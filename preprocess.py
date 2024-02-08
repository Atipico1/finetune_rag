import torch
import spacy
import argparse
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, DPRQuestionEncoder
from src.utils import str2bool
import numpy as np
import cupy as cp

def masking(doc) -> str:
    if not len(doc.ents):
        return doc.text
    text_list =[]
    for d in doc:
        if d.pos_ == "PUNCT":
            text_list.append("@"+d.text)
        elif d.pos_ == "AUX" and d.text == "'s":
            text_list.append("@"+d.text)
        else:
            text_list.append(d.text)
    for ent in doc.ents:
        text_list[ent.start:ent.end] = ["[B]"]* (ent.end - ent.start)
        text_list[ent.start] = "[MASK]"
    return " ".join(text_list).replace(" [B]", "").replace(" @", "")

def query_masking(nlp, dataset: Dataset):
    ctxs = dataset["question"]
    result = []
    for i in tqdm(range(0, len(ctxs), 2000), desc="Masking..."):
        batch = ctxs[i:i+2000]
        batch_docs = list(nlp.pipe(batch, batch_size=2000))
        masked_quries = [masking(doc) for doc in batch_docs]
        result.extend(masked_quries)
    assert len(result) == len(ctxs), "Length doesn't match"
    return dataset.add_column("masked_query", result)

def query_embedding(model, tokenizer, dataset: Dataset, args):
    queries = dataset["masked_query"]
    result = []
    for i in tqdm(range(0, len(queries), args.batch_size), desc="Embedding..."):
        batch = queries[i:i+args.batch_size]
        output = tokenizer(batch, padding="max_length", truncation=True, max_length=64, return_tensors="pt").to("cuda")
        with torch.no_grad():
            embeddings = model(**output).pooler_output.detach().cpu().numpy() # [args.batch_size, hidden_dim]
        result.extend([emb for emb in embeddings])
    assert len(result) == len(queries), "Length doesn't match"
    return dataset.add_column("query_embedding", result)

def remove_duplicate_by_similarity(dataset: Dataset, tokenizer, model, args):
    questions = dataset["question"]
    result = []
    if args.sentence_embedder == "dpr":
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(questions), args.batch_size), desc="Embedding..."):
                batch = questions[i:i+args.batch_size]
                output = tokenizer(batch, padding="max_length", truncation=True, max_length=64, return_tensors="pt").to("cuda")
                embeddings = model(**output).pooler_output.detach().cpu().numpy() # [args.batch_size, hidden_dim]
                result.extend([emb for emb in embeddings])
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2', device="cuda")
        model.max_seq_length = 64
        with torch.no_grad():
            for i in tqdm(range(0, len(questions), args.batch_size), desc="Embedding..."):
                batch = questions[i:i+args.batch_size]
                result.extend(model.encode(batch, batch_size=args.batch_size, show_progress_bar=False))
    torch.cuda.empty_cache()
    matrix = cp.array([v/np.linalg.norm(v) for v in result], dtype=cp.float16)
    key_matrix_indices = []
    for i in tqdm(range(len(matrix)), desc="Removing duplicates by similarity..."):
        if len(key_matrix_indices) == 0:
            key_matrix_indices.append(i)
        else:
            current_matrix = matrix[i:i+1]
            key_matrix_subset = matrix[key_matrix_indices]
            similarity = cp.dot(current_matrix, key_matrix_subset.T)
            if not cp.any(similarity >= args.remove_duplicate_thres):
                key_matrix_indices.append(i)
    print(f"Remove duplicates by similarity-> Before : {len(dataset)} | After : {len(key_matrix_indices)}")
    dataset = dataset.select(key_matrix_indices, writer_batch_size=50000)
    return dataset

def remove_duplicate(data: Dataset):
    masked_queries = data["masked_query"]
    unique_queries = set()
    result_idxs = []
    for idx, query in enumerate(masked_queries):
        if query not in unique_queries:
            unique_queries.add(query)
            result_idxs.append(idx)
    print(f"Remove duplicates by string match -> Before : {len(data)} | After : {len(result_idxs)}")
    filtered_data = data.select(result_idxs, writer_batch_size=50000)
    return filtered_data

def _preprocess(dataset: Dataset, args):
    if "query_embedding" not in dataset.column_names:
        spacy.prefer_gpu()
        nlp = spacy.load("en_core_web_trf")
        tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to("cuda")
        dataset = query_masking(nlp, dataset)
        if args.remove_duplicate:  
            dataset = remove_duplicate(dataset)
        dataset = query_embedding(model, tokenizer, dataset, args)
        return dataset
    else:
        return dataset

def main(args):
    if args.split == "all":
        dataset = load_dataset(args.dataset)
        train, test = dataset["train"], dataset["test"]
        train = _preprocess(train, args)
        test = _preprocess(test, args)
        result = DatasetDict({"train": train, "test": test})
        if args.push_to_hub:
            result.push_to_hub(f"{args.dataset}_preprocessed")
    else:
        dataset = load_dataset(args.dataset, split=args.split)
        if args.test:
            dataset = dataset.select(range(5000))
        print(f"{args.dataset} Loaded! -> Size : {len(dataset)}")
        dataset = _preprocess(dataset, args)
        print(f"Preprocessing Done! -> Size : {len(dataset)}")
        if args.push_to_hub and not args.test:
            dataset.push_to_hub(f"{args.dataset}_{args.split}_preprocessed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=False, default="")
    parser.add_argument("--split", type=str, required=False, default="train")
    parser.add_argument("--batch_size", type=int, required=False, default=512)
    parser.add_argument("--test", type=str2bool, required=False, default=False)
    parser.add_argument("--push_to_hub", type=str2bool, required=False, default=False)
    parser.add_argument("--remove_duplicate", type=str2bool, required=False, default=True)
    args = parser.parse_args()
    main(args)