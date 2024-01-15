import torch
import random
import joblib, os
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm
from src.index import build_index_with_ids
from src.utils import normalize_answer, text_has_answer_wo_normalization
import numpy as np
import regex
import unicodedata
import faiss
from transformers import AutoTokenizer, DPRQuestionEncoder, DPRContextEncoder

def dpr_embed(dataset: Dataset, col: str, args) -> list[np.ndarray]:     
    inputs = list(set(dataset[col]))
    if not args.test:
        if os.path.exists(f"/data/seongilpark/{col}_embeddings.pkl"):
            cached_array = joblib.load(f"/data/seongilpark/{col}_embeddings.pkl")
            if len(cached_array) == len(inputs):
                print(f"{col} embeddings already exist")
                return cached_array, inputs
    result = []
    if col == "q":
        tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to("cuda")
    else:
        tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to("cuda")
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), args.batch_size), desc="DPR Embedding..."):
            batch = inputs[i:i+args.batch_size]
            output = tokenizer(batch,
                            padding="max_length",
                            truncation=True,
                            max_length=128 if col == "question" else 256,
                            return_tensors="pt").to("cuda")
            embeddings = model(**output).pooler_output.detach().cpu().numpy() # [args.batch_size, hidden_dim]
            result.extend([emb for emb in embeddings])
    normalized_arrays = [arr / np.linalg.norm(arr) for arr in result]
    assert len(normalized_arrays) == len(result), "Length doesn't match"
    if not args.test:
        joblib.dump(normalized_arrays, f"/data/seongilpark/{col}_embeddings.pkl")
        print(f"{col} embeddings saved")
    return normalized_arrays, inputs
                
def find_similar_questions(dataset: Dataset, args):
    simple_tokenizer = SimpleTokenizer()
    embedding, questions = dpr_embed(dataset=dataset, col="question", args=args)
    embedding = np.array(embedding).astype("float32")
    index = build_index_with_ids(embedding, "/data/seongilpark/index", "question", is_save=False)
    assert len(questions) == len(dataset), "There is a not unique question in the dataset"
    new_context = []
    answers = dataset["answers"]
    contexts = dataset["context"]

    _, I = index.search(embedding, args.topk)
    nearest_neighbors = I[:, 1:] # [len(query_vectors), topk-1]
    for neighbors, answer in tqdm(zip(nearest_neighbors, answers), desc="Faiss question searching...", total=len(nearest_neighbors)):
        is_valid = False
        for idx in neighbors:
            if not has_answer(answer, contexts[idx], simple_tokenizer):
                new_context.append(contexts[idx])
                is_valid = True
                break
        if not is_valid:
            print(f"There is no similar context -> Orignal Answers: {answer}")
            new_context.append(None)
    assert len(new_context) == len(dataset), f"Length doesn't match {len(new_context)} != {len(dataset)}"
    dataset = dataset.add_column("Q_similar_context", new_context)
    dataset = dataset.filter(lambda x: x["Q_similar_context"] is not None, num_proc=8)
    return dataset, embedding

def find_similar_contexts(dataset: Dataset, args):
    simple_tokenizer = SimpleTokenizer()
    embedding, contexts = dpr_embed(dataset=dataset, col="context", args=args)
    embedding = np.array(embedding).astype('float32')
    c2embs = {c:emb for c, emb in zip(contexts, embedding)}
    c2ids = {c:i for i, c in enumerate(contexts)}
    index = build_index_with_ids(embedding, save_dir="/data/seongilpark/index", name="context", is_save=False)
    old2new_context = dict()
    _, I = index.search(embedding, args.topk)
    nearest_neighbors = I[:, 1:] # [len(contexts), topk-1]
    for i, row in tqdm(enumerate(dataset), total=len(dataset), desc="Faiss contexts searching..."):
        query_context_id = c2ids[row["context"]]
        answers = row["answers"]
        neighbors = nearest_neighbors[query_context_id]
        is_valid = False
        for idx in neighbors:
            if not has_answer(answers, contexts[idx], simple_tokenizer):
                old2new_context[row["context"]] = contexts[idx]
                is_valid = True
                break
        if not is_valid:
            print(f"There is no similar context -> Orignal Answers : {row['answers']}")
            old2new_context[row["context"]] = None
    assert len(old2new_context) == len(contexts), f"Length doesn't match {len(old2new_context)} != {len(contexts)}"
    dataset = dataset.map(lambda x: {"C_similar_context": old2new_context[x["context"]]}, num_proc=8)
    dataset = dataset.filter(lambda x: x["C_similar_context"] is not None, num_proc=8)
    return dataset, c2embs

def find_similar_contexts_with_questions(q_embs, c2embs, dataset: Dataset, args):
    simple_tokenizer = SimpleTokenizer()
    embedding = []
    contexts = dataset["context"]
    answers = dataset["answers"]
    for i in range(len(dataset)):
        q_emb = q_embs[i]
        c_emb = c2embs[contexts[i]]
        embedding.append(args.alpha * q_emb + (1-args.alpha) * c_emb)
    embedding = np.array(embedding).astype('float32')
    index = build_index_with_ids(embedding, save_dir="/data/seongilpark/index", name="question-context", is_save=False)
    _, I = index.search(embedding, args.topk)
    nearest_neighbors = I[:, 1:] # [len(contexts), topk-1]
    new_context = []
    for neighbors, answer in tqdm(
        zip(nearest_neighbors, answers), desc="Faiss mixed searching...", total=len(nearest_neighbors)):
        is_valid = False
        for idx in neighbors:
            if not has_answer(answer, contexts[idx], simple_tokenizer):
                new_context.append(contexts[idx])
                is_valid = True
                break
        if not is_valid:
            print(f"There is no similar context -> Orignal Answers : {answer}")
            new_context.append(None)
    assert len(new_context) == len(dataset), "Length doesn't match"
    dataset = dataset.add_column("QC_similar_context", new_context)
    dataset = dataset.filter(lambda x: x["QC_similar_context"] is not None, num_proc=8)
    return dataset

def find_random_contexts(dataset: Dataset):
    simple_tokenizer = SimpleTokenizer()
    contexts = list(set(dataset["context"]))
    new_context = []
    for row in tqdm(dataset, desc="Finding random contexts..."):
        max_count = 0
        a = row["answers"]
        random_ctx = random.choice(contexts)
        while has_answer(a, random_ctx, simple_tokenizer) and max_count < 10:
            random_ctx = random.choice(contexts)
            max_count += 1
        if max_count < 10:
            new_context.append(random_ctx)
        else:
            new_context.append(None)
    assert len(new_context) == len(dataset), "Length doesn't match"
    dataset = dataset.add_column("random_context", new_context)
    dataset = dataset.filter(lambda x: x["random_context"] is not None, num_proc=8)
    return dataset             

## From Contriever repo : https://github.com/facebookresearch/contriever/blob/main/src/evaluation.py#L23

class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens

def has_answer(answers: list[str], text, tokenizer) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False

def _normalize(text):
    return unicodedata.normalize('NFD', text)