import torch
import random
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
    return normalized_arrays, inputs
                
def find_similar_questions(dataset: Dataset, args):
    simple_tokenizer = SimpleTokenizer()
    embedding, questions = dpr_embed(dataset=dataset, col="question", args=args)
    embedding = np.array(embedding).astype("float32")
    index = build_index_with_ids(embedding, "/data/seongilpark/index", "question", is_save=False)
    assert len(questions) == len(dataset), "There is a not unique question in the dataset"
    new_context = []
    answers = dataset["answer_in_context"]
    contexts = dataset["context"]

    for i, query in tqdm(enumerate(embedding), total=len(embedding), desc="Faiss question searching..."):
        _, I = index.search(np.array([query]).astype("float32"), args.topk+1)
        is_valid = False
        nearest_neighbors = I[0][1:] # [topk]
        for idx in nearest_neighbors:
            if not has_answer([answers[i]], contexts[idx], simple_tokenizer):
                new_context.append(contexts[idx])
                is_valid = True
                break
        if not is_valid:
            print(f"There is no similar context -> Orignal Answers: {answers[i]}")
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
    index = build_index_with_ids(embedding, save_dir="", name="", is_save=False)
    old2new_context = dict()
    for i, row in tqdm(enumerate(dataset), total=len(dataset), desc="Faiss contexts searching..."):
        query_context_emb = c2embs[row["context"]]
        answers = [row["answer_in_context"]]
        _, I = index.search(np.array([query_context_emb]).astype("float32"), args.topk+1)
        is_valid = False
        nearest_neighbors = I[0][1:]
        for idx in nearest_neighbors:
            if not has_answer(answers, contexts[idx], simple_tokenizer):
                old2new_context[row["context"]] = contexts[idx]
                is_valid = True
                break
        if not is_valid:
            print(f"There is no similar context -> Orignal Answers : {answers}")
            old2new_context[row["context"]] = None
    assert len(old2new_context) == len(contexts), f"Length doesn't match {len(old2new_context)} != {len(contexts)}"
    dataset = dataset.map(lambda x: {"C_similar_context": old2new_context[x["context"]]}, num_proc=8)
    dataset = dataset.filter(lambda x: x["C_similar_context"] is not None, num_proc=8)
    return dataset, c2embs

def find_similar_contexts_with_questions(q_embs, c2embs, dataset: Dataset, args):
    simple_tokenizer = SimpleTokenizer()
    embedding = []
    contexts = dataset["context"]
    answers = dataset["answer_in_context"]
    for i in range(len(dataset)):
        q_emb = q_embs[i]
        c_emb = c2embs[contexts[i]]
        embedding.append(args.alpha * q_emb + (1-args.alpha) * c_emb)
    embedding = np.array(embedding).astype('float32')
    index = build_index_with_ids(embedding, save_dir="", name="", is_save=False)
    new_context = []
    for i, query in tqdm(enumerate(embedding), total=len(embedding), desc="Faiss mixed searching..."):
        _, I = index.search(np.array([query]).astype("float32"), args.topk+1)
        is_valid = False
        nearest_neighbors = I[0][1:] # [topk]
        for idx in nearest_neighbors:
            if not has_answer([answers[i]], contexts[idx], simple_tokenizer):
                new_context.append(contexts[idx])
                is_valid = True
                break
        if not is_valid:
            print(f"There is no similar context -> Orignal Answers: {answers[i]}")
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
        a = row["answer_in_context"]
        random_ctx = random.choice(contexts)
        while has_answer([a], random_ctx, simple_tokenizer) and max_count < 100:
            random_ctx = random.choice(contexts)
            max_count += 1
        if max_count < 100:
            new_context.append(random_ctx)
        else:
            new_context.append(None)
    assert len(new_context) == len(dataset), "Length doesn't match"
    dataset = dataset.add_column("random_context", new_context)
    dataset = dataset.filter(lambda x: x["random_context"] is not None, num_proc=8)
    return dataset
# Op1 : 가장 비슷한 question의 context
# Op2 : 가장 비슷한 context
# Op3 : 가장 비슷한 context+question

def search_topk_without_same_answers(queries: list[str],
                                     query_answers: list[list[str]],
                                     query_vectors: np.ndarray,
                                     index,
                                     id2vec: dict,
                                     args) -> dict[str, str]:
    # Op1에서 id2vec은 key : question_id, value : context
    # Op2에서 id2vec은 key : context_id, value : context
    # Op3에서 id2vec은 key : context_id, value : context+question
    simple_tokenizer = SimpleTokenizer()
    D, I = index.search(query_vectors, args.topk+1)
    output = dict()
    nearest_neighbors = I[:, 1:] # [len(query_vectors), topk]
    similarities = D[:, 1:] # [len(query_vectors), topk]
    for neighbors, similarity, answers, query in tqdm(
        zip(nearest_neighbors, similarities, query_answers, queries), desc="Searching...", total=len(queries)
        ):
        for idx, score in zip(neighbors, similarity):
            if not has_answer(answers, id2vec[idx], simple_tokenizer):
                output[query] = id2vec[idx]
    return output
                

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