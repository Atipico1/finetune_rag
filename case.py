import argparse
import numpy as np
import faiss
from src.utils import str2bool
from datasets import load_dataset, Dataset
from collections import defaultdict
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from src.index import build_multiple_indexes
from nltk.tokenize import sent_tokenize

WH_WORDS = ["what", "when", "where", "who", "why", "how","which","whom"]

def match_case(qa_dataset: Dataset, multiple_index, args):
    cnt = 0
    output = []
    for row in tqdm(qa_dataset, desc="CASE Matching..."):
        head_word = row["question"].strip().lower().split()[0]
        if (head_word not in WH_WORDS) or (head_word not in multiple_index.keys()):
            head_word = "original"
        index, id2q = multiple_index[head_word]["index"], multiple_index[head_word]["id2q"]
        query = np.array([row["query_embedding"]]).astype("float32")
        distances, indices = index.search(query, args.num_cbr)
        cases = []
        for dist, idx in zip(distances[0], indices[0]):
            matched_row = id2q[idx]
            matched_row.update({"distance":str(dist)})
            cases.append(matched_row)
            cnt += 1
            if args.printing:
                if cnt % (len(qa_dataset) // 5) == 0:
                    print("Original Question: ", row["question"])
                    for k, v in matched_row.items():
                        print(f"Matched {k}:{v}")
                    print("-"*100)
        output.append(cases)
    return output

def _filter_long_ctx(example: Dict, max_len: int):
    if len(example["context"].split()) > max_len:
        return False
    else:
        return True

def shorten_ctx(context: str, answers, max_len: int):
    if len(context.split()) > max_len:
        sents = sent_tokenize(context)
        answer_sent_idx = -1
        for sent_idx, sent in enumerate(sents):
            for ans in answers["text"]:
                if ans in sent:
                    answer_sent_idx = sent_idx
                    break
            if answer_sent_idx != -1:
                break
        if answer_sent_idx == -1:
            return context
        else:
            new_ctx = " ".join(sents[max(0, answer_sent_idx-3):min(len(sents), answer_sent_idx+4)])
            return new_ctx
    else:
        return context

def _preprocess(dataset: Dataset, args):
    if args.short_ctx:
        try:
            return load_dataset(f"SQuAD_under_{args.short_ctx_len}", split=args.qa_split)
        except:
            dataset = dataset.map(lambda x: {"context":shorten_ctx(x["context"], x["answers"], args.short_ctx_len)})
            dataset = dataset.filter(lambda x: _filter_long_ctx(x, args.short_ctx_len))
            print(f"case preprocessed! -> Size : {len(dataset)}")
            dataset.push_to_hub(f"Atipico1/SQuAD_under_{args.short_ctx_len}")
            return dataset
    else:
        return dataset

def main(args):
    qa_dataset = load_dataset(args.qa_dataset, split=args.qa_split)
    original_case = load_dataset("Seongill/SQuAD_unique_questions", split="train")
    original_case = _preprocess(original_case, args)
    if args.test:
        qa_dataset = qa_dataset.select(range(5000))
    print(f"{args.qa_dataset} Loaded! -> Size : {len(qa_dataset)}")
    sub_original = defaultdict(list)
    for row in original_case:
        head_word = row["question"].strip().lower().split()[0]
        if head_word not in WH_WORDS:
            head_word = "original"
        sub_original[head_word].append(({"question":row["question"], "context":row["context"],"answer":row["answers"]["text"][0]}, row["query_embedding"]))
    multiple_indexs_origin: Dict = build_multiple_indexes(sub_original, [k for k in sub_original.keys()])
    original_case = match_case(qa_dataset, multiple_indexs_origin, args)
    qa_dataset = qa_dataset.add_column("original_case", original_case)
    qa_dataset = qa_dataset.remove_columns(["query_embedding"])
    if not args.test:
        qa_dataset.push_to_hub(f"{args.qa_dataset}_with_so_case")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_dataset", type=str, required=True, default="")
    parser.add_argument("--qa_split", type=str, required=False, default="train")
    parser.add_argument("--num_cbr", type=int, required=False, default=5)
    parser.add_argument("--short_ctx", type=str2bool, required=False, default=False)
    parser.add_argument("--short_ctx_len", type=int, required=False, default=150)
    parser.add_argument("--test", type=str2bool, required=False, default=False)
    parser.add_argument("--printing", type=str2bool, required=False, default=False)
    args = parser.parse_args()
    main(args)