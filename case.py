import faiss
import argparse
import numpy as np
from src.utils import normalize_answer, str2bool
from src.search import has_answer, SimpleTokenizer
from datasets import load_dataset, Dataset, DatasetDict
from collections import defaultdict
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from src.index import build_multiple_indexes

WH_WORDS = ["what", "when", "where", "who", "why", "how","which","whom"]
UNANSWERABLE_ALIAS = ["unanswerable",
                      "no answer",
                      "no answers",
                      "I don't know.",
                      "no answer at all",
                      "no answers at all",
                      "?"]
CONFLICT_ALIAS = ["conflict",
                  "conflicting",
                  "conflictive",
                  "contradictory",
                  "contradicting",
                  "contradictive",
                  "inconsistent",
                  "inconsistency"]

def _filter_case(case: Dict, question: str, answers: List[str], tokenizer):
    if args.filter_case == "same_q":
        return normalize_answer(case["question"]) == normalize_answer(question)
    elif args.filter_case == "same_qa":
        if normalize_answer(case["question"]) == normalize_answer(question):
            return False
        for answer in case["original_answers"]:
            for ans in answers:
                if normalize_answer(ans)==normalize_answer(answer):
                    return False
        return True
    elif args.filter_case == "same_qa_hasanswer":
        if normalize_answer(case["question"]) == normalize_answer(question):
            return False
        for answer in case["original_answers"]:
            for ans in answers:
                if normalize_answer(ans)==normalize_answer(answer):
                    return False
        if has_answer(answers, case["context"], tokenizer):
            return False
        return True
    else:
        return True

def _make_label(args, key: str):
    if key == "unanswerable":
        if args.case_label == "random":
            return np.random.choice(UNANSWERABLE_ALIAS)
        elif args.case_label == "fixed":
            return "unanswerable"
    elif key == "conflict":
        if args.case_label == "random":
            return np.random.choice(CONFLICT_ALIAS)
        elif args.case_label == "fixed":
            return "conflict"

def match_case(qa_dataset: Dataset, multiple_index, args):
    tokenizer = SimpleTokenizer()
    cnt = 0
    output = []
    for row in tqdm(qa_dataset, desc="CASE Matching..."):
        head_word = row["question"].strip().lower().split()[0]
        if (head_word not in WH_WORDS) or (head_word not in multiple_index.keys()):
            head_word = "original"
        index, id2q = multiple_index[head_word]["index"], multiple_index[head_word]["id2q"]
        query = np.array([row["query_embedding"]]).astype("float32")
        distances, indices = index.search(query, 100)
        cases = []
        for dist, idx in zip(distances[0], indices[0]):
            if len(cases) == args.num_cbr:
                break
            matched_row = id2q[idx]
            if not _filter_case(matched_row, row["question"], row["answers"], tokenizer):
                continue
            matched_row.update({"distance":str(dist)})
            cases.append(matched_row)
            cnt += 1
            if args.printing:
                if cnt % (len(qa_dataset) // 5) == 0:
                    print("Original Question: ", row["question"])
                    for k, v in matched_row.items():
                        print(f"Matched {k}: {v}")
                    print("-"*100)
        assert len(cases) == args.num_cbr, f"Number of cases is not {args.num_cbr} -> {len(cases)}"
        output.append(cases)
    return output

def _remove_squad_duplicate(squad: Dataset, data2):
    masked_queries = data2["masked_query"]
    input_queries = squad["masked_query"]
    result_idxs = []
    for idx, query in enumerate(input_queries):
        if query not in masked_queries:
            result_idxs.append(idx)
    print(f"Remove SQuAD2 duplicates -> Before : {len(squad)} | After : {len(result_idxs)}")
    filtered_data = squad.select(result_idxs, writer_batch_size=10000)
    return filtered_data

def make_indexs_for_case(case_dataset: Dataset, key: str, args):
    output = defaultdict(list)
    for row in case_dataset:
        head_word = row["question"].strip().lower().split()[0]
        if head_word not in WH_WORDS:
            head_word = "original"
        if key == "original":
            output[head_word].append(({"question":row["question"],
                                       "context":row["context"],
                                       "answer":row["answer_in_context"][0],
                                       "original_answers":row["answer_in_context"]}, row["query_embedding"]))
        elif key == "unanswerable":
            output[head_word].append(({"question":row["question"],
                                       "context":row[args.unans_ctx_col],
                                       "answer": _make_label(args, key=key),
                                       "original_answers":row["answer_in_context"]}, row["query_embedding"]))
        elif key == "conflict":
            output[head_word].append(({"question":row["question"],
                                      "context":row["context"],
                                      "conflict_context":row["conflict_context"],
                                      "answer": _make_label(args, key=key),
                                      "original_answers":row["answer_in_context"]},row["query_embedding"]))
            pass
        elif key == "adversary":
            pass
        else:
            raise ValueError("Wrong key!")
    if args.add_squad2 and key == "unanswerable":
        print("Add SQuAD2.0")
        squad2 = load_dataset("Atipico1/squad_v2_unique_questions", split="train")
        squad2 = _remove_squad_duplicate(squad2, case_dataset)
        for row in squad2:
            head_word = row["question"].strip().lower().split()[0]
            if head_word not in WH_WORDS:
                head_word = "original"
            output[head_word].append(({"question":row["question"],
                                       "context":row["context"],
                                       "answer":"unanswerable",
                                       "answers":["unanswerable"],
                                       "original_answers":["unanswerable"]}, row["query_embedding"]))
    return build_multiple_indexes(output, [k for k in output.keys()])

def _filter_cases(case_set: Dataset, args):
    return case_set.filter(lambda x: (len(x["answer_in_context"][0].split()) < args.case_answer_max_len) and ("<Tr>" not in x["context"]), num_proc=8)

def main(args):
    qa_dataset = load_dataset(args.qa_dataset)
    if args.test:
        for subset in qa_dataset.keys():
            qa_dataset[subset] = qa_dataset[subset].select(range(1000))
            print(f"{args.qa_dataset} {subset} Loaded! -> Size : {len(qa_dataset[subset])}")
    original_case = load_dataset("Atipico1/mrqa_preprocessed_thres-0.95_by-dpr", split="train")
    original_case = _filter_cases(original_case, args)
    unans_case = load_dataset("Atipico1/mrqa_v2_unans", split="train")
    unans_case = _filter_cases(unans_case, args)
    conflict_case = load_dataset("Atipico1/mrqa_preprocessed_with_substitution-rewritten-conflict", split="train")
    conflict_case = _filter_cases(conflict_case, args)
    original_case_index = make_indexs_for_case(original_case, "original", args)
    unans_case_index = make_indexs_for_case(unans_case, "unanswerable", args)
    conflict_case_index = make_indexs_for_case(conflict_case, "conflict", args)
    for subset in qa_dataset.keys():
        subset_original_case = match_case(qa_dataset[subset], original_case_index, args)
        subset_unans_case = match_case(qa_dataset[subset], unans_case_index, args)
        subset_conflict_case = match_case(qa_dataset[subset], conflict_case_index, args)
        qa_dataset[subset] = qa_dataset[subset].add_column("original_case", subset_original_case)
        qa_dataset[subset] = qa_dataset[subset].add_column("unans_case", subset_unans_case)
        qa_dataset[subset] = qa_dataset[subset].add_column("conflict_case", subset_conflict_case)
        qa_dataset[subset] = qa_dataset[subset].remove_columns(["query_embedding"])
    if not args.test:
        qa_dataset.push_to_hub(f"{args.save_dir}") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_dataset", type=str, required=True, default="")
    parser.add_argument("--num_cbr", type=int, required=False, default=5)
    parser.add_argument("--add_squad2", type=str2bool, required=False, default=False)
    parser.add_argument("--unans_ctx_col", type=str, required=False, default="QC_similar_context")
    parser.add_argument("--filter_case", type=str, required=False, default="")
    parser.add_argument("--case_label", type=str, required=False, default="random")
    parser.add_argument("--save_dir", type=str, required=False, default="")
    parser.add_argument("--case_answer_max_len", type=int, required=False, default=10)
    parser.add_argument("--test", type=str2bool, required=False, default=False)
    parser.add_argument("--printing", type=str2bool, required=False, default=False)
    args = parser.parse_args()
    main(args)