from dataset import (
    split_sentence,
    preprocess_text,
    make_spacy_docs,
    annotate_answer_type
)
from datasets import load_dataset
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from datasets import Dataset, DatasetDict, concatenate_datasets
import spacy
import argparse
import wandb
from src.utils import (
    str2bool,
    normalize_answer
)
from src.search import (
    find_similar_questions,
    find_similar_contexts,
    find_similar_contexts_with_questions,
    find_random_contexts
)
from src.index import build_index_with_ids, search_index
from preprocess import query_masking, remove_duplicate, query_embedding
from transformers import AutoTokenizer, DPRQuestionEncoder

def _load_dataset(args):
    if args.task == "preprocess":
        return load_mrqa(args)
    else:
        dataset = load_dataset(args.processed_dataset, split="train")
        if args.test:
            return dataset.shuffle(seed=42).select(range(10000))
        print(f"{args.dataset} Loaded! -> Size : {len(dataset)}")
        return dataset

def load_mrqa(args):
    mrqa = load_dataset("mrqa")
    train, test, valid = mrqa["train"], mrqa["test"], mrqa["validation"]
    mrqa = concatenate_datasets([train, test, valid])
    if args.test:
        mrqa = mrqa.shuffle(seed=42).select(range(5000))
    mrqa: Dataset = mrqa.filter(lambda x: x["subset"] not in args.except_subset, num_proc=8)
    print(f"MRQA Loaded without {args.except_subset} ! -> Size : {len(mrqa)}")
    return mrqa

def preprocess(args, mrqa):
    nlp = spacy.load(args.spacy_model)
    tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to("cuda")
    mrqa = query_masking(nlp, mrqa)
    del nlp
    nlp = spacy.load(args.spliter_model)
    mrqa = remove_duplicate(mrqa)
    mrqa = preprocess_text(mrqa)
    mrqa = split_sentence(mrqa, nlp, args)
    mrqa = query_embedding(model, tokenizer, mrqa, args)
    if args.test:
        sample = mrqa.shuffle().select(range(10))
        for row in sample:
            q, sc, a, c = row["question"], row["short_context"], row["answers"], row["context"]
            print(f"Question : {q}\nShort Context : {sc}\nAnswer : {a}\nContext : {c}\n\n\n\n")
    else:
        mrqa = mrqa.remove_columns(["context","context_tokens","question_tokens","detected_answers"])
        mrqa = mrqa.rename_column("short_context", "context")
        mrqa.push_to_hub("Atipico1/mrqa_preprocessed")

def generate_unans(args, dataset):
    dataset, q_embs = find_similar_questions(dataset, args)
    dataset, c2embs = find_similar_contexts(dataset, args)
    dataset = find_similar_contexts_with_questions(q_embs, c2embs, dataset, args)
    dataset = find_random_contexts(dataset)
    wandb.init(project="craft-cases", name="unanswerable" if not args.test else "test-unanswerable", config=vars(args))
    df = pd.DataFrame(dataset)
    df["answer_in_context"] = df["answer_in_context"].apply(lambda x: ", ".join(x))
    df = df[["question","context","answer_in_context","Q_similar_context","C_similar_context","QC_similar_context","random_context"]].sample(100)
    wandb.log({"samples": wandb.Table(dataframe=df)})
    if not args.test:
        dataset.push_to_hub("Atipico1/mrqa_unanswerable_v2")

def generate_adversary(args, dataset):
    nlp = spacy.load(args.spacy_model)
    output = make_spacy_docs(dataset, nlp, args)
    dataset= annotate_answer_type(dataset, output, args)
    
    
def generate_conflict(args):
    pass
    

def generate_all(args):
    pass

if __name__=="__main__":
    # Main parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mrqa")
    parser.add_argument("--processed_dataset", type=str, default="Atipico1/mrqa_preprocessed")
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--test", type=str2bool, default=False)
    parser.add_argument("--spacy_model", type=str, default="en_core_web_trf")
    parser.add_argument("--spliter_model", type=str, default="en_core_web_sm")
    parser.add_argument("--batch_size", type=int, default=1024)
    # Make subparsers
    subparsers = parser.add_subparsers(dest="task", help='Possible tasks: preprocess | all | unans | adversary | conflict')

    # Preprocess parser
    preprocess_parser = subparsers.add_parser('preprocess', help='Run preprocess')
    preprocess_parser.add_argument('--ctx_len', type=int, default=150)
    preprocess_parser.add_argument(
        '--except_subset', nargs='+', type=str, default='',
        help='Possible Subsets: SearchQA | SQuAD | NaturalQuestionsShort | HotpotQA | NewsQA'
        )
    
    # Unans parser
    unans_parser = subparsers.add_parser('unans', help='Generate unanswerable cases')
    unans_parser.add_argument('--alpha', type=float, default=0.5)
    unans_parser.add_argument('--topk', type=int, default=10, help='Top k nearest neighbors for faiss search')
    # Adversary parser
    adv_parser = subparsers.add_parser('adversary', help='Run task 3')
    adv_parser.add_argument('--save_dir', help='path to save the docs', type=str, default='/data/seongilpark/adversary')
    
    args = parser.parse_args()
    if args.use_gpu:
        spacy.prefer_gpu(args.gpu_id)
    dataset = _load_dataset(args)
    if args.task == "preprocess":
        preprocess(args, dataset)
    elif args.task == "unans":
        generate_unans(args, dataset)
    elif args.task == "adversary":
        generate_adversary(args, dataset)
    elif args.task == "conflict":
        generate_conflict(args, dataset)
    elif args.task == "all":
        generate_all(args, dataset)
    else:
        raise ValueError(f"Invalid task: {args.task}")