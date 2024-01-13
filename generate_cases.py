import os
from datasets import load_dataset
import pandas as pd
import json
import re
from tqdm.auto import tqdm
from datasets import Dataset, DatasetDict, concatenate_datasets
import spacy
import argparse
from src.utils import str2bool, normalize_answer, normalize_question
from preprocess import query_masking, remove_duplicate, query_embedding
from transformers import AutoTokenizer, DPRQuestionEncoder

def load_mrqa(args):
    mrqa = load_dataset("mrqa")
    train, test, valid = mrqa["train"], mrqa["test"], mrqa["validation"]
    mrqa = concatenate_datasets([train, test, valid])
    if args.test:
        mrqa = mrqa.shuffle(seed=42).select(range(5000))
    mrqa: Dataset = mrqa.filter(lambda x: x["subset"] not in args.except_columns, num_proc=8)
    print(f"MRQA Loaded without {args.except_columns} ! -> Size : {len(mrqa)}")
    return mrqa

def preprocess_text(text, subset_name: str):
    if subset_name == "NewsQA":
        start = text.find("--")
        end = start + 2
        text = text[end:]
    if ("<Table>" in text) or ("<Ol>" in text) or ("<Li>" in text):
        return None
    text = text.replace("<P>", "")
    text = text.replace("</P>", "")
    text = text.replace("[PAR]", "")
    text = text.replace("[DOC]", "")
    text = text.replace("[TLE]", "")
    text = re.sub('\n+', '', text)
    text = re.sub(' +', ' ', text)
    return text

def _preprocess(dataset: Dataset) -> Dataset:
    dataset = dataset.map(lambda x: {"context": preprocess_text(x["context"], x["subset"])}, num_proc=8, desc="Preprocessing...")
    print("Before preprocess: ", len(dataset))
    dataset = dataset.filter(lambda x: x["context"] is not None, num_proc=8)
    print("After preprocess: ", len(dataset))
    return dataset

def split_sentence(dataset: Dataset, nlp, args):
    output = []
    answer_sents = []
    answers_in_context = []
    for i in tqdm(range(0, len(dataset), args.batch_size), desc="Splitting sentence..."):
        batch = dataset[i:i+args.batch_size]
        docs = list(nlp.pipe(batch["context"], batch_size=args.batch_size, disable=["ner"]))
        answers = batch["answers"]
        for doc, answer in zip(docs, answers):
            answer_sent_idx = -1
            sents = [sent.text for sent in doc.sents]
            for idx, sent in enumerate(sents):
                if answer_sent_idx != -1: break
                for ans in answer:
                    if normalize_answer(ans) in normalize_answer(sent):
                        answer_sent_idx = idx
                        answer_sents.append(sent)
                        answers_in_context.append(ans)
                        break
            if answer_sent_idx == -1:
                answer_sents.append(None)
                answers_in_context.append(None)
                output.append(None)
            else:
                start_idx, end_idx = max(0, answer_sent_idx-3), min(len(sents), answer_sent_idx+4)
                output.append(" ".join([sent.strip() for sent in sents[start_idx:end_idx]]))
    assert len(list(set([len(output), len(dataset), len(answer_sents), len(answers_in_context)])))==1, "Length doesn't match"
    print("Before split: ", len(dataset))
    dataset = dataset.add_column("short_context", output)
    dataset = dataset.add_column("answer_sent", answer_sents)
    dataset = dataset.add_column("answer_in_context", answers_in_context)
    dataset = dataset.filter(lambda x: x["short_context"] is not None and len(x["short_context"].split())< args.ctx_len, num_proc=4)
    print("After split: ", len(dataset))
    return dataset

def main(args):
    mrqa = load_mrqa(args)
    if args.use_gpu:
        spacy.prefer_gpu(args.gpu_id)
    tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to("cuda")
    nlp = spacy.load(args.masking_model)
    mrqa = query_masking(nlp, mrqa)
    del nlp
    nlp = spacy.load(args.spliter_model)
    mrqa = remove_duplicate(mrqa)
    mrqa = _preprocess(mrqa)
    mrqa = split_sentence(mrqa, nlp, args)
    del nlp
    mrqa = query_embedding(model, tokenizer, mrqa, args)
    if args.test:
        sample = mrqa.shuffle().select(range(10))
        for row in sample:
            q, sc, a, c = row["question"], row["short_context"], row["answers"], row["context"]
            print(f"Question : {q}\nShort Context : {sc}\nAnswer : {a}\nContext : {c}\n\n\n\n")
        mrqa = mrqa.remove_columns(["context", "answers","context_tokens","question_tokens","detected_answers"])
        mrqa = mrqa.rename_column("short_context", "context")
    else:
        mrqa = mrqa.remove_columns(["context", "answers","context_tokens","question_tokens","detected_answers"])
        mrqa = mrqa.rename_column("short_context", "context")
        mrqa.push_to_hub("Atipico1/mrqa_preprocessed")
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mrqa")
    parser.add_argument(
        "--except_columns", nargs="+", type=str, default="", help="Columns: SearchQA | SQuAD | NaturalQuestionsShort | HotpotQA | NewsQA")
    parser.add_argument("--use_gpu", type=str2bool, default=False)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--test", type=str2bool, default=False)
    parser.add_argument("--masking_model", type=str, default="en_core_web_trf")
    parser.add_argument("--spliter_model", type=str, default="en_core_web_sm")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--ctx_len", type=int, default=150)
    args = parser.parse_args()
    main(args)