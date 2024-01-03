import argparse
from tqdm.auto import tqdm
from transformers import pipeline
from utils import str2bool
from utils import *
from datasets import load_dataset
import pandas as pd
import json

def preprocess_dataset(args):
    dataset = load_dataset(args.dataset, split=args.split)
    if args.q_field:
        dataset = dataset.remane_column(args.q_field, "question")
    if args.a_field:
        dataset = dataset.remane_column(args.a_field, "answers")
    if args.test:
        dataset = dataset.select(range(5))
    return dataset

def main(args):
    pipe = pipeline("text-generation", model=args.model, device_map="auto")
    dataset = preprocess_dataset(args)
    res = []
    for row in tqdm(dataset, desc="Evaluating"):
        q = normalize_question(row['question'])
        q = "Answer the following question\n\nQ: " + q + "\n" + "A:"
        a = row["answers"]
        output = pipe(q, do_sample=False, max_new_tokens=30)[0]["generated_text"]
        question_length = len(q)
        output = output[question_length:].strip()
        is_em, is_acc = exact_match_score(output, a), text_has_answer(a, output)
        res.append([q, a, output, is_em, is_acc])
    df = pd.DataFrame(res, columns=["Question", "Answers", "Prediction", "EM", "ACC"])
    length, total_em, total_acc = len(df), round(df["EM"].mean(),3)*100, round(df["ACC"].mean(),3)*100
    with open(f"zeroshot/{args.dataset.split('/')[-1]}.json", "w") as f:
        json.dump({"length": length, "EM": total_em, "ACC": total_acc}, f)
    df.to_csv(f"zeroshot/{args.dataset.split('/')[-1]}.csv", index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--dataset", type=str, required=False, default="Seongill/nq")
    parser.add_argument("--q_field", type=str, required=False, default=None)
    parser.add_argument("--a_field", type=str, required=False, default=None)
    parser.add_argument("--split", type=str, required=False, default="test")
    parser.add_argument("--test", type=str2bool, required=False, default=False)
    main(parser.parse_args())