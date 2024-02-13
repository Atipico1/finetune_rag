import torch, wandb
import pandas as pd
from vllm import LLM, SamplingParams
import argparse
from src.utils import str2bool
from datasets import load_dataset, Dataset
from src.utils import exact_match_score, f1_score, text_has_answer
from tqdm.auto import tqdm

INST = """Doc : {DOC}\n
Based on the above document, answer the following question. Say only the answer without saying the full sentence:
Question : {QUESTION}
Answer:"""
INST_ADV = """Doc 0: {ORIGIN_DOC}
Doc 1: {ADV_DOC}\n
Based on the above documents, answer the following question. Say only the answer without saying the full sentence:
Question : {QUESTION}
Answer:"""


def make_original_prompt(dataset, args):
    docs, questions = dataset["context"], dataset["question"]
    return [INST.format(DOC=doc, QUESTION=q) for doc, q in zip(docs, questions)]

def make_adv_prompt(dataset, args):
    docs, adv_docs, questions = dataset["context"], dataset["adversarial_passage"], dataset["question"]
    return [INST_ADV.format(ORIGIN_DOC=doc, ADV_DOC=adv_doc, QUESTION=q) for doc, adv_doc, q in zip(
        docs, adv_docs, questions)]
    
def selelct_prompt_func(args):
    if args.prompt_func == "origin":
        return make_original_prompt
    elif args.prompt_func == "adv":
        return make_adv_prompt

def main(args):
    llm=LLM(model=args.model, tensor_parallel_size=4, seed=42, dtype="auto")
    sampling_params = SamplingParams(max_tokens=args.max_tokens)
    dataset = load_dataset(args.dataset, split=args.split)
    if args.filter:
        dataset = dataset.filter(lambda x: x["replace_count"]>=2)
    prompt_func = selelct_prompt_func(args)
    prompts = prompt_func(dataset, args)
    outputs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), 64)):
            output = llm.generate(prompts[i:i+64], sampling_params)
            for o in output:
                if o.outputs[0].text.strip() == "":
                    print(o)
                outputs.append(o.outputs[0].text.strip())
    dataset = dataset.add_column("pred", outputs)
    df = pd.DataFrame(dataset)
    df["prompt"] = prompts
    df["is_em"] = df.apply(lambda x: exact_match_score(x["pred"], x[args.ans_col]), axis=1)
    df["f1"] = df.apply(lambda x: f1_score(x["pred"], x[args.ans_col]), axis=1)
    df["is_accurate"] = df.apply(lambda x: text_has_answer(x[args.ans_col], x["pred"]), axis=1)
    df[args.ans_col] = df[args.ans_col].apply(lambda x: ", ".join(x) if len(x)>1 else x[0])
    df = df[["question", args.ans_col,"prompt", "pred", "is_em", "f1", "is_accurate"]]
    wandb.init(project="evaluate-LLM", name=args.prompt_func)
    wandb.log({"raw_data":df,
               "Acc": round(df["is_accurate"].mean()*100, 2),
               "EM": round(df["is_em"].mean()*100, 2),
               "F1": round(df["f1"].mean()*100, 2)})
    wandb.finish()
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mrqa")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--model", type=str, default="", help="hub path or local path to save")
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--prompt_func", type=str, default="gen_answer_sent")
    parser.add_argument("--ans_col", type=str, default="answer_in_context")
    parser.add_argument("--test", type=str2bool, default=False)
    parser.add_argument("--filter", type=str2bool, default=True)
    args = parser.parse_args()
    main(args)