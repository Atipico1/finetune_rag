import torch, wandb
import pandas as pd
from vllm import LLM, SamplingParams
import argparse
from src.utils import str2bool
from datasets import load_dataset, Dataset
from src.utils import exact_match_score, f1_score, text_has_answer
from tqdm.auto import tqdm
from typing import List
import random
import numpy as np

INST = """<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nDoc : {DOC}\n
Based on the above document, answer the following question. Please provide the answer as a single word or term, without forming a complete sentence:
Question : {QUESTION}
Answer:<|im_end|>\n<|im_start|>assistant\n"""
INST_ADV = """<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nDoc 0: {ORIGIN_DOC}
Doc 1: {ADV_DOC}\n
Based on the above documents, answer the following question. Please provide the answer as a single word or term, without forming a complete sentence:
Question : {QUESTION}
Answer:<|im_end|>\n<|im_start|>assistant\n"""

def make_original_prompt(dataset, q_type: str, args):
    docs, questions = dataset["context"], dataset[q_type]
    return [INST.format(DOC=doc, QUESTION=q) for doc, q in zip(docs, questions)]

def make_original_sent(dataset, q_type, args):
    docs, questions = dataset["answer_sent"], dataset[q_type]
    return [INST.format(DOC=doc, QUESTION=q) for doc, q in zip(docs, questions)]

def adv_prompt(dataset, q_type, args):
    if args.random:
        output = []
        docs, adv_docs, questions = dataset["context"], dataset[args.adv_doc], dataset[q_type]
        half = len(docs)//2
        for i in range(half):
            output.append(INST_ADV.format(ORIGIN_DOC=docs[i], ADV_DOC=adv_docs[i], QUESTION=questions[i]))
        for i in range(half, len(docs)):
            output.append(INST_ADV.format(ORIGIN_DOC=adv_docs[i], ADV_DOC=docs[i], QUESTION=questions[i]))
        return output
    else:
        docs, adv_docs, questions = dataset["context"], dataset[args.adv_doc], dataset[q_type]
        return [INST_ADV.format(ORIGIN_DOC=doc, ADV_DOC=adv_doc, QUESTION=q) for doc, adv_doc, q in zip(
            docs, adv_docs, questions)]

def adv_sent_prompt(dataset, q_type, args):
    if args.random:
        output = []
        docs, adv_docs, questions = dataset["answer_sent"], dataset[args.adv_sent], dataset[q_type]
        half = len(docs)//2
        for i in range(half):
            output.append(INST_ADV.format(ORIGIN_DOC=docs[i], ADV_DOC=adv_docs[i], QUESTION=questions[i]))
        for i in range(half, len(docs)):
            output.append(INST_ADV.format(ORIGIN_DOC=adv_docs[i], ADV_DOC=docs[i], QUESTION=questions[i]))
        return output
    else:
        docs, adv_docs, questions = dataset["answer_sent"], dataset[args.adv_sent], dataset[q_type]
        return [INST_ADV.format(ORIGIN_DOC=doc, ADV_DOC=adv_doc, QUESTION=q) for doc, adv_doc, q in zip(
            docs, adv_docs, questions)]

def make_random_prompts(dataset, q_type, args):
    docs, questions = dataset["context"], dataset[q_type]
    result = []
    for i, (doc, question) in enumerate(zip(docs, questions)):
        random_doc = random.choice(docs)
        while random_doc == doc:
            random_doc = random.choice(docs)
        if args.random:
            if i < len(docs)//2:
                result.append(INST_ADV.format(ORIGIN_DOC=doc, ADV_DOC=random_doc, QUESTION=question))
            else:
                result.append(INST_ADV.format(ORIGIN_DOC=random_doc, ADV_DOC=doc, QUESTION=question))
        else:
            result.append(INST_ADV.format(ORIGIN_DOC=doc, ADV_DOC=random_doc, QUESTION=question))
    return result

def make_random_prompt(dataset, q_type):
    docs, questions = dataset["context"], dataset[q_type]
    result = []
    for doc, question in zip(docs, questions):
        random_doc = random.choice(docs)
        while random_doc == doc:
            random_doc = random.choice(docs)
        result.append(INST.format(DOC=random_doc, QUESTION=question))
    return result

def selelct_prompt_func(key: str, q_type: str):
    if key == "origin":
        return make_original_prompt
    else:
        return adv_prompt

def select_sent_func(key: str):
    if key == "origin":
        return make_original_sent
    else:
        return adv_sent_prompt
    
def make_new_question(questions : List[str], llm):
    PROMPT = """<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPlease paraphrase the following question. You should maintain original meaning and information.
    Question : {QUESTION}
    Paraphrased question:<|im_end|>\n<|im_start|>assistant\n"""
    prompts = [PROMPT.format(QUESTION=q) for q in questions]
    sampling_params = SamplingParams(max_tokens=50)
    paraphrased_questions = llm.generate(prompts, sampling_params, use_tqdm=True)
    paraphrased_questions = [o.outputs[0].text.strip() for o in paraphrased_questions]
    return paraphrased_questions

def sent_level_adv(dataset, llm, sampling_params, args):
    result = []
    answers = dataset[args.ans_col]
    with torch.no_grad(): 
        for q_type in ["question", "new_question"]:
            for key in tqdm(["origin", "adv"]):
                prompt_func = select_sent_func(key)
                prompts = prompt_func(dataset, q_type, args)
                outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
                outputs = [o.outputs[0].text.strip() for o in outputs]
                ems = [exact_match_score(pred, label) for pred, label in zip(outputs, answers)]
                f1s = [f1_score(pred, label) for pred, label in zip(outputs, answers)]
                accs = [text_has_answer(label, pred) for label, pred in zip(answers, outputs)]
                metrics = ["sent_"+key+"_"+q_type]
                for metric_name, values in zip(["EM", "F1", "Acc"], [ems, f1s, accs]):
                    metrics.append(round(np.mean(values)*100, 3))
                result.append(metrics)
    df = pd.DataFrame(data=result, columns=["prompt", "EM", "F1", "Acc"])
    return df

def passage_level_adv(dataset, llm, sampling_params, args):
    result = []
    answers = dataset[args.ans_col]
    with torch.no_grad():
        for q_type in ["question", "new_question"]:
            for key in tqdm(["origin", "adv-only-gpt"]):
                prompt_func = selelct_prompt_func(key, q_type)
                prompts = prompt_func(dataset, q_type, args)
                outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
                outputs = [o.outputs[0].text.strip() for o in outputs]
                ems = [exact_match_score(pred, label) for pred, label in zip(outputs, answers)]
                f1s = [f1_score(pred, label) for pred, label in zip(outputs, answers)]
                accs = [text_has_answer(label, pred) for label, pred in zip(answers, outputs)]
                metrics = ["passage_"+ key+"_"+q_type]
                for metric_name, values in zip(["EM", "F1", "Acc"], [ems, f1s, accs]):
                    metrics.append(round(np.mean(values)*100, 3))
                result.append(metrics)
    df = pd.DataFrame(data=result, columns=["prompt", "EM", "F1", "Acc"])
    return df

def origin_random(dataset, llm, sampling_params, args):
    answers = dataset[args.ans_col]
    result = []
    with torch.no_grad():
        for q_type in ["question", "new_question"]:
            prompts = make_random_prompts(dataset, q_type, args)
            outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
            outputs = [o.outputs[0].text.strip() for o in outputs]
            ems = [exact_match_score(pred, label) for pred, label in zip(outputs, answers)]
            f1s = [f1_score(pred, label) for pred, label in zip(outputs, answers)]
            accs = [text_has_answer(label, pred) for label, pred in zip(answers, outputs)]
            metrics = ["R+O_"+q_type]
            for metric_name, values in zip(["EM", "F1", "Acc"], [ems, f1s, accs]):
                metrics.append(round(np.mean(values)*100, 3))
            result.append(metrics)
    return pd.DataFrame(data=result, columns=["prompt", "EM", "F1", "Acc"])

def main(args):
    llm=LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), seed=42, dtype="auto")
    sampling_params = SamplingParams(max_tokens=args.max_tokens)
    if not args.cache_q:
        dataset = load_dataset(args.dataset, split=args.split)
        questions = make_new_question(dataset["question"], llm)
        dataset = dataset.add_column("new_question", questions)
        dataset.push_to_hub(args.dataset + "-new_question")
    else:
        dataset = load_dataset(args.dataset + "-new_question", split=args.split)
    args.model = args.model.split("/")[-1]
    if args.adv_doc == "gpt_adv_sent_passage":
        args.adv_type = "gptonly"
    elif args.adv_doc == "gpt_passage":
        args.adv_type = "gpt"
    else:
        args.adv_type = "custom"
    if args.random:
        dataset = dataset.shuffle(seed=42)
    sent_df = sent_level_adv(dataset, llm, sampling_params, args)
    pass_df = passage_level_adv(dataset, llm, sampling_params, args)
    ro_df = origin_random(dataset, llm, sampling_params, args)
    pd.concat([sent_df, pass_df, ro_df]).to_excel(f"{'' if not args.random else 'random_'}{args.adv_type}_{args.model}.xlsx", index=False)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mrqa")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--cache_q", type=str2bool, default=False)
    parser.add_argument("--model", type=str, default="", help="hub path or local path to save")
    parser.add_argument("--max_tokens", type=int, default=20)
    parser.add_argument("--ans_col", type=str, default="answer_in_context")
    parser.add_argument("--test", type=str2bool, default=False)
    parser.add_argument("--random", type=str2bool, default=False)
    parser.add_argument("--adv_doc", type=str, default="adv_context")
    parser.add_argument("--adv_sent", type=str, default="adv_answer_sent")
    args = parser.parse_args()
    main(args)