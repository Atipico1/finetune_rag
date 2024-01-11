import os
os.environ["CUDA_VISIBLE_DEVICES"]="6,7"
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, PeftConfig, PeftModel
from src.utils import exact_match_score, f1_score, normalize_question, str2bool, text_has_answer, normalize_answer
from tqdm.auto import tqdm
import torch, wandb
import argparse
import numpy as np
import pandas as pd
import json
from train import make_case_text

def _preprocess_dataset(dataset, args):
    if "case" in args.dataset_name.lower():
        columns_names = dataset.column_names
        if "case" in columns_names:
            if args.num_cases == "random":
                ceiling = len(dataset["case"][0])
                return dataset.map(lambda x: {"case": sorted(x["case"], key=lambda y: float(y["distance"]), reverse=True)[:np.random.randint(1, ceiling+1)]})
            return dataset.map(lambda x: {"case": sorted(x["case"], key=lambda y: float(y["distance"]), reverse=True)[:args.num_cases]})
        else:
            for col in columns_names:
                if "case" in col:
                    dataset = dataset.rename_column(col, "case")
                    if args.num_cases == "random":
                        ceiling = len(dataset["case"][0])
                        return dataset.map(lambda x: {"case": sorted(x["case"], key=lambda y: float(y["distance"]), reverse=True)[:np.random.randint(1, ceiling+1)]})
                    return dataset.map(lambda x: {"case": sorted(x["case"], key=lambda y: float(y["distance"]), reverse=True)[:args.num_cases]})
    else:
        return dataset

def formatting_for_evaluation(row, args):
    ctxs = "\n".join([f"Doc {idx}: {ctx['text']}" for idx, ctx in enumerate(row["ctxs"][:args.num_contexts])])
    q = normalize_question(row['question'])
    if args.num_cases > 0:
        case_text = make_case_text(row["case"][:args.num_cases])
        if args.num_contexts == 0:
            text = case_text + f"### Q: {q}\n### A:"
        else:
            text = case_text + f"### Background:\n{ctxs}\n\n ### Q: {q}\n### A:"
    else:
        if args.num_contexts == 0:
            text = f"### Q: {q}\n### A:"
        else:
            text = f"### Background:\n{ctxs}\n\n ### Q: {q}\n### A:"
    return text

def main(args):    
    dataset = load_dataset(args.dataset_name, split="test")
    if args.revision is not None:
        config = PeftConfig.from_pretrained(args.model, revision=args.revision)
        basemodel = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(basemodel, args.model, config=config)
    else:
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.model, device_map="auto", torch_dtype=torch.bfloat16
        )
    # model = model.merge_and_unload() 
    # if args.revision is not None:
    #     config = PeftConfig.from_pretrained(args.model, revision=args.revision)
    # else:
    #     config = PeftConfig.from_pretrained(args.model)
    # inference_model = AutoModelForCausalLM.from_pretrained(
    #     config.base_model_name_or_path, device_map="auto"
    # )
    # model = PeftModel.from_pretrained(inference_model, args.model)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    config = GenerationConfig(
        do_sample=False
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        generation_config=config,
        framework="pt"
    )
    run_name = args.model.split("/")[-1]
    if args.test:
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.select(range(10))
        run_name += "-test"
    dataset = _preprocess_dataset(dataset, args)
    result, ems, accs, f1_scores = [], [], [], []
    wandb.init(
        project='evaluate-robust-rag', 
        job_type="evaluation",
        name=run_name,
        config=vars(args)
        )
    iterator = tqdm(dataset, desc="Generating...")
    for idx, row in enumerate(iterator):
        prompt = formatting_for_evaluation(row, args)
        if idx < 5:
            print(prompt)
            print("\n\n\n\n")
        hasanswer = any([bool(ctx["hasanswer"]) for ctx in row["ctxs"][:args.num_contexts]])
        output = pipe(prompt,max_new_tokens=args.max_new_tokens)[0]["generated_text"]
        output_wo_prompt = output[len(prompt):]
        is_em = exact_match_score(output_wo_prompt, row["answers"])
        is_acc = int(text_has_answer(row["answers"], output_wo_prompt))
        f1_score_ = f1_score(output_wo_prompt, row["answers"])
        ems.append(is_em)
        accs.append(is_acc)
        result.append([row["question"], ", ".join(row["answers"]), prompt, output_wo_prompt, is_em, is_acc, f1_score_, hasanswer])
        iterator.set_description(desc=f"EM : {sum(ems)/len(ems):.2f} | Acc : {sum(accs)/len(accs):.2f}")
    df = pd.DataFrame(result, columns=["Question", "Answers", "Prompt", "Prediction", "EM", "ACC", "F1", "hasanswer"])
    data = df[["EM", "ACC", "F1", "hasanswer"]].mean().to_dict()
    data = {k:round(v*100,2) for k,v in data.items()}
    metric = wandb.Table(dataframe=pd.DataFrame(index=[0], data=data))
    table = wandb.Table(dataframe=df)
    wandb.log({"raw_output": table, "metrics": metric})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="Atipico1/NQ")
    parser.add_argument("--model", type=str, default="Atipico1/NQ-base-v4")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--num_cases", type=int, default=0)
    parser.add_argument("--num_contexts", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--test", type=str2bool, default=False)
    args = parser.parse_args()
    main(args)



    # config = PeftConfig.from_pretrained(args.model, revision=args.revision)
    # inference_model = AutoModelForCausalLM.from_pretrained(
    #     config.base_model_name_or_path, device_map="auto"
    # )
    # model = PeftModel.from_pretrained(inference_model, args.model)
    
            # output = tokenizer(prompt, return_tensors="pt").to(args.device)
        # output = model.generate(**output, max_new_tokens=args.max_new_tokens)
        # pred = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        # pred_wo_prompt = pred[len(prompt):]
        # pred_wo_prompt_first_line = pred_wo_prompt.split("\n")[0]
        # is_em = exact_match_score(pred_wo_prompt_first_line, a, normalize_answer)
        # is_acc = int(text_has_answer(a, pred_wo_prompt_first_line))
        # result.append([q,a,prompt, pred,pred_wo_prompt,pred_wo_prompt_first_line, is_em, is_acc])