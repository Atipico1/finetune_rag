# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from datasets import load_dataset
from dataset import preprocess_dataset
from peft import AutoPeftModelForCausalLM, PeftConfig, PeftModel
from src.utils import exact_match_score, f1_score, normalize_question, str2bool, text_has_answer, normalize_answer
from tqdm.auto import tqdm
import torch, wandb
import argparse
import pandas as pd
import time
from dataset import make_case_text, make_custom_case_text

NAME_TO_PATH = {
    "NQ": "Atipico1/NQ_preprocessed_with_o-u_case",
    "TQA": "Atipico1/trivia-top5_preprocessed_with_o-u_case",
    "WEBQ": "Atipico1/webq-top5_preprocessed_with_o-u_case"
    }
PATH_TO_NAME = {v:k for k,v in NAME_TO_PATH.items()}

def formatting_for_evaluation(row, args):
    ctxs = "\n".join([f"Doc {idx}: {ctx['text']}" for idx, ctx in enumerate(row["ctxs"][:args.num_contexts])])
    q = normalize_question(row['question'])
    if args.cbr > 0:
        if args.custom_loss:
            case_text = make_custom_case_text(row["case"])
        else:
            case_text = make_case_text(row["case"])
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

def evaluate(pipe, dataset, run_name:str, args):
    result, ems, accs, f1_scores = [], [], [], []
    iterator = tqdm(dataset, desc="Generating...")
    with torch.no_grad():
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
            iterator.set_description(desc=f"EM : {sum(ems)/len(ems)*100:.2f} | Acc : {sum(accs)/len(accs)*100:.2f}")
    df = pd.DataFrame(result, columns=["Question", "Answers", "Prompt", "Prediction", "EM", "ACC", "F1", "hasanswer"])
    data = df[["EM", "ACC", "F1", "hasanswer"]].mean().to_dict()
    answerable_em, unanswerable_em = df[df["hasanswer"] == True]["EM"].mean(), df[df["hasanswer"] == False]["EM"].mean()
    answerable_acc, unanswerable_acc = df[df["hasanswer"] == True]["ACC"].mean(), df[df["hasanswer"] == False]["ACC"].mean()
    data.update({"answerable_EM": answerable_em,
                 "unanswerable_EM": unanswerable_em,
                 "answerable_ACC": answerable_acc,
                 "unanswerable_ACC": unanswerable_acc})
    data = {k:round(v*100,2) for k,v in data.items()}
    wandb.init(
        project='evaluate-robust-rag', 
        job_type="evaluation",
        name=run_name,
        config=vars(args)
        )
    metric = wandb.Table(dataframe=pd.DataFrame(index=[0], data=data))
    table = wandb.Table(dataframe=df)
    wandb.log({"raw_output": table, "metrics": metric})
    wandb.finish()

def main(args):
    if args.sleep:
        time.sleep(60)
    config = PeftConfig.from_pretrained(args.model)
    basemodel = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(basemodel, args.model, config=config)
    if args.save:
        model_name= args.model.split("/")[-2]
        model.push_to_hub(f"Atipico1/{model_name}")
        raise ValueError("Model saved to hub")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.unk_token
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
    model_name = args.model.split("/")[-2] if len(args.model.split("/")) > 2 else args.model.split("/")[-1]
    if args.test:
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.select(range(10))
        run_name += "-test"
    
    if args.datasets != []:
        for dataset_name in args.datasets:
            dataset = load_dataset(NAME_TO_PATH[dataset_name], split="test")
            dataset = preprocess_dataset(dataset, args, "test")
            run_name = f"{dataset_name}-{model_name}"
            if args.prefix:
                run_name = f"{args.prefix}-{run_name}"
            evaluate(pipe, dataset, run_name, args)
    else:
        dataset = load_dataset(args.dataset_name, split="test")
        dataset = preprocess_dataset(dataset, args, "test")
        dataset_name = PATH_TO_NAME[args.dataset_name]
        run_name = f"{dataset_name}-{model_name}"
        evaluate(pipe, dataset, run_name, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="Atipico1/NQ")
    parser.add_argument("--datasets", type=str,nargs="+", default=[])
    parser.add_argument("--model", type=str, default="Atipico1/NQ-base-v4")
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--save", type=str2bool, default=False)
    parser.add_argument("--unanswerable", type=str2bool, default=False)
    parser.add_argument("--cbr", type=str2bool, default=False)
    parser.add_argument("--cbr_original", type=int, default=0)
    parser.add_argument("--cbr_unans", type=int, default=0)
    parser.add_argument("--cbr_adv", type=int, default=0)
    parser.add_argument("--cbr_conflict", type=int, default=0)
    parser.add_argument("--sleep", type=str2bool, default=False)
    parser.add_argument("--num_contexts", type=int, default=5)
    parser.add_argument("--custom_loss", type=str2bool, default=False)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--test", type=str2bool, default=False)
    parser.add_argument("--only_has_answer", type=str2bool, default=False)
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