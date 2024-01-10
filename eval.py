from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from datasets import load_dataset
from peft import PeftModel, PeftConfig
from src.utils import exact_match_score, normalize_question, str2bool, text_has_answer, normalize_answer
from tqdm.auto import tqdm
import torch
import argparse
import os


def preprocess_dataset(dataset, args):
    if "case" in args.dataset_name.lower():
        columns_names = dataset.column_names
        if "case" in columns_names:
            return dataset.map(lambda x: {"case": sorted(x["case"], key=lambda y: float(y["distance"]), reverse=True)[:args.num_cases]})
        else:
            for col in columns_names:
                if "case" in col:
                    dataset = dataset.rename_column(col, "case")
                    return dataset.map(lambda x: {"case": sorted(x["case"], key=lambda y: float(y["distance"]), reverse=True)[:args.num_cases]})
    else:
        return dataset

def formatting_for_evaluation(row):
    ctxs = "\n".join([f"Doc {idx}: {ctx['text']}" for idx, ctx in enumerate(row["ctxs"])])
    q = normalize_question(row['question'])
    text = f"### Background:\n{ctxs}\n\n ### Q: {q}\n### A:"
    return text

def main(args):    
    dataset = load_dataset(args.dataset_name, split="test")
    from peft import AutoPeftModelForCausalLM

    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16
    )
    model = model.merge_and_unload() 
    # config = PeftConfig.from_pretrained(args.model, revision=args.revision)
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
    if args.test:
        dataset = dataset.shuffle()
        dataset = dataset.select(range(20))
    dataset = preprocess_dataset(dataset, args)
    result,ems, accs = [], [], []
    iterator = tqdm(dataset, desc="Generating...")
    with open('log.txt', 'w') as file:

        for row in iterator:
            prompt = formatting_for_evaluation(row)
            output = pipe(prompt,max_new_tokens=args.max_new_tokens)[0]["generated_text"]
            output_wo_prompt = output[len(prompt):]
            is_em = exact_match_score(output_wo_prompt, row["answers"])
            is_acc = int(text_has_answer(row["answers"], output_wo_prompt))
            ems.append(is_em)
            accs.append(is_acc)
            file.write(f"Answer : {', '.join(row['answers'])}\nPredict : {output_wo_prompt}\n\n")
            result.append([row["question"], row["answers"], prompt, output, output_wo_prompt, is_em, is_acc])
            iterator.set_description(desc=f"EM : {sum(ems)/len(ems):.2f} | Acc : {sum(accs)/len(accs):.2f}")
        # output = tokenizer(prompt, return_tensors="pt").to(args.device)
        # output = model.generate(**output, max_new_tokens=args.max_new_tokens)
        # pred = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        # pred_wo_prompt = pred[len(prompt):]
        # pred_wo_prompt_first_line = pred_wo_prompt.split("\n")[0]
        # is_em = exact_match_score(pred_wo_prompt_first_line, a, normalize_answer)
        # is_acc = int(text_has_answer(a, pred_wo_prompt_first_line))
        # result.append([q,a,prompt, pred,pred_wo_prompt,pred_wo_prompt_first_line, is_em, is_acc])
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="Atipico1/NQ")
    parser.add_argument("--model", type=str, default="Atipico1/NQ-base-v4")
    parser.add_argument("--revision", type=str, default="841c65352910c4a9c4ed81b5a41383897748a37c")
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--test", type=str2bool, default=False)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args)