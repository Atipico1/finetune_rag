import argparse, os, wandb, torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from dataset import preprocess_dataset
from datasets import load_dataset
from eval import formatting_for_evaluation
from src.utils import exact_match_score, str2bool, f1_score, text_has_answer
import pandas as pd

NAME_TO_PATH = {
    #"NQ": "Atipico1/NQ_preprocessed_with_o-u_case",
    #"NQ":"Atipico1/NQ_preprocessed_with_o-u-c_case",
    #"NQ":"Atipico1/NQ-colbert-10k-case",
    "NQ":"Atipico1/nq-test-adv-replace-v2",
    "TQA": "Atipico1/trivia-top5_preprocessed_with_o-u_case",
    "WEBQ": "Atipico1/webq-top5_preprocessed_with_o-u_case"
    }
PATH_TO_NAME = {v:k for k,v in NAME_TO_PATH.items()}

    
def save_results(dataset, ems, f1s, accs,run_name, args):
    df = pd.DataFrame({
        "Question": dataset["question"],
        "Answers": dataset["answers"],
        "Prompt": dataset["prompt"],
        "Prediction": dataset["Prediction"],
        "hasanswer": dataset["hasanswer"],
        "EM": ems,
        "F1": f1s,
        "Acc": accs
    })
    answerable_em, unanswerable_em = df[df["hasanswer"] == True]["EM"].mean(), df[df["hasanswer"] == False]["EM"].mean()
    answerable_acc, unanswerable_acc = df[df["hasanswer"] == True]["Acc"].mean(), df[df["hasanswer"] == False]["Acc"].mean()
    data = df[["EM", "Acc", "F1", "hasanswer"]].mean().to_dict()
    data.update({"EM (ans)": answerable_em,
                 "EM (unans)": unanswerable_em,
                 "Acc (ans)": answerable_acc,
                 "Acc (unans)": unanswerable_acc})
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

def evaluate(dataset, run_name, args):
    llm = LLM(model="meta-llama/Llama-2-7b-hf",
              tensor_parallel_size=torch.cuda.device_count(),
              seed=42,
              enable_lora=True,
              max_lora_rank=64)
    dataset = dataset.map(lambda x: {"prompt":formatting_for_evaluation(x, args)}, num_proc=os.cpu_count())
    sampling_params = SamplingParams(temperature=0, max_tokens=20)
    outputs = llm.generate(dataset["prompt"],sampling_params,lora_request=LoRARequest("lora", 1, args.model))
    outputs = [o.outputs[0].text.strip() for o in outputs]
    dataset = dataset.add_column("Prediction",outputs)
    ems = [exact_match_score(pred, label) for pred, label in zip(outputs, dataset["answers"])]
    f1s = [f1_score(pred, label) for pred, label in zip(outputs, dataset["answers"])]
    accs = [text_has_answer(label, pred) for label, pred in zip(dataset["answers"], outputs)]
    save_results(dataset, ems, f1s, accs, run_name, args)
    
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
    parser.add_argument("--conflict", type=str2bool, default=False)
    parser.add_argument("--conflict_only", type=str2bool, default=True)
    parser.add_argument("--both", type=str2bool, default=False)
    parser.add_argument("--anonymize", type=str2bool, default=False)
    args = parser.parse_args()
    model_name = args.model.split("/")[-2] if len(args.model.split("/")) > 2 else args.model.split("/")[-1]
    for dataset_name in args.datasets:
        dataset = load_dataset(NAME_TO_PATH[dataset_name], split="test")
        dataset = preprocess_dataset(dataset, args, "test")
        run_name = f"{dataset_name}-{model_name}"
        if args.test:
            dataset = dataset.shuffle(seed=42)
            dataset = dataset.select(range(10))
            run_name += "-test"
        if args.prefix:
            run_name = f"{args.prefix}-{run_name}"
        evaluate(dataset, run_name, args)