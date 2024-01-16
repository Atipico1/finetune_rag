import torch, wandb
import numpy as np
from dataset import preprocess_cbr_dataset, determine_answerable
from src.script_arguments import ScriptArguments
from src.utils import normalize_question
from datasets import load_dataset, Dataset
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer,
    DataCollatorForLanguageModeling
)
# os.environ["WANDB_PROJECT"] = "finetune-robust-rag"

class QADataCollator(DataCollatorForLanguageModeling):
    answer_start_token_id = 835  # "###" token id

    def __call__(self, examples):
        batch = super().__call__(examples)

        # Only apply cross entropy loss to the answer part of the labels
        for idx, label in enumerate(batch["labels"]):
            answer_end = torch.where(label == -100)[0][0]
            answer_start = torch.where(label == self.answer_start_token_id)[0][-1]
            label[:answer_start+3] = -100
            label[answer_end] = 2
            batch["labels"][idx] = label

        return batch
    
def make_case_text(case_exs):
    output = "[CASE]\n"
    for case_ex in case_exs:
        q, c, a = case_ex["question"], case_ex["context"], case_ex["answer"]
        output += f"Background:\nDoc 0: {c}\nQ: {q}\nA: {a}\n\n"
    output += "[/CASE]\n\n"
    return output

def formatting_for_original(example):
    output_texts = []
    for i in range(len(example['question'])):
        q = normalize_question(example['question'][i])
        if num_contexts == 0:
            text = f"### Q: {q}\n### A: {example['answers'][i][0]}</s>"
        else:
            ctxs = "\n".join([f"Doc {idx}: {ctx['text']}" for idx, ctx in enumerate(example["ctxs"][i][:num_contexts])])
            text = f"### Background:\n{ctxs}\n\n### Q: {q}\n### A: {example['answers'][i][0]}</s>"
        output_texts.append(text)
    return output_texts

def formatting_for_cbr(example):
    output_texts = []
    for i in range(len(example['question'])):
        q = normalize_question(example['question'][i])
        ctxs = "\n".join([f"Doc {idx}: {ctx['text']}" for idx, ctx in enumerate(example["ctxs"][i][:num_contexts])])  
        case_text = make_case_text(example["case"][i])
        text = case_text + f"### Background:\n{ctxs}\n\n ### Q: {q}\n### A: {example['answers'][i][0]}</s>"
        output_texts.append(text)
    return output_texts
    
def main(args):
    global num_contexts
    num_contexts = args.num_contexts
    try:
        args.num_cases = int(args.num_cases)
    except:
        pass
    dataset = load_dataset(args.dataset_name, split="train")
    if args.cbr:
        dataset = preprocess_cbr_dataset(dataset, args)
    response_template = "### A:"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.unk_token
    #collator = DataCollatorForCompletionOnlyLM(tokenizer.encode(response_template, add_special_tokens = False)[2:], tokenizer=tokenizer, mlm=False)
    collator = QADataCollator(tokenizer=tokenizer, mlm=False)
    if args.unanswerable:
        dataset = dataset.map(lambda x: {"answers": determine_answerable(x)})
    formatting_func = formatting_for_cbr if args.cbr else formatting_for_original
    wandb.init(
        project='finetune-robust-rag', 
        job_type="training",
        name=args.run_name if not args.test else "test",
        config=vars(args)
        )
    samples = formatting_func(dataset[:20])
    max_length = max(tokenizer(formatting_func(dataset[:]), return_length=True)["length"])
    print("Max length: ", max_length)
    sample_table = wandb.Table(columns=["idx", "sample"], data=[[i, sample]for i, sample in enumerate(samples)])
    wandb.log({"samples": sample_table})
    if args.test:
        raise ValueError("Test mode")
    bnb_config = BitsAndBytesConfig(
            load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        attn_implementation="flash_attention_2")
    training_args = TrainingArguments(
        output_dir=args.output_dir+args.run_name,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        optim=args.optim,
        bf16=True,
        group_by_length=True,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        hub_strategy="checkpoint",
        save_total_limit=args.save_total_limit,
        push_to_hub=args.push_to_hub,
        hub_model_id=f"Atipico1/{args.run_name}",
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        gradient_checkpointing=args.gradient_checkpointing)
    
    lora_config = LoraConfig(
                r=args.peft_lora_r,
                lora_alpha=args.peft_lora_alpha,
                bias="none",
                task_type="CAUSAL_LM"
                )
    trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=lora_config,
            max_seq_length=args.seq_length,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=collator,
            formatting_func=formatting_func,
        )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    main(script_args)