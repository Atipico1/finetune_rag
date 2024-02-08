import time
import torch, wandb
import pandas as pd
from dataset import preprocess_dataset, get_formatting_func, CustomDataCollator
from src.script_arguments import ScriptArguments
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer
)
# os.environ["WANDB_PROJECT"] = "finetune-robust-rag"

def save_samples(dataset, args):
    sample = dataset[:50]
    formatting_func = get_formatting_func(args)
    samples = formatting_func(sample)
    sample = pd.DataFrame(sample)
    sample["prompt"] = samples
    sample["answers"] = sample["answers"].apply(lambda x: ", ".join(x) if len(x) > 1 else x[0])
    cols = ["question", "answers", "prompt", "hasanswer"]
    if args.unanswerable:
        cols.insert(2, "original_answers")
        sample["original_answers"] = sample["original_answers"].apply(lambda x: ", ".join(x) if len(x) > 1 else x[0])
    if args.conflict:
        cols.insert(2, "original_answers")
        sample["original_answers"] = sample["original_answers"].apply(lambda x: ", ".join(x) if len(x) > 1 else x[0])
        cols.append("is_conflict")
    sample = sample[cols]
    sample_table = wandb.Table(dataframe=sample)
    wandb.log({"samples": sample_table})

def main(args):
    global num_contexts
    num_contexts = args.num_contexts
    dataset = load_dataset(args.dataset_name, split="train")
    dataset = preprocess_dataset(dataset, args)
    response_template = "### A:"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.unk_token
    if args.custom_loss:
        collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)
    else:
        collator = DataCollatorForCompletionOnlyLM(tokenizer.encode(response_template, add_special_tokens = False)[2:], tokenizer=tokenizer, mlm=False)
    #collator = QADataCollator(tokenizer=tokenizer, mlm=False)

    formatting_func = get_formatting_func(args)
    wandb.init(
        project='finetune-robust-rag', 
        job_type="training",
        name=args.run_name if not args.test else "test",
        config=vars(args)
        )
    save_samples(dataset, args)
    max_length = max(tokenizer(formatting_func(dataset[:]), return_length=True)["length"])
    print("Max length: ", max_length)
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