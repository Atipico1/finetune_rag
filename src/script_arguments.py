from dataclasses import dataclass, field
from typing import Optional, Union

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco", metadata={"help": "the dataset name"}
    )
    dataset_text_field: Optional[str] = field(default="prompt", metadata={"help": "the text field of the dataset"})
    run_name: Optional[str] = field(default="run", metadata={"help": "the name of the run"})
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    weight_decay: Optional[float] = field(default=0.001)
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="linear",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
        },
    )
    warmup_ratio: Optional[float] = field(default=0.01, metadata={"help": "the warmup ratio"})
    packing: Optional[bool] = field(default=False, metadata={"help": "use packing"})
    cal_max_len: Optional[bool] = field(default=False, metadata={"help": "calculate max length of the dataset"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "load the model in 4 bits precision"})
    output_dir: Optional[str] = field(default="/data/seongilpark/checkpoints/", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=100, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_strategy: Optional[str] = field(default="epoch", metadata={"help": "The checkpoint save strategy to use."})
    num_contexts: Optional[int] = field(default=1, metadata={"help": "Number of contexts to use for training"})
    num_cases: Optional[Union[int, str]] = field(default=0, metadata={"help": "Number of cases to use for training"})
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
    push_to_hub: Optional[bool] = field(default=True, metadata={"help": "Push the model to HF Hub"})
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use gradient checkpointing or no"}
    )
    dataset_split: Optional[str] = field(
        default="train", metadata={"help": "The dataset split to use for training"}
    )
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "key word arguments to be passed along `torch.utils.checkpoint.checkpoint` method - e.g. `use_reentrant=False`"
        },
    )
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})
    resume_from_checkpoint: Optional[bool] = field(default=False, metadata={"help": "Resume from checkpoint"})