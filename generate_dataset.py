import argparse
from src.utils import str2bool
from datasets import load_dataset, DatasetDict

def generate_unans_dataset(args):
    dataset = load_dataset(args.dataset)
    train = dataset["train"].map(lambda x: {"answers": x["answers"] if any([ctx["hasanswer"] for ctx in x["ctxs"]]) else ["unanswerable"]}, num_proc=8)
    test = dataset["test"].map(lambda x: {"answers": x["answers"] if any([ctx["hasanswer"] for ctx in x["ctxs"]]) else ["unanswerable"]}, num_proc=8)
    DatasetDict({"train":train, "test":test}).push_to_hub(f"Atipico1/{args.dataset.split('/')[-1]}_unans")

def generate_conflict_dataset(args):
    dataset = load_dataset(args.dataset)
    train = dataset["train"].map(lambda x: {"answers": x["answers"] if len(x["answers"]) > 1 else ["conflict"]}, num_proc=8)
    test = dataset["test"].map(lambda x: {"answers": x["answers"] if len(x["answers"]) > 1 else ["conflict"]}, num_proc=8)
    DatasetDict({"train":train, "test":test}).push_to_hub(f"Atipico1/{args.dataset.split('/')[-1]}_conflict")

if __name__=="__main__":
    # Main parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mrqa")
    parser.add_argument("--test", type=str2bool, default=False)
    # Make subparsers
    subparsers = parser.add_subparsers(dest="task", help='Possible tasks: all | unans | adversary | conflict')

    # Preprocess parser
    unans_parser = subparsers.add_parser('unans', help='Generate unanswerable dataset')
    # unans_parser.add_argument('--train_size', type=int, default=150)
    # unans_parser.add_argument('--valid_size', type=int, default=150)

    # Adversary parser
    adv_parser = subparsers.add_parser('adversary', help='Run task 3')
    adv_parser.add_argument('param3', help='Parameter 3 for task 3')
    args = parser.parse_args()
    if args.task == "unans":
        generate_unans_dataset(args)
    elif args.task == "conflict":
        generate_conflict_dataset(args)
    elif args.task == "adversarial-unans":
        generate_adversarial_unans_dataset(args)
    elif args.task == "adversarial-conflict":
        generate_adversarial_conflict_dataset(args)
    elif args.task == "all":
        generate_all(args)
    else:
        raise ValueError(f"Invalid task: {args.task}")