import faiss
import os
from dataset import (
    split_sentence_and_make_short_context,
    preprocess_text,
    annotate_answer_type,
    ner,
    gen_entity_vector,
    generate_entity_pool
)
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
import joblib
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import spacy, torch
import argparse
import wandb
from src.utils import (
    str2bool,
    cal_cosine_similarities,
    normalize_question
)
from src.index import build_index_with_ids
from typing import List
from preprocess import query_masking, remove_duplicate, query_embedding, remove_duplicate_by_similarity
from transformers import AutoTokenizer, DPRQuestionEncoder
from src.search import (
    find_similar_questions,
    find_similar_contexts,
    find_similar_contexts_with_questions,
    find_random_contexts,
    find_similar_entity,
    find_random_entity
)

def _load_dataset(args):
    if args.task == "preprocess":
        return load_mrqa(args)
    else:
        if args.split != "all":
            dataset = load_dataset(args.dataset, split="train")
            if args.test:
                return dataset.shuffle(seed=42).select(range(args.test_size))
            print(f"{args.dataset} Loaded! -> Size : {len(dataset)}")
        else:
            dataset = load_dataset(args.dataset)
            if args.test:
                for subset in dataset.keys():
                    dataset[subset] = dataset[subset].shuffle(seed=42).select(range(args.test_size))
                    print(f"{args.dataset} {subset} Loaded! -> Size : {len(dataset[subset])}")
        return dataset

def load_mrqa(args):
    mrqa = load_dataset("mrqa")
    train, test, valid = mrqa["train"], mrqa["test"], mrqa["validation"]
    mrqa = concatenate_datasets([train, test, valid])
    mrqa: Dataset = mrqa.filter(lambda x: x["subset"] not in args.except_subset, num_proc=os.cpu_count())
    print(f"MRQA Loaded without {args.except_subset} ! -> Size : {len(mrqa)}")
    if args.test:
        mrqa = mrqa.shuffle(seed=42).select(range(10000))
        print(f"MRQA-TEST Loaded ! -> Size : {len(mrqa)}")
    return mrqa

def preprocess(args, mrqa):
    nlp = spacy.load(args.spacy_model)
    tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to("cuda")
    mrqa = query_masking(nlp, mrqa)
    del nlp
    nlp = spacy.load(args.spliter_model)
    mrqa = remove_duplicate(mrqa)
    mrqa = preprocess_text(mrqa, args)
    mrqa = split_sentence_and_make_short_context(mrqa, nlp, args)
    mrqa = query_embedding(model, tokenizer, mrqa, args)
    del nlp
    mrqa = remove_duplicate_by_similarity(mrqa, tokenizer, model, args)
    if args.test:
        sample = mrqa.shuffle().select(range(10))
        for row in sample:
            q, sc, a, c = row["question"], row["short_context"], row["answers"], row["context"]
            print(f"Question : {q}\nShort Context : {sc}\nAnswer : {a}\nContext : {c}\n\n\n\n")
    else:
        mrqa = mrqa.remove_columns(["context","context_tokens","question_tokens","detected_answers"])
        mrqa = mrqa.rename_column("short_context", "context")
        mrqa.push_to_hub(f"Atipico1/mrqa_preprocessed_thres-{args.remove_duplicate_thres}_by-{args.sentence_embedder}")

def generate_unans(args, dataset):
    dataset, q_embs = find_similar_questions(dataset, args)
    dataset, c2embs = find_similar_contexts(dataset, args)
    dataset = find_similar_contexts_with_questions(q_embs, c2embs, dataset, args)
    dataset = find_random_contexts(dataset)
    df = pd.DataFrame(dataset)
    df["answer_in_context"] = df["answer_in_context"].apply(lambda x: ", ".join(x))
    df = df[["question","context","answer_in_context","Q_similar_context","C_similar_context","QC_similar_context","random_context"]].sample(100)
    try:
        wandb.init(project="craft-cases", name="unanswerable" if not args.test else "test-unanswerable", config=vars(args))
        wandb.log({"samples": wandb.Table(dataframe=df)})
    except:
        df.to_csv("unanswerable_result.csv") 
    if not args.test:
        dataset.push_to_hub("Atipico1/mrqa_v2_unanswerable" if args.save_dir == "" else args.save_dir)

def generate_similar_context(args, dataset):
    from vllm import LLM, SamplingParams
    from src.search import (
    has_answer,
    SimpleTokenizer
)
    raw_text = """Rewrite the following ariticle, maintaining its original meaning. Do not add any new information. Keep the specified words unchanged.
Words to Keep Unchanged: {ANSWERS}
Original Article: {CONTEXT}
Rewritten Article:"""
    def make_prompt(ans, ctx, key: str):
        if key=="solar":
            prompt = raw_text.format(ANSWERS=", ".join(ans), CONTEXT=ctx)
            #prompt = tokenizer(prompt, return_tensors="pt").to(model.device)
            return prompt
        else:
            pass
    #dataset = load_dataset("Atipico1/mrqa_preprocessed_with_substitution", split="train")
    if args.test:
        dataset = dataset.shuffle(seed=42).select(range(100))
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    llm = LLM(model=args.lm, tensor_parallel_size=2, seed=42)
    generated_texts = []
    answers, contexts = dataset["answer_in_context"], dataset["context"]
    with torch.no_grad():
        for i in tqdm(range(0, len(answers), args.batch_size), desc="Generating Rewrites"):
            batch_answers = answers[i:i+args.batch_size]
            batch_contexts = contexts[i:i+args.batch_size]
            prompts = [make_prompt(ans, ctx, key="solar") for ans, ctx in zip(batch_answers, batch_contexts)]
            outputs = llm.generate(prompts, sampling_params)
            for output in outputs:
                generated_texts.append(output.outputs[0].text.strip())
    dataset = dataset.add_column("rewritten_context", generated_texts)
    tokenizer = SimpleTokenizer()
    dataset = dataset.map(lambda x: {"valid": has_answer(x["answer_in_context"], x["rewritten_context"], tokenizer)}, num_proc=8)
    wandb.init(project="craft-cases", name="similar-context" if not args.test else "test-similar-context", config=vars(args))
    df = pd.DataFrame(dataset)
    df["answer_in_context"] = df["answer_in_context"].apply(lambda x: ", ".join(x))
    df = df[["question","answer_in_context","context","rewritten_context","valid"]]
    wandb.log({"samples": wandb.Table(dataframe=df.sample(100)),
              "valid_ratio": df["valid"].mean()})
    if not args.test:
        dataset.push_to_hub("Atipico1/mrqa_preprocessed_with_substitution-rewritten")                

def generate_support_context(args):
    pass
    
def generate_adversary(args, dataset):
    raise NotImplementedError("This function is not implemented yet")

def generate_similar_entity(args, dataset: Dataset):
    if args.entity_vector_path:
        entities_groupby_type = joblib.load(args.entity_vector_path.replace("_vec.pkl", ".pkl"))
        entity_vector_groupby_type = joblib.load(args.entity_vector_path)
    else:
        entities_groupby_type, entity_vector_groupby_type = generate_entity_pool(dataset, args)
    dataset = ner(dataset, args, "context")
    dataset = gen_entity_vector(dataset, args, args.ans_col)
    index_per_entity = {}
    for k, v in entity_vector_groupby_type.items():
        v = np.array(v).astype('float32')
        index_per_entity[k] = build_index_with_ids(v, save_dir="", name=k, is_save=False, gpu_id=args.gpu_id)
    similar_entities, similar_scores = find_similar_entity(dataset, args, entities_groupby_type, index_per_entity)
    dataset = dataset.add_column("similar_entity",  similar_entities)
    dataset = dataset.add_column("similar_entity_score", similar_scores)
    dataset = dataset.filter(lambda x: x["similar_entity"] is not None)
    random_entities = find_random_entity(dataset, entities_groupby_type, args)
    dataset = dataset.add_column("random_entity", random_entities)
    dataset = dataset.filter(lambda x: x["random_entity"] is not None)
    random_scores = cal_cosine_similarities(dataset["entity_vector"], dataset["random_entity"], args)
    dataset = dataset.add_column("random_entity_score", random_scores)
    dataset = dataset.remove_columns(["entity_vector"])
    if not args.test and args.save_dir != "":
        if args.split != "all":
            dataset.push_to_hub(args.save_dir)
            return None
        else:
            return dataset
    else:
        return None

def generate_conflict_context(args, dataset):
    from vllm import LLM, SamplingParams
    from src.search import (
    has_answer,
    SimpleTokenizer
)
    sentence_format = """Please write a single sentence claim using the follwoing question and answer. The claim should include the answer and be as realistic as possible:
Question: {QUESTION}
Answer: {ANSWER}
Claim:"""
    claim_format = """Given a claim, please write a concise, factual passage to support it. You can make up fake content and supporting evidence but it should be as realistic as possible. The passage must be less than 100 words.
Claim: {CLAIM}
Passage:"""
    if args.test:
        dataset = dataset.shuffle(seed=42).select(range(1000))
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    llm = LLM(model=args.lm, tensor_parallel_size=2, seed=42)
    generated_texts = []
    tokenizer = SimpleTokenizer()
    questions, answers, answer_sents, substitutions = dataset["question"], dataset["answer_in_context"], dataset["answer_sent"], dataset["similar_entity"]
    single_sentence_claim_prompts = [sentence_format.format(QUESTION=normalize_question(question), ANSWER=answer) for question, answer in zip(questions, substitutions)]
    with torch.no_grad():
        for i in tqdm(range(0, len(single_sentence_claim_prompts), args.batch_size), desc="Generating Rewrites"):
            batch = single_sentence_claim_prompts[i:i+args.batch_size]
            outputs = llm.generate(batch, sampling_params)
            for output in outputs:
                generated_texts.append(output.outputs[0].text.strip())
    dataset = dataset.add_column("conflict_claim", generated_texts)
    dataset = dataset.map(lambda x: {"conflict_valid": has_answer([x["similar_entity"]], x["conflict_claim"], tokenizer)}, num_proc=8)
    dataset = dataset.filter(lambda x: x["conflict_valid"], num_proc=8)
    print("After filtering:", len(dataset))
    dataset = dataset.remove_columns(["conflict_valid"])
    #claims = [update_context_with_substitution_string(sent, ans, sub) for ans, sent, sub in zip(answers, answer_sents, substitutions)]
    claims = dataset["conflict_claim"]
    generated_texts = []
    with torch.no_grad():
        for i in tqdm(range(0, len(claims), args.batch_size), desc="Generating Rewrites"):
            batch = claims[i:i+args.batch_size]
            prompts = [claim_format.format(CLAIM=claim) for claim in batch]
            outputs = llm.generate(prompts, sampling_params)
            for output in outputs:
                generated_texts.append(output.outputs[0].text.strip())
    dataset = dataset.add_column("conflict_context", generated_texts)
    #dataset = dataset.add_column("conflict_claim", claims)
    dataset = dataset.map(lambda x: {"conflict_valid": has_answer([x["similar_entity"]], x["conflict_context"], tokenizer)}, num_proc=8)
    wandb.init(project="craft-cases", name="conflict-context" if not args.test else "test-conflict-context", config=vars(args))
    df = pd.DataFrame(dataset)
    df["answer_in_context"] = df["answer_in_context"].apply(lambda x: ", ".join(x))
    df = df[["question","answer_in_context","context","conflict_claim", "conflict_context","similar_entity", "conflict_valid"]]
    wandb.log({"samples": wandb.Table(dataframe=df.sample(100)),
              "valid_ratio": df["conflict_valid"].mean()})
    if not args.test:
        dataset.push_to_hub("Atipico1/mrqa_preprocessed_with_substitution-rewritten-conflict")  
def generate_all(args):
    pass

if __name__=="__main__":
    # Main parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mrqa")
    parser.add_argument("--save_dir", type=str, default="", help="hub path or local path to save")
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--test", type=str2bool, default=False)
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--spacy_model", type=str, default="en_core_web_trf")
    parser.add_argument("--spliter_model", type=str, default="en_core_web_sm")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--split", type=str, default="")
    # Make subparsers
    subparsers = parser.add_subparsers(dest="task", help='Possible tasks: preprocess | all | unans | adversary | conflict')

    # Preprocess parser
    preprocess_parser = subparsers.add_parser('preprocess', help='Run preprocess')
    preprocess_parser.add_argument('--ctx_min_len', type=int, default=50)
    preprocess_parser.add_argument('--ctx_avg_len', type=int, default=70)
    preprocess_parser.add_argument("--ctx_max_len", type=int, default=120)
    preprocess_parser.add_argument(
        '--except_subset', nargs='+', type=str, default='SearchQA',
        help='Possible Subsets: SearchQA | SQuAD | NaturalQuestionsShort | HotpotQA | NewsQA'
        )
    preprocess_parser.add_argument('--sentence_embedder', type=str, default="dpr")
    preprocess_parser.add_argument('--answer_max_len', type=int, default=10)
    preprocess_parser.add_argument('--remove_duplicate_thres', type=float, default=0.95)
    # Unans parser
    unans_parser = subparsers.add_parser('unans', help='Generate unanswerable cases')
    unans_parser.add_argument('--alpha', type=float, default=0.2, help='alpha for weighted average unanswerable case generation')
    unans_parser.add_argument('--topk', type=int, default=10, help='Top k nearest neighbors for faiss search')
    # Adversary parser
    adv_parser = subparsers.add_parser('adversary', help='Run task 3')
    adv_parser.add_argument('--save_dir', help='path to save the docs', type=str, default='/data/seongilpark/adversary')
    
    # Entity parser
    entity_parser = subparsers.add_parser('entity', help='Generate similar entity')
    entity_parser.add_argument('--entity_vector_path', help='path to entity vector', type=str, default='/data/seongilpark/dataset/mrqa_valid_entities_vec.pkl')
    entity_parser.add_argument('--ans_col', help="answer column name", type=str, default='answer_in_context')
    entity_parser.add_argument("--threshold", type=float, default=0.8)
    entity_parser.add_argument("--use_wikitext", type=str2bool, default=True)
    # Similar context parser
    similar_context_parser = subparsers.add_parser('similar_context', help='Generate similar context')
    similar_context_parser.add_argument('--temperature', type=float, default=0.8)
    similar_context_parser.add_argument('--top_p', type=float, default=0.95)
    similar_context_parser.add_argument('--max_tokens', type=int, default=300)
    similar_context_parser.add_argument('--lm', type=str, default='Upstage/SOLAR-10.7B-Instruct-v1.0')
    
    # Conflict parser
    conflict_context_parser = subparsers.add_parser('conflict_context', help='Generate conflict context')
    conflict_context_parser.add_argument('--temperature', type=float, default=0.8)
    conflict_context_parser.add_argument('--top_p', type=float, default=0.95)
    conflict_context_parser.add_argument('--max_tokens', type=int, default=200)
    conflict_context_parser.add_argument('--lm', type=str, default='Upstage/SOLAR-10.7B-Instruct-v1.0')
    
    args = parser.parse_args()
    if args.use_gpu:
        if args.gpu_id >= 0:
            gpu_available=spacy.prefer_gpu(args.gpu_id)
        else:
            gpu_available=spacy.prefer_gpu(1)
        print("Spacy GPU Available:", gpu_available)
    dataset = _load_dataset(args)
    if args.task == "preprocess":
        preprocess(args, dataset)
    elif args.task == "unans":
        generate_unans(args, dataset)
    elif args.task == "adversary":
        generate_adversary(args, dataset)
    elif args.task == "similar_context":
        generate_similar_context(args, dataset)
    elif args.task == "conflict_context":
        generate_conflict_context(args, dataset)
    elif args.task == "entity":
        if args.split != "all":
            generate_similar_entity(args, dataset)
        else:
            output = {}
            for subset in dataset.keys():
                output[subset] = generate_similar_entity(args, dataset[subset])
            DatasetDict(output).push_to_hub(args.save_dir)
    else:
        raise ValueError(f"Invalid task: {args.task}")