import re, random, spacy, os, string, joblib, wandb
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from collections import defaultdict
from src.search import SimpleTokenizer, has_answer
from src.utils import (
    normalize_question,
    update_context_with_substitution_string,
    normalize_answer
)
import torch
from typing import List
from transformers import DataCollatorForLanguageModeling
import numpy as np
import pandas as pd
import datetime as dt

#ANSWER_POS = ["ADV", "ADJ", "NOUN", "NUM", "SYM", "PROPN"]
EDIT_CANDIDATE_POS = ["VERB", "NOUN", "ADJ", "ADV"]
ANSWER_POS = ["ADV", "ADJ", "NOUN", "VERB", "NUM"]
NUM_PROC = os.cpu_count()
def _preprocess_context(text, subset_name: str):
    if subset_name == "NewsQA":
        start = text.find("--")
        end = start + 2
        text = text[end:]
    if ("<Table>" in text) or ("<Ol>" in text) or ("<Li>" in text) or ("<Tr>" in text):
        return None
    text = text.replace("<P>", "")
    text = text.replace("</P>", "")
    text = text.replace("[PAR]", "")
    text = text.replace("[DOC]", "")
    text = text.replace("[TLE]", "")
    text = text.replace("[SEP]", "")
    text = re.sub('\n+', '', text)
    text = re.sub(' +', ' ', text)
    return text

def preprocess_text(dataset: Dataset, args) -> Dataset:
    dataset = dataset.map(lambda x: {"context": _preprocess_context(x["context"], x["subset"])}, num_proc=NUM_PROC, desc="Preprocessing...")
    print("Before context preprocess: ", len(dataset))
    dataset = dataset.filter(lambda x: x["context"] is not None, num_proc=NUM_PROC)
    print("After context preprocess: ", len(dataset))
    return dataset

def split_sentence_and_make_short_context(dataset: Dataset, nlp, args):
    simple_tokenizer = SimpleTokenizer()
    answer_passages = []
    answer_sents = []
    answers_in_context = []
    for i in tqdm(range(0, len(dataset), args.batch_size), desc="Splitting sentence..."):
        batch = dataset[i:i+args.batch_size]
        docs = list(nlp.pipe(batch["context"], batch_size=args.batch_size, disable=["ner"]))
        answers = batch["answers"]
        for doc, answer in zip(docs, answers):
            answer_sent_idx = -1
            sents = [sent.text for sent in doc.sents]
            for idx, sent in enumerate(sents):
                if has_answer(answer, sent, simple_tokenizer):
                    start_idx, end_idx = max(0, idx-3), min(len(sents), idx+4)
                    answer_passage = " ".join([sent.strip() for sent in sents[start_idx:end_idx]])
                    while len(answer_passage.split()) < args.ctx_avg_len:
                        if start_idx == 0 and end_idx == len(sents):
                            break
                        elif start_idx == 0 and end_idx < len(sents):
                            end_idx += 1
                            answer_passage = " ".join([sent.strip() for sent in sents[start_idx:end_idx]])
                        elif start_idx > 0 and end_idx == len(sents):
                            start_idx -= 1
                            answer_passage = " ".join([sent.strip() for sent in sents[start_idx:end_idx]])
                        else:
                            start_idx -= 1
                            end_idx += 1
                            answer_passage = " ".join([sent.strip() for sent in sents[start_idx:end_idx]])
                    answer_sents.append(sent)
                    buffer = []
                    for ans in answer:
                        if has_answer([ans], answer_passage, simple_tokenizer):
                            buffer.append(ans)
                    if buffer == []:
                        print(f"Answer not found in context!\nAnswer : {answer}\nContext : {answer_passage}")
                        answer_passages.append(None)
                        answers_in_context.append(None)
                    else:
                        answers_in_context.append(buffer)
                        answer_passages.append(answer_passage)
                    answer_sent_idx = idx
                    break
            if answer_sent_idx == -1:
                answer_sents.append(None)
                answers_in_context.append(None)
                answer_passages.append(None)
                
    assert len(list(set([len(answer_passages), len(dataset), len(answer_sents), len(answers_in_context)])))==1, "Length doesn't match"
    print("Before split: ", len(dataset))
    dataset = dataset.add_column("short_context", answer_passages)
    dataset = dataset.add_column("answer_sent", answer_sents)
    dataset = dataset.add_column("answer_in_context", answers_in_context)
    dataset = dataset.filter(lambda x: x["short_context"] is not None, num_proc=NUM_PROC)
    print("After split: ", len(dataset))
    dataset = dataset.filter(lambda x: len(x["short_context"].split())< args.ctx_max_len and len(x["short_context"].split()) > args.ctx_min_len, num_proc=NUM_PROC)
    print("After context length filtering: ", len(dataset))
    dataset = dataset.filter(lambda x: all([len(ans.split())< args.answer_max_len for ans in x["answer_in_context"]]), num_proc=NUM_PROC, desc="Max answer len filtering...")
    print("After answer length filtering: ", len(dataset))
    return dataset

def make_spacys(dataset: Dataset, nlp, args):
    total_texts = list(set(dataset["context"] + [normalize_question(q) for q in dataset["question"]]))
    docs = []
    for i in tqdm(range(0, len(total_texts), args.batch_size), desc="Making spacy docs..."):
        batch = total_texts[i:i+args.batch_size]
        docs.extend(list(nlp.pipe(batch)))
    ent2text = defaultdict(list)
    text2ent = dict()
    pos2text = defaultdict(list)
    text2pos = dict()
    for doc in docs:
        for ent in doc.ents:
            ent2text[ent.label_].append(ent.text)
        for token in doc:
            if not token.ent_type_:
                pos2text[token.pos_].append(token.text)
    for k, v in ent2text.items():
        ent2text[k] = list(set(v))
    for k, v in ent2text.items():
        for vv in v:
            text2ent[vv] = k
    pos2text = {k:list(set(v)) for k, v in pos2text.items() if k in ANSWER_POS}
    for k, v in pos2text.items():
        for vv in v:
            text2pos[vv] = k
    [print(f"{k}: {len(v)}") for k, v in ent2text.items()]
    [print(f"{k}: {len(v)}") for k, v in pos2text.items()]
    #[joblib.dump(data, f"{args.output_dir}/{data.__name__}.pkl") for data in [ent2text, text2ent, pos2text, text2pos]]
    joblib.dump(ent2text, f"{args.output_dir}/ent2text.pkl")
    joblib.dump(text2ent, f"{args.output_dir}/text2ent.pkl")
    joblib.dump(pos2text, f"{args.output_dir}/pos2text.pkl")
    joblib.dump(text2pos, f"{args.output_dir}/text2pos.pkl")
    return None

def annotate_answer_type(dataset, map_dict: dict, args):
    answer_types = []
    for answer in tqdm(dataset["answer_in_context"], desc="Annotating answer type..."):
        for ans in answer:
            if ans in map_dict["text2ent"]:
                answer_types.append(map_dict["text2ent"][ans])
                break
            elif ans in map_dict["text2pos"]:
                answer_types.append(map_dict["text2pos"][ans])
                break
            else:
                continue
    dataset = dataset.add_column("answer_type", answer_types)
    return dataset

def filter_and_sort(data):
    sorted_data = sorted(data, key=lambda x: float(x["distance"]), reverse=True)
    filtered_data = [d for d in sorted_data if "<Tr>" not in d["context"]]
    if filtered_data == []:
        print([len(d["answer"].split()) for d in sorted_data])
    return filtered_data

def aggregate_cases(example, args):
    output = []
    if args.cbr_original > 0:
        output.extend(filter_and_sort(example["original_case"])[:args.cbr_original])
    if args.cbr_unans > 0:
        output.extend(filter_and_sort(example["unans_case"])[:args.cbr_unans])
    if args.cbr_adv > 0:
        output.extend(filter_and_sort(example["adversary_case"])[:args.cbr_adv])
    if args.cbr_conflict > 0:
        output.extend(filter_and_sort(example["conflict_case"])[:args.cbr_conflict])
    random.shuffle(output)
    if len(output) == 0:
        #raise ValueError("No cases!")
        print(output)
    
    return output

def find_answer_in_context(answers: List[str], ctxs: List[str], tokenizer):
    for ans in answers:
        for ctx in ctxs:
            if has_answer([ans], ctx["text"], tokenizer):
                return ans
    return answers[0]

def add_conflict_context(example, args):
    if args.both:
        pass
    elif args.conflict_only:
        if not example["valid_conflict"]:
            return example["ctxs"]
        hasanswer_cnt = sum([int(ctx["hasanswer"]) for ctx in example["ctxs"]])
        if hasanswer_cnt > 1 and hasanswer_cnt < args.num_contexts:
            for i in range(len(example["ctxs"])):
                if not bool(example["ctxs"][i]["hasanswer"]):
                    example["ctxs"][i]["text"] = example["conflict_context"]
                    return example["ctxs"]
            print("No context without answer!")
        else:
            return example["ctxs"]

def is_conflict(example, args):
    if args.both:
        pass
    elif args.conflict_only:
        if not example["valid_conflict"]:
            return False
        hasanswer_cnt = sum([int(ctx["hasanswer"]) for ctx in example["ctxs"]])
        if hasanswer_cnt > 1 and hasanswer_cnt < args.num_contexts:
            return True
        else:
            return False

def anonymize(example, mode):
    for c in example["case"]:
        num_chars = random.randint(4, 6) if mode == "train" else random.randint(6, 8)
        random_chars = "".join([random.choice(string.ascii_letters) for _ in range(num_chars)])
        if "original_answers" not in c:
            c["original_answers"] = [c["answer"]]
        c["context"] = update_context_with_substitution_string(c["context"], c["original_answers"], random_chars)
        c["answer"] = random_chars
    return example["case"]

def preprocess_dataset(dataset, args, mode="train"):
    dataset = dataset.map(lambda x: {"question": normalize_question(x["question"])}, num_proc=NUM_PROC, desc="Normalizing question...")
    dataset = dataset.map(lambda x: {"ctxs": x["ctxs"][:args.num_contexts]}, num_proc=NUM_PROC, desc="Selecting contexts...")
    dataset = dataset.map(lambda x: {"hasanswer": determine_answerable(x)}, num_proc=NUM_PROC, desc="Determining answerable...")
    if mode=="train" and args.only_has_answer:
        dataset = dataset.filter(lambda x: x["hasanswer"], num_proc=NUM_PROC)
        print("After ONLY-HAS-ANSWER filtering: ", len(dataset))
    if mode=="train" and args.answer_in_context:
        tokenizer = SimpleTokenizer()
        dataset = dataset.map(lambda x: {"answers": [find_answer_in_context(x["answers"], x["ctxs"], tokenizer)]}, num_proc=NUM_PROC, desc="Finding answer in context...")
    if args.cbr:
        dataset = dataset.map(lambda x: {"case": aggregate_cases(x, args)}, num_proc=NUM_PROC, desc="Aggregating cases...")
        if args.anonymize:
            dataset = dataset.map(lambda x: {"case": anonymize(x, mode)}, num_proc=NUM_PROC, desc="Anonymizing...")
    if args.unanswerable:
        dataset = dataset.map(lambda x: {"original_answers": x["answers"]}, num_proc=NUM_PROC, desc="Saving original answers...")
        dataset = dataset.map(lambda x: {"answers": ["unanswerable"] if not x["hasanswer"] else x["answers"]}, num_proc=NUM_PROC, desc="Replacing answers...")
    if args.conflict:
        dataset = dataset.map(lambda x: {"ctxs": add_conflict_context(x, args)}, num_proc=NUM_PROC, desc="Add conflicting contexts...")
        dataset = dataset.map(lambda x: {"is_conflict": is_conflict(x, args)}, num_proc=NUM_PROC, desc="Determining conflict...")
        dataset = dataset.map(lambda x: {"original_answers": x["answers"]}, num_proc=NUM_PROC, desc="Saving original answers...")
        dataset = dataset.map(lambda x: {"answers": ["conflict"] if x["is_conflict"] else x["answers"]}, num_proc=NUM_PROC, desc="Replacing answers...")
        print("Number of is_conflict: ", sum(dataset["is_conflict"]))
    return dataset

def determine_answerable(example):
    return any([e["hasanswer"] for e in example["ctxs"]])

def make_case_text(case_exs):
    output = "[CASE]\n"
    for case_ex in case_exs:
        q, c, a = normalize_question(case_ex["question"]), case_ex["context"], case_ex["answer"]
        output += f"Background:\nDoc 0: {c}\nQ: {q}\nA: {a}\n\n"
    output += "[/CASE]\n\n"
    return output

def make_custom_case_text(case_exs):
    output = "[CASE]\n"
    for case_ex in case_exs:
        q, c, a = normalize_question(case_ex["question"]), case_ex["context"], case_ex["answer"]
        output += f"Background:\nDoc 0: {c.replace('####', ' ')}\nQ: {q}\n####A: {a}\n####\n\n"
    output += "[/CASE]\n\n"
    return output

def formatting_for_original(example):
    output_texts = []
    for i in range(len(example['question'])):
        if len(example["ctxs"]) == 0:
            text = f"### Q: {example['question'][i]}\n ### A: {example['answers'][i][0]}</s>"
        else:
            ctxs = "\n".join([f"Doc {idx}: {ctx['text']}" for idx, ctx in enumerate(example["ctxs"][i])])
            text = f"### Background:\n{ctxs}\n ### Q: {example['question'][i]}\n ### A: {example['answers'][i][0]}</s>"
        output_texts.append(text)
    return output_texts

def old_formatting_for_original(example):
    output_texts = []
    for i in range(len(example['question'])):
        if len(example["ctxs"]) == 0:
            text = f"### Q: {example['question'][i]}\n ### A: {example['answers'][i][0]}</s>"
        else:
            ctxs = "\n".join([f"Doc {idx}: {ctx['text']}" for idx, ctx in enumerate(example["ctxs"][i])])
            text = f"### Background:\n{ctxs}\n ### Q: {example['question'][i]}\n### A: {example['answers'][i][0]}</s>"
        output_texts.append(text)
    return output_texts

def formatting_for_cbr(example):
    output_texts = []
    for i in range(len(example['question'])):
        ctxs = "\n".join([f"Doc {idx}: {ctx['text']}" for idx, ctx in enumerate(example["ctxs"][i])])  
        case_text = make_case_text(example["case"][i])
        text = case_text + f"### Background:\n{ctxs}\n ### Q: {example['question'][i]}\n ### A: {example['answers'][i][0]}</s>"
        output_texts.append(text)
    return output_texts

def old_formatting_for_cbr(example):
    output_texts = []
    for i in range(len(example['question'])):
        ctxs = "\n".join([f"Doc {idx}: {ctx['text']}" for idx, ctx in enumerate(example["ctxs"][i])])  
        case_text = make_case_text(example["case"][i])
        text = case_text + f"### Background:\n{ctxs}\n ### Q: {example['question'][i]}\n### A: {example['answers'][i][0]}</s>"
        output_texts.append(text)
    return output_texts

def formatting_for_custom_loss(example):
    output_texts = []
    for i in range(len(example['question'])):
        ctxs = "\n".join([f"Doc {idx}: {ctx['text']}" for idx, ctx in enumerate(example["ctxs"][i])])  
        case_text = make_custom_case_text(example["case"][i])
        text = case_text + f"### Background:\n{ctxs}\n ### Q: {example['question'][i]}\n ### A: {example['answers'][i][0]}</s>"
        output_texts.append(text)
    return output_texts

def get_formatting_func(args):
    if args.cbr:
        if args.custom_loss:
            return formatting_for_custom_loss
        return formatting_for_cbr
    else:
        return formatting_for_original
    
class CustomDataCollator(DataCollatorForLanguageModeling):
    answer_start_token_id = 835  # "_```"
    case_answer_token_id = 4136
    def __call__(self, examples):
        batch = super().__call__(examples)
        for idx, label in enumerate(batch["labels"]):
            qa_answer_start = torch.where(label == 835)[0][-1]
            case_answer_token_ids = torch.where(label == 4136)[0]
            token_save = []
            label_copy = label.clone()
            for i in range(0, len(case_answer_token_ids), 2):
                case_start, case_end = case_answer_token_ids[i:i+2]
                token_save.append(label_copy[case_start+3:case_end])
            label[:qa_answer_start+3] = -100
            for i, tokens in zip(range(0, len(case_answer_token_ids), 2), token_save):
                case_start, case_end = case_answer_token_ids[i], case_answer_token_ids[i+1]
                label[case_start+3:case_end] = tokens
            batch["labels"][idx] = label
        return batch
"""
1. text를 입력으로 받아서
2. text의 pos를 알아낸 다음 (or entity)
3. pos에 있는 text들 중에서
4. 가장 유사한 걸 찾기
"""

def ner(dataset: Dataset, args, col_name: str):
    res = []
    nlp = spacy.load(args.spacy_model)
    entities = defaultdict(list)
    inputs = dataset[col_name]
    answers = dataset[args.ans_col]
    for i in tqdm(range(0, len(inputs), args.batch_size), desc="Extracting entities..."):
        batch = inputs[i:i+args.batch_size]
        batch_answers = answers[i:i+args.batch_size]
        docs = list(nlp.pipe(batch, batch_size=args.batch_size))
        for doc, ans in zip(docs, batch_answers):
            hit = False
            if doc.ents:
                for ent in doc.ents:
                    entities[ent.label_].append(ent.text)
                    if not hit and normalize_answer(ent.text) == normalize_answer(ans[0]):
                        res.append(ent.label_)
                        hit = True
            if not hit:
                res.append(None)
    dataset = dataset.add_column("entity", res)
    return dataset.filter(lambda x: x["entity"] is not None, num_proc=NUM_PROC)

def gen_entity_vector(dataset: Dataset, args, col_name: str):
    nlp = spacy.load("en_core_web_lg")
    entities = dataset[col_name]
    if isinstance(entities[0], list):
        entities = [ent[0] for ent in entities]
    vectors = []
    for i in range(0, len(entities), args.batch_size):
        batch = entities[i:i+args.batch_size]
        docs = list(nlp.pipe(batch, batch_size=args.batch_size))
        if args.use_gpu and torch.cuda.is_available():
            vectors.extend([doc.vector.get() for doc in docs])
        else:
            vectors.extend([doc.vector for doc in docs])
    vectors = [v/np.linalg.norm(v) for v in vectors]
    dataset = dataset.add_column("entity_vector", vectors)
    return dataset.filter(lambda x: not np.isnan(x["entity_vector"]).any(), batch_size=4000, num_proc=NUM_PROC)

def is_valid_entity(entity: str, label: str) -> bool:
    if "@" in entity:
        return False
    if label == "DATE" and "through" in entity:
        return False
    if label == "DATE" and "and" in entity:
        return False
    return True

def generate_entity_pool(dataset: Dataset, args):
    contexts = []
    if args.use_wikitext:
        dataset = load_dataset("wikitext","wikitext-103-raw-v1",split="train")
        dataset = dataset.filter(lambda x: len(x["text"]) > 50, num_proc=NUM_PROC)
        dataset = dataset.map(lambda x: {"text": x["text"].strip()}, num_proc=NUM_PROC)
        contexts += dataset["text"]
    else:
        contexts += dataset["context"]
    if args.test:
        contexts = random.sample(contexts, 10000)
    print("Total number of contexts: ", len(contexts))
    nlp = spacy.load("en_core_web_trf")
    entities = defaultdict(list)
    for i in tqdm(range(0, len(contexts), 2000), desc="Extracting entities from entity pool..."):
        batch = contexts[i:i+2000]
        docs = list(nlp.pipe(batch, batch_size=2000))
        for doc in docs:
            if doc.ents:
                for ent in doc.ents:
                    if is_valid_entity(ent.text.lower(), ent.label_):
                        entities[ent.label_].append(ent.text)
    entities = {k: list(set(v)) for k, v in entities.items()}
    nlp = spacy.load("en_core_web_lg")
    
    entities_to_vector = defaultdict(list)
    valid_entity_group = defaultdict(list)
    for k, v in entities.items():
        vectors, valid_entities, valid_vectors = [], [], []
        for i in range(0, len(v), 2000):
            batch = v[i:i+2000]
            docs = list(nlp.pipe(batch, batch_size=2000))
            if args.use_gpu and torch.cuda.is_available():
                vectors.extend([doc.vector.get() for doc in docs])
            else:
                vectors.extend([doc.vector for doc in docs])
        vectors = [v/np.linalg.norm(v) for v in vectors]
        for vec, ent in zip(vectors, v):
            if not np.isnan(vec).any():
                valid_entities.append(ent)
                valid_vectors.append(vec)
        entities_to_vector[k] = np.array(valid_vectors)
        valid_entity_group[k] = valid_entities
    for k, v in valid_entity_group.items():
        print(f"{k}: {len(v)}")
    if not args.test:
        joblib.dump(entities_to_vector, f"/data/seongilpark/dataset/entity_group_vec.pkl")
        joblib.dump(valid_entity_group, f"/data/seongilpark/dataset/entity_group.pkl")
        print("Entity pool saved!")
        print("Saved path: /data/seongilpark/dataset/entity_group.pkl")
    return valid_entity_group, entities_to_vector
        
def save_samples_to_wandb(dataset: Dataset, args):
    now = dt.datetime.now().strftime("%m-%d-%H-%M")
    df = pd.DataFrame(dataset)
    if args.task == "unans":
        df["answer_in_context"] = df["answer_in_context"].apply(lambda x: ", ".join(x))
        df = df[["question","context","answer_in_context","Q_similar_context","C_similar_context","QC_similar_context","random_context"]].sample(100)
        try:
            wandb.init(project="craft-cases", name="unanswerable" if not args.test else "test-unanswerable", config=vars(args))
            wandb.log({"samples": wandb.Table(dataframe=df)})
        except:
            df.to_csv(f"output/{args.task}_{now}.csv")
    elif args.task == "entity":
        df[args.ans_col] = df[args.ans_col].apply(lambda x: ", ".join(x))
        df = df[["question", args.ans_col, "entity", "similar_entity","similar_entity_score","random_entity","random_entity_score"]].sample(100)
        try:
            wandb.init(project="craft-cases", name="similar_entity" if not args.test else "test-similar_entity", config=vars(args))
            wandb.log({"samples": wandb.Table(dataframe=df),
                "similar_entity_score_mean": df["similar_entity_score"].mean(),
                "random_entity_score_mean": df["random_entity_score"].mean()})
        except:
            df.to_csv(f"output/{args.task}_{now}.csv")
    elif args.task == "similar_context":
        try:
            wandb.init(project="craft-cases", name="similar_context" if not args.test else "test-similar_context", config=vars(args))
            df[args.ans_col] = df[args.ans_col].apply(lambda x: ", ".join(x))
            df = df[["question",args.ans_col,"context","rewritten_context","valid"]]
            wandb.log({"samples": wandb.Table(dataframe=df.sample(100)),
                    "valid_ratio": df["valid"].mean()})
            wandb.log({"samples": wandb.Table(dataframe=df)})
        except:
            df.to_csv(f"output/{args.task}_{now}.csv")
    wandb.finish()
