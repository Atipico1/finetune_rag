import re, random
from datasets import Dataset
from tqdm.auto import tqdm
from collections import defaultdict
from src.search import SimpleTokenizer, has_answer
from src.utils import normalize_question
import joblib

#ANSWER_POS = ["ADV", "ADJ", "NOUN", "NUM", "SYM", "PROPN"]
EDIT_CANDIDATE_POS = ["VERB", "NOUN", "ADJ", "ADV"]
ANSWER_POS = ["ADV", "ADJ", "NOUN", "VERB", "NUM"]

def _preprocess(text, subset_name: str):
    if subset_name == "NewsQA":
        start = text.find("--")
        end = start + 2
        text = text[end:]
    if ("<Table>" in text) or ("<Ol>" in text) or ("<Li>" in text):
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

def preprocess_text(dataset: Dataset) -> Dataset:
    dataset = dataset.map(lambda x: {"context": _preprocess(x["context"], x["subset"])}, num_proc=8, desc="Preprocessing...")
    print("Before preprocess: ", len(dataset))
    dataset = dataset.filter(lambda x: x["context"] is not None, num_proc=8)
    print("After preprocess: ", len(dataset))
    return dataset

def split_sentence(dataset: Dataset, nlp, args):
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
    dataset = dataset.filter(lambda x: x["short_context"] is not None and len(x["short_context"].split())< args.ctx_len, num_proc=8)
    print("After split: ", len(dataset))
    return dataset

def make_spacy_docs(dataset: Dataset, nlp, args):
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
        raise ValueError("No Only text data!")
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
        raise ValueError("No cases!")
    
    return output
    
def preprocess_cbr_dataset(dataset, args):
    dataset = dataset.map(lambda x: {"case": aggregate_cases(x, args)}, num_proc=8, desc="Aggregating cases...")
    return dataset

def determine_answerable(example):
    hasanswer = any([e["hasanswer"] for e in example["ctxs"]])
    if hasanswer:
        return ["unanswerable"]
    else:
        return example["answers"]
    
"""
1. text를 입력으로 받아서
2. text의 pos를 알아낸 다음 (or entity)
3. pos에 있는 text들 중에서
4. 가장 유사한 걸 찾기
"""