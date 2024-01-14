from datasets import Dataset
from src.search import SimpleTokenizer, has_answer
from tqdm.auto import tqdm
import re

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