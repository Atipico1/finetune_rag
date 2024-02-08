import argparse
import time
from collections import Counter
import re
import string
from typing import Callable, List
import numpy as np
from tqdm.auto import tqdm

def normalize_answer(s: str):
    if not s:
        return ""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def em(prediction, ground_truth, normalize_fn):
    return float(normalize_fn(prediction) == normalize_fn(ground_truth))

def f1(prediction, ground_truth, normalize_fn):
    prediction_tokens = normalize_fn(prediction).split()
    ground_truth_tokens = normalize_fn(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def f1_score(prediction, ground_truths):
    return max([f1(prediction, gt, normalize_answer) for gt in ground_truths])

def exact_match_score(prediction, ground_truths):
    return max([em(prediction, gt, normalize_answer) for gt in ground_truths])

def normalize_question(question: str):
    if not question.endswith("?"):
        question = question + "?"

    return question[0].lower() + question[1:]

def accuracy(preds, labels):
    match_count = 0
    for pred, label in zip(preds, labels):
        target = label[0]
        if pred == target:
            match_count += 1

    return 100 * (match_count / len(preds))

def text_has_answer(answers, text) -> bool:
    if isinstance(answers, str):
        answers = [answers]
    text = normalize_answer(text)
    for single_answer in answers:
        single_answer = normalize_answer(single_answer)
        if single_answer in text:
            return True
    return False

def text_has_answer_wo_normalization(answers, text) -> bool:
    if isinstance(answers, str):
        answers = [answers]
    for single_answer in answers:
        if single_answer in text:
            return True
    return False

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def cosine_similarity(a, b):
    if not isinstance(b, np.ndarray):
        b = b.get()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def cal_cosine_similarities(queries_vector: List[np.ndarray], entities: List[str], args):
    import spacy
    scores = []
    nlp = spacy.load("en_core_web_lg")
    assert len(queries_vector) == len(entities), "Length of queries and entities should be same"
    for i in tqdm(range(0, len(queries_vector), args.batch_size), desc="Generating random entity score"):
        docs = nlp.pipe(entities[i:i+args.batch_size], batch_size=args.batch_size)
        for doc, query in zip(docs, queries_vector[i:i+args.batch_size]):
            scores.append(cosine_similarity(query, doc.vector))
    return scores

def find_answer_in_context(answer_text: str, context: str):
    if isinstance(context, str):
        context_spans = [
            (m.start(), m.end())
            for m in re.finditer(re.escape(answer_text.lower()), context.lower())
        ]
        return context_spans
    else:
        return [""]

def update_context_with_substitution_string(
    context: str, originals:List[str], substitution: str, replace_every_string=True
) -> str:
    replace_spans = []
    for orig_answer in originals:
        replace_spans.extend(find_answer_in_context(orig_answer, context))
    replace_strs = set([context[span[0] : span[1]] for span in replace_spans])
    for replace_str in replace_strs:
        context = context.replace(replace_str, substitution)
    return context

def generate_answer_from_gpt(prompt: List[str], client , config: dict):
    max_try = 0
    while max_try < 3:
        try:
            response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=config["max_tokens"],
            seed=42,
            temperature=config["temperature"],
            top_p=config["top_p"]
            )
            return [res.text for res in response.choices]
        except Exception as e:
            print(f"GPT API Error : {e}")
            max_try += 1
            time.sleep(3)
    print("GPT Failed to generate answer")
    return ""