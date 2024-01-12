import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from tqdm.auto import tqdm
from datasets import load_dataset
import pandas as pd
import json
from datasets import Dataset, DatasetDict, concatenate_datasets
import spacy
import joblib
import torch
#spacy.prefer_gpu(gpu_id=0)
nlp = spacy.load("en_core_web_md")
df = joblib.load("/data/seongilpark/mrqa_df.joblib")
contexts = list(set(df.norm_context.tolist()))
print(len(contexts))
ctx2doc = dict()
BATCH_SIZE = 128
for i in tqdm(range(0, len(contexts), BATCH_SIZE)):
    batch = contexts[i:i+BATCH_SIZE]
    docs = list(nlp.pipe(batch, batch_size=BATCH_SIZE, n_process=2))
    for ctx, doc in zip(batch, docs):
        ctx2doc[ctx] = {"doc":doc}
    del docs
    del batch
    print(len(ctx2doc))
joblib.dump(ctx2doc, "/data/seongilpark/mrqa_ctx2doc.joblib")
print("Done, save path is /data/seongilpark/mrqa_ctx2doc.joblib")