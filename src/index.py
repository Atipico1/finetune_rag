import numpy as np
from typing import Union, List, Any, Dict
import faiss

def build_multiple_indexes(input_dict: Dict, subsets: List[str]):
    output = dict()
    for subset in subsets:
        if (input_dict.get(subset) == []) or (input_dict.get(subset) == None):
            continue
        else:
            rows, embeddings = [d[0] for d in input_dict[subset]], [d[1] for d in input_dict[subset]]
            index = build_index_with_ids(np.array(embeddings), "/data/seongilpark/index", subset, is_save=False)
            output.update({subset:{"index":index, "id2q":{_id:row for _id, row in zip(np.arange(len(embeddings)).astype('int64'), rows)}}})
    return output

def build_index_with_ids(vectors: np.ndarray, save_dir: str, name: str, is_save: bool = True):
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatIP(len(vectors[0]))
    index = faiss.IndexIDMap(index_flat)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    ids = np.arange(len(vectors)).astype('int64')
    gpu_index.add_with_ids(vectors, ids)
    if is_save:
        faiss.gpu_to_cpu(index)
        faiss.write_index(index, f"{save_dir}/{name}.index")
        print(f"{name} Index saved")
    return gpu_index

def load_index(index_path):
    return faiss.read_index(index_path)

def search_index(query_vector, index, k):
    distances, indices = index.search(query_vector, k)
    return distances, indices
