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

def build_index_with_ids(vectors: np.ndarray, save_dir: str, name: str, is_save: bool = True, gpu_id: int =0):
    index_flat = faiss.IndexFlatIP(len(vectors[0]))
    index = faiss.IndexIDMap(index_flat)
    ids = np.arange(len(vectors)).astype('int64')
    if gpu_id != -100 and faiss.get_num_gpus() != 0:
        if faiss.get_num_gpus() > 1:
            gpu_index = faiss.index_cpu_to_all_gpus(index)
        else:
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, index)
            gpu_index.add_with_ids(vectors, ids)
        return gpu_index
    else:
        index.add_with_ids(vectors, ids)
        return index

def load_index(index_path):
    return faiss.read_index(index_path)

def search_index(query_vector, index, k):
    distances, indices = index.search(query_vector, k)
    return distances, indices
