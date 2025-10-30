import numpy as np
from typing import List

def recall_at_k(retrieved: List[str], relevant: List[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_k & relevant_set) / len(relevant_set)

def mrr_at_k(retrieved: List[str], relevant: List[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    for rank, item_id in enumerate(retrieved[:k], start=1):
        if item_id in relevant_set:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    
    relevant_set = set(relevant)
    dcg = sum(1.0 / np.log2(rank + 1) 
              for rank, item in enumerate(retrieved[:k], start=1) 
              if item in relevant_set)
    idcg = sum(1.0 / np.log2(rank + 1) 
               for rank in range(1, min(len(relevant), k) + 1))
    
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_ranking(retrieved_lists: List[List[str]], 
                     relevant_lists: List[List[str]], 
                     k: int = 10):
    recalls = [recall_at_k(ret, rel, k) for ret, rel in zip(retrieved_lists, relevant_lists)]
    mrrs = [mrr_at_k(ret, rel, k) for ret, rel in zip(retrieved_lists, relevant_lists)]
    ndcgs = [ndcg_at_k(ret, rel, k) for ret, rel in zip(retrieved_lists, relevant_lists)]
    
    return {
        f"Recall@{k}": np.mean(recalls),
        f"MRR@{k}": np.mean(mrrs),
        f"NDCG@{k}": np.mean(ndcgs)
    }
