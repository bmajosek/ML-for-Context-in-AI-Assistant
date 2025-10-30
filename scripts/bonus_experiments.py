import json
from tqdm import tqdm
from qdrant_client.models import Distance
from src.search_engine import SearchEngine
from src.metrics import evaluate_ranking
from src.data import load_cosqa_eval
from config import MODEL_NAME, EVAL_K

def extract_function_name(code_text):
    """Extract function name from code if possible"""
    lines = code_text.split('\n')
    for line in lines:
        if 'def ' in line:
            parts = line.split('def ')[1].split('(')[0].strip()
            return parts
    return code_text[:50]

def evaluate_with_function_names():
    corpus, queries, qrels = load_cosqa_eval()
    
    name_corpus = {k: extract_function_name(v) for k, v in corpus.items()}
    
    engine = SearchEngine(model_name=MODEL_NAME, collection_name="cosqa_funcnames")
    corpus_ids = list(name_corpus.keys())
    corpus_texts = [name_corpus[cid] for cid in corpus_ids]
    engine.add_documents(corpus_texts, doc_ids=corpus_ids)
    
    retrieved_lists = []
    relevant_lists = []
    
    for qid in tqdm(qrels.keys(), desc="Function names eval"):
        results = engine.search(queries[qid], top_k=EVAL_K)
        retrieved_lists.append([r['doc_id'] for r in results])
        relevant_lists.append(qrels[qid])
    
    metrics = evaluate_ranking(retrieved_lists, relevant_lists, k=EVAL_K)
    
    print("Function names experiment:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics


def evaluate_with_distance_metrics():
    corpus, queries, qrels = load_cosqa_eval()
    results = {}
    
    for dist in [Distance.COSINE, Distance.DOT, Distance.EUCLID]:
        print(f"\nTesting {dist.name} distance...")
        engine = SearchEngine(model_name=MODEL_NAME, collection_name=f"cosqa_{dist.name.lower()}", distance=dist)
        corpus_ids = list(corpus.keys())
        corpus_texts = [corpus[cid] for cid in corpus_ids]
        engine.add_documents(corpus_texts, doc_ids=corpus_ids)
        
        retrieved_lists = []
        relevant_lists = []
        
        for qid in tqdm(qrels.keys(), desc=f"{dist.name} eval"):
            results_list = engine.search(queries[qid], top_k=EVAL_K)
            retrieved_lists.append([r['doc_id'] for r in results_list])
            relevant_lists.append(qrels[qid])
        
        metrics = evaluate_ranking(retrieved_lists, relevant_lists, k=EVAL_K)
        results[dist.name] = metrics
        
        print(f"{dist.name} results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    return results

def main():
    func_results = evaluate_with_function_names()
    with open("bonus_function_names.json", "w") as f:
        json.dump(func_results, f, indent=2)
    
    distance_results = evaluate_with_distance_metrics()
    with open("bonus_distance_metrics.json", "w") as f:
        json.dump(distance_results, f, indent=2)

if __name__ == "__main__":
    main()

