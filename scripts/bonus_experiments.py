import json
from tqdm import tqdm
from qdrant_client.models import Distance
from src.search_engine import SearchEngine
from src.metrics import evaluate_ranking
from src.data import load_cosqa_eval

def extract_function_name(code_text):
    """Extract function name from code if possible"""
    lines = code_text.split('\n')
    for line in lines:
        if 'def ' in line:
            # simple extraction - just get the function name
            parts = line.split('def ')[1].split('(')[0].strip()
            return parts
    return code_text[:50]  # fallback to first 50 chars

def evaluate_with_function_names():
    print("\nExperiment 1: Function Names vs Full Bodies")
    
    corpus, queries, qrels = load_cosqa_eval()
    engine_full = SearchEngine(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                               collection_name="full_bodies")
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid] for cid in corpus_ids]
    engine_full.add_documents(corpus_texts, doc_ids=corpus_ids)
    
    retrieved_full, relevant_full = [], []
    for qid in tqdm(list(qrels.keys())[:500]):  # sample for speed
        if qid not in queries:
            continue
        results = engine_full.search(queries[qid], top_k=10)
        retrieved_full.append([r['doc_id'] for r in results])
        relevant_full.append(qrels[qid])
    
    metrics_full = evaluate_ranking(retrieved_full, relevant_full, k=10)
    
    engine_names = SearchEngine(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                collection_name="func_names")
    func_names = [extract_function_name(corpus[cid]) for cid in corpus_ids]
    engine_names.add_documents(func_names, doc_ids=corpus_ids)
    
    retrieved_names, relevant_names = [], []
    for qid in tqdm(list(qrels.keys())[:500]):
        if qid not in queries:
            continue
        results = engine_names.search(queries[qid], top_k=10)
        retrieved_names.append([r['doc_id'] for r in results])
        relevant_names.append(qrels[qid])
    
    metrics_names = evaluate_ranking(retrieved_names, relevant_names, k=10)
    
    print("\nResults:")
    print(f"{'Metric':<15} {'Full Bodies':<15} {'Func Names':<15} {'Diff'}")
    for metric in metrics_full:
        full = metrics_full[metric]
        names = metrics_names[metric]
        diff = names - full
        print(f"{metric:<15} {full:<15.4f} {names:<15.4f} {diff:+.4f}")
    
    return {"full_bodies": metrics_full, "function_names": metrics_names}

def evaluate_with_distance_metrics():
    print("\nExperiment 2: Distance Metrics")
    
    corpus, queries, qrels = load_cosqa_eval()
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid] for cid in corpus_ids]
    
    distances = {
        "COSINE": Distance.COSINE,
        "EUCLID": Distance.EUCLID,
        "DOT": Distance.DOT
    }
    
    results = {}
    
    for dist_name, dist_type in distances.items():
        engine = SearchEngine(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            collection_name=f"dist_{dist_name.lower()}",
            distance=dist_type
        )
        engine.add_documents(corpus_texts, doc_ids=corpus_ids)
        
        retrieved, relevant = [], []
        for qid in tqdm(list(qrels.keys())[:500]):
            if qid not in queries:
                continue
            search_results = engine.search(queries[qid], top_k=10)
            retrieved.append([r['doc_id'] for r in search_results])
            relevant.append(qrels[qid])
        
        metrics = evaluate_ranking(retrieved, relevant, k=10)
        results[dist_name] = metrics
    
    print("\nResults:")
    print(f"{'Metric':<15} {'COSINE':<15} {'EUCLID':<15} {'DOT':<15}")
    for metric in results["COSINE"]:
        cos = results["COSINE"][metric]
        euc = results["EUCLID"][metric]
        dot = results["DOT"][metric]
        print(f"{metric:<15} {cos:<15.4f} {euc:<15.4f} {dot:<15.4f}")
    
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

