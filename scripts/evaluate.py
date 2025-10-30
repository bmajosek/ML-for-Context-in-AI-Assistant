import json
from tqdm import tqdm
from src.search_engine import SearchEngine
from src.metrics import evaluate_ranking
from src.data import load_cosqa_eval
from config import MODEL_NAME, EVAL_K

def evaluate_model(model_name, k=EVAL_K):
    corpus, queries, qrels = load_cosqa_eval()
    
    engine = SearchEngine(model_name=model_name, collection_name="cosqa")
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid] for cid in corpus_ids]
    engine.add_documents(corpus_texts, doc_ids=corpus_ids)
    
    retrieved_lists = []
    relevant_lists = []
    
    for qid in tqdm(qrels.keys()):
        results = engine.search(queries[qid], top_k=k)
        retrieved_lists.append([r['doc_id'] for r in results])
        relevant_lists.append(qrels[qid])
    
    metrics = evaluate_ranking(retrieved_lists, relevant_lists, k=k)
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

def main():
    results = evaluate_model(MODEL_NAME, k=EVAL_K)
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
