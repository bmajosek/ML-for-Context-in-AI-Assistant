from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm

class CodeSearchDataset(Dataset):
    def __init__(self, queries, codes):
        self.queries = queries
        self.codes = codes
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        return self.queries[idx], self.codes[idx]

def _load_base_datasets():
    dataset = load_dataset("CoIR-Retrieval/cosqa")
    corpus_dataset = load_dataset("CoIR-Retrieval/cosqa", "corpus")
    queries_dataset = load_dataset("CoIR-Retrieval/cosqa", "queries")
    
    corpus = {item['_id']: item['text'] for item in corpus_dataset['corpus']}
    queries = {item['_id']: item['text'] for item in queries_dataset['queries']}
    
    return dataset, corpus, queries

def _extract_pairs(data_split, corpus, queries, max_samples, desc):
    pairs_queries, pairs_codes = [], []
    for item in tqdm(data_split, desc=desc):
        if item['score'] > 0 and len(pairs_queries) < max_samples:
            qid, cid = item['query-id'], item['corpus-id']
            if qid in queries and cid in corpus:
                pairs_queries.append(queries[qid])
                pairs_codes.append(corpus[cid])
    return pairs_queries, pairs_codes

def load_cosqa_data(max_train_samples=5000, max_val_samples=500):
    dataset, corpus, queries = _load_base_datasets()
    
    train_queries, train_codes = _extract_pairs(
        dataset['train'], corpus, queries, max_train_samples, "Loading training data"
    )
    val_queries, val_codes = _extract_pairs(
        dataset['test'], corpus, queries, max_val_samples, "Loading validation data"
    )
    
    train_dataset = CodeSearchDataset(train_queries, train_codes)
    val_dataset = CodeSearchDataset(val_queries, val_codes)
    
    return train_dataset, val_dataset, corpus, queries

def load_cosqa_eval():
    dataset, corpus, queries = _load_base_datasets()
    
    qrels = {}
    for item in dataset['test']:
        if item['score'] > 0:
            qid, cid = item['query-id'], item['corpus-id']
            if qid not in qrels:
                qrels[qid] = []
            qrels[qid].append(cid)
    
    return corpus, queries, qrels

