# ML for Context in AI Assistant

A simple embeddings-based semantic search engine for code, evaluated and fine-tuned on the **CoSQA** dataset.

## Overview

This project implements and evaluates a lightweight code search engine based on text embeddings. The goal is to explore how pretrained language models can be used for semantic code retrieval, and how fine-tuning can improve performance on real code search data.

### Part 1: Embeddings-Based Search

- Loads a collection of documents (e.g., code snippets or function bodies)
- Provides a simple API for text-based search
- Uses a pretrained model from **Hugging Face** to generate embeddings
- Stores and retrieves vectors using **Qdrant**

### Part 2: Evaluation

- Evaluated on the **CoSQA** dataset
- Implements ranking metrics:
  - **Recall@10**
  - **MRR@10**
  - **NDCG@10**
- Reports baseline metrics using the pretrained model before fine-tuning

### Part 3: Fine-Tuning

- Fine-tunes the model on the CoSQA training split
- Uses **cross-entropy loss** with in-batch negatives
- Demonstrates metric improvements on the test set after fine-tuning
- Includes a training loss plot to visualize convergence

### Bonus Experiments

- Compare results when searching over **function names** instead of full function bodies
- Explore how vector DB parameters (e.g., metric type) affect performance

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Quick Start

You can adjust paths, model names, and hyperparameters in `config.py`.

**Demo:**
```bash
python -m scripts.demo
```

**Full pipeline:**
```bash
python -m scripts.run_pipeline
```

**Run individual steps:**
```bash
python -m scripts.train
python -m scripts.evaluate
python -m scripts.bonus_experiments
```

**Start API server:**
```bash
python api.py
```

---

### Testing the REST API

**1. Check service health:**
```bash
curl http://localhost:8000/health
```

**2. Initialize the engine:**
```bash
curl -X POST "http://localhost:8000/initialize" -H "Content-Type: application/json"
```

**3. Add documents to the engine:**
```bash
curl -X POST "http://localhost:8000/add_documents" \
  -H "Content-Type: application/json" \
  -d '{"documents": [
        {"text": "def hello(): print(\"Hello, world!\")", "doc_id": "1"},
        {"text": "def add(a, b): return a + b", "doc_id": "2"}
      ]}'
```

**4. Search for relevant code:**
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I print hello?", "top_k": 2}'
```

---

## Project Structure

```
rag_jb/
├── config.py                     # Global configuration
├── api.py                        # REST API for code search
├── src/
│   ├── models/
│   │   └── code_search_model.py  # Model definition (PyTorch Lightning)
│   ├── data/
│   │   └── cosqa_dataset.py      # Dataset loader for CoSQA
│   ├── search_engine.py          # Vector search logic
│   └── metrics.py                # Evaluation metrics (Recall@10, etc.)
├── scripts/
│   ├── train.py                  # Fine-tuning script
│   ├── evaluate.py               # Evaluation script
│   ├── run_pipeline.py           # End-to-end pipeline
│   ├── bonus_experiments.py      # Extra experiments
│   └── demo.py                   # Small example run
├── report.ipynb                  # Notebook with results and visualizations
└── requirements.txt
```

---

## Implementation Notes

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector store:** Qdrant
- **Loss:** Cross-entropy with in-batch negatives  
- **Dataset:** [CoSQA](https://github.com/wasiahmad/CoSQA) — around 20k query-function pairs

**Metrics explained:**
- *Recall@10*: Did the relevant code appear in the top 10?
- *MRR@10*: How early did the correct result appear?
- *NDCG@10*: Measures ranking quality considering multiple relevant results

---

## Deliverables

This repository includes:
- All code for model training, evaluation
- A clear project structure with documented scripts  
- A `report.ipynb` notebook showing results and plots  
- Working scripts that reproduce results end-to-end
