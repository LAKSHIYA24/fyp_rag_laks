# RBI & SEBI Financial Regulatory RAG Module

An advanced Retrieval-Augmented Generation (RAG) system designed for Indian Financial Regulations (RBI and SEBI circulars, master directions, and notifications). 

The system features structure-aware chunking, hybrid retrieval (Dense Semantic + Sparse BM25 with Reciprocal Rank Fusion), cross-encoder reranking, and local LLM generation with discrete claim extraction for downstream verifiers.

---

## 🏗️ Architecture

```
[ PDF Documents ] (RBI / SEBI Circulars)
        │
        ▼
[ Regulatory Structure Chunker ] ───► Preserves document hierarchies, sections, and metadata
        │
   ┌────┴────────────────────────┐
   ▼                             ▼
[ Dense Vector Store ]     [ Sparse BM25 Index ]
(ChromaDB + BGE-Small)     (Lexical Token Matcher)
   │                             │
   └──────────────┬──────────────┘
                  ▼
   [ Hybrid RRF & Cross-Encoder Reranker ]
   (ms-marco-MiniLM-L-6-v2)
                  │
                  ▼
     [ Local LLM Generator ]
     (Ollama: Qwen2.5-7B / Phi-3.5)
                  │
                  ▼
   [ Grounded Answer + Citations + Discrete Claims ]
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally

### 2. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/LAKSHIYA24/fyp_rag_laks.git
cd fyp_rag_laks
pip install -r requirements.txt
```

### 3. Setup LLM
Pull the default LLM in Ollama:
```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
```

---

## 📖 Usage

### Ingestion & Indexing
Place regulatory PDFs inside the `RBI-GUIDELINES/` folder and build the dual indices:
```bash
# Index all documents
python main.py ingest

# Or test with a small batch
python main.py ingest --max-docs 10
```

### Querying
Execute queries against the regulatory knowledge base:
```bash
python main.py query "What are the KYC compliance guidelines for banks?"
```

### Evaluation & Testing
Run automated benchmarks:
```bash
# Evaluate retrieval & generation quality
python main.py eval

# Run test suite
pytest
```

---

## 📁 Repository Structure

```
├── config.py                 # Core configurations & hyperparameters
├── main.py                   # CLI entry point (ingest, query, eval)
├── requirements.txt          # Project dependencies
├── data/
│   └── eval_dataset.json     # Benchmark evaluation questions
├── tests/
│   └── test_rag_pipeline.py  # Unit & pipeline test suite
└── rag_module/
    ├── chunking/             # Regulatory structure-aware chunking
    ├── indexing/             # ChromaDB vector store
    ├── ingestion/            # PDF parsing & dataset scanning
    ├── retrieval/            # BM25, RRF, and Cross-Encoder reranking
    ├── generation/           # Ollama generator & claims extraction
    └── evaluation/           # Evaluation metrics (Hit Rate, MRR, Precision@K)
```
