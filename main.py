import argparse
import json
import sys
from pathlib import Path

from config import (
    DATASET_DIR, PERSIST_DIRECTORY, SPARSE_INDEX_PATH,
    EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME, OLLAMA_MODEL_NAME,
    OLLAMA_HOST, TOP_K_DENSE, TOP_K_SPARSE, TOP_K_RERANKED
)
from rag_module.ingestion import DatasetScanner, RegulatoryPDFParser
from rag_module.chunking import RegulatoryStructureChunker
from rag_module.indexing import ChromaVectorStore
from rag_module.retrieval import BM25Retriever, HybridRerankRetriever
from rag_module.generation import RegulatoryGenerator
from rag_module.evaluation import RAGEvaluator

def build_pipeline():
    """Initializes vector store, BM25 retriever, hybrid reranker, and generator."""
    vector_store = ChromaVectorStore(persist_dir=PERSIST_DIRECTORY, model_name=EMBEDDING_MODEL_NAME)

    bm25_retriever = BM25Retriever(index_path=SPARSE_INDEX_PATH)
    if not bm25_retriever.load_index(SPARSE_INDEX_PATH):
        print("[Warning] BM25 index file not found. Run 'python main.py ingest' first.")

    hybrid_retriever = HybridRerankRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
        reranker_model_name=RERANKER_MODEL_NAME
    )

    generator = RegulatoryGenerator(model_name=OLLAMA_MODEL_NAME, ollama_host=OLLAMA_HOST)

    return vector_store, bm25_retriever, hybrid_retriever, generator

def handle_ingest(args):
    """Handles dataset discovery, PDF parsing, structure chunking, and dual indexing."""
    print("=== RBI & SEBI REGULATORY INGESTION PIPELINE ===")
    scanner = DatasetScanner(root_dir=DATASET_DIR)
    doc_descriptors = scanner.scan_documents(max_docs=args.max_docs)

    if not doc_descriptors:
        print(f"[Error] No PDF files found in {DATASET_DIR}")
        return

    parser = RegulatoryPDFParser()
    chunker = RegulatoryStructureChunker()

    all_chunks = []
    print(f"\n[Ingestion] Parsing {len(doc_descriptors)} PDF documents...")

    for idx, doc_meta in enumerate(doc_descriptors, start=1):
        print(f"[{idx}/{len(doc_descriptors)}] Parsing: {doc_meta['filename']}")
        pages = parser.parse_pdf(doc_meta["file_path"])
        chunks = chunker.chunk_document(doc_meta, pages)
        all_chunks.extend(chunks)

    print(f"\n[Ingestion] Total structure-aware chunks generated: {len(all_chunks)}")

    # Index Dense Vectors in ChromaDB
    vector_store = ChromaVectorStore(persist_dir=PERSIST_DIRECTORY, model_name=EMBEDDING_MODEL_NAME)
    vector_store.add_chunks(all_chunks)

    # Index Sparse BM25
    bm25_retriever = BM25Retriever(index_path=SPARSE_INDEX_PATH)
    bm25_retriever.build_index(all_chunks)

    print("\n[Success] Ingestion & dual indexing complete!")

def handle_query(args):
    """Executes hybrid retrieval, reranking, and generation with claims separation."""
    print(f"\n=== REGULATORY QUERY: '{args.query}' ===")
    vector_store, bm25_retriever, hybrid_retriever, generator = build_pipeline()

    context_chunks = hybrid_retriever.retrieve(
        query=args.query,
        top_k_dense=TOP_K_DENSE,
        top_k_sparse=TOP_K_SPARSE,
        top_k_final=TOP_K_RERANKED
    )

    result = generator.generate_answer(args.query, context_chunks)

    print("\n--- GROUNDED REGULATORY ANSWER ---")
    print(result["answer"])

    print("\n--- SOURCE CITATIONS ---")
    for c in result["citations"]:
        print(f" - {c}")

    print("\n--- DISCRETE CLAIMS (FOR DOWNSTREAM SLM VERIFIER) ---")
    for idx, claim in enumerate(result["claims"], start=1):
        print(f" [{idx}] {claim}")

    print(f"\n[Engine Used: {result['model_used']}]")

def handle_eval(args):
    """Runs automated evaluation framework over benchmark dataset."""
    print("=== RUNNING RAG EVALUATION BENCHMARK ===")
    vector_store, bm25_retriever, hybrid_retriever, generator = build_pipeline()

    evaluator = RAGEvaluator(retriever=hybrid_retriever, generator=generator)
    eval_dataset_path = Path(__file__).parent / "data" / "eval_dataset.json"

    summary = evaluator.evaluate_dataset(eval_dataset_path=eval_dataset_path, top_k=TOP_K_RERANKED)
    output_report_path = Path(__file__).parent / "evaluation_report.json"
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Evaluation] Full report saved to {output_report_path}")

def main():
    parser = argparse.ArgumentParser(description="RBI & SEBI Financial Regulatory RAG Module")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Parse dataset PDFs and build vector + BM25 index")
    ingest_parser.add_argument("--max-docs", type=int, default=None, help="Maximum number of PDF documents to index (optional)")

    # Query subcommand
    query_parser = subparsers.add_parser("query", help="Query the regulatory RAG pipeline")
    query_parser.add_argument("query", type=str, help="Regulatory query string")

    # Eval subcommand
    eval_parser = subparsers.add_parser("eval", help="Run benchmark evaluation suite")

    args = parser.parse_args()

    if args.command == "ingest":
        handle_ingest(args)
    elif args.command == "query":
        handle_query(args)
    elif args.command == "eval":
        handle_eval(args)

if __name__ == "__main__":
    main()
