
import json
import re
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "hospital_knowledge.jsonl"
EVAL_PATH = ROOT / "data" / "evaluation" / "rag_dev_evaluation.csv"
OUTPUT_PATH = ROOT / "data" / "evaluation" / "rag_dev_retrieval_results.csv"
QDRANT_PATH = ROOT / "qdrant_data"

COLLECTION_NAME = "mediguide_hospital_knowledge"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DENSE_K = 20
BM25_K = 20
RRF_K = 60
RERANK_K = 20
FINAL_K = 5


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def main():
    with DATA_PATH.open("r", encoding="utf-8") as file:
        chunks = [json.loads(line) for line in file if line.strip()]

    chunks = [
        chunk for chunk in chunks
        if chunk.get("chunk_id")
        and isinstance(chunk.get("text"), str)
        and chunk["text"].strip()
    ]

    if not chunks:
        raise ValueError(f"No valid chunks found in {DATA_PATH}")

    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])

    evaluations = pd.read_csv(EVAL_PATH, dtype={"eval_id": str})

    required_columns = {"eval_id", "question"}
    if not required_columns.issubset(evaluations.columns):
        raise ValueError(f"Evaluation file needs columns: {required_columns}")

    if evaluations["question"].isna().any():
        raise ValueError("Evaluation file contains blank questions.")

    print(f"Loaded {len(chunks)} corpus chunks.")
    print(f"Evaluating {len(evaluations)} questions.")

    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print("Loading cross-encoder...")
    reranker = CrossEncoder(RERANKER_MODEL)

    client = QdrantClient(path=str(QDRANT_PATH))
    rows = []

    try:
        if not client.collection_exists(COLLECTION_NAME):
            raise RuntimeError(
                f"Qdrant collection '{COLLECTION_NAME}' not found."
            )

        for question_number, (_, evaluation) in enumerate(
            evaluations.iterrows(), start=1
        ):
            eval_id = evaluation["eval_id"]
            query = str(evaluation["question"]).strip()

            print(f"[{question_number}/{len(evaluations)}] {eval_id}: {query}")

            # 1. Dense retrieval
            query_vector = embedder.encode(
                query, normalize_embeddings=True
            ).tolist()

            dense_points = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=DENSE_K,
                with_payload=True,
            ).points

            dense_ids = [
                point.payload.get("chunk_id")
                for point in dense_points
                if point.payload
                and point.payload.get("chunk_id") in chunk_by_id
            ]

            # 2. BM25 retrieval
            bm25_scores = bm25.get_scores(tokenize(query))
            bm25_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: float(bm25_scores[i]),
                reverse=True,
            )[:BM25_K]

            bm25_ids = [chunks[i]["chunk_id"] for i in bm25_indices]

            # 3. Reciprocal Rank Fusion
            rrf_scores = {}

            for ranked_ids in (dense_ids, bm25_ids):
                for rank, chunk_id in enumerate(ranked_ids, start=1):
                    rrf_scores[chunk_id] = (
                        rrf_scores.get(chunk_id, 0.0)
                        + 1.0 / (RRF_K + rank)
                    )

            rrf_ranked = sorted(
                rrf_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:RERANK_K]

            if not rrf_ranked:
                rows.append({
                    "eval_id": eval_id,
                    "question": query,
                    "final_rank": None,
                    "chunk_id": None,
                    "reranker_score": None,
                    "rrf_score": None,
                    "category": None,
                    "topic": None,
                    "chunk_text": None,
                })
                continue

            # 4. Cross-encoder reranking
            candidates = [
                chunk_by_id[chunk_id]
                for chunk_id, _ in rrf_ranked
            ]

            pairs = [(query, chunk["text"]) for chunk in candidates]
            scores = reranker.predict(pairs)

            reranked = sorted(
                zip(candidates, scores),
                key=lambda item: float(item[1]),
                reverse=True,
            )[:FINAL_K]

            # 5. Save every top-five candidate for review
            for rank, (chunk, score) in enumerate(reranked, start=1):
                rows.append({
                    "eval_id": eval_id,
                    "question": query,
                    "final_rank": rank,
                    "chunk_id": chunk["chunk_id"],
                    "reranker_score": float(score),
                    "rrf_score": rrf_scores[chunk["chunk_id"]],
                    "category": chunk.get("category", "N/A"),
                    "topic": chunk.get("topic", "N/A"),
                    "chunk_text": chunk["text"],
                })

    finally:
        client.close()

    result = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\nQuestions processed: {len(evaluations)}")
    print(f"Result rows saved: {len(result)}")
    print(f"Saved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
