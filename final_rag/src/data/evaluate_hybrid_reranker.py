
import json
import re
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "hospital_knowledge.jsonl"
EVAL_PATH = ROOT / "data" / "evaluation" / "rag_evaluation.csv"
OUTPUT_PATH = ROOT / "data" / "evaluation" / "hybrid_reranker_results.csv"
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
    return re.findall(r"\b\w+\b", str(text).lower())


def main():
    with DATA_PATH.open("r", encoding="utf-8") as file:
        chunks = [json.loads(line) for line in file if line.strip()]

    chunks = [
        c for c in chunks
        if c.get("chunk_id")
        and isinstance(c.get("text"), str)
        and c["text"].strip()
    ]
    chunk_by_id = {c["chunk_id"]: c for c in chunks}

    evaluation = pd.read_csv(EVAL_PATH)
    evaluation = evaluation[
        evaluation["split"].astype(str).str.lower().eq("test")
    ].copy()

    if evaluation.empty:
        raise ValueError("No evaluation rows with split='test' found.")

    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    print(f"Chunks: {len(chunks)}")
    print(f"Evaluation questions: {len(evaluation)}")
    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    print("Loading cross-encoder...")
    reranker = CrossEncoder(RERANKER_MODEL)

    client = QdrantClient(path=str(QDRANT_PATH))
    rows = []

    try:
        if not client.collection_exists(COLLECTION_NAME):
            raise RuntimeError(f"Qdrant collection missing: {COLLECTION_NAME}")

        for count, (_, item) in enumerate(evaluation.iterrows(), start=1):
            question = str(item["question"]).strip()
            if not question:
                continue

            vector = embedder.encode(
                question, normalize_embeddings=True
            ).tolist()

            dense_points = client.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                limit=DENSE_K,
                with_payload=True,
            ).points

            dense_ids = [
                p.payload.get("chunk_id")
                for p in dense_points
                if p.payload
                and p.payload.get("chunk_id") in chunk_by_id
            ]

            sparse_scores = bm25.get_scores(tokenize(question))
            sparse_indices = sorted(
                range(len(sparse_scores)),
                key=lambda i: float(sparse_scores[i]),
                reverse=True,
            )[:BM25_K]
            bm25_ids = [chunks[i]["chunk_id"] for i in sparse_indices]

            rrf_scores = {}
            for ranking in (dense_ids, bm25_ids):
                for rank, chunk_id in enumerate(ranking, start=1):
                    rrf_scores[chunk_id] = (
                        rrf_scores.get(chunk_id, 0.0)
                        + 1.0 / (RRF_K + rank)
                    )

            candidates = sorted(
                rrf_scores,
                key=lambda cid: rrf_scores[cid],
                reverse=True,
            )[:RERANK_K]

            candidate_chunks = [chunk_by_id[cid] for cid in candidates]
            pairs = [(question, c["text"]) for c in candidate_chunks]
            scores = reranker.predict(pairs) if pairs else []

            reranked = sorted(
                zip(candidate_chunks, scores),
                key=lambda pair: float(pair[1]),
                reverse=True,
            )[:FINAL_K]

            top_ids = [c["chunk_id"] for c, _ in reranked]
            top_texts = [c["text"] for c, _ in reranked]
            top_scores = [float(score) for _, score in reranked]

            rows.append({
                "eval_id": item["eval_id"],
                "question": question,
                "category": item["category"],
                "acceptance_criterion": item["acceptance_criterion"],
                "expected_answer_outline": item["expected_answer_outline"],
                "top_chunk_ids": ";".join(top_ids),
                "top_reranker_scores": ";".join(
                    f"{score:.4f}" for score in top_scores
                ),
                "top_chunk_texts": " || ".join(top_texts),
                "candidate_count": len(candidates),
                "candidate_retrieval_ids": ";".join(candidates),
            })

            if count % 10 == 0 or count == len(evaluation):
                print(f"Processed {count}/{len(evaluation)}")

    finally:
        client.close()

    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"\nSaved results to: {OUTPUT_PATH}")
    print(f"Rows saved: {len(rows)}")


if __name__ == "__main__":
    main()
