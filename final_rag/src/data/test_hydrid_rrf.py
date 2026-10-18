
import json
import re
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "hospital_knowledge.jsonl"
QDRANT_PATH = ROOT / "qdrant_data"

COLLECTION_NAME = "mediguide_hospital_knowledge"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
RRF_K = 60


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def main():
    with DATA_PATH.open("r", encoding="utf-8") as file:
        chunks = [json.loads(line) for line in file if line.strip()]

    chunks = [
        chunk for chunk in chunks
        if isinstance(chunk.get("text"), str) and chunk["text"].strip()
    ]

    bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])
    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(path=str(QDRANT_PATH))

    try:
        if not client.collection_exists(COLLECTION_NAME):
            raise RuntimeError(f"Missing Qdrant collection: {COLLECTION_NAME}")

        id_to_chunk = {chunk["chunk_id"]: chunk for chunk in chunks}

        while True:
            query = input("\nEnter question (or 'exit'): ").strip()
            if query.lower() == "exit":
                break
            if not query:
                continue

            # Dense retrieval
            vector = model.encode(
                query, normalize_embeddings=True
            ).tolist()

            dense_results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                limit=TOP_K,
                with_payload=True,
            ).points

            dense_ids = [
                point.payload.get("chunk_id")
                for point in dense_results
                if point.payload and point.payload.get("chunk_id")
            ]

            # BM25 retrieval
            bm25_scores = bm25.get_scores(tokenize(query))
            bm25_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True,
            )[:TOP_K]

            bm25_ids = [
                chunks[i]["chunk_id"] for i in bm25_indices
            ]

            # Reciprocal Rank Fusion
            rrf_scores = {}

            for ranked_ids in (dense_ids, bm25_ids):
                for rank, chunk_id in enumerate(ranked_ids, start=1):
                    rrf_scores[chunk_id] = (
                        rrf_scores.get(chunk_id, 0)
                        + 1 / (RRF_K + rank)
                    )

            ranked_chunks = sorted(
                rrf_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:TOP_K]

            print("\nHybrid RRF results:")

            for rank, (chunk_id, score) in enumerate(ranked_chunks, start=1):
                chunk = id_to_chunk.get(chunk_id)

                if chunk is None:
                    print(f"\n{rank}. {chunk_id} | RRF: {score:.5f}")
                    print("Chunk text not found in local JSONL.")
                    continue

                print(f"\n--- Result {rank} ---")
                print(f"Chunk ID: {chunk_id}")
                print(f"RRF score: {score:.5f}")
                print(f"Category: {chunk.get('category', 'N/A')}")
                print(f"Topic: {chunk.get('topic', 'N/A')}")
                print(f"Text: {chunk['text']}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
