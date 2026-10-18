
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "hospital_knowledge.jsonl"
TOP_K = 5


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    with DATA_PATH.open("r", encoding="utf-8") as file:
        chunks = [json.loads(line) for line in file if line.strip()]

    chunks = [
        chunk for chunk in chunks
        if isinstance(chunk.get("text"), str) and chunk["text"].strip()
    ]

    if not chunks:
        raise ValueError("No valid text chunks found.")

    corpus = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(corpus)

    print(f"Loaded {len(chunks)} chunks.")

    while True:
        query = input("\nEnter question (or 'exit'): ").strip()

        if query.lower() == "exit":
            break
        if not query:
            continue

        scores = bm25.get_scores(tokenize(query))
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:TOP_K]

        print(f"\nTop {len(ranked_indices)} BM25 results:")

        for rank, index in enumerate(ranked_indices, start=1):
            chunk = chunks[index]
            print(f"\n--- Result {rank} ---")
            print(f"Chunk ID: {chunk.get('chunk_id', 'N/A')}")
            print(f"Category: {chunk.get('category', 'N/A')}")
            print(f"Topic: {chunk.get('topic', 'N/A')}")
            print(f"Score: {scores[index]:.4f}")
            print(f"Text: {chunk['text']}")


if __name__ == "__main__":
    main()
