
import json
from pathlib import Path

from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "hospital_knowledge.jsonl"
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 5
CANDIDATE_K = 20


def main():
    with DATA_PATH.open("r", encoding="utf-8") as file:
        chunks = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    chunks = [
        chunk for chunk in chunks
        if isinstance(chunk.get("text"), str) and chunk["text"].strip()
    ]

    if not chunks:
        raise ValueError("No valid chunks found.")

    print(f"Loaded {len(chunks)} chunks.")
    print(f"Loading reranker: {MODEL_NAME}")

    model = CrossEncoder(MODEL_NAME)

    while True:
        query = input("\nEnter question (or 'exit'): ").strip()

        if query.lower() == "exit":
            break
        if not query:
            continue

        candidates = chunks[:CANDIDATE_K]
        pairs = [(query, chunk["text"]) for chunk in candidates]
        scores = model.predict(pairs)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:TOP_K]

        print("\nCross-encoder results:")

        for rank, (chunk, score) in enumerate(ranked, start=1):
            print(f"\n--- Result {rank} ---")
            print(f"Chunk ID: {chunk.get('chunk_id', 'N/A')}")
            print(f"Score: {float(score):.4f}")
            print(f"Category: {chunk.get('category', 'N/A')}")
            print(f"Topic: {chunk.get('topic', 'N/A')}")
            print(f"Text: {chunk['text']}")

    print("Finished.")


if __name__ == "__main__":
    main()
