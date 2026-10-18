
import csv
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
QDRANT_PATH = ROOT / "qdrant_data"
OUTPUT_PATH = ROOT / "data" / "evaluation" / "retrieval_baseline.csv"

COLLECTION_NAME = "mediguide_hospital_knowledge"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

# Provisional labels based on the retrieval results we inspected.
# Add or change IDs only after checking the actual chunk text.
TEST_CASES = [
    {
        "question": "What should I do for my first appointment?",
        "relevant_ids": {"MGCH-0013"},
        "status": "provisional_labels",
    },
    {
        "question": "How do I request wheelchair assistance?",
        "relevant_ids": {"MGCH-0294", "MGCH-0008", "MGCH-0126"},
        "status": "provisional_labels",
    },
    {
        "question": "What should a visitor know before entering the hospital?",
        "relevant_ids": {"MGCH-0221", "MGCH-0075"},
        "status": "provisional_labels",
    },
    {
        "question": "Where is Cardiology?",
        "relevant_ids": set(),
        "status": "missing_indexed_evidence",
    },
    {
        "question": "Can you book an appointment now?",
        "relevant_ids": {"MGCH-0013", "MGCH-0014"},
        "status": "provisional_labels",
    },
]


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(path=str(QDRANT_PATH))
    rows = []

    try:
        if not client.collection_exists(COLLECTION_NAME):
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' does not exist. "
                "Build the vector store first."
            )

        for case in TEST_CASES:
            question = case["question"]
            relevant_ids = case["relevant_ids"]

            vector = model.encode(
                question,
                normalize_embeddings=True,
            ).tolist()

            results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                limit=TOP_K,
                with_payload=True,
            ).points

            retrieved_ids = []
            print(f"\nQuestion: {question}")
            print(f"Assessment: {case['status']}")

            for rank, result in enumerate(results, start=1):
                payload = result.payload or {}
                chunk_id = payload.get("chunk_id", "N/A")
                retrieved_ids.append(chunk_id)

                print(
                    f"  Rank {rank}: {chunk_id} "
                    f"(score={result.score:.4f})"
                )

            if case["status"] == "missing_indexed_evidence":
                print(
                    "  Ranking metrics: NOT CALCULATED. "
                    "Expected evidence is missing from the indexed corpus."
                )
                recall_at_1 = ""
                recall_at_3 = ""
                recall_at_5 = ""
            else:
                # These metrics are provisional until relevance labels
                # have been reviewed and approved.
                recall_at_1 = (
                    int(bool(set(retrieved_ids[:1]) & relevant_ids))
                )
                recall_at_3 = (
                    int(bool(set(retrieved_ids[:3]) & relevant_ids))
                )
                recall_at_5 = (
                    int(bool(set(retrieved_ids[:5]) & relevant_ids))
                )

            rows.append({
                "question": question,
                "assessment": case["status"],
                "provisional_relevant_ids": ";".join(
                    sorted(relevant_ids)
                ),
                "retrieved_ids_ranked": ";".join(retrieved_ids),
                "hit_at_1": recall_at_1,
                "hit_at_3": recall_at_3,
                "hit_at_5": recall_at_5,
            })

    finally:
        client.close()

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nBaseline report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
