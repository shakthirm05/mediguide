
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
QDRANT_PATH = ROOT / "qdrant_data"

COLLECTION_NAME = "mediguide_hospital_knowledge"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


def main():
    model = SentenceTransformer(MODEL_NAME)

    client = QdrantClient(path=str(QDRANT_PATH))

    try:
        if not client.collection_exists(COLLECTION_NAME):
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' does not exist. "
                "Run build_vector_store.py first."
            )

        while True:
            question = input(
                "\nEnter a hospital question "
                "(or type 'exit' to stop): "
            ).strip()

            if question.lower() == "exit":
                break

            if not question:
                print("Please enter a question.")
                continue

            query_vector = model.encode(
                question,
                normalize_embeddings=True,
            ).tolist()

            results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=TOP_K,
                with_payload=True,
            ).points

            if not results:
                print("No matching chunks found.")
                continue

            print(f"\nTop {len(results)} results:")

            for rank, result in enumerate(results, start=1):
                payload = result.payload or {}

                print(f"\n--- Result {rank} ---")
                print(f"Chunk ID: {payload.get('chunk_id', 'N/A')}")
                print(f"Category: {payload.get('category', 'N/A')}")
                print(f"Topic: {payload.get('topic', 'N/A')}")
                print(f"Score: {result.score:.4f}")
                print(f"Text: {payload.get('text', 'N/A')}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
