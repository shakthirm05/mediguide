
import json
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# Project root: final_rag/
ROOT = Path(__file__).resolve().parents[2]

JSONL_PATH = (
    ROOT / "data" / "processed" / "hospital_knowledge.jsonl"
)
QDRANT_PATH = ROOT / "qdrant_data"

COLLECTION_NAME = "mediguide_hospital_knowledge"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


def load_chunks():
    if not JSONL_PATH.exists():
        raise FileNotFoundError(
            f"Validated JSONL file not found: {JSONL_PATH}"
        )

    chunks = []

    with JSONL_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from exc

            if not chunk.get("chunk_id") or not chunk.get("text"):
                raise ValueError(
                    f"Missing chunk_id or text on line {line_number}"
                )

            chunks.append(chunk)

    if not chunks:
        raise ValueError("No knowledge chunks found.")

    ids = [str(chunk["chunk_id"]) for chunk in chunks]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs found in JSONL.")

    return chunks


def make_qdrant_id(chunk_id):
    """Create a stable UUID while preserving the original chunk ID."""
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"mediguide:{chunk_id}",
        )
    )


def main():
    chunks = load_chunks()

    print(f"Loaded {len(chunks)} knowledge chunks.")
    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)
    vector_size = model.get_embedding_dimension()

    print(f"Embedding dimension: {vector_size}")

    QDRANT_PATH.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path=str(QDRANT_PATH))

    try:
        # Rebuild only this collection from the validated JSONL.
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)
            print(f"Rebuilding collection: {COLLECTION_NAME}")

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start:start + BATCH_SIZE]
            texts = [str(chunk["text"]) for chunk in batch]

            embeddings = model.encode(
                texts,
                batch_size=BATCH_SIZE,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

            points = []

            for chunk, embedding in zip(batch, embeddings):
                # Keep all source metadata, including the original ID.
                payload = {
                    key: value
                    for key, value in chunk.items()
                    if key != "text"
                }
                payload["chunk_id"] = str(chunk["chunk_id"])
                payload["text"] = str(chunk["text"])

                points.append(
                    models.PointStruct(
                        id=make_qdrant_id(chunk["chunk_id"]),
                        vector=embedding.tolist(),
                        payload=payload,
                    )
                )

            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True,
            )

            processed = min(start + len(batch), len(chunks))
            print(f"Indexed {processed}/{len(chunks)} chunks")

        info = client.get_collection(COLLECTION_NAME)

        print("\n=== MEDIGUIDE VECTOR STORE ===")
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Stored points: {info.points_count}")
        print(f"Embedding dimension: {vector_size}")
        print(f"Qdrant storage: {QDRANT_PATH}")

        if info.points_count != len(chunks):
            raise RuntimeError(
                f"Expected {len(chunks)} points, "
                f"but Qdrant reports {info.points_count}."
            )

        print("Vector store build completed successfully.")

    finally:
        client.close()


if __name__ == "__main__":
    main()
