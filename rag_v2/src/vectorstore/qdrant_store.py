from pathlib import Path
import json

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDINGS_FILE = (
    PROJECT_ROOT
    / "embeddings"
    / "hospital_embeddings.npy"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "embeddings"
    / "hospital_embeddings_metadata.jsonl"
)

QDRANT_PATH = PROJECT_ROOT / "qdrant_data"


# --------------------------------------------------
# Qdrant configuration
# --------------------------------------------------

COLLECTION_NAME = "hospital_knowledge"

VECTOR_SIZE = 1024

BATCH_SIZE = 64


# --------------------------------------------------
# Load embeddings
# --------------------------------------------------

print("Loading embeddings...")

embeddings = np.load(EMBEDDINGS_FILE)

print(f"Embedding shape: {embeddings.shape}")


# --------------------------------------------------
# Load metadata
# --------------------------------------------------

print("Loading metadata...")

metadata = []

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        line = line.strip()

        if line:
            metadata.append(json.loads(line))


print(f"Metadata records: {len(metadata)}")


# --------------------------------------------------
# Validate
# --------------------------------------------------

if len(embeddings) != len(metadata):
    raise ValueError(
        "Number of embeddings and metadata records do not match"
    )

if embeddings.shape[1] != VECTOR_SIZE:
    raise ValueError(
        f"Expected vector dimension {VECTOR_SIZE}, "
        f"got {embeddings.shape[1]}"
    )


# --------------------------------------------------
# Create Qdrant client
# --------------------------------------------------

print("Starting Qdrant...")

client = QdrantClient(
    path=str(QDRANT_PATH)
)


# --------------------------------------------------
# Create collection
# --------------------------------------------------

existing_collections = [
    collection.name
    for collection in client.get_collections().collections
]


if COLLECTION_NAME in existing_collections:

    print(
        f"Collection '{COLLECTION_NAME}' already exists."
    )

else:

    print(
        f"Creating collection '{COLLECTION_NAME}'..."
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )


# --------------------------------------------------
# Insert vectors
# --------------------------------------------------

print("Uploading vectors to Qdrant...")

for start in range(
    0,
    len(embeddings),
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        len(embeddings)
    )

    points = []

    for index in range(start, end):

        record = metadata[index]

        point = PointStruct(
            id=index,

            vector=embeddings[index].tolist(),

            payload={
                "chunk_id": record.get("chunk_id"),
                "document_id": record.get("document_id"),
                "document_type": record.get("document_type"),
                "category": record.get("category"),
                "title": record.get("title"),
                "chunk_type": record.get("chunk_type"),
                "chunk_text": record.get("chunk_text"),
                "metadata": record.get("metadata", {}),
            },
        )

        points.append(point)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(
        f"Uploaded {end}/{len(embeddings)}"
    )


# --------------------------------------------------
# Verify collection
# --------------------------------------------------

collection_info = client.get_collection(
    COLLECTION_NAME
)

print("\nQdrant collection information:")

print(
    f"Collection: {COLLECTION_NAME}"
)

print(
    f"Vectors: {collection_info.points_count}"
)

print(
    f"Vector dimension: "
    f"{collection_info.config.params.vectors.size}"
)

print(
    f"Distance: "
    f"{collection_info.config.params.vectors.distance}"
)


# --------------------------------------------------
# Close
# --------------------------------------------------

client.close()

print("\nQdrant indexing completed successfully.")