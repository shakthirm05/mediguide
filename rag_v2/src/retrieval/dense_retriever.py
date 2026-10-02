from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# ==================================================
# Project paths
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

QDRANT_PATH = PROJECT_ROOT / "qdrant_data"


# ==================================================
# Configuration
# ==================================================

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

COLLECTION_NAME = "hospital_knowledge"

TOP_K = 5


# ==================================================
# Load embedding model
# ==================================================

print("Loading embedding model...")

model = SentenceTransformer(
    MODEL_NAME
)

print("Embedding model loaded.")


# ==================================================
# Connect to Qdrant
# ==================================================

print("Connecting to Qdrant...")

client = QdrantClient(
    path=str(QDRANT_PATH)
)

print("Connected to Qdrant.")


# ==================================================
# Verify collection
# ==================================================

collections = [
    collection.name
    for collection in client.get_collections().collections
]

if COLLECTION_NAME not in collections:

    raise ValueError(
        f"Collection '{COLLECTION_NAME}' does not exist."
    )


collection_info = client.get_collection(
    COLLECTION_NAME
)

print(
    f"Collection: {COLLECTION_NAME}"
)

print(
    f"Vectors: {collection_info.points_count}"
)


# ==================================================
# Dense retrieval
# ==================================================

def retrieve(
    query: str,
    top_k: int = TOP_K
):

    # ------------------------------------------------
    # Convert query into embedding
    # ------------------------------------------------

    query_vector = model.encode(
        query,
        normalize_embeddings=True
    )


    # ------------------------------------------------
    # Search Qdrant
    # ------------------------------------------------

    results = client.query_points(

        collection_name=COLLECTION_NAME,

        query=query_vector.tolist(),

        limit=top_k,

        with_payload=True
    )


    return results.points