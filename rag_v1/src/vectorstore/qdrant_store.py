import json
import os

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)


# ============================================================
# CONFIGURATION
# ============================================================

# Project root
PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

# Embedding files
EMBEDDINGS_FILE = os.path.join(
    PROJECT_ROOT,
    "embeddings",
    "hospital_embeddings.npy"
)

METADATA_FILE = os.path.join(
    PROJECT_ROOT,
    "embeddings",
    "hospital_embeddings_metadata.jsonl"
)

# Qdrant local database
QDRANT_PATH = os.path.join(
    PROJECT_ROOT,
    "qdrant_data"
)

# Collection name
COLLECTION_NAME = "hospital_knowledge"

# Number of vectors inserted per batch
BATCH_SIZE = 64


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings():

    print("\nLoading embeddings...")

    embeddings = np.load(
        EMBEDDINGS_FILE
    )

    print(
        f"Loaded embeddings: {embeddings.shape}"
    )

    return embeddings


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():

    print("\nLoading metadata...")

    metadata = []

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            metadata.append(
                json.loads(line)
            )

    print(
        f"Loaded metadata: {len(metadata)}"
    )

    return metadata


# ============================================================
# CREATE QDRANT CLIENT
# ============================================================

def create_qdrant_client():

    print("\nStarting Qdrant...")

    client = QdrantClient(
        path=QDRANT_PATH
    )

    print(
        f"Qdrant database: {QDRANT_PATH}"
    )

    return client


# ============================================================
# CREATE COLLECTION
# ============================================================

def create_collection(
    client,
    vector_dimension
):

    print(
        f"\nCreating collection: {COLLECTION_NAME}"
    )

    # Delete collection if it already exists.
    # This makes this script safe to re-run while developing.
    if client.collection_exists(
        COLLECTION_NAME
    ):

        print(
            "Collection already exists."
        )

        print(
            "Deleting old collection..."
        )

        client.delete_collection(
            COLLECTION_NAME
        )

    client.create_collection(

        collection_name=COLLECTION_NAME,

        vectors_config=VectorParams(

            size=vector_dimension,

            distance=Distance.COSINE
        )
    )

    print(
        "Collection created successfully."
    )


# ============================================================
# CREATE POINTS
# ============================================================

def create_points(
    embeddings,
    metadata
):

    points = []

    for index in range(
        len(embeddings)
    ):

        payload = metadata[index]

        point = PointStruct(

            # Qdrant point ID
            id=index,

            # Embedding vector
            vector=embeddings[index].tolist(),

            # Metadata
            payload=payload
        )

        points.append(point)

    return points


# ============================================================
# INSERT INTO QDRANT
# ============================================================

def insert_points(
    client,
    points
):

    print(
        f"\nInserting {len(points)} vectors..."
    )

    for start in range(
        0,
        len(points),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(points)
        )

        batch = points[start:end]

        client.upsert(

            collection_name=COLLECTION_NAME,

            points=batch
        )

        print(
            f"Inserted {end}/{len(points)}"
        )

    print(
        "\nAll vectors inserted successfully."
    )


# ============================================================
# VERIFY COLLECTION
# ============================================================

def verify_collection(
    client
):

    print(
        "\nVerifying Qdrant collection..."
    )

    info = client.get_collection(
        COLLECTION_NAME
    )

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    print(
        f"Vectors stored: {info.points_count}"
    )

    print(
        f"Vector dimension: "
        f"{info.config.params.vectors.size}"
    )


# ============================================================
# TEST ONE VECTOR SEARCH
# ============================================================

def test_search(
    client,
    embeddings,
    metadata
):

    print(
        "\nTesting vector search..."
    )

    # Use the first chunk's vector
    test_vector = embeddings[0].tolist()

    results = client.query_points(

        collection_name=COLLECTION_NAME,

        query=test_vector,

        limit=5,

        with_payload=True
    ).points

    print(
        "\nTop 5 results:"
    )

    for rank, result in enumerate(
        results,
        start=1
    ):

        payload = result.payload

        print(
            "\n--------------------------------"
        )

        print(
            f"Rank: {rank}"
        )

        print(
            f"Score: {result.score:.4f}"
        )

        print(
            f"Chunk ID: "
            f"{payload.get('chunk_id')}"
        )

        print(
            f"Title: "
            f"{payload.get('title')}"
        )

        print(
            f"Category: "
            f"{payload.get('category')}"
        )

        print(
            f"Chunk type: "
            f"{payload.get('chunk_type')}"
        )

        print(
            f"Text: "
            f"{payload.get('chunk_text')}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "MEDIGUIDE HOSPITAL RAG"
    )

    print(
        "QDRANT VECTOR DATABASE SETUP"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # STEP 1
    # Load embeddings
    # --------------------------------------------------------

    embeddings = load_embeddings()

    # --------------------------------------------------------
    # STEP 2
    # Load metadata
    # --------------------------------------------------------

    metadata = load_metadata()

    # --------------------------------------------------------
    # STEP 3
    # Validate
    # --------------------------------------------------------

    if len(embeddings) != len(metadata):

        raise ValueError(
            "Number of embeddings does not match "
            "number of metadata records!"
        )

    # --------------------------------------------------------
    # STEP 4
    # Start Qdrant
    # --------------------------------------------------------

    client = create_qdrant_client()

    # --------------------------------------------------------
    # STEP 5
    # Create collection
    # --------------------------------------------------------

    vector_dimension = embeddings.shape[1]

    create_collection(
        client,
        vector_dimension
    )

    # --------------------------------------------------------
    # STEP 6
    # Create Qdrant points
    # --------------------------------------------------------

    points = create_points(
        embeddings,
        metadata
    )

    # --------------------------------------------------------
    # STEP 7
    # Insert vectors
    # --------------------------------------------------------

    insert_points(
        client,
        points
    )

    # --------------------------------------------------------
    # STEP 8
    # Verify
    # --------------------------------------------------------

    verify_collection(
        client
    )

    # --------------------------------------------------------
    # STEP 9
    # Test search
    # --------------------------------------------------------

    test_search(
        client,
        embeddings,
        metadata
    )

    print("\n")
    print("=" * 60)
    print("QDRANT SETUP COMPLETED")
    print("=" * 60)


if __name__ == "__main__":

    main()