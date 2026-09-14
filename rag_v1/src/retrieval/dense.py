import os

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

# Same embedding model used when creating the document embeddings
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

# Local Qdrant database
QDRANT_PATH = os.path.join(
    PROJECT_ROOT,
    "qdrant_data"
)

COLLECTION_NAME = "hospital_knowledge"

TOP_K = 5


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_embedding_model():

    print("Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Embedding model loaded.")

    return model


# ============================================================
# CONNECT TO QDRANT
# ============================================================

def connect_qdrant():

    print("Connecting to Qdrant...")

    client = QdrantClient(
        path=QDRANT_PATH
    )

    print("Connected to Qdrant.")

    return client


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve(
    question,
    model,
    client,
    top_k=TOP_K
):

    # --------------------------------------------------------
    # Convert the user question into an embedding
    # --------------------------------------------------------

    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    # Convert numpy array to normal Python list
    question_embedding = question_embedding.tolist()

    # --------------------------------------------------------
    # Search Qdrant
    # --------------------------------------------------------

    results = client.query_points(

        collection_name=COLLECTION_NAME,

        query=question_embedding,

        limit=top_k,

        with_payload=True
    ).points

    return results


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    question,
    results
):

    print("\n")
    print("=" * 70)
    print("QUESTION")
    print("=" * 70)

    print(question)

    print("\n")
    print("=" * 70)
    print("RETRIEVED CHUNKS")
    print("=" * 70)

    for rank, result in enumerate(
        results,
        start=1
    ):

        payload = result.payload

        print("\n" + "-" * 70)

        print(
            f"Rank: {rank}"
        )

        print(
            f"Similarity Score: {result.score:.4f}"
        )

        print(
            f"Chunk ID: "
            f"{payload.get('chunk_id')}"
        )

        print(
            f"Document Type: "
            f"{payload.get('document_type')}"
        )

        print(
            f"Category: "
            f"{payload.get('category')}"
        )

        print(
            f"Title: "
            f"{payload.get('title')}"
        )

        print(
            f"Chunk Type: "
            f"{payload.get('chunk_type')}"
        )

        print("\nContent:")

        print(
            payload.get(
                "chunk_text",
                ""
            )
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MEDIGUIDE - SIMPLE DENSE RETRIEVAL")
    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_embedding_model()

    # --------------------------------------------------------
    # Connect to Qdrant
    # --------------------------------------------------------

    client = connect_qdrant()

    # --------------------------------------------------------
    # Interactive question loop
    # --------------------------------------------------------

    while True:

        print("\n")

        question = input(
            "Ask a hospital question "
            "(type 'exit' to stop): "
        )

        if question.lower().strip() == "exit":

            print("Exiting...")

            break

        if not question.strip():

            print("Please enter a question.")

            continue

        # ----------------------------------------------------
        # Retrieve
        # ----------------------------------------------------

        results = retrieve(
            question,
            model,
            client,
            TOP_K
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        display_results(
            question,
            results
        )


if __name__ == "__main__":

    main()