import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

INPUT_FILE = "../../data/chunks/mediguide_rag_chunks_final.jsonl"

OUTPUT_DIR = "../../embeddings"

OUTPUT_VECTORS = os.path.join(
    OUTPUT_DIR,
    "hospital_embeddings.npy"
)

OUTPUT_METADATA = os.path.join(
    OUTPUT_DIR,
    "hospital_embeddings_metadata.jsonl"
)

BATCH_SIZE = 16


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks(file_path):

    chunks = []

    print(f"\nLoading chunks from:")
    print(file_path)

    with open(file_path, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            chunk = json.loads(line)

            chunks.append(chunk)

    print(f"Loaded {len(chunks)} chunks")

    return chunks


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_model():

    print("\nLoading embedding model...")
    print(f"Model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded successfully")

    return model


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(model, chunks):

    # Extract only the text that should be embedded.
    texts = [
        chunk["chunk_text"]
        for chunk in chunks
    ]

    print("\nCreating embeddings...")
    print(f"Number of texts: {len(texts)}")
    print(f"Batch size: {BATCH_SIZE}")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(embeddings)

    print("\nEmbedding generation completed")

    print(f"Embedding shape: {embeddings.shape}")

    return embeddings


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

def save_embeddings(embeddings):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(
        OUTPUT_VECTORS,
        embeddings
    )

    print("\nEmbeddings saved to:")
    print(OUTPUT_VECTORS)


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata(chunks):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(
        OUTPUT_METADATA,
        "w",
        encoding="utf-8"
    ) as file:

        for index, chunk in enumerate(chunks):

            metadata = {
                "vector_index": index,

                "chunk_id": chunk["chunk_id"],

                "document_id": chunk["document_id"],

                "document_type": chunk["document_type"],

                "knowledge_domain": chunk["knowledge_domain"],

                "category": chunk["category"],

                "title": chunk["title"],

                "chunk_type": chunk["chunk_type"],

                "chunk_text": chunk["chunk_text"],

                "source_id": chunk["source_id"],

                "source_name": chunk["source_name"],

                "source_url": chunk["source_url"],

                "hospital_name": chunk["hospital_name"],

                "scope": chunk["scope"],

                "status": chunk["status"]
            }

            file.write(
                json.dumps(
                    metadata,
                    ensure_ascii=False
                )
                + "\n"
            )

    print("\nMetadata saved to:")
    print(OUTPUT_METADATA)


# ============================================================
# VALIDATE EMBEDDINGS
# ============================================================

def validate_embeddings(embeddings, chunks):

    print("\n")
    print("=" * 60)
    print("EMBEDDING VALIDATION")
    print("=" * 60)

    print(f"Number of chunks     : {len(chunks)}")
    print(f"Number of embeddings : {len(embeddings)}")
    print(f"Embedding dimensions : {embeddings.shape[1]}")

    # Check that every chunk has an embedding
    if len(chunks) != len(embeddings):

        raise ValueError(
            "Number of chunks and embeddings do not match!"
        )

    # Check for NaN values
    if np.isnan(embeddings).any():

        raise ValueError(
            "Embeddings contain NaN values!"
        )

    # Check for infinite values
    if np.isinf(embeddings).any():

        raise ValueError(
            "Embeddings contain infinite values!"
        )

    # Because normalize_embeddings=True,
    # vectors should have approximately unit length.
    norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    print(
        f"Minimum vector norm  : {norms.min():.4f}"
    )

    print(
        f"Maximum vector norm  : {norms.max():.4f}"
    )

    print(
        f"Average vector norm  : {norms.mean():.4f}"
    )

    print("\nValidation successful")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MEDIGUIDE HOSPITAL RAG - EMBEDDING PIPELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # STEP 1
    # Load final chunks
    # --------------------------------------------------------

    chunks = load_chunks(
        INPUT_FILE
    )

    # --------------------------------------------------------
    # STEP 2
    # Load embedding model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # STEP 3
    # Generate embeddings
    # --------------------------------------------------------

    embeddings = create_embeddings(
        model,
        chunks
    )

    # --------------------------------------------------------
    # STEP 4
    # Validate embeddings
    # --------------------------------------------------------

    validate_embeddings(
        embeddings,
        chunks
    )

    # --------------------------------------------------------
    # STEP 5
    # Save vectors
    # --------------------------------------------------------

    save_embeddings(
        embeddings
    )

    # --------------------------------------------------------
    # STEP 6
    # Save metadata
    # --------------------------------------------------------

    save_metadata(
        chunks
    )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("EMBEDDING PIPELINE COMPLETED")
    print("=" * 60)

    print("\nGenerated files:")

    print(
        f"1. {OUTPUT_VECTORS}"
    )

    print(
        f"2. {OUTPUT_METADATA}"
    )

    print("\nNext step:")
    print("Load these vectors into Qdrant.")


if __name__ == "__main__":

    main()