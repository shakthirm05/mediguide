
from functools import lru_cache
import logging

from ollama import Client
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from src.backend.services.patient_preference_service import (
    load_patient_preferences,
)
from src.data.answer_question import (
    load_chunks,
    retrieve,
    generate_answer,
    tokenize,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    OLLAMA_HOST,
    QDRANT_PATH,
    COLLECTION_NAME,
)

logger = logging.getLogger(__name__)


class RAGService:
    """Reusable hospital RAG service. Initialize once per API process."""

    def __init__(self):
        self.chunks = load_chunks()
        self.chunk_by_id = {
            chunk["chunk_id"]: chunk for chunk in self.chunks
        }

        self.bm25 = BM25Okapi(
            [tokenize(chunk["text"]) for chunk in self.chunks]
        )

        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.reranker = CrossEncoder(RERANKER_MODEL)
        self.llm = Client(host=OLLAMA_HOST)
        self.qdrant = QdrantClient(path=str(QDRANT_PATH))

        if not self.qdrant.collection_exists(COLLECTION_NAME):
            self.qdrant.close()
            raise RuntimeError(
                f"Qdrant collection '{COLLECTION_NAME}' was not found."
            )

        # Verify Ollama connectivity during initialization.
        self.llm.list()

    def ask(
        self,
        question: str,
        user_id: str | None = None,
    ) -> dict:
        """Return an evidence-grounded hospital answer and its evidence."""

        evidence = retrieve(
            query=question,
            chunks=self.chunks,
            chunk_by_id=self.chunk_by_id,
            bm25=self.bm25,
            embedder=self.embedder,
            reranker=self.reranker,
            qdrant=self.qdrant,
        )

        # Personalization is optional. A memory failure must not prevent
        # the hospital knowledge pipeline from answering.
        preferences = {}

        if user_id:
            try:
                preferences = load_patient_preferences(user_id)
            except Exception:
                logger.exception(
                    "Could not load patient preferences; "
                    "using default response style."
                )

        answer = generate_answer(
            query=question,
            evidence=evidence,
            llm=self.llm,
            preferences=preferences,
        )

        return {
            "answer": answer,
            "evidence": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "title": chunk.get("title", "Untitled"),
                    "score": chunk.get("_reranker_score"),
                }
                for chunk in evidence
            ],
        }

    def close(self):
        """Close the Qdrant client."""
        self.qdrant.close()


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    """Cache one service instance for the running API process."""
    return RAGService()
