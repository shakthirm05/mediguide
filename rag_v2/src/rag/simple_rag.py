from retrieval.dense_retriever import DenseRetriever
from llm.llm import QwenLLM


class SimpleRAG:

    def __init__(self):
        print("\nInitializing Dense Retriever...")
        self.retriever = DenseRetriever()

        print("\nInitializing LLM...")
        self.llm = QwenLLM()

    def answer(self, question: str, top_k: int = 5):

        print("\nRetrieving hospital knowledge...")

        results = self.retriever.search(
            question,
            top_k=top_k
        )

        # Build context from retrieved chunks
        context_parts = []

        for i, result in enumerate(results, start=1):

            context_parts.append(
                f"""
--- Evidence {i} ---
Title: {result['title']}
Category: {result['category']}
Content:
{result['chunk_text']}
"""
            )

        context = "\n".join(context_parts)

        print("\nGenerating grounded answer...")

        answer = self.llm.generate(
            question=question,
            context=context
        )

        return answer, results