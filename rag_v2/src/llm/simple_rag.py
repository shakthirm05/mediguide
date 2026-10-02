from ..retrieval.dense_retriever import retrieve
from .llm import QwenLLM
# ==================================================
# Configuration
# ==================================================

TOP_K = 5


# ==================================================
# Build context
# ==================================================

def build_context(results):

    context_parts = []


    for rank, result in enumerate(
        results,
        start=1
    ):

        payload = result.payload or {}


        # --------------------------------------------------
        # Extract metadata
        # --------------------------------------------------

        document_type = payload.get(
            "document_type",
            ""
        )

        category = payload.get(
            "category",
            ""
        )

        title = payload.get(
            "title",
            ""
        )

        chunk_text = payload.get(
            "chunk_text",
            ""
        )


        # --------------------------------------------------
        # Create evidence block
        # --------------------------------------------------

        evidence = f"""
--- Hospital Knowledge {rank} ---

Document Type:
{document_type}

Category:
{category}

Title:
{title}

Content:
{chunk_text}
"""


        context_parts.append(
            evidence
        )


    return "\n".join(
        context_parts
    )


# ==================================================
# Simple RAG
# ==================================================

class SimpleRAG:

    def __init__(self):

        print("\n" + "=" * 80)

        print(
            "INITIALIZING MEDIGUIDE SIMPLE RAG"
        )

        print("=" * 80)


        # --------------------------------------------------
        # Load LLM
        # --------------------------------------------------

        print(
            "\nInitializing Qwen LLM..."
        )

        self.llm = QwenLLM()


        print(
            "\nSimple RAG initialized successfully."
        )


    # ==================================================
    # Answer question
    # ==================================================

    def answer(
        self,
        question: str,
        top_k: int = TOP_K
    ):


        # --------------------------------------------------
        # Dense retrieval
        # --------------------------------------------------

        print(
            "\nRetrieving hospital knowledge..."
        )


        results = retrieve(

            query=question,

            top_k=top_k
        )


        # --------------------------------------------------
        # Check results
        # --------------------------------------------------

        if not results:

            return (
                "I could not find relevant hospital "
                "information in the knowledge base."
            )


        print(
            f"\nRetrieved {len(results)} "
            f"knowledge chunks."
        )


        # ==================================================
        # Display retrieved evidence
        # ==================================================

        print(
            "\n" + "-" * 80
        )

        print(
            "RETRIEVED HOSPITAL KNOWLEDGE"
        )

        print(
            "-" * 80
        )


        for rank, result in enumerate(
            results,
            start=1
        ):

            payload = result.payload or {}


            print(
                f"\n[{rank}]"
            )

            print(
                f"Score: "
                f"{result.score:.4f}"
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
                f"Chunk ID: "
                f"{payload.get('chunk_id')}"
            )

            print(
                f"Content:\n"
                f"{payload.get('chunk_text')}"
            )


        # --------------------------------------------------
        # Build context for LLM
        # --------------------------------------------------

        context = build_context(
            results
        )


        # --------------------------------------------------
        # Generate grounded answer
        # --------------------------------------------------

        answer = self.llm.generate(

            question=question,

            context=context
        )


        return answer


# ==================================================
# Main application
# ==================================================

def main():

    rag = SimpleRAG()


    print(
        "\n" + "=" * 80
    )

    print(
        "MEDIGUIDE HOSPITAL ASSISTANT"
    )

    print(
        "=" * 80
    )

    print(
        "\nAsk a question about the hospital."
    )

    print(
        "Type 'exit' to stop."
    )


    # --------------------------------------------------
    # Conversation loop
    # --------------------------------------------------

    while True:


        question = input(
            "\nQuestion: "
        ).strip()


        # --------------------------------------------------
        # Exit
        # --------------------------------------------------

        if question.lower() == "exit":

            break


        # --------------------------------------------------
        # Ignore empty input
        # --------------------------------------------------

        if not question:

            continue


        # --------------------------------------------------
        # Generate answer
        # --------------------------------------------------

        answer = rag.answer(
            question
        )


        # --------------------------------------------------
        # Display final answer
        # --------------------------------------------------

        print(
            "\n" + "=" * 80
        )

        print(
            "FINAL ANSWER"
        )

        print(
            "=" * 80
        )

        print(
            answer
        )


    print(
        "\nExiting MediGuide."
    )


# ==================================================
# Entry point
# ==================================================

if __name__ == "__main__":

    main()