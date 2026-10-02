from ..conversation.context_query_builder import (
    ContextQueryBuilder
)

from ..retrieval.dense_retriever import (
    retrieve
)


# ==================================================
# CONFIGURATION
# ==================================================

TOP_K = 5


class ConversationRAG:

    def __init__(self, llm):

        print(
            "\nInitializing Conversation RAG..."
        )

        # Use the already loaded LLM.
        # Do NOT create another QwenLLM here.
        self.llm = llm

        self.query_builder = (
            ContextQueryBuilder()
        )

        print(
            "\nConversation RAG initialized."
        )


    # ==================================================
    # BUILD RETRIEVAL CONTEXT
    # ==================================================

    def build_context(
        self,
        results
    ):

        context_parts = []


        for rank, result in enumerate(
            results,
            start=1
        ):

            payload = result.payload or {}

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
    # EXTRACT RETRIEVAL INFORMATION
    # ==================================================

    def get_retrieved_documents(
        self,
        results
    ):

        documents = []


        for rank, result in enumerate(
            results,
            start=1
        ):

            payload = result.payload or {}


            document = {

                "rank": rank,

                "score": round(
                    float(result.score),
                    4
                ),

                "document_type": payload.get(
                    "document_type",
                    ""
                ),

                "category": payload.get(
                    "category",
                    ""
                ),

                "title": payload.get(
                    "title",
                    ""
                ),

                "chunk_id": payload.get(
                    "chunk_id",
                    ""
                ),

                "content": payload.get(
                    "chunk_text",
                    ""
                )
            }


            documents.append(
                document
            )


        return documents


    # ==================================================
    # ANSWER QUESTION
    # ==================================================

    def answer(
        self,
        user_message: str,
        dialogue_state: dict,
        top_k: int = TOP_K
    ):

        # --------------------------------------------------
        # 1. Build context-aware retrieval query
        # --------------------------------------------------

        retrieval_query = (
            self.query_builder.build(
                user_message=user_message,
                dialogue_state=dialogue_state
            )
        )


        print(
            "\n" + "=" * 80
        )

        print(
            "RETRIEVAL QUERY"
        )

        print(
            "=" * 80
        )

        print(
            retrieval_query
        )


        # --------------------------------------------------
        # 2. Dense retrieval
        # --------------------------------------------------

        print(
            "\nRetrieving hospital knowledge..."
        )

        results = retrieve(
            query=retrieval_query,
            top_k=top_k
        )


        if not results:

            return {

                "answer": (
                    "I could not find relevant hospital "
                    "information in the knowledge base."
                ),

                "retrieval_query": retrieval_query,

                "retrieved_documents": []

            }


        print(
            f"\nRetrieved {len(results)} "
            f"knowledge chunks."
        )


        # --------------------------------------------------
        # 3. Display retrieved knowledge
        # --------------------------------------------------

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
                f"Score: {result.score:.4f}"
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
        # 4. Build LLM context
        # --------------------------------------------------

        context = self.build_context(
            results
        )


        # --------------------------------------------------
        # 5. Generate final answer
        # --------------------------------------------------

        answer = self.llm.generate(

            question=user_message,

            context=context

        )


        # --------------------------------------------------
        # 6. Extract retrieval information
        # --------------------------------------------------

        retrieved_documents = (
            self.get_retrieved_documents(
                results
            )
        )


        # --------------------------------------------------
        # 7. Return structured result
        # --------------------------------------------------

        return {

            "answer": answer,

            "retrieval_query": retrieval_query,

            "retrieved_documents":
                retrieved_documents

        }