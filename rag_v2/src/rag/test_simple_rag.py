from simple_rag import SimpleRAG


def main():

    rag = SimpleRAG()

    question = "What information do I need for registration?"

    answer, results = rag.answer(
        question,
        top_k=5
    )

    print("\n" + "=" * 70)
    print("USER QUESTION")
    print("=" * 70)
    print(question)

    print("\n" + "=" * 70)
    print("RETRIEVED EVIDENCE")
    print("=" * 70)

    for i, result in enumerate(results, start=1):

        print(f"\n[{i}] {result['title']}")
        print(f"Score: {result['score']}")
        print(f"Category: {result['category']}")
        print(result['chunk_text'])

    print("\n" + "=" * 70)
    print("LLM ANSWER")
    print("=" * 70)
    print(answer)


if __name__ == "__main__":
    main()