
import json
import re
from pathlib import Path

from ollama import Client
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "hospital_knowledge.jsonl"
QDRANT_PATH = ROOT / "qdrant_data"

COLLECTION_NAME = "mediguide_hospital_knowledge"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_HOST = "http://localhost:11434"

DENSE_K = 20
BM25_K = 20
RRF_K = 60
RERANK_K = 20
FINAL_K = 5


# --------------------------------------------------
# Tokenization
# --------------------------------------------------

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


# --------------------------------------------------
# Load hospital knowledge
# --------------------------------------------------

def load_chunks():
    with DATA_PATH.open("r", encoding="utf-8") as file:
        chunks = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    chunks = [
        chunk
        for chunk in chunks
        if chunk.get("chunk_id")
        and isinstance(chunk.get("text"), str)
        and chunk["text"].strip()
        and chunk.get("index_policy") != "NOT_INDEXABLE"
    ]

    if not chunks:
        raise ValueError(
            f"No valid hospital knowledge chunks found: {DATA_PATH}"
        )

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs found in the corpus.")

    return chunks


# --------------------------------------------------
# Query intent detection
# --------------------------------------------------

def detect_intent(query):
    q = query.lower().strip()

    complaint_phrases = [
        "unhappy with the service",
        "unhappy with service",
        "complaint",
        "complain about",
        "report a service issue",
        "raise a concern",
        "poor service",
        "dissatisfied with",
        "make a complaint",
    ]

    location_phrases = [
        "where is",
        "where do i go",
        "where can i find",
        "location of",
        "how do i get to",
        "directions to",
        "heart-related",
        "heart related",
        "cardiology",
        "radiology",
    ]

    if any(phrase in q for phrase in complaint_phrases):
        return "complaint"

    if any(phrase in q for phrase in location_phrases):
        if any(term in q for term in ["heart", "cardiology", "cardiac"]):
            return "cardiology_location"

        if "radiology" in q or "imaging department" in q:
            return "radiology_location"

        return "department_location"

    return None


# --------------------------------------------------
# Intent relevance
# --------------------------------------------------

def intent_match_score(chunk, intent):
    """
    Returns a small ranking boost when a chunk clearly matches
    the detected intent. This does not create new hospital facts.
    """

    title = chunk.get("title", "").lower()
    text = chunk.get("text", "").lower()
    content = f"{title}\n{text}"

    if intent == "cardiology_location":
        has_cardiology = any(
            term in content
            for term in ["cardiology", "cardiac", "heart department"]
        )

        has_location = any(
            term in content
            for term in [
                "location",
                "located",
                "room",
                "level",
                "floor",
                "direction",
                "c-201",
            ]
        )

        if has_cardiology and has_location:
            return 5.0

    elif intent == "radiology_location":
        has_radiology = any(
            term in content
            for term in ["radiology", "imaging department"]
        )

        has_location = any(
            term in content
            for term in [
                "location",
                "located",
                "room",
                "level",
                "floor",
                "direction",
                "rad-01",
            ]
        )

        if has_radiology and has_location:
            return 5.0

    elif intent == "department_location":
        has_location = any(
            term in content
            for term in [
                "location",
                "located",
                "room",
                "level",
                "floor",
                "direction",
            ]
        )

        if has_location:
            return 1.0

    elif intent == "complaint":
        insurance_only = (
            "insurance-grievance-routing" in title
            or (
                "insurance" in title
                and "complaint" not in title
                and "patient relations" not in title
            )
        )

        has_patient_complaint = any(
            term in content
            for term in [
                "patient complaint",
                "complaint process",
                "raise a complaint",
                "patient relations",
                "service concern",
                "complaint during admission",
                "how to raise a complaint",
                "making a complaint",
            ]
        )

        if has_patient_complaint and not insurance_only:
            return 5.0

    return 0.0


# --------------------------------------------------
# Hybrid retrieval: Dense + BM25 + RRF + Reranker
# --------------------------------------------------

def retrieve(
    query,
    chunks,
    chunk_by_id,
    bm25,
    embedder,
    reranker,
    qdrant,
):
    intent = detect_intent(query)

    # 1. Dense retrieval using Qdrant

    query_vector = embedder.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    dense_points = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=DENSE_K,
        with_payload=True,
    ).points

    dense_ids = [
        point.payload.get("chunk_id")
        for point in dense_points
        if point.payload
        and point.payload.get("chunk_id") in chunk_by_id
    ]

    # 2. Sparse retrieval using BM25

    bm25_scores = bm25.get_scores(tokenize(query))

    bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda index: float(bm25_scores[index]),
        reverse=True,
    )[:BM25_K]

    bm25_ids = [
        chunks[index]["chunk_id"]
        for index in bm25_indices
    ]

    # 3. Reciprocal Rank Fusion

    rrf_scores = {}

    for ranked_ids in (dense_ids, bm25_ids):
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0.0)
                + 1.0 / (RRF_K + rank)
            )

    # 4. Select candidates from RRF and directly relevant corpus chunks

    candidate_ids = list(rrf_scores.keys())

    if intent:
        for chunk in chunks:
            if intent_match_score(chunk, intent) > 0:
                chunk_id = chunk["chunk_id"]

                if chunk_id not in rrf_scores:
                    candidate_ids.append(chunk_id)

    if not candidate_ids:
        return []

    candidates = [
        chunk_by_id[chunk_id]
        for chunk_id in candidate_ids
    ]

    # 5. Cross-encoder reranking

    pairs = [
        (
            query,
            f"{chunk.get('title', '')}\n{chunk['text']}",
        )
        for chunk in candidates
    ]

    reranker_scores = reranker.predict(pairs)

    # 6. Apply a small intent boost to matching evidence

    scored_chunks = []

    for chunk, raw_score in zip(candidates, reranker_scores):
        score = float(raw_score)
        score += intent_match_score(chunk, intent)
        scored_chunks.append((chunk, score))

    reranked = sorted(
        scored_chunks,
        key=lambda item: item[1],
        reverse=True,
    )[:FINAL_K]

    return [
        {
            **chunk,
            "_reranker_score": score,
        }
        for chunk, score in reranked
    ]


# --------------------------------------------------
# Answer generation using local Ollama LLM
# --------------------------------------------------

def generate_answer(query, evidence, llm):
    if not evidence:
        return (
            "I don't have enough verified hospital information "
            "to answer that. Please ask the hospital help desk."
        )

    context_parts = []

    for chunk in evidence:
        context_parts.append(
            f"[{chunk['chunk_id']}] "
            f"Title: {chunk.get('title', 'Untitled')}\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are MediGuide, an AI-powered hospital information and navigation assistant
for MediGuide General Hospital.

Answer the patient's exact question directly, using only the supplied
hospital evidence.

CORE RULES

1. DIRECT ANSWERS
- Give the specific answer first.
- For simple factual questions, use one or two sentences.
- Avoid generic introductions, repetition, and unnecessary closing statements.
- Do not add unrelated advice.

2. EVIDENCE
- Use retrieved evidence as the source of truth for hospital-specific facts.
- Do not invent locations, room numbers, floor numbers, desk codes,
  opening hours, procedures, or staff responsibilities.
- Do not combine unrelated documents into one answer.
- If the evidence is insufficient, state what cannot be verified.

3. LOCATIONS
- For department-location questions, provide the verified room and floor
  when they appear in the evidence.
- Do not substitute appointment advice for location information.
- If a location is unavailable, say that its exact location cannot be verified.
- Do not claim a physical route is accessible or navigable without
  confirmation from an authorised navigation tool.

4. AMBIGUOUS REQUESTS
- Ask one short clarifying question when necessary.
- For "Can you help me with my appointment?", ask whether the patient needs
  help booking, rescheduling, checking details, or finding the department.
- For "Can I get help with my documents?", ask which document type they mean.
- Do not list every possible procedure.

5. APPOINTMENTS AND LIVE INFORMATION
- Do not guess current availability, waiting times, appointment status,
  live closures, or patient-specific billing and insurance information.
- State when an authorised live system is needed.
- Never claim an appointment was booked, changed, cancelled, or confirmed
  unless an authorised tool verifies the action.

6. COMPLAINTS
- For complaints about hospital service, prioritise the hospital's patient
  complaint or Patient Relations process when supported by the evidence.
- Do not default to insurance grievance procedures unless the question
  concerns insurance.
- Do not promise a resolution, disciplinary action, or response time.

7. DOCUMENTS
- Distinguish registration, admission, insurance, appointment, and discharge
  documents.
- Answer only for the document type requested.
- Do not merge unrelated checklists.

8. SAFETY AND PRIVACY
- Do not diagnose conditions or prescribe treatment.
- Do not make clinical decisions or disclose another patient's information.
- Direct urgent medical or immediate safety concerns to appropriate staff.

9. STYLE
- Use simple, natural language suitable for patients and visitors.
- Include specific facts only when supported by evidence.
- Do not mention internal retrieval scores, embeddings, or implementation.

10. INSUFFICIENT EVIDENCE
- Do not fill gaps with assumptions.
- State the missing information briefly.
- Suggest a next step only when supported by the evidence.

A short, precise, evidence-grounded answer is better than a long, generic one.
"""

    user_prompt = f"""
PATIENT QUESTION:
{query}

RETRIEVED HOSPITAL EVIDENCE:
{context}

Answer the exact question using the evidence and rules above.
If the evidence is insufficient, say so instead of guessing.
"""

    response = llm.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        options={
            "temperature": 0,
        },
    )

    return response["message"]["content"].strip()


# --------------------------------------------------
# Main application
# --------------------------------------------------

def main():
    print("=" * 60)
    print("MediGuide Hospital RAG")
    print("=" * 60)

    chunks = load_chunks()

    chunk_by_id = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    print(f"Loaded knowledge chunks: {len(chunks)}")

    # Build BM25 from the same canonical corpus

    bm25 = BM25Okapi(
        [
            tokenize(chunk["text"])
            for chunk in chunks
        ]
    )

    # Load retrieval models

    print("\nLoading embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print("Loading cross-encoder reranker...")
    reranker = CrossEncoder(RERANKER_MODEL)

    # Connect to Ollama

    print("Connecting to Ollama...")
    llm = Client(host=OLLAMA_HOST)

    try:
        models = llm.list()
        available_models = [
            model.model
            for model in models.models
        ]

        if not any(
            name == OLLAMA_MODEL
            or name.startswith(OLLAMA_MODEL + ":")
            for name in available_models
        ):
            raise RuntimeError(
                f"Ollama model '{OLLAMA_MODEL}' is not installed. "
                f"Run: ollama pull {OLLAMA_MODEL}"
            )

    except Exception as error:
        raise RuntimeError(
            "Could not connect to Ollama. Ensure Ollama is running "
            "and the model is installed."
        ) from error

    # Connect to Qdrant

    qdrant = QdrantClient(path=str(QDRANT_PATH))

    try:
        if not qdrant.collection_exists(COLLECTION_NAME):
            raise RuntimeError(
                f"Qdrant collection '{COLLECTION_NAME}' was not found."
            )

        print(f"LLM: {OLLAMA_MODEL}")
        print(f"Qdrant collection: {COLLECTION_NAME}")
        print("\nMediGuide is ready.")
        print("Type 'exit' or 'quit' to stop.")

        while True:
            try:
                query = input("\nPatient: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting MediGuide.")
                break

            if query.lower() in {"exit", "quit"}:
                print("Exiting MediGuide.")
                break

            if not query:
                continue

            try:
                print("\nRetrieving hospital evidence...")

                evidence = retrieve(
                    query=query,
                    chunks=chunks,
                    chunk_by_id=chunk_by_id,
                    bm25=bm25,
                    embedder=embedder,
                    reranker=reranker,
                    qdrant=qdrant,
                )

                print("\nGenerating answer with Ollama...")

                answer = generate_answer(
                    query=query,
                    evidence=evidence,
                    llm=llm,
                )

                print("\n" + "=" * 60)
                print("MEDIGUIDE")
                print("=" * 60)
                print(answer)

                print("\nRetrieved evidence:")

                if evidence:
                    for index, chunk in enumerate(evidence, start=1):
                        score = chunk.get("_reranker_score", 0.0)

                        print(
                            f"{index}. {chunk['chunk_id']} | "
                            f"{chunk.get('title', 'Untitled')} | "
                            f"reranker score: {score:.4f}"
                        )
                else:
                    print("No evidence retrieved.")

                print("=" * 60)

            except Exception as error:
                print(f"\nError processing question: {error}")
                print(
                    "Check that Ollama and Qdrant are running "
                    "and the required models are installed."
                )

    finally:
        qdrant.close()


if __name__ == "__main__":
    main()
