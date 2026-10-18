
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

EVAL_FILE = ROOT / "data" / "evaluation" / "rag_dev_evaluation.csv"
CORPUS_FILE = ROOT / "data" / "processed" / "hospital_knowledge.jsonl"
OUTPUT_FILE = ROOT / "data" / "evaluation" / "rag_corpus_coverage.csv"

def normalize(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())

def load_chunks(path):
    chunks = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            # Support common JSONL chunk-text field names.
            chunk_id = (
                record.get("chunk_id")
                or record.get("id")
                or f"line-{line_number}"
            )

            chunk_text = next(
                (
                    record[key]
                    for key in ("chunk_text", "text", "content", "page_content")
                    if isinstance(record.get(key), str)
                ),
                "",
            )

            if chunk_text:
                chunks.append({
                    "chunk_id": str(chunk_id),
                    "text": chunk_text,
                })

    return chunks

def content_terms(text):
    # Ignore common words that provide little lexical evidence.
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "do", "does",
        "did", "i", "my", "me", "we", "our", "you", "your", "to",
        "for", "of", "in", "on", "at", "and", "or", "it", "this",
        "that", "what", "how", "can", "will", "should", "from",
        "have", "has", "had", "be", "been", "being", "about",
        "need", "want", "please", "doesnt", "not"
    }
    return {
        token for token in normalize(text)
        if len(token) > 2 and token not in stopwords
    }

evaluations = pd.read_csv(EVAL_FILE).fillna("")
chunks = load_chunks(CORPUS_FILE)

if not chunks:
    raise ValueError(
        "No chunk text found. Inspect the JSONL field names in "
        "hospital_knowledge.jsonl and update load_chunks()."
    )

results = []

for _, row in evaluations.iterrows():
    query_terms = content_terms(row["question"])
    outline_terms = content_terms(row["expected_answer_outline"])
    target_terms = query_terms | outline_terms

    ranked = []

    for chunk in chunks:
        chunk_terms = content_terms(chunk["text"])
        overlap = target_terms & chunk_terms

        score = (
            len(overlap) / len(target_terms)
            if target_terms else 0
        )

        ranked.append((score, chunk, overlap))

    ranked.sort(key=lambda item: item[0], reverse=True)
    best_score, best_chunk, overlap = ranked[0]

    results.append({
        "eval_id": row["eval_id"],
        "question": row["question"],
        "category": row["category"],
        "best_lexical_overlap": round(best_score, 4),
        "candidate_chunk_id": best_chunk["chunk_id"],
        "matched_terms": ", ".join(sorted(overlap)),
        "candidate_chunk_text": best_chunk["text"][:500],
        "coverage_status": (
            "REVIEW_CANDIDATE" if best_score > 0
            else "NO_LEXICAL_MATCH"
        ),
    })

result_df = pd.DataFrame(results)
result_df.to_csv(OUTPUT_FILE, index=False)

print(f"Corpus chunks loaded: {len(chunks)}")
print(f"Questions checked: {len(result_df)}")
print(f"Saved report: {OUTPUT_FILE}")
print("\nCandidate matches:")
print(
    result_df[
        ["eval_id", "question", "best_lexical_overlap",
         "candidate_chunk_id", "coverage_status"]
    ].to_string(index=False)
)
