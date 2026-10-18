
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

COVERAGE_FILE = ROOT / "data/evaluation/rag_corpus_coverage.csv"
CORPUS_FILE = ROOT / "data/processed/hospital_knowledge.jsonl"

coverage = pd.read_csv(COVERAGE_FILE).fillna("")

chunks = {}

with CORPUS_FILE.open("r", encoding="utf-8") as file:
    for line in file:
        if not line.strip():
            continue

        record = json.loads(line)
        chunk_id = str(
            record.get("chunk_id")
            or record.get("id")
            or ""
        )

        text = next(
            (
                record[key]
                for key in ("chunk_text", "text", "content", "page_content")
                if isinstance(record.get(key), str)
            ),
            "",
        )

        if chunk_id and text:
            chunks[chunk_id] = text

for _, row in coverage.iterrows():
    eval_id = row["eval_id"]
    chunk_id = str(row["candidate_chunk_id"])
    question = row["question"]

    print("\n" + "=" * 90)
    print(f"{eval_id}: {question}")
    print(f"Candidate chunk: {chunk_id}")
    print(f"Lexical overlap: {row['best_lexical_overlap']}")

    text = chunks.get(chunk_id)

    if text:
        print("\nCHUNK TEXT:")
        print(text)
    else:
        print("\nWARNING: Candidate chunk ID not found in JSONL.")
