
import pandas as pd
from pathlib import Path

root = Path(".")
results_path = root / "data/evaluation/hybrid_reranker_results.csv"
output_path = root / "data/evaluation/manual_relevance_labels.csv"

results = pd.read_csv(results_path)
evaluation = pd.read_csv(
    root / "data/evaluation/rag_evaluation.csv"
)

first_20 = evaluation.head(20)[
    ["eval_id", "question", "category"]
]

merged = first_20.merge(
    results[["eval_id", "top_chunk_ids"]],
    on="eval_id",
    how="left"
)

rows = []

for _, item in merged.iterrows():
    chunk_ids = str(item["top_chunk_ids"]).split(";")

    for rank, chunk_id in enumerate(chunk_ids, start=1):
        if not chunk_id or chunk_id == "nan":
            continue

        rows.append({
            "eval_id": item["eval_id"],
            "question": item["question"],
            "category": item["category"],
            "rank": rank,
            "chunk_id": chunk_id,
            "relevance": "",
            "review_notes": ""
        })

pd.DataFrame(rows).to_csv(output_path, index=False)

print(f"Created: {output_path}")
print(f"Questions: {len(first_20)}")
print(f"Chunk judgments to review: {len(rows)}")
