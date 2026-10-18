
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "evaluation" / "manual_relevance_labels.csv"
OUTPUT_FILE = ROOT / "data" / "evaluation" / "ranking_metrics.csv"

df = pd.read_csv(INPUT_FILE)

required = {"eval_id", "rank", "chunk_id", "relevance"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df["relevance"] = pd.to_numeric(df["relevance"], errors="coerce")
df["rank"] = pd.to_numeric(df["rank"], errors="coerce")

if df[["eval_id", "rank", "chunk_id", "relevance"]].isna().any().any():
    raise ValueError(
        "Missing or invalid evaluation values. Check eval_id, rank, "
        "chunk_id and relevance."
    )

if not df["relevance"].between(0, 2).all():
    raise ValueError("Relevance must be 0, 1 or 2.")

if df.duplicated(["eval_id", "rank"]).any():
    raise ValueError("Duplicate rank found within an evaluation question.")

df = df.sort_values(["eval_id", "rank"])

results = []

for eval_id, group in df.groupby("eval_id", sort=True):
    group = group.sort_values("rank")
    grades = group["relevance"].to_numpy(dtype=float)

    # Hit Rate@5: at least one relevant result appears in the top five.
    top5 = grades[:5]
    hit_rate = float(np.any(top5 > 0))

    # MRR@5: reciprocal rank of the first relevant result.
    relevant_ranks = np.flatnonzero(top5 > 0)
    mrr = (
        1.0 / (relevant_ranks[0] + 1)
        if len(relevant_ranks)
        else 0.0
    )

    # nDCG@5: use graded relevance (0, 1, 2).
    discounts = 1.0 / np.log2(np.arange(2, len(top5) + 2))
    gains = np.power(2.0, top5) - 1.0
    dcg = float(np.sum(gains * discounts))

    ideal_grades = np.sort(top5)[::-1]
    ideal_gains = np.power(2.0, ideal_grades) - 1.0
    idcg = float(np.sum(ideal_gains * discounts))

    ndcg = dcg / idcg if idcg > 0 else np.nan

    results.append({
        "eval_id": eval_id,
        "judged_chunks": len(group),
        "hit_rate_at_5": hit_rate,
        "mrr_at_5": mrr,
        "ndcg_at_5": ndcg,
        "no_relevant_in_judged_results": not np.any(grades > 0),
    })

metrics = pd.DataFrame(results)
metrics.to_csv(OUTPUT_FILE, index=False)

print(f"Questions evaluated: {len(metrics)}")
print(f"Mean Hit Rate@5: {metrics['hit_rate_at_5'].mean():.4f}")
print(f"Mean MRR@5: {metrics['mrr_at_5'].mean():.4f}")
print(f"Mean nDCG@5: {metrics['ndcg_at_5'].mean():.4f}")
print(f"Questions with no relevant judged chunks: "
      f"{metrics['no_relevant_in_judged_results'].sum()}")
print(f"Saved per-question metrics to: {OUTPUT_FILE}")

print("\nLowest nDCG@5 questions:")
print(
    metrics.sort_values("ndcg_at_5", na_position="first")
    .head(10)
    .to_string(index=False)
)
