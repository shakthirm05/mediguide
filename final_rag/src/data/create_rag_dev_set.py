
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / "data" / "evaluation" / "rag_evaluation.csv"
OUTPUT = ROOT / "data" / "evaluation" / "rag_dev_evaluation.csv"

# Candidate RAG questions selected from the existing dev split.
rag_eval_ids = [
    "EVAL-PROD-013",  # Planned admission documents
    "EVAL-PROD-018",  # Discharge documents and procedure
    "EVAL-PROD-032",  # Visiting hours policy
    "EVAL-PROD-033",  # Children visiting ICU
    "EVAL-PROD-034",  # Food brought for a patient
    "EVAL-PROD-043",  # Language and interpreter assistance
    "EVAL-PROD-044",  # Lost property procedure
    "EVAL-PROD-047",  # Complaint procedure
    "EVAL-PROD-049",  # Cafeteria hours and abstention
    "EVAL-PROD-062",  # Visiting-hours exception
    "EVAL-PROD-064",  # Help reading forms
    "EVAL-PROD-067",  # Conflicting visiting-hours sources
    "EVAL-PROD-068",  # Policy provenance
    "EVAL-PROD-069",  # Avoiding overgeneralisation
]

df = pd.read_csv(SOURCE, dtype={"eval_id": str})

dev = df[df["split"].str.lower() == "dev"].copy()

missing = sorted(set(rag_eval_ids) - set(dev["eval_id"]))

if missing:
    raise ValueError(
        "These selected IDs are missing from the dev split: "
        + ", ".join(missing)
    )

rag_dev = dev[dev["eval_id"].isin(rag_eval_ids)].copy()

# Preserve the original evaluation data and mark this subset explicitly.
rag_dev["evaluation_track"] = "rag_retrieval_candidate"
rag_dev["review_status"] = "NEEDS_CORPUS_AND_EXPECTED_ANSWER_REVIEW"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
rag_dev.to_csv(OUTPUT, index=False)

print(f"Candidate questions created: {len(rag_dev)}")
print(f"Saved to: {OUTPUT}")
print("\nSelected questions:")
print(rag_dev[["eval_id", "question", "category"]].to_string(index=False))
