
import json
from pathlib import Path

import pandas as pd

# Project root: MediGuide/
ROOT = Path(__file__).resolve().parents[2]

EXCEL_PATH = (
    ROOT
    / "data"
    / "raw"
    / "MediGuide_Final_Hospital_Dataset_Expanded.xlsx"
)

PROCESSED_DIR = ROOT / "data" / "processed"
EVALUATION_DIR = ROOT / "data" / "evaluation"

JSONL_PATH = PROCESSED_DIR / "hospital_knowledge.jsonl"
EVALUATION_PATH = EVALUATION_DIR / "rag_evaluation.csv"

REQUIRED_COLUMNS = {
    "chunk_id",
    "hospital_id",
    "category",
    "topic",
    "title",
    "text",
    "source_ids",
    "data_status",
    "review_status",
    "effective_date",
    "sensitivity",
    "index_policy",
}


def main():
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(
            f"Excel dataset not found: {EXCEL_PATH}"
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load the RAG chunks.
    chunks = pd.read_excel(
        EXCEL_PATH,
        sheet_name="RAG_Chunks"
    )

    # 2. Validate the expected schema.
    missing_columns = REQUIRED_COLUMNS - set(chunks.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    if chunks.empty:
        raise ValueError("RAG_Chunks is empty.")

    # 3. Check IDs.
    if chunks["chunk_id"].isna().any():
        raise ValueError("Some chunks have missing IDs.")

    if chunks["chunk_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Some chunks have empty IDs.")

    if chunks["chunk_id"].duplicated().any():
        duplicates = chunks.loc[
            chunks["chunk_id"].duplicated(keep=False),
            "chunk_id"
        ].tolist()

        raise ValueError(
            f"Duplicate chunk IDs found: {duplicates}"
        )

    # 4. Check text fields.
    for column in ["title", "text", "category"]:
        invalid = (
            chunks[column].isna()
            | chunks[column].astype(str).str.strip().eq("")
        )

        if invalid.any():
            raise ValueError(
                f"{invalid.sum()} rows have empty {column}."
            )

    # 5. Select approved, indexable records only.
    eligible = chunks[
        (chunks["review_status"] == "APPROVED_FOR_SIMULATION")
        & (
            chunks["index_policy"]
            == "INDEXABLE_SYNTHETIC_HOSPITAL_DATA"
        )
    ].copy()

    if eligible.empty:
        raise ValueError("No approved, indexable chunks found.")

    # Report excluded records instead of silently ignoring them.
    excluded_count = len(chunks) - len(eligible)

    # 6. Export the validated candidate corpus as JSONL.
    with JSONL_PATH.open("w", encoding="utf-8") as file:
        for record in eligible.to_dict(orient="records"):
            cleaned = {}

            for key, value in record.items():
                if pd.isna(value):
                    cleaned[key] = None
                elif hasattr(value, "isoformat"):
                    cleaned[key] = value.isoformat()
                else:
                    cleaned[key] = value

            file.write(
                json.dumps(cleaned, ensure_ascii=False) + "\n"
            )

    # 7. Load evaluation questions separately.
    evaluation = pd.read_excel(
        EXCEL_PATH,
        sheet_name="RAG_Evaluation_Set"
    )

    required_eval_columns = {
        "eval_id",
        "question",
        "expected_answer_outline",
        "acceptance_criterion",
    }

    missing_eval_columns = (
        required_eval_columns - set(evaluation.columns)
    )

    if missing_eval_columns:
        raise ValueError(
            "Missing evaluation columns: "
            f"{sorted(missing_eval_columns)}"
        )

    if evaluation["eval_id"].duplicated().any():
        raise ValueError("Duplicate evaluation IDs found.")

    if evaluation["question"].isna().any():
        raise ValueError("Evaluation contains empty questions.")

    evaluation.to_csv(
        EVALUATION_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # 8. Print the validation summary.
    print("\n=== MEDIGUIDE DATASET VALIDATION ===")
    print(f"Total workbook chunks: {len(chunks)}")
    print(f"Approved/indexable chunks: {len(eligible)}")
    print(f"Excluded chunks: {excluded_count}")
    print(f"Unique categories: {eligible['category'].nunique()}")
    print(f"Evaluation questions: {len(evaluation)}")
    print(f"\nJSONL output: {JSONL_PATH}")
    print(f"Evaluation output: {EVALUATION_PATH}")
    print("\nValidation completed successfully.")


if __name__ == "__main__":
    main()
