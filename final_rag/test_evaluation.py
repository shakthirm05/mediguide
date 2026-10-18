
import csv
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "evaluation" / "rag_evaluation.csv"
REPORT_DIR = ROOT / "reports" / "evaluation"

sys.path.insert(0, str(ROOT))

from src.backend.services.rag_service import get_rag_service


WIDTH = 78


def line(char="=", width=WIDTH):
    print(char * width)


def heading(title):
    print()
    line("=")
    print(f" {title}")
    line("=")


def table(headers, rows, widths=None):
    """Print a readable ASCII table."""
    headers = [str(h) for h in headers]
    rows = [[str(v) for v in row] for row in rows]

    if widths is None:
        widths = [
            max(len(headers[i]), *(len(row[i]) for row in rows))
            for i in range(len(headers))
        ]

    def print_row(row):
        print("| " + " | ".join(
            value[:widths[i]].ljust(widths[i])
            for i, value in enumerate(row)
        ) + " |")

    separator = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    print(separator)
    print_row(headers)
    print(separator)

    for row in rows:
        print_row(row)

    print(separator)


def percentile(values, percentile_value):
    """Calculate a percentile using linear interpolation."""
    if not values:
        return 0.0

    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value / 100
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower
    return ordered[lower] + (
        ordered[upper] - ordered[lower]
    ) * fraction


def main():
    if not DATASET.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET}"
        )

    with DATASET.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)
        required = {
            "eval_id",
            "question",
            "expected_answer_outline",
            "acceptance_criterion",
            "split",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Missing required CSV columns: {sorted(missing)}"
            )

        all_rows = list(reader)

    test_rows = [
        row for row in all_rows
        if (row.get("split") or "").strip().lower() == "test"
    ]

    dev_count = sum(
        (row.get("split") or "").strip().lower() == "dev"
        for row in all_rows
    )

    if not test_rows:
        raise ValueError("No test questions found in the dataset.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    latencies = []
    service = None

    heading("MEDIGUIDE RAG EVALUATION REPORT")

    print(f"Generated at       : {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}")
    print(f"Dataset            : {DATASET}")
    print(f"Total questions    : {len(all_rows)}")
    print(f"Test set           : {len(test_rows)}")
    print(f"Development set    : {dev_count}")
    print("Embedding model    : sentence-transformers/all-MiniLM-L6-v2")
    print("Vector database    : Qdrant")
    print("Collection         : mediguide_hospital_knowledge")
    print("Retrieval pipeline : Dense + BM25 + RRF + reranker")
    print("Evaluation scope   : Generation execution and latency")

    try:
        service = get_rag_service()

        heading("PER-QUESTION EXECUTION RESULTS")

        print(
            f"{'No.':<5} {'Evaluation ID':<18} "
            f"{'Status':<10} {'Latency (s)':>12}"
        )
        line("-")

        for index, row in enumerate(test_rows, start=1):
            question = (row.get("question") or "").strip()
            started = time.perf_counter()

            answer = ""
            evidence = []
            error = ""

            try:
                response = service.ask(question)

                if isinstance(response, dict):
                    answer = str(response.get("answer") or "")
                    evidence = response.get("evidence") or []
                else:
                    answer = str(response)

            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            elapsed = time.perf_counter() - started
            latencies.append(elapsed)

            successful = bool(answer.strip()) and not error

            results.append({
                "eval_id": row.get("eval_id", ""),
                "split": row.get("split", ""),
                "category": row.get("category", ""),
                "question": question,
                "expected_answer_outline": row.get(
                    "expected_answer_outline", ""
                ),
                "acceptance_criterion": row.get(
                    "acceptance_criterion", ""
                ),
                "answer": answer,
                "evidence": json.dumps(
                    evidence, ensure_ascii=False, default=str
                ),
                "latency_seconds": round(elapsed, 4),
                "generation_success": successful,
                "error": error,
            })

            status = "OK" if successful else "FAILED"

            print(
                f"{index:03}/{len(test_rows):03} "
                f"{row.get('eval_id', ''):<18} "
                f"{status:<10} {elapsed:>10.2f}"
            )

            if error:
                print(f"    Error: {error}")
            elif not answer.strip():
                print("    Error: Empty answer")

    finally:
        if service is not None:
            close = getattr(service, "close", None)
            if callable(close):
                close()

    total = len(results)
    successes = sum(
        result["generation_success"] for result in results
    )
    failures = total - successes

    success_rate = successes / total if total else 0.0
    failure_rate = failures / total if total else 0.0

    average = statistics.mean(latencies) if latencies else 0.0
    median = statistics.median(latencies) if latencies else 0.0
    p95 = percentile(latencies, 95)
    minimum = min(latencies) if latencies else 0.0
    maximum = max(latencies) if latencies else 0.0

    # Save detailed per-question output.
    results_path = REPORT_DIR / "rag_evaluation_results.csv"

    with results_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "total_dataset_questions": len(all_rows),
        "test_questions": total,
        "development_questions": dev_count,
        "generation_success_count": successes,
        "generation_failure_count": failures,
        "generation_success_rate": round(success_rate, 4),
        "generation_failure_rate": round(failure_rate, 4),
        "latency_seconds": {
            "average": round(average, 4),
            "median": round(median, 4),
            "p95": round(p95, 4),
            "minimum": round(minimum, 4),
            "maximum": round(maximum, 4),
        },
        "retrieval_metrics": None,
        "intent_classification_metrics": None,
        "answer_quality_metrics": None,
        "acceptance_criterion_pass_rate": None,
        "limitations": [
            "Generation success does not establish answer correctness.",
            "Retrieval relevance labels are unavailable.",
            "Acceptance criteria were not automatically scored.",
        ],
    }

    summary_path = REPORT_DIR / "rag_evaluation_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    heading("GENERATION EXECUTION METRICS")

    table(
        ["Metric", "Measured value"],
        [
            ["Total test questions", str(total)],
            ["Successful generations", str(successes)],
            ["Failed generations", str(failures)],
            ["Generation success rate", f"{success_rate * 100:.2f}%"],
            ["Generation failure rate", f"{failure_rate * 100:.2f}%"],
        ],
        [34, 25],
    )

    heading("RESPONSE TIME STATISTICS")

    table(
        ["Statistic", "Latency (seconds)"],
        [
            ["Average", f"{average:.3f}"],
            ["Median", f"{median:.3f}"],
            ["95th percentile (P95)", f"{p95:.3f}"],
            ["Minimum", f"{minimum:.3f}"],
            ["Maximum", f"{maximum:.3f}"],
        ],
        [34, 25],
    )

    heading("RETRIEVAL QUALITY")

    table(
        ["Metric", "Status"],
        [
            ["Hit Rate @ K", "NOT EVALUATED"],
            ["Precision @ K", "NOT EVALUATED"],
            ["Recall @ K", "NOT EVALUATED"],
            ["F1 Score", "NOT EVALUATED"],
            ["Mean Reciprocal Rank", "NOT EVALUATED"],
        ],
        [34, 25],
    )

    print(
        "Reason: the evaluation CSV has no verified relevant chunk IDs."
    )

    heading("ANSWER QUALITY AND INTENT CLASSIFICATION")

    table(
        ["Evaluation dimension", "Status"],
        [
            ["Answer relevance", "NOT EVALUATED"],
            ["Faithfulness", "NOT EVALUATED"],
            ["Context relevance", "NOT EVALUATED"],
            ["Acceptance-criterion pass rate", "NOT EVALUATED"],
            ["Intent classification metrics", "NOT EVALUATED"],
            ["Confusion matrix", "NOT EVALUATED"],
        ],
        [38, 21],
    )

    heading("FINAL EVALUATION SUMMARY")

    table(
        ["Category", "Result"],
        [
            ["Execution", "COMPLETED"],
            ["Test questions processed", f"{total}/{total}"],
            ["Generation success rate", f"{success_rate * 100:.2f}%"],
            ["Average response time", f"{average:.3f} seconds"],
            ["Median response time", f"{median:.3f} seconds"],
            ["Answer correctness", "NOT ESTABLISHED"],
            ["Retrieval accuracy", "NOT ESTABLISHED"],
        ],
        [34, 28],
    )

    print("\nInterpretation:")
    print(
        "- Successful generation means a non-empty answer was returned "
        "without an exception."
    )
    print(
        "- It does not mean the answer was correct or supported by evidence."
    )
    print(
        "- Retrieval and answer-quality metrics require additional labels "
        "and evaluation."
    )

    print("\nSaved artifacts:")
    print(f"Detailed results : {results_path}")
    print(f"Summary JSON     : {summary_path}")

    line("=")
    print("Evaluation run finished.")


if __name__ == "__main__":
    main()
