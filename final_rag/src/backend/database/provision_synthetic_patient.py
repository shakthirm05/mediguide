
import argparse
import secrets
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pwdlib import PasswordHash


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

password_hasher = PasswordHash.recommended()


def provision_patient(patient_identifier: str) -> None:
    database_url = __import__("os").getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is missing from the project .env file.")

    # High-entropy, one-time claim code.
    claim_code = secrets.token_urlsafe(24)
    claim_code_hash = password_hasher.hash(claim_code)

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO patient_registry (patient_identifier, claim_code_hash)
                VALUES (%s, %s)
                """,
                (patient_identifier, claim_code_hash),
            )

    # Display the code only after the database transaction succeeds.
    print("\nSynthetic patient provisioned successfully.")
    print(f"Patient identifier: {patient_identifier}")
    print(f"One-time claim code: {claim_code}")
    print("\nGive this code only to the intended test patient.")
    print("It cannot be recovered from the database after this output is lost.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Provision a synthetic MediGuide patient."
    )
    parser.add_argument(
        "patient_identifier",
        help="Synthetic PID, for example MEDI-P001",
    )
    args = parser.parse_args()

    patient_identifier = args.patient_identifier.strip()
    if not patient_identifier or len(patient_identifier) > 50:
        parser.error("Patient identifier must contain 1–50 characters.")

    try:
        provision_patient(patient_identifier)
    except psycopg.errors.UniqueViolation:
        print(
            "Error: that patient identifier already exists. "
            "No existing record was changed.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    except Exception as exc:
        # Do not print claim codes or database credentials in errors.
        print(
            f"Provisioning failed ({type(exc).__name__}). "
            "Check the database connection and schema.",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
