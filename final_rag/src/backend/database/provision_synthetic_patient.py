
import argparse
import os
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
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is missing from the project .env file.")

    claim_code = secrets.token_urlsafe(24)
    claim_code_hash = password_hasher.hash(claim_code)

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO patient_registry (
                    patient_identifier,
                    claim_code_hash
                )
                VALUES (%s, %s)
                """,
                (patient_identifier, claim_code_hash),
            )

    print("\nSynthetic patient provisioned successfully.")
    print(f"Patient identifier: {patient_identifier}")
    print(f"One-time claim code: {claim_code}")
    print("\nGive this code only to the intended test patient.")


def reset_claim_code(patient_identifier: str) -> None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is missing from the project .env file.")

    claim_code = secrets.token_urlsafe(24)
    claim_code_hash = password_hasher.hash(claim_code)

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT linked_user_id
                FROM patient_registry
                WHERE patient_identifier = %s
                FOR UPDATE
                """,
                (patient_identifier,),
            )
            row = cur.fetchone()

            if row is None:
                raise ValueError(
                    "Patient identifier does not exist. "
                    "Provision the patient first."
                )

            if row[0] is not None:
                raise ValueError(
                    "Patient record is already linked. "
                    "Claim code was not changed."
                )

            cur.execute(
                """
                UPDATE patient_registry
                SET claim_code_hash = %s
                WHERE patient_identifier = %s
                  AND linked_user_id IS NULL
                """,
                (claim_code_hash, patient_identifier),
            )

            if cur.rowcount != 1:
                raise ValueError(
                    "Patient record could not be updated safely."
                )

    print("\nClaim code reset successfully.")
    print(f"Patient identifier: {patient_identifier}")
    print(f"New one-time claim code: {claim_code}")
    print("\nGive this code only to the intended test patient.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Provision a synthetic MediGuide patient."
    )
    parser.add_argument(
        "patient_identifier",
        help="Synthetic patient identifier, for example MEDI-P001",
    )
    parser.add_argument(
        "--reset-claim-code",
        action="store_true",
        help="Issue a new code for an existing, unlinked patient record.",
    )
    args = parser.parse_args()

    patient_identifier = args.patient_identifier.strip()
    if not patient_identifier or len(patient_identifier) > 50:
        parser.error("Patient identifier must contain 1-50 characters.")

    try:
        if args.reset_claim_code:
            reset_claim_code(patient_identifier)
        else:
            provision_patient(patient_identifier)
    except psycopg.errors.UniqueViolation:
        print(
            "Error: that patient identifier already exists. "
            "No existing record was changed.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        print(
            f"Operation failed ({type(exc).__name__}). "
            "Check the database connection and schema.",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
