
from pathlib import Path

from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parents[2]
QDRANT_PATH = ROOT / "qdrant_data"
COLLECTION_NAME = "mediguide_hospital_knowledge"

TARGET_IDS = {
    "MGCH-0013",
    "MGCH-0014",
    "MGCH-0226",
    "MGCH-0283",
}


def main():
    client = QdrantClient(path=str(QDRANT_PATH))

    try:
        if not client.collection_exists(COLLECTION_NAME):
            raise RuntimeError(
                f"Collection not found: {COLLECTION_NAME}"
            )

        found = set()

        records, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )

        while True:
            for record in records:
                payload = record.payload or {}
                chunk_id = payload.get("chunk_id")

                if chunk_id in TARGET_IDS:
                    found.add(chunk_id)
                    print(f"\nChunk ID: {chunk_id}")
                    print(f"Title: {payload.get('title')}")
                    print(f"Topic: {payload.get('topic')}")
                    print(f"Text: {payload.get('text')}")

            if next_offset is None:
                break

            records, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                offset=next_offset,
                limit=100,
                with_payload=True,
                with_vectors=False,
            )

        print("\n=== VERIFICATION SUMMARY ===")
        for chunk_id in sorted(TARGET_IDS):
            status = "FOUND" if chunk_id in found else "NOT FOUND"
            print(f"{chunk_id}: {status}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
