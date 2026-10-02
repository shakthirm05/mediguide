from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "chunks"
    / "mediguide_rag_chunks_final.jsonl"
)

print("=" * 70)
print("CHECK 1: AFTER REGISTRATION KNOWLEDGE")
print("=" * 70)

matches = []

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)

        text = doc.get("chunk_text", "").lower()
        title = doc.get("title", "").lower()
        category = doc.get("category", "").lower()

        if (
            "after registration" in text
            or "after registration" in title
            or "after registration" in category
        ):
            matches.append(doc)

print(f"Found: {len(matches)} matching chunks")

for doc in matches:
    print("\n---")
    print("Chunk ID:", doc.get("chunk_id"))
    print("Document ID:", doc.get("document_id"))
    print("Title:", doc.get("title"))
    print("Category:", doc.get("category"))
    print("Text:")
    print(doc.get("chunk_text"))