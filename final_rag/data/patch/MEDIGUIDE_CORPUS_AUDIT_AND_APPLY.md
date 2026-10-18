# MediGuide corpus audit and next-step patch
Date: 2026-10-12

## What was verified
- The active working project is reported to use a 306-chunk `MGCH-*` JSONL corpus, 136 separate evaluation questions, Qdrant dense retrieval + BM25 + RRF + cross-encoder reranking.
- The available library CSV `MediGuide_RAG_Chunks.csv` is a different 210-row `MGKB-*` draft snapshot. Do not replace the active 306-row working JSONL with this older/different snapshot.
- The evaluation file has 136 questions; the questions must remain out of the retrieval index.
- The 1,912-chunk external crawl contains institution-specific policy/directory content and must not be indexed wholesale as MediGuide facts. Extract generalizable process principles, preserve source attribution, and create MediGuide-specific synthetic records for local details.

## Known retrieval observations from the existing 14-question dev run
- Admission and discharge questions retrieved relevant policy chunks in top five.
- Visitor-hour questions retrieve related chunks; the active data contains draft/general hours, so ensure the answer clearly uses the defined MediGuide synthetic schedule.
- Restricted-area/children visitor query retrieves related ICU/children/restricted-area chunks at ranks 2–3; this is a ranking opportunity, not a reason to invent a universal age limit.
- Language-support and accessibility queries retrieve relevant chunks near the top.
- Lost-property query retrieves relevant chunks at ranks 1–3.
- Cafeteria closing-time query did not retrieve cafeteria hours in top five. This was a knowledge gap, not merely a reranking problem.
- Department-hours/source-of-truth query retrieves the source-of-truth chunk at rank 1.
- Cardiology location is an exact fictional directory fact and should be stored in structured directory data and/or a clearly synthetic directory chunk; robot routing must still validate the map node.

## Supplemental files
1. `mediguide_supplemental_chunks.jsonl` — 10 additional records using the active MGCH-style schema. These are intended as an additive patch, not a replacement corpus.
2. `mediguide_source_register_additions.csv` — official source links and synthetic source IDs used by the additions.
3. `evaluation_patch.csv` — updates EVAL-PROD-049 because the cafeteria schedule is now defined. Apply the patch to the existing evaluation file rather than creating a second conflicting evaluation row.

## Important application design
- General procedure guidance is research-informed from official sources (MoHFW patient-rights charter, Government of India ORS, and IRDAI).
- Institution-specific details such as MediGuide hours, room codes, desks and routes are synthetic project data.
- Keep exact current schedules, doctors, department directory, appointments and operational state in PostgreSQL as the source of truth. RAG may explain procedures but should not override structured state.
- Do not index external crawled hospital pages as local MediGuide policies.
- Keep synthetic patient records isolated by authenticated patient identity; never mix patient-specific memory into the shared hospital knowledge collection.

## Apply and validate
1. Back up `data/processed/hospital_knowledge.jsonl`.
2. Check the 10 supplemental IDs/text against the active corpus for duplicates, then append only records that are not already present.
3. Rebuild Qdrant from the resulting canonical JSONL and rebuild/reload BM25 from the same corpus; do not update only one retriever.
4. Apply `evaluation_patch.csv` to the existing EVAL-PROD-049 row.
5. Rerun the existing dev retrieval script on the 14 dev questions.
6. Rerun answer-level evaluation and the 94-question held-out test only after the changes are frozen. Do not use the 94 test questions for iterative tuning.
7. Record new metrics only from actual script output. The existing manual relevance labels were assistant-assigned and should be described as provisional until human-reviewed.
