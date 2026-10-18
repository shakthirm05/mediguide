#!/usr/bin/env python3
"""
MediGuide public-source collector + chunker + old-dataset merger.

What it does:
1) Starts from a curated list of public patient-information websites.
2) Checks robots.txt before requesting each page and uses a polite delay.
3) Crawls relevant same-domain pages and follows sitemaps when available.
4) Extracts visible HTML text and relevant links.
5) Deduplicates documents and chunks by normalized text.
6) Optionally merges the old MediGuide 710-row CSV without deleting audit metadata.
7) Writes raw document registry, new chunks, merged chunks, and a build report.

IMPORTANT:
- This script does not bypass logins, paywalls, CAPTCHAs, or access restrictions.
- Check each website's terms/licence and robots.txt before collecting or reusing content.
- Respect copyright. Keep source URLs and attribution; use content only where permitted.
- It cannot guarantee 6,000 chunks. It reports the real count. Add permitted sources if
  the target is not met; never pad the corpus with duplicates or generated filler.
- Every newly collected chunk is marked UNREVIEWED and not production-answerable by default.
- External hospitals' policies (hours, prices, contact details, rules, eligibility) must
  not be presented as MediGuide General Hospital policy.
- The existing CSV is preserved as-is in the merged output; quarantined rows stay quarantined.
- HTML pages only in this version; PDF extraction is not implemented here.

Install:
  python -m pip install requests beautifulsoup4 lxml pandas

Example (run from final_rag\\dataset_work):
  python mediguide_collect_and_chunk.py --outdir output --target-chunks 6000 --max-pages 12000 --delay 1.5

Merge existing dataset:
  python mediguide_collect_and_chunk.py --outdir output --old-csv "..\\MediGuide_RAG_Final_Audited_710.csv" --target-chunks 6000
"""

import argparse
import csv
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.robotparser
from collections import deque
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup

USER_AGENT = "MediGuideResearchBot/1.0 (educational corpus research; polite public-page collector)"

# Start small and expand only after reviewing the sites' terms and crawl rules.
SEED_URLS = [
    "https://www.nhs.uk/nhs-services/hospitals/",
    "https://www.nhs.uk/nhs-services/hospitals/going-into-hospital/",
    "https://www.nhs.uk/nhs-services/hospitals/going-into-hospital/outpatients-and-day-patients/",
    "https://www.nhs.uk/nhs-services/hospitals/going-into-hospital/visiting-someone-in-hospital/",
    "https://www.nhs.uk/nhs-services/hospitals/going-into-hospital/being-discharged-from-hospital/",
    "https://medlineplus.gov/",
    "https://medlineplus.gov/personalhealthissues.html",
    "https://www.ahrq.gov/patients-consumers/index.html",
    "https://www.cdc.gov/infection-control/",
    "https://www.who.int/health-topics/patient-safety",
    "https://my.clevelandclinic.org/patients/information",
    "https://www.mayoclinic.org/patient-visitor-guide",
]

# Exact host allowlist: links to other domains are never crawled automatically.
ALLOWED_HOSTS = {
    "nhs.uk", "www.nhs.uk",
    "medlineplus.gov", "www.medlineplus.gov",
    "ahrq.gov", "www.ahrq.gov",
    "cdc.gov", "www.cdc.gov",
    "who.int", "www.who.int",
    "my.clevelandclinic.org", "clevelandclinic.org", "www.clevelandclinic.org",
    "mayoclinic.org", "www.mayoclinic.org",
}

RELEVANT_TERMS = [
    "hospital", "patient", "appointment", "outpatient", "admission", "discharge",
    "visitor", "visiting", "accessibility", "accessible", "interpreter", "language",
    "transport", "parking", "patient-right", "patient right", "privacy", "consent",
    "safety", "infection-control", "infection control", "hand-hygiene", "hand hygiene",
    "billing", "insurance", "financial-assistance", "department", "service", "clinic",
    "wayfinding", "prepare", "hospital-stay", "hospital stay", "support", "complaint",
    "feedback", "communication", "caregiver", "carer", "disability", "medical-record",
    "medical record", "patient-experience", "patient experience", "healthcare",
]
EXCLUDED_PATH_TERMS = [
    "/login", "/signin", "/sign-in", "/account", "/shop", "/donate", "/jobs",
    "/careers", "/press", "/newsroom", "/research-study", "/clinical-trial",
]
TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid"}

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def normalize_url(url):
    if not url:
        return None
    url, _ = urldefrag(url.strip())
    p = urlparse(url)
    host = (p.hostname or "").lower()
    if p.scheme not in ("http", "https") or host not in ALLOWED_HOSTS:
        return None
    # Remove common tracking parameters; skip search/filter query URLs.
    q = urllib.parse.parse_qsl(p.query, keep_blank_values=False)
    if any(k.lower() not in TRACKING_PARAMS for k, _ in q):
        return None
    path = re.sub(r"/{2,}", "/", p.path or "/")
    return urllib.parse.urlunparse((p.scheme, host, path.rstrip("/") or "/", "", "", ""))

def relevant_url(url):
    p = urlparse(url)
    low = (p.path or "/").lower()
    if any(term in low for term in EXCLUDED_PATH_TERMS):
        return False
    # Allow seeded pages and most relevant paths; avoid indiscriminate site-wide crawling.
    return any(term in low or term in url.lower() for term in RELEVANT_TERMS) or low in ("/", "")

def normalized_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()

def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def host_origin(url):
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"

def robots_allowed(url, session, cache):
    origin = host_origin(url)
    if origin not in cache:
        rp = urllib.robotparser.RobotFileParser()
        robots_url = origin + "/robots.txt"
        try:
            resp = session.get(robots_url, timeout=20)
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
                cache[origin] = rp
            elif resp.status_code in (401, 403):
                cache[origin] = False
            else:
                # If robots.txt is unavailable, stop and ask the operator to review manually.
                cache[origin] = None
        except requests.RequestException:
            cache[origin] = None
    rp = cache[origin]
    if rp is False:
        return False
    if rp is None:
        return False
    return rp.can_fetch(USER_AGENT, url)

def extract_html(url, session):
    response = session.get(url, timeout=30, allow_redirects=True)
    if response.status_code != 200:
        return None, []
    content_type = response.headers.get("Content-Type", "").lower()
    if "text/html" not in content_type and "application/xhtml" not in content_type:
        return None, []
    soup = BeautifulSoup(response.text, "lxml")
    for element in soup(["script", "style", "nav", "footer", "header", "aside", "form",
                         "noscript", "svg", "button", "iframe"]):
        element.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else url
    main = soup.find("main") or soup.find("article") or soup.find(id=re.compile("main|content", re.I)) or soup.body or soup
    headings = [h.get_text(" ", strip=True) for h in main.find_all(["h1", "h2", "h3"])]
    pieces = []
    for element in main.find_all(["p", "li", "dt", "dd", "th", "td"]):
        text = re.sub(r"\s+", " ", element.get_text(" ", strip=True)).strip()
        if len(text) >= 40 and text.lower() not in {"menu", "search", "skip to main content"}:
            pieces.append(text)
    # Keep paragraph order but remove exact repeats.
    body = "\n".join(dict.fromkeys(pieces))
    links = []
    for a in soup.find_all("a", href=True):
        candidate = normalize_url(urljoin(response.url, a["href"]))
        if candidate and urlparse(candidate).hostname == urlparse(url).hostname and relevant_url(candidate):
            links.append(candidate)
    if len(body) < 500:
        return None, links
    return {
        "requested_url": url,
        "source_url": response.url,
        "title": title[:300],
        "headings": headings,
        "text": body,
        "retrieved_at": utc_now(),
        "last_modified": response.headers.get("Last-Modified", ""),
        "source_domain": urlparse(response.url).hostname,
    }, links

def infer_category(url, title, text):
    s = (url + " " + title + " " + text[:1500]).lower()
    rules = [
        ("appointments_checkin", ["appointment", "outpatient", "check-in", "check in"]),
        ("admission_inpatient", ["admission", "inpatient", "hospital stay", "prepare for hospital"]),
        ("discharge_planning", ["discharge", "leaving hospital", "going home"]),
        ("visitors", ["visitor", "visiting", "family and friends"]),
        ("accessibility_language", ["accessibility", "accessible", "interpreter", "language access", "disability"]),
        ("patient_rights_privacy", ["patient rights", "privacy", "medical records", "consent", "confidentiality"]),
        ("patient_safety", ["patient safety", "infection control", "hand hygiene", "fall prevention"]),
        ("transport_facilities", ["transport", "parking", "directions", "getting here", "wayfinding"]),
        ("billing_financial", ["billing", "insurance", "financial assistance", "payment"]),
        ("hospital_services", ["department", "services", "clinic", "hospital services"]),
        ("communication_support", ["communication", "support", "complaint", "feedback", "caregiver"]),
    ]
    for category, terms in rules:
        if any(term in s for term in terms):
            return category
    return "general_patient_support"

def scope_label(url, title, text):
    s = (url + " " + title + " " + text[:2500]).lower()
    external_policy_terms = [
        "visiting hours", "phone:", "telephone", "parking fee", "opening hours",
        "at cleveland clinic", "at mayo clinic", "our hospital", "our clinic",
        "insurance plan", "eligible for", "eligibility", "must be accompanied",
        "our policy", "our visitor policy", "contact us", "call us",
    ]
    clinical_terms = [
        "diagnose", "diagnosis", "treatment options", "dosage", "clinical guideline",
        "take this medication", "prescribe", "symptoms of", "medical emergency",
    ]
    if any(term in s for term in external_policy_terms):
        return "EXTERNAL_POLICY_REFERENCE"
    if any(term in s for term in clinical_terms):
        return "CLINICAL_OR_SAFETY_REVIEW_REQUIRED"
    return "GENERAL_REFERENCE_REVIEW_REQUIRED"

def chunk_paragraphs(text, max_chars=2200, overlap=200, min_chars=350):
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        sentences = [para]
        if len(para) > max_chars:
            sentences = re.split(r"(?<=[.!?])\s+", para)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current) + len(sentence) + 1 <= max_chars:
                current = (current + "\n" + sentence).strip()
            else:
                if len(current) >= min_chars:
                    chunks.append(current)
                tail = current[-overlap:] if overlap and current else ""
                current = (tail + "\n" + sentence).strip()
    if len(current) >= min_chars:
        chunks.append(current)
    return chunks

def read_old_csv(path):
    if not path:
        return []
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            # Preserve every original field and status as supplied.
            row["_dataset_origin"] = "existing_710_dataset"
            rows.append(row)
    return rows

def write_csv(path, rows):
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="output")
    parser.add_argument("--old-csv", default="")
    parser.add_argument("--target-chunks", type=int, default=6000)
    parser.add_argument("--max-pages", type=int, default=12000)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--max-chars", type=int, default=2200)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint.json in --outdir")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    checkpoint_path = os.path.join(args.outdir, "checkpoint.json")
    docs_path = os.path.join(args.outdir, "MediGuide_Web_Documents.csv")
    chunks_path = os.path.join(args.outdir, "MediGuide_New_Web_Chunks.csv")
    failures_path = os.path.join(args.outdir, "MediGuide_Crawl_Failures.csv")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    session.max_redirects = 5
    robots_cache = {}

    documents, new_chunks, failures = [], [], []
    seen_urls, seen_doc_hashes, seen_chunk_hashes = set(), set(), set()
    queue = deque(SEED_URLS)

    if args.resume and os.path.exists(checkpoint_path):
        try:
            state = json.load(open(checkpoint_path, encoding="utf-8"))
            queue = deque(state.get("queue", []))
            seen_urls = set(state.get("seen_urls", []))
            documents = state.get("documents", [])
            new_chunks = state.get("new_chunks", [])
            failures = state.get("failures", [])
            seen_doc_hashes = {digest(normalized_text(d.get("text", ""))) for d in documents}
            seen_chunk_hashes = {digest(normalized_text(c.get("chunk_text", ""))) for c in new_chunks}
            print(f"Resumed: {len(seen_urls)} pages visited, {len(new_chunks)} chunks saved, {len(queue)} URLs queued.")
        except Exception as exc:
            raise SystemExit(f"Could not read checkpoint: {exc}")

    def save_checkpoint():
        state = {
            "queue": list(queue),
            "seen_urls": list(seen_urls),
            "documents": documents,
            "new_chunks": new_chunks,
            "failures": failures,
            "saved_at": utc_now(),
        }
        tmp = checkpoint_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
        os.replace(tmp, checkpoint_path)
        write_csv(docs_path, documents)
        write_csv(chunks_path, new_chunks)
        write_csv(failures_path, failures)

    # Apply the requested timeout to all subsequent HTTP requests.
    original_get = session.get
    def quick_get(url, *a, **kw):
        kw["timeout"] = args.timeout
        return original_get(url, *a, **kw)
    session.get = quick_get

    processed_since_save = 0
    started = time.time()
    while queue and len(seen_urls) < args.max_pages and len(new_chunks) < args.target_chunks:
        raw_url = queue.popleft()
        url = normalize_url(raw_url)
        if not url or url in seen_urls or not relevant_url(url):
            continue
        seen_urls.add(url)

        if not robots_allowed(url, session, robots_cache):
            failures.append({"url": url, "reason": "robots disallowed or robots.txt could not be checked"})
            processed_since_save += 1
        else:
            try:
                page, links = extract_html(url, session)
            except requests.RequestException as exc:
                failures.append({"url": url, "reason": str(exc)[:300]})
                page, links = None, []
            except Exception as exc:
                failures.append({"url": url, "reason": f"parse error: {str(exc)[:250]}"})
                page, links = None, []

            for link in links:
                if link not in seen_urls and len(queue) < args.max_pages * 3:
                    queue.append(link)

            if page:
                dh = digest(normalized_text(page["text"]))
                if dh not in seen_doc_hashes:
                    seen_doc_hashes.add(dh)
                    page["document_id"] = "DOC_" + dh[:16]
                    page["category"] = infer_category(page["source_url"], page["title"], page["text"])
                    page["scope_label"] = scope_label(page["source_url"], page["title"], page["text"])
                    documents.append(page)
                    for idx, chunk in enumerate(chunk_paragraphs(
                        page["text"], max_chars=args.max_chars, overlap=args.overlap
                    ), start=1):
                        ch = normalized_text(chunk)
                        if len(ch) < 250:
                            continue
                        chash = digest(ch)
                        if chash in seen_chunk_hashes:
                            continue
                        seen_chunk_hashes.add(chash)
                        new_chunks.append({
                            "chunk_id": f"{page['document_id']}_CH{idx:03d}",
                            "document_id": page["document_id"],
                            "source_url": page["source_url"],
                            "source_domain": page["source_domain"],
                            "source_title": page["title"],
                            "retrieved_at": page["retrieved_at"],
                            "category": page["category"],
                            "scope_label": page["scope_label"],
                            "review_status": "UNREVIEWED",
                            "production_eligible": "NO",
                            "chunk_text": chunk,
                            "chunk_sha256": chash,
                        })
                        if len(new_chunks) >= args.target_chunks:
                            break
            processed_since_save += 1

        if processed_since_save >= args.checkpoint_every:
            save_checkpoint()
            elapsed = int(time.time() - started)
            print(f"[{elapsed}s] visited={len(seen_urls)} docs={len(documents)} chunks={len(new_chunks)} queued={len(queue)} failures={len(failures)}")
            processed_since_save = 0
        if args.delay > 0:
            time.sleep(args.delay)

    save_checkpoint()

    # Preserve old data fields and audit labels. Merge only if requested.
    old_rows = read_old_csv(args.old_csv) if args.old_csv else []
    merged_rows = old_rows + new_chunks
    if args.old_csv:
        write_csv(os.path.join(args.outdir, "MediGuide_Merged_Chunks.csv"), merged_rows)
        with open(os.path.join(args.outdir, "MediGuide_Merged_Chunks.jsonl"), "w", encoding="utf-8") as f:
            for row in merged_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(os.path.join(args.outdir, "MediGuide_New_Web_Chunks.jsonl"), "w", encoding="utf-8") as f:
        for row in new_chunks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "generated_at": utc_now(),
        "visited_urls": len(seen_urls),
        "unique_documents": len(documents),
        "new_unique_chunks": len(new_chunks),
        "old_rows_merged": len(old_rows),
        "merged_total_rows": len(merged_rows) if args.old_csv else None,
        "target_chunks": args.target_chunks,
        "target_reached": len(new_chunks) >= args.target_chunks,
        "queue_remaining": len(queue),
        "failures": len(failures),
        "checkpoint_file": checkpoint_path,
        "warning": "New web chunks are unreviewed. Do not ingest them into production without source/licence review, scope review, deduplication against old corpus, and safety review."
    }
    with open(os.path.join(args.outdir, "MediGuide_Build_Report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print("Finished. If interrupted, rerun the same command with --resume to continue from the last checkpoint.")

if __name__ == "__main__":
    main()
