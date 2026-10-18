#!/usr/bin/env python3
"""
MediGuide General Hospital: source-first public-web corpus builder.

This script crawls only explicitly allowlisted public domains, discovers pages from
robots.txt/sitemaps and a seed manifest, extracts page text, deduplicates pages and
chunks, and exports a provenance-rich JSONL/CSV dataset. It DOES NOT invent policy
facts and it does not certify medical content as safe.

Install:
  pip install requests beautifulsoup4 lxml pandas

Run:
  python build_mediguide_internet_corpus.py \
      --manifest MediGuide_Internet_Source_Manifest.csv \
      --outdir mediguide_corpus \
      --target-chunks 6000

Important:
- Check each source site's robots.txt, terms, and reuse/licensing requirements before use.
- Use polite crawl delays and do not bypass access controls/paywalls.
- Target is a minimum target, not a promise: the script reports the real number it found.
- Only keep content relevant to hospital navigation, services, patient communication,
  appointments, admission/discharge, accessibility, patient rights, safety and support.
- Exclude clinical diagnosis/treatment advice from the MediGuide answerable corpus.
- Any institution-specific details are tagged EXTERNAL_POLICY_REFERENCE and must not
  be presented as MediGuide General Hospital policy.
"""

import argparse, csv, hashlib, json, os, re, time, urllib.parse, urllib.robotparser
from collections import deque
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse, urldefrag
import requests
from bs4 import BeautifulSoup

USER_AGENT = "MediGuideResearchCorpusBot/1.0 (educational research; contact project maintainer)"
ALLOWLIST = {
    "nhs.uk", "www.nhs.uk",
    "medlineplus.gov", "www.medlineplus.gov",
    "cdc.gov", "www.cdc.gov",
    "ahrq.gov", "www.ahrq.gov",
    "my.clevelandclinic.org", "clevelandclinic.org", "www.clevelandclinic.org",
    "mayoclinic.org", "www.mayoclinic.org",
    "healthinfonet.ecu.edu.au", "www.healthinfonet.ecu.edu.au",
    "health.gov", "www.health.gov",
    "who.int", "www.who.int",
}
INCLUDE_TERMS = [
    "patient", "hospital", "appointment", "admission", "discharge", "visitor",
    "visiting", "accessibility", "accessible", "interpreter", "language",
    "transport", "parking", "medical-record", "medical record", "patient-right",
    "patient right", "privacy", "consent", "safety", "infection-prevention",
    "infection prevention", "hand hygiene", "billing", "insurance", "financial",
    "department", "service", "clinic", "wayfinding", "prepare", "prepare-for",
    "hospital-stay", "hospital stay", "support", "complaint", "feedback",
    "communication", "caregiver", "carer", "disability", "discharge-planning",
    "emergency-department", "emergency department", "test", "procedure",
]
EXCLUDE_URL_TERMS = [
    "/drugs/", "/drug/", "/prescription/", "/symptoms/", "/diagnosis/",
    "/treatment/", "/disease/", "/condition/", "/clinical-guideline/",
    "/hcp/", "/professionals/", "/research/", "/news/", "/press/",
    "/jobs/", "/careers/", "/donate", "/shop", "/login", "/signin",
]
BOILERPLATE = {
    "skip to main content", "menu", "search", "cookie settings", "accept all cookies",
    "privacy policy", "terms and conditions", "back to top",
}

def normalize_url(url):
    url, _ = urldefrag(url)
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return None
    if p.hostname not in ALLOWLIST:
        return None
    if p.query:
        # Most query variants are tracking or search; skip to reduce duplication.
        return None
    path = re.sub(r"/{2,}", "/", p.path or "/")
    return urllib.parse.urlunparse((p.scheme, p.netloc.lower(), path.rstrip("/") or "/", "", "", ""))

def url_relevant(url):
    u = url.lower()
    if any(t in u for t in EXCLUDE_URL_TERMS):
        return False
    return any(t in u for t in INCLUDE_TERMS) or u.endswith((".pdf", ".html", ".htm"))

def canonical_text(s):
    s = re.sub(r"\s+", " ", s).strip().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()

def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def robots_allowed(url, session, cache):
    origin = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    if origin not in cache:
        rp = urllib.robotparser.RobotFileParser()
        robots_url = origin + "/robots.txt"
        try:
            r = session.get(robots_url, timeout=15)
            if r.status_code == 200:
                rp.parse(r.text.splitlines())
            else:
                rp = None
        except Exception:
            rp = None
        cache[origin] = rp
    rp = cache[origin]
    return True if rp is None else rp.can_fetch(USER_AGENT, url)

def get_sitemaps(domain, session):
    urls = [f"https://{domain}/robots.txt", f"https://{domain}/sitemap.xml"]
    sitemaps = set()
    for u in urls:
        try:
            r = session.get(u, timeout=20)
            if r.status_code != 200:
                continue
            if u.endswith("robots.txt"):
                for line in r.text.splitlines():
                    if line.lower().startswith("sitemap:"):
                        sitemaps.add(line.split(":", 1)[1].strip())
            else:
                sitemaps.add(u)
        except Exception:
            pass
    return sorted(sitemaps)

def sitemap_urls(sitemap, session, seen=None, max_sitemaps=200):
    if seen is None:
        seen = set()
    found = []
    if sitemap in seen or len(seen) >= max_sitemaps:
        return found
    seen.add(sitemap)
    try:
        r = session.get(sitemap, timeout=30)
        if r.status_code != 200 or not r.text.lstrip().startswith("<"):
            return found
        soup = BeautifulSoup(r.text, "xml")
        if soup.find("sitemapindex"):
            for loc in soup.find_all("loc"):
                found.extend(sitemap_urls(loc.get_text(strip=True), session, seen, max_sitemaps))
        else:
            found.extend(loc.get_text(strip=True) for loc in soup.find_all("loc"))
    except Exception:
        pass
    return found

def extract_page(url, session):
    headers = {"User-Agent": USER_AGENT}
    r = session.get(url, headers=headers, timeout=25)
    if r.status_code != 200:
        return None
    ctype = r.headers.get("Content-Type", "").lower()
    if "text/html" not in ctype and "application/xhtml" not in ctype:
        return None
    soup = BeautifulSoup(r.text, "lxml")
    for el in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript", "svg"]):
        el.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else urlparse(url).path.rsplit("/", 1)[-1]
    main = soup.find("main") or soup.find("article") or soup.find(id=re.compile("main|content", re.I)) or soup.body or soup
    headings = []
    for h in main.find_all(["h1", "h2", "h3"]):
        t = h.get_text(" ", strip=True)
        if t and len(t) < 180:
            headings.append(t)
    paragraphs = []
    for el in main.find_all(["p", "li", "dt", "dd", "th", "td"]):
        t = el.get_text(" ", strip=True)
        t = re.sub(r"\s+", " ", t)
        if len(t) >= 45 and t.lower() not in BOILERPLATE:
            paragraphs.append(t)
    body = "\n".join(dict.fromkeys(paragraphs))
    if len(body) < 450:
        return None
    return {"url": url, "title": title[:300], "headings": headings, "text": body,
            "last_modified": r.headers.get("Last-Modified", "")}

def infer_category(url, title, text):
    s = (url + " " + title + " " + text[:1000]).lower()
    rules = [
        ("appointments", ["appointment", "outpatient", "check-in", "check in"]),
        ("admissions", ["admission", "inpatient", "hospital stay", "prepare for hospital"]),
        ("discharge", ["discharge", "leaving hospital", "going home"]),
        ("visitors", ["visitor", "visiting", "family and friends"]),
        ("accessibility", ["accessibility", "accessible", "interpreter", "language access", "disability"]),
        ("patient_rights_privacy", ["patient rights", "privacy", "medical records", "consent", "confidentiality"]),
        ("patient_safety", ["patient safety", "infection prevention", "hand hygiene", "fall prevention"]),
        ("transport_facilities", ["transport", "parking", "directions", "getting here", "wayfinding"]),
        ("billing_financial", ["billing", "insurance", "financial assistance", "payment"]),
        ("hospital_services", ["department", "services", "clinic", "hospital services"]),
        ("patient_communication_support", ["talking with your doctor", "communication", "support", "complaint", "feedback"]),
    ]
    for cat, terms in rules:
        if any(t in s for t in terms):
            return cat
    return "general_patient_support"

def scope_label(url, title, text):
    s = (url + " " + title + " " + text[:1800]).lower()
    # External institutional rules, contact details, prices, exact hours, portal names,
    # location names, and eligibility criteria must not be treated as MediGuide policy.
    institution_specific = [
        "visiting hours", "call ", "phone:", "telephone", "£", "$", "nhs number",
        "mychart", "pals", "insurance plan", "eligible for", "eligibility",
        "our hospital", "at cleveland clinic", "at mayo clinic", "at this hospital",
        "parking fee", "opening hours", "visitors are allowed", "must be accompanied",
    ]
    if any(x in s for x in institution_specific):
        return "EXTERNAL_POLICY_REFERENCE"
    if any(x in s for x in ["diagnose", "treatment options", "take this medication", "dosage", "clinical guideline"]):
        return "CLINICAL_REVIEW_REQUIRED"
    return "GENERAL_REFERENCE_REVIEW_REQUIRED"

def chunk_text(text, max_chars=2400, overlap=250):
    # Paragraph-aware chunking: chunks remain semantically coherent and overlap modestly.
    paras = [p.strip() for p in text.splitlines() if p.strip()]
    chunks, cur = [], ""
    for para in paras:
        if len(para) > max_chars:
            # Split long paragraphs at sentence boundaries first.
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                if len(cur) + len(sentence) + 1 <= max_chars:
                    cur = (cur + " " + sentence).strip()
                else:
                    if len(cur) >= 350:
                        chunks.append(cur)
                    tail = cur[-overlap:] if overlap and cur else ""
                    cur = (tail + " " + sentence).strip()
        elif len(cur) + len(para) + 2 <= max_chars:
            cur = (cur + "\n" + para).strip()
        else:
            if len(cur) >= 350:
                chunks.append(cur)
            tail = cur[-overlap:] if overlap and cur else ""
            cur = (tail + "\n" + para).strip()
    if len(cur) >= 350:
        chunks.append(cur)
    return chunks

def load_manifest(path):
    seeds = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            url = normalize_url(row.get("seed_url", "").strip())
            if url:
                seeds.append(url)
    return list(dict.fromkeys(seeds))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="MediGuide_Internet_Source_Manifest.csv")
    ap.add_argument("--outdir", default="mediguide_corpus")
    ap.add_argument("--target-chunks", type=int, default=6000)
    ap.add_argument("--max-pages", type=int, default=50000)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--max-chars", type=int, default=2400)
    ap.add_argument("--overlap", type=int, default=250)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    robots_cache = {}
    seeds = load_manifest(args.manifest)
    queue = deque(seeds)
    for domain in sorted(ALLOWLIST):
        for sm in get_sitemaps(domain, session):
            for u in sitemap_urls(sm, session):
                nu = normalize_url(u)
                if nu and url_relevant(nu):
                    queue.append(nu)

    seen_urls, seen_pages, seen_chunks = set(), set(), set()
    pages, chunks = [], []
    while queue and len(seen_urls) < args.max_pages:
        url = normalize_url(queue.popleft())
        if not url or url in seen_urls or not url_relevant(url):
            continue
        seen_urls.add(url)
        if not robots_allowed(url, session, robots_cache):
            continue
        try:
            page = extract_page(url, session)
        except Exception:
            page = None
        if page:
            normalized = canonical_text(page["text"])
            ph = sha(normalized)
            if ph not in seen_pages:
                seen_pages.add(ph)
                page_id = "doc_" + sha(url)[:16]
                cat = infer_category(url, page["title"], page["text"])
                scope = scope_label(url, page["title"], page["text"])
                pages.append({
                    "document_id": page_id, "title": page["title"], "source_url": url,
                    "source_domain": urlparse(url).netloc, "source_organization": urlparse(url).netloc,
                    "category": cat, "scope_status": scope, "last_modified": page["last_modified"],
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "content_sha256": ph, "text_chars": len(page["text"]),
                })
                for idx, txt in enumerate(chunk_text(page["text"], args.max_chars, args.overlap), 1):
                    norm = canonical_text(txt)
                    if len(norm) < 300:
                        continue
                    ch = sha(norm)
                    if ch in seen_chunks:
                        continue
                    seen_chunks.add(ch)
                    chunks.append({
                        "chunk_id": "chunk_" + ch[:20], "document_id": page_id,
                        "title": page["title"], "source_url": url,
                        "source_domain": urlparse(url).netloc, "category": cat,
                        "scope_status": scope, "chunk_index": idx, "chunk_text": txt,
                        "content_sha256": ch, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "answerable_by_default": False,
                        "review_status": "UNREVIEWED_SOURCE_EXTRACT",
                        "hospital_scope": "GENERAL_REFERENCE_ONLY",
                    })
            # Discover relevant same-domain links from the page.
            try:
                # Re-fetch already fetched HTML only to keep extractor simple; link discovery
                # is best-effort and is limited to the same allowlisted host.
                r = session.get(url, timeout=20)
                soup = BeautifulSoup(r.text, "lxml")
                for a in soup.find_all("a", href=True):
                    candidate = normalize_url(urljoin(url, a["href"]))
                    if candidate and urlparse(candidate).netloc == urlparse(url).netloc and url_relevant(candidate):
                        if candidate not in seen_urls:
                            queue.append(candidate)
            except Exception:
                pass
        time.sleep(max(0.2, args.delay))

    # JSONL is canonical; CSV is convenient for audits.
    jsonl_path = os.path.join(args.outdir, "MediGuide_Internet_Chunks.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in chunks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(args.outdir, "MediGuide_Internet_Chunks.csv"), "w", newline="", encoding="utf-8") as f:
        fields = list(chunks[0].keys()) if chunks else [
            "chunk_id","document_id","title","source_url","source_domain","category",
            "scope_status","chunk_index","chunk_text","content_sha256","retrieved_at",
            "answerable_by_default","review_status","hospital_scope"
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(chunks)
    with open(os.path.join(args.outdir, "MediGuide_Internet_Documents.csv"), "w", newline="", encoding="utf-8") as f:
        fields = list(pages[0].keys()) if pages else [
            "document_id","title","source_url","source_domain","source_organization",
            "category","scope_status","last_modified","retrieved_at","content_sha256","text_chars"
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(pages)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "unique_urls_attempted": len(seen_urls), "unique_documents": len(pages),
        "unique_chunks": len(chunks), "target_chunks": args.target_chunks,
        "target_met": len(chunks) >= args.target_chunks,
        "unreviewed_chunks": len(chunks),
        "warning": "A chunk count is not an approval. Review source content, licensing, clinical scope, and institution-specific details before production."
    }
    with open(os.path.join(args.outdir, "MediGuide_Corpus_Build_Report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if len(chunks) < args.target_chunks:
        print("TARGET NOT MET: add relevant, permitted sources or expand crawl coverage; do not fabricate or duplicate chunks.")

if __name__ == "__main__":
    main()
