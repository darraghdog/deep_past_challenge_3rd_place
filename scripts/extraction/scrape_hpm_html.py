#!/usr/bin/env python3
"""Scrape missing HPM Hecker HTML corpora and parse transliterations to JSONL."""

import json
import os
import re
import sys
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent.parent  # scripts/extraction/ → repo root
DATA_DIR = Path(os.environ.get("DATA_DIR", str(REPO_DIR / "data")))
PROMPTS_DIR = REPO_DIR / "prompts"
sys.path.insert(0, str(REPO_DIR / "scripts"))

BASE_URL = "https://www.hethport.uni-wuerzburg.de/altass/html"
OUTPUT_DIR = DATA_DIR / "missing_283_transliterations" / "round_4" / "hpm_hecker_web" / "html_corpora"

# 13 missing corpora: (name, url_suffix)
CORPORA = [
    ("Adana", "Liste_Adana.html"),
    ("Gol", "Liste_Gol.html"),
    ("KTS2", "Liste_Kts2.html"),
    ("kt_bk", "Liste_%26bk.html"),
    ("kt_80", "Liste_%2680.html"),
    ("kt_87", "Liste_%2687.html"),
    ("kt_88", "Liste_%2688.html"),
    ("kt_89", "Liste_%2689.html"),
    ("kt_94", "Liste_%2694.html"),
    ("kt_95", "Liste_%2695.html"),
    ("kt_96", "Liste_%2696.html"),
    ("kt_97", "Liste_%2697.html"),
    ("kt_98", "Liste_%2698.html"),
]


def fetch_url(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                print(f"  FAILED: {url} -> {e}")
                return None


def extract_tablet_links(html):
    """Extract tablet page filenames from index HTML."""
    links = re.findall(r"href='([^']+\.html)'", html)
    # Filter out navigation links
    return [l for l in links if not any(x in l.lower() for x in ["liste", "altass", "gesamt", "anfang", "hethiter"])]


def parse_tablet_html(html):
    """Parse a single tablet HTML page into structured data."""
    # Extract title
    title_match = re.search(r"<h1[^>]*><font[^>]*>([^<]+)</font></h1>", html)
    if not title_match:
        title_match = re.search(r"<h1[^>]*>([^<]+)</h1>", html)
    tablet_id = title_match.group(1).strip() if title_match else "unknown"

    # Extract metadata (everything before the transliteration table)
    metadata_lines = []
    meta_matches = re.findall(r"<font color='blue'>([^<]*)</font></b></td><td>([^<]*)", html)
    for label, value in meta_matches:
        metadata_lines.append(f"{label.strip()} {value.strip()}")

    # Split HTML at the <br/><table> boundary between metadata and transliteration.
    # The first <table> is metadata (Museums-Nr, Lit, Photo, etc.).
    # The second <table> has the actual transliteration lines.
    table_splits = re.split(r"<br/>\s*<table>", html)
    translit_html = table_splits[-1] if len(table_splits) > 1 else html

    # Remove <fn>...</fn> inline editorial comments (NB:, Anm:, etc.)
    translit_html = re.sub(r"<fn>.*?</fn>", "", translit_html, flags=re.DOTALL)

    # Extract transliteration lines from table rows
    lines = []
    # Pattern: <tr><td> line_marker </td><td> transliteration </td></tr>
    row_pattern = re.compile(r"<tr><td>([^<]*)</td><td>(.*?)</td></tr>", re.DOTALL)

    for match in row_pattern.finditer(translit_html):
        line_marker = match.group(1).strip()
        raw_content = match.group(2).strip()

        if not raw_content or raw_content == "":
            continue

        # Clean HTML tags to get plain transliteration
        text = clean_html_transliteration(raw_content)
        if text.strip():
            lines.append((line_marker, text.strip()))

    return tablet_id, metadata_lines, lines


def clean_html_transliteration(html_text):
    """Convert HTML-formatted transliteration to plain text."""
    text = html_text

    # Remove PN/PP tags (personal/place names)
    text = re.sub(r"</?PN>", "", text)
    text = re.sub(r"</?PP>", "", text)
    text = re.sub(r"</?GN>", "", text)
    text = re.sub(r"</?DN>", "", text)

    # Convert <small> (Sumerograms) - keep text as-is (already uppercase)
    text = re.sub(r"<small>(.*?)</small>", r"\1", text)

    # Convert <span class='k'> (Akkadian italic) - keep text
    text = re.sub(r"<span class='k'\s*>(.*?)</span>", r"\1", text)

    # Convert <span class='g'> (editorial marks like [ ]) - keep text
    text = re.sub(r"<span class='g'\s*>(.*?)</span>", r"\1", text)

    # Convert <span class='h'> - keep text
    text = re.sub(r"<span class='h'\s*>(.*?)</span>", r"\1", text)

    # Convert <sup> (superscript) to inline
    text = re.sub(r"<sup>(.*?)</sup>", r"\1", text)

    # Convert <sub> to inline
    text = re.sub(r"<sub>(.*?)</sub>", r"\1", text)

    # Remove <b>, <i>, <font> tags
    text = re.sub(r"</?b>", "", text)
    text = re.sub(r"</?i>", "", text)
    text = re.sub(r"<font[^>]*>", "", text)
    text = re.sub(r"</font>", "", text)

    # Remove any remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Fix = line continuations (join with hyphen)
    text = text.replace("= ", "-")

    return text


def format_transliteration(lines):
    """Format parsed lines into a single transliteration string."""
    parts = []
    for marker, text in lines:
        if marker:
            # Surface markers
            marker_clean = marker.strip()
            if marker_clean in ("Vs.", "Rs.", "u.K.", "o.K.", "l.S.", "l.Rd.", "r.Rd."):
                parts.append(f"@{marker_clean}")
            elif re.match(r"^\d+", marker_clean):
                pass  # Line number, don't add separately
        parts.append(text)
    return " ".join(parts)


def main():
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)

    total_tablets = 0
    total_with_translit = 0
    total_lines = 0

    all_records = []

    for corpus_name, list_url_suffix in CORPORA:
        print(f"\n=== {corpus_name} ===")
        list_url = f"{BASE_URL}/{list_url_suffix}"
        list_html = fetch_url(list_url)
        if not list_html:
            continue

        tablet_files = extract_tablet_links(list_html)
        print(f"  Found {len(tablet_files)} tablet pages")

        corpus_records = []
        for i, tablet_file in enumerate(tablet_files):
            tablet_url = f"{BASE_URL}/{tablet_file}"
            tablet_html = fetch_url(tablet_url)
            if not tablet_html:
                continue

            tablet_id, metadata, lines = parse_tablet_html(tablet_html)
            total_tablets += 1

            if lines:
                total_with_translit += 1
                total_lines += len(lines)
                translit = format_transliteration(lines)

                record = {
                    "id": tablet_id,
                    "corpus": corpus_name,
                    "transliteration": translit,
                    "translation": "MISSING",
                    "line_count": len(lines),
                    "source_url": tablet_url,
                }
                corpus_records.append(record)
                all_records.append(record)

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(tablet_files)} ({len(corpus_records)} with transliterations)")

            # Be polite to the server
            time.sleep(0.1)

        print(f"  Result: {len(corpus_records)}/{len(tablet_files)} tablets with transliterations, {sum(r['line_count'] for r in corpus_records)} lines")

        # Write per-corpus checkpoint
        if corpus_records:
            ckpt_path = OUTPUT_DIR / f"checkpoint_{corpus_name}.jsonl"
            with open(ckpt_path, "w") as f:
                for r in corpus_records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write combined output
    output_path = OUTPUT_DIR / "hpm_html_transliterations.jsonl"
    with open(output_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n=== SUMMARY ===")
    print(f"Total tablet pages: {total_tablets}")
    print(f"With transliterations: {total_with_translit}")
    print(f"Total lines: {total_lines}")
    print(f"Output: {output_path}")

    # Per-corpus breakdown
    from collections import Counter
    corpus_counts = Counter(r["corpus"] for r in all_records)
    corpus_lines = {}
    for r in all_records:
        corpus_lines[r["corpus"]] = corpus_lines.get(r["corpus"], 0) + r["line_count"]
    print(f"\nPer-corpus:")
    for c in sorted(corpus_counts.keys()):
        print(f"  {c:10s}: {corpus_counts[c]:4d} tablets, {corpus_lines[c]:5d} lines")


if __name__ == "__main__":
    main()
