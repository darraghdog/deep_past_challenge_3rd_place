"""
Deduplicate synthetic translations v19.

Record-level dedup by exact transliteration matching: Remove synthetic records
where any sentence pair's transliteration exactly matches (after rough_normalize)
an expert high-quality sentence pair or an AKT transliteration.

Usage:
    python scripts/preparation/dedup_synthetic_v19.py
    python scripts/preparation/dedup_synthetic_v19.py --dry-run
    python scripts/preparation/dedup_synthetic_v19.py --expert-file PATH --akt-file PATH
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent.parent  # scripts/preparation/ → repo root
DATA_DIR = Path(os.environ.get("DATA_DIR", str(REPO_DIR / "data")))
sys.path.insert(0, str(REPO_DIR / "scripts"))

DEFAULT_INPUT = str(DATA_DIR / "synthetic_translations_sentence_v16_etxra1.jsonl")
DEFAULT_OUTPUT = str(DATA_DIR / "synthetic_translations_sentence_v19_dedup.jsonl")
DEFAULT_EXPERT = str(DATA_DIR / "expert_translations_repaired_sentence_output_v19_dedup.jsonl")
DEFAULT_AKT = str(DATA_DIR / "akt_combined_v19_dedup.jsonl")


def rough_normalize(text: str) -> str:
    """Rough normalization for fuzzy matching."""
    if not text:
        return ""
    sub_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    text = text.translate(sub_map)
    text = text.replace("ₓ", "")
    text = text.replace("ḫ", "h").replace("Ḫ", "H")
    text = text.replace("<big_gap>", "<gap>")
    while "<gap> <gap>" in text:
        text = text.replace("<gap> <gap>", "<gap>")
    for ch in "[](){}⸢⸣⌈⌉˹˺":
        text = text.replace(ch, "")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def load_expert_high_quality_translits(expert_file: str) -> set:
    """Load all high-quality expert sentence transliterations (rough normalized)."""
    translits = set()
    with open(expert_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                for sp in rec.get("sentence_pairs", []):
                    if sp.get("quality", "").lower() != "high":
                        continue
                    t = rough_normalize(sp.get("transliteration", ""))
                    if t:
                        translits.add(t)
    return translits


def load_akt_translits(akt_file: str) -> set:
    """Load all AKT transliterations (rough normalized)."""
    translits = set()
    with open(akt_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                t = rough_normalize(rec.get("transliteration", ""))
                if t:
                    translits.add(t)
    return translits


def parse_args():
    parser = argparse.ArgumentParser(description="Deduplicate synthetic translations v19")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT,
                        help=f"Input synthetic JSONL (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help=f"Output deduped JSONL (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--expert-file", type=str, default=DEFAULT_EXPERT,
                        help=f"Expert deduped JSONL (default: {DEFAULT_EXPERT})")
    parser.add_argument("--akt-file", type=str, default=DEFAULT_AKT,
                        help=f"AKT deduped JSONL (default: {DEFAULT_AKT})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing output")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Synthetic V19 Deduplication")
    print("=" * 60)

    # Load synthetic records
    print(f"\nLoading {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"  Loaded {len(records)} synthetic records")

    # Build exclude transliteration sets
    exclude_expert = set()
    exclude_akt = set()

    # === Expert high-quality sentence transliterations ===
    if os.path.exists(args.expert_file):
        print(f"\nLoading expert high-quality transliterations from {args.expert_file}...")
        exclude_expert = load_expert_high_quality_translits(args.expert_file)
        print(f"  Expert high-quality transliterations: {len(exclude_expert)}")
    else:
        print(f"\n  Expert file not found: {args.expert_file}")
        print("  Skipping expert transliteration exclusion.")

    # === AKT transliterations ===
    if os.path.exists(args.akt_file):
        print(f"\nLoading AKT transliterations from {args.akt_file}...")
        exclude_akt = load_akt_translits(args.akt_file)
        print(f"  AKT transliterations: {len(exclude_akt)}")
    else:
        print(f"\n  AKT file not found: {args.akt_file}")
        print("  Skipping AKT transliteration exclusion.")

    exclude_all = exclude_expert | exclude_akt
    print(f"\n  Total exclude transliterations: {len(exclude_all)} (expert: {len(exclude_expert)}, akt: {len(exclude_akt)}, overlap: {len(exclude_expert & exclude_akt)})")

    # === Dedup: remove synthetic records with matching sentence transliterations ===
    print("\n--- Record-level exact transliteration dedup ---")

    kept = []
    removed_expert = 0
    removed_akt = 0

    for rec in records:
        # Check if any sentence pair transliteration matches
        matched_expert = False
        matched_akt = False
        for sp in rec.get("sentence_pairs", []):
            t = rough_normalize(sp.get("transliteration", ""))
            if not t:
                continue
            if t in exclude_expert:
                matched_expert = True
                break
            if t in exclude_akt:
                matched_akt = True
                break

        if matched_expert:
            removed_expert += 1
        elif matched_akt:
            removed_akt += 1
        else:
            kept.append(rec)

    # === Summary ===
    print(f"  Removed (expert translit match): {removed_expert}")
    print(f"  Removed (AKT translit match):    {removed_akt}")
    print(f"  Kept: {len(kept)} records")

    # Quality distribution
    quality_dist = defaultdict(int)
    total_pairs = 0
    for rec in kept:
        for sp in rec.get("sentence_pairs", []):
            q = sp.get("quality", "unknown").lower()
            quality_dist[q] += 1
            total_pairs += 1

    print(f"\n{'=' * 60}")
    print(f"Summary")
    print(f"{'=' * 60}")
    print(f"  Input:          {len(records)} records")
    print(f"  Expert overlap: -{removed_expert}")
    print(f"  AKT overlap:    -{removed_akt}")
    print(f"  Output:         {len(kept)} records")
    print(f"  Total pairs:    {total_pairs}")
    print(f"\n  Quality distribution:")
    for q, count in sorted(quality_dist.items()):
        print(f"    {q}: {count}")

    if args.dry_run:
        print(f"\n  [DRY RUN] No output written.")
        return

    # Write output
    print(f"\n  Writing {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Done. Wrote {len(kept)} records.")


if __name__ == "__main__":
    main()
