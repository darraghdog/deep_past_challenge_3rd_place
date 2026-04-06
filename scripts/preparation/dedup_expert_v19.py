"""
Deduplicate expert translations v19.

Oare_id dedup only (NO quality filter). Group by oare_id; for duplicates,
keep the record with more high-quality pairs. Confirm via rough_normalize()
+ SequenceMatcher (>0.96) — near-exact matches only.

Usage:
    python scripts/preparation/dedup_expert_v19.py
    python scripts/preparation/dedup_expert_v19.py --dry-run
    python scripts/preparation/dedup_expert_v19.py --input data/my_expert.jsonl
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent.parent  # scripts/preparation/ → repo root
DATA_DIR = Path(os.environ.get("DATA_DIR", str(REPO_DIR / "data")))
sys.path.insert(0, str(REPO_DIR / "scripts"))

DEFAULT_INPUT = str(DATA_DIR / "expert_translations_repaired_sentence_output_v16.jsonl")
DEFAULT_OUTPUT = str(DATA_DIR / "expert_translations_repaired_sentence_output_v19_dedup.jsonl")


def rough_normalize(text: str) -> str:
    """Rough normalization for fuzzy matching.

    - subscripts → digits, remove ₓ
    - ḫ/Ḫ → h/H
    - <big_gap> → <gap>, merge adjacent gaps
    - Remove brackets: [] () {} ⸢⸣ ⌈⌉ ˹˺
    - Collapse whitespace, lowercase
    """
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


def count_high_quality(record: dict) -> int:
    """Count high-quality sentence pairs in a record."""
    return sum(
        1 for sp in record.get("sentence_pairs", [])
        if sp.get("quality", "").lower() == "high"
    )


def quality_distribution(record: dict) -> dict:
    """Get quality distribution for a record's sentence pairs."""
    dist = defaultdict(int)
    for sp in record.get("sentence_pairs", []):
        dist[sp.get("quality", "unknown").lower()] += 1
    return dict(dist)


def are_duplicates(rec_a: dict, rec_b: dict, sim_threshold: float = 0.96) -> bool:
    """Check if two records are true duplicates via transliteration similarity."""
    t_a = rough_normalize(rec_a.get("transliteration", ""))
    t_b = rough_normalize(rec_b.get("transliteration", ""))

    if not t_a or not t_b:
        return False

    # Exact match after normalization
    if t_a == t_b:
        return True

    # Fuzzy match (near-exact only)
    ratio = SequenceMatcher(None, t_a, t_b).ratio()
    return ratio > sim_threshold


def parse_args():
    parser = argparse.ArgumentParser(description="Deduplicate expert translations v19")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT,
                        help=f"Input JSONL file (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help=f"Output JSONL file (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--sim-threshold", type=float, default=0.96,
                        help="SequenceMatcher threshold for oare_id dedup (default: 0.96)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing output")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Expert V19 Deduplication")
    print("=" * 60)

    # Load input
    print(f"\nLoading {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"  Loaded {len(records)} records")

    # === Oare_id dedup ===
    print(f"\n--- Oare_id dedup (sim_threshold={args.sim_threshold}) ---")

    # Group by oare_id
    by_oid = defaultdict(list)
    for rec in records:
        oid = rec.get("oare_id", "")
        if oid:
            by_oid[oid].append(rec)
        else:
            by_oid["__no_oid__"].append(rec)

    dup_groups = {oid: recs for oid, recs in by_oid.items()
                  if len(recs) > 1 and oid != "__no_oid__"}
    print(f"  Unique oare_ids: {len(by_oid)}")
    print(f"  Oare_ids with duplicates: {len(dup_groups)}")

    kept = []
    removed_oid = []

    for oid, recs in by_oid.items():
        if len(recs) == 1 or oid == "__no_oid__":
            kept.extend(recs)
            continue

        # Sort by high-quality count descending — keep the best
        recs_sorted = sorted(recs, key=lambda r: count_high_quality(r), reverse=True)
        best = recs_sorted[0]
        kept.append(best)

        for other in recs_sorted[1:]:
            if are_duplicates(best, other, sim_threshold=args.sim_threshold):
                removed_oid.append(other)
            else:
                # Not a true dupe despite same oare_id — keep both
                kept.append(other)

    print(f"  Removed: {len(removed_oid)} duplicate records")
    print(f"  Kept: {len(kept)} records")

    if removed_oid:
        print(f"\n  Sample removed duplicates:")
        for rec in removed_oid[:5]:
            oid = rec.get("oare_id", "?")[:12]
            src = rec.get("source", "?")
            n_high = count_high_quality(rec)
            print(f"    {oid}... | source={src} | high={n_high}")

    # === Summary ===
    print(f"\n{'=' * 60}")
    print(f"Summary")
    print(f"{'=' * 60}")
    print(f"  Input:           {len(records)} records")
    print(f"  Oid dup removed: {len(removed_oid)}")
    print(f"  Output:          {len(kept)} records")

    # Quality distribution of final output
    final_dist = defaultdict(int)
    final_high_total = 0
    for rec in kept:
        for sp in rec.get("sentence_pairs", []):
            q = sp.get("quality", "unknown").lower()
            final_dist[q] += 1
            if q == "high":
                final_high_total += 1
    print(f"\n  Final quality distribution:")
    for q, count in sorted(final_dist.items()):
        print(f"    {q}: {count}")
    print(f"  Total high-quality pairs: {final_high_total}")

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
