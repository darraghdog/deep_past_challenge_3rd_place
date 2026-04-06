"""
Prepare V23 sentence-level training data — sliding-window augmentation for all sources.

Changes from V22:
- Expert: sliding-window augmentation (--expert-sw), 14 copies with 512-byte cap
- Synthetic: sliding-window augmentation (--synthetic-sw), 8 copies with 512-byte cap
- AKT/Dergipark/Michel: sliding-window with max_bytes enforcement (prevents >512 overflow)
- CAD: 4x upsample (down from 6x)
- Segments split at low-quality pairs (expert/synthetic): never merge across quality gaps
- Output: synth_claude_v23_aug1/

Usage:
    python scripts/preparation/prepare_sentence_data_23.py --dergipark --michel --expert-sw --synthetic-sw
    python scripts/preparation/prepare_sentence_data_23.py --dergipark --michel --expert-sw --synthetic-sw --holdout
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent.parent  # scripts/preparation/ → repo root
DATA_DIR = Path(os.environ.get("DATA_DIR", str(REPO_DIR / "data")))
sys.path.insert(0, str(REPO_DIR / "scripts"))

from normalization import (
    normalize_fractions, denormalize_fractions, normalize_slash_fractions,
    normalize_h_dot, normalize_gaps, normalize_determinatives,
    normalize_brackets, normalize_whitespace, remove_scribal_insertions,
    normalize_special_chars, normalize_punctuation_spacing, normalize_half_brackets,
    normalize_unmatched_brackets, normalize_line_dividers,
    normalize_ceiling_brackets, normalize_figure_dash, normalize_circumflex_to_macron,
    postprocess_translation, post_normalize_big_gap_to_gap,
)
from akt_matching import match_akt_to_oare_ids, match_akt_by_transliteration
from consolidate_akt_v20 import _clean_akt_transliteration, _clean_akt_translation

# Override normalize_subscripts with V15-era behavior: ₓ→x (not ₓ→removed)
_V15_SUBSCRIPT_TRANS = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₓ", "0123456789x")
def normalize_subscripts(text: str) -> str:
    return text.translate(_V15_SUBSCRIPT_TRANS)


def normalize_s_dot(text: str) -> str:
    """Replace 's.' abbreviation with 'son of' in translations."""
    return re.sub(r'\bs\.\s', 'son of ', text)

# Paths
OUTPUT_DIR = str(DATA_DIR / "synth_claude_v23_aug1")

INPUT_FILE = str(DATA_DIR / "expert_translations_repaired_sentence_output_v19_dedup.jsonl")
SYNTHETIC_V19_FILE = str(DATA_DIR / "synthetic_translations_sentence_v19_dedup.jsonl")
SYNTHETIC_V22_FILE = str(DATA_DIR / "synthetic_translations_sentence_v22.jsonl")
SYNTHETIC_V24_FILE = str(DATA_DIR / "synthetic_translations_sentence_v24.jsonl")
SYNTHETIC_V26_FILE = str(DATA_DIR / "synthetic_translations_sentence_v26.jsonl")
SYNTHETIC_V27_FILE = str(DATA_DIR / "synthetic_translations_sentence_v27.jsonl")
PUBLISHED_TEXTS_CSV = str(DATA_DIR / "published_texts.csv")
HOLDOUT_FILE = str(DATA_DIR / "holdout_oare_ids.txt")
CAD_FILE = str(DATA_DIR / "CAD_open_extracted_v20_pro31" / "cad_pairs_v20_normalized.jsonl")

# V24 AKT extraction paths
AKT_V24_BASE = str(DATA_DIR / "OCR_V20")
AKT_V24_FILES = [
    ("side_by_side", "side_by_side/extracted_v24_pro31/akt_pairs_v24_pro31.jsonl"),
    ("top_bottom", "top_bottom/extracted_v24_pro31/akt_pairs_v24_pro31.jsonl"),
    ("ocr", "ocr/extracted_v24_pro31/akt_pairs_v24_pro31.jsonl"),
]

# Dergipark V24 extraction path
DERGIPARK_FILE = str(DATA_DIR / "extra_1003" / "dergipark_v1"
                     / "extracted_v24_pro31" / "akt_pairs_v24_pro31.jsonl")

# Michel V24 extraction path
MICHEL_FILE = str(DATA_DIR / "extra_1003" / "michel"
                  / "extracted_v24_pro31" / "akt_pairs_v24_pro31.jsonl")

# Hecker HPM synthetic translations
HECKER_FILE = str(DATA_DIR / "hecker_hpm_translations_v22.jsonl")
HECKER_V26_FILE = str(DATA_DIR / "hecker_translations_v26.jsonl")

# Round 2+4 additional extraction files (not in aug2)
R4_BASE = DATA_DIR / "missing_283_transliterations"
ROUND4_FILES = [
    ("round2_tr", str(R4_BASE / "round_2" / "turkish"
                      / "extracted_v24_pro31" / "akt_pairs_v24_pro31.jsonl")),
    ("round2_en", str(R4_BASE / "round_2" / "english"
                      / "extracted_v24_pro31" / "akt_pairs_v24_pro31.jsonl")),
    # ("round4_gelb", ...),  # EXCLUDED: 1935 SOV English syntax + double-labeled Sumerograms. Needs cleanup.
    ("round4_en", str(R4_BASE / "round_4" / "dergipark_en"
                      / "extracted_v24_pro31" / "akt_pairs_v24_pro31_new_only.jsonl")),
    ("round4_de", str(R4_BASE / "round_4" / "dergipark_de"
                      / "extracted_v24_pro31" / "akt_pairs_v24_pro31.jsonl")),
    ("round4_tr", str(R4_BASE / "round_4" / "dergipark_tr"
                      / "extracted_v24_pro31" / "akt_pairs_v24_pro31.jsonl")),
]
ROUND4_COPIES = 14

# Upsample multipliers
EXPERT_COPIES = 8
SYNTHETIC_COPIES = 4
AKT_COPIES = 14
CAD_COPIES = 4
DERGIPARK_COPIES = 14
MICHEL_COPIES = 14
HECKER_COPIES = 8

# Sliding-window copies when --expert-sw / --synthetic-sw
EXPERT_SW_COPIES = 14
SYNTHETIC_SW_COPIES = 6
SYNTHETIC_V24_SW_COPIES = 12

# Sliding window size weights: favor smaller merges
# Sizes 2-6 with weights (0.35, 0.25, 0.2, 0.1, 0.1)
WINDOW_SIZES = [2, 3, 4, 5, 6]
WINDOW_WEIGHTS = [0.35, 0.25, 0.20, 0.10, 0.10]

SYSTEM_PROMPT = """You are an expert translator of Old Assyrian Akkadian texts (~1950-1850 BCE).

Transliteration conventions:
- Syllables separated by hyphens (e.g., "um-ma" = "thus")
- Logograms in capitals (KU.BABBAR = silver)
- Sign readings use integers not subscripts: bi4, il5, DU10
- <gap> or <big_gap> = damaged/missing text

Translate the Akkadian transliteration to English."""


def is_repetitive(text: str, threshold: int = 4) -> bool:
    """Detect repetitive ellipsis patterns that could cause degeneration."""
    return bool(re.search(r'(\.\.\. ){' + str(threshold) + r',}', text))


def load_jsonl(path: str) -> list:
    """Load records from a JSONL file."""
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def load_optional_jsonl(path: str, label: str) -> list:
    """Load JSONL file if it exists, with logging."""
    if not os.path.exists(path):
        print(f"{label} file not found, skipping: {path}")
        return []
    print(f"Loading {path}...")
    records = load_jsonl(path)
    print(f"  Loaded {len(records)} {label.lower()} records")
    return records


def save_jsonl(records: list, path: str):
    """Save records to a JSONL file."""
    with open(path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def extract_sentence_pairs(records: list, quality_filter: str = "high",
                           include_medium: bool = False) -> list:
    """Extract sentence pairs from records, filtering by quality.

    Args:
        quality_filter: Keep pairs matching this quality level.
        include_medium: If True, also keep "medium" quality pairs.
    """
    pairs = []
    quality_counts = defaultdict(int)
    keep = {quality_filter.lower()}
    if include_medium:
        keep.add("medium")

    for record in records:
        for sp in record.get('sentence_pairs', []):
            quality = sp.get('quality', '').lower()
            quality_counts[quality] += 1

            if quality in keep:
                translit = sp.get('transliteration', '').strip()
                trans = sp.get('translation', '').strip()
                if translit and trans:
                    pairs.append((translit, trans))

    print("Quality distribution:")
    for q, count in sorted(quality_counts.items()):
        marker = " <-" if q in keep else ""
        print(f"  {q}: {count}{marker}")
    return pairs


def _sliding_window_merge(lines, rng, max_bytes=None):
    """Merge ordered line pairs with weighted random window sizes 2-6.

    Window sizes are drawn from WINDOW_SIZES with WINDOW_WEIGHTS,
    favoring smaller merges (more 2s, fewer 5-6s).

    Args:
        lines: list of (transliteration, translation) tuples, ordered by start_line
        rng: random.Random instance for reproducibility
        max_bytes: if set, reject window merges that exceed this byte count per side

    Returns:
        list of merged (transliteration, translation) tuples
    """
    merged = []
    i = 0
    while i < len(lines):
        remaining = len(lines) - i
        if remaining == 1:
            merged.append(lines[i])
            i += 1
        else:
            # Filter sizes/weights to those <= remaining
            valid = [(s, w) for s, w in zip(WINDOW_SIZES, WINDOW_WEIGHTS) if s <= remaining]

            # Filter by byte length if max_bytes set
            if max_bytes:
                byte_valid = []
                for s, w in valid:
                    window = lines[i:i + s]
                    mt = " ".join(l[0] for l in window)
                    mtr = " ".join(l[1] for l in window)
                    if len(mt.encode('utf-8')) <= max_bytes and len(mtr.encode('utf-8')) <= max_bytes:
                        byte_valid.append((s, w))
                if byte_valid:
                    valid = byte_valid
                else:
                    # No merge fits — emit single pair
                    merged.append(lines[i])
                    i += 1
                    continue

            sizes, weights = zip(*valid)
            w = rng.choices(sizes, weights=weights, k=1)[0]
            window = lines[i:i + w]
            merged_translit = " ".join(l[0] for l in window)
            merged_trans = " ".join(l[1] for l in window)
            merged.append((merged_translit, merged_trans))
            i += w
    return merged


def load_akt_v24_raw(quality_filter: str = "high"):
    """Load all V24 AKT records from JSONL files.

    Returns:
        list of raw record dicts
    """
    raw_records = []
    quality_counts = defaultdict(int)

    for mode, rel_path in AKT_V24_FILES:
        path = os.path.join(AKT_V24_BASE, rel_path)
        if not os.path.exists(path):
            print(f"  AKT V24 {mode} not found, skipping: {path}")
            continue

        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    raw_records.append(data)
                    quality_counts[data.get('match_quality', '').lower()] += 1
                    count += 1

        print(f"  AKT V24 {mode}: {count} records")

    print("AKT V24 match_quality distribution:")
    for q, count in sorted(quality_counts.items()):
        print(f"  {q}: {count}")

    return raw_records


def load_akt_v24_raw_from_file(path, label="AKT"):
    """Load V24 records from a single JSONL file with quality reporting."""
    quality_counts = defaultdict(int)
    records = []
    if not os.path.exists(path):
        print(f"  {label} file not found, skipping: {path}")
        return records

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                records.append(data)
                quality_counts[data.get('match_quality', '').lower()] += 1

    print(f"  {label}: {len(records)} records")
    print(f"  {label} match_quality distribution:")
    for q, count in sorted(quality_counts.items()):
        print(f"    {q}: {count}")
    return records


_TURKISH_CHARS = set('çğıöşüÇĞİÖŞÜ')


def load_akt_v24_tablets(raw_records, filter_turkish=False):
    """Parse raw V24 records into tablet-grouped structure.

    Filters out low-quality and MISSING lines. Groups by tablet ID,
    sorted by start_line within each tablet.

    Args:
        filter_turkish: If True, remove pairs with Turkish chars (for Dergipark/round4 sources
            where Turkish damage notes like 'Birkaç satır kırık' leak in). Do NOT enable for
            AKT sources — İ/ş in AKT are valid name chars / OCR variants, not contamination.

    Returns:
        dict mapping tablet_id -> list of (translit, trans) tuples (ordered by start_line)
    """
    tablets = defaultdict(list)
    skipped = 0

    for rec in raw_records:
        akt_id = rec.get('id', '')
        translit = rec.get('transliteration', '').strip()
        trans = (rec.get('english_translation') or rec.get('translation', '')).strip()
        match_q = rec.get('match_quality', '').lower()
        start_line = rec.get('start_line', 0)

        if not translit or not trans:
            continue

        if match_q == "low" or "MISSING" in translit or "MISSING" in trans:
            skipped += 1
            continue

        # Filter Turkish chars only for Dergipark/round4 sources
        if filter_turkish and (_TURKISH_CHARS & set(translit) or _TURKISH_CHARS & set(trans)):
            skipped += 1
            continue

        tablets[akt_id].append((translit, trans, start_line))

    # Sort lines within each tablet by start_line, then drop start_line
    for tablet_id in tablets:
        tablets[tablet_id].sort(key=lambda x: x[2])
        tablets[tablet_id] = [(t, tr) for t, tr, _ in tablets[tablet_id]]

    total_lines = sum(len(lines) for lines in tablets.values())
    print(f"  Tablets: {len(tablets)}, lines: {total_lines}, skipped (low/MISSING): {skipped}")

    return dict(tablets)


def load_expert_synthetic_tablets(records):
    """Group expert/synthetic records by oare_id into segmented tablet structure.

    Splits at low-quality pairs: sliding window merges WITHIN segments only,
    never across low-quality gaps. High+medium pairs kept.
    Returns dict[oare_id -> list of segments], each segment = list[(translit, trans)].
    """
    tablets = {}
    skipped_low = 0
    skipped_empty = 0

    for rec in records:
        oare_id = rec.get('oare_id', '')
        if not oare_id:
            continue

        segments = []
        current = []
        for sp in rec.get('sentence_pairs', []):
            quality = sp.get('quality', '').lower()
            if quality == 'low':
                skipped_low += 1
                if current:
                    segments.append(current)
                    current = []
                continue
            t = sp.get('transliteration', '').strip()
            tr = sp.get('translation', '').strip()
            if t and tr:
                current.append((t, tr))
            else:
                skipped_empty += 1
        if current:
            segments.append(current)

        if segments:
            tablets[oare_id] = segments

    total_lines = sum(len(l) for segs in tablets.values() for l in segs)
    total_segs = sum(len(segs) for segs in tablets.values())
    print(f"  Tablets: {len(tablets)}, segments: {total_segs}, lines: {total_lines}, "
          f"skipped low: {skipped_low}, skipped empty: {skipped_empty}")
    return tablets


def generate_sliding_window_copies(tablets, seed, n_copies, norm_t, norm_tr,
                                   source_tag="akt", max_bytes=None, segmented=False):
    """Generate n_copies of data with sliding-window augmentation.

    Each copy applies a different random sliding-window merge to each tablet,
    then normalizes the merged pairs.

    Args:
        tablets: dict[id -> list[(t,tr)]]           if segmented=False (AKT-style flat)
                 dict[id -> list[list[(t,tr)]]]      if segmented=True  (expert/synthetic segments)
        seed: base random seed
        n_copies: number of copies to generate
        norm_t: transliteration normalization function
        norm_tr: translation normalization function
        source_tag: source label for output tuples
        max_bytes: if set, reject window merges exceeding this byte count per side
        segmented: if True, tablets values are list of segments (list of list of pairs)

    Returns:
        list of (normalized_translit, normalized_trans, source_tag) tuples
    """
    all_pairs = []
    for copy_idx in range(n_copies):
        rng = random.Random(seed + copy_idx)
        copy_count = 0
        for tablet_id, data in tablets.items():
            segments = data if segmented else [data]
            for segment in segments:
                merged = _sliding_window_merge(segment, rng, max_bytes=max_bytes)
                for t, tr in merged:
                    nt = norm_t(t)
                    ntr = norm_tr(tr)
                    if nt and ntr:
                        all_pairs.append((nt, ntr, source_tag))
                        copy_count += 1
        if copy_idx == 0:
            print(f"    Copy 0: {copy_count} pairs")

    print(f"  Total {source_tag} pairs ({n_copies} copies): {len(all_pairs)}")
    return all_pairs


def load_cad_pairs() -> list:
    """Load CAD dictionary pairs from normalized v20 file."""
    if not os.path.exists(CAD_FILE):
        print(f"  CAD file not found, skipping: {CAD_FILE}")
        return []

    pairs = []
    with open(CAD_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                t = data.get('transliteration', '').strip()
                tr = data.get('translation', '').strip()
                if t and tr:
                    pairs.append((t, tr))

    return pairs


def deduplicate_pairs(pairs: list) -> list:
    """Remove duplicate (transliteration, translation) pairs. Preserves extra fields like source."""
    seen = set()
    unique = []
    for item in pairs:
        key = (item[0].lower(), item[1].lower())
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _apply_common_normalizations(text: str, keep_half_brackets: bool = False) -> str:
    """Apply normalizations common to both transliteration and translation."""
    if not keep_half_brackets:
        text = normalize_half_brackets(text)
    text = normalize_ceiling_brackets(text)
    text = normalize_fractions(text)
    text = normalize_slash_fractions(text)
    text = normalize_subscripts(text)
    text = normalize_h_dot(text)
    text = normalize_gaps(text)
    return text


def _apply_final_normalizations(text: str) -> str:
    """Apply final normalizations common to both transliteration and translation."""
    text = normalize_brackets(text)
    text = normalize_unmatched_brackets(text)
    text = normalize_line_dividers(text)
    text = remove_scribal_insertions(text)
    text = normalize_special_chars(text)
    text = normalize_figure_dash(text)
    return text


def normalize_transliteration(text: str, keep_half_brackets: bool = False) -> str:
    """Full normalization for transliteration with Unicode fractions. Fix-AKT always on."""
    text = _apply_common_normalizations(text, keep_half_brackets=keep_half_brackets)
    text = normalize_determinatives(text)
    text = _apply_final_normalizations(text)
    text = denormalize_fractions(text)
    text = normalize_slash_fractions(text)
    text = normalize_punctuation_spacing(text)
    text = normalize_whitespace(text)
    # Fix-AKT always on
    text = _clean_akt_transliteration(text, fix_silver=True, fix_ocr=True)
    text = normalize_whitespace(text)
    return text


def normalize_translation(text: str, keep_half_brackets: bool = False) -> str:
    """Full normalization for translation with Unicode fractions. Fix-AKT always on."""
    text = _apply_common_normalizations(text, keep_half_brackets=keep_half_brackets)
    text = _apply_final_normalizations(text)
    text = normalize_circumflex_to_macron(text)
    text = denormalize_fractions(text)
    text = normalize_slash_fractions(text)
    text = normalize_punctuation_spacing(text)
    text = normalize_whitespace(text)
    # Fix-AKT always on
    text = _clean_akt_translation(text)
    text = normalize_whitespace(text)
    return text


def format_chat_message(transliteration: str, translation: str, source: str = None) -> dict:
    """Convert a pair to OpenAI chat format with optional source tag."""
    msg = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transliteration},
            {"role": "assistant", "content": translation}
        ]
    }
    if source:
        msg["source"] = source
    return msg


def load_holdout_file(path: str) -> set:
    """Load holdout oare_ids from a text file (one per line)."""
    with open(path, 'r', encoding='utf-8') as f:
        ids = {line.strip() for line in f if line.strip()}
    print(f"  Loaded {len(ids)} holdout oare_ids from {path}")
    return ids


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare V23 sentence-level training data (sliding-window augmentation)")
    parser.add_argument("--holdout", action="store_true",
                        help="Exclude holdout oare_ids from training (clean train/val split)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (auto-generated if not specified)")
    parser.add_argument("--quality", type=str, default="high", help="Quality filter (default: high)")
    parser.add_argument("--holdout-expert", action="store_true",
                        help="Hold out expert data from training")
    parser.add_argument("--holdout-synthetic", action="store_true",
                        help="Hold out synthetic data from training")
    parser.add_argument("--holdout-akt", action="store_true",
                        help="Hold out AKT data from training")
    parser.add_argument("--holdout-cad", action="store_true",
                        help="Hold out CAD dictionary data from training")
    parser.add_argument("--keep-half-brackets", action="store_true",
                        help="Keep half brackets instead of removing them")
    parser.add_argument("--postprocess", action="store_true",
                        help="Apply postprocess_translation to targets")
    parser.add_argument("--no-dergipark", action="store_true",
                        help="Exclude Dergipark V24 data")
    parser.add_argument("--no-michel", action="store_true",
                        help="Exclude Michel V24 data")
    parser.add_argument("--no-expert-sw", action="store_true",
                        help="Use simple upsample for expert (instead of sliding-window)")
    parser.add_argument("--no-synthetic-sw", action="store_true",
                        help="Use simple upsample for synthetic (instead of sliding-window)")
    parser.add_argument("--synthetic-v24", action="store_true",
                        help="Use V24 synthetic data (12x SW) instead of V19+V22")
    parser.add_argument("--synthetic-v26", action="store_true",
                        help="Use V26 semantic few-shot synthetic data (12x SW) instead of V19+V22")
    parser.add_argument("--synthetic-v27", action="store_true",
                        help="Use V27 semantic few-shot synthetic data (384-dim, sample 12/24, 12x SW) instead of V19+V22")
    parser.add_argument("--round4", action="store_true",
                        help="Include round 2+4 additional extractions (Gelb, Dergipark EN/DE/TR)")
    parser.add_argument("--hecker", action="store_true",
                        help="Include Hecker HPM synthetic translations (high+medium quality)")
    parser.add_argument("--hecker-v26", action="store_true",
                        help="Use Hecker V26 (semantic few-shot) instead of V22")
    parser.add_argument("--hecker-file", type=str, default=HECKER_FILE,
                        help=f"Hecker translations JSONL (default: {os.path.basename(HECKER_FILE)})")
    parser.add_argument("--cad-copies", type=int, default=None,
                        help=f"Override CAD upsample copies (default: {CAD_COPIES})")
    parser.add_argument("--max-merge-bytes", type=int, default=512,
                        help="Max bytes per side for sliding-window merged pairs (default: 512)")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # Derive positive flags from --no-* flags (defaults are on in v23)
    args.dergipark = not args.no_dergipark
    args.michel = not args.no_michel
    args.expert_sw = not args.no_expert_sw
    args.synthetic_sw = not args.no_synthetic_sw

    # Override multipliers if specified
    global CAD_COPIES, HECKER_COPIES
    if args.cad_copies is not None:
        CAD_COPIES = args.cad_copies

    # Build source-holdout list for output directory naming
    source_holdouts = []
    if args.holdout_expert:
        source_holdouts.append("expert")
    if args.holdout_synthetic:
        source_holdouts.append("synthetic")
    if args.holdout_akt:
        source_holdouts.append("akt")
    if args.holdout_cad:
        source_holdouts.append("cad")

    # Auto-generate output dir
    if args.output_dir:
        output_dir = args.output_dir
    else:
        base = str(DATA_DIR / "synth_claude_pp_v23_aug1") if args.postprocess else OUTPUT_DIR
        suffix = ""
        if source_holdouts == ["cad"]:
            suffix += "_nocad"
        elif source_holdouts:
            suffix += "_" + "_".join(source_holdouts)
        if not args.dergipark:
            suffix += "_nodg"
        if not args.michel:
            suffix += "_nomi"
        if not args.expert_sw:
            suffix += "_noesw"
        if not args.synthetic_sw:
            suffix += "_nossw"
        if args.synthetic_v24:
            suffix += "_sv24"
        if args.synthetic_v26:
            suffix += "_sv26"
        if args.synthetic_v27:
            suffix += "_sv27"
        if args.round4:
            suffix += "_r4"
        if args.hecker and args.hecker_v26:
            suffix += "_hk2"
        elif args.hecker:
            suffix += "_hk"
        if args.holdout:
            suffix += "_holdout"
        output_dir = base + suffix

    print("=" * 60)
    print("Preparing V23 Sentence-Level Training Data (Augmented)")
    print("=" * 60)
    print(f"Quality: {args.quality} | Seed: {args.seed}")
    if args.expert_sw:
        print(f"Expert: {EXPERT_SW_COPIES}x sliding-window (max_bytes={args.max_merge_bytes})")
    else:
        print(f"Expert: {EXPERT_COPIES}x upsample")
    if args.synthetic_v27:
        print(f"Synthetic: V27 (semantic few-shot, 384-dim, sample 12/24), {SYNTHETIC_V24_SW_COPIES}x sliding-window (max_bytes={args.max_merge_bytes})")
    elif args.synthetic_v26:
        print(f"Synthetic: V26 (semantic few-shot), {SYNTHETIC_V24_SW_COPIES}x sliding-window (max_bytes={args.max_merge_bytes})")
    elif args.synthetic_v24:
        print(f"Synthetic: V24, {SYNTHETIC_V24_SW_COPIES}x sliding-window (max_bytes={args.max_merge_bytes})")
    elif args.synthetic_sw:
        print(f"Synthetic: V19 + V22, {SYNTHETIC_SW_COPIES}x sliding-window (max_bytes={args.max_merge_bytes})")
    else:
        print(f"Synthetic: V19 + V22 combined, {SYNTHETIC_COPIES}x upsample")
    print(f"AKT source: V24 (semantic chunking, 576 DPI, Gemini Pro 3.1)")
    print(f"AKT: {AKT_COPIES}x sliding-window copies (no dedup)")
    print(f"CAD: {CAD_COPIES}x upsample")
    if args.dergipark:
        print(f"Dergipark: {DERGIPARK_COPIES}x sliding-window copies")
    if args.michel:
        print(f"Michel: {MICHEL_COPIES}x sliding-window copies")
    if args.round4:
        print(f"Round 2+4: {ROUND4_COPIES}x sliding-window copies")
    if args.hecker and args.hecker_v26:
        print(f"Hecker: V26 (semantic few-shot), {HECKER_COPIES}x upsample (high+medium quality)")
    elif args.hecker:
        print(f"Hecker: V22, {HECKER_COPIES}x upsample (high+medium quality)")
    print(f"Fix-AKT: ALWAYS ON (fix_silver + fix_ocr)")
    if args.keep_half_brackets:
        print("Half brackets: KEEP")
    if args.postprocess:
        print("Postprocess: ENABLED")
    if args.holdout:
        print("Holdout: ENABLED")
    if source_holdouts:
        print(f"Source holdout: {', '.join(source_holdouts)}")
    print(f"Output: {output_dir}\n")

    # Load holdout oare_ids (always needed for val)
    print("Loading holdout oare_ids...")
    holdout_ids = load_holdout_file(HOLDOUT_FILE)

    # Load expert data
    if args.holdout_expert:
        expert_records = []
        print("Skipping expert records (source holdout)")
    else:
        expert_records = load_optional_jsonl(INPUT_FILE, "Expert")

    # Load synthetic data
    if args.holdout_synthetic:
        synthetic_v19_records = []
        synthetic_v22_records = []
        synthetic_v24_records = []
        print("Skipping synthetic records (source holdout)")
    elif args.synthetic_v27:
        synthetic_v19_records = []
        synthetic_v22_records = []
        synthetic_v24_records = load_optional_jsonl(SYNTHETIC_V27_FILE, "Synthetic V27")
    elif args.synthetic_v26:
        synthetic_v19_records = []
        synthetic_v22_records = []
        synthetic_v24_records = load_optional_jsonl(SYNTHETIC_V26_FILE, "Synthetic V26")
    elif args.synthetic_v24:
        synthetic_v19_records = []
        synthetic_v22_records = []
        synthetic_v24_records = load_optional_jsonl(SYNTHETIC_V24_FILE, "Synthetic V24")
    else:
        synthetic_v19_records = load_optional_jsonl(SYNTHETIC_V19_FILE, "Synthetic V19")
        synthetic_v22_records = load_optional_jsonl(SYNTHETIC_V22_FILE, "Synthetic V22")
        synthetic_v24_records = []

    # Load AKT V24 records
    if args.holdout_akt:
        akt_raw_records = []
        print("\nSkipping AKT pairs (source holdout)")
    else:
        print("\nLoading AKT V24 records...")
        akt_raw_records = load_akt_v24_raw(quality_filter=args.quality)
        print(f"  Total raw records: {len(akt_raw_records)}")

    # Load CAD pairs (normalized v20 — no quality filter needed)
    if args.holdout_cad:
        cad_pairs = []
        print("\nSkipping CAD pairs (source holdout)")
    else:
        print("\nLoading CAD pairs...")
        cad_pairs = load_cad_pairs()
        print(f"  Total CAD pairs: {len(cad_pairs)}")

    # Load Dergipark V24 records
    if args.dergipark:
        print("\nLoading Dergipark V24 records...")
        dergipark_raw = load_akt_v24_raw_from_file(DERGIPARK_FILE, "Dergipark")
        print(f"  Total raw records: {len(dergipark_raw)}")
    else:
        dergipark_raw = []

    # Load Michel V24 records
    if args.michel:
        print("\nLoading Michel V24 records...")
        michel_raw = load_akt_v24_raw_from_file(MICHEL_FILE, "Michel")
        print(f"  Total raw records: {len(michel_raw)}")
    else:
        michel_raw = []

    # Load round 2+4 additional extractions
    if args.round4:
        print("\nLoading round 2+4 additional extractions...")
        round4_raw = []
        for label, path in ROUND4_FILES:
            recs = load_akt_v24_raw_from_file(path, label)
            round4_raw.extend(recs)
        print(f"  Total round 2+4 records: {len(round4_raw)}")
    else:
        round4_raw = []

    # Load Hecker HPM synthetic translations
    if args.hecker:
        hecker_file = HECKER_V26_FILE if args.hecker_v26 else args.hecker_file
        hecker_label = "Hecker V26" if args.hecker_v26 else "Hecker"
        print(f"\nLoading {hecker_label} translations...")
        hecker_records = load_optional_jsonl(hecker_file, hecker_label)
    else:
        hecker_records = []

    # Match AKT → oare_ids (needed for holdout exclusion)
    if akt_raw_records:
        print("\nMatching AKT records to oare_ids...")
        akt_to_oare = match_akt_to_oare_ids(akt_raw_records, PUBLISHED_TEXTS_CSV)
        # Text-based fallback for unmatched groups
        all_expert_for_matching = load_jsonl(INPUT_FILE) if os.path.exists(INPUT_FILE) else []
        if all_expert_for_matching:
            text_matches = match_akt_by_transliteration(
                akt_raw_records, all_expert_for_matching, akt_to_oare)
            akt_to_oare.update(text_matches)
    else:
        akt_to_oare = {}

    # Apply holdout exclusion from training
    if args.holdout:
        print("\nApplying holdout exclusion...")
        pre = len(expert_records)
        expert_records = [r for r in expert_records if r.get('oare_id', '') not in holdout_ids]
        print(f"  Expert: {pre} -> {len(expert_records)} (excluded {pre - len(expert_records)})")

        pre = len(synthetic_v19_records)
        synthetic_v19_records = [r for r in synthetic_v19_records if r.get('oare_id', '') not in holdout_ids]
        print(f"  Synthetic V19: {pre} -> {len(synthetic_v19_records)} (excluded {pre - len(synthetic_v19_records)})")

        pre = len(synthetic_v22_records)
        synthetic_v22_records = [r for r in synthetic_v22_records if r.get('oare_id', '') not in holdout_ids]
        print(f"  Synthetic V22: {pre} -> {len(synthetic_v22_records)} (excluded {pre - len(synthetic_v22_records)})")

        if akt_raw_records:
            pre = len(akt_raw_records)
            akt_raw_records = [
                r for r in akt_raw_records
                if akt_to_oare.get(r.get('id', ''), '') not in holdout_ids
            ]
            print(f"  AKT raw: {pre} -> {len(akt_raw_records)} (excluded {pre - len(akt_raw_records)})")

    # Extract expert pairs
    print(f"\nExtracting expert sentence pairs (quality={args.quality})...")
    expert_pairs = extract_sentence_pairs(expert_records, quality_filter=args.quality)
    print(f"  Extracted {len(expert_pairs)} expert pairs")

    # Extract synthetic pairs from each file independently, then combine
    print(f"\nExtracting synthetic V19 sentence pairs (quality={args.quality})...")
    synthetic_v19_pairs = extract_sentence_pairs(synthetic_v19_records, quality_filter=args.quality)
    print(f"  Extracted {len(synthetic_v19_pairs)} V19 pairs")

    print(f"\nExtracting synthetic V22 sentence pairs (quality={args.quality})...")
    synthetic_v22_pairs = extract_sentence_pairs(synthetic_v22_records, quality_filter=args.quality)
    print(f"  Extracted {len(synthetic_v22_pairs)} V22 pairs")

    if synthetic_v24_records:
        print(f"\nExtracting synthetic V24 sentence pairs (quality={args.quality})...")
        synthetic_v24_pairs = extract_sentence_pairs(synthetic_v24_records, quality_filter=args.quality)
        print(f"  Extracted {len(synthetic_v24_pairs)} V24 pairs")
    else:
        synthetic_v24_pairs = []

    synthetic_pairs = synthetic_v19_pairs + synthetic_v22_pairs + synthetic_v24_pairs
    print(f"  Combined synthetic pairs: {len(synthetic_pairs)}")

    # Extract Hecker pairs (high + medium quality)
    if hecker_records:
        print(f"\nExtracting Hecker sentence pairs (high+medium)...")
        hecker_pairs = extract_sentence_pairs(hecker_records, quality_filter=args.quality,
                                              include_medium=True)
        print(f"  Extracted {len(hecker_pairs)} Hecker pairs")
    else:
        hecker_pairs = []

    # Parse AKT into tablet structure (filter low/MISSING, group by tablet)
    if akt_raw_records:
        print(f"\nBuilding AKT tablet structure...")
        akt_tablets = load_akt_v24_tablets(akt_raw_records)
    else:
        akt_tablets = {}

    # === Non-AKT pipeline: tag, dedup, normalize, upsample ===
    # When --expert-sw or --synthetic-sw, those sources skip this pipeline and use sliding-window instead
    print("\n--- Non-AKT sources: dedup + normalize ---")
    tagged_pairs = []
    if not args.expert_sw:
        tagged_pairs += [(t, tr, "expert") for t, tr in expert_pairs]
    if not args.synthetic_sw and not args.synthetic_v24 and not args.synthetic_v26 and not args.synthetic_v27:
        tagged_pairs += [(t, tr, "synthetic") for t, tr in synthetic_pairs]
    tagged_pairs += [(t, tr, "cad") for t, tr in cad_pairs]
    if hecker_pairs:
        tagged_pairs += [(t, tr, "hecker") for t, tr in hecker_pairs]

    total_non_akt = len(tagged_pairs)
    print(f"Total non-AKT pairs (simple upsample): {total_non_akt}")

    print("Deduplicating non-AKT pairs...")
    unique_pairs = deduplicate_pairs(tagged_pairs)
    dupe_count = total_non_akt - len(unique_pairs)
    print(f"  Removed {dupe_count} duplicates, {len(unique_pairs)} unique")

    print("Filtering repetitive patterns...")
    pre_filter = len(unique_pairs)
    unique_pairs = [(t, tr, s) for t, tr, s in unique_pairs if not is_repetitive(tr)]
    rep_count = pre_filter - len(unique_pairs)
    print(f"  Removed {rep_count} repetitive, {len(unique_pairs)} final")

    print("Applying normalization to non-AKT pairs...")
    unique_pairs = [
        (post_normalize_big_gap_to_gap(normalize_transliteration(t, keep_half_brackets=args.keep_half_brackets)),
         normalize_s_dot(post_normalize_big_gap_to_gap(normalize_translation(tr, keep_half_brackets=args.keep_half_brackets))),
         s)
        for t, tr, s in unique_pairs
    ]

    if args.postprocess:
        print("Applying postprocess_translation to targets...")
        unique_pairs = [(t, postprocess_translation(tr), s) for t, tr, s in unique_pairs]

    # Upsample non-AKT sources (only those not using sliding-window)
    print(f"\nUpsampling non-AKT sources...")
    upsampled_pairs = []
    source_base_counts = defaultdict(int)
    for _, _, s in unique_pairs:
        source_base_counts[s] += 1

    for t, tr, s in unique_pairs:
        if s == "expert":
            for _ in range(EXPERT_COPIES):
                upsampled_pairs.append((t, tr, s))
        elif s == "synthetic":
            for _ in range(SYNTHETIC_COPIES):
                upsampled_pairs.append((t, tr, s))
        elif s == "cad":
            for _ in range(CAD_COPIES):
                upsampled_pairs.append((t, tr, s))
        elif s == "hecker":
            for _ in range(HECKER_COPIES):
                upsampled_pairs.append((t, tr, s))

    if not args.expert_sw:
        print(f"  Expert: {source_base_counts['expert']} x {EXPERT_COPIES} = {source_base_counts['expert'] * EXPERT_COPIES}")
    if not args.synthetic_sw:
        print(f"  Synthetic: {source_base_counts['synthetic']} x {SYNTHETIC_COPIES} = {source_base_counts['synthetic'] * SYNTHETIC_COPIES}")
    print(f"  CAD: {source_base_counts['cad']} x {CAD_COPIES} = {source_base_counts['cad'] * CAD_COPIES}")
    if hecker_pairs:
        print(f"  Hecker: {source_base_counts['hecker']} x {HECKER_COPIES} = {source_base_counts['hecker'] * HECKER_COPIES}")
    print(f"  Non-AKT total (simple upsample): {len(upsampled_pairs)}")

    # === Sliding-window normalization functions ===
    norm_t = lambda t: post_normalize_big_gap_to_gap(normalize_transliteration(t, keep_half_brackets=args.keep_half_brackets))
    _norm_tr = lambda t: normalize_s_dot(post_normalize_big_gap_to_gap(normalize_translation(t, keep_half_brackets=args.keep_half_brackets)))
    pp_norm_tr = (lambda t: postprocess_translation(_norm_tr(t))) if args.postprocess else _norm_tr
    sw_max_bytes = args.max_merge_bytes

    # === Expert sliding-window pipeline ===
    if args.expert_sw and expert_records:
        print(f"\n--- Expert: {EXPERT_SW_COPIES}x sliding-window augmentation (max_bytes={sw_max_bytes}) ---")
        expert_tablets = load_expert_synthetic_tablets(expert_records)
        expert_sw_augmented = generate_sliding_window_copies(
            expert_tablets, seed=args.seed + 30000, n_copies=EXPERT_SW_COPIES,
            norm_t=norm_t, norm_tr=pp_norm_tr,
            source_tag="expert", max_bytes=sw_max_bytes, segmented=True)
    else:
        expert_sw_augmented = []

    # === Synthetic sliding-window pipeline ===
    if (args.synthetic_v24 or args.synthetic_v26 or args.synthetic_v27) and synthetic_v24_records:
        synth_label = "V27" if args.synthetic_v27 else ("V26" if args.synthetic_v26 else "V24")
        print(f"\n--- Synthetic {synth_label}: {SYNTHETIC_V24_SW_COPIES}x sliding-window augmentation (max_bytes={sw_max_bytes}) ---")
        synth_v24_tablets = load_expert_synthetic_tablets(synthetic_v24_records)
        synth_sw_augmented = generate_sliding_window_copies(
            synth_v24_tablets, seed=args.seed + 60000, n_copies=SYNTHETIC_V24_SW_COPIES,
            norm_t=norm_t, norm_tr=pp_norm_tr,
            source_tag="synthetic", max_bytes=sw_max_bytes, segmented=True)
        print(f"  Total synthetic V24 sliding-window pairs: {len(synth_sw_augmented)}")
    elif args.synthetic_sw and (synthetic_v19_records or synthetic_v22_records):
        print(f"\n--- Synthetic: {SYNTHETIC_SW_COPIES}x sliding-window augmentation (max_bytes={sw_max_bytes}) ---")
        synth_sw_augmented = []
        if synthetic_v19_records:
            print("  Synthetic V19:")
            synth_v19_tablets = load_expert_synthetic_tablets(synthetic_v19_records)
            synth_sw_augmented += generate_sliding_window_copies(
                synth_v19_tablets, seed=args.seed + 40000, n_copies=SYNTHETIC_SW_COPIES,
                norm_t=norm_t, norm_tr=pp_norm_tr,
                source_tag="synthetic", max_bytes=sw_max_bytes, segmented=True)
        if synthetic_v22_records:
            print("  Synthetic V22:")
            synth_v22_tablets = load_expert_synthetic_tablets(synthetic_v22_records)
            synth_sw_augmented += generate_sliding_window_copies(
                synth_v22_tablets, seed=args.seed + 50000, n_copies=SYNTHETIC_SW_COPIES,
                norm_t=norm_t, norm_tr=pp_norm_tr,
                source_tag="synthetic", max_bytes=sw_max_bytes, segmented=True)
        print(f"  Total synthetic sliding-window pairs: {len(synth_sw_augmented)}")
    else:
        synth_sw_augmented = []

    # === AKT pipeline: sliding-window augmentation (no dedup) ===
    print(f"\n--- AKT: {AKT_COPIES}x sliding-window augmentation ---")

    if akt_tablets:
        akt_augmented = generate_sliding_window_copies(
            akt_tablets, seed=args.seed, n_copies=AKT_COPIES,
            norm_t=norm_t, norm_tr=pp_norm_tr,
            source_tag="akt", max_bytes=sw_max_bytes)
    else:
        akt_augmented = []

    # === Dergipark pipeline: same sliding-window augmentation as AKT ===
    if dergipark_raw:
        print(f"\n--- Dergipark: {DERGIPARK_COPIES}x sliding-window augmentation ---")
        dergipark_tablets = load_akt_v24_tablets(dergipark_raw, filter_turkish=True)
        dergipark_augmented = generate_sliding_window_copies(
            dergipark_tablets, seed=args.seed + 10000, n_copies=DERGIPARK_COPIES,
            norm_t=norm_t, norm_tr=pp_norm_tr,
            source_tag="dergipark", max_bytes=sw_max_bytes)
    else:
        dergipark_augmented = []

    # === Michel pipeline: same sliding-window augmentation as AKT ===
    if michel_raw:
        print(f"\n--- Michel: {MICHEL_COPIES}x sliding-window augmentation ---")
        michel_tablets = load_akt_v24_tablets(michel_raw, filter_turkish=True)
        michel_augmented = generate_sliding_window_copies(
            michel_tablets, seed=args.seed + 20000, n_copies=MICHEL_COPIES,
            norm_t=norm_t, norm_tr=pp_norm_tr,
            source_tag="michel", max_bytes=sw_max_bytes)
    else:
        michel_augmented = []

    # === Round 2+4 pipeline: same sliding-window augmentation ===
    if round4_raw:
        print(f"\n--- Round 2+4: {ROUND4_COPIES}x sliding-window augmentation ---")
        round4_tablets = load_akt_v24_tablets(round4_raw, filter_turkish=True)
        round4_augmented = generate_sliding_window_copies(
            round4_tablets, seed=args.seed + 30000, n_copies=ROUND4_COPIES,
            norm_t=norm_t, norm_tr=pp_norm_tr,
            source_tag="round4", max_bytes=sw_max_bytes)
    else:
        round4_augmented = []

    # Combine all
    all_pairs = (upsampled_pairs + expert_sw_augmented + synth_sw_augmented
                 + akt_augmented + dergipark_augmented + michel_augmented
                 + round4_augmented)
    print(f"\nTotal training pairs: {len(all_pairs)}")

    # Convert to chat format with source tags
    print("Converting to chat format...")
    train_records = [format_chat_message(t, tr, source=s) for t, tr, s in all_pairs]
    random.shuffle(train_records)
    print(f"  Training records: {len(train_records)}")

    # Source distribution
    source_counts = defaultdict(int)
    for r in train_records:
        source_counts[r.get("source", "unknown")] += 1
    print("  Source distribution:")
    for src, count in sorted(source_counts.items()):
        print(f"    {src}: {count}")

    # Build validation from holdout expert records
    print(f"\nBuilding validation from holdout oare_ids...")
    all_expert = load_optional_jsonl(INPUT_FILE, "Expert (for val)")
    val_experts = [r for r in all_expert if r.get('oare_id', '') in holdout_ids]
    print(f"  Found {len(val_experts)} expert records with holdout oare_ids")

    val_norm_t = norm_t
    val_norm_tr = pp_norm_tr

    val_records = []
    for r in val_experts:
        translit = r.get('transliteration', '').strip()
        trans = r.get('corrected_translation', '').strip()
        if not trans:
            trans = r.get('translation', '').strip()
        if translit and trans:
            val_records.append({
                **format_chat_message(val_norm_t(translit), val_norm_tr(trans)),
                "kaggle_translation": val_norm_tr(trans),
            })
    print(f"  Validation records: {len(val_records)}")

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    suffix = "_hb" if args.keep_half_brackets else ""
    train_path = os.path.join(output_dir, f"akkadian_train{suffix}.jsonl")
    val_path = os.path.join(output_dir, f"akkadian_val{suffix}.jsonl")
    save_jsonl(train_records, train_path)
    save_jsonl(val_records, val_path)

    akt_base_lines = sum(len(lines) for lines in akt_tablets.values())
    stats = {
        "version": "v23_aug1",
        "akt_source": "V24 (semantic chunking, 576 DPI, Gemini Pro 3.1)",
        "akt_augmentation": f"{AKT_COPIES}x sliding-window (no dedup)",
        "akt_tablets": len(akt_tablets),
        "akt_base_lines": akt_base_lines,
        "synthetic_source": "V27 (semantic few-shot, 384-dim, sample 12/24)" if args.synthetic_v27 else ("V26 (semantic few-shot)" if args.synthetic_v26 else ("V24" if args.synthetic_v24 else "V19 + V22 combined")),
        "expert_sw": args.expert_sw,
        "expert_augmentation": f"{EXPERT_SW_COPIES}x sliding-window" if args.expert_sw else f"{EXPERT_COPIES}x upsample",
        "synthetic_v24": args.synthetic_v24,
        "synthetic_v26": args.synthetic_v26,
        "synthetic_v27": args.synthetic_v27,
        "synthetic_sw": args.synthetic_sw or args.synthetic_v24 or args.synthetic_v26 or args.synthetic_v27,
        "synthetic_augmentation": f"{SYNTHETIC_V24_SW_COPIES}x sliding-window (V27)" if args.synthetic_v27 else (f"{SYNTHETIC_V24_SW_COPIES}x sliding-window (V26)" if args.synthetic_v26 else (f"{SYNTHETIC_V24_SW_COPIES}x sliding-window (V24)" if args.synthetic_v24 else (f"{SYNTHETIC_SW_COPIES}x sliding-window" if args.synthetic_sw else f"{SYNTHETIC_COPIES}x upsample"))),
        "max_merge_bytes": args.max_merge_bytes,
        "cad_upsample": f"{CAD_COPIES}x",
        "cad_source": "cad_pairs_v20_normalized.jsonl",
        "fix_silver": True,
        "fix_ocr": True,
        "holdout": args.holdout,
        "source_holdouts": source_holdouts,
        "holdout_expert": args.holdout_expert,
        "holdout_synthetic": args.holdout_synthetic,
        "holdout_akt": args.holdout_akt,
        "holdout_cad": args.holdout_cad,
        "keep_half_brackets": args.keep_half_brackets,
        "postprocess": args.postprocess,
        "expert_records": len(expert_records),
        "synthetic_v19_records": len(synthetic_v19_records),
        "synthetic_v22_records": len(synthetic_v22_records),
        "synthetic_v24_records": len(synthetic_v24_records),
        "expert_pairs_base": len(expert_pairs),
        "synthetic_v19_pairs": len(synthetic_v19_pairs),
        "synthetic_v22_pairs": len(synthetic_v22_pairs),
        "synthetic_v24_pairs": len(synthetic_v24_pairs),
        "synthetic_pairs_combined": len(synthetic_pairs),
        "akt_augmented_pairs": len(akt_augmented),
        "akt_raw_records": len(akt_raw_records),
        "dergipark": args.dergipark,
        "dergipark_raw_records": len(dergipark_raw),
        "dergipark_augmented_pairs": len(dergipark_augmented),
        "michel": args.michel,
        "michel_raw_records": len(michel_raw),
        "michel_augmented_pairs": len(michel_augmented),
        "hecker": args.hecker,
        "hecker_v26": args.hecker_v26,
        "hecker_records": len(hecker_records),
        "hecker_pairs_base": len(hecker_pairs),
        "hecker_upsample": f"{HECKER_COPIES}x",
        "expert_sw_augmented_pairs": len(expert_sw_augmented),
        "synthetic_sw_augmented_pairs": len(synth_sw_augmented),
        "cad_pairs": len(cad_pairs),
        "non_akt_before_dedup": total_non_akt,
        "non_akt_duplicates_removed": dupe_count,
        "non_akt_repetitive_removed": rep_count,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "source_counts": dict(source_counts),
        "seed": args.seed,
    }
    stats_path = os.path.join(output_dir, f"stats{suffix}.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved: {train_path}, {val_path}, {stats_path}")

    # Summary
    print("\n" + "=" * 60)
    print(f"Summary: {len(train_records)} train, {len(val_records)} val")
    if train_records:
        lengths = [len(r['messages'][2]['content']) for r in train_records]
        print(f"Response lengths: {min(lengths)}-{max(lengths)} chars (mean: {sum(lengths)//len(lengths)})")


if __name__ == "__main__":
    main()
