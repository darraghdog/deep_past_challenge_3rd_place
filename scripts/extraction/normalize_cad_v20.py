"""
Normalize and filter CAD V20 pairs for training.

Addresses issues found in blind test analysis:
- Citations in translation (0.2%) → remove
- Etymological name translations (0.3%) → remove
- 'sic' in translation (0.2%) → clean
- Short fragments <4 words (15.4%) → remove
- Apply full cleaning pipeline (clean_transliteration_chars, clean_translation_chars, postprocess_translation)

Usage:
    python scripts/normalize_cad_v20.py
    python scripts/normalize_cad_v20.py --min-words 3
    python scripts/normalize_cad_v20.py --include-medium
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent.parent  # scripts/extraction/ → repo root
DATA_DIR = Path(os.environ.get("DATA_DIR", str(REPO_DIR / "data")))
PROMPTS_DIR = REPO_DIR / "prompts"
sys.path.insert(0, str(REPO_DIR / "scripts"))

from normalization import (
    normalize_fractions, denormalize_fractions, normalize_slash_fractions,
    normalize_h_dot, normalize_gaps, normalize_determinatives,
    normalize_brackets, normalize_whitespace, remove_scribal_insertions,
    normalize_special_chars, normalize_punctuation_spacing, normalize_half_brackets,
    normalize_unmatched_brackets, normalize_line_dividers,
    normalize_ceiling_brackets, normalize_figure_dash, normalize_circumflex_to_macron,
    postprocess_translation, clean_transliteration_chars, clean_translation_chars,
)

_V15_SUBSCRIPT_TRANS = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₓ", "0123456789x")
def normalize_subscripts(text: str) -> str:
    return text.translate(_V15_SUBSCRIPT_TRANS)

# Citation pattern: publication abbreviation + volume number
CITATION_RE = re.compile(
    r'\b(BIN|CCT|KTS|KTH|KTB|ICK|AKT|TC|ATHE|OIP|VS|HUCA|JCS|AfO|RA|JAOS|PBS|YOS|AASOR|TTC|EL)\s+\d'
)

# Etymological name translations: "His-Arm-is-Long", "My-God-Take-Care"
ETYMO_RE = re.compile(
    r'(His-|My-|God-|The-god|The-God|\(the god)'
)


def _apply_common(text: str) -> str:
    text = normalize_half_brackets(text)
    text = normalize_ceiling_brackets(text)
    text = normalize_fractions(text)
    text = normalize_slash_fractions(text)
    text = normalize_subscripts(text)
    text = normalize_h_dot(text)
    text = normalize_gaps(text)
    return text


def _apply_final(text: str) -> str:
    text = normalize_brackets(text)
    text = normalize_unmatched_brackets(text)
    text = normalize_line_dividers(text)
    text = remove_scribal_insertions(text)
    text = normalize_special_chars(text)
    text = normalize_figure_dash(text)
    return text


def normalize_cad_transliteration(text: str) -> str:
    text = _apply_common(text)
    text = normalize_determinatives(text)
    text = _apply_final(text)
    text = denormalize_fractions(text)
    text = normalize_slash_fractions(text)
    text = normalize_punctuation_spacing(text)
    text = normalize_whitespace(text)
    text = clean_transliteration_chars(text)
    return text


def normalize_cad_translation(text: str) -> str:
    text = _apply_common(text)
    text = _apply_final(text)
    text = normalize_circumflex_to_macron(text)
    text = denormalize_fractions(text)
    text = normalize_slash_fractions(text)
    text = normalize_punctuation_spacing(text)
    text = normalize_whitespace(text)
    text = clean_translation_chars(text)
    text = postprocess_translation(text)
    # Clean remaining 'sic' annotations
    text = re.sub(r'\s*\{sic\}\s*', ' ', text)
    text = re.sub(r'\s*\(sic\)\s*', ' ', text)
    text = re.sub(r'\bsic\b', '', text)
    text = normalize_whitespace(text)
    return text


def is_citation(translation: str) -> bool:
    """Check if translation is/contains an academic citation."""
    return bool(CITATION_RE.search(translation))


def is_etymological(translation: str) -> bool:
    """Check if translation contains etymological name breakdown."""
    return bool(ETYMO_RE.search(translation))


def is_echo(transliteration: str, translation: str) -> bool:
    """Check if translation just echoes the transliteration."""
    # Normalize both and check similarity
    t_lower = re.sub(r'[^a-z0-9 ]', '', transliteration.lower())
    tr_lower = re.sub(r'[^a-z0-9 ]', '', translation.lower())
    if not t_lower or not tr_lower:
        return False
    # If >80% of translation chars appear in transliteration, it's an echo
    overlap = sum(1 for c in tr_lower if c in t_lower)
    return overlap > len(tr_lower) * 0.8


def filter_and_normalize_cad(
    input_path: str,
    output_path: str,
    min_words: int = 4,
    include_medium: bool = False,
    dry_run: bool = False,
) -> dict:
    """Filter and normalize CAD V20 pairs.

    Returns stats dict.
    """
    allowed = {"high"}
    if include_medium:
        allowed.add("medium")

    stats = {
        'total': 0, 'quality_filtered': 0,
        'citation': 0, 'etymological': 0, 'short': 0, 'echo': 0, 'empty': 0,
        'kept': 0,
    }
    records = []

    with open(input_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            stats['total'] += 1

            if rec.get('oa_confidence', '').lower() not in allowed:
                stats['quality_filtered'] += 1
                continue

            translit = rec.get('mt_transliteration', '').strip()
            trans = rec.get('mt_translation', '').strip()

            if not translit or not trans or translit == 'N/A' or trans == 'N/A':
                stats['empty'] += 1
                continue

            # Normalize
            translit = normalize_cad_transliteration(translit)
            trans = normalize_cad_translation(trans)

            if not translit or not trans:
                stats['empty'] += 1
                continue

            # Filter citations
            if is_citation(trans):
                stats['citation'] += 1
                continue

            # Filter etymological name translations
            if is_etymological(trans):
                stats['etymological'] += 1
                continue

            # Filter short fragments
            if len(trans.split()) < min_words:
                stats['short'] += 1
                continue

            # Filter echoed transliterations
            if is_echo(translit, trans):
                stats['echo'] += 1
                continue

            stats['kept'] += 1
            records.append({
                'transliteration': translit,
                'translation': trans,
                'headword': rec.get('headword', ''),
                'source_volume': rec.get('source_volume', ''),
                'source_ref': rec.get('source_ref', ''),
                'oa_confidence': rec.get('oa_confidence', ''),
            })

    if not dry_run:
        with open(output_path, 'w') as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    return stats


def main():
    parser = argparse.ArgumentParser(description="Normalize and filter CAD V20 pairs")
    parser.add_argument("--min-words", type=int, default=4, help="Min words in translation (default: 4)")
    parser.add_argument("--include-medium", action="store_true", help="Include medium confidence")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing output")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    input_path = args.input or str(DATA_DIR / "CAD_open_extracted_v20_pro31" / "cad_pairs_v20_pro31.jsonl")
    output_path = args.output or str(DATA_DIR / "CAD_open_extracted_v20_pro31" / "cad_pairs_v20_normalized.jsonl")

    print("=" * 60)
    print("Normalizing CAD V20 Pairs")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Min words: {args.min_words}")
    print(f"Include medium: {args.include_medium}")

    stats = filter_and_normalize_cad(
        input_path, output_path,
        min_words=args.min_words,
        include_medium=args.include_medium,
        dry_run=args.dry_run,
    )

    print(f"\nResults:")
    print(f"  Total:            {stats['total']}")
    print(f"  Quality filtered: {stats['quality_filtered']}")
    print(f"  Empty:            {stats['empty']}")
    print(f"  Citations:        {stats['citation']}")
    print(f"  Etymological:     {stats['etymological']}")
    print(f"  Short (<{args.min_words} words): {stats['short']}")
    print(f"  Echo:             {stats['echo']}")
    print(f"  Kept:             {stats['kept']}")

    if not args.dry_run:
        print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
