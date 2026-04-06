"""
Cross-reference Hecker HPM extracted transliterations against our data sources
to identify which tablets already have translations.

Matching strategy:
1. Publication ID match: Hecker ID (e.g. "BIN 4, 1") → normalize → match aliases in published_texts.csv
2. Museum number match: Hecker museum_number → match label field in published_texts.csv
3. Excavation number match: Hecker excavation_number (kt number) → match excavation_no in published_texts.csv
4. Transliteration fuzzy match: token inverted index pre-filter + rapidfuzz ratio (fast)

Usage:
    python scripts/crossref_hecker.py --input data/.../akt_pairs_v24_pro31.jsonl
    python scripts/crossref_hecker.py --input ... --no-fuzzy
    python scripts/crossref_hecker.py --input ... --fuzzy-threshold 80
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from rapidfuzz import fuzz

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent.parent  # scripts/extraction/ → repo root
DATA_DIR = Path(os.environ.get("DATA_DIR", str(REPO_DIR / "data")))
PROMPTS_DIR = REPO_DIR / "prompts"
sys.path.insert(0, str(REPO_DIR / "scripts"))

COMP_DIR = DATA_DIR / "deep-past-initiative-machine-translation_v2"


def normalize_hecker_id(hecker_id: str) -> str:
    """Normalize Hecker ID to match published_texts aliases format."""
    norm = re.sub(r'_part\d+$', '', hecker_id)
    norm = norm.replace(',', '')
    norm = re.sub(r'\s+', ' ', norm).strip()
    return norm


def tokenize(text: str) -> list:
    """Split transliteration into tokens for inverted index."""
    return [t for t in re.split(r'[\s\-=/:]+', text.lower()) if len(t) > 1]


class TokenIndex:
    """Inverted index for fast fuzzy candidate retrieval."""

    def __init__(self):
        self.token_to_ids = defaultdict(set)  # token → set of oare_ids
        self.id_to_text = {}  # oare_id → full transliteration

    def add(self, oare_id: str, text: str):
        self.id_to_text[oare_id] = text
        for token in tokenize(text):
            self.token_to_ids[token].add(oare_id)

    def query(self, text: str, top_k: int = 20) -> list:
        """Find top_k candidate oare_ids by token overlap count."""
        tokens = tokenize(text)
        if not tokens:
            return []
        candidate_counts = Counter()
        for token in tokens:
            for oare_id in self.token_to_ids.get(token, ()):
                candidate_counts[oare_id] += 1
        # Return top_k by overlap count
        return [oid for oid, _ in candidate_counts.most_common(top_k)]

    def fuzzy_match(self, text: str, threshold: float = 75.0, top_k: int = 20):
        """Find best fuzzy match using token pre-filter + fuzz.ratio.

        Returns (oare_id, score) or (None, 0).
        """
        candidates = self.query(text, top_k=top_k)
        if not candidates:
            return None, 0

        best_id = None
        best_score = 0
        for oare_id in candidates:
            ref = self.id_to_text[oare_id]
            # Use ratio (not partial_ratio) — much faster, good enough with pre-filter
            score = fuzz.ratio(text, ref)
            if score > best_score:
                best_score = score
                best_id = oare_id

        if best_score >= threshold:
            return best_id, best_score
        return None, 0


def load_published_texts():
    """Load published_texts.csv and build lookup indices."""
    path = COMP_DIR / "published_texts.csv"
    aliases_to_oare = {}
    museum_to_oare = {}
    excav_to_oare = {}
    oare_to_translit = {}

    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            oare_id = row['oare_id']
            aliases = row.get('aliases', '').strip()
            label = row.get('label', '')
            excav = row.get('excavation_no', '').strip()
            translit = row.get('transliteration', '')

            if aliases:
                aliases_to_oare[aliases] = oare_id
            if excav:
                excav_to_oare[excav] = oare_id

            m = re.search(r'\(([^)]+)\)', label)
            if m:
                museum_to_oare[m.group(1).strip()] = oare_id

            if translit:
                oare_to_translit[oare_id] = translit

    return aliases_to_oare, museum_to_oare, excav_to_oare, oare_to_translit


def load_train_translations():
    path = COMP_DIR / "train.csv"
    oare_to_trans = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            oare_to_trans[row['oare_id']] = row['translation']
    return oare_to_trans


def load_sentences_oare():
    path = COMP_DIR / "Sentences_Oare_FirstWord_LinNum.csv"
    uuid_to_sentences = defaultdict(list)
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            text_uuid = row['text_uuid']
            trans = row.get('translation', '')
            if trans:
                uuid_to_sentences[text_uuid].append({
                    'display_name': row.get('display_name', ''),
                    'translation': trans,
                })
    return uuid_to_sentences


def load_expert_sentences():
    path = DATA_DIR / "expert_translations_repaired_sentence_output_v16.jsonl"
    oare_to_expert = {}
    if not path.exists():
        print(f"  Warning: {path} not found, skipping expert data")
        return oare_to_expert
    with open(path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                oare_to_expert[rec['oare_id']] = rec
    return oare_to_expert


def load_synthetic_sentences():
    path = DATA_DIR / "synthetic_translations_sentence_v16_etxra1.jsonl"
    oare_to_synth = {}
    if not path.exists():
        print(f"  Warning: {path} not found, skipping synthetic data")
        return oare_to_synth
    with open(path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                oare_to_synth[rec['oare_id']] = rec
    return oare_to_synth


def load_hecker_pairs(input_path: str):
    pairs = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def match_by_id(hecker_id, aliases_to_oare, museum_number, museum_to_oare,
                excav_number, excav_to_oare):
    norm_id = normalize_hecker_id(hecker_id)
    if norm_id in aliases_to_oare:
        return aliases_to_oare[norm_id], "alias"
    if museum_number and museum_number in museum_to_oare:
        return museum_to_oare[museum_number], "museum_number"
    if excav_number and excav_number in excav_to_oare:
        return excav_to_oare[excav_number], "excavation_number"
    return None, None


def find_best_translation(oare_id, train_trans, sentences_oare, expert, synthetic):
    if oare_id in train_trans:
        return train_trans[oare_id], "train"
    if oare_id in expert:
        rec = expert[oare_id]
        trans = rec.get('corrected_translation', '') or rec.get('translation', '')
        if trans:
            return trans, "expert"
    if oare_id in synthetic:
        rec = synthetic[oare_id]
        trans = rec.get('corrected_translation', '') or rec.get('translation', '')
        if trans:
            return trans, "synthetic"
    if oare_id in sentences_oare:
        sents = sentences_oare[oare_id]
        if sents:
            combined = " ".join(s['translation'] for s in sents)
            return combined, "sentences_oare"
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Cross-reference Hecker extractions against data sources")
    parser.add_argument("--input", required=True, help="Path to Hecker extracted JSONL")
    parser.add_argument("--output", default=None, help="Output enriched JSONL (default: input with _crossref suffix)")
    parser.add_argument("--fuzzy-threshold", type=int, default=75,
                        help="Min fuzz.ratio score for transliteration matching (default: 75)")
    parser.add_argument("--no-fuzzy", action="store_true",
                        help="Disable fuzzy transliteration matching")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_crossref.jsonl")

    print("Loading data sources...")
    aliases_to_oare, museum_to_oare, excav_to_oare, oare_to_translit = load_published_texts()
    print(f"  published_texts: {len(aliases_to_oare)} aliases, {len(museum_to_oare)} museum numbers, {len(excav_to_oare)} excavation numbers")

    train_trans = load_train_translations()
    print(f"  train.csv: {len(train_trans)} translations")

    sentences_oare = load_sentences_oare()
    print(f"  Sentences_Oare: {len(sentences_oare)} tablets")

    expert = load_expert_sentences()
    print(f"  expert: {len(expert)} records")

    synthetic = load_synthetic_sentences()
    print(f"  synthetic: {len(synthetic)} records")

    hecker_pairs = load_hecker_pairs(args.input)
    print(f"  Hecker input: {len(hecker_pairs)} pairs")

    # Build token inverted index for fuzzy matching
    token_index = None
    if not args.no_fuzzy:
        token_index = TokenIndex()
        n_indexed = 0
        for oare_id, translit in oare_to_translit.items():
            has_trans = (oare_id in train_trans or oare_id in expert or
                         oare_id in synthetic or oare_id in sentences_oare)
            if has_trans and translit:
                token_index.add(oare_id, translit)
                n_indexed += 1
        print(f"  Token index: {n_indexed} entries, {len(token_index.token_to_ids)} unique tokens")

    # --- Pre-build full tablet transliterations for fuzzy matching ---
    # Concatenate all _partN chunks into full tablet text
    tablet_translits = defaultdict(list)  # tablet_key → [(part_idx, translit, museum, excav)]
    for pair in hecker_pairs:
        hecker_id = pair.get('id', '')
        norm_id = normalize_hecker_id(hecker_id)
        translit = pair.get('transliteration', '')
        # Extract part number for ordering
        part_match = re.search(r'_part(\d+)$', hecker_id)
        part_idx = int(part_match.group(1)) if part_match else 0
        tablet_translits[norm_id].append((
            part_idx, translit,
            pair.get('museum_number', ''),
            pair.get('excavation_number', ''),
        ))

    # Sort parts and concatenate
    tablet_full_text = {}  # tablet_key → full concatenated transliteration
    tablet_metadata = {}   # tablet_key → (museum_number, excavation_number)
    for tablet_key, parts in tablet_translits.items():
        parts.sort(key=lambda x: x[0])
        tablet_full_text[tablet_key] = ' '.join(p[1] for p in parts if p[1])
        # Use metadata from first part
        tablet_metadata[tablet_key] = (parts[0][2], parts[0][3])

    print(f"  Full tablet texts: {len(tablet_full_text)} tablets")

    # --- Cross-reference ---
    print("\nCross-referencing...")
    t0 = time.time()

    stats = defaultdict(int)
    tablet_matches = {}
    enriched = []

    # Pass 1: Match all tablets (ID-based first, then fuzzy on full text)
    for tablet_key in tablet_full_text:
        museum_num, excav_num = tablet_metadata[tablet_key]

        # Try ID-based matching using any pair's ID for this tablet
        oare_id, method = match_by_id(
            tablet_key, aliases_to_oare,
            museum_num, museum_to_oare,
            excav_num, excav_to_oare,
        )

        # Fuzzy match on full concatenated tablet text
        if oare_id is None and token_index:
            full_text = tablet_full_text[tablet_key]
            if len(full_text) > 30:
                matched_id, score = token_index.fuzzy_match(
                    full_text, threshold=args.fuzzy_threshold
                )
                if matched_id:
                    oare_id = matched_id
                    method = f"fuzzy_translit({score:.0f})"

        tablet_matches[tablet_key] = (oare_id, method)

    # Pass 2: Enrich individual pairs using tablet-level matches
    for pair in hecker_pairs:
        hecker_id = pair.get('id', '')
        norm_id = normalize_hecker_id(hecker_id)
        oare_id, method = tablet_matches.get(norm_id, (None, None))

        # Find translation
        translation = None
        trans_source = None
        if oare_id:
            translation, trans_source = find_best_translation(
                oare_id, train_trans, sentences_oare, expert, synthetic
            )

        enriched_pair = dict(pair)
        enriched_pair['matched_oare_id'] = oare_id or ""
        enriched_pair['match_method'] = method or ""
        enriched_pair['matched_source'] = trans_source or ""
        if translation:
            enriched_pair['matched_translation'] = translation
            stats['pairs_with_translation'] += 1
        else:
            stats['pairs_no_translation'] += 1

        if oare_id:
            stats[f'method_{method.split("(")[0]}'] += 1
        else:
            stats['no_match'] += 1

        enriched.append(enriched_pair)

    elapsed = time.time() - t0

    # --- Tablet-level stats ---
    tablets_matched = sum(1 for v in tablet_matches.values() if v[0] is not None)
    tablets_total = len(tablet_matches)
    tablets_with_trans = 0
    for tablet_key, (oare_id, method) in tablet_matches.items():
        if oare_id:
            trans, _ = find_best_translation(oare_id, train_trans, sentences_oare, expert, synthetic)
            if trans:
                tablets_with_trans += 1

    # Write output
    with open(output_path, 'w') as f:
        for pair in enriched:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    # --- Report ---
    print(f"\n{'='*70}")
    print("CROSS-REFERENCE RESULTS")
    print(f"{'='*70}")
    print(f"Total Hecker pairs:        {len(hecker_pairs)}")
    print(f"Unique tablets:            {tablets_total}")
    print(f"Tablets matched (oare_id): {tablets_matched} ({100*tablets_matched/tablets_total:.1f}%)")
    print(f"Tablets with translation:  {tablets_with_trans} ({100*tablets_with_trans/tablets_total:.1f}%)")
    print(f"Tablets unmatched:         {tablets_total - tablets_matched}")
    print(f"\nPair-level:")
    print(f"  Pairs with translation:  {stats['pairs_with_translation']}")
    print(f"  Pairs no translation:    {stats['pairs_no_translation']}")
    print(f"\nMatch methods:")
    for k, v in sorted(stats.items()):
        if k.startswith('method_'):
            print(f"  {k[7:]:25s} {v}")
    print(f"  {'no_match':25s} {stats['no_match']}")
    print(f"\nTime: {elapsed:.1f}s")
    print(f"Output: {output_path}")

    # Show sample matches
    print(f"\n--- Sample matches ---")
    shown = 0
    for pair in enriched:
        if pair.get('matched_oare_id') and pair.get('matched_translation'):
            tid = pair['id']
            method = pair['match_method']
            src = pair['matched_source']
            trans = pair['matched_translation'][:80]
            print(f"  {tid:30s}  [{method}/{src}]  {trans}...")
            shown += 1
            if shown >= 10:
                break

    # Show sample unmatched
    print(f"\n--- Sample unmatched ---")
    shown = 0
    for pair in enriched:
        if not pair.get('matched_oare_id'):
            tid = pair['id']
            translit = pair['transliteration'][:60]
            print(f"  {tid:30s}  {translit}...")
            shown += 1
            if shown >= 5:
                break


if __name__ == "__main__":
    main()
