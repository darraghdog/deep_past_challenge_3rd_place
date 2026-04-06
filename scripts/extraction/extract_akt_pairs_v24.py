"""
Extract transliteration/translation pairs from AKT/OAA PDFs using Gemini Flash/Pro.
Version 24: Semantic chunking with start_line/end_line, dual quality scores
(ocr_quality + match_quality), 576 DPI. Side-by-side only (v24 prompt).
Other modes fall back to v23 prompts.

All 22 PDFs live in datamount/OCR_V20/{side_by_side,top_bottom,ocr}/.

Usage:
    # Dry-run to verify prompt assignments
    python scripts/extract_akt_pairs_v24.py --mode side_by_side --dry-run

    # Test on single small PDF
    python scripts/extract_akt_pairs_v24.py --mode side_by_side --pdf-filter Kouwenberg

    # Full extraction runs
    python -u scripts/extract_akt_pairs_v24.py --mode side_by_side

    # Re-flatten checkpoints without re-extracting
    python scripts/extract_akt_pairs_v24.py --mode side_by_side --flatten-only

    # Use Pro model instead of Flash
    python -u scripts/extract_akt_pairs_v24.py --mode side_by_side --model pro
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from rapidfuzz import fuzz

import fitz  # PyMuPDF
import requests

# Directories
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent.parent  # scripts/extraction/ → repo root
DATA_DIR = Path(os.environ.get("DATA_DIR", str(REPO_DIR / "data")))
PROMPTS_DIR = REPO_DIR / "prompts"
sys.path.insert(0, str(REPO_DIR / "scripts"))

# API Configuration
API_URL = "https://inference-api.nvidia.com/v1/chat/completions"
MODELS = {
    "flash": "gcp/google/gemini-3-flash-preview",
    "pro": "gcp/google/gemini-3-pro-preview",
    "pro31": "gcp/google/gemini-3.1-pro-preview",
}
TEMPERATURE = 0.3
MAX_TOKENS = 65536

# Default configuration
DEFAULT_CHUNK_SIZE = 10
DEFAULT_DPI = 576
DEFAULT_OVERLAP = 2
DEFAULT_BASE_DIR = "OCR_V20"
DEFAULT_MODEL = "pro31"
MAX_RETRIES = 3

# --- Config-driven prompt selection ---

# Per-PDF language overrides (key = prefix matched against PDF stem)
LANG_MAP = {
    "AKT_03": "de",
    "ICK4": "de",
    "Kouwenberg": "kouwenberg_en",
}

# Default language per layout mode
LAYOUT_DEFAULTS = {
    "side_by_side": "en",
    "top_bottom": "tr",
    "ocr": "tr",
    "dergipark": "tr",
    "michel": "en",
    "hecker": "en",
    "gelb": "en",
    "veenhof": "en",
}

# (mode, language) -> prompt filename
PROMPT_MAP = {
    ("side_by_side", "en"): "akt_side_by_side_parsing_v24.txt",
    ("side_by_side", "kouwenberg_en"): "akt_side_by_side_kouwenberg_parsing_v24.txt",
    ("top_bottom", "tr"): "akt_top_bottom_turkish_align_v24.txt",
    ("top_bottom", "de"): "akt_top_bottom_and_align_parsing_v24.txt",
    ("ocr", "tr"): "akt_ocr_turkish_align_v24.txt",
    ("dergipark", "tr"): "dergipark_inline_parsing_v24.txt",
    ("michel", "en"): "michel_en_parsing_v24.txt",
    ("hecker", "en"): "hecker_translit_parsing_v24.txt",
    ("gelb", "en"): "gelb_oip27_parsing_v24.txt",
    ("veenhof", "en"): "veenhof_1972_v24.txt",
}


def get_language_for_pdf(pdf_name: str, mode: str) -> str:
    """Resolve language for a PDF from LANG_MAP + layout defaults.

    For hecker mode, skip LANG_MAP overrides — transliteration-only extraction
    uses the same prompt regardless of the original publication language.
    """
    if mode not in ("hecker", "michel", "gelb", "veenhof"):
        for prefix, lang in LANG_MAP.items():
            if pdf_name.startswith(prefix):
                return lang
    return LAYOUT_DEFAULTS[mode]


def get_prompt_for_pdf(pdf_name: str, mode: str) -> str:
    """Resolve prompt filename for a PDF."""
    lang = get_language_for_pdf(pdf_name, mode)
    key = (mode, lang)
    if key not in PROMPT_MAP:
        raise ValueError(f"No prompt for ({mode}, {lang}) — PDF: {pdf_name}")
    return PROMPT_MAP[key]


def load_prompt(prompt_filename: str) -> str:
    """Load the extraction prompt from file."""
    prompt_path = PROMPTS_DIR / prompt_filename
    if prompt_path.exists():
        return prompt_path.read_text()
    raise FileNotFoundError(f"Prompt file not found: {prompt_path}")


def pdf_to_base64_images(pdf_path: Path, start_page: int, end_page: int, dpi: int = DEFAULT_DPI) -> List[str]:
    """Convert PDF pages to base64-encoded PNG images."""
    images = []
    doc = fitz.open(pdf_path)
    end_page = min(end_page, len(doc))

    for page_num in range(start_page, end_page):
        page = doc[page_num]
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")
        b64_str = base64.b64encode(png_bytes).decode("utf-8")
        images.append(b64_str)

    doc.close()
    return images


def call_gemini_multimodal(
    images: List[str],
    prompt: str,
    api_key: str,
    model_key: str = DEFAULT_MODEL,
    max_retries: int = MAX_RETRIES,
) -> Optional[str]:
    """Send images to Gemini via NVIDIA API with retry + exponential backoff."""
    model = MODELS[model_key]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    user_content = [
        {"type": "text", "text": "Extract transliteration/translation pairs from these PDF pages:"}
    ]
    for img_b64 in images:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
        })

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=600)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            if status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 5
                print(f"    HTTP {status_code} error, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            print(f"  HTTP Error: {e}")
            print(f"  Status Code: {status_code}")
            print(f"  Response Body: {e.response.text[:500]}")
            return None
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 5
                print(f"    Timeout, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            print(f"  Timeout after {max_retries} attempts")
            return None
        except requests.exceptions.RequestException as e:
            print(f"  Request Error: {e}")
            return None
        except KeyError as e:
            print(f"  Response Parse Error: Missing key {e}")
            return None
        except Exception as e:
            print(f"  Unexpected Error: {type(e).__name__}: {e}")
            return None

    return None


def parse_extraction_response(response: str) -> List[Dict]:
    """Extract JSON array from between delimiters in response.

    Supports both v24 delimiters (PARSED CHUNKS) and v23 (PARSED LINES).
    """
    pairs = []

    # Try v24 delimiters first, then v23, then fallback
    for start_marker, end_marker in [
        ("------------------------PARSED CHUNKS------------------------",
         "------------------------End Of chunks------------------------"),
        ("------------------------PARSED LINES------------------------",
         "------------------------End Of lines------------------------"),
    ]:
        start_idx = response.find(start_marker)
        end_idx = response.find(end_marker)
        if start_idx != -1 and end_idx != -1:
            json_str = response[start_idx + len(start_marker):end_idx].strip()
            break
    else:
        # Fallback: try array first, then object
        match = re.search(r'\[[\s\S]*\]', response)
        if not match:
            match = re.search(r'\{[\s\S]*\}', response)
        if match:
            json_str = match.group()
        else:
            return []

    # Strip markdown code fences that Gemini sometimes wraps around JSON
    json_str = re.sub(r'^```(?:json)?\s*', '', json_str)
    json_str = re.sub(r'\s*```\s*$', '', json_str)

    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, dict) and ('embedded_pairs' in parsed or 'tablet_sentence_pairs' in parsed):
            # Veenhof-style dual-list object: flatten both lists with pair_type tags
            for pair in parsed.get('embedded_pairs', []):
                pair['pair_type'] = 'embedded'
                pair.setdefault('start_line', 0)
                pair.setdefault('end_line', 0)
                pairs.append(pair)
            for pair in parsed.get('tablet_sentence_pairs', []):
                pair['pair_type'] = 'tablet_sentence'
                pairs.append(pair)
        elif isinstance(parsed, list):
            pairs = parsed
        else:
            pairs = [parsed]
    except json.JSONDecodeError as e:
        # Try raw_decode to ignore trailing junk after valid JSON
        try:
            decoder = json.JSONDecoder()
            parsed, _ = decoder.raw_decode(json_str)
            if isinstance(parsed, dict) and ('embedded_pairs' in parsed or 'tablet_sentence_pairs' in parsed):
                for pair in parsed.get('embedded_pairs', []):
                    pair['pair_type'] = 'embedded'
                    pair.setdefault('start_line', 0)
                    pair.setdefault('end_line', 0)
                    pairs.append(pair)
                for pair in parsed.get('tablet_sentence_pairs', []):
                    pair['pair_type'] = 'tablet_sentence'
                    pairs.append(pair)
                return pairs
            elif isinstance(parsed, list):
                return parsed
            else:
                return [parsed]
        except (json.JSONDecodeError, ValueError):
            pass
        print(f"  JSON parse error: {e}")
        obj_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(obj_pattern, json_str if 'json_str' in dir() else response)
        for m in matches:
            try:
                obj = json.loads(m)
                if 'transliteration' in obj and 'translation' in obj:
                    pairs.append(obj)
            except json.JSONDecodeError:
                continue

    return pairs


def get_pdf_page_count(pdf_path: Path) -> int:
    """Get the number of pages in a PDF."""
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Dict]:
    """Load checkpoint file and return completed chunks."""
    completed = {}
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    chunk_name = entry.get('chunk', '')
                    completed[chunk_name] = entry
    return completed


def save_checkpoint(checkpoint_path: Path, chunk_data: Dict):
    """Append chunk results to checkpoint file."""
    with open(checkpoint_path, 'a') as f:
        f.write(json.dumps(chunk_data, ensure_ascii=False) + '\n')


def calculate_chunks(page_count: int, chunk_size: int, overlap: int) -> List[Tuple[int, int]]:
    """Calculate chunk boundaries with optional overlap."""
    chunks = []

    if overlap > 0:
        stride = chunk_size - overlap
        start = 0
        while start < page_count:
            end = min(start + chunk_size, page_count)
            chunks.append((start, end))
            start += stride
            if start >= page_count:
                break
    else:
        for start in range(0, page_count, chunk_size):
            end = min(start + chunk_size, page_count)
            chunks.append((start, end))

    return chunks


def normalize_translation(text: str) -> str:
    """Normalize translation text (sheqel -> shekel)."""
    if not text:
        return text
    text = re.sub(r'\bsheqels\b', 'shekels', text, flags=re.IGNORECASE)
    text = re.sub(r'\bSheqels\b', 'Shekels', text)
    text = re.sub(r'\bSHEQELS\b', 'SHEKELS', text)
    text = re.sub(r'\bsheqel\b', 'shekel', text, flags=re.IGNORECASE)
    text = re.sub(r'\bSheqel\b', 'Shekel', text)
    text = re.sub(r'\bSHEQEL\b', 'SHEKEL', text)
    return text


def process_pdf(
    pdf_path: Path,
    prompt: str,
    api_key: str,
    chunk_size: int,
    overlap: int,
    checkpoint_path: Path,
    completed_chunks: Dict[str, Dict],
    language: str,
    model_key: str = DEFAULT_MODEL,
    dpi: int = DEFAULT_DPI,
    dry_run: bool = False,
    max_chunks: int = 0,
) -> List[Dict]:
    """Process a single PDF file in chunks with per-PDF checkpoint."""
    pdf_name = pdf_path.stem
    page_count = get_pdf_page_count(pdf_path)
    all_pairs = []

    print(f"\nProcessing {pdf_name} ({page_count} pages, lang={language})")

    chunk_boundaries = calculate_chunks(page_count, chunk_size, overlap)
    num_chunks = len(chunk_boundaries)

    if overlap > 0:
        print(f"  Chunks: {num_chunks} (size={chunk_size}, overlap={overlap}, stride={chunk_size - overlap})")

    chunks_processed = 0
    for chunk_idx, (start_page, end_page) in enumerate(chunk_boundaries):
        if max_chunks > 0 and chunks_processed >= max_chunks:
            print(f"  Reached --max-chunks={max_chunks}, stopping.")
            break

        chunk_name = f"{pdf_name}_p{start_page+1:03d}-{end_page:03d}"

        if chunk_name in completed_chunks:
            print(f"  Chunk {chunk_idx+1}/{num_chunks} ({chunk_name}): SKIPPED (checkpoint)")
            chunk_data = completed_chunks[chunk_name]
            all_pairs.extend(chunk_data.get('pairs', []))
            continue

        print(f"  Chunk {chunk_idx+1}/{num_chunks}: pages {start_page+1}-{end_page}")

        if dry_run:
            print(f"    [DRY RUN] Would process {end_page - start_page} pages")
            continue

        print(f"    Converting pages to images (dpi={dpi})...")
        images = pdf_to_base64_images(pdf_path, start_page, end_page, dpi=dpi)
        print(f"    Calling {model_key} API with {len(images)} images...")

        start_time = time.time()
        response = call_gemini_multimodal(images, prompt, api_key, model_key=model_key)
        elapsed = time.time() - start_time

        if response is None:
            print(f"    API call failed! Skipping chunk.")
            chunks_processed += 1
            continue

        pairs = parse_extraction_response(response)
        print(f"    Extracted {len(pairs)} pairs in {elapsed:.1f}s")

        for pair in pairs:
            pair['chunk'] = chunk_name

        all_pairs.extend(pairs)

        chunk_data = {
            'chunk': chunk_name,
            'pdf_name': pdf_name,
            'start_page': start_page + 1,
            'end_page': end_page,
            'num_pairs': len(pairs),
            'language': language,
            'model': model_key,
            'pairs': pairs,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        save_checkpoint(checkpoint_path, chunk_data)

        chunks_processed += 1

        if chunk_idx < num_chunks - 1:
            time.sleep(2)

    return all_pairs


# Quality priority for dedup — use match_quality if available, fall back to quality
QUALITY_RANK = {"high": 0, "medium": 1, "low": 2}


def _get_quality(pair: Dict) -> int:
    """Get quality rank, preferring match_quality over quality."""
    q = pair.get('match_quality', pair.get('quality', 'low'))
    return QUALITY_RANK.get(q, 2)


def _pick_best(a: Dict, b: Dict) -> Dict:
    """Pick the better of two duplicate pairs. Prefer real translation, then higher quality."""
    def score(p):
        trans = p.get('translation', '') or p.get('english_translation', '')
        is_missing = 1 if trans == 'MISSING' else 0
        return (is_missing, _get_quality(p))
    return a if score(a) <= score(b) else b


def _dedup_chunk_boundary(tail: List[Dict], head: List[Dict], edge: int = 5) -> Tuple[List[Dict], int]:
    """Deduplicate at the boundary between two consecutive chunks for the same tablet.

    Takes the last `edge` pairs of the previous chunk and first `edge` pairs of
    the next chunk. Finds the best alignment offset by sliding and counting fuzzy
    matches (rapidfuzz.fuzz.ratio >= 80). At the best offset, matched pairs are
    merged (keep best), unmatched pairs from both sides are kept.
    """
    tail_edge = tail[-edge:] if len(tail) > edge else tail[:]
    head_edge = head[:edge] if len(head) > edge else head[:]

    if not tail_edge or not head_edge:
        return tail + head, 0

    tail_translits = [p.get('transliteration', '') for p in tail_edge]
    head_translits = [p.get('transliteration', '') for p in head_edge]

    # Try all offsets: head[0] could match tail[k] for k in range(len(tail_edge))
    best_offset = -1
    best_matches = 0

    for offset in range(len(tail_edge)):
        matches = 0
        for j in range(len(head_edge)):
            i = offset + j
            if i >= len(tail_edge):
                break
            if tail_translits[i] and head_translits[j]:
                ratio = fuzz.ratio(tail_translits[i], head_translits[j])
                if ratio >= 80:
                    matches += 1
        if matches > best_matches:
            best_matches = matches
            best_offset = offset

    if best_matches == 0:
        return tail + head, 0

    # Build merged result
    result = tail[:-len(tail_edge)] if len(tail) > len(tail_edge) else []
    result.extend(tail_edge[:best_offset])
    dupes = 0
    overlap_len = min(len(tail_edge) - best_offset, len(head_edge))
    for j in range(overlap_len):
        i = best_offset + j
        t_trans = tail_translits[i] if i < len(tail_translits) else ''
        h_trans = head_translits[j]
        if t_trans and h_trans and fuzz.ratio(t_trans, h_trans) >= 80:
            result.append(_pick_best(tail_edge[i], head_edge[j]))
            dupes += 1
        else:
            result.append(tail_edge[i])
            result.append(head_edge[j])
    for i in range(best_offset + overlap_len, len(tail_edge)):
        result.append(tail_edge[i])
    for j in range(overlap_len, len(head_edge)):
        result.append(head_edge[j])
    if len(head) > len(head_edge):
        result.extend(head[len(head_edge):])

    return result, dupes


def flatten_all_checkpoints(output_dir: Path, output_path: Path) -> Tuple[int, int]:
    """Merge all checkpoint_*.jsonl in output_dir, boundary + fuzzy dedup, write output.

    Two-pass dedup:
    1. Boundary dedup: align overlapping chunk edges per tablet
    2. Fuzzy dedup: within each tablet, remove pairs with similar transliteration (>=80)
    """
    raw_pairs = []

    checkpoint_files = sorted(output_dir.glob("checkpoint_*.jsonl"))
    if not checkpoint_files:
        print(f"  No checkpoint files found in {output_dir}")
        return 0, 0

    print(f"  Merging {len(checkpoint_files)} checkpoint files...")

    for ckpt_path in checkpoint_files:
        with open(ckpt_path, 'r') as f:
            for line_text in f:
                if line_text.strip():
                    entry = json.loads(line_text)
                    chunk_name = entry.get('chunk', '')
                    for pair in entry.get('pairs', []):
                        if 'english_translation' in pair:
                            pair['english_translation'] = normalize_translation(pair['english_translation'])
                        if 'translation' in pair:
                            pair['translation'] = normalize_translation(pair['translation'])
                        if 'chunk' not in pair:
                            pair['chunk'] = chunk_name
                        raw_pairs.append(pair)

    total_before = len(raw_pairs)

    # Pass 0: Intra-chunk dedup — remove exact duplicates within the same chunk
    # Gemini occasionally outputs the same pair twice in one API response.
    # Key on (id, start_line, end_line) within each chunk.
    seen_intra = set()
    deduped_pairs = []
    intra_dupes = 0
    for pair in raw_pairs:
        sl, el = pair.get('start_line', 0), pair.get('end_line', 0)
        # Embedded pairs (start_line=0, end_line=0) need transliteration in key
        # to avoid collapsing distinct pairs from the same tablet
        extra = pair.get('transliteration', '')[:50] if sl == 0 and el == 0 else ''
        key = (pair.get('chunk', ''), pair.get('id', ''), sl, el, extra)
        if key in seen_intra:
            intra_dupes += 1
            continue
        seen_intra.add(key)
        deduped_pairs.append(pair)

    if intra_dupes > 0:
        print(f"  Intra-chunk dedup: {total_before} -> {len(deduped_pairs)} ({intra_dupes} duplicates removed)")
    raw_pairs = deduped_pairs

    # Pass 1: For each tablet in multiple chunks, pick the best chunk
    # Priority: more lines covered > fewer MISSING > higher quality > more chars
    tablet_chunks: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))

    for pair in raw_pairs:
        tablet_id = pair.get('id', '')
        chunk = pair.get('chunk', '')
        tablet_chunks[tablet_id][chunk].append(pair)

    def _chunk_score(pairs: List[Dict]) -> tuple:
        """Score a chunk's extraction of a tablet. Higher = better."""
        n_pairs = len(pairs)
        n_missing = sum(
            1 for p in pairs
            if (p.get('translation', '') or p.get('english_translation', '')) == 'MISSING'
        )
        n_real = n_pairs - n_missing
        avg_quality = sum(2 - _get_quality(p) for p in pairs) / n_pairs if n_pairs else 0
        total_chars = sum(len(p.get('transliteration', '')) for p in pairs)
        return (n_pairs, n_real, avg_quality, total_chars)

    all_pairs = []
    chunk_dupes_removed = 0

    for tablet_id in sorted(tablet_chunks.keys()):
        chunks = tablet_chunks[tablet_id]

        if len(chunks) == 1:
            for pairs in chunks.values():
                all_pairs.extend(pairs)
            continue

        # Pick best chunk by score
        best_chunk = max(chunks.keys(), key=lambda c: _chunk_score(chunks[c]))
        all_pairs.extend(chunks[best_chunk])
        chunk_dupes_removed += sum(len(pairs) for c, pairs in chunks.items() if c != best_chunk)

    print(f"  Chunk pick:     {total_before} -> {len(all_pairs)} ({chunk_dupes_removed} removed from non-best chunks)")

    final_pairs = all_pairs

    with open(output_path, 'w') as f:
        for pair in final_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    return total_before, len(final_pairs)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract AKT/OAA transliteration/translation pairs (v24, semantic chunking, 576 DPI)"
    )
    parser.add_argument(
        "--mode",
        choices=["side_by_side", "top_bottom", "ocr", "dergipark", "michel", "hecker", "gelb", "veenhof"],
        required=True,
        help="Layout mode: side_by_side, top_bottom, ocr, dergipark, michel, hecker, gelb, or veenhof",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=DEFAULT_BASE_DIR,
        help=f"Base directory containing mode subdirs (default: {DEFAULT_BASE_DIR})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Pages per API call (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help=f"Page overlap between chunks (default: {DEFAULT_OVERLAP}, stride=chunk_size-overlap)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"DPI for PDF rendering (default: {DEFAULT_DPI})",
    )
    parser.add_argument(
        "--model",
        choices=["flash", "pro", "pro31"],
        default=DEFAULT_MODEL,
        help=f"Gemini model: flash or pro (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--pdf-filter",
        type=str,
        default=None,
        help="Process only PDFs whose name contains this string (e.g. 'AKT_05', 'Kouwenberg')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview prompt assignments and chunk counts without API calls",
    )
    parser.add_argument(
        "--flatten-only",
        action="store_true",
        help="Re-flatten existing checkpoints to output without re-extracting",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Max chunks to process per PDF (0=all, useful for testing)",
    )
    parser.add_argument(
        "--shard",
        type=str,
        default=None,
        help="Process shard K of N (e.g. '0/4' = first of 4 shards). PDFs split round-robin.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve base directory (relative to DATA_DIR or absolute)
    if os.path.isabs(args.base_dir):
        base_dir = Path(args.base_dir)
    else:
        base_dir = DATA_DIR / args.base_dir

    # For most modes, PDFs are in base_dir/mode/. If that subdir doesn't exist,
    # use base_dir directly (e.g. dergipark where PDFs sit in the base dir).
    candidate_dir = base_dir / args.mode
    pdf_dir = candidate_dir if candidate_dir.is_dir() else base_dir
    output_subdir = f"extracted_v24_{args.model}"
    output_dir = pdf_dir / output_subdir
    output_path = output_dir / f"akt_pairs_v24_{args.model}.jsonl"

    # --- flatten-only mode ---
    if args.flatten_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 70)
        print(f"FLATTEN-ONLY: {output_dir}")
        print("=" * 70)
        before, after = flatten_all_checkpoints(output_dir, output_path)
        if before > 0:
            removed = before - after
            print(f"\nOutput: {output_path}")
            print(f"Total: {before} pairs -> {after} after dedup ({removed} duplicates removed)")
        else:
            print("No checkpoint data found.")
        return

    # --- Normal extraction mode ---
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for API key
    api_key = os.environ.get("AKKADIAN_KEY")
    if not api_key and not args.dry_run:
        print("Error: AKKADIAN_KEY environment variable not set")
        print("Set it with: export AKKADIAN_KEY='your-key-here'")
        return

    # Find PDFs
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if args.pdf_filter:
        pdf_files = [p for p in pdf_files if args.pdf_filter in p.name]
    if args.shard:
        shard_k, shard_n = map(int, args.shard.split("/"))
        pdf_files = [p for i, p in enumerate(pdf_files) if i % shard_n == shard_k]

    if not pdf_files:
        print(f"Error: No PDF files found in {pdf_dir}")
        if args.pdf_filter:
            print(f"  (with filter: '{args.pdf_filter}')")
        return

    # Print configuration
    print("=" * 70)
    print(f"AKT/OAA PDF EXTRACTION v24 ({args.model.upper()})")
    print("=" * 70)
    print(f"Mode:       {args.mode}")
    print(f"Model:      {MODELS[args.model]}")
    print(f"PDF dir:    {pdf_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Chunk size: {args.chunk_size} pages")
    print(f"Overlap:    {args.overlap} pages (stride={args.chunk_size - args.overlap})")
    print(f"DPI:        {args.dpi}")
    if args.pdf_filter:
        print(f"PDF filter: '{args.pdf_filter}'")
    if args.shard:
        print(f"Shard:      {args.shard}")
    if args.dry_run:
        print("*** DRY RUN — no API calls ***")
    print(f"\nPDFs to process: {len(pdf_files)}")

    # Build per-PDF plan
    pdf_plans = []
    total_pages = 0
    total_chunks = 0

    for pdf_path in pdf_files:
        pdf_name = pdf_path.stem
        lang = get_language_for_pdf(pdf_name, args.mode)
        prompt_file = get_prompt_for_pdf(pdf_name, args.mode)
        page_count = get_pdf_page_count(pdf_path)
        chunks = calculate_chunks(page_count, args.chunk_size, args.overlap)

        pdf_plans.append({
            'path': pdf_path,
            'name': pdf_name,
            'lang': lang,
            'prompt_file': prompt_file,
            'page_count': page_count,
            'num_chunks': len(chunks),
        })
        total_pages += page_count
        total_chunks += len(chunks)

        print(f"  {pdf_path.name:30s}  {page_count:4d} pages  {len(chunks):3d} chunks  lang={lang:14s}  prompt={prompt_file}")

    print(f"\nTotal: {total_pages} pages, {total_chunks} chunks")
    print("=" * 70)

    # Pre-load all unique prompts
    prompt_cache: Dict[str, str] = {}
    for plan in pdf_plans:
        pf = plan['prompt_file']
        if pf not in prompt_cache:
            prompt_cache[pf] = load_prompt(pf)
            print(f"Loaded prompt: {pf} ({len(prompt_cache[pf])} chars)")

    # Process each PDF
    start_time = time.time()
    total_pairs = 0

    for plan in pdf_plans:
        pdf_path = plan['path']
        pdf_name = plan['name']
        lang = plan['lang']
        prompt = prompt_cache[plan['prompt_file']]

        # Per-PDF checkpoint
        checkpoint_path = output_dir / f"checkpoint_{pdf_name}.jsonl"
        completed_chunks = load_checkpoint(checkpoint_path)
        if completed_chunks:
            print(f"\n  Resuming {pdf_name}: {len(completed_chunks)} chunks in checkpoint")

        pairs = process_pdf(
            pdf_path=pdf_path,
            prompt=prompt,
            api_key=api_key,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            checkpoint_path=checkpoint_path,
            completed_chunks=completed_chunks,
            language=lang,
            model_key=args.model,
            dpi=args.dpi,
            dry_run=args.dry_run,
            max_chunks=args.max_chunks,
        )
        total_pairs += len(pairs)

    # Flatten all checkpoints to aggregate output
    if not args.dry_run:
        print(f"\n{'='*70}")
        print("FLATTENING CHECKPOINTS")
        print(f"{'='*70}")
        before, after = flatten_all_checkpoints(output_dir, output_path)
        if before > 0:
            removed = before - after
            print(f"\nOutput: {output_path}")
            print(f"Total: {before} pairs -> {after} after dedup ({removed} duplicates removed)")

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"{'='*70}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Total pairs extracted: {total_pairs}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
