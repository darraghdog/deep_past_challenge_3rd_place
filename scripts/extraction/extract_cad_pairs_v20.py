"""
Extract Old Assyrian transliteration/translation pairs from CAD PDF volumes using Gemini.
Version 20: Based on extract_akt_pairs_v20.py structure (retry, per-PDF checkpointing,
model selection) with CAD-specific nested parsing from extract_cad_pairs.py.

Re-extracts all 26 CAD volumes with Gemini 3.1 Pro at 1024 DPI, 10-page overlapping chunks.
Original extraction (extract_cad_pairs.py) used Flash at 150 DPI with 30-page chunks.

Usage:
    # Dry-run to verify all PDFs and chunk counts
    python scripts/extract_cad_pairs_v20.py --dry-run

    # Test on smallest volume
    python scripts/extract_cad_pairs_v20.py --pdf-filter CAD_19

    # Full extraction
    python -u scripts/extract_cad_pairs_v20.py

    # Re-flatten checkpoints without re-extracting
    python scripts/extract_cad_pairs_v20.py --flatten-only

    # Use different model
    python -u scripts/extract_cad_pairs_v20.py --model flash
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import fitz  # PyMuPDF
import requests

# Directories
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent.parent  # scripts/extraction/ → repo root
DATA_DIR = Path(os.environ.get("DATA_DIR", str(REPO_DIR / "data")))
PROMPTS_DIR = REPO_DIR / "prompts"
sys.path.insert(0, str(REPO_DIR / "scripts"))

# Paths
PDF_DIR = DATA_DIR / "CAD_open"
PROMPT_FILE = "cad_side_by_side_parsing.txt"

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
DEFAULT_DPI = 1024
DEFAULT_OVERLAP = 2
DEFAULT_MODEL = "pro31"
MAX_RETRIES = 3


def load_prompt(prompt_filename: str) -> str:
    """Load the extraction prompt from file."""
    prompt_path = PROMPTS_DIR / prompt_filename
    if prompt_path.exists():
        return prompt_path.read_text()
    raise FileNotFoundError(f"Prompt file not found: {prompt_path}")


def pdf_to_base64_images(pdf_path: Path, start_page: int, end_page: int, dpi: int = DEFAULT_DPI) -> List[str]:
    """Convert PDF pages to base64-encoded PNG images.

    Args:
        pdf_path: Path to PDF file
        start_page: Starting page (0-indexed)
        end_page: Ending page (exclusive)
        dpi: Resolution for rendering

    Returns:
        List of base64-encoded PNG strings
    """
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
    """Send images to Gemini via NVIDIA API with retry + exponential backoff.

    Args:
        images: List of base64-encoded PNG images
        prompt: System prompt for extraction
        api_key: NVIDIA API key
        model_key: "flash", "pro", or "pro31"
        max_retries: Maximum retry attempts

    Returns:
        Model response text or None on failure
    """
    model = MODELS[model_key]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    user_content = [
        {"type": "text", "text": "Extract Old Assyrian transliteration/translation pairs from these CAD pages:"}
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
            rj = response.json()
            finish_reason = rj["choices"][0].get("finish_reason", "")
            content = rj["choices"][0]["message"].get("content")
            if finish_reason == "content_filter" or content is None:
                print(f"    Content filtered (finish_reason={finish_reason})")
                return None
            return content
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


def parse_cad_response(response: str) -> List[Dict]:
    """Extract JSON array of CAD headword entries from Gemini response.

    The CAD prompt outputs a JSON array of nested headword entries directly.

    Returns:
        List of headword entry dictionaries
    """
    match = re.search(r'\[[\s\S]*\]', response)
    if not match:
        return []

    json_str = match.group()
    try:
        entries = json.loads(json_str)
        if not isinstance(entries, list):
            entries = [entries]
        return entries
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        # Try to extract individual top-level objects
        obj_pattern = r'\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}[^{}]*)*\}'
        matches = re.findall(obj_pattern, json_str)
        entries = []
        for m in matches:
            try:
                obj = json.loads(m)
                if 'headword' in obj:
                    entries.append(obj)
            except json.JSONDecodeError:
                continue
        return entries


def get_pdf_page_count(pdf_path: Path) -> int:
    """Get the number of pages in a PDF."""
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Dict]:
    """Load checkpoint file and return completed chunks.

    Returns:
        Dict mapping chunk names to their checkpoint entries
    """
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
    """Calculate chunk boundaries with optional overlap.

    Args:
        page_count: Total number of pages
        chunk_size: Pages per chunk
        overlap: Number of overlapping pages between chunks

    Returns:
        List of (start_page, end_page) tuples (0-indexed start, exclusive end)
    """
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


def count_attestations(entries: List[Dict]) -> int:
    """Count total OA attestations across all entries."""
    return sum(
        len(att)
        for entry in entries
        for sense in entry.get('senses', [])
        for att in [sense.get('oa_attestations', [])]
    )


def process_pdf(
    pdf_path: Path,
    prompt: str,
    api_key: str,
    chunk_size: int,
    overlap: int,
    checkpoint_path: Path,
    completed_chunks: Dict[str, Dict],
    model_key: str = DEFAULT_MODEL,
    dpi: int = DEFAULT_DPI,
    dry_run: bool = False,
) -> List[Dict]:
    """Process a single PDF file in chunks with per-PDF checkpoint.

    Returns:
        List of all extracted headword entries from this PDF
    """
    pdf_name = pdf_path.stem
    page_count = get_pdf_page_count(pdf_path)
    all_entries = []

    print(f"\nProcessing {pdf_name} ({page_count} pages)")

    chunk_boundaries = calculate_chunks(page_count, chunk_size, overlap)
    num_chunks = len(chunk_boundaries)

    if overlap > 0:
        print(f"  Chunks: {num_chunks} (size={chunk_size}, overlap={overlap}, stride={chunk_size - overlap})")

    for chunk_idx, (start_page, end_page) in enumerate(chunk_boundaries):
        chunk_name = f"{pdf_name}_p{start_page+1:03d}-{end_page:03d}"

        if chunk_name in completed_chunks:
            print(f"  Chunk {chunk_idx+1}/{num_chunks} ({chunk_name}): SKIPPED (checkpoint)")
            chunk_data = completed_chunks[chunk_name]
            all_entries.extend(chunk_data.get('entries', []))
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
            continue

        entries = parse_cad_response(response)
        num_att = count_attestations(entries)
        print(f"    Extracted {len(entries)} entries ({num_att} attestations) in {elapsed:.1f}s")

        for entry in entries:
            entry['chunk'] = chunk_name
            entry['source_volume'] = pdf_name

        all_entries.extend(entries)

        chunk_data = {
            'chunk': chunk_name,
            'pdf_name': pdf_name,
            'start_page': start_page + 1,
            'end_page': end_page,
            'num_entries': len(entries),
            'num_attestations': num_att,
            'entries': entries,
            'model': model_key,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        save_checkpoint(checkpoint_path, chunk_data)

        if chunk_idx < num_chunks - 1:
            time.sleep(2)

    return all_entries


def flatten_entries_to_pairs(
    checkpoint_path: Path,
    output_path: Path,
) -> Tuple[int, int, int]:
    """Flatten checkpoint entries to one attestation pair per line with dedup.

    Walks entries[].senses[].oa_attestations[] and emits one line per attestation
    with headword metadata attached.

    Dedup key: (source_ref, mt_transliteration[:50]) — removes overlap duplicates.

    Returns:
        Tuple of (total_entries, total_pairs_before_dedup, total_pairs_after_dedup)
    """
    all_pairs = []
    seen: Set[Tuple[str, str]] = set()
    total_entries = 0
    duplicates_removed = 0

    with open(checkpoint_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            chunk_data = json.loads(line)
            chunk_name = chunk_data.get('chunk', '')
            pdf_name = chunk_data.get('pdf_name', '')

            for entry in chunk_data.get('entries', []):
                total_entries += 1
                headword = entry.get('headword', '')
                headword_variants = entry.get('headword_variants', [])
                pos = entry.get('pos', '')
                source_volume = entry.get('source_volume', pdf_name)

                for sense in entry.get('senses', []):
                    sense_id = sense.get('sense_id', '')
                    gloss = sense.get('gloss', '')
                    domain = sense.get('domain', '')

                    for att in sense.get('oa_attestations', []):
                        pair = {
                            'mt_transliteration': att.get('mt_transliteration', ''),
                            'mt_translation': att.get('mt_translation', ''),
                            'raw_transliteration': att.get('raw_transliteration', ''),
                            'raw_translation': att.get('raw_translation', ''),
                            'headword': headword,
                            'headword_variants': headword_variants,
                            'pos': pos,
                            'sense_id': sense_id,
                            'gloss': gloss,
                            'domain': domain,
                            'source_ref': att.get('source_ref', ''),
                            'oa_confidence': att.get('oa_confidence', ''),
                            'confidence_reason': att.get('confidence_reason', ''),
                            'surface_forms': att.get('surface_forms', []),
                            'chunk': entry.get('chunk', chunk_name),
                            'source_volume': source_volume,
                        }

                        dedup_key = (
                            pair['source_ref'],
                            pair['mt_transliteration'][:50],
                        )
                        if dedup_key in seen:
                            duplicates_removed += 1
                            continue
                        seen.add(dedup_key)
                        all_pairs.append(pair)

    with open(output_path, 'w') as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    total_before = len(all_pairs) + duplicates_removed
    return total_entries, total_before, len(all_pairs)


def flatten_all_checkpoints(output_dir: Path, output_path: Path) -> Tuple[int, int, int]:
    """Merge all per-PDF checkpoint files, flatten, and dedup.

    Args:
        output_dir: Directory containing checkpoint_*.jsonl files
        output_path: Path for aggregate output JSONL

    Returns:
        Tuple of (total_entries, total_pairs_before_dedup, total_pairs_after_dedup)
    """
    all_pairs = []
    seen: Set[Tuple[str, str]] = set()
    total_entries = 0
    duplicates_removed = 0

    checkpoint_files = sorted(output_dir.glob("checkpoint_*.jsonl"))
    if not checkpoint_files:
        print(f"  No checkpoint files found in {output_dir}")
        return 0, 0, 0

    print(f"  Merging {len(checkpoint_files)} checkpoint files...")

    for ckpt_path in checkpoint_files:
        with open(ckpt_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                chunk_data = json.loads(line)
                chunk_name = chunk_data.get('chunk', '')
                pdf_name = chunk_data.get('pdf_name', '')

                for entry in chunk_data.get('entries', []):
                    total_entries += 1
                    headword = entry.get('headword', '')
                    headword_variants = entry.get('headword_variants', [])
                    pos = entry.get('pos', '')
                    source_volume = entry.get('source_volume', pdf_name)

                    for sense in entry.get('senses', []):
                        sense_id = sense.get('sense_id', '')
                        gloss = sense.get('gloss', '')
                        domain = sense.get('domain', '')

                        for att in sense.get('oa_attestations', []):
                            pair = {
                                'mt_transliteration': att.get('mt_transliteration', ''),
                                'mt_translation': att.get('mt_translation', ''),
                                'raw_transliteration': att.get('raw_transliteration', ''),
                                'raw_translation': att.get('raw_translation', ''),
                                'headword': headword,
                                'headword_variants': headword_variants,
                                'pos': pos,
                                'sense_id': sense_id,
                                'gloss': gloss,
                                'domain': domain,
                                'source_ref': att.get('source_ref', ''),
                                'oa_confidence': att.get('oa_confidence', ''),
                                'confidence_reason': att.get('confidence_reason', ''),
                                'surface_forms': att.get('surface_forms', []),
                                'chunk': entry.get('chunk', chunk_name),
                                'source_volume': source_volume,
                            }

                            dedup_key = (
                                pair['source_ref'],
                                pair['mt_transliteration'][:50],
                            )
                            if dedup_key in seen:
                                duplicates_removed += 1
                                continue
                            seen.add(dedup_key)
                            all_pairs.append(pair)

    with open(output_path, 'w') as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    total_before = len(all_pairs) + duplicates_removed
    return total_entries, total_before, len(all_pairs)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract OA transliteration/translation pairs from CAD PDFs (v20, Pro 3.1 + 1024 DPI)"
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
        help=f"Gemini model: flash, pro, or pro31 (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--pdf-filter",
        type=str,
        default=None,
        help="Process only PDFs whose name contains this string (e.g. 'CAD_04', 'CAD_19')",
    )
    parser.add_argument(
        "--pdf-list",
        type=str,
        default=None,
        help="Comma-separated list of PDF stems to process (e.g. 'CAD_01-1_A-AL_open,CAD_02_B_open')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview PDFs and chunk counts without API calls",
    )
    parser.add_argument(
        "--flatten-only",
        action="store_true",
        help="Re-flatten existing checkpoints to output without re-extracting",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Output directory includes model name
    output_dir = DATA_DIR / f"CAD_open_extracted_v20_{args.model}"
    output_path = output_dir / f"cad_pairs_v20_{args.model}.jsonl"

    # --- flatten-only mode ---
    if args.flatten_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 70)
        print(f"FLATTEN-ONLY: {output_dir}")
        print("=" * 70)
        total_entries, before, after = flatten_all_checkpoints(output_dir, output_path)
        if before > 0:
            removed = before - after
            print(f"\nOutput: {output_path}")
            print(f"Total entries: {total_entries}")
            print(f"Total pairs: {before} -> {after} after dedup ({removed} duplicates removed)")
        else:
            print("No checkpoint data found.")
        return

    # --- Normal extraction mode ---
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for API key
    api_key = os.environ.get("AKKADIAN_KEY")
    if not api_key and not args.dry_run:
        print("Error: AKKADIAN_KEY environment variable not set")
        print("Set it with: source ~/.bash_profile")
        return

    # Load prompt
    try:
        prompt = load_prompt(PROMPT_FILE)
        print(f"Loaded prompt: {PROMPT_FILE} ({len(prompt)} chars)")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Find PDFs
    pdf_files = sorted(PDF_DIR.glob("CAD_*.pdf"))
    if not pdf_files:
        pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if args.pdf_filter:
        pdf_files = [p for p in pdf_files if args.pdf_filter in p.name]

    if args.pdf_list:
        stems = set(s.strip() for s in args.pdf_list.split(','))
        pdf_files = [p for p in pdf_files if p.stem in stems]

    if not pdf_files:
        print(f"Error: No PDF files found in {PDF_DIR}")
        if args.pdf_filter:
            print(f"  (with filter: '{args.pdf_filter}')")
        return

    # Print configuration
    print("=" * 70)
    print(f"CAD PDF EXTRACTION v20 ({args.model.upper()})")
    print("=" * 70)
    print(f"Model:      {MODELS[args.model]}")
    print(f"PDF dir:    {PDF_DIR}")
    print(f"Output dir: {output_dir}")
    print(f"Chunk size: {args.chunk_size} pages")
    print(f"Overlap:    {args.overlap} pages (stride={args.chunk_size - args.overlap})")
    print(f"DPI:        {args.dpi}")
    if args.pdf_filter:
        print(f"PDF filter: '{args.pdf_filter}'")
    if args.pdf_list:
        print(f"PDF list:   {len(pdf_files)} PDFs from explicit list")
    if args.dry_run:
        print("*** DRY RUN — no API calls ***")
    print(f"\nPDFs to process: {len(pdf_files)}")

    # Build per-PDF plan
    total_pages = 0
    total_chunks = 0

    for pdf_path in pdf_files:
        page_count = get_pdf_page_count(pdf_path)
        chunks = calculate_chunks(page_count, args.chunk_size, args.overlap)
        total_pages += page_count
        total_chunks += len(chunks)
        print(f"  {pdf_path.name:40s}  {page_count:4d} pages  {len(chunks):3d} chunks")

    print(f"\nTotal: {total_pages} pages, {total_chunks} chunks")
    print("=" * 70)

    # Process each PDF
    start_time = time.time()
    total_entries = 0
    total_attestations = 0

    for pdf_path in pdf_files:
        pdf_name = pdf_path.stem

        # Per-PDF checkpoint
        checkpoint_path = output_dir / f"checkpoint_{pdf_name}.jsonl"
        completed_chunks = load_checkpoint(checkpoint_path)
        if completed_chunks:
            print(f"\n  Resuming {pdf_name}: {len(completed_chunks)} chunks in checkpoint")

        entries = process_pdf(
            pdf_path=pdf_path,
            prompt=prompt,
            api_key=api_key,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            checkpoint_path=checkpoint_path,
            completed_chunks=completed_chunks,
            model_key=args.model,
            dpi=args.dpi,
            dry_run=args.dry_run,
        )
        total_entries += len(entries)
        total_attestations += count_attestations(entries)

    # Flatten all checkpoints to aggregate output
    if not args.dry_run:
        print(f"\n{'='*70}")
        print("FLATTENING CHECKPOINTS")
        print(f"{'='*70}")
        entry_count, before, after = flatten_all_checkpoints(output_dir, output_path)
        if before > 0:
            removed = before - after
            print(f"\nOutput: {output_path}")
            print(f"Total entries: {entry_count}")
            print(f"Total pairs: {before} -> {after} after dedup ({removed} duplicates removed)")

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"{'='*70}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Total entries extracted: {total_entries}")
    print(f"Total attestations: {total_attestations}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
