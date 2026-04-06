"""
Split Akkadian transliterations from published_texts.csv and generate translations.

V22: Re-generation with Gemini Pro 3.1 (upgraded from Gemini 3 Pro in V13).
Pre-filters to oare_ids that survived v19 dedup (~5,181 IDs).
Uses dynamic few-shot examples from v16 expert translations.

Based on V13 (split_published_texts_13.py). Changes:
- Model: gcp/google/gemini-3.1-pro-preview (was gemini-3-pro-preview)
- Input: v2 published_texts.csv
- Few-shot reference: v16 expert (uses v2 data)
- Pre-filter: --keep-oare-ids file (default: synthetic_v22_keep_oare_ids.txt)
- Output defaults: synthetic_translations_sentence_v22.jsonl

Features:
- Dynamic few-shot examples (80 random high-quality pairs per batch)
- Batched API calls (8 samples per call) to save input tokens
- Edit distance validation for transliteration (can't validate generated translations)
- Checkpointing for resumability (checkpoints are retained as backup)
- Temperature 1.0 for more diverse translations
- Chunked parallel generation with --start-idx/--end-idx

Usage:
    python split_published_texts_v22.py
    python split_published_texts_v22.py --batch-size 8
    python split_published_texts_v22.py --fill-gaps

    # Parallel chunked generation (run in separate terminals):
    python split_published_texts_v22.py --start-idx 0 --end-idx 1296
    python split_published_texts_v22.py --start-idx 1296 --end-idx 2592
    python split_published_texts_v22.py --start-idx 2592 --end-idx 3888
    python split_published_texts_v22.py --start-idx 3888 --end-idx 5183

    # Merge chunks after completion:
    cat data/synthetic_translations_sentence_v22_chunk_*.jsonl > merged.jsonl
"""

import argparse
import json
import os
import random
import re
import sys
import time
import requests
import pandas as pd
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Directories
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent.parent  # scripts/extraction/ → repo root
DATA_DIR = Path(os.environ.get("DATA_DIR", str(REPO_DIR / "data")))
PROMPTS_DIR = REPO_DIR / "prompts"
sys.path.insert(0, str(REPO_DIR / "scripts"))

# API Configuration
API_URL = "https://inference-api.nvidia.com/v1/chat/completions"
MODEL = "gcp/google/gemini-3.1-pro-preview"
TEMPERATURE = 1.0
MAX_TOKENS = 65536

# Default files
DEFAULT_INPUT_FILE = "deep-past-initiative-machine-translation_v2/published_texts.csv"
DEFAULT_OUTPUT_FILE = "synthetic_translations_sentence_v22.jsonl"
DEFAULT_CHECKPOINT_FILE = "synthetic_translations_sentence_checkpoint_v22.jsonl"
DEFAULT_PROMPT_FILE = "prompt_v13_published_texts_sentence_split.txt"
FEW_SHOT_REFERENCE = "expert_translations_repaired_sentence_output_v16.jsonl"
DEFAULT_KEEP_OARE_IDS = "synthetic_v22_keep_oare_ids.txt"

# Default batch size for API calls (8 samples per call)
DEFAULT_BATCH_SIZE = 8

# Validation threshold
SIMILARITY_THRESHOLD = 0.9

# Few-shot configuration
FEW_SHOT_SAMPLE_SIZE = 80


def resolve_path(filename: str, base_dir) -> str:
    """Resolve a filename to full path, using base_dir if not absolute."""
    if os.path.isabs(filename) or os.path.exists(filename):
        return filename
    return os.path.join(str(base_dir), filename)


def load_prompt_template(prompt_file: str = None) -> str:
    """Load the sentence splitting prompt template."""
    prompt_path = resolve_path(prompt_file or DEFAULT_PROMPT_FILE, PROMPTS_DIR)
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    with open(prompt_path, 'r') as f:
        return f.read()


def load_few_shot_pool(few_shot_file: str = None) -> List[Dict]:
    """Load high-quality sentence pairs from expert translations."""
    few_shot_path = resolve_path(few_shot_file or FEW_SHOT_REFERENCE, DATA_DIR)
    pool = []
    with open(few_shot_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            for pair in record.get('sentence_pairs', []):
                if pair.get('quality') != 'high':
                    continue
                translation = pair.get('translation', '').strip()
                transliteration = pair.get('transliteration', '').strip()
                if not translation or not transliteration:
                    continue
                # Filter out extreme length ratios
                ratio = len(translation) / len(transliteration)
                if ratio < 0.3 or ratio > 2.0:
                    continue
                pool.append({
                    'transliteration': transliteration,
                    'translation': translation
                })
    print(f"Loaded {len(pool)} high-quality few-shot examples")
    return pool


def sample_few_shot_examples(pool: List[Dict], n: int = FEW_SHOT_SAMPLE_SIZE) -> str:
    """Randomly sample n examples and format as JSON."""
    samples = random.sample(pool, min(n, len(pool)))
    return json.dumps(samples, ensure_ascii=False, indent=2)


def load_keep_oare_ids(keep_file: str = None) -> set:
    """Load set of oare_ids to keep from file (one per line)."""
    keep_path = resolve_path(keep_file or DEFAULT_KEEP_OARE_IDS, DATA_DIR)
    if not os.path.exists(keep_path):
        raise FileNotFoundError(f"Keep oare_ids file not found: {keep_path}")
    ids = set()
    with open(keep_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(line)
    print(f"Loaded {len(ids)} keep oare_ids from {keep_path}")
    return ids


def load_data(input_file: str = None) -> List[Dict]:
    """Load input data from CSV file."""
    input_path = resolve_path(input_file or DEFAULT_INPUT_FILE, DATA_DIR)
    df = pd.read_csv(input_path)
    data = df.to_dict('records')
    print(f"Loaded {len(data)} records from {input_path}")
    return data


def compute_similarity(original: str, reconstructed: str) -> float:
    """Compute similarity ratio using SequenceMatcher (0.0 to 1.0)."""
    return SequenceMatcher(None, original, reconstructed).ratio()


def validate_sentence_pairs(record: Dict, pairs: List[Dict], threshold: float = SIMILARITY_THRESHOLD) -> Tuple[bool, float]:
    """Validate that joined transliteration pairs match original within threshold.

    Note: Unlike the expert script, we can only validate transliteration since
    published_texts.csv has no translation column to compare against.

    Returns: (is_valid, trans_similarity)
    """
    # Concatenate transliterations (join with space)
    reconstructed_trans = ' '.join(p['transliteration'] for p in pairs if p.get('transliteration'))
    trans_sim = compute_similarity(record['transliteration'], reconstructed_trans)

    is_valid = trans_sim >= threshold
    return is_valid, trans_sim


def build_prompt_with_few_shot(prompt_template: str, few_shot_pool: List[Dict], data: any) -> str:
    """Build prompt with few-shot examples and input data."""
    few_shot_str = sample_few_shot_examples(few_shot_pool, n=FEW_SHOT_SAMPLE_SIZE)
    prompt = prompt_template.replace("{EXPERT_FEW_SHOT_EXAMPLES}", few_shot_str)
    return prompt.replace(
        "<INSERT_TRANLITERATION_TRANSLATION_PAIR_HERE>",
        json.dumps(data, ensure_ascii=False, indent=2)
    )


def build_user_message(record: Dict, prompt_template: str, few_shot_pool: List[Dict]) -> str:
    """Build user message for a single sample."""
    return build_prompt_with_few_shot(
        prompt_template,
        few_shot_pool,
        {"transliteration": record["transliteration"]}
    )


def build_batch_message(records: List[Tuple[int, Dict]], prompt_template: str, few_shot_pool: List[Dict]) -> str:
    """Build user message with array of samples for batched API call."""
    batch_data = [
        {"id": local_id, "transliteration": record["transliteration"]}
        for local_id, record in records
    ]
    return build_prompt_with_few_shot(prompt_template, few_shot_pool, batch_data)


def is_valid_sentence_pairs(pairs: List) -> bool:
    """Check if pairs is a valid list of sentence pair dicts with required fields."""
    if not isinstance(pairs, list):
        return False
    for pair in pairs:
        if not isinstance(pair, dict):
            return False
        if "transliteration" not in pair or "translation" not in pair:
            return False
    return True


def extract_json_array(response: str) -> Optional[List]:
    """Extract first JSON array from response text."""
    match = re.search(r'\[[\s\S]*\]', response)
    if not match:
        return None
    try:
        parsed = json.loads(match.group())
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    return None


def parse_sentence_pairs(response: str) -> Optional[List[Dict]]:
    """Extract and validate JSON array of sentence pairs from API response."""
    pairs = extract_json_array(response)
    if pairs is not None and is_valid_sentence_pairs(pairs):
        return pairs
    return None


def parse_batch_response(response: str, expected_ids: List[int]) -> Dict[int, Optional[List[Dict]]]:
    """Parse batched API response, matching results to input IDs."""
    results = {id_: None for id_ in expected_ids}
    parsed = extract_json_array(response)
    if parsed is None:
        return results

    # Check if batch response format (has 'id' and 'sentence_pairs' keys)
    is_batch_format = (
        parsed and
        isinstance(parsed[0], dict) and
        'id' in parsed[0] and
        'sentence_pairs' in parsed[0]
    )

    if is_batch_format:
        for item in parsed:
            if not isinstance(item, dict):
                continue
            local_id = item.get('id')
            sentence_pairs = item.get('sentence_pairs')
            if local_id in expected_ids and is_valid_sentence_pairs(sentence_pairs):
                results[local_id] = sentence_pairs
    elif len(expected_ids) == 1 and is_valid_sentence_pairs(parsed):
        # Single response format (fallback for batch size 1)
        results[expected_ids[0]] = parsed

    return results


def call_api(user_message: str, api_key: str, max_retries: int = 3) -> Optional[str]:
    """Call NVIDIA Inference API with retry logic for transient failures."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": user_message}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=600)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            # Retry on 502, 503, 504 (gateway/timeout errors)
            if status_code in (502, 503, 504) and attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 30  # 30s, 60s, 120s
                print(f"    HTTP {status_code} error, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            print(f"  HTTP Error: {e}")
            print(f"  Status Code: {status_code}")
            print(f"  Response Body: {e.response.text[:500]}")
            return None
        except requests.exceptions.Timeout as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 30
                print(f"    Timeout, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            print(f"  Timeout Error: {e}")
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


def build_result(
    record: Dict,
    sentence_pairs: Optional[List[Dict]],
    trans_sim: float,
    validation_passed: bool
) -> Dict:
    """Build result dict from record and processing results."""
    result = {
        "oare_id": record.get("oare_id"),
        "transliteration": record.get("transliteration"),
        "genre_label": record.get("genre_label"),
    }
    if sentence_pairs is not None:
        result.update({
            "sentence_pairs": sentence_pairs,
            "split_success": True,
            "num_sentence_pairs": len(sentence_pairs),
            "trans_similarity": round(trans_sim, 4),
            "validation_passed": validation_passed,
        })
    else:
        result.update({
            "sentence_pairs": [],
            "split_success": False,
            "num_sentence_pairs": 0,
            "trans_similarity": 0.0,
            "validation_passed": False,
        })
    return result


def retry_with_backoff(attempt: int, max_retries: int, message: str) -> bool:
    """Print retry message and sleep with exponential backoff. Returns True if should retry."""
    if attempt < max_retries - 1:
        wait_time = (2 ** attempt) * 30
        print(f"    {message}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
        time.sleep(wait_time)
        return True
    return False


def process_single_record(
    record: Dict,
    index: int,
    prompt_template: str,
    api_key: str,
    few_shot_pool: List[Dict],
    max_retries: int = 3
) -> Dict:
    """Process a single record through the API with retry logic."""
    user_message = build_user_message(record, prompt_template, few_shot_pool)
    sentence_pairs = None
    trans_sim = 0.0
    validation_passed = False

    for attempt in range(max_retries):
        response = call_api(user_message, api_key, max_retries=1)
        if response is None:
            if not retry_with_backoff(attempt, max_retries, "API call failed"):
                break
            continue

        sentence_pairs = parse_sentence_pairs(response)
        if sentence_pairs is None:
            if not retry_with_backoff(attempt, max_retries, "Parse failed"):
                break
            continue

        validation_passed, trans_sim = validate_sentence_pairs(record, sentence_pairs)
        if validation_passed:
            break
        if not retry_with_backoff(attempt, max_retries, f"Validation failed (trans={trans_sim:.3f})"):
            break

    return build_result(record, sentence_pairs, trans_sim, validation_passed)


def process_batch(
    records_with_indices: List[Tuple[int, Dict]],
    prompt_template: str,
    api_key: str,
    few_shot_pool: List[Dict],
    max_retries: int = 2
) -> List[Dict]:
    """Process a batch of records through the API."""
    local_records = [(i, record) for i, (_, record) in enumerate(records_with_indices)]
    expected_ids = list(range(len(records_with_indices)))

    user_message = build_batch_message(local_records, prompt_template, few_shot_pool)
    response = call_api(user_message, api_key, max_retries=3)

    # API failed - mark all as failed
    if response is None:
        results = []
        for batch_idx, record in records_with_indices:
            result = build_result(record, None, 0.0, False)
            result["batch_index"] = batch_idx
            results.append(result)
        return results

    parsed_results = parse_batch_response(response, expected_ids)
    results = [None] * len(records_with_indices)
    retry_queue = []

    for local_id, (batch_idx, record) in enumerate(records_with_indices):
        sentence_pairs = parsed_results.get(local_id)
        if sentence_pairs is None:
            retry_queue.append((local_id, batch_idx, record))
            continue

        validation_passed, trans_sim = validate_sentence_pairs(record, sentence_pairs)
        if not validation_passed:
            retry_queue.append((local_id, batch_idx, record))
            continue

        result = build_result(record, sentence_pairs, trans_sim, True)
        result["batch_index"] = batch_idx
        results[local_id] = result

    # Retry failed samples individually
    if retry_queue:
        print(f"    Retrying {len(retry_queue)} failed samples individually...")
        for local_id, batch_idx, record in retry_queue:
            oare_id = record.get('oare_id', 'unknown')[:8]
            print(f"      Retrying {oare_id}...")
            time.sleep(1)

            result = process_single_record(
                record, batch_idx, prompt_template, api_key, few_shot_pool, max_retries=max_retries
            )
            result["batch_index"] = batch_idx
            results[local_id] = result

            if result["split_success"]:
                if result["validation_passed"]:
                    print(f"        -> {result['num_sentence_pairs']} pairs (validated)")
                else:
                    print(f"        -> {result['num_sentence_pairs']} pairs (trans={result['trans_similarity']:.3f})")
            else:
                print(f"        -> FAILED")

    return results


def load_checkpoint(checkpoint_path: str) -> Tuple[List[Dict], set]:
    """Load checkpoint file and return results and completed indices."""
    if not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) == 0:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        return [], set()

    try:
        all_results = []
        completed_indices = set()
        with open(checkpoint_path, 'r') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    all_results.append(result)
                    completed_indices.add(result['batch_index'])
        max_idx = max(completed_indices) if completed_indices else -1
        print(f"Checkpoint: {len(all_results)} processed, max index {max_idx}")
        return all_results, completed_indices
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not read checkpoint file: {e}")
        print("Starting fresh...")
        os.remove(checkpoint_path)
        return [], set()


def save_checkpoint(results: List[Dict], checkpoint_path: str):
    """Append results to checkpoint file."""
    with open(checkpoint_path, 'a') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def write_final_output(all_results: List[Dict], output_path: str, checkpoint_path: str):
    """Write sorted results to output file. Checkpoint is retained as backup."""
    all_results_sorted = sorted(all_results, key=lambda x: x['batch_index'])
    with open(output_path, 'w') as f:
        for result in all_results_sorted:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"Output saved to: {output_path}")
    # Safety: retain checkpoint as backup (do not delete)
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint retained at: {checkpoint_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Split Akkadian transliterations and generate translations from published_texts.csv (V22 with Gemini Pro 3.1)"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=DEFAULT_INPUT_FILE,
        help=f"Input CSV file (default: {DEFAULT_INPUT_FILE})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output JSONL file (default: {DEFAULT_OUTPUT_FILE})"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=DEFAULT_PROMPT_FILE,
        help=f"Prompt template file (default: {DEFAULT_PROMPT_FILE})"
    )
    parser.add_argument(
        "--few-shot-file",
        type=str,
        default=FEW_SHOT_REFERENCE,
        help=f"Few-shot reference file (default: {FEW_SHOT_REFERENCE})"
    )
    parser.add_argument(
        "--keep-oare-ids",
        type=str,
        default=DEFAULT_KEEP_OARE_IDS,
        help=f"File with oare_ids to keep, one per line (default: {DEFAULT_KEEP_OARE_IDS})"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Samples per API call (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--fill-gaps",
        action="store_true",
        help="Retry failed indices from previous run"
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start index (inclusive) for processing subset of data"
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="End index (exclusive) for processing subset of data. Default: process to end"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Check for API key
    api_key = os.environ.get("AKKADIAN_KEY")
    if not api_key:
        print("Error: AKKADIAN_KEY environment variable not set")
        print("Set it with: export AKKADIAN_KEY='your-key-here'")
        return

    print("=" * 70)
    print("V22 SENTENCE SPLITTING + TRANSLATION (GEMINI PRO 3.1, PRE-FILTERED)")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Batch size (samples per API call): {args.batch_size}")
    print(f"Few-shot samples per batch: {FEW_SHOT_SAMPLE_SIZE}")
    print(f"Validation threshold: {SIMILARITY_THRESHOLD}")
    print(f"Prompt: {args.prompt}")
    print(f"Few-shot reference: {args.few_shot_file}")
    print(f"Keep oare_ids: {args.keep_oare_ids}")
    print("=" * 70)

    # Load prompt template
    try:
        prompt_template = load_prompt_template(args.prompt)
        print(f"Prompt template loaded: {args.prompt}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Load few-shot pool
    try:
        few_shot_pool = load_few_shot_pool(args.few_shot_file)
        if len(few_shot_pool) < FEW_SHOT_SAMPLE_SIZE:
            print(f"Warning: Few-shot pool has only {len(few_shot_pool)} examples (requested {FEW_SHOT_SAMPLE_SIZE})")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Load data
    print("\nLoading data...")
    data = load_data(args.input)
    total_before_filter = len(data)

    # Pre-filter by keep oare_ids
    try:
        keep_ids = load_keep_oare_ids(args.keep_oare_ids)
        data = [r for r in data if r.get('oare_id') in keep_ids]
        print(f"Filtered {total_before_filter} -> {len(data)} records ({total_before_filter - len(data)} removed)")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    total_records = len(data)

    # Handle chunk arguments
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else total_records

    # Validate range
    if start_idx >= total_records:
        print(f"Error: start_idx {start_idx} >= total records {total_records}")
        return
    end_idx = min(end_idx, total_records)

    # Set up output paths with chunk suffix if using chunks
    if start_idx > 0 or args.end_idx is not None:
        chunk_suffix = f"_chunk_{start_idx}_{end_idx}"
        base, ext = os.path.splitext(args.output)
        output_filename = f"{base}{chunk_suffix}{ext}"
        checkpoint_filename = f"checkpoint_v22{chunk_suffix}.jsonl"
        print(f"Chunk mode: indices {start_idx} to {end_idx} ({end_idx - start_idx} records)")
    else:
        output_filename = args.output
        checkpoint_filename = DEFAULT_CHECKPOINT_FILE

    output_path = resolve_path(output_filename, DATA_DIR)
    checkpoint_path = os.path.join(str(DATA_DIR), checkpoint_filename)

    # Slice data to chunk range
    data = data[start_idx:end_idx]
    print(f"Processing {len(data)} records (global indices {start_idx} to {end_idx})")

    # Load checkpoint if exists
    all_results, completed_indices = load_checkpoint(checkpoint_path)

    # Build list of indices to process (now relative to chunk)
    chunk_size = len(data)

    # Generate global indices for this chunk
    global_indices = list(range(start_idx, end_idx))

    if args.fill_gaps:
        # Process all missing indices (gaps + remaining)
        indices_to_process = [i for i in global_indices if i not in completed_indices]
    else:
        # Skip gaps, continue from end
        max_completed = max(completed_indices) if completed_indices else start_idx - 1
        indices_to_process = [i for i in global_indices if i > max_completed]

    # Report gaps if any
    if completed_indices:
        max_completed = max(completed_indices)
        expected_indices = set(range(start_idx, max_completed + 1))
        missing_indices = sorted(expected_indices - completed_indices)
        if missing_indices:
            print(f"\nFound {len(missing_indices)} missing indices (gaps from failed records)")
            if args.fill_gaps:
                print("Will fill gaps AND continue to end")
            else:
                print("Skipping gaps (use --fill-gaps to retry them)")

    print(f"Indices to process: {len(indices_to_process)}")

    if len(indices_to_process) == 0:
        print("All records already processed!")
        write_final_output(all_results, output_path, checkpoint_path)
        return

    # Process records in batches
    print(f"\nProcessing {len(indices_to_process)} records in batches of {args.batch_size}...")

    start_time = time.time()
    records_processed = 0
    api_calls = 0

    # Group indices into batches
    for batch_start in range(0, len(indices_to_process), args.batch_size):
        batch_indices = indices_to_process[batch_start:batch_start + args.batch_size]

        # Build list of (batch_index, record) tuples
        # batch_index is global, data is sliced so use local index (global_idx - start_idx)
        records_with_indices = [(idx, data[idx - start_idx]) for idx in batch_indices]

        batch_num = batch_start // args.batch_size + 1
        total_batches = (len(indices_to_process) + args.batch_size - 1) // args.batch_size

        print(f"\n[Batch {batch_num}/{total_batches}] Processing indices {batch_indices[0]}-{batch_indices[-1]} ({len(batch_indices)} samples)...")

        # Process batch
        batch_results = process_batch(
            records_with_indices,
            prompt_template,
            api_key,
            few_shot_pool,
            max_retries=1
        )
        api_calls += 1

        all_results.extend(batch_results)
        records_processed += len(batch_results)

        # Report batch results
        success_count = sum(1 for r in batch_results if r.get('split_success'))
        validated_count = sum(1 for r in batch_results if r.get('validation_passed'))
        total_pairs = sum(r.get('num_sentence_pairs', 0) for r in batch_results)

        print(f"  -> {success_count}/{len(batch_results)} succeeded, {validated_count} validated, {total_pairs} total pairs")

        # Save checkpoint
        save_checkpoint(batch_results, checkpoint_path)

        # Timing stats
        elapsed = time.time() - start_time
        records_per_sec = records_processed / elapsed if elapsed > 0 else 0
        remaining_records = len(indices_to_process) - (batch_start + len(batch_indices))
        eta_seconds = remaining_records / records_per_sec if records_per_sec > 0 else 0
        eta_minutes = eta_seconds / 60

        print(f"  Checkpoint saved: {len(all_results)} total | Rate: {records_per_sec:.2f}/s | ETA: {eta_minutes:.1f}m")

        # Rate limiting - 1 second delay between batches
        if batch_start + args.batch_size < len(indices_to_process):
            time.sleep(1)

    # Write final output
    write_final_output(all_results, output_path, checkpoint_path)

    # Print statistics
    total_elapsed = time.time() - start_time
    success_count = sum(1 for r in all_results if r.get('split_success'))
    validated_count = sum(1 for r in all_results if r.get('validation_passed'))
    total_pairs = sum(r.get('num_sentence_pairs', 0) for r in all_results)
    trans_sims = [r.get('trans_similarity', 0) for r in all_results if r.get('split_success')]
    avg_trans_sim = sum(trans_sims) / len(trans_sims) if trans_sims else 0
    n = len(all_results)

    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"{'='*70}")
    print(f"Total time: {total_elapsed/60:.1f} minutes ({total_elapsed:.0f}s)")
    print(f"API calls: {api_calls}")
    print(f"Total processed: {n}")
    print(f"Successful splits: {success_count} ({100*success_count/n:.1f}%)")
    print(f"Validation passed: {validated_count} ({100*validated_count/n:.1f}%)")
    print(f"Failed splits: {n - success_count} ({100*(n - success_count)/n:.1f}%)")
    print(f"Total sentence pairs: {total_pairs}")
    print(f"Avg pairs per record: {total_pairs/n:.1f}")
    print(f"Avg trans similarity: {avg_trans_sim:.4f}")


if __name__ == "__main__":
    main()
