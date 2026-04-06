"""
Split Akkadian transliteration/translation pairs into aligned sentence pairs.

V16: Uses v16 repaired data, enables thinking_config with thinking_level=high,
and saves reasoning traces in output.

Features:
- Batched API calls (8 samples per call) to save input tokens
- Edit distance validation with retry logic
- Checkpointing for resumability
- Thinking mode with reasoning trace capture

Usage:
    python split_expert_sentences_v16.py
    python split_expert_sentences_v16.py --batch-size 8
    python split_expert_sentences_v16.py --fill-gaps
"""

import argparse
import json
import os
import re
import sys
import time
import requests
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
MODEL = "gcp/google/gemini-3-pro-preview"
TEMPERATURE = 0.7
MAX_TOKENS = 65536

# Default files
DEFAULT_INPUT_FILE = "expert_translations_repaired_v16.jsonl"
DEFAULT_OUTPUT_FILE = "expert_translations_repaired_sentence_output_v16.jsonl"
DEFAULT_CHECKPOINT_FILE = "expert_translations_repaired_sentence_v16_checkpoint.jsonl"
DEFAULT_PROMPT_FILE = "prompt_v08_sentence_split.txt"

# Default batch size for API calls (8 samples per call)
DEFAULT_BATCH_SIZE = 8

# Validation threshold
SIMILARITY_THRESHOLD = 0.9


def load_prompt_template(prompt_file: str = None) -> str:
    """Load the sentence splitting prompt template."""
    if prompt_file is None:
        prompt_file = DEFAULT_PROMPT_FILE

    # Check if it's just a filename or a full path
    if not os.path.isabs(prompt_file) and not os.path.exists(prompt_file):
        prompt_path = os.path.join(str(PROMPTS_DIR), prompt_file)
    else:
        prompt_path = prompt_file

    if os.path.exists(prompt_path):
        with open(prompt_path, 'r') as f:
            return f.read()
    else:
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")


def load_data(input_file: str = None) -> List[Dict]:
    """Load input data from JSONL file."""
    if input_file is None:
        input_file = DEFAULT_INPUT_FILE

    if not os.path.isabs(input_file) and not os.path.exists(input_file):
        input_path = os.path.join(str(DATA_DIR), input_file)
    else:
        input_path = input_file

    data = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print(f"Loaded {len(data)} records from {input_path}")
    return data


def compute_similarity(original: str, reconstructed: str) -> float:
    """Compute similarity ratio using SequenceMatcher (0.0 to 1.0)."""
    return SequenceMatcher(None, original, reconstructed).ratio()


def validate_sentence_pairs(record: Dict, pairs: List[Dict], threshold: float = SIMILARITY_THRESHOLD) -> Tuple[bool, float, float]:
    """Validate that joined pairs match original within threshold.

    Returns: (is_valid, trans_similarity, transl_similarity)
    """
    # Concatenate transliterations (join with space)
    reconstructed_trans = ' '.join(p['transliteration'] for p in pairs if p.get('transliteration'))
    trans_sim = compute_similarity(record['transliteration'], reconstructed_trans)

    # Concatenate translations
    reconstructed_transl = ' '.join(p['translation'] for p in pairs if p.get('translation'))
    transl_sim = compute_similarity(record['corrected_translation'], reconstructed_transl)

    is_valid = trans_sim >= threshold and transl_sim >= threshold
    return is_valid, trans_sim, transl_sim


def build_user_message(record: Dict, prompt_template: str) -> str:
    """Build user message by inserting transliteration/translation into prompt template (single sample)."""
    # Build minimal dict to expose to the model
    pair_data = {
        "transliteration": record["transliteration"],
        "translation": record["corrected_translation"]
    }

    # Replace placeholder in prompt template
    user_message = prompt_template.replace(
        "<INSERT_TRANLITERATION_TRANSLATION_PAIR_HERE>",
        json.dumps(pair_data, ensure_ascii=False, indent=2)
    )

    return user_message


def build_batch_message(records: List[Tuple[int, Dict]], prompt_template: str) -> str:
    """Build user message with array of samples for batched API call.

    Args:
        records: List of (local_id, record) tuples
        prompt_template: The prompt template with placeholder

    Returns:
        User message with JSON array of samples
    """
    # Build array of samples with local IDs (0 to N-1)
    batch_data = []
    for local_id, record in records:
        batch_data.append({
            "id": local_id,
            "transliteration": record["transliteration"],
            "translation": record["corrected_translation"]
        })

    # Replace placeholder in prompt template
    user_message = prompt_template.replace(
        "<INSERT_TRANLITERATION_TRANSLATION_PAIR_HERE>",
        json.dumps(batch_data, ensure_ascii=False, indent=2)
    )

    return user_message


def parse_sentence_pairs(response: str) -> Optional[List[Dict]]:
    """Extract JSON array of sentence pairs from API response (single sample).

    Returns None if parsing fails (triggers retry).
    """
    # Try to extract JSON array from response
    try:
        match = re.search(r'\[[\s\S]*\]', response)
        if match:
            pairs = json.loads(match.group())
            # Validate it's a list
            if isinstance(pairs, list):
                # Validate each item has required fields
                for pair in pairs:
                    if not isinstance(pair, dict):
                        return None
                    if "transliteration" not in pair or "translation" not in pair:
                        return None
                return pairs
    except json.JSONDecodeError:
        pass

    return None


def parse_batch_response(response: str, expected_ids: List[int]) -> Dict[int, Optional[List[Dict]]]:
    """Parse batched API response, matching results to input IDs.

    Args:
        response: Raw API response text
        expected_ids: List of expected local IDs (0 to N-1)

    Returns:
        Dict mapping local_id -> sentence_pairs (or None if parsing failed)
    """
    results = {id_: None for id_ in expected_ids}

    try:
        # Try to extract JSON array from response
        match = re.search(r'\[[\s\S]*\]', response)
        if not match:
            return results

        parsed = json.loads(match.group())
        if not isinstance(parsed, list):
            return results

        # Check if it's a batch response (has 'id' and 'sentence_pairs' keys)
        # or a single response (list of sentence pairs directly)
        if parsed and isinstance(parsed[0], dict) and 'id' in parsed[0] and 'sentence_pairs' in parsed[0]:
            # Batch response format
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                local_id = item.get('id')
                sentence_pairs = item.get('sentence_pairs')

                if local_id is not None and local_id in expected_ids:
                    # Validate sentence_pairs
                    if isinstance(sentence_pairs, list):
                        valid = True
                        for pair in sentence_pairs:
                            if not isinstance(pair, dict):
                                valid = False
                                break
                            if "transliteration" not in pair or "translation" not in pair:
                                valid = False
                                break
                        if valid:
                            results[local_id] = sentence_pairs
        else:
            # Single response format (fallback for batch size 1)
            if len(expected_ids) == 1:
                valid = True
                for pair in parsed:
                    if not isinstance(pair, dict):
                        valid = False
                        break
                    if "transliteration" not in pair or "translation" not in pair:
                        valid = False
                        break
                if valid:
                    results[expected_ids[0]] = parsed

    except json.JSONDecodeError:
        pass

    return results


def call_api(user_message: str, api_key: str, max_retries: int = 3) -> Tuple[Optional[str], str]:
    """Call NVIDIA Inference API with thinking mode enabled.

    Returns:
        (content, reasoning) tuple. On failure, returns (None, "").
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": user_message}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "thinking_config": {
            "thinking_level": "low",
            "include_thoughts": True,
        },
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=600)
            response.raise_for_status()
            message = response.json()["choices"][0]["message"]
            reasoning = message.get("reasoning_content", "") or ""
            content = message["content"]
            return content, reasoning
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            # Retry on 502, 503, 504 (gateway/timeout errors)
            if status_code in (429, 502, 503, 504) and attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 30  # 30s, 60s, 120s
                print(f"    HTTP {status_code} error, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            print(f"  HTTP Error: {e}")
            print(f"  Status Code: {status_code}")
            print(f"  Response Body: {e.response.text[:500]}")
            return None, ""
        except requests.exceptions.Timeout as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 30
                print(f"    Timeout, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            print(f"  Timeout Error: {e}")
            return None, ""
        except requests.exceptions.RequestException as e:
            print(f"  Request Error: {e}")
            return None, ""
        except KeyError as e:
            print(f"  Response Parse Error: Missing key {e}")
            return None, ""
        except Exception as e:
            print(f"  Unexpected Error: {type(e).__name__}: {e}")
            return None, ""
    return None, ""


def process_single_record(
    record: Dict,
    index: int,
    prompt_template: str,
    api_key: str,
    max_retries: int = 3
) -> Dict:
    """Process a single record through the API (for individual retries).

    Retries on both HTTP errors and parse failures.
    """
    user_message = build_user_message(record, prompt_template)

    sentence_pairs = None
    trans_sim = 0.0
    transl_sim = 0.0
    validation_passed = False
    reasoning = ""

    for attempt in range(max_retries):
        response, attempt_reasoning = call_api(user_message, api_key, max_retries=1)

        if response is None:
            # HTTP error - will be retried by outer loop
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 30
                print(f"    API call failed, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            break

        reasoning = attempt_reasoning

        # Try to parse response
        sentence_pairs = parse_sentence_pairs(response)

        if sentence_pairs is None:
            # Parse failure - retry
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 30
                print(f"    Parse failed, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            break

        # Validate with edit distance
        validation_passed, trans_sim, transl_sim = validate_sentence_pairs(record, sentence_pairs)

        if validation_passed:
            break

        # Validation failed - retry
        if attempt < max_retries - 1:
            wait_time = (2 ** attempt) * 30
            print(f"    Validation failed (trans={trans_sim:.3f}, transl={transl_sim:.3f}), retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait_time)
            continue

    # Build result
    result = dict(record)  # Copy all input fields

    if sentence_pairs is not None:
        result["sentence_pairs"] = sentence_pairs
        result["split_success"] = True
        result["num_sentence_pairs"] = len(sentence_pairs)
        result["trans_similarity"] = round(trans_sim, 4)
        result["transl_similarity"] = round(transl_sim, 4)
        result["validation_passed"] = validation_passed
    else:
        result["sentence_pairs"] = []
        result["split_success"] = False
        result["num_sentence_pairs"] = 0
        result["trans_similarity"] = 0.0
        result["transl_similarity"] = 0.0
        result["validation_passed"] = False

    result["reasoning"] = reasoning
    return result


def process_batch(
    records_with_indices: List[Tuple[int, Dict]],
    prompt_template: str,
    api_key: str,
    max_retries: int = 2
) -> List[Dict]:
    """Process a batch of records through the API.

    Args:
        records_with_indices: List of (batch_index, record) tuples
        prompt_template: The prompt template
        api_key: API key
        max_retries: Max retries for individual validation failures

    Returns:
        List of result dicts (in same order as input)
    """
    # Build batch message with local IDs (0 to N-1)
    local_records = [(i, record) for i, (_, record) in enumerate(records_with_indices)]
    expected_ids = [i for i in range(len(records_with_indices))]

    user_message = build_batch_message(local_records, prompt_template)

    # Call API
    response, reasoning = call_api(user_message, api_key, max_retries=3)

    if response is None:
        # API failed - mark all as failed
        results = []
        for batch_idx, record in records_with_indices:
            result = dict(record)
            result["sentence_pairs"] = []
            result["split_success"] = False
            result["num_sentence_pairs"] = 0
            result["trans_similarity"] = 0.0
            result["transl_similarity"] = 0.0
            result["validation_passed"] = False
            result["reasoning"] = ""
            results.append(result)
        return results

    # Parse batch response
    parsed_results = parse_batch_response(response, expected_ids)

    # Build results and collect validation failures for retry
    results = [None] * len(records_with_indices)
    retry_queue = []  # List of (local_id, batch_idx, record) for retries

    for local_id, (batch_idx, record) in enumerate(records_with_indices):
        sentence_pairs = parsed_results.get(local_id)

        if sentence_pairs is None:
            # Parse failed - add to retry queue
            retry_queue.append((local_id, batch_idx, record))
            continue

        # Validate with edit distance
        validation_passed, trans_sim, transl_sim = validate_sentence_pairs(record, sentence_pairs)

        if not validation_passed:
            # Validation failed - add to retry queue
            retry_queue.append((local_id, batch_idx, record))
            continue

        # Success
        result = dict(record)
        result["sentence_pairs"] = sentence_pairs
        result["split_success"] = True
        result["num_sentence_pairs"] = len(sentence_pairs)
        result["trans_similarity"] = round(trans_sim, 4)
        result["transl_similarity"] = round(transl_sim, 4)
        result["validation_passed"] = True
        result["reasoning"] = reasoning
        results[local_id] = result

    # Retry failed samples individually
    if retry_queue:
        print(f"    Retrying {len(retry_queue)} failed samples individually...")
        for local_id, batch_idx, record in retry_queue:
            oare_id = record.get('oare_id', 'unknown')[:8]
            print(f"      Retrying {oare_id}...")

            # Rate limiting between retries
            time.sleep(1)

            result = process_single_record(
                record,
                batch_idx,
                prompt_template,
                api_key,
                max_retries=max_retries
            )
            results[local_id] = result

            if result["split_success"]:
                if result["validation_passed"]:
                    print(f"        -> {result['num_sentence_pairs']} pairs (validated)")
                else:
                    print(f"        -> {result['num_sentence_pairs']} pairs (trans={result['trans_similarity']:.3f}, transl={result['transl_similarity']:.3f})")
            else:
                print(f"        -> FAILED")

    return results


def load_checkpoint(checkpoint_path: str) -> Tuple[List[Dict], set]:
    """Load checkpoint file and return results and completed indices."""
    all_results = []
    completed_indices = set()

    if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 0:
        try:
            with open(checkpoint_path, 'r') as f:
                for line in f:
                    if line.strip():
                        result = json.loads(line)
                        all_results.append(result)
                        completed_indices.add(result['batch_index'])
            print(f"Checkpoint: {len(all_results)} processed, max index {max(completed_indices) if completed_indices else -1}")
        except Exception as e:
            print(f"Warning: Could not read checkpoint file: {e}")
            print("Starting fresh...")
            all_results = []
            completed_indices = set()
            os.remove(checkpoint_path)
    else:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    return all_results, completed_indices


def save_checkpoint(results: List[Dict], checkpoint_path: str):
    """Append results to checkpoint file."""
    with open(checkpoint_path, 'a') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Split Akkadian transliteration/translation pairs into sentence pairs (V16 + thinking)"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=DEFAULT_INPUT_FILE,
        help=f"Input JSONL file (default: {DEFAULT_INPUT_FILE})"
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
        default=None,
        help="Start index (inclusive) for parallel chunking"
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="End index (exclusive) for parallel chunking"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint file (default: auto-derived, use for parallel chunks)"
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
    print("SENTENCE SPLITTING V16 (v2 data + thinking high)")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Batch size (samples per API call): {args.batch_size}")
    print(f"Validation threshold: {SIMILARITY_THRESHOLD}")
    print(f"Prompt: {args.prompt}")
    print(f"Thinking: low")
    if args.start_idx is not None or args.end_idx is not None:
        print(f"Chunk: [{args.start_idx} : {args.end_idx}]")
    print("=" * 70)

    # Load prompt template
    try:
        prompt_template = load_prompt_template(args.prompt)
        print(f"Prompt template loaded: {args.prompt}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Load data
    print("\nLoading data...")
    data = load_data(args.input)

    # Set up output paths
    if not os.path.isabs(args.output):
        output_path = os.path.join(str(DATA_DIR), args.output)
    else:
        output_path = args.output

    if args.checkpoint:
        ckpt_file = args.checkpoint
        if not os.path.isabs(ckpt_file):
            checkpoint_path = os.path.join(str(DATA_DIR), ckpt_file)
        else:
            checkpoint_path = ckpt_file
    else:
        checkpoint_path = os.path.join(str(DATA_DIR), DEFAULT_CHECKPOINT_FILE)

    # Load checkpoint if exists
    all_results, completed_indices = load_checkpoint(checkpoint_path)

    # Build list of indices to process
    total_records = len(data)
    chunk_start = args.start_idx if args.start_idx is not None else 0
    chunk_end = args.end_idx if args.end_idx is not None else total_records

    if args.fill_gaps:
        # Process all missing indices (gaps + remaining) within chunk range
        indices_to_process = [i for i in range(chunk_start, chunk_end) if i not in completed_indices]
    else:
        # Skip gaps, continue from end within chunk range
        max_completed = max(completed_indices) if completed_indices else chunk_start - 1
        indices_to_process = list(range(max(max_completed + 1, chunk_start), chunk_end))

    # Report gaps if any
    if completed_indices:
        max_completed = max(completed_indices)
        expected_indices = set(range(max_completed + 1))
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
        # Write final output
        all_results_sorted = sorted(all_results, key=lambda x: x['batch_index'])
        with open(output_path, 'w') as f:
            for result in all_results_sorted:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"Output saved to: {output_path}")
        # Keep checkpoint file (do NOT delete — see CLAUDE.md data safety rules)
        print(f"Checkpoint kept at: {checkpoint_path}")
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
        records_with_indices = [(idx, data[idx]) for idx in batch_indices]

        batch_num = batch_start // args.batch_size + 1
        total_batches = (len(indices_to_process) + args.batch_size - 1) // args.batch_size

        print(f"\n[Batch {batch_num}/{total_batches}] Processing indices {batch_indices[0]}-{batch_indices[-1]} ({len(batch_indices)} samples)...")

        # Process batch
        batch_results = process_batch(
            records_with_indices,
            prompt_template,
            api_key,
            max_retries=1
        )
        api_calls += 1

        # Save results
        for result in batch_results:
            all_results.append(result)
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

    # Safety check: output and checkpoint must differ
    if os.path.realpath(output_path) == os.path.realpath(checkpoint_path):
        print(f"ERROR: output_path and checkpoint_path are the same file: {output_path}")
        print("Aborting final write to prevent data loss. Results are safe in checkpoint.")
        return

    # Write final output (sorted by batch_index)
    all_results_sorted = sorted(all_results, key=lambda x: x['batch_index'])
    with open(output_path, 'w') as f:
        for result in all_results_sorted:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    total_elapsed = time.time() - start_time
    total_minutes = total_elapsed / 60

    # Count statistics
    success_count = sum(1 for r in all_results if r.get('split_success'))
    fail_count = len(all_results) - success_count
    validated_count = sum(1 for r in all_results if r.get('validation_passed'))
    total_pairs = sum(r.get('num_sentence_pairs', 0) for r in all_results)

    # Similarity stats
    trans_sims = [r.get('trans_similarity', 0) for r in all_results if r.get('split_success')]
    transl_sims = [r.get('transl_similarity', 0) for r in all_results if r.get('split_success')]
    avg_trans_sim = sum(trans_sims) / len(trans_sims) if trans_sims else 0
    avg_transl_sim = sum(transl_sims) / len(transl_sims) if transl_sims else 0

    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"{'='*70}")
    print(f"Total time: {total_minutes:.1f} minutes ({total_elapsed:.0f}s)")
    print(f"API calls: {api_calls}")
    print(f"Total processed: {len(all_results)}")
    print(f"Successful splits: {success_count} ({100*success_count/len(all_results):.1f}%)")
    print(f"Validation passed: {validated_count} ({100*validated_count/len(all_results):.1f}%)")
    print(f"Failed splits: {fail_count} ({100*fail_count/len(all_results):.1f}%)")
    print(f"Total sentence pairs: {total_pairs}")
    print(f"Avg pairs per record: {total_pairs/len(all_results):.1f}")
    print(f"Avg trans similarity: {avg_trans_sim:.4f}")
    print(f"Avg transl similarity: {avg_transl_sim:.4f}")
    print(f"Output saved to: {output_path}")

    # Keep checkpoint file (do NOT delete — see CLAUDE.md data safety rules)
    print(f"Checkpoint kept at: {checkpoint_path}")


if __name__ == "__main__":
    main()
