"""
Repair expert translations using NVIDIA Inference API (Gemini 3 Pro) with thinking mode.

V16: Reads v2 competition data (train.csv + Sentences_Oare) directly, combines/dedupes
in-memory. Enables thinking_config with thinking_level=high and saves reasoning traces.

Usage:
    python repair_expert_translations_v16.py
    python repair_expert_translations_v16.py --batch-size 8
"""

import argparse
from collections import defaultdict
import pandas as pd
import requests
import json
import os
import re
import sys
import time
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
TEMPERATURE = 0.3  # Lower temperature for more consistent repairs
MAX_TOKENS = 65536

# Default batch configuration
DEFAULT_SAMPLES_PER_BATCH = 8

# Input files (v2 competition data)
TRAIN_CSV = DATA_DIR / "deep-past-initiative-machine-translation_v2" / "train.csv"
SENTENCES_OARE_CSV = DATA_DIR / "sentences_oare_expert_pairs.csv"

# Output files
DEFAULT_OUTPUT_FILE = "expert_translations_repaired_v16.jsonl"
DEFAULT_CHECKPOINT_FILE = "expert_translations_repaired_v16_checkpoint.jsonl"
DEFAULT_PROMPT_FILE = "prompt_repair_translations_a.txt"


def load_prompt_template(prompt_file: str = None) -> str:
    """Load the repair prompt template."""
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


def load_data() -> pd.DataFrame:
    """Load and combine expert translations from v2 train.csv and Sentences_Oare CSV.

    Same logic as extract_expert_published_texts.py but reads v2 data directly.
    """
    train = pd.read_csv(TRAIN_CSV)
    sentences = pd.read_csv(SENTENCES_OARE_CSV)

    # Get train expert pairs (use train's translation)
    train_expert = train[['oare_id', 'transliteration', 'translation']].copy()
    train_expert['source'] = 'train'

    # Get Sentences_Oare pairs NOT in train
    train_ids = set(train['oare_id'].dropna())
    sentences_new = sentences[~sentences['oare_id'].isin(train_ids)].copy()
    sentences_new = sentences_new.drop_duplicates(subset=['oare_id'], keep='first')
    sentences_new['source'] = 'sentences_oare'

    # Combine
    output = pd.concat([train_expert, sentences_new], ignore_index=True)

    print(f"Loaded {len(output)} expert translations")
    print(f"  - From v2/train.csv: {len(train_expert)}")
    print(f"  - From Sentences_Oare (new): {len(sentences_new)}")
    return output


def build_batch_request(samples_batch: pd.DataFrame, start_index: int) -> str:
    """Build user message with batch of samples to review."""
    batch_data = []
    for i, (_, row) in enumerate(samples_batch.iterrows()):
        item = {
            "oare_id": row['oare_id'],
            "transliteration": row['transliteration'],
            "translation": row['translation']
        }
        batch_data.append(item)

    return f"""Review these {len(batch_data)} expert translations and repair any errors.
Return a JSON array with 'oare_id', 'find', 'replace', and 'edit' for each item needing repair.

{json.dumps(batch_data, ensure_ascii=False, indent=2)}"""


def parse_batch_response(response: str, expected_count: int) -> List[Dict]:
    """Parse JSON array from response."""
    results = []

    # Try to extract JSON array from response
    try:
        match = re.search(r'\[[\s\S]*\]', response)
        if match:
            results = json.loads(match.group())
    except json.JSONDecodeError:
        pass

    # Fallback: try to parse individual JSON objects
    if not results:
        obj_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(obj_pattern, response)
        for match in matches:
            try:
                obj = json.loads(match)
                if 'oare_id' in obj and 'find' in obj and 'replace' in obj:
                    results.append(obj)
            except:
                continue

    return results


def call_api(messages: List[Dict], api_key: str, max_retries: int = 3) -> Tuple[Optional[str], str]:
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
        "messages": messages,
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
                print(f"  HTTP {status_code} error, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            print(f"HTTP Error: {e}")
            print(f"Status Code: {status_code}")
            print(f"Response Body: {e.response.text[:500]}")
            return None, ""
        except requests.exceptions.Timeout as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 30
                print(f"  Timeout, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            print(f"Timeout Error: {e}")
            return None, ""
        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
            return None, ""
        except KeyError as e:
            print(f"Response Parse Error: Missing key {e}")
            return None, ""
        except Exception as e:
            print(f"Unexpected Error: {type(e).__name__}: {e}")
            return None, ""
    return None, ""


def repair_batch_with_indices(
    samples_batch: pd.DataFrame,
    batch_indices: List[int],
    system_message: str,
    api_key: str
) -> List[Dict]:
    """Review and repair a batch of translations in one API call.

    Args:
        samples_batch: DataFrame of samples to process
        batch_indices: List of original indices for each sample (for batch_index field)
        system_message: System prompt
        api_key: API key
    """

    user_message = build_batch_request(samples_batch, 0)  # start_index unused now

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    print(f"  Calling API for {len(samples_batch)} samples...")
    response, reasoning = call_api(messages, api_key)

    if not response:
        print("  API call failed!")
        return []

    # Parse response
    repairs = parse_batch_response(response, len(samples_batch))
    print(f"  Parsed {len(repairs)} repairs from response")

    # Group repairs by oare_id (multiple fixes possible per translation)
    repairs_by_id = defaultdict(list)
    for r in repairs:
        if 'oare_id' in r:
            repairs_by_id[r['oare_id']].append(r)

    results = []
    for i, (_, row) in enumerate(samples_batch.iterrows()):
        oare_id = row['oare_id']
        original_index = batch_indices[i]
        original_translation = row['translation']

        warning = False
        warning_msg = []

        find_replace_ops = []

        if oare_id in repairs_by_id:
            # Apply all find/replace operations for this translation
            corrected = original_translation
            edits = []

            for repair in repairs_by_id[oare_id]:
                find_text = repair.get('find', '')
                replace_text = repair.get('replace', '')
                edit_desc = repair.get('edit', '')

                if not find_text:
                    continue

                # Store the operation
                find_replace_ops.append({
                    'find': find_text,
                    'replace': replace_text,
                    'edit': edit_desc
                })

                # Count occurrences of find_text
                match_count = corrected.count(find_text)

                if match_count == 0:
                    # Find text not found - skip this one, flag warning
                    warning = True
                    warning_msg.append(f"'find' not found: {find_text[:50]}")
                    print(f"    WARNING: 'find' text not found in {oare_id[:8]}...")
                elif match_count > 1:
                    # Multiple matches - skip this one, flag warning
                    warning = True
                    warning_msg.append(f"'find' has {match_count} matches: {find_text[:50]}")
                    print(f"    WARNING: 'find' text has {match_count} matches in {oare_id[:8]}...")
                else:
                    # Exactly one match - safe to apply
                    corrected = corrected.replace(find_text, replace_text, 1)
                    edits.append(edit_desc)

            # Apply successful corrections, include warnings in edit field
            corrected_translation = corrected
            all_messages = edits + warning_msg
            edit = '; '.join(all_messages) if all_messages else ''
        else:
            # No repairs for this item - copy original
            corrected_translation = original_translation
            edit = ''

        result = {
            'oare_id': oare_id,
            'transliteration': row['transliteration'],
            'translation': original_translation,
            'corrected_translation': corrected_translation,
            'edit': edit,
            'warning': warning,
            'find_replace': find_replace_ops if find_replace_ops else None,
            'source': row.get('source', ''),
            'batch_index': original_index,
            'reasoning': reasoning,
        }
        results.append(result)

    return results


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Repair expert translations using NVIDIA Inference API (Gemini 3 Pro) with thinking mode"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSONL file (default: expert_translations_repaired_v16.jsonl)"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=DEFAULT_PROMPT_FILE,
        help=f"Prompt file to use (default: {DEFAULT_PROMPT_FILE})"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=DEFAULT_SAMPLES_PER_BATCH,
        help=f"Samples per API call (default: {DEFAULT_SAMPLES_PER_BATCH})"
    )
    parser.add_argument(
        "--fill-gaps",
        action="store_true",
        help="Fill gaps from failed batches AND continue to end"
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
    print("EXPERT TRANSLATION REPAIR V16 (v2 data + thinking high)")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Batch size: {args.batch_size}")
    print(f"Prompt: {args.prompt}")
    print(f"Thinking: high")
    if args.start_idx is not None or args.end_idx is not None:
        print(f"Chunk: [{args.start_idx} : {args.end_idx}]")
    print("=" * 70)

    # Load data (inline extract from v2 train.csv + Sentences_Oare)
    print("\nLoading data...")
    samples = load_data()

    # Load prompt template
    try:
        system_message = load_prompt_template(args.prompt)
        print(f"Prompt template loaded: {args.prompt}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Set up output paths
    if args.output:
        output_file = args.output
    else:
        output_file = DEFAULT_OUTPUT_FILE

    if not os.path.isabs(output_file):
        output_path = os.path.join(str(DATA_DIR), output_file)
    else:
        output_path = output_file

    if args.checkpoint:
        ckpt_file = args.checkpoint
        if not os.path.isabs(ckpt_file):
            checkpoint_path = os.path.join(str(DATA_DIR), ckpt_file)
        else:
            checkpoint_path = ckpt_file
    else:
        checkpoint_path = os.path.join(str(DATA_DIR), DEFAULT_CHECKPOINT_FILE)

    # Load checkpoint if exists
    all_results = []
    completed_indices = set()

    if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 0:
        try:
            with open(checkpoint_path, 'r') as f:
                for line in f:
                    if line.strip():
                        all_results.append(json.loads(line))
            existing = pd.DataFrame(all_results)
            completed_indices = set(existing['batch_index'])
            max_completed = max(completed_indices) if completed_indices else -1

            # Find gaps in batch_index (failed batches)
            expected_indices = set(range(max_completed + 1))
            missing_indices = sorted(expected_indices - completed_indices)

            if missing_indices:
                print(f"\nFound {len(missing_indices)} missing indices (gaps from failed batches)")
                if args.fill_gaps:
                    print("Will fill gaps AND continue to end")
                else:
                    print("Skipping gaps (use --fill-gaps to retry them)")

            print(f"Checkpoint: {len(existing)} processed, max index {max_completed}")
        except Exception as e:
            print(f"Warning: Could not read checkpoint file: {e}")
            print("Starting fresh...")
            all_results = []
            completed_indices = set()
            os.remove(checkpoint_path)
    else:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    # Build list of indices to process
    total_samples = len(samples)
    chunk_start = args.start_idx if args.start_idx is not None else 0
    chunk_end = args.end_idx if args.end_idx is not None else total_samples

    if args.fill_gaps:
        # Process all missing indices (gaps + remaining) within chunk range
        indices_to_process = [i for i in range(chunk_start, chunk_end) if i not in completed_indices]
    else:
        # Skip gaps, continue from end within chunk range
        max_completed = max(completed_indices) if completed_indices else chunk_start - 1
        indices_to_process = list(range(max(max_completed + 1, chunk_start), chunk_end))

    print(f"Indices to process: {len(indices_to_process)}")

    if len(indices_to_process) == 0:
        print("All samples already processed!")
        # Write final output
        with open(output_path, 'w') as f:
            for result in all_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"Output saved to: {output_path}")
        return

    # Process in batches
    batch_size = args.batch_size
    num_batches = (len(indices_to_process) + batch_size - 1) // batch_size

    print(f"\nProcessing {num_batches} batches ({batch_size} samples each)...")

    start_time = time.time()
    samples_processed = 0

    for batch_idx in range(num_batches):
        batch_start_time = time.time()
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(indices_to_process))
        batch_indices = indices_to_process[batch_start:batch_end]
        batch = samples.iloc[batch_indices]

        print(f"\nBatch {batch_idx + 1}/{num_batches} (indices {batch_indices[0]}-{batch_indices[-1]})")

        results = repair_batch_with_indices(batch, batch_indices, system_message, api_key)
        all_results.extend(results)
        samples_processed += len(results)

        # Save checkpoint after each batch
        with open(checkpoint_path, 'a') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        # Timing stats
        batch_time = time.time() - batch_start_time
        elapsed = time.time() - start_time
        samples_per_sec = samples_processed / elapsed if elapsed > 0 else 0
        remaining_samples = len(indices_to_process) - (batch_idx + 1) * batch_size
        eta_seconds = remaining_samples / samples_per_sec if samples_per_sec > 0 else 0
        eta_minutes = eta_seconds / 60

        # Count edits and warnings in this batch
        edits_in_batch = sum(1 for r in results if r.get('edit') and not r.get('warning'))
        warnings_in_batch = sum(1 for r in results if r.get('warning'))

        print(f"  Saved: {len(all_results)} total | Edits: {edits_in_batch} | Warnings: {warnings_in_batch} | "
              f"Batch: {batch_time:.1f}s | Rate: {samples_per_sec:.2f}/s | ETA: {eta_minutes:.1f}m")

        if batch_idx < num_batches - 1:
            time.sleep(2)

    # Safety check: output and checkpoint must differ
    if os.path.realpath(output_path) == os.path.realpath(checkpoint_path):
        print(f"ERROR: output_path and checkpoint_path are the same file: {output_path}")
        print("Aborting final write to prevent data loss. Results are safe in checkpoint.")
        return

    # Write final output
    with open(output_path, 'w') as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    total_elapsed = time.time() - start_time
    total_minutes = total_elapsed / 60

    # Count total edits and warnings
    total_edits = sum(1 for r in all_results if r.get('edit') and not r.get('warning'))
    total_warnings = sum(1 for r in all_results if r.get('warning'))

    print(f"\n{'='*60}")
    print(f"COMPLETE!")
    print(f"{'='*60}")
    print(f"Total time: {total_minutes:.1f} minutes ({total_elapsed:.0f}s)")
    print(f"Total processed: {len(all_results)}")
    print(f"Total edits applied: {total_edits} ({100*total_edits/len(all_results):.1f}%)")
    print(f"Total warnings (not applied): {total_warnings} ({100*total_warnings/len(all_results):.1f}%)")
    print(f"Output saved to: {output_path}")

    # Keep checkpoint file (do NOT delete — see CLAUDE.md data safety rules)
    print(f"Checkpoint kept at: {checkpoint_path}")


if __name__ == "__main__":
    main()
