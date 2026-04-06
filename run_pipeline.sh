#!/bin/bash
set -euo pipefail

# =============================================================================
# Akkadian Translation Pipeline — 3rd Place Solution
# Runs all extraction and preparation stages sequentially.
# To parallelize, use --shard K/N args on individual scripts.
# =============================================================================

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
export DATA_DIR="${DATA_DIR:-$REPO_DIR/data}"
SCRIPTS_E="$REPO_DIR/scripts/extraction"
SCRIPTS_P="$REPO_DIR/scripts/preparation"

# Load API key
if [ -f "$REPO_DIR/.env" ]; then
    source "$REPO_DIR/.env"
fi
if [ -z "${AKKADIAN_KEY:-}" ]; then
    echo "Error: AKKADIAN_KEY not set. Copy .env.example to .env and add your key."
    exit 1
fi
export AKKADIAN_KEY

echo "=== Pipeline Configuration ==="
echo "DATA_DIR: $DATA_DIR"
echo "REPO_DIR: $REPO_DIR"
echo ""

# =============================================================================
# Stage 1: Expert data (competition CSVs → expert_v19_dedup.jsonl)
# =============================================================================

echo "=== Stage 1a: Combine expert pairs from competition CSVs ==="
python3 -u "$SCRIPTS_E/extract_expert_published_texts.py"

echo "=== Stage 1b: Repair expert translations (Gemini API, ~5h) ==="
python3 -u "$SCRIPTS_E/repair_expert_translations_v16.py" \
    --batch-size 8
# To parallelize: run 4 shards with --start-idx/--end-idx, then --fill-gaps

echo "=== Stage 1c: Split expert sentences (Gemini API, ~1h) ==="
python3 -u "$SCRIPTS_E/split_expert_sentences_v16.py" \
    --batch-size 8
# To parallelize: run 4 shards with --start-idx/--end-idx, then --fill-gaps

echo "=== Stage 1d: Dedup expert ==="
python3 -u "$SCRIPTS_P/dedup_expert_v19.py"

# =============================================================================
# Stage 2: AKT PDF extraction (PDFs → AKT v24 JSONLs)
# =============================================================================

echo "=== Stage 2a: AKT side_by_side extraction (Gemini API) ==="
python3 -u "$SCRIPTS_E/extract_akt_pairs_v24.py" --mode side_by_side
# To parallelize: --shard 0/4, --shard 1/4, etc.

echo "=== Stage 2b: AKT top_bottom extraction (Gemini API) ==="
python3 -u "$SCRIPTS_E/extract_akt_pairs_v24.py" --mode top_bottom

echo "=== Stage 2c: AKT OCR extraction (Gemini API) ==="
python3 -u "$SCRIPTS_E/extract_akt_pairs_v24.py" --mode ocr

# =============================================================================
# Stage 3: CAD extraction (PDFs → cad_normalized.jsonl)
# =============================================================================

echo "=== Stage 3a: CAD PDF extraction (Gemini API) ==="
python3 -u "$SCRIPTS_E/extract_cad_pairs_v20.py"

echo "=== Stage 3b: CAD normalization ==="
python3 -u "$SCRIPTS_E/normalize_cad_v20.py"

# =============================================================================
# Stage 4: Journal articles (PDFs → v24 JSONLs)
# =============================================================================

echo "=== Stage 4a: Dergipark batch 1 (Gemini API) ==="
python3 -u "$SCRIPTS_E/extract_akt_pairs_v24.py" --mode dergipark \
    --base-dir "$DATA_DIR/journal_articles/dergipark_v1"

echo "=== Stage 4b: Michel (Gemini API) ==="
python3 -u "$SCRIPTS_E/extract_akt_pairs_v24.py" --mode michel \
    --base-dir "$DATA_DIR/journal_articles/michel"

echo "=== Stage 4c: Round 2 Turkish (Gemini API) ==="
python3 -u "$SCRIPTS_E/extract_akt_pairs_v24.py" --mode dergipark \
    --base-dir "$DATA_DIR/journal_articles/round_2/turkish"

echo "=== Stage 4d: Round 2 English (Gemini API) ==="
python3 -u "$SCRIPTS_E/extract_akt_pairs_v24.py" --mode michel \
    --base-dir "$DATA_DIR/journal_articles/round_2/english"

echo "=== Stage 4e: Round 4 Turkish (Gemini API) ==="
python3 -u "$SCRIPTS_E/extract_akt_pairs_v24.py" --mode dergipark \
    --base-dir "$DATA_DIR/journal_articles/round_4/dergipark_tr"

echo "=== Stage 4f: Round 4 English (Gemini API) ==="
python3 -u "$SCRIPTS_E/extract_akt_pairs_v24.py" --mode michel \
    --base-dir "$DATA_DIR/journal_articles/round_4/dergipark_en"

echo "=== Stage 4g: Round 4 German (Gemini API) ==="
python3 -u "$SCRIPTS_E/extract_akt_pairs_v24.py" --mode top_bottom \
    --base-dir "$DATA_DIR/journal_articles/round_4/dergipark_de"

# =============================================================================
# Stage 5: Hecker pipeline (PDFs → hecker_hpm_translations_v22.jsonl)
# Requires Stage 1c expert data for crossref and few-shot
# =============================================================================

echo "=== Stage 5a: Hecker PDF extraction (Gemini API) ==="
python3 -u "$SCRIPTS_E/extract_akt_pairs_v24.py" --mode hecker \
    --base-dir "$DATA_DIR/hecker_hpm/pdfs"

echo "=== Stage 5b: Hecker HTML scrape (optional — data pre-provided) ==="
# Uncomment to re-scrape from HPM website:
# python3 -u "$SCRIPTS_E/scrape_hpm_html.py"

echo "=== Stage 5c: Hecker cross-reference ==="
python3 -u "$SCRIPTS_E/crossref_hecker.py" \
    --input "$DATA_DIR/hecker_hpm/pdfs/extracted_v24_pro31/akt_pairs_v24_pro31.jsonl"

echo "=== Stage 5d: Generate Hecker translations (Gemini API) ==="
python3 -u "$SCRIPTS_E/split_published_texts_v22.py" \
    --input "$DATA_DIR/hecker_hpm_translate_input.csv" \
    --output "$DATA_DIR/hecker_hpm_translations_v22.jsonl" \
    --keep-oare-ids "$DATA_DIR/hecker_hpm_translate_keep_oare_ids.txt" \
    --batch-size 8

# =============================================================================
# Stage 6: Synthetic data (published_texts.csv → synthetic_v22.jsonl)
# Requires Stage 1c expert data for few-shot examples
# =============================================================================

echo "=== Stage 6a: Synthetic V22 translations (Gemini API) ==="
python3 -u "$SCRIPTS_E/split_published_texts_v22.py" \
    --batch-size 8
# Note: synthetic_v19_dedup.jsonl is a pre-provided artifact from older pipeline.
# It is included in the akkadian-3rd-place-training-data Kaggle dataset.

# =============================================================================
# Stage 7: Final preparation (all JSONLs → augmented training set)
# =============================================================================

echo "=== Stage 7: Prepare final training dataset ==="
python3 -u "$SCRIPTS_P/prepare_sentence_data_23.py" \
    --hecker --round4

echo ""
echo "=== Pipeline complete ==="
echo "Training data written to: $DATA_DIR/synth_claude_v23_aug1/"
