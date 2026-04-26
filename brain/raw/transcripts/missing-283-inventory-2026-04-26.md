# Missing 283 Inventory — 2026-04-26

Source directory inspected: `../akk/datamount/missing_283_transliterations/`.

This directory belongs to the original working repository (`../akk/`), not the current reproduction repo. It records investigations and source collection around OARE tablets with translations but missing transliterations.

## Source Metadata

`dataset-metadata.json`:

```json
{
  "id": "darraghdog/missing-283-tablets",
  "title": "missing-283-tablets",
  "licenses": [{"name": "other"}]
}
```

## README Summary Observed

- 283 tablets in `Sentences_Oare_FirstWord_LinNum.csv` had expert translations but were not in `published_texts.csv`.
- None overlapped `train.csv`.
- All 283 had complete English translations, totaling 1,306 sentences.
- Transliterations had been found for 53 of 283 at the time of the README.
- About 230 remained unavailable, mostly unpublished or book-only.

## Directory Rounds

- `round_1/`: initial manual collection, including `missing_283_tablets.json`, Hecker aATU files, selected Turkish journal PDFs, and extracted Hecker JSON.
- `round_2/`: Michel HAL and DergiPark papers, with English, Turkish, table, and no-transliteration subfolders.
- `round_3_hecker_hpm/`: bulk Hecker/HPM PDFs.
- `round_4/`: broad collection across Turkish, English, German, OIP27/Gelb, Hecker web, and related sources.
- `round_5_cdli/`: CDLI Old Assyrian corpus clone and cross-reference output.
- `round_6/`, `round_7/`, `round_8/`: later source rounds and extraction outputs.
- `extra_1003/`: DergiPark and Michel source batches used in training-data construction.

## File Inventory

Observed total files: 558.

Observed extension counts:

- 231 `pdf`
- 170 `jsonl`
- 68 `tf`
- 16 `png`
- 13 `sample`
- 13 files without extension
- 11 `ds_store`
- 8 `ipynb`
- 4 `txt`
- 4 `json`
- 3 `yaml`
- 3 `py`
- 3 `md`
- 2 `tsv`
- 1 each: `sh`, `pack`, `idx`, `html`, `gitignore`, `gitattributes`, `flake8`, `excluded`, `css`

## Training Usage Claimed by README

The README says `synth_claude_v23_aug3`/aug4 used:

- Dergipark batch 1: 445 raw pairs, 2,650 augmented.
- Michel batch 1: 412 raw pairs, 2,093 augmented.
- Round 2 Turkish/English, Round 4 English/German/Turkish: part of 9,638 augmented pairs.
- Hecker HPM translations: 1,521 raw pairs, 26,672 augmented.
- Total listed: 4,204 raw pairs, 41,053 augmented.

Not included according to the README:

- Round 3 Hecker HPM transliteration-only set.
- Round 4 Gelb OIP27.
- Round 4 HPM HTML scrape.
- Round 5 CDLI.
