# 3rd Place Solution — Deep Past Initiative Machine Translation

Kaggle competition: [Deep Past Initiative Machine Translation](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation)

3rd place solution for translating Old Assyrian Akkadian transliterations to English.

## Data Setup

### 1. Competition Data (hosted by Kaggle)

Download from the [competition data page](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/data):

```
train.csv                            # 1,561 training pairs
test.csv                             # Test submission format
sample_submission.csv
published_texts.csv                  # 7,953 transliterations (no translations)
Sentences_Oare_FirstWord_LinNum.csv  # Sentence-level metadata
bibliography.csv
eBL_Dictionary.csv                   # Electronic Babylonian Library lexicon
OA_Lexicon_eBL.csv                   # Old Assyrian lexicon
publications.csv                     # Publication database
resources.csv
```

Place these in `data/competition/`.

### 2. Source Data (hosted on Kaggle Datasets)

Download from [akkadian-3rd-place-source-data-darragh](https://www.kaggle.com/datasets/darraghdog/akkadian-3rd-place-source-data-darragh):

```bash
kaggle datasets download darraghdog/akkadian-3rd-place-source-data-darragh -p data/ --unzip
```

This contains all PDF and reference sources used in the data extraction pipeline (~1.4 GB):

| Directory | Contents | Files | Size |
|-----------|----------|-------|------|
| `akt_subsets/side_by_side/` | AKT volumes with side-by-side transliteration/translation layout | 8 PDFs | 328M |
| `akt_subsets/top_bottom/` | AKT volumes with top-bottom layout | 9 PDFs | 44M |
| `akt_subsets/ocr/` | AKT 1 (1990) — oldest, OCR-quality scans | 1 PDF | 1.3M |
| `oaa_pihans/` | Reference publications (ICK4, Kouwenberg, Larsen, PIHANS96) | 4 PDFs | 183M |
| `cad/` | Chicago Assyrian Dictionary open-access volumes | 26 PDFs | 750M |
| `journal_articles/dergipark_v1/` | Turkish journal articles — batch 1 | 13 PDFs | 26M |
| `journal_articles/michel/` | Michel 2025 OA Legal Cases | 1 PDF | 1M |
| `journal_articles/round_2/turkish/` | Turkish journal articles — round 2 | 8 PDFs | 20M |
| `journal_articles/round_2/english/` | English journal articles — round 2 | 3 PDFs | 6M |
| `journal_articles/round_4/dergipark_tr/` | Turkish journal articles — round 4 | 26 PDFs | 96M |
| `journal_articles/round_4/dergipark_en/` | English journal articles — round 4 | 8 PDFs | 23M |
| `journal_articles/round_4/dergipark_de/` | German journal articles — round 4 | 2 PDFs | 1.6M |
| `hecker_hpm/pdfs/` | Hecker aATU/HPM transliteration corpus | 70 PDFs | 35M |
| `hecker_hpm/html_corpora/` | HPM web-scraped transliterations | 15 JSONLs | 828K |
| `hecker_hpm/POAT_Hecker.pdf` | POAT Hecker publication | 1 PDF | 237K |
| Root files | `holdout_oare_ids.txt`, `Dataset_Instructions.txt`, `Dataset_Instructions_v2.txt` | 3 files | 27K |

### Expected directory structure after setup

```
data/
  competition/           # From Kaggle competition page
    train.csv
    published_texts.csv
    ...
  akt_subsets/           # From source data dataset
    side_by_side/
    top_bottom/
    ocr/
  oaa_pihans/
  cad/
  journal_articles/
    dergipark_v1/
    michel/
    round_2/
    round_4/
  hecker_hpm/
    pdfs/
    html_corpora/
  holdout_oare_ids.txt
  Dataset_Instructions.txt
  Dataset_Instructions_v2.txt
```
