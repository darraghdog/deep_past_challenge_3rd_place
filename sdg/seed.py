"""Extract diverse seed sentences from training data for grammar-guided augmentation.

Filters out formulaic patterns (seals, witnesses, letter openings, eponymies, heavy gaps)
and samples diverse seeds balanced across genres and documents.

Usage:
    from sdg.seed import get_seeds
    seeds = get_seeds(n=2000)
"""

from pathlib import Path

import kagglehub
import pandas as pd

# Formulaic patterns to exclude (applied to English translation)
EXCLUDE_PATTERNS = [
    r"^Seal of ",  # seal impressions — pure name listing
    r"^Witness",  # witness lists — pure name listing
    r"^(From |To |Say to |Thus )",  # letter openings — formulaic, handle separately
    r"(eponym|Eponym)",  # date formulas
    r"<gap>.*<gap>.*<gap>",  # heavily broken (3+ gaps)
]

MIN_TRANSLIT_LEN = 20  # too short = not enough content to transform
MAX_TRANSLIT_LEN = 200  # too long = complex, hard to transform cleanly


def load_training_data() -> pd.DataFrame:
    """Load English training sentences from dpc-mix-a04."""
    input_dir = Path(kagglehub.dataset_download("conjuring92/dpc-mix-a04"))
    train_df = pd.read_parquet(input_dir / "train.parquet")
    return train_df


def load_oare() -> pd.DataFrame:
    """Load OARE word-level annotations."""
    helper_dir = Path(kagglehub.dataset_download("conjuring92/dpc-helper-v01"))
    return pd.read_parquet(helper_dir / "oare_processed.parquet")


def load_names() -> pd.DataFrame:
    """Load name lookup table (transliteration → English name)."""
    input_dir = Path(kagglehub.dataset_download("conjuring92/dpc-mix-a04"))
    lookup_df = pd.read_parquet(input_dir / "lookup.parquet")
    # Deduplicate: take first English name per transliteration form
    names = lookup_df[lookup_df.transliteration.notna() & lookup_df.translation.notna()].groupby("transliteration").translation.first().reset_index()
    return names


def _has_verb(tl: str) -> bool:
    """Heuristic: does the transliteration likely contain a finite verb?

    OA finite verbs have characteristic prefix patterns (i-, ta-, a-, ni-, u-, tu-, nu-)
    or are imperatives/statives. We check for tokens starting with common verb prefixes
    that are NOT known Sumerograms or prepositions.
    """
    VERB_PREFIXES = {"i-", "ta-", "a-", "ni-", "u-", "tu-", "nu-", "li-", "lu-"}
    NON_VERB_STARTS = {
        "i-na",
        "a-na",
        "a-šùr",
        "a-šur",
        "a-sur",
        "a-lim",
        "a-bi",
        "a-hi",
        "a-ha",
    }
    tokens = tl.lower().split()
    for tok in tokens:
        if any(tok.startswith(p) for p in VERB_PREFIXES):
            if tok not in NON_VERB_STARTS and not tok.startswith("i-na") and not tok.startswith("a-na"):
                return True
    return False


def filter_formulaic(df: pd.DataFrame, require_verb: bool = False) -> pd.DataFrame:
    """Remove formulaic sentences that aren't useful as transformation seeds."""
    mask = pd.Series(False, index=df.index)

    for pat in EXCLUDE_PATTERNS:
        mask |= df.translation.str.contains(pat, na=False, regex=True)

    # Length filters
    tlen = df.transliteration.str.len()
    mask |= tlen < MIN_TRANSLIT_LEN
    mask |= tlen > MAX_TRANSLIT_LEN

    # Verb presence filter
    if require_verb:
        mask |= ~df.transliteration.apply(_has_verb)

    return df[~mask].reset_index(drop=True)


def get_seeds(
    n: int = 2000,
    language: str = "en",
    balance_genre: bool = True,
    random_state: int | None = None,
    require_verb: bool = False,
) -> pd.DataFrame:
    """Get diverse seed sentences for grammar-guided augmentation.

    Args:
        n: Number of seeds to sample.
        language: Language filter ('en' for original English only,
                  'all' for all languages including translated).
        balance_genre: If True, sample proportionally across document_type
                       with a floor so rare genres are represented.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with columns: oare_id, sentence_id, transliteration, translation,
        document_type, pdf, base_alias
    """
    train_df = load_training_data()

    # Language filter
    if language == "en":
        pool = train_df[train_df.language == "en"].copy()
    elif language == "en_all":
        pool = train_df[train_df.language.isin(["en", "translated_en"])].copy()
    elif language == "all":
        pool = train_df.copy()
    else:
        pool = train_df[train_df.language == language].copy()

    # Remove formulaic
    pool = filter_formulaic(pool, require_verb=require_verb)

    # Deduplicate by transliteration (some augmented pairs share same translit)
    pool = pool.drop_duplicates(subset=["transliteration"]).reset_index(drop=True)

    n = min(n, len(pool))

    if not balance_genre:
        return pool.sample(n, random_state=random_state).reset_index(drop=True)

    # Stratified sampling: proportional to genre but with a floor
    genre_counts = pool.document_type.value_counts()
    total = genre_counts.sum()
    min_per_genre = max(1, n // (len(genre_counts) * 3))  # floor: at least ~1/3 of equal share

    # Compute target per genre
    targets = {}
    remaining = n
    for genre, count in genre_counts.items():
        target = max(min_per_genre, int(n * count / total))
        target = min(target, count)  # can't exceed available
        targets[genre] = target
        remaining -= target

    # Distribute any remaining budget to largest genres
    if remaining > 0:
        for genre in genre_counts.index:
            available = genre_counts[genre] - targets[genre]
            add = min(remaining, available)
            targets[genre] += add
            remaining -= add
            if remaining <= 0:
                break

    # Sample per genre, then also diversify within genre by document
    samples = []
    for genre, target in targets.items():
        genre_pool = pool[pool.document_type == genre]

        if target >= len(genre_pool):
            samples.append(genre_pool)
            continue

        # Within genre: sample at most 2 sentences per document for diversity
        doc_groups = genre_pool.groupby("oare_id")
        per_doc = []
        for _, group in doc_groups:
            per_doc.append(group.sample(min(2, len(group)), random_state=random_state))
        diverse_pool = pd.concat(per_doc, ignore_index=True)

        if len(diverse_pool) >= target:
            samples.append(diverse_pool.sample(target, random_state=random_state))
        else:
            # Not enough with per-doc cap, relax and sample from full genre pool
            samples.append(genre_pool.sample(target, random_state=random_state))

    result = pd.concat(samples, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return result


def base_oare_id(oare_id: str) -> str:
    """Strip prefixes to get base OARE UUID for annotation lookup."""
    return oare_id.replace("translated_", "").replace("augmented_", "").replace("EXTERNAL-", "")


def get_oare_for_seed(seed_row: pd.Series, oare_df: pd.DataFrame) -> pd.DataFrame:
    """Get OARE word-level annotations for a seed sentence's document."""
    base_id = base_oare_id(seed_row["oare_id"])
    return oare_df[oare_df.oare_id == base_id].sort_values("line_num")


if __name__ == "__main__":
    seeds = get_seeds(n=2000)
    print(f"Seeds: {len(seeds):,}")
    print(f"Genre distribution:\n{seeds.document_type.value_counts().to_string()}")
    print(f"\nTranslit length: mean={seeds.transliteration.str.len().mean():.0f}, median={seeds.transliteration.str.len().median():.0f}")
    print(f"Unique documents: {seeds.oare_id.nunique():,}")
    print("\nSample seeds:")
    for _, row in seeds.head(5).iterrows():
        print(f"  [{row.document_type:12s}] {row.transliteration[:70]}")
        print(f"  {'':14s} {row.translation[:70]}")
        print()
