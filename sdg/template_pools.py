"""Slot value pools for OA template-based data generation.

Loads name pools from ono.parquet, extracts eponyms/week officials from
training data, and exposes sampling functions that return (TL, TR) pairs.

Usage:
    from sdg.template_pools import sample_name, sample_amount, sample_eponym, ...
"""

import random

import pandas as pd

from sdg.template_constraints import ALL_PLACES, COMMODITY_PROFILES, CONTAINERS, DEADLINE_RANGES, EXPENSE_PURPOSES, FESTIVALS, FRACTIONS, INTEREST_RATES, ITEM_POOL, KINSHIP_TERMS, MONTHS, OCCUPATIONS, PENALTY_AMOUNTS, WEIGHT_COMMODITIES, lowercase_tl_name

# ---------------------------------------------------------------------------
# Data loading (runs once at import time)
# ---------------------------------------------------------------------------

# _input_dir = Path(kagglehub.dataset_download("conjuring92/dpc-mix-a04"))
# _ono_df = pd.read_parquet(_input_dir / "ono.parquet")
# _train_df = pd.read_parquet(_input_dir / "train.parquet")
# _train_en = _train_df[_train_df.language == "en"].reset_index(drop=True)
_ono_df = pd.read_parquet("/Users/rabiswas/Desktop/repos/personal/dpc/data/supl_data/onomasticon_final.parquet")


# ---------------------------------------------------------------------------
# 1. NAME POOL
# ---------------------------------------------------------------------------


def _build_name_pool():
    pool = []
    for _, row in _ono_df.iterrows():
        tl = row["transliteration"].strip()
        tr = row["translation"].strip()
        if not tl or not tr or len(tl) < 3:
            continue
        # Skip names where TL has <gap> but TR doesn't (hallucinated completions)
        if "<gap>" in tl and "<gap>" not in tr:
            continue
        # Strip determinative leaks from TR (these are TL conventions only)
        tr = tr.replace("{d}", "").replace("{f}", "").replace("{m}", "")
        # h_initial = tr.startswith("H") and len(tr) > 1 and tr[1].islower()
        pool.append(
            {
                "tl": tl,
                "tl_lower": lowercase_tl_name(tl),
                "tr": tr,
                # "tr_h_fixed": lowercase_tr_h_name(tr),
                # "h_initial": h_initial,
            }
        )
    return pool


NAME_POOL = _build_name_pool()


# ---------------------------------------------------------------------------
# 2. EPONYM POOL (hardcoded — regex extraction from training data was noisy)
# ---------------------------------------------------------------------------

EPONYM_POOL = [
    # Plain eponyms (li-mu-um NAME)
    {"tl": "A-lá-hu-um", "tr": "Ali-ahum"},
    {"tl": "A-šur-ma-lik", "tr": "Aššur-malik"},
    {"tl": "A-mur-A-šur", "tr": "Amur-Aššur"},
    {"tl": "A-šur-na-da", "tr": "Aššur-nādā"},
    {"tl": "A-šur-i-mì-tí", "tr": "Aššur-imittī"},
    {"tl": "Bu-zu-zu", "tr": "Buzuzu"},
    {"tl": "Bu-zu-ta-a", "tr": "Buzutaya"},
    {"tl": "Da-ni-a", "tr": "Dān-Ea"},
    {"tl": "DÙG-A-šur", "tr": "Ṭāb-Aššur"},
    {"tl": "E-lá-lí", "tr": "Elālī"},
    {"tl": "En-na-Sú-en6", "tr": "Ennam-Suen"},
    {"tl": "En-na-Sú-in", "tr": "Enna-Suen"},
    {"tl": "I-dí-a-bu-um", "tr": "Iddin-abum"},
    {"tl": "I-dí-A-šùr", "tr": "Iddin-Aššur"},
    {"tl": "I-ku-pí-Ištar", "tr": "Ikūn-pī-Ištar"},
    {"tl": "I-na-a", "tr": "Innaya"},
    {"tl": "Ib-ni-{d}IŠKUR", "tr": "Ibni-Adad"},
    {"tl": "Ip-hu-ru-um", "tr": "Iphurum"},
    {"tl": "Ku-bi-e-a", "tr": "Kūbiya"},
    {"tl": "A-ku-tum", "tr": "Akūtum"},
    {"tl": "Puzur4-{d}MUŠ", "tr": "Puzur-Nirah"},
    {"tl": "Šu-da-a", "tr": "Šudaya"},
    {"tl": "Šu-Hu-bur", "tr": "Šu-Hubur"},
    {"tl": "Šu-Sú-in", "tr": "Šu-Suen"},
    {"tl": "Ṣí-lu-lu", "tr": "Ṣilūlu"},
    {"tl": "A-šur-i-dí", "tr": "Aššur-idī"},
    {"tl": "A-šùr-SIG5", "tr": "Aššur-damiq"},
    {"tl": "A-šùr-SIPA", "tr": "Aššur-rē'ī"},
    {"tl": "A-al-DÙG", "tr": "Ali-ṭāb"},
    {"tl": "DINGIR-pì-lá-ah", "tr": "Ilī-pilah"},
    {"tl": "DINGIR-šu-GAL", "tr": "Ilšu-rabi"},
    {"tl": "{d}IŠKUR-ba-ni", "tr": "Adad-bāni"},
    {"tl": "{d}UTU-ba-ni", "tr": "Šamaš-bāni"},
    {"tl": "Ì-lí-dan", "tr": "Ilī-dān"},
    {"tl": "a-bi4-a", "tr": "Abiya"},
    {"tl": "A-mur-A-šùr", "tr": "Amur-Aššur"},
    # Eponyms with patronymics (li-mu-um NAME DUMU NAME)
    {"tl": "A-šur-ma-lik DUMU A-lá-hi-im", "tr": "Aššur-malik son of Ali-ahum"},
    {"tl": "Da-né-a DUMU A-bi4-wa-qar", "tr": "Dān-Ea son of Abī-waqar"},
    {"tl": "E-lá-lí DUMU I-ku-nim", "tr": "Elāli son of Ikūnum"},
    {"tl": "DINGIR-šu-GAL DUMU Ba-zi-a", "tr": "Ilšu-rabi son of Baziya"},
    {"tl": "A-al-DÙG DUMU Sà-sà-tum", "tr": "Ali-ṭāb son of Sassātum"},
    {"tl": "En-na-Sú-in DUMU Šu-A-šùr", "tr": "Enna-Suen son of Šu-Aššur"},
    {"tl": "Šu-Sú-in DUMU Ba-bi-lim", "tr": "Šu-Suen son of Babilum"},
    {"tl": "Šu-hu-bur DUMU E-lá-lí", "tr": "Šu-Hubur son of Elālī"},
]


# ---------------------------------------------------------------------------
# 3. WEEK OFFICIAL POOL (hardcoded)
# ---------------------------------------------------------------------------

WEEK_POOL = [
    # Single officials
    {"tl": "Ku-da-tim", "tr": "Kudātum"},
    {"tl": "E-na-ah-DINGIR", "tr": "Enah-ili"},
    {"tl": "{d}MAR.TU-ba-ni", "tr": "Amurrum-bāni"},
    {"tl": "I-tur4-DINGIR", "tr": "Itūr-ili"},
    {"tl": "A-šur-i-mì-tí", "tr": "Aššur-imittī"},
    {"tl": "A-šùr-i-mì-tí", "tr": "Aššur-imittī"},
    {"tl": "Kur-ub-Ištar", "tr": "Kurub-Ištar"},
    {"tl": "I-dí-na-bi-im", "tr": "Iddin-abum"},
    {"tl": "Ì-lí-dan", "tr": "Ilī-dān"},
    {"tl": "kà-ší-im", "tr": "the kaššu-official"},
    # Officials with patronymics
    {"tl": "A-šur-ma-lik DUMU Sú-kà-li-a", "tr": "Aššur-malik son of Sukkalliya"},
    {"tl": "I-tur4-DINGIR DUMU A-mur-Ištar", "tr": "Itūr-ili son of Amur-Ištar"},
    {"tl": "A-šùr-ma-lik DUMU A-lá-hi-im", "tr": "Aššur-malik son of Ali-ahum"},
    # Paired officials (two names with ù = "and")
    {"tl": "A-šur-i-mì-tí ù tap-pá-i-šu", "tr": "Aššur-imittī and his partner"},
    {"tl": "A-šur-i-mì-tí ù A-mur-Ištar", "tr": "Aššur-imittī and Amur-Ištar"},
    {"tl": "A-šùr-i-dí ù Šu-Ištar", "tr": "Aššur-idī and Šu-Ištar"},
    {"tl": "A-la-hi-im ú {d}MAR.TU-ba-ni", "tr": "Ali-ahum and Amurrum-bāni"},
    {"tl": "Ma-nu-ki-A-šur ù I-ku-nim", "tr": "Mannu-kī-Aššur and Ikūnum"},
    {"tl": "Pu-šu-ke-en6 ù Puzur4-A-šur", "tr": "Pūšu-kēn and Puzur-Aššur"},
    {"tl": "Ba-bi-lim ù Na-ra-am-ZU", "tr": "Babilum and Narām-Sīn"},
    {"tl": "i-dí-a-šur ù i-ku-nim", "tr": "Iddin-Aššur and Ikūnum"},
]


# ---------------------------------------------------------------------------
# Sampling functions
# ---------------------------------------------------------------------------


def sample_name(n: int = 1, lowercase_tl: bool = True) -> list[dict]:
    """Sample n random name pairs from the pool.

    Returns list of {tl, tr} dicts. If lowercase_tl=True, uses the
    lowercase syllabic form (matching test convention).
    """
    picks = random.sample(NAME_POOL, min(n, len(NAME_POOL)))
    results = []
    for p in picks:
        results.append(
            {
                "tl": p["tl_lower"] if lowercase_tl else p["tl"],
                "tr": p["tr"],
            }
        )
    return results


def sample_eponym() -> dict:
    """Sample a random eponym pair {tl, tr}."""
    if EPONYM_POOL:
        return random.choice(EPONYM_POOL)
    # Fallback to a name
    name = sample_name(1, lowercase_tl=False)[0]
    return name


def sample_week_officials() -> dict:
    """Sample random week official(s) {tl, tr}."""
    if WEEK_POOL:
        return random.choice(WEEK_POOL)
    # Fallback: two random names joined with ù
    names = sample_name(2, lowercase_tl=False)
    return {
        "tl": f"{names[0]['tl']} ù {names[1]['tl']}",
        "tr": f"{names[0]['tr']} and {names[1]['tr']}",
    }


def sample_month(prefer_name: bool = True) -> dict:
    """Sample a random month. Returns {tl, tr, num}.

    TR is always the month name (e.g. "Kēnātim"), never "month N",
    because templates already provide the word "month" / "Month".
    """
    month = random.choice(MONTHS)
    return {"tl": month["tl"], "tr": month["tr"], "num": month["num"]}


def sample_place() -> dict:
    """Sample a random place {tl, tr}."""
    return random.choice(ALL_PLACES)


def sample_commodity() -> tuple[str, dict]:
    """Sample a random commodity key and its profile.

    Returns (key, profile) where key is e.g. 'silver'.
    """
    key = random.choice(list(COMMODITY_PROFILES.keys()))
    return key, COMMODITY_PROFILES[key]


def sample_interest_rate() -> dict:
    """Sample an interest rate weighted by frequency. Returns {tl, tr}."""
    weights = [r["weight"] for r in INTEREST_RATES]
    rate = random.choices(INTEREST_RATES, weights=weights, k=1)[0]
    return {"tl": rate["tl"], "tr": rate["tr"]}


def sample_occupation() -> dict:
    """Sample a random occupation {tl, tr}."""
    return random.choice(OCCUPATIONS)


def sample_kinship() -> dict:
    """Sample a random kinship term."""
    return random.choice(KINSHIP_TERMS)


def sample_container(commodity_key: str | None = None) -> dict:
    """Sample a random container, optionally filtered by commodity."""
    if commodity_key:
        valid = [c for c in CONTAINERS if commodity_key in c["contents"]]
        if valid:
            return random.choice(valid)
    return random.choice(CONTAINERS)


def sample_festival() -> dict:
    """Sample a random festival {tl, tr}."""
    return random.choice(FESTIVALS)


def sample_expense() -> dict:
    """Sample a random expense purpose {tl, tr}."""
    return random.choice(EXPENSE_PURPOSES)


def sample_penalty() -> dict:
    """Sample a random penalty amount {tl, tr}."""
    return random.choice(PENALTY_AMOUNTS)


def sample_deadline() -> dict:
    """Sample a random deadline {n, tl_unit, tr_unit, tl, tr}.

    E.g., {n: 12, tl_unit: "ha-am-ša-tim", tr_unit: "weeks",
           tl: "12 ha-am-ša-tim", tr: "12 weeks"}
    """
    unit_key = random.choice(list(DEADLINE_RANGES.keys()))
    unit = DEADLINE_RANGES[unit_key]
    n = random.randint(unit["range"][0], unit["range"][1])
    return {
        "n": n,
        "tl_unit": unit["tl_unit"],
        "tr_unit": unit["tr_unit"],
        "tl": f"{n} {unit['tl_unit']}",
        "tr": f"{n} {unit['tr_unit']}",
    }


# ---------------------------------------------------------------------------
# Amount generation (the complex one)
# ---------------------------------------------------------------------------

_PLURAL_TO_SINGULAR = {
    "shekels": "shekel",
    "minas": "mina",
    "talents": "talent",
    "sacks": "sack",
    "jars": "jar",
    "grains": "grain",
}


def _maybe_singular(unit_tr: str, n: int, frac: dict | None) -> str:
    """Return singular form if n == 1 and no fraction."""
    if n == 1 and frac is None and unit_tr in _PLURAL_TO_SINGULAR:
        return _PLURAL_TO_SINGULAR[unit_tr]
    return unit_tr


def _sample_fraction() -> dict | None:
    """Maybe return a fraction (40% chance), or None."""
    if random.random() < 0.4:
        frac = random.choice(FRACTIONS)
        return frac
    return None


def sample_amount(commodity_key: str, include_commodity: bool = True) -> dict:
    """Generate a random valid amount for a commodity.

    Args:
        commodity_key: Key into COMMODITY_PROFILES.
        include_commodity: If True, append commodity Sumerogram and qualifier.
            Set to False when the template already embeds the commodity.

    Returns {tl, tr} with correctly composed compound amounts.
    E.g., {"tl": "3 ⅓ ma-na 5 GÍN KÙ.BABBAR ṣa-ru-pá-am",
           "tr": "3 ⅓ minas 5 shekels of refined silver"}
    """
    profile = COMMODITY_PROFILES[commodity_key]
    units = profile["units"]

    # Pick qualifier (only used if include_commodity)
    qualifier = random.choice(profile["qualifiers"])
    q_tl = qualifier["tl"] if include_commodity else ""
    q_tr = qualifier["tr"] if include_commodity else ""

    # Decide: simple (1 tier) or compound (2 tiers)
    # Only compound if the chosen unit has a compound_with
    unit = random.choice(units)
    use_compound = unit["compound_with"] is not None and random.random() < 0.35

    if use_compound:
        # Find the sub-unit
        sub_unit_tl = unit["compound_with"]
        sub_unit = next((u for u in units if u["tl"] == sub_unit_tl), None)
        if sub_unit is None:
            use_compound = False

    if use_compound and sub_unit:
        # Main tier
        main_n = random.randint(1, unit["range"][1] // 2)  # smaller for compound
        main_frac = _sample_fraction()
        # Sub tier
        sub_max = min(sub_unit["range"][1], 59)  # sub-unit within one main unit
        sub_n = random.randint(1, sub_max)
        sub_frac = _sample_fraction()

        # Build TL
        main_tl = f"{main_n}" + (f" {main_frac['unicode']}" if main_frac else "")
        sub_tl = f"{sub_n}" + (f" {sub_frac['unicode']}" if sub_frac else "")

        parts_tl = [f"{main_tl} {unit['tl']}", f"{sub_tl} {sub_unit['tl']}"]
        if unit["tl"] == "":  # plain count commodities don't compound
            parts_tl = [f"{main_tl}"]

        tl_str = " ".join(parts_tl)
        if include_commodity:
            commodity_tl = profile["tl"]
            if commodity_tl:
                tl_str += f" {commodity_tl}"
        if q_tl:
            tl_str += f" {q_tl}"

        # Build TR
        main_tr = f"{main_n}" + (f" {main_frac['unicode']}" if main_frac else "")
        sub_tr = f"{sub_n}" + (f" {sub_frac['unicode']}" if sub_frac else "")

        unit_tr_main = _maybe_singular(unit["tr"], main_n, main_frac) if unit["tr"] else ""
        unit_tr_sub = _maybe_singular(sub_unit["tr"], sub_n, sub_frac) if sub_unit["tr"] else ""

        parts_tr = []
        if unit_tr_main:
            parts_tr.append(f"{main_tr} {unit_tr_main}")
        else:
            parts_tr.append(main_tr)
        if unit_tr_sub:
            parts_tr.append(f"{sub_tr} {unit_tr_sub}")
        else:
            parts_tr.append(sub_tr)

        tr_str = " ".join(parts_tr)
        if include_commodity:
            commodity_tr = profile["tr"]
            if q_tr:
                tr_str += f" of {q_tr} {commodity_tr}"
            elif commodity_tr:
                tr_str += f" of {commodity_tr}"

    else:
        # Simple amount (1 tier)
        n = random.randint(unit["range"][0], unit["range"][1])
        frac = _sample_fraction()

        n_str = f"{n}" + (f" {frac['unicode']}" if frac else "")

        # TL
        unit_tl = unit["tl"]

        tl_parts = [n_str]
        if unit_tl:
            tl_parts.append(unit_tl)
        if include_commodity:
            commodity_tl = profile["tl"]
            if commodity_tl:
                tl_parts.append(commodity_tl)
        if q_tl:
            tl_parts.append(q_tl)
        tl_str = " ".join(tl_parts)

        # TR
        unit_tr = _maybe_singular(unit["tr"], n, frac) if unit["tr"] else ""

        tr_parts = [n_str]
        if unit_tr:
            tr_parts.append(unit_tr)

        tr_str = " ".join(tr_parts)
        if include_commodity:
            commodity_tr = profile["tr"]
            if q_tr:
                tr_str += f" of {q_tr} {commodity_tr}"
            elif commodity_tr:
                tr_str += f" of {commodity_tr}"

    return {"tl": tl_str, "tr": tr_str}


def sample_bare_number(commodity_key: str = "silver") -> dict:
    """Sample just a number (+ optional fraction), no unit, no commodity.

    Uses the range from a commodity profile to stay realistic.
    For templates that embed their own unit token in the TL string.
    Returns e.g. {'tl': '3 ½', 'tr': '3 ½'}.
    """
    profile = COMMODITY_PROFILES.get(commodity_key, COMMODITY_PROFILES["silver"])
    unit = random.choice(profile["units"])
    n = random.randint(unit["range"][0], unit["range"][1])
    frac = _sample_fraction()
    n_str = f"{n}" + (f" {frac['unicode']}" if frac else "")
    return {"tl": n_str, "tr": n_str}


def sample_commodity_label(
    values: list[str] | None = None,
    weight_only: bool = False,
    exclude: set[str] | None = None,
) -> dict:
    """Sample a commodity label (not an amount).

    If values provided, parse 'TL/TR' format and pick one.
    Otherwise pick from COMMODITY_PROFILES keys.
    If weight_only=True, only pick from weighable commodities (no sheep/donkeys/bread etc).
    If exclude provided, skip those commodity keys.
    """
    if values:
        pick = random.choice(values)
        tl, tr = pick.split("/", 1)
        return {"tl": tl.strip(), "tr": tr.strip()}
    pool = list(WEIGHT_COMMODITIES) if weight_only else list(COMMODITY_PROFILES.keys())
    if exclude:
        pool = [k for k in pool if k not in exclude]
    key = random.choice(pool)
    profile = COMMODITY_PROFILES[key]
    return {"tl": profile["tl"], "tr": profile["tr"]}


def sample_place_from_values(values: list[str]) -> dict:
    """Parse 'TL/TR' values list and pick one."""
    pick = random.choice(values)
    tl, tr = pick.split("/", 1)
    return {"tl": tl.strip(), "tr": tr.strip()}


def sample_rate_number() -> dict:
    """Sample just the numeric portion of an interest rate.

    E.g. {'tl': '½', 'tr': '½'} or {'tl': '1 ½', 'tr': '1 ½'}
    Templates already contain 'GÍN.TA' so we only need the number.
    """
    weights = [r["weight"] for r in INTEREST_RATES]
    rate = random.choices(INTEREST_RATES, weights=weights, k=1)[0]
    # Extract just the number part (before GÍN.TA)
    num_tl = rate["tl"].replace("GÍN.TA", "").strip()
    num_tr = rate["tr"].replace("shekels", "").replace("shekel", "").strip()
    return {"tl": num_tl, "tr": num_tr}


def sample_item() -> dict:
    """Sample a tradeable item {tl, tr}."""
    return random.choice(ITEM_POOL)


def sample_deadline_number(unit: str | None = None) -> dict:
    """Return just the number for a deadline, given the unit key.

    unit: 'weeks', 'months', 'days', 'years'. If None, pick random.
    Returns {'tl': '12', 'tr': '12'}.
    """
    if unit and unit in DEADLINE_RANGES:
        info = DEADLINE_RANGES[unit]
    else:
        unit_key = random.choice(list(DEADLINE_RANGES.keys()))
        info = DEADLINE_RANGES[unit_key]
    n = random.randint(info["range"][0], info["range"][1])
    return {"tl": str(n), "tr": str(n)}


def sample_amount_simple(commodity_key: str) -> dict:
    """Generate a simple (non-compound) amount. Always one tier, no fractions."""
    profile = COMMODITY_PROFILES[commodity_key]
    unit = random.choice(profile["units"])
    n = random.randint(unit["range"][0], min(unit["range"][1], 30))

    unit_tl = unit["tl"]
    commodity_tl = profile["tl"]
    tl_parts = [str(n)]
    if unit_tl:
        tl_parts.append(unit_tl)
    if commodity_tl:
        tl_parts.append(commodity_tl)
    tl_str = " ".join(tl_parts)

    unit_tr = _maybe_singular(unit["tr"], n, None) if unit["tr"] else ""
    commodity_tr = profile["tr"]
    tr_parts = [str(n)]
    if unit_tr:
        tr_parts.append(unit_tr)
    tr_str = " ".join(tr_parts)
    if commodity_tr:
        tr_str += f" of {commodity_tr}"

    return {"tl": tl_str, "tr": tr_str}
