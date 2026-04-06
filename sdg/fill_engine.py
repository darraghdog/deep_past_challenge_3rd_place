"""Template fill engine for OA synthetic data generation.

Loads templates from sdg/templates/*.json, resolves slots using
sampling functions from template_pools, and produces (transliteration,
translation) pairs.

Usage:
    from sdg.fill_engine import fill_template, generate
    pairs = generate(n=1000, seed=42)
"""

import json
import random
import re
import unicodedata
from pathlib import Path

from sdg.template_constraints import COMMODITY_PROFILES
from sdg.template_pools import (
    sample_amount,
    sample_bare_number,
    sample_commodity_label,
    sample_deadline_number,
    sample_eponym,
    sample_festival,
    sample_item,
    sample_month,
    sample_name,
    sample_occupation,
    sample_place,
    sample_place_from_values,
    sample_rate_number,
)

# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def load_templates() -> list[dict]:
    """Load all templates from JSON files."""
    templates = []
    for f in sorted(_TEMPLATE_DIR.glob("*.json")):
        data = json.loads(f.read_text())
        templates.extend(data)
    return templates


_ALL_TEMPLATES = load_templates()
_TEMPLATES_BY_CATEGORY = {}
for _t in _ALL_TEMPLATES:
    cat = _t.get("category", "other")
    _TEMPLATES_BY_CATEGORY.setdefault(cat, []).append(_t)


# ---------------------------------------------------------------------------
# Commodity-in-template detection
# ---------------------------------------------------------------------------


def _strip_diacritics(s: str) -> str:
    """Remove diacritics for fuzzy matching (KÙ→KU, ANŠE→ANSE, etc.)."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# Build lookup: stripped sumerogram → set of commodity keys
_STRIPPED_TO_KEYS: dict[str, set[str]] = {}
for _key, _prof in COMMODITY_PROFILES.items():
    if _prof["tl"]:
        _stripped = _strip_diacritics(_prof["tl"]).upper()
        _STRIPPED_TO_KEYS.setdefault(_stripped, set()).add(_key)

# Extra variant spellings found in templates (not in COMMODITY_PROFILES)
_VARIANT_SUMEROGRAMS = {
    "GUSKIN": "gold",  # GUŠKIN = KÙ.GI
    "KU.KI": "gold",  # KU.KI = KÙ.GI (alternate Sumerogram)
    "ANSE": "donkeys",  # ANŠE variant
    "ANSE.HI.A": "donkeys",
}


def _get_template_tl_stripped(template: dict) -> str:
    """Get template TL with slots removed and diacritics stripped, uppercased."""
    tl = template["tl"]
    tl_clean = re.sub(r"\{\w+_TL\}", "", tl)
    return _strip_diacritics(tl_clean).upper()


def _template_has_commodity_embedded(template: dict, commodity_key: str | None) -> bool:
    """Check if the template TL already contains the commodity Sumerogram
    or if another slot provides the commodity (COMM/COMMODITY slot).

    Returns True if amount slot should be BARE (no commodity in output).
    """
    # Case 1: template has a separate COMM/COMMODITY slot
    for s in template["slots"]:
        if s["type"] == "commodity":
            return True

    if commodity_key is None:
        return False

    tl_stripped = _get_template_tl_stripped(template)

    # Case 2: template TL text contains the commodity Sumerogram
    profile = COMMODITY_PROFILES.get(commodity_key)
    if profile and profile["tl"]:
        target = _strip_diacritics(profile["tl"]).upper()
        if target in tl_stripped:
            return True

    # Case 3: check variant spellings in template
    for variant, vkey in _VARIANT_SUMEROGRAMS.items():
        if vkey == commodity_key and variant in tl_stripped:
            return True

    return False


def _detect_embedded_commodity(template: dict) -> str | None:
    """Detect which commodity is hardcoded in the template TL.

    Used when AMT slot has no commodity option but template embeds one.
    Returns commodity key or None.
    """
    # Check for COMM slot first
    for s in template["slots"]:
        if s["type"] == "commodity":
            return "silver"  # COMM slot handles commodity; use silver for ranges

    tl_stripped = _get_template_tl_stripped(template)

    # Check all known sumerograms (longest match first to avoid partial matches)
    all_targets = []
    for key, prof in COMMODITY_PROFILES.items():
        if prof["tl"]:
            target = _strip_diacritics(prof["tl"]).upper()
            all_targets.append((target, key))
    for variant, vkey in _VARIANT_SUMEROGRAMS.items():
        all_targets.append((variant, vkey))

    # Sort by length descending (prefer longer matches)
    all_targets.sort(key=lambda x: -len(x[0]))

    for target, key in all_targets:
        if target in tl_stripped:
            return key

    return None


# ---------------------------------------------------------------------------
# Slot resolver
# ---------------------------------------------------------------------------

_VALID_COMMODITIES = set(COMMODITY_PROFILES.keys())


def _resolve_amount(slot: dict, template: dict) -> dict:
    """Resolve an amount slot, detecting whether to include commodity."""
    opts = slot.get("options", {})

    # bare_number mode: just a number + optional fraction, no unit, no commodity
    if opts.get("bare_number"):
        commodity_key = opts.get("commodity_key", "silver")
        if commodity_key not in _VALID_COMMODITIES:
            commodity_key = "silver"
        return sample_bare_number(commodity_key)

    commodity = opts.get("commodity")

    # commodity is a known key string → use it
    if isinstance(commodity, str) and commodity in _VALID_COMMODITIES:
        include = not _template_has_commodity_embedded(template, commodity)
        return sample_amount(commodity, include_commodity=include)

    # commodity=True or other truthy → template provides the commodity,
    # amount should be BARE. Use "silver" for unit ranges (shekels/minas
    # are universal weight units in OA, safe default).
    if commodity:
        return sample_amount("silver", include_commodity=False)

    # No commodity option → check if template embeds a commodity anyway
    detected = _detect_embedded_commodity(template)
    if detected:
        return sample_amount(detected, include_commodity=False)

    # Truly no commodity anywhere → random commodity, include it
    from sdg.template_pools import sample_commodity as _sc

    key, _ = _sc()
    return sample_amount(key, include_commodity=True)


def _resolve_number(slot: dict) -> dict:
    """Resolve a plain number slot (for NUM-type slots that templates
    incorrectly mark as 'amount'). Returns just an integer as string."""
    opts = slot.get("options", {})
    # Use deadline ranges if unit specified, otherwise small int
    unit = opts.get("unit")
    if unit:
        return sample_deadline_number(unit)
    n = random.randint(1, 12)
    return {"tl": str(n), "tr": str(n)}


def _resolve_slot(slot: dict, template: dict) -> dict:
    """Resolve a single slot definition to {tl, tr}."""
    stype = slot["type"]
    opts = slot.get("options", {})

    if stype == "name":
        return sample_name(1, lowercase_tl=True)[0]

    elif stype == "amount":
        # Detect NUM-as-amount: slot named NUM/N with type "amount"
        slot_name = slot.get("name", "")
        if slot_name in ("NUM", "N"):
            return _resolve_number(slot)
        return _resolve_amount(slot, template)

    elif stype == "commodity":
        values = opts.get("values")
        weight_only = opts.get("weight_only", False)
        exclude = None
        if opts.get("exclude_silver"):
            exclude = {"silver"}
        return sample_commodity_label(values, weight_only=weight_only, exclude=exclude)

    elif stype == "place":
        values = opts.get("values")
        if values:
            return sample_place_from_values(values)
        return sample_place()

    elif stype == "deadline":
        unit = opts.get("unit")
        return sample_deadline_number(unit)

    elif stype == "month":
        m = sample_month()
        return {"tl": m["tl"], "tr": m["tr"]}

    elif stype == "eponym":
        return sample_eponym()

    elif stype == "occupation":
        values = opts.get("values")
        if values:
            pick = random.choice(values)
            if "/" in pick:
                tl, tr = pick.split("/", 1)
                return {"tl": tl.strip(), "tr": tr.strip()}
            return {"tl": pick, "tr": pick}
        return sample_occupation()

    elif stype == "rate":
        return sample_rate_number()

    elif stype == "item":
        return sample_item()

    elif stype == "festival":
        return sample_festival()

    else:
        raise ValueError(f"Unknown slot type: {stype!r} in slot {slot}")


# ---------------------------------------------------------------------------
# -ma enclitic fix
# ---------------------------------------------------------------------------

# Pattern: {PN_TL}-ma in template TL (enclitic after name)
_MA_PATTERN = re.compile(r"\{(\w+)_TL\}-ma")


def _fix_ma_enclitic(tl: str, resolved: dict[str, dict]) -> str:
    """Strip trailing -ma from name TL values when template appends -ma.

    Prevents double -ma: e.g. ta-ší-ma + -ma → ta-ší-ma (not ta-ší-ma-ma).
    """

    def _replacer(m):
        slot_name = m.group(1)
        if slot_name in resolved:
            name_tl = resolved[slot_name]["tl"]
            if name_tl.endswith("-ma"):
                # Strip the trailing -ma; template will add its own
                name_tl = name_tl[:-3]
            return name_tl + "-ma"
        return m.group(0)  # shouldn't happen, leave as-is

    return _MA_PATTERN.sub(_replacer, tl)


# ---------------------------------------------------------------------------
# Core fill function
# ---------------------------------------------------------------------------


def fill_template(template: dict) -> dict:
    """Fill one template with sampled slot values.

    Returns:
        {
            'tl': filled transliteration string,
            'tr': filled translation string,
            'template_id': template ID,
            'category': template category,
        }
    """
    slots = template["slots"]
    tl = template["tl"]
    tr = template["tr"]

    # Pre-sample all name slots at once to ensure uniqueness
    name_indices = [i for i, s in enumerate(slots) if s["type"] == "name"]
    if name_indices:
        names = sample_name(len(name_indices), lowercase_tl=True)
        while len(names) < len(name_indices):
            names.extend(sample_name(1, lowercase_tl=True))

    # Resolve each slot
    resolved = {}
    name_counter = 0
    for slot in slots:
        slot_name = slot["name"]
        if slot["type"] == "name":
            resolved[slot_name] = names[name_counter]
            name_counter += 1
        else:
            resolved[slot_name] = _resolve_slot(slot, template)

    # Fix 4: Handle -ma enclitic before general replacement
    has_ma = _MA_PATTERN.search(tl)
    if has_ma:
        tl = _fix_ma_enclitic(tl, resolved)

    # Replace all slot references in tl and tr
    for slot_name, value in resolved.items():
        tl = tl.replace(f"{{{slot_name}_TL}}", value["tl"])
        tr = tr.replace(f"{{{slot_name}_TR}}", value["tr"])

    return {
        "tl": tl,
        "tr": tr,
        "template_id": template["id"],
        "category": template.get("category", "other"),
    }


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------


def generate(
    n: int,
    category_weights: dict[str, float] | None = None,
    seed: int = 42,
) -> list[dict]:
    """Generate n filled (transliteration, translation) pairs.

    Args:
        n: Number of pairs to generate.
        category_weights: Optional dict mapping category → weight for
            sampling distribution. If None, uniform over all templates.
        seed: Random seed for reproducibility.

    Returns:
        List of dicts with 'tl', 'tr', 'template_id', 'category'.
    """
    random.seed(seed)

    if category_weights:
        categories = list(category_weights.keys())
        weights = [category_weights[c] for c in categories]
    else:
        categories = None

    results = []
    seen: set[tuple[str, str]] = set()
    max_attempts = n * 3  # prevent infinite loop if pool exhausted
    attempts = 0

    while len(results) < n and attempts < max_attempts:
        attempts += 1
        if categories:
            cat = random.choices(categories, weights=weights, k=1)[0]
            pool = _TEMPLATES_BY_CATEGORY.get(cat, _ALL_TEMPLATES)
            tmpl = random.choice(pool)
        else:
            tmpl = random.choice(_ALL_TEMPLATES)

        filled = fill_template(tmpl)
        key = (filled["tl"], filled["tr"])
        if key not in seen:
            seen.add(key)
            results.append(filled)

    return results
