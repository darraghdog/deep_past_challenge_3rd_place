"""Constraint tables for OA template-based data generation.

These tables encode which slot values are compatible with each other,
preventing nonsensical combinations like "5 talents of refined silver"
or "washed textiles."

Used by the fill engine (Step 4) when generating (transliteration, translation) pairs.
"""

import re

# ---------------------------------------------------------------------------
# 1. COMMODITY PROFILES
#    Each commodity defines: valid units, valid qualifiers, realistic amount
#    ranges per unit, and whether it uses compound amounts.
# ---------------------------------------------------------------------------

COMMODITY_PROFILES = {
    "silver": {
        "tl": "KÙ.BABBAR",
        "tr": "silver",
        "units": [
            {"tl": "GÍN", "tr": "shekels", "range": (1, 60), "compound_with": None},
            {"tl": "ma-na", "tr": "minas", "range": (1, 50), "compound_with": "GÍN"},
            {"tl": "GÚ", "tr": "talents", "range": (1, 5), "compound_with": "ma-na"},
        ],
        "qualifiers": [
            {"tl": "", "tr": ""},
            {"tl": "ṣa-ru-pá-am", "tr": "refined"},
            {"tl": "SIG5", "tr": "good"},
            {"tl": "li-ti", "tr": "tested"},
        ],
        "interest_unit": {"tl": "GÍN.TA", "tr": "shekels"},
        "interest_base": {"tl": "ma-na-im", "tr": "mina"},
    },
    "tin": {
        "tl": "AN.NA",
        "tr": "tin",
        "units": [
            {"tl": "GÍN", "tr": "shekels", "range": (1, 60), "compound_with": None},
            {"tl": "ma-na", "tr": "minas", "range": (1, 60), "compound_with": "GÍN"},
            {"tl": "GÚ", "tr": "talents", "range": (1, 10), "compound_with": "ma-na"},
        ],
        "qualifiers": [
            {"tl": "", "tr": ""},
            {"tl": "qá-tim", "tr": "loose"},
        ],
        "interest_unit": None,
        "interest_base": None,
    },
    "copper": {
        "tl": "URUDU",
        "tr": "copper",
        "units": [
            {"tl": "ma-na", "tr": "minas", "range": (1, 300), "compound_with": None},
            {"tl": "GÚ", "tr": "talents", "range": (1, 20), "compound_with": "ma-na"},
        ],
        "qualifiers": [
            {"tl": "", "tr": ""},
            {"tl": "SIG5", "tr": "good"},
            {"tl": "ma-sí-am", "tr": "washed"},
            {"tl": "ší-ku-um", "tr": "šikku-"},
            {"tl": "lá-mu-nam", "tr": "bad"},
            {"tl": "ša-bu-ra-am", "tr": "broken"},
            {"tl": "ṣa-la-mì-im", "tr": "black"},
        ],
        "interest_unit": {"tl": "ma-na.TA", "tr": "minas"},
        "interest_base": {"tl": "GÚ", "tr": "talent"},
        "interest_alt": {
            "unit": {"tl": "ma-na.TA", "tr": "minas"},
            "base": {"tl": "10 ma-na-e", "tr": "10 minas"},
            "note": "Per-10-minas pattern for smaller copper debts",
        },
    },
    "gold": {
        "tl": "KÙ.GI",
        "tr": "gold",
        "units": [
            {"tl": "GÍN", "tr": "shekels", "range": (1, 30), "compound_with": None},
            {"tl": "ma-na", "tr": "minas", "range": (1, 10), "compound_with": "GÍN"},
        ],
        "qualifiers": [
            {"tl": "", "tr": ""},
            {"tl": "pá-ša-lam", "tr": "pašallum"},
        ],
        "interest_unit": None,
        "interest_base": None,
    },
    "textiles": {
        "tl": "TÚG",
        "tr": "textiles",
        "units": [
            {"tl": "", "tr": "", "range": (1, 300), "compound_with": None},  # plain count
        ],
        "qualifiers": [
            {"tl": "", "tr": ""},
            {"tl": "SIG5", "tr": "good"},
            {"tl": "ra-qá-tim", "tr": "thin"},
            {"tl": "ša qá-tim", "tr": "ordinary"},
        ],
        "interest_unit": None,
        "interest_base": None,
    },
    "kutanu": {
        "tl": "ku-ta-nu",
        "tr": "kutānu-textiles",
        "units": [
            {"tl": "", "tr": "", "range": (1, 200), "compound_with": None},  # plain count
        ],
        "qualifiers": [
            {"tl": "", "tr": ""},
            {"tl": "SIG5", "tr": "fine"},
            {"tl": "šu-ru-tum", "tr": "dark"},
            {"tl": "a-bar-ni-am", "tr": "Abarnian"},
        ],
        "interest_unit": None,
        "interest_base": None,
    },
    "wool": {
        "tl": "SÍG.HI.A",
        "tr": "wool",
        "units": [
            {"tl": "ma-na", "tr": "minas", "range": (1, 60), "compound_with": None},
            {"tl": "GÚ", "tr": "talents", "range": (1, 30), "compound_with": "ma-na"},
        ],
        "qualifiers": [
            {"tl": "", "tr": ""},
        ],
        "interest_unit": None,
        "interest_base": None,
    },
    "donkeys": {
        "tl": "ANŠE",
        "tr": "donkeys",
        "units": [
            {"tl": "", "tr": "", "range": (1, 30), "compound_with": None},  # plain count
        ],
        "qualifiers": [
            {"tl": "", "tr": ""},
            {"tl": "ṣa-lá-mu", "tr": "black"},
        ],
        "interest_unit": None,
        "interest_base": None,
    },
    "barley": {
        "tl": "ŠE-am",
        "tr": "barley",
        "units": [
            {"tl": "na-ru-uq", "tr": "sacks", "range": (1, 50), "compound_with": None},
            {"tl": "DUG", "tr": "jars", "range": (1, 10), "compound_with": None},
        ],
        "qualifiers": [{"tl": "", "tr": ""}],
        "interest_unit": None,
        "interest_base": None,
    },
    "wheat": {
        "tl": "GIG",
        "tr": "wheat",
        "units": [
            {"tl": "na-ru-uq", "tr": "sacks", "range": (1, 50), "compound_with": None},
            {"tl": "DUG", "tr": "jars", "range": (1, 10), "compound_with": None},
        ],
        "qualifiers": [{"tl": "", "tr": ""}],
        "interest_unit": None,
        "interest_base": None,
    },
    "iron": {
        "tl": "KÙ.AN",
        "tr": "meteoric iron",
        "units": [
            {"tl": "GÍN", "tr": "shekels", "range": (1, 30), "compound_with": None},
            {"tl": "ma-na", "tr": "minas", "range": (1, 5), "compound_with": "GÍN"},
        ],
        "qualifiers": [{"tl": "", "tr": ""}],
        "interest_unit": None,
        "interest_base": None,
    },
    "sheep": {
        "tl": "UDU",
        "tr": "sheep",
        "units": [
            {"tl": "", "tr": "", "range": (1, 25), "compound_with": None},
        ],
        "qualifiers": [{"tl": "", "tr": ""}],
        "interest_unit": None,
        "interest_base": None,
    },
    "bread": {
        "tl": "NINDA",
        "tr": "loaves of bread",
        "units": [
            {"tl": "", "tr": "", "range": (1, 100), "compound_with": None},
        ],
        "qualifiers": [{"tl": "", "tr": ""}],
        "interest_unit": None,
        "interest_base": None,
    },
    "hides": {
        "tl": "maš-ku",
        "tr": "hides",
        "units": [
            {"tl": "", "tr": "", "range": (1, 200), "compound_with": None},
        ],
        "qualifiers": [{"tl": "", "tr": ""}],
        "interest_unit": None,
        "interest_base": None,
    },
    "lapis_lazuli": {
        "tl": "hu-sà-ri-im",
        "tr": "lapis lazuli",
        "units": [
            {"tl": "GÍN", "tr": "shekels", "range": (1, 30), "compound_with": None},
            {"tl": "ma-na", "tr": "minas", "range": (1, 20), "compound_with": "GÍN"},
        ],
        "qualifiers": [{"tl": "", "tr": ""}],
        "interest_unit": None,
        "interest_base": None,
    },
}

# Commodity categories for slot constraints
WEIGHT_COMMODITIES = {"silver", "tin", "copper", "gold", "wool", "iron", "lapis_lazuli", "barley", "wheat"}
COUNT_COMMODITIES = {"textiles", "kutanu", "donkeys", "sheep", "bread", "hides"}


# ---------------------------------------------------------------------------
# 2. FRACTIONS
#    All attested Unicode fractions with TL and TR forms.
# ---------------------------------------------------------------------------

FRACTIONS = [
    {"value": 0.5, "unicode": "½"},
    {"value": 0.333, "unicode": "⅓"},
    {"value": 0.667, "unicode": "⅔"},
    {"value": 0.25, "unicode": "¼"},
    {"value": 0.167, "unicode": "⅙"},
    {"value": 0.833, "unicode": "⅚"},
]


# ---------------------------------------------------------------------------
# 3. INTEREST RATES
#    Attested rates with weights for sampling (common rates sampled more).
# ---------------------------------------------------------------------------

INTEREST_RATES = [
    {"tl": "½ GÍN.TA", "tr": "½ shekel", "weight": 5},
    {"tl": "1 GÍN.TA", "tr": "1 shekel", "weight": 12},
    {"tl": "1 ½ GÍN.TA", "tr": "1 ½ shekels", "weight": 51},
    {"tl": "2 GÍN.TA", "tr": "2 shekels", "weight": 6},
    {"tl": "3 GÍN.TA", "tr": "3 shekels", "weight": 9},
    {"tl": "⅓ GÍN.TA", "tr": "⅓ shekel", "weight": 2},
    {"tl": "¼ GÍN.TA", "tr": "¼ shekel", "weight": 2},
    {"tl": "⅚ GÍN.TA", "tr": "⅚ shekel", "weight": 1},
    {"tl": "1 ⅓ GÍN.TA", "tr": "1 ⅓ shekels", "weight": 2},
    {"tl": "1 ¼ GÍN.TA", "tr": "1 ¼ shekels", "weight": 1},
    {"tl": "⅔ GÍN.TA", "tr": "⅔ shekel", "weight": 1},
    {"tl": "1 ⅙ GÍN.TA", "tr": "1 ⅙ shekels", "weight": 2},
]

# Colony regulation formula (no explicit rate)
COLONY_INTEREST = {
    "tl": "ki-ma a-wa-at kà-ri-im ṣí-ib-tám ú-ṣa-áb",
    "tr": "he will add interest in accordance with the rule of the colony",
}


# ---------------------------------------------------------------------------
# 4. INSTITUTION-PLACE CONSTRAINTS
#    Which institutions exist in which places.
# ---------------------------------------------------------------------------

INSTITUTIONS = {
    "colony": {
        "tl": "kà-ri-im",
        "tr": "colony",
        "valid_places": [
            {"tl": "Kà-ni-iš", "tr": "Kaneš"},
            {"tl": "Wa-ah-šu-ša-na", "tr": "Wahšušana"},
            {"tl": "Dur4-hu-mì-it", "tr": "Durhumit"},
            {"tl": "Ha-ah-hu-um", "tr": "hahhum"},
            {"tl": "Za-al-pa", "tr": "Zalpa"},
            {"tl": "Pu-ru-uš-ha-dim", "tr": "Purušhaddum"},
            {"tl": "Hu-ra-ma", "tr": "hurama"},
            {"tl": "Ti-mì-il5-ki-a", "tr": "Timilkiya"},
            {"tl": "Ku-bu-ur-na-at", "tr": "Kuburnat"},
            {"tl": "Te-ga-ra-ma", "tr": "Tegarama"},
            {"tl": "Ta-wi-ni-a", "tr": "Tawiniya"},
            {"tl": "Ha-tu-uš", "tr": "Hattuš"},
        ],
    },
    "trading_station": {
        "tl": "wa-bar-tim",
        "tr": "trading station",
        "valid_places": [
            {"tl": "Za-al-pa", "tr": "Zalpa"},
            {"tl": "Ku-bu-ur-na-at", "tr": "Kuburnat"},
            {"tl": "Ša-la-tù-ar", "tr": "Šalatuar"},
            {"tl": "Ma-ma", "tr": "Mamma"},
            {"tl": "Wa-áš-ha-ni-a", "tr": "Wašhaniya"},
            {"tl": "Ba-ad-na", "tr": "Badna"},
        ],
    },
    "palace": {
        "tl": "É.GAL-lim",
        "tr": "palace",
        "valid_places": [
            {"tl": "Kà-ni-iš", "tr": "Kaneš"},
            {"tl": "Wa-ah-šu-ša-na", "tr": "Wahšušana"},
            {"tl": "Pu-ru-uš-ha-dim", "tr": "Purušhaddum"},
            {"tl": "Bu-ru-uš-ha-dim", "tr": "Burušhaddum"},
        ],
    },
    "city": {
        "tl": "a-lim{ki}",
        "tr": "the City",
        "valid_places": [
            {"tl": "A-šùr", "tr": "Aššur"},
        ],
        "note": "a-lim{ki} = Aššur always. Not used with other cities.",
    },
    "colony_office": {
        "tl": "É kà-ri-im",
        "tr": "the office of the colony",
        "valid_places": [
            {"tl": "Kà-ni-iš", "tr": "Kaneš"},
            {"tl": "Wa-ah-šu-ša-na", "tr": "Wahšušana"},
            {"tl": "Dur4-hu-mì-it", "tr": "Durhumit"},
            {"tl": "Pu-ru-uš-ha-dim", "tr": "Purušhaddum"},
        ],
    },
    "inn": {
        "tl": "É wa-áb-ri",
        "tr": "the inn",
        "valid_places": [
            {"tl": "Hu-ra-ma", "tr": "hurama"},
            {"tl": "Ha-ah-hu-um", "tr": "hahhum"},
            {"tl": "Ti-mì-il5-ki-a", "tr": "Timilkiya"},
            {"tl": "Bu-ru-ul-lim", "tr": "Burullum"},
            {"tl": "Ša-la-tù-ar", "tr": "Šalatuar"},
            {"tl": "Ša-mu-ha", "tr": "Samuha"},
            {"tl": "Wa-áš-ha-ni-a", "tr": "Wašhaniya"},
            {"tl": "Ku-sà-ra", "tr": "Kuššara"},
            {"tl": "Ha-na-ak-na-ak", "tr": "Hanaknak"},
        ],
    },
}

# Places that can appear freely (no institution constraint)
ALL_PLACES = [
    {"tl": "a-lim{ki}", "tr": "the City"},
    {"tl": "Kà-ni-iš", "tr": "Kaneš"},
    {"tl": "Wa-ah-šu-ša-na", "tr": "Wahšušana"},
    {"tl": "Za-al-pa", "tr": "Zalpa"},
    {"tl": "Pu-ru-uš-ha-dim", "tr": "Purušhaddum"},
    {"tl": "Hu-ra-ma", "tr": "hurama"},
    {"tl": "Ku-bu-ur-na-at", "tr": "Kuburnat"},
    {"tl": "Dur4-hu-mì-it", "tr": "Durhumit"},
    {"tl": "Ha-ah-hu-um", "tr": "hahhum"},
    {"tl": "Ti-mì-il5-ki-a", "tr": "Timilkiya"},
    {"tl": "Ša-la-tù-ar", "tr": "Šalatuar"},
    {"tl": "Ma-ma", "tr": "Mamma"},
    {"tl": "Ku-sà-ra", "tr": "Kuššara"},
    {"tl": "Wa-áš-ha-ni-a", "tr": "Wašhaniya"},
    {"tl": "Bu-ru-ul-lim", "tr": "Burullum"},
    {"tl": "Ša-mu-ha", "tr": "Samuha"},
    {"tl": "Ne-na-ša", "tr": "Nenaša"},
    {"tl": "Ha-tu-uš", "tr": "Hattuš"},
    {"tl": "Am-ku-wa", "tr": "Amkuwa"},
    {"tl": "Lu-hu-za-di-a", "tr": "Luhusaddiya"},
    {"tl": "Te-ga-ra-ma", "tr": "Tegarama"},
    {"tl": "Ha-na-ak-na-ak", "tr": "Hanaknak"},
    {"tl": "Ti-iš-mur-na", "tr": "Tišmurna"},
    {"tl": "Ta-wi-ni-a", "tr": "Tawiniya"},
    {"tl": "Ba-ad-na", "tr": "Badna"},
]


# ---------------------------------------------------------------------------
# 5. TAX BUNDLES
#    Which taxes go together and which commodities they apply to.
# ---------------------------------------------------------------------------

TAX_BUNDLES = {
    "import_transport": {
        "tl": "ni-is-ha-sú DIRI ša-du-a-sú ša-bu",
        "tr": "its import duty added, its transport tariff paid",
        "applies_to": ["silver", "tin", "gold", "copper"],
        "note": "Standard pair for long-distance trade goods",
    },
    "import_only": {
        "tl": "ni-is-ha-tum",
        "tr": "import duty",
        "applies_to": ["textiles", "kutanu", "tin", "donkeys"],
    },
    "export_tax": {
        "tl": "wa-ṣí-tám",
        "tr": "export duty",
        "applies_to": ["silver", "gold"],
    },
    "tithe": {
        "tl": "10-tum",
        "tr": "tithe",
        "applies_to": ["silver", "tin", "copper", "iron", "textiles"],
    },
    "five_percent": {
        "tl": "me-tum 5",
        "tr": "5 percent tax",
        "applies_to": ["textiles", "kutanu"],
    },
    "entrance_tax": {
        "tl": "e-ri-ib-tum",
        "tr": "entrance tax",
        "applies_to": ["tin", "silver", "copper"],
    },
}


# ---------------------------------------------------------------------------
# 6. COMPOUND AMOUNT RULES
#    How weight tiers compose into multi-part amounts.
# ---------------------------------------------------------------------------

COMPOUND_RULES = {
    # tier_name: {unit_tl, unit_tr, sub_tier, per_unit (how many sub-units in one unit)}
    "talent": {"tl": "GÚ", "tr": "talents", "sub_tier": "mina", "per_unit": 60},
    "mina": {"tl": "ma-na", "tr": "minas", "sub_tier": "shekel", "per_unit": 60},
    "shekel": {"tl": "GÍN", "tr": "shekels", "sub_tier": "grain", "per_unit": 180},
    "grain": {"tl": "ŠE", "tr": "grains", "sub_tier": None, "per_unit": None},
}

# Valid compound tier chains (which tiers can stack)
VALID_TIER_CHAINS = [
    ["talent", "mina"],                  # 2 GÚ 15 ma-na
    ["talent", "mina", "shekel"],        # 2 GÚ 15 ma-na 8 GÍN (rare)
    ["mina", "shekel"],                  # 3 ma-na 5 GÍN (very common)
    ["shekel", "grain"],                 # 8 ⅔ GÍN 15 ŠE
    ["mina", "shekel", "grain"],         # rare but attested
    ["talent", "mina", "shekel", "grain"],  # 27 docs attest this
]

# Subtraction patterns: N LÁ FRAC = N minus FRAC
# TL: 4 LÁ ¼ GÍN → TR: 3 ¾ shekels (sometimes kept as "4 minus ¼ shekels")
SUBTRACTION_NOTE = "LÁ means 'minus'. Sometimes TR converts (4 LÁ ¼ = 3 ¾), sometimes keeps explicit."


# ---------------------------------------------------------------------------
# 7. MONTHS
# ---------------------------------------------------------------------------

MONTHS = [
    {"num": 1, "tl": "be-el-tí-É.GAL-lim", "tr": "Bēlat-ekallim", "tr_num": "month 1"},
    {"num": 2, "tl": "ša sá-ra-tim", "tr": "Ša-sarrātim", "tr_num": "month 2"},
    {"num": 2, "tl": "ša sà-ra-tim", "tr": "Ša-sarrātim", "tr_num": "month 2"},
    {"num": 3, "tl": "ke-na-tim", "tr": "Kēnātim", "tr_num": "month 3"},
    {"num": 3, "tl": "ša ke-na-tim", "tr": "Ša-kēnātim", "tr_num": "month 3"},
    {"num": 4, "tl": "ma-hu-ur-DINGIR", "tr": "Mahhur-ilī", "tr_num": "month 4"},
    {"num": 4, "tl": "ma-hu-ur-ì-lí", "tr": "Mahhur-ilī", "tr_num": "month 4"},
    {"num": 5, "tl": "áb-ša-ra-ni", "tr": "Ab-šarrāni", "tr_num": "month 5"},
    {"num": 5, "tl": "áb ša-ra-ni", "tr": "Ab-šarrāni", "tr_num": "month 5"},
    {"num": 5, "tl": "áb-ša-ra-nu", "tr": "Ab-šarrāni", "tr_num": "month 5"},
    {"num": 6, "tl": "hu-bu-ur", "tr": "Hubur", "tr_num": "month 6"},
    {"num": 7, "tl": "ṣí-ip-im", "tr": "Ṣip'um", "tr_num": "month 7"},
    {"num": 8, "tl": "qá-ra-a-tim", "tr": "Qarra'ātum", "tr_num": "month 8"},
    {"num": 8, "tl": "qá-ra-a-tí", "tr": "Qarra'ātum", "tr_num": "month 8"},
    {"num": 9, "tl": "kán-wár-ta", "tr": "Kanwarta", "tr_num": "month 9"},
    {"num": 9, "tl": "kán-bar-ta", "tr": "Tanbarta", "tr_num": "month 9"},
    {"num": 9, "tl": "Kà-an-ma-ar-ta", "tr": "Kanwarta", "tr_num": "month 9"},
    {"num": 10, "tl": "té-i-na-tim", "tr": "Te'inātum", "tr_num": "month 10"},
    {"num": 11, "tl": "ku-zal-li", "tr": "Kuzallu", "tr_num": "month 11"},
    {"num": 11, "tl": "ku-zal-lu", "tr": "Kuzallu", "tr_num": "month 11"},
    {"num": 12, "tl": "a-lá-na-tim", "tr": "Allanātum", "tr_num": "month 12"},
    {"num": 12, "tl": "a-lá-na-tum", "tr": "Allanātum", "tr_num": "month 12"},
]


# ---------------------------------------------------------------------------
# 8. PENALTY TIERS
#    Fixed penalty amounts (typically in gold).
# ---------------------------------------------------------------------------

PENALTY_AMOUNTS = [
    {"tl": "1 ma-na KÙ.GI", "tr": "1 mina of gold"},
    {"tl": "2 ma-na KÙ.GI", "tr": "2 minas of gold"},
    {"tl": "3 ma-na KÙ.GI", "tr": "3 minas of gold"},
    {"tl": "5 ma-na KÙ.GI", "tr": "5 minas of gold"},
    {"tl": "6 ma-na KÙ.GI", "tr": "6 minas of gold"},
    {"tl": "10 ma-na KÙ.GI", "tr": "10 minas of gold"},
    {"tl": "1 ma-na KÙ.BABBAR", "tr": "1 mina of silver"},
    {"tl": "2 ma-na KÙ.BABBAR", "tr": "2 minas of silver"},
    {"tl": "3 ma-na KÙ.BABBAR", "tr": "3 minas of silver"},
    {"tl": "5 ma-na KÙ.BABBAR", "tr": "5 minas of silver"},
    {"tl": "10 ma-na KÙ.BABBAR", "tr": "10 minas of silver"},
]


# ---------------------------------------------------------------------------
# 9. NAME CASING RULES
#    How to transform names for TL lowercase convention.
# ---------------------------------------------------------------------------

# Known mixed-case Sumerograms in our data that need uppercasing.
# Only Ištar/Ìštar and Puzur variants — all other Sumerograms in our
# data are already all-uppercase (DINGIR, UTU, MAN, LUGAL, etc.).
_MIXED_SUMEROGRAM_RE = re.compile(r"^(Ištar|Ìštar|Puzur\d*|PUZUR\d*)$", re.IGNORECASE)


def lowercase_tl_name(name: str) -> str:
    """Convert a TL name to official convention: syllabic lowercase, Sumerograms uppercase.

    Rule for each hyphen-separated element:
      1. All-uppercase alpha (2+ chars) → keep (Sumerogram: DINGIR, SIG5, MAN)
      2. Matches known mixed-case Sumerogram (Ištar, Puzur4) → uppercase it
      3. Determinative {d}/{f}/{ki}/{m} → keep, process the rest
      4. Everything else → lowercase (syllabic)

    Examples:
        En-nam-A-šùr     → en-nam-a-šùr
        {d}UTU-ba-ni     → {d}UTU-ba-ni
        A-mur-Ištar      → a-mur-IŠTAR
        Puzur4-A-šur     → PUZUR4-a-šur
        SIG5-pí-A-šur    → SIG5-pí-a-šur
        DÙG-ṣí-lá-A-šur → DÙG-ṣí-lá-a-šur
        KIŠIB            → KIŠIB
    """
    parts = name.split("-")
    result = []
    for part in parts:
        # Handle determinative prefix: {d}X, {f}X, {ki}X, {m}X
        det = ""
        rest = part
        if part.startswith("{"):
            close = part.find("}")
            if close != -1:
                det = part[: close + 1]
                rest = part[close + 1 :]

        if not rest:
            result.append(det)
            continue

        # Extract only alpha characters (ignore digits, subscripts, dots)
        alpha = re.sub(r"[\d₀-₉.]+", "", rest)

        if not alpha:
            # Pure digits/subscripts — keep as-is
            result.append(det + rest)
            continue

        if alpha.upper() == alpha and len(alpha) >= 2:
            # Already all-uppercase with 2+ alpha chars → Sumerogram, keep
            result.append(det + rest)
        elif _MIXED_SUMEROGRAM_RE.match(rest.split(".")[0]):
            # Known mixed-case Sumerogram → uppercase it
            result.append(det + rest.upper())
        else:
            # Syllabic → lowercase
            result.append(det + rest.lower())

    return "-".join(result)


def lowercase_tr_h_name(name: str) -> str:
    """Lowercase TR names starting with h (from original ḫ).

    Hinnāya → hinnāya
    Hahhum → hahhum
    Hurama → hurama
    Aššur-malik → Aššur-malik (no change, not h-initial)
    """
    if name and name[0] == "H":
        return name[0].lower() + name[1:]
    return name


# ---------------------------------------------------------------------------
# 10. KINSHIP TERMS
# ---------------------------------------------------------------------------

KINSHIP_TERMS = [
    {"tl": "DUMU", "tr": "son", "possessive_tl": "DUMU-a", "possessive_tr": "my son"},
    {"tl": "a-hi", "tr": "brother", "possessive_tl": "a-hu-a", "possessive_tr": "my brother"},
    {"tl": "a-bi", "tr": "father", "possessive_tl": "a-bi-a", "possessive_tr": "my father"},
    {"tl": "be-lí", "tr": "lord", "possessive_tl": "be-lí-a", "possessive_tr": "my lord"},
    {"tl": "a-ha-tí", "tr": "sister", "possessive_tl": "a-ha-tí-a", "possessive_tr": "my sister"},
    {"tl": "um-mì", "tr": "mother", "possessive_tl": "um-mì-a", "possessive_tr": "my mother"},
    {"tl": "DAM", "tr": "wife", "possessive_tl": "a-ša-tí-a", "possessive_tr": "my wife"},
    {"tl": "DUMU.MÍ", "tr": "daughter", "possessive_tl": "DUMU.MÍ-a", "possessive_tr": "my daughter"},
]


# ---------------------------------------------------------------------------
# 11. OCCUPATION TITLES
# ---------------------------------------------------------------------------

OCCUPATIONS = [
    {"tl": "DAM.GÀR", "tr": "merchant"},
    {"tl": "DUB.SAR", "tr": "scribe"},
    {"tl": "um-me-a-ni", "tr": "investor"},
    {"tl": "ra-bi-ṣú-um", "tr": "representative"},
    {"tl": "kà-ṣa-ru-um", "tr": "packer"},
    {"tl": "ṣú-ha-ru-um", "tr": "servant"},
    {"tl": "ša-qí-il5 da-tim", "tr": "datu-payer"},
    {"tl": "sa-hi-ir", "tr": "junior merchant"},
    {"tl": "ma-lá-hi-im", "tr": "skipper"},
    {"tl": "aš-kà-pí-im", "tr": "leatherworker"},
    {"tl": "a-lá-hi-nim", "tr": "miller"},
    {"tl": "sà-an-gu-um", "tr": "priest"},
    {"tl": "lá-ki-dí-im", "tr": "runner"},
    {"tl": "NU.BANDA", "tr": "overseer"},
]


# ---------------------------------------------------------------------------
# 12. CONTAINER TYPES
# ---------------------------------------------------------------------------

CONTAINERS = [
    {"tl": "na-ru-uq", "tr": "sack", "tr_pl": "sacks", "contents": ["textiles", "barley", "wheat", "tin"]},
    {"tl": "né-pí-šu-um", "tr": "package", "tr_pl": "packages", "contents": ["silver", "gold"]},
    {"tl": "ri-ik-sú-um", "tr": "bundle", "tr_pl": "bundles", "contents": ["silver", "tin"]},
    {"tl": "ta-ma-lá-ku", "tr": "tablet container", "tr_pl": "tablet containers", "contents": ["tablets"]},
    {"tl": "DUG", "tr": "jar", "tr_pl": "jars", "contents": ["barley", "wheat", "beer", "oil", "malt"]},
    {"tl": "i-lá-tim", "tr": "container", "tr_pl": "containers", "contents": ["copper"]},
]


# ---------------------------------------------------------------------------
# 13. EXCHANGE RATES (tin-to-silver, copper-to-silver)
# ---------------------------------------------------------------------------

EXCHANGE_RATES = {
    "tin_to_silver": [
        {"tl": "8 GÍN.TA", "tr": "at a rate of 8 shekels each"},
        {"tl": "14 ½ GÍN.TA", "tr": "at a rate of 14 ½ shekels each"},
        {"tl": "15 GÍN.TA", "tr": "at a rate of 15 shekels each"},
        {"tl": "15 ⅔ GÍN.TA", "tr": "at a rate of 15 ⅔ shekels each"},
        {"tl": "16 GÍN.TA", "tr": "at a rate of 16 shekels each"},
        {"tl": "16 ½ GÍN.TA", "tr": "at a rate of 16 ½ shekels each"},
        {"tl": "17 GÍN.TA", "tr": "at a rate of 17 shekels each"},
    ],
    "gold_to_silver": [
        {"tl": "8 GÍN.TA", "tr": "at a rate of 8 shekels each"},
        {"tl": "9 ⅓ GÍN.TA", "tr": "at a rate of 9 ⅓ shekels each"},
    ],
    "copper_to_silver": [
        {"tl": "60 GÍN.TA", "tr": "at a rate of 60 to 1"},
        {"tl": "65 GÍN.TA", "tr": "at a rate of 65 to 1"},
        {"tl": "70 GÍN.TA", "tr": "at a rate of 70 to 1"},
        {"tl": "75 GÍN.TA", "tr": "at a rate of 75 to 1"},
    ],
}


# ---------------------------------------------------------------------------
# 14. PAYMENT DEADLINE RANGES
# ---------------------------------------------------------------------------

DEADLINE_RANGES = {
    "weeks": {"tl_unit": "ha-am-ša-tim", "tr_unit": "weeks", "range": (2, 70)},
    "months": {"tl_unit": "ITU.KAM", "tr_unit": "months", "range": (1, 12)},
    "days": {"tl_unit": "u4-me-e", "tr_unit": "days", "range": (2, 30)},
    "years": {"tl_unit": "ša-na-at", "tr_unit": "year(s)", "range": (1, 5)},
}


# ---------------------------------------------------------------------------
# 15. FESTIVAL NAMES (for payment deadlines)
# ---------------------------------------------------------------------------

FESTIVALS = [
    {"tl": "ha-ar-pé-e", "tr": "harvest"},
    {"tl": "bu-qú-nim", "tr": "shearing"},
    {"tl": "pa-ha-ar ad-ri-im", "tr": "gathering of the threshing floor"},
    {"tl": "Ni-pá-as", "tr": "Nipais festival"},
    {"tl": "ú-sú-me-e", "tr": "Usmue festival"},
    {"tl": "pá-ar-kà", "tr": "Parka festival"},
    {"tl": "A-na-a", "tr": "Anna festival"},
]


# ---------------------------------------------------------------------------
# 16. PURPOSE / EXPENSE CATEGORIES
# ---------------------------------------------------------------------------

EXPENSE_PURPOSES = [
    {"tl": "tí-ib-nim", "tr": "straw"},
    {"tl": "ú-ku-ul-tim", "tr": "food/provisions"},
    {"tl": "É wa-áb-ri", "tr": "the inn"},
    {"tl": "e-ri-qí-im", "tr": "the wagon"},
    {"tl": "ig-ri-im", "tr": "hire"},
    {"tl": "ni-qí-i-šu", "tr": "his sacrifices"},
    {"tl": "ik-ri-bi", "tr": "votive offerings"},
    {"tl": "bu-sà-lim", "tr": "cooking"},
    {"tl": "e-re-qí-im qá-nu-e", "tr": "a wagonload of reeds"},
    {"tl": "ša-du-a-sú", "tr": "transport tariff"},
    {"tl": "ni-is-ha-tim", "tr": "import tax"},
    {"tl": "lu-bu-ší", "tr": "clothing"},
    {"tl": "be-ú-lá-at", "tr": "working capital"},
    {"tl": "da-a-sú", "tr": "road tax"},
]


# ---------------------------------------------------------------------------
# 17. ITEM POOL (for legal "item" slots — things that can be sold/guaranteed)
# ---------------------------------------------------------------------------

ITEM_POOL = [
    {"tl": "É-tim", "tr": "house"},
    {"tl": "A.ŠÀ", "tr": "field"},
    {"tl": "SAG.GÉME", "tr": "slave girl"},
    {"tl": "SAG.ÌR", "tr": "slave"},
    {"tl": "e-ri-qí-im", "tr": "wagon"},
    {"tl": "ANŠE", "tr": "donkey"},
    {"tl": "TÚG.HI.A", "tr": "textiles"},
    {"tl": "KÙ.BABBAR", "tr": "silver"},
    {"tl": "AN.NA", "tr": "tin"},
    {"tl": "URUDU", "tr": "copper"},
]
