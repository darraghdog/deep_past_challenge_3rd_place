"""
Transliteration normalization utilities.
Converts train format to test format for consistent model input/output.
"""

import re
import unicodedata

# Unicode fraction to decimal mapping (train format uses decimals)
FRACTION_MAP = {
    '½': '0.5',
    '⅓': '0.33333',
    '⅔': '0.66666',
    '¼': '0.25',
    '¾': '0.75',
    '⅕': '0.2',
    '⅖': '0.4',
    '⅗': '0.6',
    '⅘': '0.8',
    '⅙': '0.16666',
    '⅚': '0.83333',
    '⅛': '0.125',
    '⅜': '0.375',
    '⅝': '0.625',
    '⅞': '0.875',
}

# Translation tables for efficient character mapping
SUBSCRIPT_TRANS = str.maketrans(
    {"₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
     "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
     "ₓ": None}  # ₓ is removed (not converted to x) per v2 normalization
)
H_DOT_TRANS = str.maketrans("ḫḪ", "hH")


def normalize_fractions(text: str) -> str:
    """
    Convert Unicode fractions to decimal format to match train.csv format.

    Test data may have: ⅓ ma-na 5 GÍN
    Train data uses:    0.33333 ma-na 5 GÍN
    """
    for frac, decimal in FRACTION_MAP.items():
        text = text.replace(frac, decimal)
    return text


def denormalize_fractions(text: str) -> str:
    """
    Convert decimal fractions back to Unicode fractions for submission.

    Model outputs:  0.5 mina, 1.3333 mina, 0.66666 shekel
    Test expects:   ½ mina, 1 ⅓ mina, ⅔ shekel

    Only converts the 6 base-60 Akkadian fractions (per host):
    ½ (0.5), ¼ (0.25), ⅓ (0.333...), ⅔ (0.666...), ⅙ (0.166...), ⅚ (0.833...)

    Handles floating point artifacts like 1.3333300000000001
    """
    # Order matters: handle standalone 0.x FIRST, then N.x patterns
    # This prevents 0.5 → "0 ½" instead of "½"
    # Also handle floating point errors (e.g., 1.3333300000000001)

    # ⅚ = 0.833... (50 shekels of 60)
    # Match .8333, .83333, .833334, .8333300000001, etc.
    text = re.sub(r'\b0\.8333[340]*[14]?\b', '⅚', text)         # 0.8333... → ⅚
    text = re.sub(r'(\d+)\.8333[340]*[14]?\b', r'\1 ⅚', text)  # 1.8333..., 3.83334 → 1 ⅚

    # ⅙ = 0.166... (10 shekels of 60)
    # Match .1666, .16666, .166667, .1666600000001, etc.
    text = re.sub(r'\b0\.1666[670]*[17]?\b', '⅙', text)         # 0.1666... → ⅙
    text = re.sub(r'(\d+)\.1666[670]*[17]?\b', r'\1 ⅙', text)  # 1.1666..., 1.16667 → 1 ⅙

    # ⅔ = 0.666... (40 shekels of 60)
    # Match .6666, .66666, .666667, .6666600000003, etc.
    text = re.sub(r'\b0\.6666[670]*[137]?\b', '⅔', text)         # 0.6666... → ⅔
    text = re.sub(r'(\d+)\.6666[670]*[137]?\b', r'\1 ⅔', text)  # 1.6666..., 4.66667 → 1 ⅔

    # ⅓ = 0.333... (20 shekels of 60)
    # Match .3333, .33333, .333334, .3333300000001, etc.
    text = re.sub(r'\b0\.3333[340]*[14]?\b', '⅓', text)         # 0.3333... → ⅓
    text = re.sub(r'(\d+)\.3333[340]*[14]?\b', r'\1 ⅓', text)  # 1.3333..., 1.33334 → 1 ⅓

    # ½ = 0.5 (30 shekels of 60)
    text = re.sub(r'\b0\.5\b', '½', text)                  # 0.5 → ½
    text = re.sub(r'(\d+)\.5\b', r'\1 ½', text)           # 1.5 → 1 ½

    # ¼ = 0.25 (15 shekels of 60)
    text = re.sub(r'\b0\.25\b', '¼', text)                 # 0.25 → ¼
    text = re.sub(r'(\d+)\.25\b', r'\1 ¼', text)          # 1.25 → 1 ¼

    return text


def normalize_slash_fractions(text: str) -> str:
    """
    Convert slash fractions to Unicode fractions.

    Training data has: 2/3 mina, 1 1/3 shekels, 5/6 minas, 1 / 4 shekel
    Test expects:      ⅔ mina, 1 ⅓ shekels, ⅚ minas, ¼ shekel

    Only converts the 6 base-60 Akkadian fractions:
    1/2 → ½, 1/4 → ¼, 1/3 → ⅓, 2/3 → ⅔, 1/6 → ⅙, 5/6 → ⅚

    Handles both compact (1/2) and spaced (1 / 2) forms.
    """
    # Order: longer numerators first to avoid partial matches
    # Allow optional spaces around slash: 1/2 or 1 / 2
    text = re.sub(r'\b5\s*/\s*6\b', '⅚', text)
    text = re.sub(r'\b1\s*/\s*6\b', '⅙', text)
    text = re.sub(r'\b2\s*/\s*3\b', '⅔', text)
    text = re.sub(r'\b1\s*/\s*3\b', '⅓', text)
    text = re.sub(r'\b1\s*/\s*4\b', '¼', text)
    text = re.sub(r'\b1\s*/\s*2\b', '½', text)
    return text


def normalize_subscripts(text: str) -> str:
    """
    Convert subscript numbers to regular integers to match test format.

    Train has: qí-bi₄, DU₁₀, il₅
    Test has:  qí-bi4, DU10, il5
    """
    return text.translate(SUBSCRIPT_TRANS)


def normalize_h_dot(text: str) -> str:
    """
    Convert ḫ/Ḫ to h/H - test data doesn't have ḫ character.

    Train has: ḫa, aḫ, Ḫ
    Test has:  ha, ah, H
    """
    return text.translate(H_DOT_TRANS)


def normalize_gaps(text: str) -> str:
    """
    Normalize gap markers in transliteration and translation.

    Per host: test uses <gap> for single missing sign, <big_gap> for multiple.

    Big gaps (multiple signs missing):
        x x x, [x x], [...], …, ……, ... → <big_gap>

    Single gaps:
        x, [x], (x) → <gap>

    Also merges adjacent gaps per public postprocessing.
    """
    # Step 1: Big gaps - multiple signs missing
    # Multiple x's (with spaces or hyphens between them)
    text = re.sub(r'\bx(?:[\s-]+x)+\b', '<big_gap>', text)
    # Consecutive x's without spaces (xx, xxx, xxxx)
    text = re.sub(r'\bxx+\b', '<big_gap>', text)
    # Bracketed ellipsis FIRST (before plain ellipsis)
    text = re.sub(r'\[\.+\]', '<big_gap>', text)          # [..] or [...]
    text = re.sub(r'\[…+\]', '<big_gap>', text)           # […] or [……]
    # Plain ellipsis patterns: ..., …, ……
    text = re.sub(r'\.{3,}', '<big_gap>', text)           # ... or ....
    text = re.sub(r'…+', '<big_gap>', text)               # … or ……

    # Step 2: Single gaps
    # [x] or (x) → <gap>
    text = re.sub(r'\[x\]', '<gap>', text, flags=re.I)
    text = re.sub(r'\(x\)', '<gap>', text, flags=re.I)
    # Standalone x → <gap> (preserves x-kam → <gap>-kam)
    text = re.sub(r'\bx\b', '<gap>', text)

    # Step 3: Merge adjacent gaps (per public postprocessing)
    text = re.sub(r'<gap>\s*<gap>', '<big_gap>', text)
    text = re.sub(r'<big_gap>\s*<big_gap>', '<big_gap>', text)
    text = re.sub(r'<gap>\s*<big_gap>', '<big_gap>', text)
    text = re.sub(r'<big_gap>\s*<gap>', '<big_gap>', text)

    return text


def normalize_determinatives(text: str) -> str:
    """
    Convert parentheses determinatives to curly brackets.

    Per host (Example 1): parentheses used as determinatives should use curly brackets.

    Train has: (d)UTU, šu-(d)EN.LÍL, (m)PN, (TÚG)ṣú-ba-tám
    Test has:  {d}UTU, šu-{d}EN.LÍL, {m}PN, {TÚG}ṣú-ba-tám

    Common determinatives:
    - d = divine (god names)
    - m = male (personal names)
    - f = female (personal names)
    - ki = place names
    - urudu = copper objects
    - TÚG = textile/garment classifier
    """
    # Match (X) where X is 1-5 letters (lowercase or uppercase, including accented)
    # This covers: (d), (m), (f), (ki), (urudu), (TÚG), etc.
    text = re.sub(r'\(([a-zA-ZÀ-ÿ]{1,5})\)', r'{\1}', text)
    return text


def normalize_brackets(text: str) -> str:
    """
    Remove square brackets from editorial insertions in translations.

    Per host (Example 3): bracketed insertions like "[added]" should have
    brackets removed since we want to train the phraseology, not mark
    which words are reconstructed.

    Examples:
        "its excise [added]" → "its excise added"
        "transport fee [paid]" → "transport fee paid"
        "15 shekels of [silver]" → "15 shekels of silver"
        "[my?] temple" → "my? temple"
    """
    # Remove square brackets but keep the content inside
    text = re.sub(r'\[([^\]]*)\]', r'\1', text)
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace and strip trailing dashes.

    Per public postprocessing Step 13:
    - Collapse multiple spaces to single space
    - Strip leading/trailing whitespace
    - Strip leading/trailing dashes

    Examples:
        "text  with   spaces" → "text with spaces"
        "- leading dash" → "leading dash"
        "trailing dash -" → "trailing dash"
    """
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple whitespace
    text = text.strip()               # Strip leading/trailing whitespace
    text = text.strip('-')            # Strip leading/trailing dashes
    text = text.strip()               # Strip any whitespace exposed by dash removal
    return text


def remove_scribal_insertions(text: str) -> str:
    """
    Remove modern scribal/grammatical annotations from translations.

    Per host (Example 2): "We have made sure to remove these from the
    hidden evaluation translations. So it would be best to remove the
    modern scribal insertions when you encounter them."

    Removes patterns like:
        (fem. plur.), (fem.), (plur.), (pl.), (sing.), (singular),
        (plural), (?), (!), (them), (and), etc.

    Examples:
        "you (fem. plur.) have" → "you have"
        "release (them) to" → "release to"
        "reading (?)" → "reading"
    """
    # Remove grammatical annotations: (fem), (plur), (pl), (sing), etc.
    text = re.sub(
        r'\s*\((fem|plur|pl|sing|singular|plural|masc|m|f)\.?\s*\w*\.?\)',
        '',
        text,
        flags=re.I
    )
    # Remove (?) and (!) annotations
    text = re.sub(r'\s*\(\?\)', '', text)
    text = re.sub(r'\s*\(!\)', '', text)
    # Remove common insertions like (them), (and), (here), (sent), etc.
    text = re.sub(r'\s*\((them|and|here|sent|i\.e\.|e\.g\.)\)', '', text, flags=re.I)
    return text


def normalize_special_chars(text: str) -> str:
    """
    Normalize special characters to ASCII equivalents.

    Converts:
        — (em dash) → - (hyphen)
        – (en dash) → - (hyphen)
        ' ' (curly quotes) → ' (straight apostrophe)
        " " (curly double quotes) → " (straight quotes)
        *word* → word (remove italics markers)

    Examples:
        "in Šalatuwar—from it" → "in Šalatuwar-from it"
        "Aššur-rē'ī" → "Aššur-rē'ī"
        "*wawālu*" → "wawālu"
    """
    # Em dash and en dash to hyphen
    text = text.replace('\u2014', '-')  # — em dash
    text = text.replace('\u2013', '-')  # – en dash
    # Curly apostrophes to straight
    text = text.replace('\u2018', "'")  # ' left single quote
    text = text.replace('\u2019', "'")  # ' right single quote
    # Curly double quotes to straight
    text = text.replace('\u201c', '"')  # " left double quote
    text = text.replace('\u201d', '"')  # " right double quote
    # Remove italics markers (*word* → word)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    return text


def normalize_punctuation_spacing(text: str) -> str:
    """
    Fix spacing around punctuation marks.

    Per public postprocessing Step 12:
    - Remove space before punctuation: "text ," → "text,"
    - Remove duplicate punctuation: ".." → "."

    Examples:
        "he said , that" → "he said, that"
        "the end .." → "the end."
        "what ? he" → "what? he"
    """
    # Remove space before punctuation
    text = re.sub(r'\s+([.,;:?!])', r'\1', text)
    # Remove duplicate punctuation
    text = re.sub(r'([.,;:?!])\1+', r'\1', text)
    return text


def normalize_half_brackets(text: str) -> str:
    """
    Remove half-brackets but keep the text inside.

    Half-brackets ⸢⸣ indicate text that is damaged but still legible.
    Unlike square brackets [] which indicate missing/restored text,
    half-brackets mark visible but damaged signs that can be read
    with confidence.

    Since the text IS readable, we simply remove the brackets and
    keep the content.

    Examples:
        "⸢a⸣-na" → "a-na"
        "⸢um-ma⸣" → "um-ma"
        "{⸢d⸣}UTU" → "{d}UTU"
    """
    # Remove half-brackets but keep content
    text = text.replace('⸢', '')
    text = text.replace('⸣', '')
    return text


def normalize_oracc_brackets(text: str) -> str:
    """
    Convert square brackets to <gap> markers for ORACC data.

    In ORACC transliterations, square brackets [] indicate text that
    is completely broken away and restored by modern editors.
    For training, we convert these to <gap> markers.

    Note: This is different from normalize_brackets() which removes
    brackets in TRANSLATIONS. This function is for TRANSLITERATIONS
    from ORACC data specifically.

    Examples:
        "[na]-ab" → "<gap>-ab"
        "a-[na]" → "a-<gap>"
        "[x x]-ma" → "<gap>-ma"
    """
    # Replace bracketed content with <gap>
    text = re.sub(r'\[[^\]]*\]', '<gap>', text)
    return text


def normalize_unmatched_brackets(text: str) -> str:
    """
    Remove unmatched/orphan brackets that remain after other normalization.

    Sometimes text has partial brackets like "text]" or "[text" from
    damaged or truncated source material.

    Examples:
        "a-ha-iš] it-ta" → "a-ha-iš it-ta"
        "[broken text" → "broken text"
    """
    text = text.replace('[', '')
    text = text.replace(']', '')
    return text


def normalize_line_dividers(text: str) -> str:
    """
    Remove or normalize line divider markers.

    Slashes / are sometimes used to mark line breaks in tablets.
    Section marks § indicate document sections.
    These should be removed for training.

    Examples:
        "text / more text" → "text more text"
        "§ section header" → "section header"
    """
    text = text.replace('/', ' ')
    text = text.replace('§', '')
    return text


def normalize_ceiling_brackets(text: str) -> str:
    """
    Remove ceiling brackets ⌈⌉ (similar to half-brackets).

    These are another type of bracket used to indicate damaged text.

    Examples:
        "⌈um-ma⌉" → "um-ma"
    """
    text = text.replace('⌈', '')
    text = text.replace('⌉', '')
    return text


def normalize_figure_dash(text: str) -> str:
    """
    Convert figure dash ‒ (U+2012) to regular hyphen.

    The figure dash is used in some sources but should be a regular hyphen.
    """
    text = text.replace('\u2012', '-')
    return text


def normalize_circumflex_to_macron(text: str) -> str:
    """
    Convert circumflex vowels to macron equivalents (both indicate long vowels).

    Per host's allowed characters, macron vowels (ā ī ū ē) are allowed,
    but circumflex (û î ê) are not listed. Exception: â IS allowed.

    Converts:
        û → ū
        î → ī
        ê → ē
        Û → Ū
        Î → Ī
        Ê → Ē

    Leaves â/Â unchanged (explicitly allowed by host).

    Examples:
        "Sîn" → "Sīn"
        "ikû" → "ikū"
        "bêlu" → "bēlu"
    """
    text = text.replace('û', 'ū')
    text = text.replace('î', 'ī')
    text = text.replace('ê', 'ē')
    text = text.replace('Û', 'Ū')
    text = text.replace('Î', 'Ī')
    text = text.replace('Ê', 'Ē')
    return text


def cdli_to_target(text: str) -> str:
    """Convert CDLI ATF ASCII transliteration to competition target format.

    Based on official Kaggle competition character mapping:
    https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/overview

    Character mappings (CDLI → Target):
        sz → š,  SZ → Š     (shin)
        s, → ṣ,  S, → Ṣ     (emphatic sade)
        t, → ṭ,  T, → Ṭ     (emphatic tet)
        h  → ḫ,  H  → Ḫ     (het) - Note: test set uses h, train uses ḫ
        a2 → á,  a3 → à     (vowel indices 2/3 become accents)
        e2 → é,  e3 → è
        i2 → í,  i3 → ì
        u2 → ú,  u3 → ù

    Numbers 4+ stay as integers (NOT subscripts):
        en6 → en6 (not en₆)

    Order matters: multi-character sequences must be replaced before single chars.
    """
    if not text:
        return text

    result = text

    # Step 1: Multi-character consonant sequences (order matters!)
    consonants = [
        ('sz', 'š'), ('SZ', 'Š'),
        ('s,', 'ṣ'), ('S,', 'Ṣ'),
        ('t,', 'ṭ'), ('T,', 'Ṭ'),
    ]
    for old, new in consonants:
        result = result.replace(old, new)

    # Step 2: Vowels with index 2 (acute accent) and 3 (grave accent)
    # Must match letter+number pattern to avoid false positives
    vowel_accents = [
        ('a2', 'á'), ('A2', 'Á'),
        ('e2', 'é'), ('E2', 'É'),
        ('i2', 'í'), ('I2', 'Í'),
        ('u2', 'ú'), ('U2', 'Ú'),
        ('a3', 'à'), ('A3', 'À'),
        ('e3', 'è'), ('E3', 'È'),
        ('i3', 'ì'), ('I3', 'Ì'),
        ('u3', 'ù'), ('U3', 'Ù'),
    ]
    for old, new in vowel_accents:
        result = result.replace(old, new)

    # Step 3: h → ḫ (CDLI uses plain h for het)
    # Only replace lowercase h that's part of transliteration, not in metadata
    # Be careful: don't replace h inside words like "the" in comments
    # Since we're processing ATF lines, h should be safe to replace
    result = result.replace('h', 'ḫ').replace('H', 'Ḫ')

    # Step 4: Numbers 4+ stay as integers (competition format)
    # No conversion needed - CDLI already uses integers

    return result


def normalize_v2_transliteration(text: str) -> str:
    """Normalize transliteration to match v2 competition format.

    Applies the four v2 host normalizations:
    1. Subscript digits ₀-₉ → ASCII 0-9, ₓ removed
    2. <big_gap> → <gap>
    3. Adjacent <gap> merging (<gap> <gap>, <gap>-<gap> → <gap>)

    Use this to align old data (published_texts, train.csv) with v2 format.
    """
    text = normalize_subscripts(text)
    text = text.replace('<big_gap>', '<gap>')
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r'<gap>\s*<gap>', '<gap>', text)
        text = re.sub(r'<gap>-<gap>', '<gap>', text)
    return text


def normalize_transliteration(text: str) -> str:
    """Apply all normalizations to a transliteration string."""
    text = normalize_fractions(text)
    text = normalize_subscripts(text)
    text = normalize_h_dot(text)
    text = normalize_gaps(text)
    return text


# ---------------------------------------------------------------------------
# Hecker HPM normalization: align OCR output to published_texts.csv format
# ---------------------------------------------------------------------------

# Superscript fractions in Hecker: ¹/₂ ¹/₃ ²/₃ ¹/₄ ¹/₆ ⁵/₆
_SUPERSCRIPT_DIGITS = str.maketrans("¹²³⁴⁵⁶⁷⁸⁹⁰", "1234567890")
_SUBSCRIPT_DIGITS_MAP = str.maketrans("₁₂₃₄₅₆₇₈₉₀", "1234567890")


def normalize_hecker_translit(text: str) -> str:
    """Normalize Hecker HPM OCR transliteration to match published_texts.csv format.

    Applies these transforms (in order):
    1. Superscript fractions: ¹/₂ → 0.5, ²/₃ → 0.66666, etc.
    2. Slash fractions: 1/2 → 0.5, 2/3 → 0.66666, etc.
    3. Subscript digits: ₄ → 4, ₆ → 6
    4. Bare determinatives: dEN-líl → {d}EN-líl, mPN → {m}PN
    5. Lowercase proper nouns (Hecker capitalizes, ref lowercases)
    6. Sign joins: = → - (Hecker uses = for morpheme joins)
    7. Editorial marks: strip !, ?, *, <<...>> corrections
    8. Bracket cleanup: strip <...> angle brackets around corrections
    9. Slash line breaks: / between words → space
    10. Whitespace normalization
    """
    if not text:
        return text

    # 1. Superscript/subscript/mixed fractions (must come before subscript normalization)
    # Handles: ¹/₂, ²/₃, 1/₂, ¹/2, 1/2 etc.
    def _any_frac_replace(m):
        num = m.group(1).translate(_SUPERSCRIPT_DIGITS).translate(_SUBSCRIPT_DIGITS_MAP)
        den = m.group(2).translate(_SUPERSCRIPT_DIGITS).translate(_SUBSCRIPT_DIGITS_MAP)
        return _fraction_to_decimal(num, den)
    # Match any combo of regular/super/subscript digits around /
    text = re.sub(r'([¹²³⁴⁵⁶⁷⁸⁹⁰₁₂₃₄₅₆₇₈₉₀\d]+)/([¹²³⁴⁵⁶⁷⁸⁹⁰₁₂₃₄₅₆₇₈₉₀\d]+)', _any_frac_replace, text)
    # Compound: "1 0.5" → "1.5" (from "1 1/2")
    text = re.sub(r'(\d+)\s+0\.(\d+)', r'\1.\2', text)

    # 3. Subscript digits → ASCII
    text = text.translate(SUBSCRIPT_TRANS)

    # 4. Bare determinatives → curly braces
    # dGOD-name → {d}GOD-name (d followed by uppercase letter)
    # mPERSON → {m}PERSON
    # fPERSON → {f}PERSON
    text = re.sub(r'(?<![a-zA-ZÀ-ÿ{])d([A-ZÈÉÊÀÁÂÌÍÎÙÚÛŠṢṬ])', r'{d}\1', text)
    text = re.sub(r'(?<![a-zA-ZÀ-ÿ{])m([A-ZÈÉÊÀÁÂÌÍÎÙÚÛŠṢṬ])', r'{m}\1', text)
    text = re.sub(r'(?<![a-zA-ZÀ-ÿ{])f([A-ZÈÉÊÀÁÂÌÍÎÙÚÛŠṢṬ])', r'{f}\1', text)

    # 5. Lowercase proper nouns
    # Hecker capitalizes first letter of names: Pu-šu-ke-en → pu-šu-ke-en
    # But keep Sumerograms (all-caps tokens) uppercase: KÙ.BABBAR, DUMU, GÍN
    # Strategy: lowercase any token that has mixed case (upper+lower)
    def _lowercase_mixed(m):
        token = m.group(0)
        # Extract only alpha chars (ignore digits, dots, braces) to check case
        alpha_only = re.sub(r'[^a-zA-ZÀ-ÿ]', '', token)
        if not alpha_only:
            return token
        # All-caps (Sumerograms): keep as-is — KÙ.BABBAR, GÍN, DUMU, PUZUR4
        if alpha_only == alpha_only.upper():
            return token
        # Determinative prefix: keep braces, lowercase rest
        det_match = re.match(r'(\{[dmf]\})(.*)', token)
        if det_match:
            return det_match.group(1) + det_match.group(2).lower()
        # Mixed case: lowercase — Pu-šu → pu-šu, Lá-ma → lá-ma
        return token.lower()

    text = re.sub(r'[^\s]+', _lowercase_mixed, text)

    # 6. Sign joins: = → -
    text = text.replace('=', '-')

    # 7. Editorial marks
    # Strip ! and ? from sign readings (lu!? → lu, ri! → ri)
    text = re.sub(r'([a-zA-ZÀ-ÿ₀-₉])[\!\?]+', r'\1', text)
    # Strip * corrections (*a-na → a-na)
    text = re.sub(r'\*([a-zA-ZÀ-ÿ])', r'\1', text)
    # Remove <<...>> scribal corrections
    text = re.sub(r'<<[^>]*>>', '', text)

    # 8. Angle bracket corrections: <ma> → ma
    text = re.sub(r'<([^>]+)>', r'\1', text)

    # 9. Slash line breaks: / between words → space
    # But not inside compound signs like TÚG.ḪI.A
    text = re.sub(r'\s*/\s*', ' ', text)

    # 10. Whitespace normalization
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def _fraction_to_decimal(num_str: str, den_str: str) -> str:
    """Convert fraction strings to decimal format used in published_texts."""
    try:
        num = int(num_str)
        den = int(den_str)
        if den == 0:
            return f"{num_str}/{den_str}"
        val = num / den
        # Use the same decimal representations as published_texts
        frac_map = {
            (1, 2): '0.5', (1, 3): '0.33333', (2, 3): '0.66666',
            (1, 4): '0.25', (3, 4): '0.75',
            (1, 6): '0.16666', (5, 6): '0.83333',
        }
        return frac_map.get((num, den), f"{val:.5f}".rstrip('0').rstrip('.'))
    except ValueError:
        return f"{num_str}/{den_str}"


# ---------------------------------------------------------------------------
# Post-normalization: convert model output to match updated competition labels
# ---------------------------------------------------------------------------

# Reverse mapping: decimal → Unicode fraction
DECIMAL_TO_FRACTION = {v: k for k, v in FRACTION_MAP.items()}


def post_normalize_remove_quotes(text: str) -> str:
    """Remove all double-quote characters from translation output.

    Host confirmed: 'quotations = removed' in updated labels.
    """
    return text.replace('"', '')


def post_normalize_remove_parentheses(text: str) -> str:
    """Remove parentheses but keep their content.

    Host confirmed: 'parentheses ( ) = removed' in updated labels.
    Examples:
        "(and) he" → "and he"
        "silver (?)" → "silver ?"
    """
    text = text.replace('(', '')
    text = text.replace(')', '')
    return text


def post_normalize_remove_apostrophes(text: str) -> str:
    """Remove all apostrophe characters from translation output.

    Host confirmed: "scare quotes ' ' = removed".
    Also observed: father's → fathers, Aššur's → Aššurs.
    """
    return text.replace("'", '')


def post_normalize_big_gap_to_gap(text: str) -> str:
    """Convert <big_gap> to <gap> and deduplicate adjacent gaps.

    Host confirmed: '<big_gap> replaced with <gap> then deduplicated'.
    """
    text = text.replace('<big_gap>', '<gap>')
    # Deduplicate adjacent <gap> tokens (with optional whitespace between)
    text = re.sub(r'(<gap>\s*)+<gap>', '<gap>', text)
    return text


def post_normalize_gap_spacing(text: str) -> str:
    """Ensure spaces around <gap> markers.

    Host confirmed: 'added space around <gap> in translations'.
    Preserves hyphenated fragments like '<gap>-ilī' for name continuations.
    """
    # Add space before <gap> if preceded by non-space (but not hyphen)
    text = re.sub(r'(?<!\s)(?<!-)(<gap>)', r' \1', text)
    # Add space after <gap> if followed by non-space (but not hyphen)
    text = re.sub(r'(<gap>)(?!\s)(?!-)', r'\1 ', text)
    return text


def post_normalize_fractions_to_decimal(text: str) -> str:
    """Convert Unicode fractions back to decimal format.

    Host: 'returned fractions to decimals, for alignment purposes'.
    Handles compound forms like '1 ⅓' → '1.33333'.

    Off by default — unclear if test LABELS changed.
    """
    # First handle compound: digit + space + fraction → decimal
    for frac_char, decimal_str in DECIMAL_TO_FRACTION.items():
        # "1 ½" → "1.5", "2 ⅓" → "2.33333"
        def _compound_replace(m, dec=decimal_str):
            integer = m.group(1)
            # decimal part without leading "0."
            dec_part = dec[2:]  # e.g. "0.5" → "5", "0.33333" → "33333"
            return f"{integer}.{dec_part}"
        text = re.sub(rf'(\d+)\s+{re.escape(frac_char)}', _compound_replace, text)

    # Then standalone fractions: ½ → 0.5
    for frac_char, decimal_str in DECIMAL_TO_FRACTION.items():
        text = text.replace(frac_char, decimal_str)

    return text


def postprocess_translation(text: str, fractions_to_decimal: bool = False) -> str:
    """Apply all post-normalization steps to model output.

    Converts model output (trained on old format) to match updated
    competition label format. Call this after model inference.

    Args:
        text: Raw model output translation.
        fractions_to_decimal: If True, convert Unicode fractions to decimals.
            Off by default since it's unclear if test labels changed.

    Returns:
        Post-processed translation matching new label format.
    """
    text = post_normalize_remove_quotes(text)
    text = post_normalize_remove_parentheses(text)
    text = post_normalize_remove_apostrophes(text)
    text = post_normalize_big_gap_to_gap(text)
    text = post_normalize_gap_spacing(text)
    if fractions_to_decimal:
        text = post_normalize_fractions_to_decimal(text)
    text = normalize_punctuation_spacing(text)
    text = normalize_whitespace(text)
    return text


def clean_transliteration_chars(text: str) -> str:
    """Remove/replace characters not in the test transliteration charset.

    Targets CAD normalized forms and stray editorial marks.
    """
    # Compose combining diacritics first (e.g., u + combining macron → ū)
    text = unicodedata.normalize('NFC', text)
    # Macron vowels → plain (CAD normalized forms not in test transliterations)
    for src, dst in [('ā', 'a'), ('ī', 'i'), ('ē', 'e'), ('ū', 'u'),
                     ('Ā', 'A'), ('Ī', 'I'), ('Ē', 'E'), ('Ū', 'U')]:
        text = text.replace(src, dst)
    # Circumflex vowels → plain
    for src, dst in [('â', 'a'), ('û', 'u'), ('î', 'i'), ('ê', 'e')]:
        text = text.replace(src, dst)
    # ḥ/Ḥ → h/H (not in test set, different from ḫ→h already handled)
    text = text.replace('ḥ', 'h').replace('Ḥ', 'H')
    # ś → š (wrong diacritic)
    text = text.replace('ś', 'š')
    # Superscript determinative ᵈ → {d}
    text = text.replace('ᵈ', '{d}')
    # Half-brackets (˹˺ variants)
    text = text.replace('˹', '').replace('˺', '')
    text = text.replace('⸢', '').replace('⸣', '')
    # Strip editorial marks: * ? ( ) ,
    text = text.replace('*', '').replace('?', '')
    text = text.replace('(', '').replace(')', '')
    # Apostrophe → aleph (ʾ)
    text = text.replace("'", 'ʾ')
    # <big_gap> → <gap>, deduplicate adjacent gaps
    text = text.replace('<big_gap>', '<gap>')
    while '<gap> <gap>' in text:
        text = text.replace('<gap> <gap>', '<gap>')
    # Guillemets « » → remove
    text = text.replace('«', '').replace('»', '')
    # Superscript letters (ⁱ ˢ ¹) → remove
    for ch in 'ⁱˢ¹':
        text = text.replace(ch, '')
    # Combining below ring ̩ → remove
    text = text.replace('\u0329', '')
    # × → x
    text = text.replace('×', 'x')
    # Turkish/Romanian leftovers
    text = text.replace('Ü', 'U').replace('ü', 'u')
    text = text.replace('ț', 't').replace('Ț', 'T')
    return text


def clean_translation_chars(text: str) -> str:
    """Remove/replace characters not in the test translation charset."""
    text = unicodedata.normalize('NFC', text)
    # Uppercase macron → drop macron, keep uppercase (Ā→A, Ē→E, etc.)
    for src, dst in [('Ā', 'A'), ('Ē', 'E'), ('Ī', 'I'), ('Ū', 'U')]:
        text = text.replace(src, dst)
    # Uppercase accented not in test translations
    for src, dst in [('Í', 'I'), ('Á', 'A'), ('Ú', 'U')]:
        text = text.replace(src, dst)
    # ḥ/Ḥ → h/H
    text = text.replace('ḥ', 'h').replace('Ḥ', 'H')
    # Turkish chars → closest allowed
    text = text.replace('ü', 'u').replace('Ş', 'Š').replace('İ', 'I').replace('Ș', 'Ṣ')
    # Superscript annotations
    for ch in 'ᵖˡᵘʳ':
        text = text.replace(ch, '')
    # <big_gap> → <gap>, deduplicate adjacent gaps
    text = text.replace('<big_gap>', '<gap>')
    while '<gap> <gap>' in text:
        text = text.replace('<gap> <gap>', '<gap>')
    # Combining diacritics that survived NFC (stray combining macron/caron/etc.)
    text = text.replace('\u030c', '').replace('\u0304', '').replace('\u0329', '')
    # Remaining accented uppercase not in test translations
    text = text.replace('Ì', 'I').replace('É', 'E')
    # Stray chars
    text = text.replace('#', '').replace('*', '')
    text = text.replace('ʿ', 'ʾ')  # ayin → aleph
    text = text.replace('ě', 'e').replace('ȟ', 'h')
    text = text.replace('⁄', '/')  # fraction slash → normal slash
    text = text.replace('ü', 'u')
    return text
