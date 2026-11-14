"""High-precision filters for non-clinical source and summary text.

Rules are deterministic, independent of labels and scores, and applied
identically across data splits.
"""

import re
from typing import List


# ============================================================================
# Source: backchannel noise (phonetic acknowledgments, never clinical)
# ============================================================================

# Verbal fillers with no semantic content in any context. Excludes
# "yes", "no", "correct", "sure", "right", "absolutely", "definitely",
# which can confirm/deny clinical information in dialogue.
BACKCHANNEL_EXACT = {
    "uh-huh", "uh huh", "uhuh",
    "mm-hmm", "mm hmm", "mmhmm", "mhm", "mmhm",
    "um", "uh", "um-hm", "umhm",
    "hmm", "hm",
}


# ============================================================================
# Both: pure divider/separator lines (formatting only, never clinical)
# ============================================================================

# Lines made entirely of repeated separator characters, e.g.
# "------------------------", "========", "****", "___". Requires the
# whole line (3+ repeats) to be just these chars, so a real sentence
# with an em-dash or underscore mixed into text won't match.
_SEPARATOR_LINE = re.compile(r'^[-=_*~]{3,}\s*$')


# ============================================================================
# Source: discharge-note structural tags
# ============================================================================

# Angle-bracket section headers with no clinical content after the tag.
# BHC notes use <TAG> for section breaks; only filter when what follows
# is empty or just a deidentification placeholder (___). Doesn't match
# tags with clinical content like <ALLERGIES> Codeine or
# <CHIEF COMPLAINT> confusion.
_BHC_EMPTY_TAG = re.compile(
    r'^<[A-Z][A-Z\s/&\-()]+>\s*(_+\.?\s*)?$'
)

# Tags that never carry clinical content regardless of what follows:
# <SEX> M/F (demographic), <ATTENDING> ___ (provider name),
# <SERVICE> MEDICINE/SURGERY (hospital service).
_BHC_ADMIN_TAG = re.compile(
    r'^<(?:SEX|ATTENDING|SERVICE)>\s*\S*\.?\s*$'
)


# ============================================================================
# Summary: structural artifacts (formatting, not content)
# ============================================================================

# Full-line markdown headers -- the whole line has to be the header,
# no clinical content mixed in. Both patterns are end-anchored.
HEADER_PATTERNS = [
    r"^\*\*[A-Z][A-Za-z\s/&\-:()]+\*\*[\.:]*$",   # **ASSESSMENT AND PLAN**  **Brief Hospital Course (BHC) Narrative**.
    r"^\*\*[A-Za-z\s/]+:\*\*\.?$",                  # **Tests/Procedures:**
    r"^[A-Z][A-Z\s/&\-]+:\s*$",                     # IMPRESSION:  FINDINGS:  (plain-text radiology headers)
]

_header_regexes = [re.compile(p) for p in HEADER_PATTERNS]

# Placeholder patterns
PLACEHOLDER_PATTERNS = [
    r"\[Current Date\]",
    r"\[Redacted\]",
    r"\[Your Name\]",
    r"\[Doctor'?s?\s*Name\]",
    r"\[Patient'?s?\s*Name\]",
    r"\[DATE\]",
    r"\[NAME\]",
]

_placeholder_regexes = [re.compile(p, re.IGNORECASE) for p in PLACEHOLDER_PATTERNS]

# Signature/disclaimer patterns
SIGNATURE_PATTERNS = [
    r"^\*\*Signature",
    r"^\*\*Disclaimer",
    r"^Signature:",
    r"^Disclaimer:",
]

_signature_regexes = [re.compile(p, re.IGNORECASE) for p in SIGNATURE_PATTERNS]


# Source: high-precision note-format and boilerplate artifacts.  These are
# deliberately narrower than a general "short sentence" rule: short diagnoses,
# medication names, and list items can be meaningful once their note context is
# available.  The goal is to remove units which cannot be a clinical claim.
_SOURCE_HEADER_ONLY = re.compile(
    r"^[A-Z][A-Z0-9 /&()\-]{2,}:\s*$"
)
_SOURCE_HEADER_LIST_MARKER = re.compile(
    r"^(?:IMPRESSION|FINDINGS|OPERATION(?:S)? PERFORMED|"
    r"POSTOPERATIVE DIAGNOSES|PREOPERATIVE DIAGNOSES|PAST MEDICAL HISTORY|"
    r"PROCEDURES?|SPECIMENS?|IMPLANTS?|SUMMARY OF PROCEDURES PERFORMED):"
    r"\s*(?:\d+|[IVX]+)[.)]?\s*$",
    re.IGNORECASE,
)
_NOTE_TITLE_ONLY = re.compile(
    r"^(?:[A-Za-z]+\s+){0,5}(?:note|report|consult|assessment|plan|orders?)\s*:?$",
    re.IGNORECASE,
)
_ATTESTATION_BOILERPLATE = [
    re.compile(r"^I have personally reviewed (?:the )?(?:images|imaging|tracings).*"
               r"agree with (?:the )?report", re.IGNORECASE),
    re.compile(r"^(?:this )?(?:report|note) (?:has been )?electronically signed", re.IGNORECASE),
    re.compile(r"^electronically signed by\b", re.IGNORECASE),
]
_SOURCE_TECHNICAL_DISCLAIMER = [
    re.compile(r"^please contact the .+clinical laboratory for any questions", re.IGNORECASE),
    re.compile(r"^this laboratory is certified under the clinical laboratory improvement amendments", re.IGNORECASE),
    re.compile(r"^the fda does not require this test to go through premarket", re.IGNORECASE),
    re.compile(r"^it has not been cleared or approved by the (?:u\.?s\.? )?food and drug administration", re.IGNORECASE),
    re.compile(r"^it should not be regarded as investigational or for research", re.IGNORECASE),
    re.compile(r"^this test is used for clinical purposes", re.IGNORECASE),
    re.compile(r"^the (?:frozen section )?remnant is submitted in cassette", re.IGNORECASE),
    re.compile(r"^two additional fragments are frozen for potential future", re.IGNORECASE),
]
_SOURCE_TRIAGE_TEMPLATE = [
    re.compile(r"^the appropriate initial studies and treatment (?:was|were) ordered\b", re.IGNORECASE),
    re.compile(r"^the patient was taken to have (?:her|his) studies initiated\b", re.IGNORECASE),
]


# ============================================================================
# Core filter functions
# ============================================================================

def _strip_speaker_tag(text: str) -> str:
    """Remove speaker tags like [doctor], [patient], [nurse] from start of text."""
    return re.sub(r'^\[(?:doctor|patient|nurse|provider|clinician)\]\s*', '', text, flags=re.IGNORECASE)


def _normalize_source(text: str) -> str:
    """Lowercase, strip speaker tags, whitespace, and trailing punctuation."""
    text = text.strip()
    text = _strip_speaker_tag(text)
    text = text.lower()
    text = re.sub(r'[.!?,;:]+$', '', text).strip()
    return text


def is_valid_summary_sentence(sentence: str) -> bool:
    """Filter for target summary sentences V(X). Drops empty/whitespace
    strings, pure separator lines, full-line markdown headers,
    placeholder-only sentences, and signature/disclaimer blocks. Keeps
    short diagnoses, numbered clinical items, and anything else with
    actual content."""
    text = sentence.strip()

    if not text:
        return False

    # Pure divider/separator lines ("------", "====", "****")
    if _SEPARATOR_LINE.match(text):
        return False

    # Full-line markdown headers (both patterns are end-anchored)
    for regex in _header_regexes:
        if regex.match(text):
            return False

    # Placeholder-only: remove all placeholders, check if anything remains
    stripped = text
    for regex in _placeholder_regexes:
        stripped = regex.sub('', stripped)
    remaining = re.sub(r'[\*\s:.\-_]+', '', stripped)
    if not remaining:
        return False

    # Signature/disclaimer blocks
    for regex in _signature_regexes:
        if regex.match(text):
            return False

    return True


def is_valid_source_sentence(sentence: str, dataset: str = "aci") -> bool:
    """Filter for source sentences U(X). Drops empty/whitespace strings,
    pure separator lines, and backchannel noise ("uh-huh", "mm-hmm").
    Keeps "yes"/"no"/"okay"/etc. (can confirm or deny clinical content),
    greetings/pleasantries, and anything with actual clinical content."""
    text = sentence.strip()

    if not text:
        return False

    # Pure divider/separator lines ("------", "====", "****")
    if _SEPARATOR_LINE.match(text):
        return False

    # A header, or a header accidentally glued to only a list marker (for
    # example "IMPRESSION: 1."), has no standalone proposition to judge.
    if _SOURCE_HEADER_ONLY.match(text) or _SOURCE_HEADER_LIST_MARKER.match(text):
        return False

    # A note/report title is provenance, not a source fact.  Do not reject a
    # title followed by real content; that should be handled as a structured
    # context bundle downstream.
    if _NOTE_TITLE_ONLY.match(text):
        return False

    if any(pattern.match(text) for pattern in _ATTESTATION_BOILERPLATE):
        return False

    # Regulatory, assay-validation, and specimen-processing statements cannot
    # be hospital-course evidence.  Keep result/diagnostic statements; only
    # remove the fixed technical/disclaimer language itself.
    if any(pattern.match(text) for pattern in _SOURCE_TECHNICAL_DISCLAIMER):
        return False

    if any(pattern.match(text) for pattern in _SOURCE_TRIAGE_TEMPLATE):
        return False

    normalized = _normalize_source(text)

    if not normalized:
        return False

    # Backchannel noise only
    if normalized in BACKCHANNEL_EXACT:
        return False

    # Bare list numbers ("1.", "2.", "23.") -- formatting artifacts from numbered lists
    if re.match(r'^\d+\.\s*$', text):
        return False

    # Roman numeral list markers ("II.", "III.", "IV.") -- same as above
    if re.match(r'^[IVX]+\.\s*$', text):
        return False

    # Deidentification placeholders ("___", "___.")
    if re.match(r'^_+\.?\s*$', text):
        return False

    # PubMed citation placeholders ("<cit> .", "<cit>.")
    if re.match(r'^<cit>\s*\.?\s*$', text):
        return False

    # PubMed bare anonymized numbers ("<dig> .", "<dig>.")
    if re.match(r'^<dig>\s*\.?\s*$', text):
        return False

    # BHC angle-bracket section tags with no clinical content, e.g.
    # <SOCIAL HISTORY> ___, <FOLLOWUP INSTRUCTIONS> ___, <ATTENDING> ___
    # (not <ALLERGIES> Codeine, <CHIEF COMPLAINT> confusion)
    if _BHC_EMPTY_TAG.match(text):
        return False

    # BHC administrative tags, e.g. <SEX> M/F, <ATTENDING> ___, <SERVICE> MEDICINE
    if _BHC_ADMIN_TAG.match(text):
        return False

    # omop specific filters
    if re.match(r'^NOTE_DATETIME:\s*.+\|\s*NOTE_TITLE:\s*.+$', text):
        return False

    # Attending physician attestation boilerplate
    if "ATTENDING PHYSICIAN ATTESTATION" in text.upper():
        return False

    # Request ID lines: "Request ID: [0000...]"
    if re.match(r'^Request ID:\s*\[?\d+\]?\s*$', text):
        return False

    return True


def apply_filters(doc: dict, dataset: str = "aci") -> dict:
    """Compute filter masks for a document without modifying it.

    Returns dict with:
        summary_mask: List[bool] - True = valid, for each target sentence
        source_mask: List[bool] - True = valid, for each source sentence
        summary_removed: List[str] - removed summary sentences
        source_removed: List[str] - removed source sentences
    """
    summary_mask = [
        is_valid_summary_sentence(s) for s in doc['target_sentences']
    ]
    source_mask = [
        is_valid_source_sentence(s, dataset) for s in doc['source_sentences']
    ]

    summary_removed = [
        s for s, valid in zip(doc['target_sentences'], summary_mask) if not valid
    ]
    source_removed = [
        s for s, valid in zip(doc['source_sentences'], source_mask) if not valid
    ]

    return {
        'summary_mask': summary_mask,
        'source_mask': source_mask,
        'summary_removed': summary_removed,
        'source_removed': source_removed,
    }


def filter_arrays(arrays: dict, mask: List[bool]) -> dict:
    """Apply a boolean mask to multiple parallel arrays.

    Args:
        arrays: dict of {name: list} where all lists have same length as mask
        mask: boolean list, True = keep

    Returns:
        dict with same keys, filtered arrays
    """
    return {
        name: [v for v, m in zip(arr, mask) if m]
        for name, arr in arrays.items()
    }
