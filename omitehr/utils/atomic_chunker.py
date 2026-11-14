"""Create simple, deterministic source chunks from sentence-like note spans.

The chunker keeps note and section provenance, splits only explicit list
structure, and otherwise leaves upstream PyRuSH spans unchanged.  Clinical
content is not interpreted here; filtering is handled separately.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List


_NOTE_HEADER = re.compile(
    r"^\s*NOTE_DATETIME:\s*(?P<datetime>[^|\n]+?)\s*\|\s*"
    r"NOTE_TITLE:\s*(?P<title>[^|\n]+?)"
    r"(?:\s*\|\s*NOTE_CLASS_CONCEPT_ID:\s*[^|\n]+)?\s*(?:\n|$)",
    re.IGNORECASE,
)
_NOTE_HEADER_ONLY = re.compile(
    r"^\s*NOTE_DATETIME:\s*(?P<datetime>[^|\n]+?)\s*\|\s*"
    r"NOTE_TITLE:\s*(?P<title>[^|\n]+?)\s*$",
    re.IGNORECASE,
)
_SECTION_HEADER_ONLY = re.compile(
    r"^\s*(?P<section>[A-Za-z][A-Za-z0-9 /&()_-]{1,80}):\s*(?:\n|$)"
)
_SECTION_PREFIX = re.compile(
    r"^\s*(?P<section>[A-Za-z][A-Za-z0-9 /&()_-]{1,80}):\s*\S"
)
_LINE_BULLET_ITEM = re.compile(
    r"(?m)^\s*(?:[-*•·])\s+(?=\S)"
)
_INLINE_NUMBERED_ITEM = re.compile(
    r"(?:^|\s)(?:\d+|[IVX]+)[.)]\s+(?=\S)", re.IGNORECASE
)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_list_items(text: str) -> List[str]:
    """Split only author-provided bullet or numbered-list boundaries."""
    bullet_matches = list(_LINE_BULLET_ITEM.finditer(text))
    inline_matches = list(_INLINE_NUMBERED_ITEM.finditer(text))
    if bullet_matches:
        matches = bullet_matches
    else:
        # A single number can be ordinary prose; require a true numbered list.
        matches = inline_matches if len(inline_matches) >= 2 else []
    if not matches:
        return [_clean(text)] if _clean(text) else []

    pieces: List[str] = []
    prefix = _clean(text[:matches[0].start()])
    if prefix:
        pieces.append(prefix)
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        item = _clean(text[match.end():end])
        if item:
            pieces.append(item)
    return pieces


def split_atomic_chunks(unit: str, note_title: str = "") -> List[str]:
    """Return simple labelable chunks from one upstream sentence/PyRuSH span.

    ``note_title`` is accepted for backwards-compatible callers but does not
    affect content: note-specific clinical heuristics intentionally do not
    belong in this general chunker.
    """
    del note_title
    return _split_list_items(unit)


def build_atomic_chunks(source_units: Iterable[str]) -> List[Dict[str, object]]:
    """Chunk source spans while preserving note title, time, and section context."""
    chunks: List[Dict[str, object]] = []
    note_title = "Unknown note"
    note_datetime = ""
    section_context = ""
    note_index = -1

    for parent_index, raw_unit in enumerate(source_units):
        unit = str(raw_unit or "")
        header = _NOTE_HEADER.match(unit) or _NOTE_HEADER_ONLY.match(unit)
        if header:
            note_index += 1
            note_title = _clean(header.group("title")) or "Unknown note"
            note_datetime = _clean(header.group("datetime"))
            section_context = ""
            unit = unit[header.end():]

        text = unit.strip()
        if not text:
            continue

        section_match = _SECTION_HEADER_ONLY.match(text)
        if section_match:
            section_context = _clean(section_match.group("section"))
            text = text[section_match.end():].strip()
            # A standalone section heading is context, not a source claim.
            if not text:
                continue
        else:
            section_prefix = _SECTION_PREFIX.match(text)
            if section_prefix:
                section_context = _clean(section_prefix.group("section"))

        for chunk_text in split_atomic_chunks(text, note_title):
            chunks.append({
                "text": chunk_text,
                "parent_index": parent_index,
                "note_title": note_title,
                "note_datetime": note_datetime,
                "note_index": max(0, note_index),
                "section_context": section_context,
            })

    for index, chunk in enumerate(chunks):
        chunk["previous_text"] = chunks[index - 1]["text"] if index else ""
        chunk["next_text"] = chunks[index + 1]["text"] if index + 1 < len(chunks) else ""
    return chunks
