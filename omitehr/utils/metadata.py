"""Recover aligned note and sentence metadata from admission JSONL rows."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


NOTE_HEADER_RE = re.compile(
    r"NOTE_DATETIME:\s*([^|\n]+?)\s*\|\s*NOTE_TITLE:\s*([^|\n]+?)"
    r"(?:\s*\|\s*NOTE_CLASS_CONCEPT_ID:\s*[^|\n]+)?(?:\s|$)",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
SECTION_NAMES = (
    "other", "history", "assessment_plan", "physical_exam", "labs_results",
    "medications", "procedure", "diagnosis", "followup_disposition",
)
METADATA_GROUPS = ("position", "timing", "repetition", "length", "title", "section")
NUMERIC_GROUP_INDICES = {"position": (0, 1, 2), "timing": (3, 4),
                         "repetition": (5,), "length": (6,)}


def parse_metadata_groups(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    unknown = set(values).difference(METADATA_GROUPS)
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown metadata groups: {sorted(unknown)}")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("Metadata groups must not be repeated")
    return values


@dataclass(frozen=True)
class DocumentMetadata:
    numeric: np.ndarray
    note_title_bucket: np.ndarray
    section_id: np.ndarray
    note_index: np.ndarray
    local_indices: np.ndarray
    note_title: tuple[str, ...]
    sentence_text: tuple[str, ...]


def parse_datetime(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.strip()[:19])
    except ValueError:
        return None


def title_bucket(title: str, n_buckets: int) -> int:
    if n_buckets < 2:
        raise ValueError("n_buckets must be at least 2")
    if not title.strip():
        return 0
    digest = hashlib.sha256(title.strip().lower().encode()).digest()
    return 1 + int.from_bytes(digest[:4], "big") % (n_buckets - 1)


def section_id(text: str, section_context: str = "") -> int:
    """Map a preserved section header to one of nine model categories."""
    lower = (section_context or text).lower().strip()
    if re.search(r"\b(assessment|impression|plan|a/p)\b", lower[:160]):
        return SECTION_NAMES.index("assessment_plan")
    if re.search(r"\b(history|hpi|chief complaint|indications|past medical)\b", lower[:160]):
        return SECTION_NAMES.index("history")
    if re.search(r"\b(physical exam|examination|vital signs|general:)\b", lower[:160]):
        return SECTION_NAMES.index("physical_exam")
    if re.search(r"\b(lab|result|imaging|x-ray|ct |mri|culture)\b", lower[:160]):
        return SECTION_NAMES.index("labs_results")
    if re.search(r"\b(medication|medications|administered|dose|mg\b|mcg\b)\b", lower[:160]):
        return SECTION_NAMES.index("medications")
    if re.search(r"\b(procedure|operation|surgery|performed|delivery)\b", lower[:160]):
        return SECTION_NAMES.index("procedure")
    if re.search(r"\b(diagnosis|diagnoses|problem list)\b", lower[:160]):
        return SECTION_NAMES.index("diagnosis")
    if re.search(r"\b(follow.?up|disposition|discharge|return precautions)\b", lower[:160]):
        return SECTION_NAMES.index("followup_disposition")
    return SECTION_NAMES.index("other")


def normalized_text(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


def local_context_indices(note_indices: Sequence[int], window: int) -> np.ndarray:
    if window < 0:
        raise ValueError("window must be nonnegative")
    output = np.full((len(note_indices), 2 * window), -1, dtype=np.int64)
    by_note: dict[int, list[int]] = collections.defaultdict(list)
    for sentence_index, note_index_value in enumerate(note_indices):
        by_note[int(note_index_value)].append(sentence_index)
    for positions in by_note.values():
        for offset, sentence_index in enumerate(positions):
            neighbors = positions[max(0, offset - window):offset]
            neighbors += positions[offset + 1:offset + 1 + window]
            output[sentence_index, :len(neighbors)] = neighbors
    return output


def _recover_retained_metadata(
    row: Mapping[str, Any], valid_sentences: Sequence[str],
) -> list[tuple[str, int, dt.datetime | None, str, str]]:
    """Return sentence, note index/time/title, and section for each valid chunk."""
    provenance = row.get("atomic_chunk_metadata")
    if isinstance(provenance, list) and len(provenance) == len(valid_sentences):
        retained = []
        note_index = -1
        previous_key: tuple[str, str] | None = None
        for sentence, item in zip(valid_sentences, provenance):
            if not isinstance(item, Mapping):
                raise ValueError(f"doc_id={row.get('doc_id')} has invalid atomic_chunk_metadata")
            note_datetime = str(item.get("note_datetime") or "").strip()
            note_title = str(item.get("note_title") or "").strip().lower()
            key = (note_datetime, note_title)
            if key != previous_key:
                note_index += 1
                previous_key = key
            retained.append((str(sentence), note_index, parse_datetime(note_datetime),
                             note_title, str(item.get("section_context") or "")))
        return retained

    all_sentences = row.get("source_sentences", [])
    valid_mask = row.get("source_valid_mask", [])
    if len(all_sentences) != len(valid_mask):
        raise ValueError(f"doc_id={row.get('doc_id')} has unaligned source sentences/mask")
    retained = []
    valid_index = 0
    current_time: dt.datetime | None = None
    current_title = ""
    current_note_index = -1
    for sentence, keep in zip(all_sentences, valid_mask):
        header = NOTE_HEADER_RE.search(str(sentence))
        if header:
            current_note_index += 1
            current_time = parse_datetime(header.group(1))
            current_title = header.group(2).strip().lower()
        if keep:
            if valid_index >= len(valid_sentences):
                raise ValueError(f"doc_id={row.get('doc_id')} valid mask selects too many sentences")
            retained.append((str(valid_sentences[valid_index]), current_note_index,
                             current_time, current_title, ""))
            valid_index += 1
    if valid_index != len(valid_sentences):
        raise ValueError(
            f"doc_id={row.get('doc_id')} mask selected {valid_index} sentences, "
            f"expected {len(valid_sentences)}"
        )
    return retained


def extract_document_metadata(
    row: Mapping[str, Any], local_window: int = 2, title_buckets: int = 128,
) -> DocumentMetadata:
    valid_sentences = row.get("source_sentences_valid", [])
    retained = _recover_retained_metadata(row, valid_sentences)
    summary_time = parse_datetime(row.get("discharge_summary_datetime") or row.get("note_datetime"))
    counts = collections.Counter(normalized_text(item[0]) for item in retained)
    times = [item[2] for item in retained if item[2] is not None]
    first_time = min(times) if times else None
    admission_end = summary_time or (max(times) if times else None)
    admission_length_days = (
        max(0.0, (admission_end - first_time).total_seconds() / 86400.0)
        if admission_end is not None and first_time is not None else 0.0
    )
    max_note_index = max((item[1] for item in retained), default=0)
    note_sentence_counts = collections.Counter(item[1] for item in retained)
    note_offsets: collections.Counter[int] = collections.Counter()
    numeric, title_ids, sections, note_indices = [], [], [], []
    for text, note_idx, note_time, note_title, section_context in retained:
        offset = note_offsets[note_idx]
        note_offsets[note_idx] += 1
        note_position = offset / max(1, note_sentence_counts[note_idx] - 1)
        relative_note = max(note_idx, 0) / max(1, max_note_index)
        since_start = (
            max(0.0, (note_time - first_time).total_seconds() / 86400.0)
            if note_time is not None and first_time is not None else 0.0
        )
        to_discharge = (
            max(0.0, (summary_time - note_time).total_seconds() / 86400.0)
            if note_time is not None and summary_time is not None else 0.0
        )
        numeric.append([
            2.0 * note_position - 1.0,
            math.log1p(admission_length_days),
            relative_note,
            math.log1p(since_start),
            math.log1p(to_discharge),
            math.log1p(counts[normalized_text(text)]),
            math.log1p(len(text)),
        ])
        title_ids.append(title_bucket(note_title, title_buckets))
        sections.append(section_id(text, section_context))
        note_indices.append(note_idx)
    n_sentences = len(retained)
    return DocumentMetadata(
        numeric=np.asarray(numeric, dtype=np.float32).reshape(n_sentences, 7),
        note_title_bucket=np.asarray(title_ids, dtype=np.int64),
        section_id=np.asarray(sections, dtype=np.int64),
        note_index=np.asarray(note_indices, dtype=np.int64),
        local_indices=local_context_indices(note_indices, local_window),
        note_title=tuple(item[3] for item in retained),
        sentence_text=tuple(item[0] for item in retained),
    )
