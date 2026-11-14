#!/usr/bin/env python3
"""Remove recurring template and copy-forward text with TRACE.

Admissions are split into notes for TRACE's frequency module, then rebuilt as
visit-level rows with the detected spans removed from ``source_text``.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

NOTE_SEPARATOR = "\n\n-----------------------------\n\n"
NOTE_HEADER_SPLIT = re.compile(r"(?=NOTE_DATETIME:\s*.*?\|\s*NOTE_TITLE:)")
NOTE_DATETIME = re.compile(r"NOTE_DATETIME:\s*([^|\n]+)")


def split_notes(text: str) -> list[str]:
    """Split a flattened admission record into note text blocks."""
    return [chunk.strip() for chunk in NOTE_HEADER_SPLIT.split(text or "") if chunk.strip()]


def build_trace_notes(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Convert visit-level rows to the note-level records expected by TRACE."""
    notes: list[dict[str, Any]] = []
    visit_to_note_ids: dict[str, list[str]] = {}
    for row in rows:
        visit_id = str(row["visit_occurrence_id"])
        note_ids: list[str] = []
        for index, text in enumerate(split_notes(row.get("source_text", ""))):
            note_id = f"{visit_id}:{index}"
            match = NOTE_DATETIME.search(text)
            note_ids.append(note_id)
            notes.append({
                "note_csn_id": note_id,
                "pat_mrn_id": str(row["person_id"]),
                "upd_aut_local_dttm": match.group(1).strip() if match else str(index),
                "full_note_text": text,
                "template_string": "",
                "copyforward": [],
                "visit_occurrence_id": visit_id,
                "person_id": row["person_id"],
            })
        visit_to_note_ids[visit_id] = note_ids
    return notes, visit_to_note_ids


def load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    """Read visit rows, optionally limiting the run for a smoke test."""
    rows = [json.loads(line) for line in path.open() if line.strip()]
    return rows[:limit] if limit is not None else rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=Path, required=True)
    parser.add_argument("--trace_output_jsonl", type=Path, required=True)
    parser.add_argument("--cleaned_output_jsonl", type=Path, required=True)
    parser.add_argument("--metrics_json", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # TRACE is an optional extra.  Keep the module importable (and make
    # ``--help`` useful) when that extra is not installed.
    try:
        from run_trace.trace_unsupervised import UnsupervisedTRACE
        from run_trace.utils import remove_spans_from_text
    except ImportError as error:
        raise RuntimeError(
            "TRACE is required for trace_clean; install it with "
            "python -m pip install -e '.[trace]'"
        ) from error

    rows = load_rows(args.input_jsonl, args.limit)
    notes, visit_to_note_ids = build_trace_notes(rows)
    print(f"Loaded {len(rows)} visits and {len(notes)} notes", flush=True)

    trace = UnsupervisedTRACE()
    trace.load_notes(notes)
    template_spans, copyforward_spans = trace.run()

    args.trace_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.trace_output_jsonl.open("w") as handle:
        for note in notes:
            note["template_spans"] = template_spans.get(note["note_csn_id"], [])
            note["copyforward_spans"] = copyforward_spans.get(note["note_csn_id"], [])
            json.dump(note, handle)
            handle.write("\n")

    by_note = {note["note_csn_id"]: note for note in notes}
    cleaned_rows: list[dict[str, Any]] = []
    total_removed = 0
    notes_changed = 0
    for row in rows:
        cleaned_chunks: list[str] = []
        row_removed = 0
        for note_id in visit_to_note_ids[str(row["visit_occurrence_id"])]:
            note = by_note[note_id]
            spans = note.get("template_spans", []) + note.get("copyforward_spans", [])
            text = note["full_note_text"]
            cleaned = remove_spans_from_text(text, spans)
            row_removed += len(text) - len(cleaned)
            cleaned_chunks.append(cleaned)
        cleaned_row = dict(row)
        cleaned_row["source_text"] = NOTE_SEPARATOR.join(cleaned_chunks)
        cleaned_row["trace_removed_chars"] = row_removed
        cleaned_row["trace_notes_changed"] = int(row_removed > 0)
        cleaned_rows.append(cleaned_row)
        total_removed += row_removed
        notes_changed += int(row_removed > 0)

    with args.cleaned_output_jsonl.open("w") as handle:
        for row in cleaned_rows:
            json.dump(row, handle)
            handle.write("\n")

    metrics = {
        "visits": len(rows),
        "notes": len(notes),
        "notes_with_spans": sum(bool(template_spans.get(n["note_csn_id"]) or copyforward_spans.get(n["note_csn_id"])) for n in notes),
        "visits_changed": notes_changed,
        "removed_chars": total_removed,
        "template_spans": sum(len(v) for v in template_spans.values()),
        "copyforward_spans": sum(len(v) for v in copyforward_spans.values()),
    }
    args.metrics_json.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
