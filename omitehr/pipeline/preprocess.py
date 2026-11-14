#!/usr/bin/env python3
"""Replay and time deployed note splitting, atomic chunking, and filtering."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from omitehr.utils.atomic_chunker import build_atomic_chunks  # noqa: E402
from omitehr.utils.filter import apply_filters  # noqa: E402

NOTE_HEADER_SPLIT = re.compile(r"(?=NOTE_DATETIME:\s*.*?\|\s*NOTE_TITLE:)")
NOTE_SEPARATOR_RE = re.compile(r"\n\s*-{3,}\s*\n")


def note_aware_pyrush(nlp, text: str) -> list[str]:
    """Split each note with PyRuSH and normalize the resulting spans."""
    chunks = [chunk.strip() for chunk in NOTE_HEADER_SPLIT.split(text or "") if chunk.strip()]
    units: list[str] = []
    for document in nlp.pipe(chunks, batch_size=16):
        for sentence in document.sents:
            units.extend(
                re.sub(r"\s+", " ", part).strip()
                for part in NOTE_SEPARATOR_RE.split(sentence.text)
                if part.strip()
            )
    return units


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--max-admissions", type=int)
    parser.add_argument("--exclude-doc-id", action="append", default=[])
    args = parser.parse_args()

    import medspacy
    from loguru import logger
    logger.remove()
    nlp = medspacy.load()
    for pipe_name in list(nlp.pipe_names):
        if pipe_name != "medspacy_pyrush":
            nlp.remove_pipe(pipe_name)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    mismatches = 0
    excluded = {str(doc_id) for doc_id in args.exclude_doc_id}
    with args.input_jsonl.open() as source, args.output_jsonl.open("w") as output:
        for line in source:
            if not line.strip():
                continue
            if str(json.loads(line).get("doc_id")) in excluded:
                continue
            if args.max_admissions is not None and processed >= args.max_admissions:
                break
            row = json.loads(line)
            started = time.perf_counter()
            source_sentences = note_aware_pyrush(nlp, str(row.get("source_text") or ""))
            chunks = build_atomic_chunks(source_sentences)
            rebuilt = dict(row)
            if rebuilt.get("doc_id") is None:
                fallback_doc_id = rebuilt.get("visit_occurrence_id")
                if fallback_doc_id is None:
                    raise ValueError("input row needs doc_id or visit_occurrence_id")
                rebuilt["doc_id"] = str(fallback_doc_id)
            # Raw inference exports often contain ``target_text`` (or no
            # target at all) but not the sentence-aligned field used by the
            # shared filters.  Targets are not needed for inference, so use
            # an empty aligned list when they were not pre-split.
            if not isinstance(rebuilt.get("target_sentences"), list):
                rebuilt["target_sentences"] = []
            rebuilt["source_sentences"] = [chunk["text"] for chunk in chunks]
            masks = apply_filters(rebuilt)
            rebuilt["source_valid_mask"] = masks["source_mask"]
            rebuilt["source_sentences_valid"] = [
                text for text, keep in zip(rebuilt["source_sentences"], masks["source_mask"]) if keep
            ]
            rebuilt["atomic_chunk_metadata"] = [
                chunk for chunk, keep in zip(chunks, masks["source_mask"]) if keep
            ]
            # Rechunking invalidates any prior sentence-aligned labels/context.
            for field in ("entailment_labels", "contextual_label_values", "contextual_chunk_labels", "contextual_labeling_metadata",
                          "source_sentence_contexts", "labeling_status"):
                rebuilt.pop(field, None)
            rebuilt["benchmark_preprocessing_seconds"] = time.perf_counter() - started
            previous = row.get("source_sentences_valid")
            rebuilt["benchmark_preprocessing_matches_existing"] = (
                previous == rebuilt["source_sentences_valid"] if isinstance(previous, list) else None
            )
            mismatches += rebuilt["benchmark_preprocessing_matches_existing"] is False
            output.write(json.dumps(rebuilt, ensure_ascii=False) + "\n")
            processed += 1
    print(json.dumps({"admissions": processed, "excluded_doc_ids": sorted(excluded), "preprocessing_mismatches": mismatches,
                      "output": str(args.output_jsonl)}, indent=2))


if __name__ == "__main__":
    main()
