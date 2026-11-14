#!/usr/bin/env python3
"""Select deduplicated high-salience evidence for BHC omission review.

The stage joins stored salience scores to admissions that already contain a
generated BHC, then retains the top candidates after semantic deduplication.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omitehr.utils.dedupe_post_hoc import (
    DEFAULT_MODEL as DEFAULT_DEDUPE_MODEL,
    dedupe_indices,
    embed_sentences,
    load_model as load_dedupe_model,
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_salience_predictions(path: Path) -> dict[str, dict[int, float]]:
    """Load one salience score per source-sentence index."""
    predictions: dict[str, dict[int, float]] = {}
    for line_number, row in enumerate(load_jsonl(path), 1):
        doc_id = str(row.get("doc_id"))
        try:
            index, score = int(row["sentence_idx"]), float(row["score"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid salience row {line_number}") from error
        if index < 0 or not math.isfinite(score):
            raise ValueError(f"invalid salience row {line_number}")
        by_document = predictions.setdefault(doc_id, {})
        if index in by_document:
            raise ValueError(f"duplicate prediction for {doc_id!r}, sentence {index}")
        by_document[index] = score
    return predictions


def build_salient_source_text(row: dict, indices: list[int], sentences: list[str]) -> str:
    """Render selected source sentences with available note provenance."""
    metadata = row.get("atomic_chunk_metadata")
    aligned = metadata if isinstance(metadata, list) and len(metadata) == len(sentences) else None
    chunks = []
    for index in indices:
        descriptors = []
        provenance = aligned[index] if aligned else {}
        if isinstance(provenance, dict):
            if str(provenance.get("note_title") or "").strip():
                descriptors.append(f"note={str(provenance['note_title']).strip()}")
            if str(provenance.get("section_context") or "").strip():
                descriptors.append(f"section={str(provenance['section_context']).strip()}")
        header = f"[Source sentence {index}" + ("; " + "; ".join(descriptors) if descriptors else "") + "]"
        chunks.append(f"{header}\n{sentences[index]}")
    return "\n\n".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--salience-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--dedupe-threshold", type=float, default=0.95)
    parser.add_argument("--dedupe-model", default=DEFAULT_DEDUPE_MODEL)
    parser.add_argument("--dedupe-device", default="cuda")
    parser.add_argument("--dedupe-batch-size", type=int, default=32)
    parser.add_argument("--dedupe-max-length", type=int, default=512)
    parser.add_argument("--dedupe-cache-dir", type=Path)
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if not 0 < args.dedupe_threshold <= 1:
        parser.error("--dedupe-threshold must be in (0, 1]")

    rows = load_jsonl(args.input_jsonl)
    if not rows or any(not str(row.get("llm_target_text") or "").strip() for row in rows):
        raise ValueError("every input row must contain a nonempty llm_target_text")
    predictions = load_salience_predictions(args.salience_jsonl)
    tokenizer, model, device = load_dedupe_model(args.dedupe_model, args.dedupe_device)
    selected = []
    for row in rows:
        started = time.perf_counter()
        doc_id = str(row.get("doc_id"))
        sentences = row.get("source_sentences_valid")
        scores_by_index = predictions.get(doc_id)
        if not isinstance(sentences, list) or scores_by_index is None:
            raise ValueError(f"missing source sentences or predictions for {doc_id!r}")
        if set(scores_by_index) != set(range(len(sentences))):
            raise ValueError(f"salience indices do not align for {doc_id!r}")
        scores = [scores_by_index[index] for index in range(len(sentences))]
        ranked = sorted(range(len(sentences)), key=lambda index: (-scores[index], index))
        embeddings = embed_sentences(
            [str(sentences[index]) for index in ranked], tokenizer, model, device,
            batch_size=args.dedupe_batch_size, max_length=args.dedupe_max_length,
        )
        retained = dedupe_indices(
            [str(sentences[index]) for index in ranked], embeddings,
            args.dedupe_threshold, priorities=[scores[index] for index in ranked],
            max_kept=args.top_k,
        )
        indices = sorted(ranked[position] for position in retained)
        output = dict(row)
        output["salience_scores"] = scores
        output["salient_sentence_indices"] = indices
        output["salient_sentences"] = [sentences[index] for index in indices]
        output["salient_sentence_scores"] = [scores[index] for index in indices]
        output["salient_source_text"] = build_salient_source_text(output, indices, sentences)
        output["salience_selection_metadata"] = {
            "format_version": "salience_selection_v1",
            "prediction_path": str(args.salience_jsonl),
            "score_field": "score",
            "index_field": "sentence_idx",
            "index_convention": "zero_based_index_into_source_sentences_valid",
            "top_k": args.top_k,
            "min_score": None,
            "selected_count": len(indices),
            "deduplication": {
                "method": "pairwise_cosine_similarity",
                "threshold": args.dedupe_threshold,
                "embedding_model": args.dedupe_model,
                "embedding_pooling": "mean",
                "embedding_max_length": args.dedupe_max_length,
                "embedding_cache_hit": False,
                "candidate_count_before_deduplication": len(ranked),
            },
        }
        output["benchmark_selection_dedupe_seconds"] = time.perf_counter() - started
        selected.append(output)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(args.output_jsonl)
    print(f"Wrote {len(selected)} rows -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
