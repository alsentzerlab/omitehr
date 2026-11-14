#!/usr/bin/env python3
"""Rank uncovered, salient source sentences for BHC review.

This deterministic step combines coverage labels with salience scores. It
preserves all candidates for audit and writes likely omissions in descending
salience order.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_external_scores(
    path: Optional[Path], score_field: str, index_field: str,
) -> Dict[Tuple[str, int], float]:
    if path is None:
        return {}
    scores: Dict[Tuple[str, int], float] = {}
    for line_number, row in enumerate(load_jsonl(path), 1):
        doc_id = str(row.get("doc_id"))
        index = row.get(index_field)
        score = row.get(score_field)
        if index is None or score is None:
            raise ValueError(
                f"{path}:{line_number}: missing {index_field!r} or {score_field!r}"
            )
        key = (doc_id, int(index))
        if key in scores:
            raise ValueError(f"duplicate salience prediction for {key}")
        scores[key] = float(score)
    return scores


def _numeric_scores(values: Any, expected: int, field: str) -> Optional[List[float]]:
    if not isinstance(values, list) or len(values) != expected:
        return None
    try:
        return [float(value) for value in values]
    except (TypeError, ValueError) as error:
        raise ValueError(f"field {field!r} contains a nonnumeric score") from error


def get_candidate_scores(
    row: Mapping[str, Any], candidate_count: int,
    source_indices: List[int], external_scores: Mapping[Tuple[str, int], float],
) -> List[float]:
    """Resolve scores aligned to the coverage candidates."""
    candidate_score_fields = (
        "salient_sentence_scores",
        "candidate_salience_scores",
    )
    for field in candidate_score_fields:
        values = _numeric_scores(row.get(field), candidate_count, field)
        if values is not None:
            return values

    full_score_fields = (
        "salience_scores",
        "salience_probabilities",
        "source_salience_scores",
    )
    for field in full_score_fields:
        raw = row.get(field)
        if not isinstance(raw, list):
            continue
        if len(raw) == candidate_count and len(set(source_indices)) != candidate_count:
            values = _numeric_scores(raw, candidate_count, field)
            if values is not None:
                return values
        if source_indices and max(source_indices) < len(raw):
            try:
                return [float(raw[index]) for index in source_indices]
            except (TypeError, ValueError) as error:
                raise ValueError(f"field {field!r} contains a nonnumeric score") from error

    doc_id = str(row.get("doc_id"))
    missing = [(doc_id, index) for index in source_indices if (doc_id, index) not in external_scores]
    if missing:
        raise ValueError(
            f"{doc_id!r}: no aligned salience scores; missing {len(missing)} candidates"
        )
    return [external_scores[(doc_id, index)] for index in source_indices]


def rank_row(
    row: Mapping[str, Any], candidate_field: Optional[str],
    external_scores: Mapping[Tuple[str, int], float], top_k: Optional[int],
    omitted_labels: Tuple[int, ...],
) -> Dict[str, Any]:
    metadata = row.get("coverage_labeling_metadata")
    metadata_field = metadata.get("candidate_field") if isinstance(metadata, dict) else None
    field = candidate_field or metadata_field or "source_sentences_valid"
    candidates = row.get(field)
    labels = row.get("coverage_labels")
    reasoning = row.get("coverage_reasoning")
    if not isinstance(candidates, list):
        raise ValueError(f"{row.get('doc_id')!r}: missing candidate list {field!r}")
    if not isinstance(labels, list) or len(labels) != len(candidates):
        raise ValueError(f"{row.get('doc_id')!r}: coverage_labels are not aligned with {field!r}")
    if not isinstance(reasoning, list) or len(reasoning) != len(candidates):
        raise ValueError(f"{row.get('doc_id')!r}: coverage_reasoning is not aligned with {field!r}")

    raw_indices = row.get("coverage_sentence_indices")
    if raw_indices is None:
        raw_indices = list(range(len(candidates)))
    if not isinstance(raw_indices, list) or len(raw_indices) != len(candidates):
        raise ValueError(f"{row.get('doc_id')!r}: coverage_sentence_indices are not aligned")
    source_indices = [int(index) for index in raw_indices]
    scores = get_candidate_scores(row, len(candidates), source_indices, external_scores)

    records = []
    for index, (sentence, label, explanation, source_index, score) in enumerate(
        zip(candidates, labels, reasoning, source_indices, scores)
    ):
        if label not in (0, 1, 2):
            raise ValueError(f"{row.get('doc_id')!r}: invalid coverage label {label!r} at {index}")
        records.append({
            "candidate_index": index,
            "source_sentence_index": source_index,
            "sentence": sentence,
            "salience_score": score,
            "coverage_label": label,
            "coverage_binary_label": int(label == 2),
            "omission_label": int(label in omitted_labels),
            "reasoning": explanation,
        })

    ranked = sorted(records, key=lambda record: (-record["salience_score"], record["candidate_index"]))
    omissions = [record for record in ranked if record["omission_label"] == 1]
    if top_k is not None:
        omissions = omissions[:top_k]

    output = dict(row)
    output["coverage_binary_labels"] = [int(label == 2) for label in labels]
    output["omission_ranked_sentences"] = omissions
    output["omission_sentence_indices"] = [record["source_sentence_index"] for record in omissions]
    output["all_ranked_sentences"] = ranked
    output["omission_ranking_metadata"] = {
        "format_version": "bhc_omission_ranking_v1",
        "candidate_field": field,
        "coverage_binary_definition": "coverage_binary_label = int(coverage_label == 2)",
        "omission_definition": f"omission_label = int(coverage_label in {list(omitted_labels)})",
        "ranking_score": "frozen salience_score, descending",
        "top_k": top_k,
        "retrained": False,
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--candidate-field", help="Override the coverage metadata candidate field")
    parser.add_argument("--salience-jsonl", type=Path, help="Optional external predictions with doc_id, sentence_idx, and score")
    parser.add_argument("--salience-score-field", default="score")
    parser.add_argument("--salience-index-field", default="sentence_idx")
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Keep the top-k omissions per document (default: 10).",
    )
    parser.add_argument(
        "--omitted-labels", nargs="+", type=int, default=[0],
        help="Coverage labels treated as omissions (default: 0).",
    )
    args = parser.parse_args()
    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be positive")
    if any(label not in (0, 1, 2) for label in args.omitted_labels):
        parser.error("--omitted-labels values must be 0, 1, or 2")

    external_scores = load_external_scores(
        args.salience_jsonl, args.salience_score_field, args.salience_index_field,
    )
    rows = [rank_row(row, args.candidate_field, external_scores, args.top_k, tuple(args.omitted_labels))
            for row in load_jsonl(args.input_jsonl)]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output_jsonl)
    print(f"Wrote {len(rows)} omission-ranking rows -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
