#!/usr/bin/env python3
"""Remove semantic duplicates from ranked source-sentence candidates.

Aligned fields such as candidate indices, salience scores, and coverage labels
are kept in sync. The highest-priority sentence represents each duplicate set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from omitehr.utils.embeddings import embed_sentences, load_model, public_model_identifier


DEFAULT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"


DEFAULT_ALIGNED_FIELDS = (
    "salient_sentence_indices",
    "salient_sentence_scores",
    "coverage_labels",
    "coverage_reasoning",
    "coverage_sentence_indices",
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Return pairwise cosine similarities for vectors shaped (n, d)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, 1e-12)
    return normalized @ normalized.T


def dedupe_indices(
    sentences: Sequence[str], embeddings: np.ndarray, threshold: float,
    priorities: Optional[Sequence[float]] = None, max_kept: Optional[int] = None,
) -> List[int]:
    """Select one representative from each cosine-similar candidate group."""
    if max_kept is not None and max_kept <= 0:
        raise ValueError("max_kept must be positive")
    normalized = embeddings / np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12,
    )
    if priorities is None:
        order = list(range(len(sentences)))
    else:
        order = sorted(range(len(sentences)), key=lambda i: (-priorities[i], i))

    kept: List[int] = []
    for index in order:
        similarities = normalized[index] @ normalized[kept].T if kept else np.array([])
        if not np.any(similarities >= threshold):
            kept.append(index)
            if max_kept is not None and len(kept) == max_kept:
                break
    return sorted(kept)


def dedupe_row(
    row: Dict[str, Any], candidate_field: str, aligned_fields: Sequence[str],
    threshold: float, tokenizer, model, device: str, batch_size: int,
    max_length: int, priority_field: Optional[str],
) -> Dict[str, Any]:
    candidates = row.get(candidate_field)
    if not isinstance(candidates, list):
        raise ValueError(f"{row.get('doc_id')!r}: missing list field {candidate_field!r}")
    if not all(isinstance(sentence, str) and sentence.strip() for sentence in candidates):
        raise ValueError(f"{row.get('doc_id')!r}: candidates must be nonempty strings")

    for field in aligned_fields:
        values = row.get(field)
        if values is not None and (not isinstance(values, list) or len(values) != len(candidates)):
            raise ValueError(
                f"{row.get('doc_id')!r}: {field!r} is not aligned with {candidate_field!r}"
            )

    if len(candidates) < 2:
        kept = list(range(len(candidates)))
    else:
        embeddings = np.asarray(
            embed_sentences(
                candidates, tokenizer, model, device,
                batch_size=batch_size, max_length=max_length,
            ),
            dtype=np.float32,
        )
        priorities = None
        if priority_field:
            raw_priorities = row.get(priority_field)
            if raw_priorities is not None:
                if not isinstance(raw_priorities, list) or len(raw_priorities) != len(candidates):
                    raise ValueError(
                        f"{row.get('doc_id')!r}: {priority_field!r} is not aligned"
                    )
                try:
                    priorities = [float(value) for value in raw_priorities]
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{row.get('doc_id')!r}: {priority_field!r} must be numeric"
                    ) from error
                if not all(np.isfinite(value) for value in priorities):
                    raise ValueError(
                        f"{row.get('doc_id')!r}: {priority_field!r} must be finite"
                    )
        kept = dedupe_indices(candidates, embeddings, threshold, priorities)

    output = dict(row)
    output[candidate_field] = [candidates[index] for index in kept]
    for field in aligned_fields:
        values = row.get(field)
        if isinstance(values, list):
            output[field] = [values[index] for index in kept]
    output["deduplication_metadata"] = {
        "method": "pairwise_cosine_similarity",
        "embedding_model": public_model_identifier(
            str(getattr(model.config, "_name_or_path", "") or "")
        ),
        "candidate_field": candidate_field,
        "threshold": threshold,
        "priority_field": priority_field,
        "original_count": len(candidates),
        "deduplicated_count": len(kept),
        "kept_candidate_indices": kept,
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--candidate-field", default="salient_sentences")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Cosine similarity threshold for duplicates (default: 0.95).")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--priority-field", default="salient_sentence_scores",
                        help="Aligned numeric field used to choose representatives; use '' to keep first.")
    parser.add_argument(
        "--aligned-fields", default=",".join(DEFAULT_ALIGNED_FIELDS),
        help="Comma-separated candidate-aligned fields to filter together.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.batch_size <= 0 or args.max_length <= 0:
        parser.error("--batch-size and --max-length must be positive")

    aligned_fields = [field.strip() for field in args.aligned_fields.split(",") if field.strip()]
    priority_field = args.priority_field or None
    if priority_field and priority_field not in aligned_fields:
        aligned_fields.append(priority_field)
    rows = load_jsonl(args.input_jsonl)
    tokenizer, model, device = load_model(args.model, args.device)
    output_rows = [
        dedupe_row(
            row, args.candidate_field, aligned_fields, args.threshold,
            tokenizer, model, device, args.batch_size, args.max_length,
            priority_field,
        )
        for row in rows
    ]

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output_jsonl)
    removed = sum(
        row["deduplication_metadata"]["original_count"]
        - row["deduplication_metadata"]["deduplicated_count"]
        for row in output_rows
    )
    print(f"Wrote {len(output_rows)} rows -> {args.output_jsonl}; removed {removed} duplicate candidates")


if __name__ == "__main__":
    main()
