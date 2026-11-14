#!/usr/bin/env python3
"""Assign salience labels to source chunks relative to a clinician BHC.

The output keeps the auditable structured labels and also writes the aligned
``contextual_label_values`` projection consumed by the training code.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from omitehr.utils.entailment import (
    CostTracker, _call_model, build_source_contexts, get_async_client,
    load_jsonl, load_local_env, provider_for, write_jsonl,
    add_llm_arguments, fingerprint,
)


SYSTEM_PROMPT = """You are a senior clinician auditing evidence used to construct a Brief Hospital Course (BHC).

You receive a BHC, one complete source note, and specific chunks taken from that note. Judge each chunk in the context of that note. Do not judge a chunk as an isolated sentence and do not rely on medical knowledge to fill gaps in the BHC.

The task is evidence selection for faithful compression. Imagine selecting the smallest set of source facts from which a clinician could write this BHC without adding unsupported detail. A chunk can be clinically true and important yet be 0 if it does not support a specific BHC claim.

First resolve the chunk's subject and scope from the complete source note. Words such as "complications", "stable", "negative", or "improved" refer to the event, procedure, test, or patient described by that note--not automatically to the hospital course as a whole.

Assign exactly one label:

2 = DIRECTLY INCLUDED. The BHC directly states the same clinical proposition as the chunk, allowing ordinary paraphrase but preserving material details such as subject, event, certainty, timing, severity, and location. Cite the exact BHC text.

1 = INDIRECTLY RELATED. The chunk relates to one identifiable BHC claim, but is not itself directly included. It may provide method/detail/context for that claim, be broader or narrower, conditional, temporally different, or require an inference. A 1 must name a specific BHC claim.

0 = NOT RELATED TO A BHC CLAIM. The chunk does not support a specific BHC claim at the same scope. This includes boilerplate, unrelated findings, and details from a study or procedure when the BHC only makes a general statement about the patient.

For every label return concise reasoning (20 words or fewer). For label 1 or 2 also return an exact BHC quote. Return ONLY a JSON object mapping each candidate number to {"label": 0|1|2, "bhc_evidence": "exact BHC quote or NONE", "relation": "extra_specificity|broader_target|conditionality|temporal_difference|context_only|inference|NONE", "reasoning": "brief note-scoped explanation"}."""

METHODOLOGY_VERSION = "bhc_contextual_gpt_v1"
RELATED_REASONS = {"extra_specificity", "broader_target", "conditionality",
                   "temporal_difference", "context_only", "inference"}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _terms(text: str) -> set[str]:
    stop = {"the", "and", "with", "from", "that", "this", "were", "was",
            "had", "has", "have", "patient", "during", "after", "before"}
    return {word.rstrip("s") for word in re.findall(r"[a-zA-Z]{4,}", text.lower())
            if word not in stop}


def _direct_evidence_supported(source: str, evidence: str) -> bool:
    source_terms = _terms(source)
    evidence_terms = _terms(evidence)
    shared_numbers = set(re.findall(r"\d+(?:\.\d+)?", source)) & set(
        re.findall(r"\d+(?:\.\d+)?", evidence)
    )
    if not source_terms or not source_terms & evidence_terms:
        return False
    return len(source_terms & evidence_terms) / len(source_terms) >= 0.60 or bool(shared_numbers)


def parse_response(raw: Optional[str], candidates: List[str], bhc: str) -> List[Dict[str, Any]]:
    """Parse and conservatively validate one model response."""
    data = {}
    if raw:
        try:
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw[start:end])
        except (AttributeError, json.JSONDecodeError):
            pass
    output = []
    for number, source in enumerate(candidates, 1):
        value = data.get(str(number), {}) if isinstance(data, dict) else {}
        label = value.get("label") if isinstance(value, dict) else None
        evidence = value.get("bhc_evidence", "NONE") if isinstance(value, dict) else "NONE"
        relation = value.get("relation", "NONE") if isinstance(value, dict) else "NONE"
        reasoning = value.get("reasoning", "") if isinstance(value, dict) else ""
        if type(label) is not int or label not in (0, 1, 2):
            label, reasoning = None, "Invalid or incomplete model response."
        if label == 2 and (not isinstance(evidence, str) or _norm(evidence) not in _norm(bhc)
                           or not _direct_evidence_supported(source, evidence)):
            label, relation = 1, "extra_specificity"
        if label == 1 and (not isinstance(evidence, str) or _norm(evidence) not in _norm(bhc)
                           or relation not in RELATED_REASONS):
            label, evidence, relation = None, "NONE", "NONE"
        if label == 0:
            evidence, relation = "NONE", "NONE"
        if label in (1, 2) and not isinstance(reasoning, str):
            label, evidence, relation = None, "NONE", "NONE"
        output.append({"label": label, "bhc_evidence": evidence,
                       "relation": relation, "reasoning": str(reasoning).strip()[:500]})
    return output


async def _label_row(args, client, provider, row, semaphore, tracker):
    chunks = row.get("source_sentences_valid")
    bhc = str(row.get(args.bhc_field) or "").strip()
    if not isinstance(chunks, list) or not bhc:
        raise ValueError(
            f"doc_id={row.get('doc_id')!r} needs source_sentences_valid and "
            f"a nonempty {args.bhc_field}"
        )
    contexts = build_source_contexts(row)
    by_note: Dict[str, List[int]] = {}
    for index in range(len(chunks)):
        by_note.setdefault(contexts[index].get("note_key", "0"), []).append(index)
    result: List[Optional[Dict[str, Any]]] = [None] * len(chunks)

    async def call_batch(indices: List[int]) -> List[Dict[str, Any]]:
        context = contexts[indices[0]]
        candidates = "\n\n".join(
            f"{position + 1}. {chunks[index]}" for position, index in enumerate(indices)
        )
        prompt = (f'BHC:\n"""\n{bhc}\n"""\n\n'
                  f"Note title: {context.get('note_title', 'Unknown note')}\n"
                  f'Note:\n"""\n{context.get("note_text", "")}\n"""\n\n'
                  f"Candidates:\n{candidates}")
        async with semaphore:
            raw, input_tokens, output_tokens = await _call_model(
                client, provider, args.model, prompt, args.max_output_tokens, SYSTEM_PROMPT,
                args.model_options,
            )
        tracker.record(args.model, input_tokens, output_tokens)
        return parse_response(raw, [chunks[index] for index in indices], bhc)

    for note_indices in by_note.values():
        for start in range(0, len(note_indices), args.batch_size):
            indices = note_indices[start:start + args.batch_size]
            values = await call_batch(indices)
            pending = [i for i, value in zip(indices, values) if value["label"] is None]
            values_by_index = dict(zip(indices, values))
            for retry_size in (5, 1):
                retry_pending = []
                for retry_start in range(0, len(pending), retry_size):
                    retry_indices = pending[retry_start:retry_start + retry_size]
                    retry_values = await call_batch(retry_indices)
                    for index, value in zip(retry_indices, retry_values):
                        values_by_index[index] = value
                        if value["label"] is None:
                            retry_pending.append(index)
                pending = retry_pending
            for index in indices:
                result[index] = values_by_index[index]
    if any(value is None or value["label"] is None for value in result):
        raise RuntimeError(f"incomplete labels for doc_id={row.get('doc_id')!r}")
    output = dict(row)
    output["contextual_chunk_labels"] = result
    output["contextual_label_values"] = [value["label"] for value in result]
    output["contextual_labeling_metadata"] = {
        "methodology_version": METHODOLOGY_VERSION, "target_kind": "bhc",
        "model": args.model, "provider": provider, "model_options": args.model_options,
        "input_sha256": labeling_fingerprint(row, args),
        "label_definition": {"0": "not_related_to_a_bhc_claim",
                              "1": "indirectly_related_to_a_bhc_claim",
                              "2": "directly_included_bhc_evidence"},
    }
    return output


def labeling_fingerprint(row, args) -> str:
    return fingerprint(row.get(args.bhc_field), row.get("source_sentences_valid"),
                       build_source_contexts(row), SYSTEM_PROMPT, args.model,
                       provider_for(args.model, args.provider), args.model_options,
                       args.batch_size, args.max_output_tokens)


async def run(args) -> None:
    load_local_env()
    rows = load_jsonl(args.input_jsonl)
    if args.max_documents is not None:
        rows = rows[:args.max_documents]
    doc_ids = [str(row["doc_id"]) for row in rows if row.get("doc_id") is not None]
    if len(doc_ids) != len(rows) or len(set(doc_ids)) != len(rows):
        raise ValueError("input JSONL must contain one unique non-null doc_id per row")
    provider = provider_for(args.model, args.provider)
    client = get_async_client(provider)
    tracker = CostTracker()
    semaphore = asyncio.Semaphore(args.max_concurrency)
    completed: Dict[str, Dict[str, Any]] = {}
    if args.resume and args.output_jsonl.exists():
        existing = {str(row["doc_id"]): row for row in load_jsonl(args.output_jsonl)}
        for row in rows:
            old = existing.get(str(row["doc_id"]), {})
            labels = old.get("contextual_label_values", [])
            if (old.get("labeling_status") == "complete"
                    and isinstance(labels, list)
                    and len(labels) == len(row["source_sentences_valid"])
                    and all(type(label) is int and label in (0, 1, 2) for label in labels)
                    and old.get("contextual_labeling_metadata", {}).get("input_sha256") == labeling_fingerprint(row, args)):
                completed[str(row["doc_id"])] = {**row, **{key: old[key] for key in (
                    "contextual_chunk_labels", "contextual_label_values", "contextual_labeling_metadata", "labeling_status")}}
    pending = [row for row in rows if str(row.get("doc_id")) not in completed]
    for start in range(0, len(pending), args.max_concurrency):
        window = pending[start:start + args.max_concurrency]
        tasks = [asyncio.create_task(_label_row(args, client, provider, row, semaphore, tracker))
                 for row in window]
        for task in asyncio.as_completed(tasks):
            result = await task
            result["labeling_status"] = "complete"
            completed[str(result["doc_id"])] = result
            if args.checkpoint:
                write_jsonl(list(completed.values()), args.output_jsonl)
    write_jsonl(list(completed.values()), args.output_jsonl)
    print(f"Wrote {len(completed)} labeled documents -> {args.output_jsonl}")
    print(f"Total cost: {tracker.summary()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_llm_arguments(parser)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--bhc-field", default="target_text",
        help="Field containing the clinician-written BHC (default: target_text).",
    )
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--max-output-tokens", type=int, default=3000)
    parser.add_argument("--max-concurrency", type=int, default=3)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in ("batch_size", "max_output_tokens", "max_concurrency"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
