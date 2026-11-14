#!/usr/bin/env python3
"""Generate note-grounded explanations for ranked BHC omissions.

For each candidate, the model receives the source sentence, its originating
note, the admission record, and the generated BHC. The explanation provides
review context; it does not decide whether the candidate belongs in the BHC.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omitehr.utils.entailment import (
    CostTracker,
    _call_model,
    add_llm_arguments,
    fingerprint,
    build_source_contexts,
    get_async_client,
    load_jsonl,
    load_local_env,
    provider_for,
)


SYSTEM_PROMPT = """You are an internal-medicine attending physician preparing
contextual aids for a clinician annotation study of Brief Hospital Course
(BHC) summaries.

You receive one candidate source sentence, its complete originating note, the
complete source record for the patient's admission, and the LLM-generated BHC
for that same admission. Use the originating note for exact local evidence and
the complete admission for chronology, related events, causes, treatment,
complications, and relevant history. Make the candidate easy to interpret
without asking the reviewer to infer context from the isolated sentence.

Do not label the sentence as an omission or state whether it should have been
included in the BHC. However, you should compare the source with the BHC and explicitly
explain the note-grounded hospital-course information the BHC does not state.
Frame that comparison as context for the reviewer -- not as a verdict or a
recommendation. Explain what the sentence means in practice and what it adds
to understanding the full admission. Explicitly address: (1) time context -- when
it occurred relative to admission, major events, treatment, and discharge; (2)
why it happened -- the documented indication, trigger, mechanism, or clinical
problem, or that the cause is not documented; and (3) history context -- prior
conditions, procedures, recurrence, or baseline status that changes its meaning.
Do not invent facts, diagnoses, timing, causes, outcomes, or clinical importance.
Distinguish a current-admission event from past history, a plan from an event,
and a note-specific finding from a patient-wide conclusion.

Return ONLY a JSON object with exactly these fields:
{
  "meaning": "One concise sentence explaining what the source sentence refers to in this note.",
  "time_context": "When it occurred in this admission and its relation to the hospital timeline.",
  "why_it_happened": "The documented reason, trigger, mechanism, or clinical problem; state when not documented.",
  "history_context": "Relevant prior history, baseline, recurrence, or earlier events that change its interpretation.",
  "hospital_course_relevance": "What it adds or does not add to understanding the full admission compared with the BHC."
}
Each value must be plain language, at most 50 words, and must not mention this
prompt, annotation, omission detection, or an LLM."""

USER_PROMPT_TEMPLATE = """## Candidate source sentence
\"\"\"
{sentence}
\"\"\"

## Complete originating source note
Note title: {note_title}
Note datetime: {note_datetime}
\"\"\"
{note_text}
\"\"\"

## Admission record (all available source notes and events, unless explicitly marked as truncated)
\"\"\"
{admission_text}
\"\"\"

## Current generated Brief Hospital Course
\"\"\"
{bhc}
\"\"\"
"""

BATCH_SYSTEM_PROMPT = """You are an internal-medicine attending physician preparing
contextual aids for a clinician annotation study of Brief Hospital Course
(BHC) summaries.

You receive multiple candidate source sentences from one admission, together
with each candidate's complete originating note, the complete source record
for that admission, and the LLM-generated BHC. For EVERY candidate, use its
originating note for exact local evidence and the complete admission for
chronology, related events, causes, treatment, complications, and relevant
history. Make each candidate easy to interpret without asking the reviewer to
infer context from the isolated sentence.

Do not label any sentence as an omission or state whether it should have been
included in the BHC. However, compare the source with the BHC and explicitly
explain the note-grounded hospital-course information the BHC does not state.
Frame that comparison as context for the reviewer -- not as a verdict or a
recommendation. For every candidate explicitly address: (1) time context;
(2) why it happened; and (3) history context. Do not invent facts, diagnoses,
timing, causes, outcomes, or clinical importance.

Return ONLY one JSON object in this exact shape:
{
  "rationales": [
    {
      "candidate_index": 0,
      "meaning": "One concise sentence explaining what the source sentence refers to in this note.",
      "time_context": "When it occurred in this admission and its relation to the hospital timeline.",
      "why_it_happened": "The documented reason, trigger, mechanism, or clinical problem; state when not documented.",
      "history_context": "Relevant prior history, baseline, recurrence, or earlier events that change its interpretation.",
      "hospital_course_relevance": "What it adds or does not add to understanding the full admission compared with the BHC."
    }
  ]
}
Include exactly one rationale for every provided candidate, using its supplied
candidate_index exactly once. Each text value must be plain language, at most
50 words, and must not mention this prompt, annotation, omission detection, or
an LLM."""

BATCH_USER_PROMPT_TEMPLATE = '''## Candidate source sentences
{candidate_blocks}

## Unique originating source notes
{note_blocks}

## Complete admission record (all available source notes and events)
"""
{admission_text}
"""

## Current generated Brief Hospital Course
"""
{bhc}
"""
'''

METHODOLOGY_VERSION = "omission_rationalizer_v2_full_admission_context"
DEFAULT_REQUEST_TIMEOUT = 180
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_prompt(sentence: str, context: Mapping[str, str], admission_text: str, bhc: str) -> str:
    """Build a clearly delimited, note-aware rationale request."""
    return USER_PROMPT_TEMPLATE.format(
        sentence=sentence,
        note_title=context.get("note_title") or "Unknown",
        note_datetime=context.get("note_datetime") or "Unknown",
        note_text=context.get("note_text") or "[Source note unavailable]",
        admission_text=admission_text or "[Complete admission record unavailable]",
        bhc=bhc,
    )


def build_batched_prompt(
    candidates: Sequence[Mapping[str, Any]], admission_text: str, bhc: str,
) -> str:
    """Build one request while sending each originating note only once."""
    candidate_blocks = []
    note_blocks = []
    note_ids: Dict[Tuple[str, str, str], str] = {}
    for candidate in candidates:
        context = candidate["context"]
        note_title = context.get("note_title") or "Unknown"
        note_datetime = context.get("note_datetime") or "Unknown"
        note_text = context.get("note_text") or "[Source note unavailable]"
        note_key = (note_title, note_datetime, note_text)
        note_id = note_ids.get(note_key)
        if note_id is None:
            note_id = f"note_{len(note_ids)}"
            note_ids[note_key] = note_id
            note_blocks.append(
                "### {note_id}\n"
                "Note title: {note_title}\n"
                "Note datetime: {note_datetime}\n"
                "\"\"\"\n{note_text}\n\"\"\"".format(
                    note_id=note_id,
                    note_title=note_title,
                    note_datetime=note_datetime,
                    note_text=note_text,
                )
            )
        candidate_blocks.append(
            "### Candidate {index}\n"
            "Candidate source sentence:\n\"\"\"\n{sentence}\n\"\"\"\n\n"
            "Originating source note: {note_id}".format(
                index=candidate["candidate_index"],
                sentence=candidate["sentence"],
                note_id=note_id,
            )
        )
    return BATCH_USER_PROMPT_TEMPLATE.format(
        candidate_blocks="\n\n".join(candidate_blocks),
        note_blocks="\n\n".join(note_blocks),
        admission_text=admission_text or "[Complete admission record unavailable]",
        bhc=bhc,
    )


def _clean_text(value: Any, max_words: int = 50) -> str:
    """Normalize a model field and enforce the study-facing length limit."""
    if not isinstance(value, str):
        return ""
    words = re.sub(r"\s+", " ", value).strip().split()
    return " ".join(words[:max_words])


def parse_rationale(raw: Optional[str]) -> Optional[Dict[str, str]]:
    """Validate the small JSON contract without silently substituting facts."""
    if not raw:
        return None
    match = _JSON_OBJECT_RE.search(raw)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    fields = ("meaning", "time_context", "why_it_happened", "history_context", "hospital_course_relevance")
    cleaned = {field: _clean_text(value.get(field)) for field in fields}
    if any(not cleaned[field] for field in fields):
        return None
    return cleaned


def parse_batched_rationales(
    raw: Optional[str], expected_candidate_indices: Sequence[int],
) -> Optional[Dict[int, Dict[str, str]]]:
    """Validate a batched response and require one complete result per candidate."""
    if not raw:
        return None
    match = _JSON_OBJECT_RE.search(raw)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    rationales = value.get("rationales") if isinstance(value, dict) else None
    if not isinstance(rationales, list) or len(rationales) != len(expected_candidate_indices):
        return None
    expected = set(expected_candidate_indices)
    parsed: Dict[int, Dict[str, str]] = {}
    for rationale in rationales:
        if not isinstance(rationale, dict):
            return None
        index = rationale.get("candidate_index")
        if not isinstance(index, int) or index not in expected or index in parsed:
            return None
        cleaned = parse_rationale(json.dumps(rationale))
        if cleaned is None:
            return None
        parsed[index] = cleaned
    return parsed if set(parsed) == expected else None


def cap_note_around_sentence(note: str, sentence: str, max_chars: Optional[int]) -> Tuple[str, bool]:
    """Limit exceptional notes while retaining the candidate's local context.

    Most notes are passed in full.  For a note beyond ``max_chars``, retain a
    centered window around the candidate and label the gap for the model,
    avoiding a misleading claim that it saw a complete note.
    """
    if max_chars is None or len(note) <= max_chars:
        return note, False
    if max_chars < 256:
        raise ValueError("--max-note-chars must be at least 256 when set")
    location = note.lower().find(sentence.lower()) if sentence else -1
    if location < 0:
        location = len(note) // 2
    start = max(0, location - max_chars // 2)
    end = min(len(note), start + max_chars)
    start = max(0, end - max_chars)
    prefix = "[Earlier portion of this source note omitted for model context.]\n"
    suffix = "\n[Later portion of this source note omitted for model context.]"
    return prefix + note[start:end] + suffix, True


def cap_admission_from_end(admission: str, max_chars: Optional[int]) -> Tuple[str, bool]:
    """Cap an exceptional admission by retaining its beginning and marking the omitted tail.

    The complete originating note is supplied separately for every candidate,
    so this guard only limits redundant admission-wide context when the source
    record exceeds the model context window.
    """
    if max_chars is None or len(admission) <= max_chars:
        return admission, False
    if max_chars < 1_024:
        raise ValueError("--max-admission-chars must be at least 1024 when set")
    marker = "\n\n[Later portion of this admission record omitted for model context.]"
    return admission[:max_chars - len(marker)] + marker, True


def _candidate_sentence(candidate: Mapping[str, Any], sentence_field: str) -> str:
    sentence = candidate.get(sentence_field)
    if not isinstance(sentence, str) or not sentence.strip():
        raise ValueError(f"candidate has no nonempty {sentence_field!r}")
    return sentence.strip()


def prepare_candidates(
    row: Mapping[str, Any], candidate_field: str, sentence_field: str,
    source_index_field: str, max_candidates: Optional[int],
) -> List[Dict[str, Any]]:
    """Read ranked candidate records and retain their original ordering."""
    candidates = row.get(candidate_field)
    if not isinstance(candidates, list):
        raise ValueError(f"doc_id={row.get('doc_id')!r} has no list {candidate_field!r}")
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    prepared = []
    for candidate_index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"doc_id={row.get('doc_id')!r}: candidate {candidate_index} is not an object")
        sentence = _candidate_sentence(candidate, sentence_field)
        try:
            source_index = int(candidate[source_index_field])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"doc_id={row.get('doc_id')!r}: candidate {candidate_index} "
                f"needs integer {source_index_field!r}"
            ) from error
        prepared.append({
            "candidate_index": candidate_index,
            "source_sentence_index": source_index,
            "sentence": sentence,
            "candidate": candidate,
        })
    return prepared


def rationale_input_sha256(admission_text: str, bhc: str, candidates: Sequence[Mapping[str, Any]]) -> str:
    """Fingerprint the exact admission, BHC, and candidates sent to the model."""
    payload = {
        "admission_text": admission_text,
        "bhc": bhc,
        "candidates": [
            {
                "source_sentence_index": candidate["source_sentence_index"],
                "sentence": candidate["sentence"],
            }
            for candidate in candidates
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


async def rationalize_candidate(
    client: Any, provider: str, model: str, sentence: str,
    context: Mapping[str, str], admission_text: str, bhc: str, semaphore: asyncio.Semaphore,
    tracker: CostTracker, max_output_tokens: int, max_retries: int,
    request_timeout: int, model_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Call the model, retrying only transient/malformed responses."""
    prompt = build_prompt(sentence, context, admission_text, bhc)
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            async with semaphore:
                raw, input_tokens, output_tokens = await asyncio.wait_for(
                    _call_model(client, provider, model, prompt, max_output_tokens, SYSTEM_PROMPT, model_options),
                    timeout=request_timeout,
                )
            tracker.record(model, input_tokens, output_tokens)
            parsed = parse_rationale(raw)
            if parsed is None:
                raise ValueError("model returned malformed or incomplete rationale JSON")
            return parsed
        except Exception as error:
            last_error = error
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)
    raise RuntimeError(f"rationale generation failed: {last_error}")


async def rationalize_candidates_batched(
    client: Any, provider: str, model: str, candidates: Sequence[Mapping[str, Any]],
    admission_text: str, bhc: str, semaphore: asyncio.Semaphore, tracker: CostTracker,
    max_output_tokens: int, max_retries: int, request_timeout: int,
    model_options: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[int, Dict[str, str]], Dict[str, Any]]:
    """Generate every rationale for one admission in a single model request."""
    prompt = build_batched_prompt(candidates, admission_text, bhc)
    expected_indices = [candidate["candidate_index"] for candidate in candidates]
    last_error: Optional[Exception] = None
    started: Optional[float] = None
    for attempt in range(max_retries + 1):
        try:
            async with semaphore:
                if started is None:
                    # Exclude global request-queue time so this matches the
                    # per-admission stage timing used in the latency benchmark.
                    started = time.perf_counter()
                raw, input_tokens, output_tokens = await asyncio.wait_for(
                    _call_model(client, provider, model, prompt, max_output_tokens, BATCH_SYSTEM_PROMPT, model_options),
                    timeout=request_timeout,
                )
            tracker.record(model, input_tokens, output_tokens)
            parsed = parse_batched_rationales(raw, expected_indices)
            if parsed is None:
                raise ValueError("model returned malformed or incomplete batched rationale JSON")
            return parsed, {
                "attempt_count": attempt + 1,
                "successful_input_tokens": input_tokens or 0,
                "successful_output_tokens": output_tokens or 0,
                "wall_clock_seconds": time.perf_counter() - started,
            }
        except Exception as error:
            last_error = error
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)
    raise RuntimeError(f"batched rationale generation failed: {last_error}")


async def rationalize_row(
    row: Dict[str, Any], args: argparse.Namespace, client: Any, provider: str,
    semaphore: asyncio.Semaphore, tracker: CostTracker,
) -> Dict[str, Any]:
    batch_candidates_per_document = getattr(args, "batch_candidates_per_document", False)
    started = time.perf_counter()
    bhc = row.get(args.bhc_field)
    if not isinstance(bhc, str) or not bhc.strip():
        raise ValueError(f"doc_id={row.get('doc_id')!r} has no nonempty {args.bhc_field!r}")
    admission_text = row.get("source_text")
    if not isinstance(admission_text, str) or not admission_text.strip():
        raise ValueError(f"doc_id={row.get('doc_id')!r} has no nonempty source_text for full-admission context")
    contexts = build_source_contexts(row)
    admission_for_model, admission_was_truncated = cap_admission_from_end(
        admission_text.strip(), args.max_admission_chars,
    )
    candidates = prepare_candidates(
        row, args.candidate_field, args.sentence_field, args.source_index_field,
        args.max_candidates_per_document,
    )
    input_sha256 = rationale_input_sha256(admission_text.strip(), bhc.strip(), candidates)

    prepared_contexts = []
    for candidate in candidates:
        source_index = candidate["source_sentence_index"]
        if source_index not in contexts:
            raise ValueError(
                f"doc_id={row.get('doc_id')!r}: source sentence index {source_index} "
                "does not align with source_sentences_valid"
            )
        context = dict(contexts[source_index])
        note_text, note_was_truncated = cap_note_around_sentence(
            context.get("note_text", ""), candidate["sentence"], args.max_note_chars,
        )
        context["note_text"] = note_text
        prepared_contexts.append({
            **candidate,
            "context": context,
            "note_was_truncated_for_model": note_was_truncated,
        })

    async def one(candidate: Mapping[str, Any]) -> Dict[str, Any]:
        context = candidate["context"]
        rationale = await rationalize_candidate(
            client, provider, args.model, candidate["sentence"], context, admission_for_model, bhc.strip(),
            semaphore, tracker, args.max_output_tokens, args.max_retries,
            args.request_timeout, args.model_options,
        )
        return {
            "candidate_index": candidate["candidate_index"],
            "source_sentence_index": candidate["source_sentence_index"],
            "sentence": candidate["sentence"],
            "note_title": context.get("note_title", ""),
            "note_datetime": context.get("note_datetime", ""),
            "note_was_truncated_for_model": candidate["note_was_truncated_for_model"],
            "admission_was_truncated_for_model": admission_was_truncated,
            **rationale,
        }

    if batch_candidates_per_document:
        batch_rationales, batch_usage = await rationalize_candidates_batched(
            client, provider, args.model, prepared_contexts, admission_for_model, bhc.strip(), semaphore,
            tracker, args.max_output_tokens, args.max_retries, args.request_timeout, args.model_options,
        )
        rationales = [
            {
                "candidate_index": candidate["candidate_index"],
                "source_sentence_index": candidate["source_sentence_index"],
                "sentence": candidate["sentence"],
                "note_title": candidate["context"].get("note_title", ""),
                "note_datetime": candidate["context"].get("note_datetime", ""),
                "note_was_truncated_for_model": candidate["note_was_truncated_for_model"],
                "admission_was_truncated_for_model": admission_was_truncated,
                **batch_rationales[candidate["candidate_index"]],
            }
            for candidate in prepared_contexts
        ]
    else:
        rationales = await asyncio.gather(*(one(candidate) for candidate in prepared_contexts))
    output = dict(row)
    output["omission_rationales"] = rationales
    output["omission_rationalization_metadata"] = {
        "methodology_version": METHODOLOGY_VERSION,
        "model": args.model, "provider": provider, "model_options": args.model_options,
        "candidate_field": args.candidate_field,
        "sentence_field": args.sentence_field,
        "source_index_field": args.source_index_field,
        "bhc_field": args.bhc_field,
        "full_admission_context": True,
        "max_candidates_per_document": args.max_candidates_per_document,
        "max_note_chars": args.max_note_chars,
        "batch_candidates_per_document": batch_candidates_per_document,
        "max_output_tokens": args.max_output_tokens,
        "max_admission_chars": args.max_admission_chars,
        "admission_was_truncated_for_model": admission_was_truncated,
        "rationale_count": len(rationales),
        "input_sha256": input_sha256,
        "config_sha256": rationale_config_fingerprint(row, args),
        "benchmark_rationale_seconds": time.perf_counter() - started,
    }
    if batch_candidates_per_document:
        output["omission_rationalization_metadata"]["batched_request_metrics"] = batch_usage
    return output


def write_results(rows: Sequence[Mapping[str, Any]], results: Mapping[Any, Dict[str, Any]], path: Path) -> None:
    """Atomically checkpoint results in input order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            result = results.get(row.get("doc_id"))
            if result is not None:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    temporary.replace(path)


def rationale_config_fingerprint(row: Dict[str, Any], args: argparse.Namespace) -> str:
    return fingerprint(
        SYSTEM_PROMPT, BATCH_SYSTEM_PROMPT, build_source_contexts(row),
        args.model, provider_for(args.model, args.provider), args.model_options,
        args.max_note_chars, args.max_admission_chars, args.max_output_tokens,
        args.batch_candidates_per_document, args.max_candidates_per_document,
    )


async def run(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.input_jsonl)
    if args.max_documents is not None:
        rows = rows[:args.max_documents]
    doc_ids = [row.get("doc_id") for row in rows]
    if any(doc_id is None for doc_id in doc_ids) or len(set(doc_ids)) != len(doc_ids):
        raise ValueError("input JSONL must contain one unique non-null doc_id per row")

    existing_rows: Dict[Any, Dict[str, Any]] = {}
    if args.resume and args.output_jsonl.exists():
        existing_rows = {
            row.get("doc_id"): row for row in load_jsonl(args.output_jsonl)
            if isinstance(row.get("omission_rationalization_metadata"), dict)
            and isinstance(row.get("omission_rationales"), list)
            and row["omission_rationalization_metadata"].get("model") == args.model
            and row["omission_rationalization_metadata"].get(
                "max_candidates_per_document"
            ) == args.max_candidates_per_document
            and bool(row["omission_rationalization_metadata"].get(
                "batch_candidates_per_document", False
            )) == args.batch_candidates_per_document
        }
    existing: Dict[Any, Dict[str, Any]] = {}
    for row in rows:
        old = existing_rows.get(row.get("doc_id"))
        if old is None:
            continue
        bhc = row.get(args.bhc_field)
        if not isinstance(bhc, str) or not bhc.strip():
            continue
        candidates = prepare_candidates(
            row, args.candidate_field, args.sentence_field,
            args.source_index_field, args.max_candidates_per_document,
        )
        admission_text = row.get("source_text")
        if not isinstance(admission_text, str) or not admission_text.strip():
            continue
        if old["omission_rationalization_metadata"].get("config_sha256") != rationale_config_fingerprint(row, args):
            continue
        if old["omission_rationalization_metadata"].get("input_sha256") == rationale_input_sha256(
            admission_text.strip(), bhc.strip(), candidates
        ):
            existing[row.get("doc_id")] = old

    results: Dict[Any, Dict[str, Any]] = {}
    for row in rows:
        old = existing.get(row.get("doc_id"))
        if old is None:
            continue
        resumed = dict(row)
        resumed["omission_rationales"] = old["omission_rationales"]
        resumed["omission_rationalization_metadata"] = old[
            "omission_rationalization_metadata"
        ]
        results[row.get("doc_id")] = resumed
    pending = [row for row in rows if row.get("doc_id") not in existing]
    provider = provider_for(args.model, args.provider)
    client = get_async_client(provider)
    semaphore = asyncio.Semaphore(args.max_concurrency)
    tracker = CostTracker()
    tasks = [asyncio.create_task(rationalize_row(row, args, client, provider, semaphore, tracker))
             for row in pending]

    completed = 0
    for task in asyncio.as_completed(tasks):
        result = await task
        results[result["doc_id"]] = result
        completed += 1
        if completed % args.checkpoint_every == 0:
            write_results(rows, results, args.output_jsonl)
            print(f"{completed}/{len(pending)} documents completed; {tracker.summary()}", flush=True)

    write_results(rows, results, args.output_jsonl)
    print(f"Wrote {len(results)} rows -> {args.output_jsonl}")
    print(f"Cost: {tracker.summary()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_llm_arguments(parser)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--candidate-field", default="omission_ranked_sentences")
    parser.add_argument("--sentence-field", default="sentence")
    parser.add_argument("--source-index-field", default="source_sentence_index")
    parser.add_argument(
        "--bhc-field", default="llm_target_text",
        help="Field containing the generated BHC; use target_text for a reference BHC.",
    )
    parser.add_argument("--max-candidates-per-document", type=int)
    parser.add_argument(
        "--max-note-chars", type=int, default=100_000,
        help="Cap exceptional source notes around the candidate; set no cap with --max-note-chars 0.",
    )
    parser.add_argument(
        "--batch-candidates-per-document", action="store_true",
        help=("Generate all selected candidates for an admission in one request, "
              "sharing the admission record and BHC across them."),
    )
    parser.add_argument(
        "--max-output-tokens", type=int,
        help="Maximum completion tokens per request (default: 1000, or 6000 with batching).",
    )
    parser.add_argument(
        "--max-admission-chars", type=int, default=0,
        help="Cap exceptional full-admission context from the end; set 0 for no cap.",
    )
    parser.add_argument("--max-concurrency", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_output_tokens is None:
        args.max_output_tokens = 6000 if args.batch_candidates_per_document else 1000
    if args.max_note_chars == 0:
        args.max_note_chars = None
    if args.max_admission_chars == 0:
        args.max_admission_chars = None
    for option in ("max_candidates_per_document", "max_output_tokens", "max_concurrency",
                   "max_retries", "request_timeout", "checkpoint_every", "max_documents"):
        value = getattr(args, option)
        if value is not None and value <= 0:
            parser.error(f"--{option.replace('_', '-')} must be positive")
    if args.max_note_chars is not None and args.max_note_chars < 256:
        parser.error("--max-note-chars must be 0 (unlimited) or at least 256")
    if args.max_admission_chars is not None and args.max_admission_chars < 1024:
        parser.error("--max-admission-chars must be 0 (unlimited) or at least 1024")

    load_local_env()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
