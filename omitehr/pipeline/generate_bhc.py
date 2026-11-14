#!/usr/bin/env python3
"""Generate Brief Hospital Courses from admission source notes.

Each input admission's complete source-note text is supplied to an LLM. The
generated summary is written to ``llm_target_text``; any existing reference
summary is retained in ``reference_target_text`` for later comparison.
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
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omitehr.utils.entailment import (
    CostTracker,
    _call_model,
    add_llm_arguments,
    fingerprint,
    get_async_client,
    load_jsonl,
    load_local_env,
    provider_for,
)


SYSTEM_PROMPT = "You are a physician who takes care of patients in a hospital."
BASELINE_PROMPT = '''## Task:
Write a "Brief Hospital Course" section for a patient's discharge summary using the provided
information from the patient's electronic health record from this hospital admission. Respond only
with content for the "Brief Hospital course" section without any elaboration or explanation.

## Information from Patient's Electronic Health Record:
{text}

## Brief Hospital Course:'''

PROMPT_OPTIMIZED = '''## Task
Write the Brief Hospital Course section of a discharge summary for this admission. This should be a concise, clinically 
comprehensive narrative that synthesizes the patient's hospitalization, emphasizing the most clinically significant events, 
management decisions, and outcomes.

Include the following components, as clinically relevant and documented:
1. The reason for admission and significant presenting findings;
2. The major diagnoses and active problems addressed during hospitalization;
3. Key diagnostic findings, procedures, treatments, and important changes in management;
4. The clinical course, including treatment response, complications, and unresolved problems; and
5. The patient's condition at discharge, with relevant disposition and follow-up plans when documented.

Evidence and style constraints:
- Use only information supported by the provided medical record. Do not invent or infer undocumented details.
- Preserve clinically important distinctions, including diagnostic uncertainty, timing, severity, and planned versus completed care.
- Synthesize information across notes, accounting for duplicated or conflicting documentation without losing important earlier events.
- Omit routine normal findings, boilerplate, unrelated chronic history, administrative details, and
  details that do not explain the hospitalization or outcome.
- Write a concise, clinically coherent narrative in professional discharge-summary prose.

## Information from Patient's Electronic Health Record:
{text}

## Brief Hospital Course:'''

PROMPT_TEMPLATES = {
    "baseline": BASELINE_PROMPT,
    "few_shot": BASELINE_PROMPT,
    "prompt_optimized": PROMPT_OPTIMIZED,
}


def build_prompt(
    source_text: str, prompt_variant: str = "baseline",
    few_shot_examples: Optional[list[str]] = None,
) -> str:
    try:
        template = PROMPT_TEMPLATES[prompt_variant]
    except KeyError as error:
        raise ValueError(
            f"unknown prompt variant {prompt_variant!r}; "
            f"choose from {', '.join(PROMPT_TEMPLATES)}"
        ) from error
    if prompt_variant == "few_shot":
        examples = few_shot_examples or []
        demonstrations = "\n\n".join(
            f"### Example {i}\nClinician-written Brief Hospital Course:\n{example}"
            for i, example in enumerate(examples, 1)
        )
        if demonstrations:
            return (
                "## Demonstrations\n"
                "The following are examples of clinician-written BHC style. They are examples only; "
                "do not copy their patient-specific facts.\n\n"
                + demonstrations
                + "\n\n## New admission\n"
                + BASELINE_PROMPT.format(text=source_text)
            )
    return template.format(text=source_text)


def clean_summary(raw: Optional[str]) -> str:
    if not raw:
        return ""
    text = raw.strip()
    text = re.sub(r"^```(?:text|markdown)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(
        r"^(?:Here(?:'s| is) (?:the )?(?:BHC|summary):)\s*", "", text, flags=re.I
    )
    return text.strip()


def get_summary_source(row: Dict[str, Any], source_field: str) -> str:
    """Read a string or list-valued source field as prompt text."""
    value = row.get(source_field)
    if isinstance(value, list):
        source = "\n\n".join(str(item).strip() for item in value if str(item).strip())
    else:
        source = str(value or "").strip()
    if not source:
        raise ValueError(
            f"row {row.get('doc_id')!r} has empty summary source field {source_field!r}"
        )
    return source


def load_few_shot_examples(path: Path, count: int) -> list[str]:
    """Load fixed examples, retaining JSONL selection as a backwards-compatible fallback."""
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_examples = payload.get("examples", []) if isinstance(payload, dict) else payload
        if not isinstance(raw_examples, list):
            raise ValueError(f"{path} must contain a list of examples")
        examples = [
            str(item.get("bhc", "")).strip() if isinstance(item, dict) else str(item).strip()
            for item in raw_examples
        ]
        examples = [example for example in examples if example]
        if len(examples) < count:
            raise ValueError(f"{path} contains only {len(examples)} usable examples; need {count}")
        return examples[:count]

    candidates: list[str] = []
    for row in load_jsonl(path):
        target = str(row.get("target_text") or "")
        start, end = row.get("start_pos"), row.get("end_pos")
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(target):
            bhc = target[start:end].strip()
        else:
            bhc = str(row.get("bhc_final") or "").strip()
        if 500 <= len(bhc) <= 5000 and row.get("has_hospital_course", True):
            candidates.append(bhc)
    if not candidates:
        raise ValueError(f"no usable clinician-written BHC examples found in {path}")
    # Deterministic length-stratified selection avoids choosing several near-duplicates.
    candidates.sort(key=lambda text: (len(text), text))
    positions = [round(i * (len(candidates) - 1) / max(count - 1, 1)) for i in range(count)]
    return [candidates[position] for position in positions]


async def call_bhc_model(
    client, provider: str, model: str, prompt: str, max_tokens: int,
    model_options: Optional[Dict[str, Any]] = None,
) -> tuple[Optional[str], int, int]:
    return await _call_model(client, provider, model, prompt, max_tokens, SYSTEM_PROMPT, model_options)


def generation_fingerprint(source, provider, model, prompt_variant, examples, max_tokens, options):
    return fingerprint(SYSTEM_PROMPT, build_prompt(source, prompt_variant, examples),
                       provider, model, max_tokens, options or {})


async def generate_row(
    client, provider: str, row: Dict[str, Any], model: str, max_tokens: int,
    max_retries: int, semaphore: asyncio.Semaphore, tracker: CostTracker,
    source_field: str, prompt_variant: str, few_shot_examples: list[str],
    model_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    started = time.perf_counter()
    source = get_summary_source(row, source_field)

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            async with semaphore:
                raw, input_tokens, output_tokens = await call_bhc_model(
                    client, provider, model,
                    build_prompt(source, prompt_variant, few_shot_examples), max_tokens, model_options
                )
            tracker.record(model, input_tokens, output_tokens)
            summary = clean_summary(raw)
            if not summary:
                raise ValueError("model returned an empty BHC")
            output = dict(row)
            output.setdefault("reference_target_text", row.get("target_text"))
            output.pop("target_text", None)
            output["llm_target_text"] = summary
            output["target_kind"] = "bhc"
            output["llm_bhc_generation_metadata"] = {
                "methodology_version": "llm_bhc_generation_v2",
                "prompt_version": f"omitehr_bhc_prompt_{prompt_variant}",
                "prompt_variant": prompt_variant,
                "few_shot_example_count": len(few_shot_examples),
                "model": model, "provider": provider, "model_options": model_options or {},
                "source_field": source_field,
                "max_output_tokens": max_tokens,
                "output_tokens": output_tokens,
                "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "input_sha256": generation_fingerprint(
                    source, provider, model, prompt_variant, few_shot_examples, max_tokens, model_options),
                "generation_seconds": time.perf_counter() - started,
            }
            return output
        except Exception as error:
            last_error = error
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)
    raise RuntimeError(
        f"BHC generation failed for doc_id={row.get('doc_id')!r}: {last_error}"
    )


def write_results(
    rows: list[Dict[str, Any]], results: Dict[Any, Dict[str, Any]], output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            result = results.get(row.get("doc_id"))
            if result is not None:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    temp_path.replace(output_path)


async def run(args: argparse.Namespace) -> None:
    provider = provider_for(args.model, args.provider)
    client = get_async_client(provider)
    rows = load_jsonl(Path(args.input_jsonl))
    doc_ids = [row.get("doc_id") for row in rows]
    if None in doc_ids or len(set(doc_ids)) != len(doc_ids):
        raise ValueError("input JSONL must contain one unique non-null doc_id per row")
    few_shot_examples = (
        load_few_shot_examples(Path(args.few_shot_jsonl), args.num_few_shot)
        if args.prompt_variant == "few_shot"
        else []
    )
    output_path = Path(args.output_jsonl)
    existing: Dict[Any, Dict[str, Any]] = {}
    if args.resume and output_path.exists():
        existing = {
            row.get("doc_id"): row for row in load_jsonl(output_path)
            if row.get("llm_target_text") and row.get("llm_bhc_generation_metadata")
        }

    def has_current_bhc(row: Dict[str, Any], old: Dict[str, Any]) -> bool:
        metadata = old.get("llm_bhc_generation_metadata")
        if not isinstance(metadata, dict):
            return False
        source = get_summary_source(row, args.source_field)
        return metadata.get("input_sha256") == generation_fingerprint(
            source, provider, args.model, args.prompt_variant, few_shot_examples,
            args.max_tokens, args.model_options,
        )

    results: Dict[Any, Dict[str, Any]] = {}
    pending = []
    for row in rows:
        old = existing.get(row.get("doc_id"))
        if old is None or not has_current_bhc(row, old):
            pending.append(row)
        else:
            resumed = dict(row)
            resumed["reference_target_text"] = row.get("reference_target_text", row.get("target_text"))
            resumed.pop("target_text", None)
            resumed["llm_target_text"] = old["llm_target_text"]
            resumed["target_kind"] = old.get("target_kind", "bhc")
            resumed["llm_bhc_generation_metadata"] = old["llm_bhc_generation_metadata"]
            results[row.get("doc_id")] = resumed

    semaphore = asyncio.Semaphore(args.max_concurrency)
    tracker = CostTracker()
    tasks = [
        asyncio.create_task(
            generate_row(
                client, provider, row, args.model, args.max_tokens,
                args.max_retries, semaphore, tracker, args.source_field,
                args.prompt_variant,
                few_shot_examples, args.model_options,
            )
        )
        for row in pending
    ]
    completed = 0
    for task in asyncio.as_completed(tasks):
        result = await task
        results[result.get("doc_id")] = result
        completed += 1
        if completed % args.checkpoint == 0:
            write_results(rows, results, output_path)
            print(f"{completed}/{len(pending)} completed; {tracker.summary()}", flush=True)

    write_results(rows, results, output_path)
    print(f"Wrote {len(results)} rows -> {output_path}")
    print(f"Cost: {tracker.summary()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_llm_arguments(parser)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument(
        "--prompt-variant", choices=tuple(PROMPT_TEMPLATES), default="baseline",
        help="Prompt style used to generate the BHC (default: baseline).",
    )
    parser.add_argument(
        "--few-shot-jsonl",
        default=None,
        help="Path to fixed clinician-written BHC examples (JSON list, or JSONL "
        "with target_text). Required when --prompt-variant is few_shot.",
    )
    parser.add_argument("--num-few-shot", type=int, default=3)
    parser.add_argument(
        "--source-field", default="source_text",
        help="Input field containing the complete admission source notes.",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-concurrency", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--checkpoint", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for option in ("max_tokens", "max_concurrency", "max_retries", "checkpoint"):
        if getattr(args, option) <= 0:
            parser.error(f"--{option.replace('_', '-')} must be positive")
    if args.prompt_variant == "few_shot" and not args.few_shot_jsonl:
        parser.error("--few-shot-jsonl is required when --prompt-variant is few_shot")
    load_local_env()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
