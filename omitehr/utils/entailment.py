"""Shared LLM runtime helpers for the current omission pipeline.

Prompt and task logic lives with each pipeline stage.  This module only owns
provider routing, request-shape differences, cost tracking, environment
loading, note-context reconstruction, and JSONL I/O.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from omitehr.llm_backend import (
    get_async_anthropic_client,
    get_async_gemini_client,
    get_async_openai_client,
    log_call,
    price_for,
)


def load_local_env(path: Union[Path, str] = Path(".env")) -> None:
    """Load simple KEY=VALUE entries without overwriting shell variables."""
    path = Path(path)
    if not path.is_file():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


class CostTracker:
    """Track token use and estimated cost for one run."""

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0

    def record(
        self, model: str, input_tokens: Optional[int], output_tokens: Optional[int]
    ) -> None:
        log_call(model=model, input_tokens=input_tokens, output_tokens=output_tokens)
        if input_tokens is None or output_tokens is None:
            return
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        price = price_for(model)
        if price is not None:
            input_price, output_price = price
            self.cost += (
                input_tokens * input_price + output_tokens * output_price
            ) / 1_000_000

    def summary(self) -> str:
        return (
            f"{self.calls} calls, {self.input_tokens:,} in / "
            f"{self.output_tokens:,} out tok, ~${self.cost:.4f}"
        )


class DebugBudget:
    """Print prompts and responses for a limited number of calls."""

    def __init__(self, n: int) -> None:
        self.remaining = max(0, n)

    def maybe_print(self, prompt: str, raw_text: str) -> None:
        if self.remaining <= 0:
            return
        self.remaining -= 1
        print("\n" + "=" * 70)
        print("[DEBUG] PROMPT SENT:")
        print(prompt)
        print("-" * 70)
        print("[DEBUG] RAW MODEL RESPONSE:")
        print(raw_text)
        print("=" * 70 + "\n")


def provider_for(model: str, provider: Optional[str] = None) -> str:
    """Infer the LLM provider (openai/anthropic/gemini) from a model identifier."""
    if provider is not None:
        return provider
    model_lower = model.lower()
    if "gemini" in model_lower:
        return "gemini"
    if "claude" in model_lower or "anthropic" in model_lower:
        return "anthropic"
    return "openai"


def add_llm_arguments(parser: argparse.ArgumentParser) -> None:
    def model_options(value: str) -> Dict[str, Any]:
        try:
            options = json.loads(value)
        except json.JSONDecodeError as error:
            raise argparse.ArgumentTypeError(str(error)) from error
        if not isinstance(options, dict):
            raise argparse.ArgumentTypeError("--model-options must be a JSON object")
        reserved = {"model", "messages", "contents", "system", "stream",
                    "max_tokens", "max_completion_tokens", "max_output_tokens"}
        if reserved.intersection(options):
            raise argparse.ArgumentTypeError(
                "--model-options cannot override model, prompt, streaming, or token-limit arguments")
        return options

    parser.add_argument("--provider", choices=("openai", "anthropic", "gemini"),
                        help="API protocol; inferred from --model when omitted.")
    parser.add_argument("--model-options", type=model_options, default={},
                        help="JSON generation options passed to the provider (Gemini: config fields).")


def fingerprint(*values: Any) -> str:
    """Identify the exact inputs/settings of a resumable stage."""
    payload = json.dumps(values, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_async_client(provider: str):
    if provider == "anthropic":
        return get_async_anthropic_client()
    if provider == "gemini":
        return get_async_gemini_client()
    if provider == "openai":
        return get_async_openai_client()
    raise ValueError(f"Unknown provider {provider!r}")


_NEEDS_MAX_COMPLETION_TOKENS: set[str] = set()


def _gemini_thinking_config(model: str) -> Optional[Dict[str, int]]:
    model_lower = model.lower()
    if "2.5-flash" in model_lower:
        return {"thinking_budget": 0}
    if "2.5-pro" in model_lower:
        return {"thinking_budget": 128}
    return None


def _is_openai_reasoning_model(model: str) -> bool:
    return "gpt-5" in model.lower()


async def _call_model(
    client, provider: str, model: str, user_prompt: str, max_tokens: int,
    system_prompt: Optional[str] = None,
    model_options: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[int], Optional[int]]:
    """Call one provider and return response text plus token usage."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    system_prompt = system_prompt or "You are a helpful clinical language model."
    options = dict(model_options or {})

    if provider == "anthropic":
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            **options,
        )
        if getattr(response, "stop_reason", None) == "max_tokens":
            raise ValueError("Anthropic response exceeded the output-token limit")
        text = next(
            (block.text for block in response.content
             if hasattr(block, "text") and isinstance(block.text, str)),
            None,
        )
        if text is None:
            raise ValueError("Anthropic response contained no text block")
        return text, response.usage.input_tokens, response.usage.output_tokens

    if provider == "gemini":
        thinking_config = options.get("thinking_config", _gemini_thinking_config(model))
        output_tokens = max_tokens + max(0, (thinking_config or {}).get("thinking_budget", 0))
        config: Dict[str, Any] = {"max_output_tokens": output_tokens}
        if thinking_config is not None:
            config["thinking_config"] = thinking_config
        config.update(options)
        response = await client.aio.models.generate_content(
            model=model,
            contents=f"{system_prompt}\n\n{user_prompt}",
            config=config,
        )
        usage = getattr(response, "usage_metadata", None)
        candidates = getattr(response, "candidates", None) or []
        if candidates and "MAX_TOKENS" in str(getattr(candidates[0], "finish_reason", "")).upper():
            raise ValueError("Gemini response exceeded the output-token limit")
        return (
            response.text,
            getattr(usage, "prompt_token_count", None) if usage else None,
            getattr(usage, "candidates_token_count", None) if usage else None,
        )

    if provider != "openai":
        raise ValueError(f"Unknown provider {provider!r}")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    token_kwarg = (
        "max_completion_tokens" if model in _NEEDS_MAX_COMPLETION_TOKENS else "max_tokens"
    )
    extra: Dict[str, Any] = {}
    if _is_openai_reasoning_model(model):
        extra["reasoning_effort"] = "minimal"
    extra.update(options)
    try:
        response = await client.chat.completions.create(
            model=model, messages=messages, **{token_kwarg: max_tokens}, **extra
        )
    except Exception as error:
        if token_kwarg == "max_tokens" and "max_completion_tokens" in str(error):
            _NEEDS_MAX_COMPLETION_TOKENS.add(model)
            response = await client.chat.completions.create(
                model=model, messages=messages,
                max_completion_tokens=max_tokens, **extra,
            )
        else:
            raise
    if getattr(response.choices[0], "finish_reason", None) == "length":
        raise ValueError("OpenAI-compatible response exceeded the output-token limit")
    return (
        response.choices[0].message.content,
        getattr(response.usage, "prompt_tokens", None),
        getattr(response.usage, "completion_tokens", None),
    )


NOTE_HEADER_RE = re.compile(
    r"NOTE_DATETIME:\s*(?P<datetime>.*?)\s*\|\s*"
    r"NOTE_TITLE:\s*(?P<title>[^\n|]+)", re.IGNORECASE
)


def _normalized_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def build_source_contexts(doc: Dict[str, Any]) -> Dict[int, Dict[str, str]]:
    """Recover note title, timestamp, full note, and local neighbors."""
    persisted = doc.get("source_sentence_contexts")
    valid = doc.get("source_sentences_valid", [])
    if isinstance(persisted, list) and len(persisted) == len(valid):
        return {i: dict(context) for i, context in enumerate(persisted)}

    source = doc.get("source_text", "") or ""
    headers = list(NOTE_HEADER_RE.finditer(source))
    notes: List[Dict[str, str]] = []
    if headers:
        for index, header in enumerate(headers):
            end = headers[index + 1].start() if index + 1 < len(headers) else len(source)
            notes.append({
                "note_title": header.group("title").strip(),
                "note_datetime": header.group("datetime").strip(),
                "note_text": source[header.start():end].strip(),
            })
    else:
        notes.append({"note_title": "Unknown note", "note_datetime": "", "note_text": source})

    normalized_notes = [_normalized_for_match(note["note_text"]) for note in notes]
    note_indices = []
    provenance = doc.get("atomic_chunk_metadata")
    aligned = isinstance(provenance, list) and len(provenance) == len(valid)
    for index, sentence in enumerate(valid):
        item = provenance[index] if aligned else {}
        note_index = item.get("note_index")
        if isinstance(note_index, int) and not isinstance(note_index, bool) and 0 <= note_index < len(notes):
            note_indices.append(note_index)
            continue
        normalized_sentence = _normalized_for_match(sentence)
        matches = [i for i, note in enumerate(normalized_notes)
                   if normalized_sentence and normalized_sentence in note]
        if aligned:
            matches = [i for i in matches
                       if _normalized_for_match(notes[i]["note_title"]) == _normalized_for_match(item.get("note_title", ""))
                       and notes[i]["note_datetime"] == item.get("note_datetime", "")]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous or missing source note for sentence {index}; rerun preprocessing")
        note_indices.append(matches[0])

    contexts: Dict[int, Dict[str, str]] = {}
    for index, _sentence in enumerate(valid):
        note_index = note_indices[index]
        context = dict(notes[note_index])
        context["previous_sentence"] = next(
            (valid[j] for j in range(index - 1, -1, -1)
             if note_indices[j] == note_index), ""
        )
        context["next_sentence"] = next(
            (valid[j] for j in range(index + 1, len(valid))
             if note_indices[j] == note_index), ""
        )
        context["note_key"] = str(note_index)
        contexts[index] = context
    return contexts


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)
