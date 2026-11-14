"""Placeholder: OpenAI-compatible async client for the omission pipeline.

Fill in ``get_async_openai_client`` and return an object that supports:

    response = await client.chat.completions.create(
        model=model, messages=[{"role": "system", "content": ...},
                                {"role": "user", "content": ...}],
        max_tokens=max_tokens,  # or max_completion_tokens for some models
    )

The pipeline reads the reply from ``response.choices[0].message.content``
and token usage from ``response.usage.prompt_tokens`` /
``response.usage.completion_tokens``. The official ``openai`` package's
``AsyncOpenAI`` client satisfies this shape directly; any other client
(a different gateway, a self-hosted server, a mock for testing) only needs
to match the same three attributes.

Called whenever ``--model`` resolves to the "openai" provider (i.e. any
model name that isn't recognized as Gemini or Claude/Anthropic; see
``omitehr.utils.entailment.provider_for``).
"""
from __future__ import annotations


def get_async_openai_client():
    # Construct and return your own OpenAI-compatible async client here,
    # e.g. `from openai import AsyncOpenAI; return AsyncOpenAI(api_key=...)`.
    raise NotImplementedError(
        "Implement get_async_openai_client() in "
        "omitehr/llm_backend/openai_client.py before running "
        "an OpenAI-backed pipeline stage."
    )
