"""Placeholder: Anthropic-compatible async client for the omission pipeline.

Fill in ``get_async_anthropic_client`` and return an object that supports:

    response = await client.messages.create(
        model=model, max_tokens=max_tokens, system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

The pipeline reads the reply text from the first text block in
``response.content`` and token usage from ``response.usage.input_tokens`` /
``response.usage.output_tokens``. The official ``anthropic`` package's
``AsyncAnthropic`` client satisfies this shape directly; any other client
(a different gateway, a self-hosted server, a mock for testing) only needs
to match the same attributes.

Called whenever ``--model`` resolves to the "anthropic" provider (model
name contains "claude" or "anthropic"; see
``omitehr.utils.entailment.provider_for``).
"""
from __future__ import annotations


def get_async_anthropic_client():
    # Construct and return your own Anthropic-compatible async client here,
    # e.g. `from anthropic import AsyncAnthropic; return AsyncAnthropic(api_key=...)`.
    raise NotImplementedError(
        "Implement get_async_anthropic_client() in "
        "omitehr/llm_backend/anthropic_client.py before running "
        "a Claude/Anthropic-backed pipeline stage."
    )
