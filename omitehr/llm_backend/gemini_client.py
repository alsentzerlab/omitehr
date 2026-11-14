"""Placeholder: Gemini-compatible async client for the omission pipeline.

Fill in ``get_async_gemini_client`` and return an object that supports:

    response = await client.aio.models.generate_content(
        model=model, contents=f"{system_prompt}\\n\\n{user_prompt}",
        config={"max_output_tokens": max_tokens, ...},
    )

The pipeline reads the reply text from ``response.text`` and token usage
from ``response.usage_metadata.prompt_token_count`` /
``response.usage_metadata.candidates_token_count``. The official
``google-genai`` package's ``genai.Client`` satisfies this shape directly
(its ``.aio`` namespace is the async interface); any other client (a
different gateway, a self-hosted server, a mock for testing) only needs to
match the same attributes.

Called whenever ``--model`` resolves to the "gemini" provider (model name
contains "gemini"; see ``omitehr.utils.entailment.provider_for``).
"""
from __future__ import annotations


def get_async_gemini_client():
    # Construct and return your own Gemini-compatible async client here,
    # e.g. `from google import genai; return genai.Client(api_key=...)`.
    raise NotImplementedError(
        "Implement get_async_gemini_client() in "
        "omitehr/llm_backend/gemini_client.py before running "
        "a Gemini-backed pipeline stage."
    )
