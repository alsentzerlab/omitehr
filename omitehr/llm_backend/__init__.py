"""Pluggable LLM backend seam.

``omitehr.utils.entailment`` imports the five names below and nothing
else about the LLM backend. Swap any of the underlying files for your own
implementation (a different gateway, a self-hosted server, a mock for
testing) without touching any pipeline or algorithm code.
"""
from __future__ import annotations

from .anthropic_client import get_async_anthropic_client
from .cost_tracking import log_call, price_for
from .gemini_client import get_async_gemini_client
from .openai_client import get_async_openai_client

__all__ = [
    "get_async_anthropic_client",
    "get_async_gemini_client",
    "get_async_openai_client",
    "log_call",
    "price_for",
]
