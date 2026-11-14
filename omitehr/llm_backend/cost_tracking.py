"""Optional cost/usage telemetry hooks for the omission pipeline.

Unlike the provider client files in this package, these are working no-ops
rather than placeholders: ``CostTracker`` (in ``omitehr.utils.entailment``)
calls ``log_call`` after every model response and ``price_for`` to estimate
spend, and it already treats a ``None`` price as "skip cost math, still
count tokens." Leaving these as no-ops keeps token/call counts working out
of the box; fill them in only if you want your own usage logging or a
dollar-cost estimate.
"""
from __future__ import annotations

from typing import Optional, Tuple


def log_call(*, model: str, input_tokens: Optional[int], output_tokens: Optional[int]) -> None:
    return None


def price_for(model: str) -> Optional[Tuple[float, float]]:
    """Return (input_price_per_million_tokens, output_price_per_million_tokens), or None."""
    return None
