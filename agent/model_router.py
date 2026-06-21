"""Coding-model ranking engine — Pilar B of the Hermes super-coding upgrade.

Given available providers and a model catalog, recommends the best model for
coding subtasks. Pure computation — no network, no side effects. The caller
is responsible for providing the candidate pool (e.g. from models_dev).

  >>> choice = recommend_coding_model(available_providers={"anthropic", "openrouter"}, ...)
  >>> choice.provider
  'anthropic'
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ── Public types ───────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class CodingModelChoice:
    """The router's recommendation: which provider:model to use for coding."""

    provider: str
    model: str
    score: float
    reason: str


# ── Scoring constants ──────────────────────────────────────────────────────────

# Coding-family bonus: models whose family string indicates strong coding
# training get a multiplicative bonus in scoring.
_CODING_FAMILIES = frozenset({
    "claude", "gpt", "codex", "gemini", "deepseek", "qwen", "kimi", "glm", "grok",
})

# Minimum context window (in tokens) for a model to be considered for coding.
_MIN_CONTEXT_WINDOW = 32_000


# ── Core scoring ───────────────────────────────────────────────────────────────


def _score_coding_model(model) -> float:
    """Score a single model's suitability for coding tasks.

    Returns a float 0.0 (unusable) to ~1.0 (excellent). Models without
    tool-calling or marked deprecated get 0.0 — they can't be coding agents.
    """
    # Hard gates: must have tool calling and not be deprecated.
    if not getattr(model, "tool_call", False):
        return 0.0
    status = getattr(model, "status", "") or ""
    if status.lower() == "deprecated":
        return 0.0

    score = 0.2  # baseline: tool_call = usable

    context = getattr(model, "context_window", 0) or 0
    if context < _MIN_CONTEXT_WINDOW:
        score -= 0.2
    elif context > 100_000:
        score += 0.25
    elif context > 64_000:
        score += 0.15

    if getattr(model, "reasoning", False):
        score += 0.3

    family = (getattr(model, "family", "") or "").strip().lower()
    if family in _CODING_FAMILIES:
        score += 0.2

    return max(0.0, min(1.0, score))


def _model_key(model) -> tuple[str, str]:
    """Return ``(provider, model_id)`` tuple for dedup/exclusion."""
    pid = getattr(model, "provider_id", "") or ""
    mid = getattr(model, "id", "") or ""
    return (pid, mid)


def recommend_coding_model(
    *,
    available_providers: Set[str],
    all_candidate_models: Callable[[], Dict[str, List]],
    mode: str = "balanced",
    exclude: Optional[Set[str]] = None,
    only: Optional[Set[str]] = None,
    ignore: Optional[Set[str]] = None,
    min_context: int = _MIN_CONTEXT_WINDOW,
) -> Optional[CodingModelChoice]:
    """Rank all candidate models and return the best coding-model recommendation.

    Args:
        available_providers: Provider IDs (e.g. ``{"anthropic", "openrouter"}``)
            that have working credentials. Only these are considered.
        all_candidate_models: A zero-arg callable returning ``{provider_id: [model_objects]}``.
            Designed so the test can inject stubs and real code can inject
            ``models_dev``-backed data.
        mode: ``"max"`` (pure capability), ``"balanced"`` (capability with
            mild cost penalty), or ``"economy"`` (minimum capability then cheapest).
        exclude: Provider:model keys (``"anthropic:claude-sonnet-4.6"``) to skip,
            e.g. the session's own model to avoid routing to itself.
        only: If set, only consider these provider IDs.
        ignore: If set, skip these provider IDs entirely.
        min_context: Minimum context window (tokens). Default 32K.

    Returns:
        A :class:`CodingModelChoice` or ``None`` if no model qualifies.
    """
    if not available_providers:
        return None

    providers = available_providers
    if only:
        providers &= only
    if ignore:
        providers -= ignore
    if not providers:
        return None

    exclude = exclude or set()

    # Gather candidates from all eligible providers
    candidates: list = []
    try:
        catalog = all_candidate_models()
    except Exception as exc:
        logger.warning("model_router: all_candidate_models() raised %s", exc)
        catalog = {}

    for prov in providers:
        models = catalog.get(prov, [])
        for m in models:
            if (getattr(m, "context_window", 0) or 0) < min_context:
                continue
            pk = _model_key(m)
            key = f"{pk[0]}:{pk[1]}"
            if key in exclude:
                continue
            score = _score_coding_model(m)
            if score <= 0:
                continue
            mid = getattr(m, "id", "") or ""
            cost_input = getattr(m, "cost_input", 0.0) or 0.0
            cost_output = getattr(m, "cost_output", 0.0) or 0.0
            candidates.append((m, pk[0], mid, score, cost_input + cost_output))

    if not candidates:
        return None

    if mode == "max":
        # Sort by score descending, then by context descending (more context = better),
        # then by total cost descending (more expensive = more capable within same tier).
        candidates.sort(key=lambda c: (
            -c[3],
            -getattr(c[0], "context_window", 0),
            -c[4],
        ))
        best = candidates[0]
        return CodingModelChoice(
            provider=best[1],
            model=best[2],
            score=best[3],
            reason=f"ranked #1 in max mode: score={best[3]:.2f}, "
                   f"family={getattr(best[0], 'family', '?')}, "
                   f"tool_call={getattr(best[0], 'tool_call', False)}, "
                   f"reasoning={getattr(best[0], 'reasoning', False)}, "
                   f"context={getattr(best[0], 'context_window', '?')}",
        )

    elif mode == "economy":
        # Minimum threshold: must score at least 0.4 (tool_call + reasoning or >100k ctx)
        viable = [c for c in candidates if c[3] >= 0.4]
        if not viable:
            viable = [c for c in candidates if c[3] >= 0.3]
        if not viable:
            return None
        # Among viable, pick cheapest by total cost (input + output / M)
        viable.sort(key=lambda c: c[4])
        best = viable[0]
        return CodingModelChoice(
            provider=best[1],
            model=best[2],
            score=best[3],
            reason=f"cheapest viable in economy mode: score={best[3]:.2f}, "
                   f"cost=${best[4]:.2f}/M",
        )

    else:
        # "balanced" (default): score - mild cost penalty
        def _balanced_key(c):
            return c[3] - (c[4] / 150.0)  # cost penalty: $150/M total ≈ -1.0 score

        candidates.sort(key=_balanced_key, reverse=True)
        best = candidates[0]
        return CodingModelChoice(
            provider=best[1],
            model=best[2],
            score=best[3],
            reason=f"ranked #1 in balanced mode: score={best[3]:.2f}, "
                   f"cost=${best[4]:.2f}/M, adjusted={_balanced_key(best):.2f}",
        )
