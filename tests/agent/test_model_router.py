"""Tests for agent.model_router — coding model ranking engine."""

from dataclasses import dataclass
from typing import List, Dict

import pytest

from agent.model_router import (
    CodingModelChoice,
    _score_coding_model,
    recommend_coding_model,
)


@dataclass
class StubModelInfo:
    """Minimal stub matching the ModelInfo fields model_router actually reads."""
    id: str
    provider_id: str
    family: str
    reasoning: bool = False
    tool_call: bool = False
    context_window: int = 0
    cost_input: float = 0.0
    cost_output: float = 0.0
    status: str = ""
    name: str = ""


def _stub_models() -> Dict[str, List[StubModelInfo]]:
    """Stub models for testing — no network or catalog involved."""
    return {
        "anthropic": [
            StubModelInfo(id="claude-opus-4.8", provider_id="anthropic", family="claude",
                          reasoning=True, tool_call=True, context_window=200000,
                          cost_input=15.0, cost_output=75.0),
            StubModelInfo(id="claude-sonnet-4.6", provider_id="anthropic", family="claude",
                          reasoning=True, tool_call=True, context_window=200000,
                          cost_input=3.0, cost_output=15.0),
            StubModelInfo(id="claude-haiku-4.5", provider_id="anthropic", family="claude",
                          reasoning=False, tool_call=True, context_window=200000,
                          cost_input=0.25, cost_output=1.25),
        ],
        "openrouter": [
            StubModelInfo(id="google/gemini-3-pro", provider_id="openrouter", family="gemini",
                          reasoning=True, tool_call=True, context_window=200000,
                          cost_input=0.5, cost_output=2.0),
            StubModelInfo(id="deepseek/deepseek-v4", provider_id="openrouter", family="deepseek",
                          reasoning=True, tool_call=True, context_window=128000,
                          cost_input=0.5, cost_output=2.0),
            StubModelInfo(id="qwen/qwq-32b", provider_id="openrouter", family="qwen",
                          reasoning=False, tool_call=True, context_window=32000,
                          cost_input=0.2, cost_output=0.6),
        ],
    }


class TestScoreCodingModel:
    def test_reasoning_tools_high_context_scores_high(self):
        m = StubModelInfo(id="claude-opus-4.8", provider_id="anthropic", family="claude",
                          reasoning=True, tool_call=True, context_window=200000)
        score = _score_coding_model(m)
        assert score > 0

    def test_no_tool_call_scores_zero(self):
        m = StubModelInfo(id="bad-model", provider_id="anthropic", family="claude",
                          reasoning=True, tool_call=False, context_window=200000)
        score = _score_coding_model(m)
        assert score == 0.0

    def test_deprecated_model_scores_zero(self):
        m = StubModelInfo(id="old-model", provider_id="anthropic", family="claude",
                          reasoning=True, tool_call=True, context_window=200000,
                          status="deprecated")
        score = _score_coding_model(m)
        assert score == 0.0


class TestRecommendCodingModel:
    def test_picks_best_coding_model_from_providers(self):
        result = recommend_coding_model(
            available_providers={"anthropic", "openrouter"},
            all_candidate_models=_stub_models,
            mode="max",
        )
        assert result is not None
        # Opus should rank highest among these
        assert result.provider == "anthropic"
        assert "opus" in result.model

    def test_economy_picks_cheaper_model(self):
        result = recommend_coding_model(
            available_providers={"anthropic", "openrouter"},
            all_candidate_models=_stub_models,
            mode="economy",
        )
        assert result is not None
        # Economy should pick cheapest with minimum capability threshold
        assert result.score > 0

    def test_returns_none_when_no_models(self):
        result = recommend_coding_model(
            available_providers=set(),
            all_candidate_models=lambda: {},
        )
        assert result is None

    def test_exclude_prevents_self_routing(self):
        """Exclude the chat session model so routing avoids sending work to itself."""
        result = recommend_coding_model(
            available_providers={"anthropic"},
            all_candidate_models=_stub_models,
            exclude={"anthropic:claude-opus-4.8"},
            mode="max",
        )
        assert result is not None
        # Should pick sonnet since opus is excluded
        assert "sonnet" in result.model

    def test_only_filters_allowed_providers(self):
        result = recommend_coding_model(
            available_providers={"anthropic", "openrouter"},
            all_candidate_models=_stub_models,
            only={"openrouter"},
            mode="max",
        )
        assert result is not None
        assert result.provider == "openrouter"

    def test_ignore_skips_providers(self):
        result = recommend_coding_model(
            available_providers={"anthropic", "openrouter"},
            all_candidate_models=_stub_models,
            ignore={"anthropic"},
            mode="max",
        )
        assert result is not None
        assert result.provider != "anthropic"
