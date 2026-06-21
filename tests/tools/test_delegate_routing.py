"""Tests for delegation config injection from model_router."""

from tools.delegate_tool import _inject_routing_model


class TestInjectRoutingModel:
    def test_respects_explicit_model(self):
        """When delegation.model is already set, _inject returns unchanged."""
        cfg = {"model": "gpt-5.4", "provider": "openai"}
        result = _inject_routing_model(cfg)
        assert result["model"] == "gpt-5.4"
        assert result is cfg  # same object, not a copy

    def test_empty_cfg_does_not_crash(self):
        """Empty config returns unchanged without errors."""
        result = _inject_routing_model({})
        assert "model" not in result
