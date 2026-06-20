from unittest.mock import patch
from agent import auxiliary_client


def test_reasoning_config_from_task():
    cfg = {"auxiliary": {"compression": {"reasoning_effort": "high"}}}
    with patch("hermes_cli.config.load_config", return_value=cfg):
        result = auxiliary_client._get_task_reasoning_config("compression")
    assert result == {"enabled": True, "effort": "high"}


def test_reasoning_config_none_when_empty():
    cfg = {"auxiliary": {"compression": {"reasoning_effort": ""}}}
    with patch("hermes_cli.config.load_config", return_value=cfg):
        result = auxiliary_client._get_task_reasoning_config("compression")
    assert result is None


def test_reasoning_config_none_keyword():
    cfg = {"auxiliary": {"mcp": {"reasoning_effort": "none"}}}
    with patch("hermes_cli.config.load_config", return_value=cfg):
        result = auxiliary_client._get_task_reasoning_config("mcp")
    assert result == {"enabled": False}


def test_reasoning_config_empty_task_name():
    result = auxiliary_client._get_task_reasoning_config("")
    assert result is None


def test_call_llm_injects_task_reasoning(monkeypatch):
    """call_llm should merge auxiliary.<task>.reasoning_effort into extra_body['reasoning']."""
    import agent.auxiliary_client as ac

    captured = {}

    def fake_build_call_kwargs(provider, model, messages, **kw):
        captured["extra_body"] = kw.get("extra_body")
        return {"model": model, "messages": messages, "extra_body": kw.get("extra_body")}

    class _FakeCompletions:
        @staticmethod
        def create(**kwargs):
            raise RuntimeError("stop-after-kwargs")

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        base_url = "http://127.0.0.1:20128/v1"
        chat = _FakeChat()

    monkeypatch.setattr(ac, "_build_call_kwargs", fake_build_call_kwargs)
    monkeypatch.setattr(ac, "_get_cached_client", lambda *a, **k: (_FakeClient(), "cc/claude-opus-4-8"))

    cfg = {"auxiliary": {"mcp": {
        "provider": "custom", "model": "cc/claude-opus-4-8",
        "base_url": "http://127.0.0.1:20128/v1", "api_key": "x",
        "reasoning_effort": "high",
    }}}

    with patch("hermes_cli.config.load_config", return_value=cfg):
        try:
            ac.call_llm("mcp", messages=[{"role": "user", "content": "hi"}])
        except Exception:
            pass

    assert captured.get("extra_body", {}).get("reasoning") == {"enabled": True, "effort": "high"}
