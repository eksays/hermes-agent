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
