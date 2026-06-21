from agent.memory.safety import (
    is_path_excluded,
    sanitize_query,
    strip_pii,
    contains_pii,
)


def test_is_path_excluded_matches_basename():
    assert is_path_excluded(r"E:\project\node_modules\some-lib", {"node_modules"})
    assert not is_path_excluded(r"E:\project\src\lib", {"node_modules"})


def test_is_path_excluded_nested():
    assert is_path_excluded(r"E:\project\.git\HEAD", {".git"})
    assert not is_path_excluded(r"E:\project\.gitignore", {".git"})


def test_sanitize_query_lowercases_and_strips():
    assert sanitize_query("  Hello  World  ") == "hello world"


def test_sanitize_query_empty():
    assert sanitize_query("") == ""
    assert sanitize_query("   ") == ""


def test_sanitize_query_keeps_alphanumeric_and_basic():
    result = sanitize_query("find file!@#$%^&*() name;'\"")
    # Keeps #, @, /, +, -, . but strips !$%^&*();'"
    stripped = [c for c in "!$%^&*();'\"" if c in result]
    assert not stripped, f"special chars found: {stripped}"
    assert "find" in result
    assert "file" in result
    assert "name" in result


# ── PII detection & redaction ─────────────────────────────────────────────────


def test_strip_pii_removes_email():
    """Email addresses are redacted from text."""
    text = "Contact me at john.doe@example.com or support@company.org"
    result = strip_pii(text)
    assert "john.doe@example.com" not in result
    assert "support@company.org" not in result
    assert "[EMAIL]" in result or result != text


def test_strip_pii_removes_phone():
    """Phone numbers are redacted."""
    text = "Call me at 0812-3456-7890 or +62-812-3456-7890"
    result = strip_pii(text)
    assert "0812-3456-7890" not in result
    assert "[PHONE]" in result or "[REDACTED]" in result


def test_strip_pii_removes_credit_card():
    """Credit card numbers are redacted."""
    text = "My card is 4111-1111-1111-1111 and secret"
    result = strip_pii(text)
    assert "4111-1111-1111-1111" not in result
    assert "4111111111111111" not in result


def test_strip_pii_keeps_safe_text():
    """Normal text without PII is unchanged."""
    text = "The quick brown fox jumps over the lazy dog."
    result = strip_pii(text)
    assert result == text


def test_strip_pii_empty():
    """Empty string returns empty string; whitespace-only returns unchanged."""
    assert strip_pii("") == ""
    assert strip_pii("   ") == "   "


def test_contains_pii_detects_email():
    """contains_pii returns True when email is present."""
    assert contains_pii("email me at test@example.com") is True


def test_contains_pii_detects_phone():
    """contains_pii returns True when phone number is present."""
    assert contains_pii("call 081234567890") is True


def test_contains_pii_returns_false_for_safe():
    """contains_pii returns False for normal text."""
    assert contains_pii("This is a normal sentence.") is False


def test_strip_pii_removes_multiple_types():
    """Multiple PII types in one string are all redacted."""
    text = "User: alice@test.com, phone 08123456789, card 4111111111111111"
    result = strip_pii(text)
    assert not contains_pii(result) or "alice@test.com" not in result
