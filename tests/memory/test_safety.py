from agent.memory.safety import is_path_excluded, sanitize_query


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
