"""Tests for agent.memory.project_parser — project detection and git info."""

import os
import tempfile

import pytest

from agent.memory.project_parser import (
    ProjectInfo,
    ProjectDep,
    GitInfo,
    detect_project,
    parse_dependencies,
    detect_git_info,
    find_projects,
)


# ── detect_project ────────────────────────────────────────────────────────────


def test_detect_project_python():
    """pyproject.toml with [build-system] using poetry → python + poetry."""
    with tempfile.TemporaryDirectory(prefix="hermes_test_") as root:
        _write_file(root, "pyproject.toml", _PYPROJECT_POETRY)
        info = detect_project(root)
        assert info is not None
        assert info.root_path == os.path.abspath(root)
        assert info.project_type == "python"
        assert info.build_tool == "poetry"
        assert os.path.basename(root) in info.name


def test_detect_project_node():
    """package.json with next dependency → node + nextjs."""
    with tempfile.TemporaryDirectory(prefix="hermes_test_") as root:
        _write_file(
            root,
            "package.json",
            '{"dependencies": {"next": "13.0.0"}}',
        )
        info = detect_project(root)
        assert info is not None
        assert info.root_path == os.path.abspath(root)
        assert info.project_type == "node"
        assert info.framework == "nextjs"
        assert info.build_tool == "pnpm"


def test_detect_project_node_react():
    """package.json with react but no next → react framework."""
    with tempfile.TemporaryDirectory(prefix="hermes_test_") as root:
        _write_file(
            root,
            "package.json",
            '{"dependencies": {"react": "18.0.0"}}',
        )
        info = detect_project(root)
        assert info is not None
        assert info.project_type == "node"
        assert info.framework == "react"


def test_detect_project_node_vue():
    """package.json with vue → vue framework."""
    with tempfile.TemporaryDirectory(prefix="hermes_test_") as root:
        _write_file(
            root,
            "package.json",
            '{"dependencies": {"vue": "3.0.0"}}',
        )
        info = detect_project(root)
        assert info is not None
        assert info.project_type == "node"
        assert info.framework == "vue"


def test_detect_project_node_express():
    """package.json with express → express framework."""
    with tempfile.TemporaryDirectory(prefix="hermes_test_") as root:
        _write_file(
            root,
            "package.json",
            '{"dependencies": {"express": "4.0.0"}}',
        )
        info = detect_project(root)
        assert info is not None
        assert info.project_type == "node"
        assert info.framework == "express"


def test_detect_project_node_plain():
    """package.json with no known framework → node with empty framework."""
    with tempfile.TemporaryDirectory(prefix="hermes_test_") as root:
        _write_file(
            root,
            "package.json",
            '{"dependencies": {"lodash": "4.0.0"}}',
        )
        info = detect_project(root)
        assert info is not None
        assert info.project_type == "node"
        assert info.framework == ""


def test_detect_project_rust():
    """Cargo.toml → rust + cargo."""
    with tempfile.TemporaryDirectory(prefix="hermes_test_") as root:
        _write_file(
            root,
            "Cargo.toml",
            "[package]\nname = \"myapp\"\nversion = \"0.1.0\"\n",
        )
        info = detect_project(root)
        assert info is not None
        assert info.project_type == "rust"
        assert info.build_tool == "cargo"


def test_detect_project_go():
    """go.mod → go."""
    with tempfile.TemporaryDirectory(prefix="hermes_test_") as root:
        _write_file(
            root,
            "go.mod",
            "module myapp\ngo 1.21\n",
        )
        info = detect_project(root)
        assert info is not None
        assert info.project_type == "go"


def test_detect_project_dotnet():
    """*.sln file → dotnet."""
    with tempfile.TemporaryDirectory(prefix="hermes_test_") as root:
        _write_file(root, "MyApp.sln", "\n")
        info = detect_project(root)
        assert info is not None
        assert info.project_type == "dotnet"


def test_detect_project_csproj():
    """*.csproj file → dotnet."""
    with tempfile.TemporaryDirectory(prefix="hermes_test_") as root:
        _write_file(root, "MyApp.csproj", "<Project />")
        info = detect_project(root)
        assert info is not None
        assert info.project_type == "dotnet"


def test_detect_project_none():
    """Empty directory with no project markers returns None."""
    with tempfile.TemporaryDirectory(prefix="hermes_test_") as root:
        info = detect_project(root)
        assert info is None


def test_detect_project_none_no_dir():
    """Non-existent directory returns None."""
    info = detect_project(r"E:\nonexistent_dir_xyzzy")
    assert info is None


def test_detect_project_python_none_in_build_system():
    """pyproject.toml without [build-system] → python, empty build_tool."""
    with tempfile.TemporaryDirectory(prefix="hermes_test_") as root:
        _write_file(
            root,
            "pyproject.toml",
            "[project]\nname = \"myapp\"\n",
        )
        info = detect_project(root)
        assert info is not None
        assert info.project_type == "python"
        assert info.build_tool == ""


# ── parse_dependencies ────────────────────────────────────────────────────────


def test_parse_dependencies_npm():
    """Parse package.json and return correct ProjectDep entries."""
    with tempfile.TemporaryDirectory(prefix="hermes_test_") as root:
        _write_file(
            root,
            "package.json",
            _PACKAGE_JSON_DEPS,
        )
        deps = parse_dependencies("node", root)
        assert len(deps) == 3

        dep_map = {(d.name, d.is_dev): d for d in deps}

        # Runtime dep
        prod = dep_map.get(("next", False))
        assert prod is not None
        assert prod.version == "13.0.0"
        assert prod.dep_type == "npm"

        # Dev dep
        dev = dep_map.get(("typescript", True))
        assert dev is not None
        assert dev.version == "5.0.0"
        assert dev.dep_type == "npm"

        # Another dev dep
        dev2 = dep_map.get(("@types/react", True))
        assert dev2 is not None
        assert dev2.version == "18.0.0"
        assert dev2.dep_type == "npm"


def test_parse_dependencies_python():
    """Parse pyproject.toml and return correct ProjectDep entries."""
    with tempfile.TemporaryDirectory(prefix="hermes_test_") as root:
        _write_file(root, "pyproject.toml", _PYPROJECT_DEPS)
        deps = parse_dependencies("python", root)
        assert len(deps) >= 2
        dep_map = {d.name: d for d in deps}

        fastapi = dep_map.get("fastapi")
        assert fastapi is not None
        assert fastapi.version == "0.104.0"
        assert fastapi.is_dev is False
        assert fastapi.dep_type == "pip"

        pytest_dep = dep_map.get("pytest")
        assert pytest_dep is not None
        assert pytest_dep.version == "8.0.0"
        assert pytest_dep.is_dev is True
        assert pytest_dep.dep_type == "pip"


def test_parse_dependencies_python_requirements():
    """Parse requirements.txt and return correct ProjectDep entries."""
    with tempfile.TemporaryDirectory(prefix="hermes_test_") as root:
        _write_file(
            root,
            "requirements.txt",
            "requests==2.31.0\nflask==3.0.0\n",
        )
        deps = parse_dependencies("python", root)
        assert len(deps) == 2
        dep_map = {d.name: d for d in deps}

        requests = dep_map.get("requests")
        assert requests is not None
        assert requests.version == "2.31.0"
        assert requests.is_dev is False

        flask = dep_map.get("flask")
        assert flask is not None
        assert flask.version == "3.0.0"


def test_parse_dependencies_rust():
    """Parse Cargo.toml and return correct ProjectDep entries."""
    with tempfile.TemporaryDirectory(prefix="hermes_test_") as root:
        _write_file(
            root,
            "Cargo.toml",
            _CARGO_DEPS,
        )
        deps = parse_dependencies("rust", root)
        assert len(deps) > 0
        dep_map = {d.name: d for d in deps}

        serde = dep_map.get("serde")
        assert serde is not None
        assert serde.version == "1.0"
        assert serde.is_dev is False
        assert serde.dep_type == "cargo"


def test_parse_dependencies_unknown_type():
    """Unknown project type returns empty list."""
    deps = parse_dependencies("unknown", r"C:\nonexistent")
    assert deps == []


def test_parse_dependencies_missing_file():
    """Missing dependency file returns empty list (no crash)."""
    with tempfile.TemporaryDirectory(prefix="hermes_test_") as root:
        deps = parse_dependencies("node", root)
        assert deps == []


def test_parse_dependencies_bad_json():
    """Corrupt package.json returns empty list (no crash)."""
    with tempfile.TemporaryDirectory(prefix="hermes_test_") as root:
        _write_file(root, "package.json", "not valid json")
        deps = parse_dependencies("node", root)
        assert deps == []


# ── detect_git_info ───────────────────────────────────────────────────────────


def test_detect_git_info():
    """Detect git info by reading .git files directly."""
    with tempfile.TemporaryDirectory(prefix="hermes_test_") as root:
        _init_git_dir(root)

        info = detect_git_info(root)

        assert info is not None
        assert info.git_path == os.path.abspath(os.path.join(root, ".git"))
        assert info.default_branch in ("main", "master")
        assert "example.com" in info.remote_url or "github.com" in info.remote_url
        assert len(info.last_commit_hash) == 40
        assert info.last_commit_date.startswith("202")
        assert len(info.last_commit_message) > 0
        assert info.commit_count >= 1
        assert info.branch_count >= 1


def test_detect_git_no_repo():
    """No .git directory returns None."""
    with tempfile.TemporaryDirectory(prefix="hermes_test_") as root:
        info = detect_git_info(root)
        assert info is None


def test_detect_git_detached_head():
    """Detached HEAD state (HEAD with hash) works correctly."""
    with tempfile.TemporaryDirectory(prefix="hermes_test_") as root:
        git_dir = os.path.join(root, ".git")
        os.makedirs(os.path.join(git_dir, "refs", "heads"))
        os.makedirs(os.path.join(git_dir, "objects"))
        os.makedirs(os.path.join(git_dir, "logs"))

        # HEAD contains a hash (detached)
        _write_file(
            git_dir,
            "HEAD",
            "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2\n",
        )
        # Config with remote
        _write_file(
            git_dir,
            "config",
            _GIT_CONFIG,
        )
        # Branch list
        os.makedirs(os.path.join(git_dir, "refs", "heads"), exist_ok=True)
        _write_file(
            os.path.join(git_dir, "refs", "heads"),
            "main",
            "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2\n",
        )
        # Log with commit info
        _write_file(
            git_dir,
            "logs/HEAD",
            _GIT_LOG_ENTRY,
        )

        info = detect_git_info(root)
        assert info is not None
        assert info.last_commit_hash == "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
        assert info.default_branch in ("main", "master")


# ── find_projects ─────────────────────────────────────────────────────────────


def test_find_projects():
    """Find project directories from root paths."""
    with tempfile.TemporaryDirectory(prefix="hermes_test_") as root:
        # Create a python project
        py_dir = os.path.join(root, "backend")
        os.makedirs(py_dir)
        _write_file(py_dir, "pyproject.toml", _PYPROJECT_POETRY)

        # Create a node project
        node_dir = os.path.join(root, "frontend")
        os.makedirs(node_dir)
        _write_file(
            node_dir,
            "package.json",
            '{"dependencies": {"react": "18.0.0"}}',
        )

        # Create a non-project dir
        other_dir = os.path.join(root, "docs")
        os.makedirs(other_dir)
        _write_file(other_dir, "readme.txt", "hello")

        found = find_projects([root], set())
        found_abs = [os.path.abspath(p) for p in found]
        assert os.path.abspath(py_dir) in found_abs
        assert os.path.abspath(node_dir) in found_abs
        assert os.path.abspath(other_dir) not in found_abs


def test_find_projects_skips_excluded():
    """Skip directories matching exclude patterns."""
    with tempfile.TemporaryDirectory(prefix="hermes_test_") as root:
        # Project in root
        _write_file(
            root,
            "package.json",
            '{"dependencies": {"express": "4.0.0"}}',
        )

        # Node_modules sub-project should be excluded
        nm_dir = os.path.join(root, "node_modules", "some-lib")
        os.makedirs(nm_dir)
        _write_file(
            nm_dir,
            "package.json",
            '{"dependencies": {"left-pad": "1.0.0"}}',
        )

        found = find_projects([root], {"node_modules"})
        found_abs = [os.path.abspath(p) for p in found]
        assert os.path.abspath(root) in found_abs
        assert os.path.abspath(nm_dir) not in found_abs


def test_find_projects_max_depth():
    """find_projects respects max depth of 5 levels."""
    with tempfile.TemporaryDirectory(prefix="hermes_test_") as root:
        # Create a deep nested project at depth 6
        deep_dir = root
        for _ in range(6):
            deep_dir = os.path.join(deep_dir, "sub")
            os.makedirs(deep_dir, exist_ok=True)
        _write_file(
            deep_dir,
            "package.json",
            '{"dependencies": {"react": "18.0.0"}}',
        )

        found = find_projects([root], set())
        deep_abs = os.path.abspath(deep_dir)
        # Depth 6 should not be found (max is 5)
        assert deep_abs not in [os.path.abspath(p) for p in found]


# ── Fixture helpers ───────────────────────────────────────────────────────────


def _write_file(directory: str, filename: str, content: str) -> str:
    """Write *content* to *directory*/*filename* and return the full path."""
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def _init_git_dir(root: str) -> None:
    """Manually create a minimal .git directory structure (no git CLI)."""
    git_dir = os.path.join(root, ".git")
    os.makedirs(os.path.join(git_dir, "refs", "heads"))
    os.makedirs(os.path.join(git_dir, "objects"))
    os.makedirs(os.path.join(git_dir, "logs"))

    # HEAD points to refs/heads/main
    _write_file(git_dir, "HEAD", "ref: refs/heads/main\n")
    # Config with remote origin
    _write_file(git_dir, "config", _GIT_CONFIG)
    # Commit hash for main branch
    _write_file(
        os.path.join(git_dir, "refs", "heads"),
        "main",
        "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2\n",
    )
    # Also create a master branch to test default_branch detection
    _write_file(
        os.path.join(git_dir, "refs", "heads"),
        "master",
        "b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3\n",
    )
    # Log entry with commit details
    _write_file(
        git_dir,
        "logs/HEAD",
        _GIT_LOG_ENTRY,
    )


# ── Test data ─────────────────────────────────────────────────────────────────

_PYPROJECT_POETRY = """\
[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[project]
name = "myapp"
version = "0.1.0"
"""

_PYPROJECT_DEPS = """\
[project]
name = "myapp"
version = "0.1.0"
dependencies = [
    "fastapi==0.104.0",
    "uvicorn[standard]>=0.24.0",
]

[project.optional-dependencies]
dev = [
    "pytest==8.0.0",
]
"""

_PACKAGE_JSON_DEPS = """\
{
    "dependencies": {
        "next": "13.0.0"
    },
    "devDependencies": {
        "typescript": "5.0.0",
        "@types/react": "18.0.0"
    }
}
"""

_CARGO_DEPS = """\
[package]
name = "myapp"
version = "0.1.0"

[dependencies]
serde = "1.0"
tokio = { version = "1.0", features = ["full"] }
"""

_GIT_CONFIG = """\
[core]
	repositoryformatversion = 0
	filemode = true
	bare = false
	logallrefupdates = true
[remote "origin"]
	url = https://github.com/example/myapp.git
	fetch = +refs/heads/*:refs/remotes/origin/*
[branch "main"]
	remote = origin
	merge = refs/heads/main
"""

_GIT_LOG_ENTRY = """\
0000000000000000000000000000000000000000 a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2 John Doe <john@example.com> 1700000000 +0000	commit (initial): Initial commit
"""
