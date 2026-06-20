"""Deterministic project parser — detects project type, framework, build tool,
and dependencies from a root directory.

No git CLI calls; all .git data is read from the filesystem directly.
No network access. Pure file I/O.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass
class ProjectInfo:
    root_path: str
    name: str
    project_type: str  # 'node' | 'python' | 'rust' | 'go' | 'dotnet' | 'other'
    framework: str  # 'nextjs' | 'fastapi' | 'actix' | 'django' | '' etc
    build_tool: str  # 'pnpm' | 'cargo' | 'poetry' | 'pip' | 'gradle' | '' etc


@dataclass
class ProjectDep:
    name: str
    version: str
    is_dev: bool
    dep_type: str  # 'npm' | 'pip' | 'cargo' | 'nuget' | 'go'


@dataclass
class GitInfo:
    git_path: str
    default_branch: str  # 'main' or 'master' (auto-detect)
    remote_url: str
    last_commit_hash: str
    last_commit_date: str  # ISO 8601
    last_commit_message: str
    commit_count: int
    branch_count: int


# ── Framework detection maps ──────────────────────────────────────────────────

_NODE_FRAMEWORKS: Dict[str, str] = {
    "next": "nextjs",
    "react": "react",
    "vue": "vue",
    "express": "express",
}


# ── Public API ────────────────────────────────────────────────────────────────


def detect_project(root_path: str) -> Optional[ProjectInfo]:
    """Inspect *root_path* and return a ``ProjectInfo`` if a known project
    marker file is present, or ``None`` if the directory does not contain
    any recognised project.

    Detection order: pyproject.toml → package.json → Cargo.toml → go.mod
    → *.sln / *.csproj.
    """
    if not os.path.isdir(root_path):
        return None

    root_path = os.path.abspath(root_path)
    name = os.path.basename(root_path)

    # 1. pyproject.toml → python
    if os.path.isfile(os.path.join(root_path, "pyproject.toml")):
        pyproject = _read_file(os.path.join(root_path, "pyproject.toml"))
        build_tool = _detect_python_build_tool(pyproject)
        framework = _detect_python_framework(pyproject)
        return ProjectInfo(
            root_path=root_path,
            name=name,
            project_type="python",
            framework=framework,
            build_tool=build_tool,
        )

    # 2. package.json → node
    pkg_path = os.path.join(root_path, "package.json")
    if os.path.isfile(pkg_path):
        pkg_data = _read_json_safe(pkg_path)
        framework = _detect_node_framework(pkg_data)
        build_tool = _detect_node_build_tool(root_path, pkg_data)
        return ProjectInfo(
            root_path=root_path,
            name=name,
            project_type="node",
            framework=framework,
            build_tool=build_tool,
        )

    # 3. Cargo.toml → rust
    if os.path.isfile(os.path.join(root_path, "Cargo.toml")):
        return ProjectInfo(
            root_path=root_path,
            name=name,
            project_type="rust",
            framework="",
            build_tool="cargo",
        )

    # 4. go.mod → go
    if os.path.isfile(os.path.join(root_path, "go.mod")):
        return ProjectInfo(
            root_path=root_path,
            name=name,
            project_type="go",
            framework="",
            build_tool="",
        )

    # 5. *.sln or *.csproj → dotnet
    try:
        for entry in os.scandir(root_path):
            if entry.is_file():
                lower = entry.name.lower()
                if lower.endswith(".sln") or lower.endswith(".csproj"):
                    return ProjectInfo(
                        root_path=root_path,
                        name=name,
                        project_type="dotnet",
                        framework="",
                        build_tool="",
                    )
    except PermissionError:
        pass

    return None


def parse_dependencies(project_type: str, root_path: str) -> List[ProjectDep]:
    """Parse dependency files for *project_type* under *root_path*.

    Returns an empty list on any parse error (never raises).
    """
    project_type = project_type.lower()

    parsers = {
        "node": _parse_npm_deps,
        "python": _parse_python_deps,
        "rust": _parse_cargo_deps,
        "go": _parse_go_deps,
        "dotnet": _parse_dotnet_deps,
    }

    parser = parsers.get(project_type)
    if parser is None:
        return []
    try:
        return parser(root_path)
    except Exception as exc:
        logger.warning(
            "Failed to parse %s dependencies in %s: %s",
            project_type,
            root_path,
            exc,
        )
        return []


def detect_git_info(root_path: str) -> Optional[GitInfo]:
    """Read ``.git`` files directly to extract repository metadata.

    No git CLI calls — only uses ``os`` and basic file I/O so it is
    deterministic and works even when git is not installed.

    Returns ``None`` if ``.git`` does not exist or cannot be read.
    """
    git_path = os.path.join(root_path, ".git")
    if not os.path.isdir(git_path):
        return None

    try:
        head_content = _read_file(os.path.join(git_path, "HEAD")) or ""
        head_ref = head_content.strip()

        # Determine default branch: read HEAD to find current branch name,
        # then check which of main/master exists.
        current_branch = ""
        if head_ref.startswith("ref: "):
            current_branch = head_ref[5:].removeprefix("refs/heads/")
        else:
            # Detached HEAD — hash only
            current_branch = ""

        default_branch = _detect_default_branch(git_path, current_branch)
        remote_url = _read_remote_url(git_path)
        last_commit_hash, last_commit_message = _read_last_commit(git_path)
        last_commit_date = _read_last_commit_date(git_path)
        commit_count = _count_commits(git_path)
        branch_count = _count_branches(git_path)

        return GitInfo(
            git_path=git_path,
            default_branch=default_branch or "main",
            remote_url=remote_url or "",
            last_commit_hash=last_commit_hash or "",
            last_commit_date=last_commit_date or "",
            last_commit_message=last_commit_message or "",
            commit_count=commit_count,
            branch_count=branch_count,
        )
    except Exception as exc:
        logger.warning("Failed to read git info in %s: %s", root_path, exc)
        return None


def find_projects(roots: List[str], exclude_patterns: Set[str]) -> List[str]:
    """Walk *roots* and return paths where ``detect_project()`` would succeed.

    Skips directories whose basename matches any pattern in *exclude_patterns*.
    Walks at most 5 levels deep.
    """
    found: List[str] = []

    for root in roots:
        if not os.path.isdir(root):
            continue
        root = os.path.abspath(root)
        _walk_for_projects(root, exclude_patterns, depth=0, max_depth=5, found=found)

    # Deduplicate while preserving order
    seen: Set[str] = set()
    unique: List[str] = []
    for p in found:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


# ── Internal helpers ──────────────────────────────────────────────────────────


def _walk_for_projects(
    directory: str,
    exclude_patterns: Set[str],
    depth: int,
    max_depth: int,
    found: List[str],
) -> None:
    """Recursive walk helper for ``find_projects``."""
    if depth > max_depth:
        return

    basename = os.path.basename(directory)
    if basename in exclude_patterns:
        return

    if _has_project_marker(directory):
        found.append(directory)
        # Do NOT descend into a found project's subdirectories — treat it
        # as a leaf.
        return

    if depth == max_depth:
        return

    try:
        for entry in os.scandir(directory):
            if entry.is_dir():
                _walk_for_projects(
                    entry.path,
                    exclude_patterns,
                    depth + 1,
                    max_depth,
                    found,
                )
    except PermissionError:
        pass


def _has_project_marker(directory: str) -> bool:
    """Quick check — does *directory* contain any known project marker?"""
    markers = {
        "package.json",
        "pyproject.toml",
        "Cargo.toml",
        "go.mod",
    }
    try:
        for entry in os.scandir(directory):
            if entry.is_file() and entry.name in markers:
                return True
            if entry.is_file():
                lower = entry.name.lower()
                if lower.endswith(".sln") or lower.endswith(".csproj"):
                    return True
    except PermissionError:
        pass
    return False


# ── File I/O helpers ──────────────────────────────────────────────────────────


def _read_file(path: str) -> str:
    """Read a text file, return empty string on error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except (FileNotFoundError, PermissionError, OSError):
        return ""


def _read_json_safe(path: str) -> Optional[dict]:
    """Read and parse a JSON file, return ``None`` on error."""
    content = _read_file(path)
    if not content:
        return None
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return data
        return None
    except (json.JSONDecodeError, ValueError):
        return None


# ── Python detection ──────────────────────────────────────────────────────────


def _detect_python_build_tool(content: str) -> str:
    """Detect python build tool from pyproject.toml contents."""
    # Very basic TOML parsing — only look at [build-system] build-backend
    match = re.search(
        r'^build-backend\s*=\s*"([^"]+)"',
        content,
        re.MULTILINE,
    )
    if not match:
        return ""

    backend = match.group(1)
    if "poetry" in backend:
        return "poetry"
    if "hatch" in backend:
        return "hatch"
    if "pdm" in backend:
        return "pdm"
    if "setuptools" in backend:
        return "pip"
    return ""


def _detect_python_framework(content: str) -> str:
    """Detect python framework from pyproject.toml dependencies."""
    # Quick check for common frameworks in [project] dependencies
    if re.search(r'"fastapi"', content, re.MULTILINE):
        return "fastapi"
    if re.search(r'"django"', content, re.MULTILINE):
        return "django"
    if re.search(r'"flask"', content, re.MULTILINE):
        return "flask"
    return ""


# ── Node detection ────────────────────────────────────────────────────────────


def _detect_node_framework(pkg_data: Optional[dict]) -> str:
    """Detect node framework from parsed package.json."""
    if pkg_data is None:
        return ""

    all_deps: Dict[str, str] = {}
    for key in ("dependencies", "devDependencies"):
        deps = pkg_data.get(key, {})
        if isinstance(deps, dict):
            all_deps.update(deps)

    for dep_name, framework_name in _NODE_FRAMEWORKS.items():
        if dep_name in all_deps:
            return framework_name

    return ""


def _detect_node_build_tool(root_path: str, pkg_data: Optional[dict]) -> str:
    """Detect node build tool from lockfiles or package.json metadata."""
    if os.path.isfile(os.path.join(root_path, "pnpm-lock.yaml")):
        return "pnpm"
    if os.path.isfile(os.path.join(root_path, "yarn.lock")):
        return "yarn"
    if os.path.isfile(os.path.join(root_path, "package-lock.json")):
        return "npm"
    return "pnpm"


# ── Dependency parsers ────────────────────────────────────────────────────────


def _parse_npm_deps(root_path: str) -> List[ProjectDep]:
    """Parse package.json and return npm dependencies."""
    pkg_path = os.path.join(root_path, "package.json")
    data = _read_json_safe(pkg_path)
    if data is None:
        return []

    deps: List[ProjectDep] = []

    raw: Dict[str, dict] = data.get("dependencies", {})
    if isinstance(raw, dict):
        for name, version in raw.items():
            if isinstance(version, str):
                deps.append(
                    ProjectDep(
                        name=name,
                        version=version,
                        is_dev=False,
                        dep_type="npm",
                    )
                )

    dev_raw: Dict[str, dict] = data.get("devDependencies", {})
    if isinstance(dev_raw, dict):
        for name, version in dev_raw.items():
            if isinstance(version, str):
                deps.append(
                    ProjectDep(
                        name=name,
                        version=version,
                        is_dev=True,
                        dep_type="npm",
                    )
                )

    return deps


def _parse_python_deps(root_path: str) -> List[ProjectDep]:
    """Parse pyproject.toml and/or requirements.txt."""
    deps: List[ProjectDep] = []

    # Try pyproject.toml first
    pyproject_path = os.path.join(root_path, "pyproject.toml")
    if os.path.isfile(pyproject_path):
        content = _read_file(pyproject_path)
        if content:
            deps.extend(_parse_pyproject_deps(content))

    # Also parse requirements.txt if it exists
    req_path = os.path.join(root_path, "requirements.txt")
    if os.path.isfile(req_path):
        content = _read_file(req_path)
        if content:
            deps.extend(_parse_requirements_txt(content))

    return deps


def _parse_pyproject_deps(content: str) -> List[ProjectDep]:
    """Parse ``[project] dependencies`` and ``[project.optional-dependencies]``."""
    deps: List[ProjectDep] = []

    # [project] dependencies — look for lines between "dependencies = [" and "]"
    in_deps = False
    bracket_depth = 0
    for line in content.splitlines():
        stripped = line.strip()

        # Detect start of dependencies array
        if stripped.startswith("dependencies"):
            if "[" in stripped and "]" in stripped:
                # Single line like `dependencies = ["foo==1.0",]`
                arr_start = stripped.index("[")
                arr_end = stripped.rindex("]")
                arr_content = stripped[arr_start + 1 : arr_end]
                for item in _split_array_items(arr_content):
                    dep = _parse_pip_dep(item)
                    if dep:
                        deps.append(dep)
                continue
            elif "[" in stripped:
                in_deps = True
                bracket_depth = stripped.count("[") - stripped.count("]")
                # Extract any items after the [ on the same line
                arr_start = stripped.index("[")
                after = stripped[arr_start + 1 :]
                for item in _split_array_items(after):
                    dep = _parse_pip_dep(item)
                    if dep:
                        deps.append(dep)
                if bracket_depth <= 0:
                    in_deps = False
                continue

        if in_deps:
            bracket_depth += stripped.count("[") - stripped.count("]")
            for item in _split_array_items(stripped):
                dep = _parse_pip_dep(item)
                if dep:
                    deps.append(dep)
            if bracket_depth <= 0:
                in_deps = False

    # [project.optional-dependencies] — all are dev deps
    in_optional = False
    bracket_depth = 0
    for line in content.splitlines():
        stripped = line.strip()

        # Match [project.optional-dependencies] or [project.optional-dependencies.X]
        if re.match(r"^\[project\.optional-dependencies", stripped):
            in_optional = True
            bracket_depth = 0
            continue

        if in_optional:
            if stripped.startswith("["):
                in_optional = False
                continue

            # TOML key = [ ... ] — start of an option group array
            if "=" in stripped and "[" in stripped:
                # Parse everything after the =
                _, _, rest = stripped.partition("=")
                rest = rest.strip()
                if rest.startswith("["):
                    bracket_depth = rest.count("[") - rest.count("]")
                    arr_content = rest[1:] if bracket_depth <= 0 else rest[1:]
                    for item in _split_array_items(arr_content):
                        dep = _parse_pip_dep(item)
                        if dep:
                            dep.is_dev = True
                            deps.append(dep)
                    if bracket_depth <= 0:
                        bracket_depth = 0
                continue

            # Inside an array continuation
            if bracket_depth > 0:
                bracket_depth += stripped.count("[") - stripped.count("]")
                for item in _split_array_items(stripped):
                    dep = _parse_pip_dep(item)
                    if dep:
                        dep.is_dev = True
                        deps.append(dep)
                continue

    return deps


def _parse_requirements_txt(content: str) -> List[ProjectDep]:
    """Parse a requirements.txt file."""
    deps: List[ProjectDep] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            continue
        dep = _parse_pip_dep(stripped)
        if dep:
            deps.append(dep)
    return deps


def _parse_pip_dep(item: str) -> Optional[ProjectDep]:
    """Parse a single pip-style dependency string.

    Handles: ``name==version``, ``name>=version``, ``name``,
    ``name[extra]>=version``, and quoted items.
    """
    item = item.strip().strip('",\'')

    if not item or item.startswith("#"):
        return None

    # Markers after ; are not relevant for name/version extraction
    if ";" in item:
        item = item.split(";", 1)[0].strip()

    # Strip extras like [standard]
    name_part = re.sub(r"\[.*?\]", "", item).strip()
    extras = ""  # noqa: F841

    # Match name + version operator + version
    match = re.match(
        r"^([a-zA-Z0-9_][a-zA-Z0-9_.-]*?)\s*"
        r"(?:==|>=|<=|!=|~=|>|<|===)\s*"
        r"([a-zA-Z0-9_.*]+(?:\.[a-zA-Z0-9_.*]+)*)",
        name_part,
    )
    if match:
        return ProjectDep(
            name=match.group(1),
            version=match.group(2),
            is_dev=False,
            dep_type="pip",
        )

    # No version constraint — just the name
    name_match = re.match(r"^([a-zA-Z0-9_][a-zA-Z0-9_.-]*)", name_part)
    if name_match:
        return ProjectDep(
            name=name_match.group(1),
            version="",
            is_dev=False,
            dep_type="pip",
        )

    return None


def _split_array_items(content: str) -> List[str]:
    """Split a TOML/INI array content string into individual items.

    Handles multi-line and comma-separated entries.
    """
    items: List[str] = []
    current: List[str] = []
    in_quote = False
    quote_char = ""
    for char in content:
        if in_quote:
            if char == quote_char:
                in_quote = False
            current.append(char)
        elif char in ('"', "'"):
            in_quote = True
            quote_char = char
            current.append(char)
        elif char == ",":
            item = "".join(current).strip()
            if item:
                items.append(item)
            current = []
        elif char not in ("\n", "\r", " ", "\t"):
            current.append(char)
        else:
            current.append(char)

    item = "".join(current).strip()
    if item:
        items.append(item)
    return items


def _parse_cargo_deps(root_path: str) -> List[ProjectDep]:
    """Parse Cargo.toml [dependencies] section."""
    content = _read_file(os.path.join(root_path, "Cargo.toml"))
    if not content:
        return []

    deps: List[ProjectDep] = []
    in_deps = False

    for line in content.splitlines():
        stripped = line.strip()

        if stripped == "[dependencies]":
            in_deps = True
            continue

        if in_deps:
            # Stop at next section
            if stripped.startswith("["):
                break

            if not stripped or stripped.startswith("#"):
                continue

            # Parse "name = "version"" or "name = { version = "1.0", ... }"
            dep = _parse_cargo_dep_line(stripped)
            if dep:
                deps.append(dep)

    return deps


def _parse_cargo_dep_line(line: str) -> Optional[ProjectDep]:
    """Parse a single Cargo dependency line."""
    # "name = { version = "1.0", features = [...] }"
    inline_match = re.match(
        r'^\s*([a-zA-Z0-9_-]+)\s*=\s*\{\s*version\s*=\s*"([^"]+)"',
        line,
    )
    if inline_match:
        return ProjectDep(
            name=inline_match.group(1),
            version=inline_match.group(2),
            is_dev=False,
            dep_type="cargo",
        )

    # "name = "1.0""
    simple_match = re.match(
        r'^\s*([a-zA-Z0-9_-]+)\s*=\s*"([^"]+)"',
        line,
    )
    if simple_match:
        return ProjectDep(
            name=simple_match.group(1),
            version=simple_match.group(2),
            is_dev=False,
            dep_type="cargo",
        )

    # "name"  (simple name without version — path/git dep)
    bare_match = re.match(r"^\s*([a-zA-Z0-9_-]+)\s*$", line)
    if bare_match:
        return ProjectDep(
            name=bare_match.group(1),
            version="",
            is_dev=False,
            dep_type="cargo",
        )

    return None


def _parse_go_deps(root_path: str) -> List[ProjectDep]:
    """Parse go.mod for require blocks."""
    # Basic parsing — limited scope
    return []


def _parse_dotnet_deps(root_path: str) -> List[ProjectDep]:
    """Parse .csproj for PackageReference entries."""
    # Basic parsing — limited scope
    return []


# ── Git info helpers ──────────────────────────────────────────────────────────


def _detect_default_branch(git_path: str, current_branch: str) -> str:
    """Detect default branch by checking refs/heads/main vs master."""
    # If HEAD points to a known branch, use it if it exists
    if current_branch:
        branch_path = os.path.join(git_path, "refs", "heads", current_branch)
        if os.path.isfile(branch_path):
            return current_branch

    # Otherwise check which of main/master has a ref file
    if os.path.isfile(os.path.join(git_path, "refs", "heads", "main")):
        return "main"
    if os.path.isfile(os.path.join(git_path, "refs", "heads", "master")):
        return "master"

    # Last resort: scan refs/heads for any branch
    heads_dir = os.path.join(git_path, "refs", "heads")
    if os.path.isdir(heads_dir):
        try:
            for entry in os.scandir(heads_dir):
                if entry.is_file():
                    return entry.name
        except PermissionError:
            pass

    return current_branch


def _read_remote_url(git_path: str) -> str:
    """Read remote origin URL from .git/config."""
    config_path = os.path.join(git_path, "config")
    content = _read_file(config_path)
    if not content:
        return ""

    # Look for [remote "origin"] section
    in_origin = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("[remote"):
            in_origin = '"origin"' in stripped
        elif in_origin:
            if stripped.startswith("["):
                break
            url_match = re.match(r'\s*url\s*=\s*(.+)$', stripped)
            if url_match:
                return url_match.group(1).strip()

    return ""


def _read_last_commit(git_path: str) -> tuple:
    """Read last commit hash and message from .git/logs/HEAD."""
    log_path = os.path.join(git_path, "logs", "HEAD")
    content = _read_file(log_path)
    if not content:
        return ("", "")

    lines = [l for l in content.splitlines() if l.strip()]
    if not lines:
        return ("", "")

    last_line = lines[-1]
    parts = last_line.split("\t", 1)
    if len(parts) < 2:
        return ("", "")

    # Hash is the second token in the first part
    log_parts = parts[0].split()
    if len(log_parts) < 2:
        # Less than two tokens — try to extract from what we have
        hash = log_parts[-1] if log_parts else ""
    else:
        # new_hash is the second field
        hash = log_parts[1] if len(log_parts) > 1 else log_parts[0] if log_parts else ""

    message = parts[1].strip()

    return (hash, message)


def _read_last_commit_date(git_path: str) -> str:
    """Read last commit date (ISO 8601) from .git/logs/HEAD."""
    log_path = os.path.join(git_path, "logs", "HEAD")
    content = _read_file(log_path)
    if not content:
        return ""

    lines = [l for l in content.splitlines() if l.strip()]
    if not lines:
        return ""

    last_line = lines[-1]
    # Format: old_hash new_hash author_name <author_email> timestamp offset\tmessage
    # The timestamp is the first whitespace-delimited token AFTER the closing > of the email.
    # Author names with spaces (e.g. "John Doe") make simple split()+index unreliable.
    match = re.search(r">\s+(\d+)\s+[+-]\d{4}", last_line)
    if match:
        try:
            timestamp = int(match.group(1))
            import datetime

            dt = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
            return dt.isoformat()
        except (ValueError, IndexError):
            pass

    # Fallback: try looking for a unix timestamp in the last 60 chars (compact format)
    # without angle-bracket email
    try:
        parts = last_line.split()
        for i, p in enumerate(parts):
            if p.isdigit() and len(p) >= 8 and i + 1 < len(parts) and re.match(r"[+-]\d{4}", parts[i + 1]):
                timestamp = int(p)
                import datetime

                dt = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
                return dt.isoformat()
    except (ValueError, IndexError):
        pass

    return ""


def _count_commits(git_path: str) -> int:
    """Count commits by counting lines in .git/logs/HEAD."""
    log_path = os.path.join(git_path, "logs", "HEAD")
    content = _read_file(log_path)
    if not content:
        return 0
    lines = [l for l in content.splitlines() if l.strip()]
    return len(lines)


def _count_branches(git_path: str) -> int:
    """Count branches by listing .git/refs/heads/ entries."""
    heads_dir = os.path.join(git_path, "refs", "heads")
    if not os.path.isdir(heads_dir):
        return 0
    try:
        count = 0
        for entry in os.scandir(heads_dir):
            if entry.is_file():
                count += 1
        return count
    except PermissionError:
        return 0
