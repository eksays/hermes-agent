"""Default paths, exclusions, and intervals for the Hermes Memory System.

All values can be overridden in `.hermes/config.yaml` under the `memory:` key.
"""

DEFAULT_ROOTS = [
    "E:\\File\\Project AI Agent",
]

DEFAULT_EXCLUDE_PATTERNS = {
    "node_modules",
    ".git",
    ".venv",
    "__pycache__",
    "dist",
    "build",
    ".next",
    ".cache",
    ".hermes",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".turbo",
}

SKIP_EXTENSIONS = {
    ".exe", ".dll", ".so", ".dylib",
    ".bin", ".dat", ".db", ".sqlite",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico", ".svg",
    ".mp3", ".mp4", ".avi", ".mkv", ".mov",
    ".zip", ".tar", ".gz", ".7z", ".rar",
    ".xlsx", ".pptx",
    ".pyc", ".pyo",
    ".ttf", ".otf", ".woff", ".woff2",
}

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024

DEFAULT_SCAN_INTERVAL_S = 30 * 60

DEFAULT_SEARCH_LIMIT = 20
