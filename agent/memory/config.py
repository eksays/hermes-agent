"""Default paths, exclusions, and intervals for the Hermes Memory System.

All values can be overridden in `.hermes/config.yaml` under the `memory:` key.
"""

DEFAULT_ROOTS = [
    "C:\\",
    "D:\\",
    "E:\\",
]

# Full disk: empty exclude set. Exclusions can be added through
# `.hermes/config.yaml` or environment overrides if specific directories
# (e.g. Windows, Program Files) need to be skipped for performance.
# File-level filtering is still applied via SKIP_EXTENSIONS and
# MAX_FILE_SIZE_BYTES.
DEFAULT_EXCLUDE_PATTERNS = set()

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

# ── Scoring & Recommendation Engine ──────────────────────────────────────
SCORING_WEIGHTS = {
    "recency": 0.25,
    "frequency": 0.20,
    "user_boost": 0.20,
    "git_activity": 0.15,
    "project_relevance": 0.10,
    "similarity": 0.10,  # TF-IDF content similarity
}
assert abs(sum(SCORING_WEIGHTS.values()) - 1.0) < 0.001, "scoring weights must sum to 1.0"

SCORING_INTERVAL_S = 60 * 60  # recompute recommendation scores every hour
STALE_DEFAULT_DAYS = 14
TFIDF_MAX_FILES = 5000
TFIDF_MAX_FILE_BYTES = 10 * 1024 * 1024  # lower than MAX_FILE_SIZE — skip large payloads from feature extraction
