"""Central, portable resolution of the persistent model/data cache.

Code never hardcodes a cache path. It resolves one here, in priority order:
  1. GEOPROBE_MODEL_CACHE   (project-specific; set this on a server)
  2. HF_HOME                (standard Hugging Face variable)
  3. ~/.cache/huggingface   (HF default; persistent on macOS/Linux)

`.env` at the repo root is read automatically (no dependency), so transferring
to a server is one line in `.env` — nothing in the code changes.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def model_cache_dir() -> Path:
    _load_dotenv()
    for var in ("GEOPROBE_MODEL_CACHE", "HF_HOME"):
        value = os.environ.get(var)
        if value:
            return Path(value).expanduser()
    return Path.home() / ".cache" / "huggingface"


def ensure_hf_env() -> Path:
    """Point HF libraries at the persistent cache. Call BEFORE importing transformers."""
    cache = model_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))
    return cache
