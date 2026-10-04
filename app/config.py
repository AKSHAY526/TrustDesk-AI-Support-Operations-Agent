"""Application settings, loaded from environment with a tiny .env reader."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KB_DIR = DATA_DIR / "knowledge_base"
DB_PATH = BASE_DIR / "trustdesk.db"

PROMPT_VERSION = "v1"


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader so we do not need an extra dependency."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Real environment variables win over the file.
        os.environ.setdefault(key, value)


_load_dotenv(BASE_DIR / ".env")


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


class Settings:
    """Runtime configuration."""

    # AI provider. An empty API key means the offline stand-in model is used.
    llm_base_url: str = os.getenv("LLM_BASE_URL", "").rstrip("/")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    llm_timeout_seconds: float = _float("LLM_TIMEOUT_SECONDS", 20.0)

    # Demo auth: one token per role.
    token_agent: str = os.getenv("TOKEN_AGENT", "demo-agent-token")
    token_manager: str = os.getenv("TOKEN_MANAGER", "demo-manager-token")
    token_admin: str = os.getenv("TOKEN_ADMIN", "demo-admin-token")

    # Retrieval
    # Seed data, sentence-transformers: real tickets >= 0.40, unrelated questions <= 0.30.
    retrieval_min_score: float = _float("RETRIEVAL_MIN_SCORE", 0.35)
    retrieval_top_k: int = _int("RETRIEVAL_TOP_K", 4)

    @property
    def use_real_model(self) -> bool:
        return bool(self.llm_api_key and self.llm_base_url)


settings = Settings()
