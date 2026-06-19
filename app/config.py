"""Central configuration, overridable via environment variables.

Copy ``.env.example`` to ``.env`` (it is auto-loaded if python-dotenv is
installed) or export the variables directly.
"""

from __future__ import annotations

import os
from pathlib import Path

# Load a .env file if python-dotenv is available (optional convenience).
try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


# Repository root (one level up from this file's package).
ROOT = Path(__file__).resolve().parent.parent

# --- Ollama (local LLM host) ---------------------------------------------
OLLAMA_HOST: str = _env("OLLAMA_HOST", "http://localhost:11434")
CHAT_MODEL: str = _env("CHAT_MODEL", "llama3")
EMBED_MODEL: str = _env("EMBED_MODEL", "nomic-embed-text")

# --- Data / storage locations --------------------------------------------
DATA_DIR: Path = Path(_env("DATA_DIR", str(ROOT / "data"))).expanduser()
CHROMA_DIR: Path = Path(_env("CHROMA_DIR", str(ROOT / "chroma_db"))).expanduser()
COLLECTION: str = _env("COLLECTION", "knowledge")

# --- Chunking / retrieval knobs ------------------------------------------
CHUNK_SIZE: int = int(_env("CHUNK_SIZE", "1000"))      # characters per chunk
CHUNK_OVERLAP: int = int(_env("CHUNK_OVERLAP", "150"))  # overlap between chunks
TOP_K: int = int(_env("TOP_K", "5"))                    # chunks retrieved per query

# Request timeout (seconds) for Ollama calls. This is the *read* budget — how
# long to wait for the model to load and generate. Cold-starting a large model
# on CPU can take a while, so keep this generous.
OLLAMA_TIMEOUT: float = float(_env("OLLAMA_TIMEOUT", "300"))
# Separate, short connect timeout so an unreachable server fails fast instead
# of waiting the full read budget.
OLLAMA_CONNECT_TIMEOUT: float = float(_env("OLLAMA_CONNECT_TIMEOUT", "10"))

# --- Performance knobs ----------------------------------------------------
# How long Ollama keeps a model resident after a request. "30m" avoids paying
# the model load cost on every query; "-1" keeps it loaded indefinitely.
OLLAMA_KEEP_ALIVE: str = _env("OLLAMA_KEEP_ALIVE", "30m")
# Cap generated tokens so the model can't ramble (also bounds latency).
NUM_PREDICT: int = int(_env("NUM_PREDICT", "512"))
# Context window. Smaller = faster prompt processing; must fit prompt+context.
NUM_CTX: int = int(_env("NUM_CTX", "4096"))


def summary() -> str:
    """Human-readable view of the active configuration."""
    return (
        f"Ollama host : {OLLAMA_HOST}\n"
        f"Chat model  : {CHAT_MODEL}\n"
        f"Embed model : {EMBED_MODEL}\n"
        f"Data dir    : {DATA_DIR}\n"
        f"Chroma dir  : {CHROMA_DIR}\n"
        f"Collection  : {COLLECTION}\n"
        f"Chunk size  : {CHUNK_SIZE} (overlap {CHUNK_OVERLAP})\n"
        f"Top-K       : {TOP_K}\n"
        f"Keep-alive  : {OLLAMA_KEEP_ALIVE}\n"
        f"num_predict : {NUM_PREDICT}  num_ctx: {NUM_CTX}\n"
    )
