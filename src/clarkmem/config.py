"""
Central config — the portability layer.

Everything that ties ClarkMem to a particular machine lives here and is driven
by environment variables with sane defaults, so the same code runs unchanged on
a laptop, a server, or an isolated agent box. No hardcoded paths, no secrets.

CLARKMEM_* is the primary prefix. Legacy COGNIFY_* names (pre-rename installs)
are still honored everywhere via _env().
"""
from __future__ import annotations

# pypdf/chromadb -> xml.parsers.expat needs a working expat. On Homebrew macOS the
# stock pyexpat is broken; point DYLD at brew's expat if present and not already set.
import os
import sys
from pathlib import Path

if sys.platform == "darwin" and "DYLD_LIBRARY_PATH" not in os.environ:
    brew_expat = "/opt/homebrew/opt/expat/lib"
    if os.path.isdir(brew_expat):
        os.environ["DYLD_LIBRARY_PATH"] = f"{brew_expat}:/opt/homebrew/lib"


def _env(name: str, default: str | None = None) -> str | None:
    """CLARKMEM_<name> wins; legacy COGNIFY_<name> is still honored."""
    v = os.environ.get(f"CLARKMEM_{name}")
    if v is None:
        v = os.environ.get(f"COGNIFY_{name}")
    return default if v is None else v


# --- where data lives -------------------------------------------------------
def _default_data_dir() -> str:
    new, legacy = Path.home() / ".clarkmem", Path.home() / ".cognify"
    if not new.exists() and legacy.exists():
        return str(legacy)  # keep reading a pre-rename store in place
    return str(new)


DATA_DIR = Path(_env("DATA_DIR", _default_data_dir())).expanduser()

# --- embedding model --------------------------------------------------------
EMBED_MODEL = _env("EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM = int(_env("EMBED_DIM", "384"))
# Provider for the shared embedder: "st" (sentence-transformers) or "fastembed"
# (ONNX, torch-free — fits small server boxes). Both emit the same L2-normalized
# 384d space, so indexes built under one are queryable under the other.
EMBED_PROVIDER = (_env("EMBED_PROVIDER", "st") or "st").lower()


# --- backend selection ------------------------------------------------------
def backend_default() -> str:
    """'local' (ChromaDB+networkx) or 'neo4j' (TurboVec+Neo4j)."""
    return _env("BACKEND", "local") or "local"


# --- ingest -----------------------------------------------------------------
# Parallel per-chunk extraction (the LLM call is network-bound). 1 = serial.
EXTRACT_WORKERS = max(1, int(_env("EXTRACT_WORKERS", "1")))


# --- temporal facts ----------------------------------------------------------
def functional_predicates() -> set[str]:
    """Predicates where a subject holds ONE current object (WORKS_AT, LOCATED_IN,
    ...): asserting a new object auto-invalidates the subject's older facts for
    that predicate. Comma-separated env; empty (default) = never auto-invalidate."""
    raw = _env("FUNCTIONAL_PREDICATES", "") or ""
    return {p.strip().upper() for p in raw.split(",") if p.strip()}


# --- LLM extractor ----------------------------------------------------------
# Provider: "openai" (any OpenAI-compatible /chat/completions: OpenRouter, OpenAI,
# vLLM, Ollama) or "anthropic" (Claude messages API, native). Auto-detects Claude
# if ANTHROPIC_API_KEY is set and no provider/OpenRouter key is configured.
def _default_provider() -> str:
    explicit = _env("LLM_PROVIDER")
    if explicit:
        return explicit.lower()
    if os.environ.get("ANTHROPIC_API_KEY") and not (
        _env("LLM_KEY") or os.environ.get("OPENROUTER_API_KEY")
    ):
        return "anthropic"
    return "openai"


LLM_PROVIDER = _default_provider()
LLM_BASE = (_env("LLM_BASE", "https://openrouter.ai/api/v1") or "").rstrip("/")
LLM_MODEL = _env("LLM_MODEL", "openai/gpt-4o-mini")
LLM_KEY_ENV = _env("LLM_KEYENV", "OPENROUTER_API_KEY")

# Anthropic (Claude) native
ANTHROPIC_BASE = (_env("ANTHROPIC_BASE", "https://api.anthropic.com") or "").rstrip("/")
ANTHROPIC_MODEL = _env("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")


def llm_key() -> str | None:
    """Resolve the extractor API key. For anthropic: CLARKMEM_LLM_KEY or
    ANTHROPIC_API_KEY. For openai-compatible: CLARKMEM_LLM_KEY or the named key
    env var (default OPENROUTER_API_KEY)."""
    if _env("LLM_KEY"):
        return _env("LLM_KEY")
    if LLM_PROVIDER == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY")
    return os.environ.get(LLM_KEY_ENV)


# --- HTTP server (clarkmem-serve) --------------------------------------------
def api_key() -> str | None:
    """x-api-key required on every endpoint except /health when set."""
    return _env("API_KEY")


def serve_hosts() -> list[str]:
    """Comma-separated binds, e.g. '127.0.0.1,100.x.y.z'. Never 0.0.0.0."""
    raw = _env("HOST", "127.0.0.1") or "127.0.0.1"
    return [h.strip() for h in raw.split(",") if h.strip()]


def serve_port() -> int:
    return int(_env("PORT", "8799"))


def ingest_root() -> str | None:
    """Directory under which server-side {"path": ...} ingestion is allowed.
    None (unset) = path ingestion over HTTP is disabled. Read per request so
    it can be changed without a code reload (and monkeypatched in tests)."""
    return _env("INGEST_ROOT")


def max_text() -> int:
    """Cap on HTTP text bodies, in characters."""
    return int(_env("MAX_TEXT", "2000000"))


# --- Neo4j (fleet backend only) --------------------------------------------
def neo4j_creds() -> dict:
    """Read Neo4j creds from env. Optionally seed from a dotenv-style file named
    by CLARKMEM_NEO4J_ENV_FILE (env still wins for any key it sets)."""
    creds = {
        "uri": os.environ.get("NEO4J_URI"),
        "user": os.environ.get("NEO4J_USER", "neo4j"),
        "password": os.environ.get("NEO4J_PASSWORD"),
    }
    env_file = _env("NEO4J_ENV_FILE")
    if env_file and os.path.exists(env_file):
        import re
        txt = open(env_file).read()
        for key, slot in (("NEO4J_URI", "uri"), ("NEO4J_USER", "user"), ("NEO4J_PASSWORD", "password")):
            if not creds[slot]:  # env wins; file only fills gaps
                m = re.search(rf"^{key}=(.+)$", txt, re.M)
                if m:
                    creds[slot] = m.group(1).strip().strip('"').strip("'")
    return creds
