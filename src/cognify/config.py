"""
Central config — the portability layer.

Everything that ties Cognify to a particular machine lives here and is driven by
environment variables with sane defaults, so the same code runs unchanged on a
laptop, a server, or an isolated agent box. No hardcoded paths, no secrets.
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

# --- where data lives -------------------------------------------------------
DATA_DIR = Path(os.environ.get("COGNIFY_DATA_DIR", str(Path.home() / ".cognify"))).expanduser()

# --- embedding model --------------------------------------------------------
EMBED_MODEL = os.environ.get("COGNIFY_EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM = int(os.environ.get("COGNIFY_EMBED_DIM", "384"))
# Provider for the shared embedder: "st" (sentence-transformers) or "fastembed"
# (ONNX, torch-free — fits small server boxes). Both emit the same L2-normalized
# 384d space, so indexes built under one are queryable under the other.
EMBED_PROVIDER = os.environ.get("COGNIFY_EMBED_PROVIDER", "st").lower()

# --- ingest -----------------------------------------------------------------
# Parallel per-chunk extraction (the LLM call is network-bound). 1 = serial.
EXTRACT_WORKERS = max(1, int(os.environ.get("COGNIFY_EXTRACT_WORKERS", "1")))


# --- temporal facts ----------------------------------------------------------
def functional_predicates() -> set[str]:
    """Predicates where a subject holds ONE current object (WORKS_AT, LOCATED_IN,
    ...): asserting a new object auto-invalidates the subject's older facts for
    that predicate. Comma-separated env; empty (default) = never auto-invalidate."""
    raw = os.environ.get("COGNIFY_FUNCTIONAL_PREDICATES", "")
    return {p.strip().upper() for p in raw.split(",") if p.strip()}

# --- LLM extractor ----------------------------------------------------------
# Provider: "openai" (any OpenAI-compatible /chat/completions: OpenRouter, OpenAI,
# vLLM, Ollama) or "anthropic" (Claude messages API, native). Auto-detects Claude
# if ANTHROPIC_API_KEY is set and no provider/OpenRouter key is configured.
def _default_provider() -> str:
    if os.environ.get("COGNIFY_LLM_PROVIDER"):
        return os.environ["COGNIFY_LLM_PROVIDER"].lower()
    if os.environ.get("ANTHROPIC_API_KEY") and not (
        os.environ.get("COGNIFY_LLM_KEY") or os.environ.get("OPENROUTER_API_KEY")
    ):
        return "anthropic"
    return "openai"


LLM_PROVIDER = _default_provider()
LLM_BASE = os.environ.get("COGNIFY_LLM_BASE", "https://openrouter.ai/api/v1").rstrip("/")
LLM_MODEL = os.environ.get("COGNIFY_LLM_MODEL", "openai/gpt-4o-mini")
LLM_KEY_ENV = os.environ.get("COGNIFY_LLM_KEYENV", "OPENROUTER_API_KEY")

# Anthropic (Claude) native
ANTHROPIC_BASE = os.environ.get("COGNIFY_ANTHROPIC_BASE", "https://api.anthropic.com").rstrip("/")
ANTHROPIC_MODEL = os.environ.get("COGNIFY_ANTHROPIC_MODEL", "claude-3-5-haiku-latest")


def llm_key() -> str | None:
    """Resolve the extractor API key. For anthropic: COGNIFY_LLM_KEY or
    ANTHROPIC_API_KEY. For openai-compatible: COGNIFY_LLM_KEY or the named key env
    var (default OPENROUTER_API_KEY)."""
    if os.environ.get("COGNIFY_LLM_KEY"):
        return os.environ["COGNIFY_LLM_KEY"]
    if LLM_PROVIDER == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY")
    return os.environ.get(LLM_KEY_ENV)


# --- HTTP server hardening ---------------------------------------------------
def ingest_root() -> str | None:
    """Directory under which server-side {"path": ...} ingestion is allowed.
    None (unset) = path ingestion over HTTP is disabled. Read per request so
    it can be changed without a code reload (and monkeypatched in tests)."""
    return os.environ.get("COGNIFY_INGEST_ROOT")


def max_text() -> int:
    """Cap on HTTP text bodies, in characters."""
    return int(os.environ.get("COGNIFY_MAX_TEXT", "2000000"))


# --- Neo4j (fleet backend only) --------------------------------------------
def neo4j_creds() -> dict:
    """Read Neo4j creds from env. Optionally seed from a dotenv-style file named
    by COGNIFY_NEO4J_ENV_FILE (env still wins for any key it sets)."""
    creds = {
        "uri": os.environ.get("NEO4J_URI"),
        "user": os.environ.get("NEO4J_USER", "neo4j"),
        "password": os.environ.get("NEO4J_PASSWORD"),
    }
    env_file = os.environ.get("COGNIFY_NEO4J_ENV_FILE")
    if env_file and os.path.exists(env_file):
        import re
        txt = open(env_file).read()
        for key, slot in (("NEO4J_URI", "uri"), ("NEO4J_USER", "user"), ("NEO4J_PASSWORD", "password")):
            if not creds[slot]:  # env wins; file only fills gaps
                m = re.search(rf"^{key}=(.+)$", txt, re.M)
                if m:
                    creds[slot] = m.group(1).strip().strip('"').strip("'")
    return creds
