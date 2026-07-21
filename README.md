# ClarkMem

[![ci](https://github.com/S3YED/clarkmem/actions/workflows/ci.yml/badge.svg)](https://github.com/S3YED/clarkmem/actions/workflows/ci.yml)

**The persistent memory engine behind [Clark](https://getclark.app).**
Give any AI agent durable, structured, *temporal* memory: drop in documents,
notes, or conversation facts — get back a typed knowledge graph with hybrid
recall that knows not just **what** is true, but **how strongly evidenced** it
is and **whether it still holds**.

Proprietary software by Weblyfe · formerly published as "Cognify" (≤0.5.0, MIT).

## Why agents need this

Vector RAG finds similar text. Agents need more: *who works where now*, *what
replaced what*, *how facts connect across documents ingested weeks apart*.
ClarkMem keeps three layers in one engine:

1. **Chunks + embeddings** — fuzzy recall over everything ingested.
2. **A typed knowledge graph** — entities (`Person`, `Project`, `Technology`, …)
   and typed relations (`WORKS_AT`, `USES`, `DEPENDS_ON`, …) extracted by a
   cheap LLM at write time, merged across documents.
3. **Time** — every fact carries `observed_at`, an **evidence count** that grows
   as more sources assert it, and an `invalid_at` when the world moves on.
   Closed facts stay queryable as history; recall hides them by default.

## What's inside

| Capability | How |
|---|---|
| Temporal facts | evidence counting, explicit `invalidate`, optional one-current-object predicates (`CLARKMEM_FUNCTIONAL_PREDICATES`), history preserved |
| Hybrid recall | vector search + **entity-anchored retrieval** (chunks mentioning entities named in the query), fused by reciprocal-rank fusion — no LLM call at recall time |
| Multi-hop graph expansion | `hops=1..3` over typed relations, invalidated edges break the chain |
| Multi-tenant + namespaces | every node keyed by tenant; namespaces partition a tenant (docs / memory / transcripts / …) |
| Stable note identity | `key=` gives an evolving note update-in-place semantics; inline text is otherwise content-addressed |
| Self-maintenance | `maintain` — integrity pass: dangling chunks, orphan entities, vector/graph reconciliation |
| Two backends, one API | `local` = ChromaDB + networkx, zero external services, torch-free · `neo4j` = TurboVec + Neo4j for a shared fleet graph |
| Claude-native | MCP server (`clarkmem-mcp`) → tools in Claude Code / Claude Desktop |
| Runtime-agnostic | HTTP API (`clarkmem-serve`) for Hermes, n8n, cron, curl — plus a CLI |
| Agent skills included | `skills/` — drop-in SKILL.md files that teach agents to *use* and to *improve* ClarkMem |

Same 384d embedding space on both backends, so a graph built locally is
queryable on the server and vice versa.

## Core technology

- **Python 3.10+** — ~2k lines, src layout, frozen dataclasses, one file per concern
- **ChromaDB** (embedded) with its bundled **ONNX MiniLM** — local vectors, no torch, no services
- **networkx** MultiDiGraph — local typed graph, persisted as JSON per tenant
- **Neo4j 5** — fleet-scale property graph (C-prefixed labels, tenant-keyed nodes)
- **TurboVec** — per-tenant server-side vector index
- **fastembed / sentence-transformers** — one L2-normalized 384d space everywhere (`all-MiniLM-L6-v2`)
- **FastAPI + Uvicorn** — the HTTP surface (multi-bind, constant-time API-key auth)
- **MCP SDK** (FastMCP) — native tools for Claude Code / Claude Desktop
- **pypdf** for PDF ingestion; **numpy** and **requests** are the only hard dependencies
- **Extraction LLM** — any OpenAI-compatible endpoint (OpenRouter, OpenAI, vLLM, Ollama) or
  native Anthropic Claude; one cheap call per chunk at write time, **zero LLM calls at recall**

## 60-second start

```bash
./setup.sh local            # venv + deps + .env  (zero external services)
source .venv/bin/activate
echo 'OPENROUTER_API_KEY=sk-or-...' >> .env      # or ANTHROPIC_API_KEY
set -a && . ./.env && set +a

clarkmem ingest examples/sample_docs/acme.md --tenant demo
clarkmem recall "what does Pathfinder run on and who owns it?" --tenant demo
clarkmem stats --tenant demo
```

Python:

```python
import clarkmem
be = clarkmem.get_backend("local")
clarkmem.ingest(be, "handbook.pdf", tenant="acme", namespace="hr")
clarkmem.ingest(be, "Sam now leads support.", tenant="acme", key="sam-role")  # evolving note
res = clarkmem.recall(be, "who leads support?", tenant="acme")
print(res.chunks, res.relations)                  # facts carry evidence counts
clarkmem.invalidate(be, "Sam", predicate="LEADS", tenant="acme")  # world changed
```

## Recall pipeline

```
remember ->  Extract   file/text -> heading-aware ~512-token chunks
             Cognize   per chunk, cheap LLM -> typed entities + relations
             Load      embed (384d) -> vectors ; graph merge (evidence++, temporal)
recall   ->  vector top-k  +  entity-anchored chunks   ->  RRF fuse
             -> expand typed graph around hits (1..3 hops, live facts only)
             -> chunks + entities + relations, ready to ground an answer
maintain ->  integrity pass: dangling chunks, orphan entities, vector drift
```

## Use with Claude

```bash
pip install 'clarkmem[local,claude]'
claude mcp add clarkmem -- clarkmem-mcp
```

Claude gets `clarkmem_ingest`, `clarkmem_recall`, `clarkmem_invalidate`,
`clarkmem_forget`, `clarkmem_stats`. Claude can also BE the extractor — set
`ANTHROPIC_API_KEY` and it's auto-detected. Drop `skills/clarkmem/` into
`~/.claude/skills/` to teach agents the memory workflow (recall-first,
remember-with-key, invalidate-on-change).

## Use with Hermes (and any runtime)

The CLI works as-is for shell-native agents; drop `skills/clarkmem/SKILL.md`
into the agent's skills directory. For a shared long-running graph, run the
server:

```bash
pip install 'clarkmem[neo4j,serve,fastembed]'
export CLARKMEM_BACKEND=neo4j CLARKMEM_EMBED_PROVIDER=fastembed
export CLARKMEM_HOST=127.0.0.1,100.x.y.z CLARKMEM_API_KEY=$(openssl rand -hex 24)
clarkmem-serve   # /health open; everything else needs x-api-key
```

```bash
curl -s localhost:8799/ingest  -H 'content-type: application/json' \
  -d '{"text":"the refund policy is 14 days","title":"policy","tenant":"acme"}'
curl -s localhost:8799/recall  -H 'content-type: application/json' \
  -d '{"query":"refund policy?","tenant":"acme"}'
curl -s localhost:8799/invalidate -H 'content-type: application/json' \
  -d '{"subject":"Acme","predicate":"OFFERS","tenant":"acme"}'
```

Server-side `{"path": ...}` ingestion is **off by default**; allow it for one
directory with `CLARKMEM_INGEST_ROOT`. Text bodies are capped by
`CLARKMEM_MAX_TEXT`. `k`/`hops` are clamped engine-wide.

## How it compares

| | **ClarkMem** | Cognee | Mem0 | Graphiti / Zep | LightRAG | plain RAG |
|---|---|---|---|---|---|---|
| Typed entity+relation graph | ✅ | ✅ | partial | ✅ | ✅ | ❌ |
| Temporal facts (evidence, invalidation, history) | ✅ deterministic | ❌ | partial | ✅ (LLM-driven) | ❌ | ❌ |
| Graph as a retrieval signal (anchored hybrid recall) | ✅ | ⚠️ | ❌ | ✅ | ✅ | ❌ |
| Zero external services mode | ✅ | ❌ | ❌ | ❌ | ⚠️ | ✅ |
| Torch-free local install | ✅ | ❌ | ❌ | ❌ | ❌ | varies |
| Same API, swap local ↔ server | ✅ | ⚠️ | ❌ | ❌ | ❌ | n/a |
| Built-in multi-tenancy | ✅ | ⚠️ | ✅ | ✅ | ❌ | ❌ |
| MCP server for Claude | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Agent skills shipped in-repo | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Core small enough to read in a sitting | ✅ ~2k lines | ❌ | ❌ | ❌ | ⚠️ | ✅ |

**Honest edges elsewhere:** Zep/Graphiti does LLM-driven contradiction
detection and SOC2/HIPAA compliance; Cognee has more managed connectors; Mem0
is the quickest hosted drop-in for pure conversational personalization.
**ClarkMem wins when you want a real temporal knowledge graph an agent can run
anywhere** — a laptop, an isolated client box, or a shared fleet server — with
one dependency-light install and one API across all of them.

## Extraction models

One cheap LLM call per ~2KB chunk at ingest (never at recall). Recommended:
`openai/gpt-4o-mini` via OpenRouter (~$0.20 per 40-doc KB), `gemini-2.0-flash`,
a local Qwen/Llama via Ollama/vLLM (free, point `CLARKMEM_LLM_BASE` at it), or
Claude Haiku (`ANTHROPIC_API_KEY`, auto-detected — best quality of the cheap
tier). Avoid `:free` OpenRouter variants for bulk (heavy 429s).

## Configuration

All via env — see `.env.example`. Primary prefix `CLARKMEM_*`; legacy
`COGNIFY_*` names from the pre-rename era keep working. Key ones:
`CLARKMEM_BACKEND`, `CLARKMEM_DATA_DIR`, `CLARKMEM_LLM_BASE/MODEL/KEY`,
`CLARKMEM_EMBED_PROVIDER` (`st`|`fastembed`), `CLARKMEM_EXTRACT_WORKERS`,
`CLARKMEM_HOST/PORT/API_KEY`, `CLARKMEM_INGEST_ROOT`, `CLARKMEM_MAX_TEXT`,
`CLARKMEM_FUNCTIONAL_PREDICATES`, `NEO4J_URI/USER/PASSWORD`.

## For agents working on this repo

`CLAUDE.md` is the operating guide, `ARCHITECTURE.md` the design,
`BLUEPRINT.md` a from-scratch reconstruction spec, and
`skills/clarkmem-dev/` the contributor skill (invariants, test harnesses,
roadmap). Hand an agent this repo and it can use it, extend it, or rebuild it.

## Migrating from Cognify

The GitHub URL redirects. `pip install clarkmem` replaces `cognify-kg`
(never published to PyPI, so no package migration). Legacy `cognify`,
`cognify-serve`, `cognify-mcp` commands and `COGNIFY_*` env vars still work.
On-disk stores are read in place — including `~/.cognify` — no data migration.

## License

**Proprietary — © 2026 Weblyfe.** Source available for evaluation and security
review; production use requires a license (see `LICENSE`). Versions ≤0.5.0
were released under MIT as "Cognify" and remain MIT.
