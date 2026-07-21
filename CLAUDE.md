# ClarkMem — agent operating guide

You are working in **ClarkMem** (formerly Cognify), a document-ingestion +
typed **temporal** knowledge-graph memory engine. Read this before changing
anything. For deeper contributor knowledge (invariants, harnesses, roadmap)
read `skills/clarkmem-dev/SKILL.md`.

## What it is

`ingest(document)` → chunks → cheap-LLM typed entity/relation extraction →
embed + merge into a graph where every fact carries `observed_at` / `evidence`
/ `invalid_at`. `recall(query)` → vector search + entity-anchored chunks →
RRF fuse → expand the live graph around the hits. `invalidate()` closes facts
without erasing history. `maintain()` is the integrity pass. Two backends
(`local` = ChromaDB+networkx, `neo4j` = TurboVec+Neo4j) behind one API in
`core.py`.

## Layout

```
src/clarkmem/
  config.py            env-driven paths/creds/models — the ONLY place machine
                       specifics live. CLARKMEM_* primary, legacy COGNIFY_*
                       honored via _env(). No hardcoded paths or secrets.
  loader.py            file/text -> heading-aware chunks; doc identity
                       (key > file source+head > inline full-text hash)
  extractor.py         LLM -> typed Extraction; OpenAI-compatible or Anthropic
  core.py              ECL orchestration, embedder, RRF fusion, clamps,
                       Backend protocol, invalidate()
  backends/
    local_backend.py   ChromaDB + networkx MultiDiGraph (torch-free)
    neo4j_backend.py   TurboVec + Neo4j (shared/server)
    __init__.py        get_backend() factory
  cli.py               ingest / ingest-dir / recall / invalidate / maintain /
                       forget / stats
  mcp_server.py        MCP tools for Claude Code/Desktop  (clarkmem-mcp)
  server.py            FastAPI HTTP server                (clarkmem-serve)
skills/clarkmem/       drop-in agent skill: HOW to use memory well
skills/clarkmem-dev/   contributor skill: invariants, harnesses, roadmap
examples/  tests/  setup.sh  pyproject.toml  .env.example
```

## Setup & run

```bash
./setup.sh local                      # or: neo4j / all
source .venv/bin/activate && set -a && . ./.env && set +a
clarkmem ingest <path|-> --tenant T
clarkmem recall "<q>" --tenant T      # hybrid; --mode vector to compare
pytest -q                             # e2e skips without an LLM key
```

## Rules of the codebase

- **Immutable data**: dataclasses are `frozen=True`; functions return new values.
- **Backend symmetry**: any behavior added to one backend must exist on the
  other with identical semantics. `core.py` must never import a concrete backend.
- **All config through `config.py`** — never read env or hardcode elsewhere.
- **Degrade, don't crash**: extraction failures fall back to chunks-only; log it.
- **Tenancy is non-negotiable**: every node carries `tenant`; every query
  (including new chunk/doc MATCHes in Neo4j) filters by it.
- **Temporal contract**: contradiction ⇒ `invalid_at`, never hard-delete;
  new evidence revives; recall excludes invalidated facts by default.
- **Data compat**: on-disk names stay (`cognify_<tenant>` collections,
  `CDocument/CChunk/CEntity` labels, `graph-<tenant>.json`); legacy env/CLI
  aliases keep working.
- **Keep it lightweight**: local backend stays torch-free; no heavy deps on the
  default path; no LLM calls at recall time.

## Common extensions (and where)

- New file type → `loader.read_file()`.
- Different LLM/endpoint → env only (`CLARKMEM_LLM_*`), no code change.
- New vector/graph store → new file in `backends/`, register in factory,
  implement the full Backend protocol (see `core.py`) symmetrically.
- Richer retrieval (BM25 lane, PPR expand) → `core.recall()` + backend methods;
  see the roadmap in `skills/clarkmem-dev/SKILL.md`.

If you need to rebuild from scratch, follow `BLUEPRINT.md`.
