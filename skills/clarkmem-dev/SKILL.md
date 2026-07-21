---
name: clarkmem-dev
description: Use when developing, extending, reviewing, or debugging the ClarkMem repo itself (the memory engine, not memory usage) - adding backends, changing recall, touching the extractor, running the test/verification harnesses, or picking up a roadmap item.
---

# ClarkMem contributor skill

Read `CLAUDE.md` (operating rules) and `ARCHITECTURE.md` (design) before any
change. This skill adds the working knowledge that lives outside those files.

## Map (who calls what)

```
cli.py ─┐
server.py ─┼─> core.py (ingest / recall / invalidate; clamps; RRF fuse; Backend protocol)
mcp_server.py ─┘        │
                        ├─> loader.py     file/text -> chunks; doc identity (key > source+head > full text)
                        ├─> extractor.py  chunk -> typed Extraction (caps hostile output)
                        └─> backends/     local_backend.py | neo4j_backend.py  (symmetric!)
```

## Invariants you must not break

1. **Backend symmetry** — every behavior exists on BOTH backends with identical
   semantics (temporal fields, anchoring, maintain, shrink cleanup, global
   stats). If you add to one, add to the other in the same change.
2. **Tenancy** — every node/query is tenant-scoped. Neo4j doc/chunk nodes are
   keyed `(id, tenant)` (ids are content-derived and NOT tenant-prefixed);
   entity ids ARE tenant-prefixed. Any new chunk/doc MATCH must constrain
   tenant explicitly.
3. **Temporal contract** — never hard-delete a fact on contradiction: set
   `invalid_at`. Fresh evidence revives (`invalid_at=null`, `evidence+=1`).
   Recall/expand must exclude invalidated edges from BOTH results and traversal
   unless `include_invalidated`.
4. **Degrade, don't crash** — extraction failure ⇒ chunks-only ingest; missing
   optional backend methods ⇒ recall still returns vector hits.
5. **Data compat** — on-disk names stay: Chroma collections `cognify_<tenant>`,
   Neo4j labels `CDocument/CChunk/CEntity`, graph files `graph-<tenant>.json`.
   Legacy `COGNIFY_*` env + `cognify*` CLI aliases keep working (config._env).
6. **Bounds** — k≤100, hops≤3 are clamped ONCE in `core.recall`; extractor caps
   entities/relations/name length; server caps body size and gates path ingest.

## Verification (do all three for backend-touching changes)

```bash
.venv-test/bin/ruff check src tests && .venv-test/bin/python -m pytest -q
```

Neo4j paths cannot run in CI. Validate against a REAL Neo4j with the harness
pattern (see git history: `neo4j_harness*.py`): rsync the working tree to the
server, run with `CLARKMEM_DATA_DIR=/tmp/...` (never the prod data dir) and
throwaway tenants (`nb-audit-*`), assert, then delete the tenants and verify
stats are zero. Never write to production tenants.

For live-service checks: `GET /health`, then a recall probe against a real
tenant, read-only.

## Roadmap (validated against 2026 SOTA; pick from the top)

1. **Entity resolution** — alias merging at write time (normalized-name +
   embedding similarity ≥ threshold, optional LLM adjudication). Biggest
   recall-quality win; Mem0/Zep/Cognee all do a version of this.
2. **BM25 keyword lane** — third RRF list alongside vector + anchor (tiny
   inverted index per tenant, or Neo4j fulltext index server-side).
3. **Personalized-PageRank expand** — score expansion by PPR from anchored
   entities (HippoRAG 2) instead of flat BFS.
4. **Consolidation job** — scheduled `maintain` + summarize-and-reingest of
   old namespaces (Letta-style sleep-time compute).
5. **Relation provenance list** — store top-N source doc_ids per relation, not
   just the last writer; delete-doc can then decrement/invalidate cleanly.
6. **Eval harness** — LongMemEval/LoCoMo-style QA set over a seeded corpus, run
   in CI against the local backend; track recall@k and fact-accuracy.
7. **Multi-writer safety** — DB-side uniqueness constraint (or tenant-prefixed
   doc/chunk ids at 2.0) so multiple server processes can share one Neo4j.

## Gotchas

- `local` graph is a MultiDiGraph keyed by predicate; `add_edge` without `key=`
  creates duplicate parallel edges — always pass `key=`.
- Chroma ≥0.6 `list_collections()` returns names, not objects (`_stats_all`
  handles both).
- TurboVec index + meta.json are saved as a pair per tenant; save the index
  ONCE per load_document, after the graph write.
- The FastAPI server runs sync handlers in a thread pool — every local-backend
  read must hold `self._lock` (ingest mutates cached graphs in place).
