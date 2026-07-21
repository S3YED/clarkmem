# Changelog

## 1.0.0 — ClarkMem (2026-07-21)

Renamed **Cognify → ClarkMem** and upgraded to a temporal memory engine.
License changed to proprietary (≤0.5.0 stays MIT). Module `clarkmem`, package
`clarkmem`, CLIs `clarkmem` / `clarkmem-serve` / `clarkmem-mcp` (legacy
`cognify*` aliases and `COGNIFY_*` env vars still work; on-disk stores are read
in place, `~/.cognify` included).

### Added
- **Temporal facts**: relations carry `observed_at` / `updated_at` / `evidence`;
  re-assertion bumps evidence and revives closed facts; `invalidate()` (CLI /
  HTTP `/invalidate` / MCP `clarkmem_invalidate`) closes a fact without erasing
  it; recall hides closed facts unless `include_invalidated`;
  `CLARKMEM_FUNCTIONAL_PREDICATES` auto-closes one-current-object facts.
- **Hybrid recall**: entity-anchored retrieval fused with vector search via
  reciprocal-rank fusion (`mode="vector"` opts out). No LLM at recall time.
- **`maintain`** (CLI / HTTP `/maintain`): per-tenant integrity pass — dangling
  chunks, orphan entities, vector/graph reconciliation (heals pre-0.5 ghosts).
- **`key=` ingest param**: stable update-in-place identity for evolving notes.
- **`skills/`**: drop-in SKILL.md files for agents (usage + contributor).
- Local graph is now a MultiDiGraph — two different typed relations between the
  same entity pair no longer overwrite each other (transparent migration).

### Changed
- Recall `score` is the RRF-fused score in hybrid mode; chunks found by
  anchoring carry an `anchor` field; relations carry `evidence`.

## 0.5.0 (2026-07-21)

Correctness + hardening release (verified against the live fleet graph):
- neo4j: doc/chunk nodes keyed `(id, tenant)` — cross-tenant clobbering fixed.
- Both backends: shrinking re-ingest removes stale chunks, vectors, and
  stranded entities (recallable ghost chunks fixed).
- neo4j: over-fetch `max(k*3, 30)`; dedup by chunk id (prefix-dedup starved k);
  graph written before the vector index is saved.
- Inline docs content-addressed on full text (shared-preamble collisions lost
  data); ids for ≤512-char inline text unchanged.
- Server: path ingestion gated by `INGEST_ROOT` (default off), body-size cap,
  central k/hops clamps, namespace-without-tenant is a 400.
- Local backend: thread-safe reads; global stats parity with neo4j.
- CLI: `--namespace` no longer force-filters recall/stats to `default`.
- Extractor: entity/relation/name caps against hostile model output.

## 0.4.0 and earlier — as "Cognify" (MIT)

Deletion API, real multi-hop expand, namespace stats, extractor retry, CI,
optional API-key auth, fleet/server deployment support.
