# Architecture

## The idea

Plain RAG loses structure: it can find a relevant chunk but not how facts connect.
ClarkMem keeps both layers. Every document becomes (a) embedded chunks for fuzzy
recall and (b) a typed graph (entities + relations) for precise, multi-hop
traversal. Retrieval starts in the vector layer and expands into the graph.

## Data model

```
Document ──PART_OF◄── Chunk ◄──MENTIONED_IN── Entity ──REL{type}──► Entity
```

- **Document**: one source file/string. Files: id = sha256(source + first 512
  chars)[:16], so edits update the same doc in place. Inline text: id =
  sha256(full text)[:16], so distinct notes sharing a preamble never collide.
- **Chunk**: heading-aware ~512-token slice. id = `{doc_id}_{ord}`. Carries the
  embedding.
- **Entity**: canonical typed node. id = `{tenant}::{name.lower}::{type}` so the
  same entity merges across documents but never across tenants.
- **REL**: a typed directed edge (`USES`, `BUILT`, ...) extracted by the LLM,
  grounded — both endpoints must be entities found in the same chunk.

Every node carries `tenant` + `namespace`.

## Flow

### Ingest (Extract → Cognize → Load)
1. **Extract** (`loader.py`): read md/txt/pdf or raw text, strip frontmatter, split
   on markdown headings, window long segments to ~2048 chars with 256 overlap,
   snapping to sentence boundaries.
2. **Cognize** (`extractor.py`): one cheap LLM call per chunk returns strict JSON
   of typed entities + relations. Output is validated: unknown types collapse to
   `Concept`, relations whose endpoints aren't extracted entities are dropped,
   everything deduped. Transport errors raise; bad JSON returns empty.
3. **Load** (`backends/*`): embed chunks to 384d; write vectors to the vector store
   and the Document/Chunk/Entity nodes + edges to the graph store. Idempotent:
   re-ingesting a doc removes its old chunks first (ids are content-derived).

### Recall (hybrid)
1. Embed the query (same model/space as ingest).
2. Vector search within the tenant (and namespace if given), dedup by chunk id,
   fused with entity-anchored chunks (see Hybrid recall below).
3. Expand: from the hit chunks, fetch mentioned entities and their typed
   relations (1 hop) from the graph.
4. Return chunks + the entity/relation subgraph. An LLM can compose a final
   answer over this; ClarkMem returns the structured evidence.

## Why two backends

The pipeline is backend-agnostic (`core.py` depends only on the `Backend`
protocol). The embedder is pluggable: if a backend exposes `embed_texts`, core
uses it. That lets the **local** backend embed with ChromaDB's bundled ONNX
MiniLM (no torch, no downloads, fits a small box) while the **neo4j** backend uses
`sentence-transformers` — both produce the same 384d normalized vectors, so a
graph built on one is queryable by the other.

- **local**: ChromaDB persistent collection per tenant + a networkx DiGraph
  persisted as JSON. Zero external services. For agent boxes and laptops.
- **neo4j**: TurboVec `IdMapIndex` per tenant on disk + Neo4j with C-prefixed
  labels (so it coexists with other graphs in the same database). Document and
  chunk nodes are keyed `(id, tenant)` — ids are content-derived, so two tenants
  ingesting the same document must get separate nodes. For a shared fleet/server
  graph at scale.

## Boundaries & failure modes

- LLM extraction is the cost and latency driver (one call per chunk). Use
  `--no-extract` for a vectors-only pass; use `--cache` so re-ingest skips
  unchanged files.
- Graph traversal adds latency vs pure vector lookup; recall expands 1 hop by
  default.
- The local graph is loaded into memory per tenant; fine for box-scale corpora,
  not millions of nodes (use the neo4j backend there).

## Temporal facts (1.0)

Every `REL` edge carries `observed_at`, `updated_at`, `evidence` (how many
ingests asserted it) and optionally `invalid_at`. Re-assertion bumps evidence
and revives a closed fact. `invalidate()` closes facts by name (optionally
narrowed by predicate/object). Functional predicates (env) auto-close a
subject's older objects. Recall and multi-hop traversal skip invalidated edges
unless `include_invalidated` — history is preserved, never rewritten.

## Hybrid recall (1.0)

Vector top-k and entity-anchored chunks (entities literally named in the query,
longest names first) are fused with reciprocal-rank fusion. Rank-only fusion
sidesteps score-scale mismatches; the graph acts as a retrieval signal, not
just decoration. `maintain()` reconciles graph ↔ vectors per tenant.
