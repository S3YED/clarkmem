# Blueprint — rebuild ClarkMem from scratch

This is a complete specification. An agent given only this file can reconstruct
ClarkMem. Build in the order below; each step is independently testable.

## Goal

A library + CLI that ingests documents into a typed knowledge graph and answers
queries with hybrid (vector + graph) retrieval. Two interchangeable backends.
Lightweight, no hardcoded machine specifics, multi-tenant.

## Stack

- Python ≥ 3.10. Package under `src/clarkmem/` (src layout, `pyproject.toml`).
- Embeddings: `all-MiniLM-L6-v2`, 384d, L2-normalized.
  - local backend: ChromaDB's bundled ONNX MiniLM (no torch).
  - server backend: `sentence-transformers`.
- LLM: any OpenAI-compatible `/chat/completions` endpoint. Default
  `openai/gpt-4o-mini` via OpenRouter.
- Vector store: ChromaDB (local) or TurboVec `IdMapIndex` (server).
- Graph store: networkx DiGraph as JSON (local) or Neo4j (server).

## Step 1 — `config.py` (portability layer)

Read everything from env with defaults. Expose: `DATA_DIR`
(`COGNIFY_DATA_DIR`, default `~/.clarkmem`), `EMBED_MODEL`, `EMBED_DIM=384`,
`LLM_BASE/LLM_MODEL/LLM_KEY_ENV`, `llm_key()`, `neo4j_creds()`. On macOS, if
`/opt/homebrew/opt/expat/lib` exists and `DYLD_LIBRARY_PATH` is unset, set it (so
pypdf/chromadb import). No secrets, no hardcoded paths anywhere else in the tree.

## Step 2 — `loader.py` (Extract)

`load(path_or_text, is_path=None, title=None) -> Document`. Read md/txt/pdf
(`pypdf`) or raw text. Strip YAML frontmatter. Split on markdown headings into
segments; window segments > 2048 chars into pieces with 256-char overlap, snapping
to `". "` when past the halfway point. Drop pieces < 40 chars.
- `Document(id, title, source, chunks)`, `Chunk(id, doc_id, ord, heading, text)`,
  both frozen dataclasses.
- `doc_id = sha256(source + "::" + text[:512])[:16]`; `chunk.id = f"{doc_id}_{ord}"`.

## Step 3 — `extractor.py` (ClarkMem)

`extract(text) -> Extraction(entities, relations)` (frozen dataclasses;
`Entity(name,type)`, `Relation(subject,predicate,object)`).
- System prompt: return STRICT JSON `{entities:[{name,type}], relations:[{subject,
  predicate,object}]}`. Entity `type` ∈ {Person, Organization, Project, Product,
  Technology, Location, Concept, Event, Document, Metric}. Predicates UPPER_SNAKE
  verbs. Only relations whose subject AND object are in entities.
- POST to `{LLM_BASE}/chat/completions`, `temperature=0`,
  `response_format={"type":"json_object"}`; on HTTP 400 retry without that field.
- Validate: extract first `{...}`, json.loads, unknown types → `Concept`, drop
  ungrounded/duplicate relations. Empty Extraction for text < 40 chars or bad JSON.
  Raise on transport errors (caller degrades).

## Step 4 — `core.py` (orchestration)

- `embed(texts)` via a lazy singleton (384d, L2-normalized). Provider from
  `COGNIFY_EMBED_PROVIDER`: `st` (sentence-transformers, default) or `fastembed`
  (ONNX, torch-free) — identical output space.
- `_embed_texts(backend, texts)`: use `backend.embed_texts` if present, else
  `embed`.
- `Backend` Protocol: `load_document(doc,*,tenant,namespace,agent,chunk_vecs,
  extractions)`, `search(qvec,*,tenant,namespace,k)->list[dict]`,
  `expand(chunk_ids,*,tenant,hops)->{entities,relations}`, `stats(*,tenant)`.
- `ingest(backend, path_or_text, *, tenant, namespace, agent, is_path, title,
  do_extract=True, workers=None)`: load → embed chunks → per-chunk extract
  (catch+log failures; thread pool when `workers or COGNIFY_EXTRACT_WORKERS` > 1)
  → `backend.load_document` → `IngestResult`.
- `recall(backend, query, *, tenant, namespace, k=8, hops=1)`: embed query →
  `backend.search` → `backend.expand` on hit chunk ids → `RecallResult(chunks,
  entities, relations)`. `expand` honors `hops` 1..3: hop 1 = relations out of
  the hit chunks' entities; each further hop follows discovered objects
  (entities beyond the seeds are returned only when hops > 1).
- `backend.delete_document(doc_id, *, tenant)`: remove doc + chunks + vectors,
  then prune entities with no remaining MENTIONED_IN edge. Both backends.
- `backend.stats(*, tenant, namespace=None)`: with namespace, documents/chunks
  are filtered; entities/relations stay tenant-wide (they merge across
  namespaces).

## Step 5 — backends (implement the Protocol)

Graph model for both:
```
(Document) <-PART_OF- (Chunk) <-MENTIONED_IN- (Entity) -REL{type}-> (Entity)
```
Entity id = `f"{tenant}::{name.lower()}::{type}"`. Every node carries `tenant` +
`namespace`. Re-ingest is idempotent (delete a doc's prior chunks first).

- **local_backend.py**: ChromaDB `PersistentClient` at `DATA_DIR/local/chroma`, one
  collection `clarkmem_<tenant>` (cosine). `embed_texts` = ChromaDB
  `ONNXMiniLM_L6_V2`. Graph = networkx DiGraph per tenant at
  `DATA_DIR/local/graph-<tenant>.json` (node-link JSON). search via
  `collection.query` (where namespace); expand via predecessors of `chunk::<id>`
  nodes that are entities, then their `REL` out-edges.
- **neo4j_backend.py**: TurboVec `IdMapIndex(dim=384)` per tenant at
  `DATA_DIR/neo4j/<tenant>/index.turbo` + `meta.json` (uint64 id = sha256(chunk.id)
  [:8] little-endian, top bit cleared). C-prefixed labels `CDocument/CChunk/
  CEntity`, edges `PART_OF/MENTIONED_IN/REL{type}`. Create id + tenant indexes.
  Bulk writes via `UNWIND`. search = turbovec search + dedup by text; expand =
  Cypher over `MENTIONED_IN` then `REL`, tenant-filtered.
- `backends/__init__.py`: `get_backend(kind=env COGNIFY_BACKEND or "local")`.

## Step 6 — `cli.py`

`clarkmem {ingest,ingest-dir,recall,forget,stats}` with `--backend/--tenant/
--namespace/--agent`. Both ingest commands take `--workers`; `recall` takes
`--hops`. `ingest-dir` takes `--glob`, `--limit`, `--no-extract`, and `--cache`
(skip files whose sha256 is unchanged since last run; cache at
`DATA_DIR/cache/ingest-<tenant>--<namespace>.json` — namespaced so the same dir
can be ingested into two namespaces). Console script entry in `pyproject.toml`.

## Step 7 — packaging & tests

`pyproject.toml` (src layout, extras `local`/`neo4j`/`all`, `clarkmem` script),
`.env.example`, `setup.sh` (venv + `pip install -e .[extra]` + copy .env),
`tests/test_smoke.py` (chunking, extraction-parse incl. dropping ungrounded
relations, and an LLM-key-gated local e2e).

## Acceptance

```bash
./setup.sh local && source .venv/bin/activate && set -a && . ./.env && set +a
clarkmem ingest examples/sample_docs/clark.md --tenant demo
clarkmem recall "what does Clark use for memory?" --tenant demo
# expect: chunks returned, entities incl. Clark/Neo4j/TurboVec,
#         relations incl. Clark -USES-> Neo4j
pytest -q
```

## Invariants (do not violate)

1. `core.py` never imports a concrete backend.
2. Backends are symmetric — identical method semantics.
3. All machine specifics live in `config.py`; no hardcoded paths/keys.
4. Tenancy enforced on every read and write.
5. Local backend stays torch-free.
6. Frozen dataclasses; pure functions; degrade on extraction failure.
