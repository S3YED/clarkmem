---
name: clarkmem
description: Use ClarkMem — persistent temporal knowledge-graph memory — whenever you need to remember facts across sessions, recall what is known about a person/project/tool, record that something changed, or build a knowledge base from documents. Triggers - remember this, what do we know about, memory, knowledge base, who works on, did anything change.
---

# ClarkMem — the agent memory workflow

ClarkMem is your durable memory: ingest text/files → typed temporal knowledge
graph → hybrid recall (chunks + entities + relations with evidence counts).
Works identically over CLI, HTTP, and MCP.

## Setup (pick ONE transport)

**CLI (local store, zero services):**
```bash
pip install 'clarkmem[local]'   # + set ANTHROPIC_API_KEY or OPENROUTER_API_KEY
```

**HTTP (shared graph, e.g. a fleet server):** nothing to install — curl the
server. Ask your operator for the base URL and `x-api-key`.

**MCP (Claude Code / Desktop):**
```bash
claude mcp add clarkmem -- clarkmem-mcp
```
Tools: `clarkmem_ingest`, `clarkmem_recall`, `clarkmem_invalidate`,
`clarkmem_forget`, `clarkmem_stats`.

## The workflow (this is the important part)

1. **Recall first.** Before answering anything that may touch stored knowledge,
   recall — do not guess from your own context:
   ```bash
   clarkmem recall "who owns onboarding and what tool do they use?" --tenant <T>
   # HTTP: POST /recall {"query":"...","tenant":"<T>","k":8}
   ```
   Use the returned `relations` (they carry evidence counts) as ground truth;
   quote `chunks` for detail. If the API is unreachable, SAY SO — never invent
   graph results.

2. **Remember sparingly, structurally.** Store durable facts, decisions, and
   documents — not logs or scratch: 
   ```bash
   echo "Acme moved its dashboard to Vercel on 2026-07-01" | clarkmem ingest - --tenant <T> --namespace decisions
   ```
   For an **evolving** note (a status, a config, a person's role) pass a stable
   `--key` so re-ingesting updates in place instead of duplicating:
   ```bash
   echo "Sam leads support since July" | clarkmem ingest - --tenant <T> --key sam-role
   ```

3. **Invalidate when the world changes.** Never leave stale facts current:
   ```bash
   clarkmem invalidate "Sam" --predicate WORKS_AT --tenant <T>
   # HTTP: POST /invalidate {"subject":"Sam","predicate":"WORKS_AT","tenant":"<T>"}
   ```
   The fact stays as history (`--include-invalidated` shows it); recall hides it.

4. **Scope with tenant + namespace.** Always pass your agreed `--tenant`
   (isolation is per tenant — never read another tenant). Use namespaces to
   partition: `docs`, `memory`, `decisions`, `transcripts`. Prefer `k<=8` and a
   namespace filter when you know the domain.

5. **Housekeeping** (occasionally, or via cron):
   ```bash
   clarkmem maintain --tenant <T>     # integrity: orphans, dangling, vector drift
   clarkmem stats --tenant <T>
   ```

## Cost + judgment rules

- Ingest = one cheap LLM call per ~2KB chunk. Fine for notes and documents; do
  NOT dump raw logs or huge exports. `--no-extract` gives vectors-only.
- Re-ingesting identical text is idempotent; edited text without `--key` makes
  a new document.
- `forget <doc_id>` fully deletes a document (chunks, vectors, orphaned
  entities). Use `invalidate` for facts, `forget` for documents.
