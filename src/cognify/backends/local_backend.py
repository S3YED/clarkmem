"""
Local/agent backend: ChromaDB (vectors) + networkx (typed graph).

Fully self-contained, no external DB, no torch — embeddings come from ChromaDB's
bundled ONNX all-MiniLM (same 384d space as the server backend). Ideal to drop
into an isolated agent box.
"""
from __future__ import annotations

import json
import re
import threading
import time

import numpy as np

from .. import config


def _safe(t: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", t)


def _entity_id(tenant: str, name: str, etype: str) -> str:
    return f"{tenant}::{name.lower().strip()}::{etype}"


def _orphan_entities(g) -> list:
    """Entity nodes with no remaining MENTIONED_IN edge to any chunk."""
    return [n for n, d in g.nodes(data=True) if d.get("kind") == "entity"
            and not any(g.nodes[t].get("kind") == "chunk" for t in g.successors(n))]


def _upsert_rel(g, sid: str, oid: str, pred: str, doc_id: str, ts: float, functional: set):
    """Typed edge keyed by predicate; repeat assertions bump evidence and revive
    an invalidated fact. Functional predicates close the subject's other current
    objects for that predicate (bi-temporal, Graphiti-style but deterministic)."""
    d = g.get_edge_data(sid, oid, key=pred)
    if d is not None:
        d.update(updated_at=ts, doc_id=doc_id,
                 evidence=d.get("evidence", 1) + 1, invalid_at=None)
        return
    if pred in functional:
        for _, tgt, dd in g.out_edges(sid, data=True):
            if (dd.get("rel") == "REL" and dd.get("type") == pred
                    and tgt != oid and not dd.get("invalid_at")):
                dd["invalid_at"] = ts
    g.add_edge(sid, oid, key=pred, rel="REL", type=pred, doc_id=doc_id,
               observed_at=ts, updated_at=ts, evidence=1, invalid_at=None)


class LocalBackend:
    def __init__(self):
        import chromadb
        from chromadb.utils import embedding_functions
        self.root = config.DATA_DIR / "local"
        self.root.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.root / "chroma"))
        self._ef = embedding_functions.ONNXMiniLM_L6_V2()
        self._lock = threading.Lock()
        self._graphs = {}

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, config.EMBED_DIM), dtype=np.float32)
        return np.asarray(self._ef(texts), dtype=np.float32)

    def _collection(self, tenant):
        return self._client.get_or_create_collection(
            name=f"cognify_{_safe(tenant)}", metadata={"hnsw:space": "cosine"})

    def _graph_path(self, tenant):
        return self.root / f"graph-{_safe(tenant)}.json"

    def _load_graph(self, tenant):
        import networkx as nx
        if tenant in self._graphs:
            return self._graphs[tenant]
        p = self._graph_path(tenant)
        if p.exists():
            g = nx.node_link_graph(json.loads(p.read_text()), directed=True, edges="links")
            if not g.is_multigraph():
                # pre-1.0 stores held ONE edge per node pair, silently dropping a
                # second relation type between the same entities; upgrade in place
                g = nx.MultiDiGraph(g)
        else:
            g = nx.MultiDiGraph()
        self._graphs[tenant] = g
        return g

    def _save_graph(self, tenant):
        import networkx as nx
        self._graph_path(tenant).write_text(
            json.dumps(nx.node_link_data(self._graphs[tenant], edges="links")))

    def load_document(self, doc, *, tenant, namespace, agent, chunk_vecs, extractions):
        ts = time.time()
        functional = config.functional_predicates()
        with self._lock:
            col = self._collection(tenant)
            try:
                col.delete(where={"doc_id": doc.id})
            except Exception:
                pass
            col.add(
                ids=[c.id for c in doc.chunks],
                embeddings=[v.tolist() for v in np.asarray(chunk_vecs, dtype=np.float32)],
                documents=[c.text[:2000] for c in doc.chunks],
                metadatas=[{"doc_id": c.doc_id, "ord": c.ord, "heading": c.heading or "",
                            "tenant": tenant, "namespace": namespace, "title": doc.title}
                           for c in doc.chunks],
            )
            g = self._load_graph(tenant)
            # idempotent re-ingest: drop graph chunks from a previous, longer
            # version of this doc (vectors were already replaced above), then
            # prune entities that lose their last mention (same as neo4j backend)
            keep = {f"chunk::{c.id}" for c in doc.chunks}
            stale = [n for n, d in g.nodes(data=True) if d.get("kind") == "chunk"
                     and d.get("doc_id") == doc.id and n not in keep]
            g.remove_nodes_from(stale)
            g.add_node(f"doc::{doc.id}", kind="document", title=doc.title, source=doc.source)
            for c in doc.chunks:
                g.add_node(f"chunk::{c.id}", kind="chunk", doc_id=c.doc_id, namespace=namespace)
                ex = extractions.get(c.id)
                if not ex:
                    continue
                n2i = {}
                for e in ex.entities:
                    eid = _entity_id(tenant, e.name, e.type)
                    n2i[e.name.lower()] = eid
                    g.add_node(eid, kind="entity", name=e.name, etype=e.type, namespace=namespace)
                    g.add_edge(eid, f"chunk::{c.id}", key="MENTIONED_IN", rel="MENTIONED_IN")
                for r in ex.relations:
                    sid, oid = n2i.get(r.subject.lower()), n2i.get(r.object.lower())
                    if sid and oid and sid != oid:
                        _upsert_rel(g, sid, oid, r.predicate, doc.id, ts, functional)
            if stale:
                g.remove_nodes_from(_orphan_entities(g))
            self._save_graph(tenant)

    def search(self, qvec, *, tenant, namespace, k):
        # the lock also guards readers: ingest mutates the cached graph/collection
        # in place, and the HTTP server calls this from a thread pool
        with self._lock:
            col = self._collection(tenant)
            try:
                res = col.query(query_embeddings=[np.asarray(qvec, dtype=np.float32)[0].tolist()],
                                n_results=k, where={"namespace": namespace} if namespace else None)
            except Exception:
                return []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out = []
        for i, cid in enumerate(ids):
            m = metas[i] or {}
            out.append({"id": cid, "text": docs[i] if i < len(docs) else "",
                        "heading": m.get("heading", ""), "doc_id": m.get("doc_id", ""),
                        "title": m.get("title", ""), "namespace": m.get("namespace", ""),
                        "score": round(1.0 - float(dists[i]), 4) if i < len(dists) else 0.0})
        return out

    def expand(self, chunk_ids, *, tenant, hops, include_invalidated=False):
        with self._lock:
            g = self._load_graph(tenant)
            ent_ids = set()
            for cid in chunk_ids:
                cn = f"chunk::{cid}"
                if g.has_node(cn):
                    ent_ids.update(p for p in g.predecessors(cn)
                                   if g.nodes[p].get("kind") == "entity")
            # BFS over REL edges: hop 1 = relations out of the hit entities, each
            # further hop follows the objects discovered on the previous hop.
            # Invalidated facts neither surface nor carry the traversal onward.
            visited, frontier, relations = set(ent_ids), set(ent_ids), []
            for _ in range(max(1, min(hops, 3))):
                nxt = set()
                for e in frontier:
                    for _, tgt, d in g.out_edges(e, data=True):
                        if d.get("rel") != "REL":
                            continue
                        if d.get("invalid_at") and not include_invalidated:
                            continue
                        rel = {"subject": g.nodes[e].get("name"),
                               "predicate": d.get("type"),
                               "object": g.nodes[tgt].get("name"),
                               "evidence": d.get("evidence", 1)}
                        if d.get("invalid_at"):
                            rel["invalid_at"] = d["invalid_at"]
                        relations.append(rel)
                        if tgt not in visited:
                            nxt.add(tgt)
                visited |= nxt
                frontier = nxt
                if not frontier or len(relations) >= 200:
                    break
            # hop-1 keeps the classic contract: entities = the chunks' own mentions;
            # deeper hops also surface the entities discovered along the paths.
            shown = ent_ids | (visited - ent_ids if max(1, min(hops, 3)) > 1 else set())
            entities = [{"id": e, "name": g.nodes[e].get("name"), "etype": g.nodes[e].get("etype")}
                        for e in shown if g.nodes[e].get("kind") == "entity"][:100]
            return {"entities": entities, "relations": relations[:200]}

    def anchor_chunks(self, query, *, tenant, namespace, limit):
        """Chunks that mention entities literally named in the query — the graph
        as a retrieval signal (HippoRAG-style anchoring, no LLM, no new deps).
        Longest entity names match first: they are the most specific."""
        q = " " + " ".join(re.sub(r"[^\w\s.-]", " ", query.lower()).split()) + " "
        with self._lock:
            g = self._load_graph(tenant)
            ents = [(n, d.get("name", "")) for n, d in g.nodes(data=True)
                    if d.get("kind") == "entity" and len(d.get("name", "")) >= 3
                    and f" {d.get('name', '').lower()} " in q]
            ents.sort(key=lambda nd: -len(nd[1]))
            hits, seen = [], set()
            for eid, name in ents[:8]:
                for _, cn, d in g.out_edges(eid, data=True):
                    if d.get("rel") != "MENTIONED_IN" or cn in seen:
                        continue
                    cd = g.nodes[cn]
                    if cd.get("kind") != "chunk":
                        continue
                    if namespace and cd.get("namespace") != namespace:
                        continue
                    seen.add(cn)
                    hits.append((cn[len("chunk::"):], name, cd))
                    if len(hits) >= limit:
                        break
                if len(hits) >= limit:
                    break
            if not hits:
                return []
            col = self._collection(tenant)
            got = col.get(ids=[cid for cid, *_ in hits], include=["documents", "metadatas"])
        by_id = {i: (t, m or {}) for i, t, m in
                 zip(got.get("ids", []), got.get("documents", []), got.get("metadatas", []))}
        out = []
        for cid, anchor, cd in hits:
            text, m = by_id.get(cid, ("", {}))
            out.append({"id": cid, "text": text or "", "heading": m.get("heading", ""),
                        "doc_id": m.get("doc_id", cd.get("doc_id", "")),
                        "title": m.get("title", ""), "namespace": m.get("namespace", ""),
                        "score": 0.0, "anchor": anchor})
        return out

    def invalidate_relations(self, subject, *, tenant, predicate=None, object=None):
        """Close matching CURRENT facts (set invalid_at; kept for history)."""
        ts, n = time.time(), 0
        subj = subject.lower().strip()
        pred = predicate.upper().strip() if predicate else None
        obj = object.lower().strip() if object else None
        with self._lock:
            g = self._load_graph(tenant)
            for u, v, d in g.edges(data=True):
                if d.get("rel") != "REL" or d.get("invalid_at"):
                    continue
                if g.nodes[u].get("name", "").lower() != subj:
                    continue
                if pred and d.get("type") != pred:
                    continue
                if obj and g.nodes[v].get("name", "").lower() != obj:
                    continue
                d["invalid_at"] = ts
                n += 1
            if n:
                self._save_graph(tenant)
        return n

    def maintain(self, *, tenant):
        """Integrity pass: drop chunks whose document is gone, prune orphan
        entities, reconcile vectors with graph chunks. Safe to run anytime."""
        with self._lock:
            g = self._load_graph(tenant)
            col = self._collection(tenant)
            docs = {n[len("doc::"):] for n, d in g.nodes(data=True)
                    if d.get("kind") == "document"}
            dangling = [n for n, d in g.nodes(data=True)
                        if d.get("kind") == "chunk" and d.get("doc_id") not in docs]
            g.remove_nodes_from(dangling)
            orphans = _orphan_entities(g)
            g.remove_nodes_from(orphans)
            chunk_ids = {n[len("chunk::"):] for n, d in g.nodes(data=True)
                         if d.get("kind") == "chunk"}
            stored = set((col.get(include=[]) or {}).get("ids", []))
            extra = sorted(stored - chunk_ids)
            if extra:
                col.delete(ids=extra)
            self._save_graph(tenant)
        return {"tenant": tenant, "dangling_chunks_removed": len(dangling),
                "entities_pruned": len(orphans), "vectors_removed": len(extra),
                "vectors_missing": len(chunk_ids - stored)}

    def delete_document(self, doc_id, *, tenant):
        """Remove a document, its chunks/vectors, and any entities left with no
        remaining MENTIONED_IN edge (their REL edges go with them)."""
        with self._lock:
            col = self._collection(tenant)
            try:
                col.delete(where={"doc_id": doc_id})
            except Exception:
                pass
            g = self._load_graph(tenant)
            chunks = [n for n, d in g.nodes(data=True)
                      if d.get("kind") == "chunk" and d.get("doc_id") == doc_id]
            g.remove_nodes_from(chunks)
            if g.has_node(f"doc::{doc_id}"):
                g.remove_node(f"doc::{doc_id}")
            orphans = _orphan_entities(g)
            g.remove_nodes_from(orphans)
            self._save_graph(tenant)
            return {"doc_id": doc_id, "chunks_deleted": len(chunks), "entities_pruned": len(orphans)}

    def stats(self, *, tenant, namespace=None):
        """Counts for a tenant. With namespace: documents/chunks are filtered
        (entities/relations stay tenant-wide — they merge across namespaces).
        Without a tenant: global counts across every tenant on this box (same
        meaning as the neo4j backend, so /stats behaves identically)."""
        if not tenant:
            return self._stats_all()
        kinds = {"document": 0, "chunk": 0, "entity": 0}
        rels, ns_docs = 0, set()
        with self._lock:
            g = self._load_graph(tenant)
            col = self._collection(tenant)
            for _, d in g.nodes(data=True):
                k = d.get("kind")
                if k not in kinds:
                    continue
                if namespace and k == "chunk":
                    if d.get("namespace") == namespace:
                        kinds["chunk"] += 1
                        ns_docs.add(d.get("doc_id"))
                elif not namespace or k == "entity":
                    kinds[k] += 1
            rels = sum(1 for *_e, d in g.edges(data=True) if d.get("rel") == "REL")
            vectors = col.count()
        docs = len(ns_docs) if namespace else kinds["document"]
        out = {"documents": docs, "chunks": kinds["chunk"], "entities": kinds["entity"],
               "relations": rels, "vectors": vectors}
        if namespace:
            out["namespace"] = namespace
        return out

    def _stats_all(self):
        """Global counts: read every tenant's persisted graph + collection."""
        out = {"documents": 0, "chunks": 0, "entities": 0, "relations": 0, "vectors": 0}
        with self._lock:
            for p in sorted(self.root.glob("graph-*.json")):
                try:
                    data = json.loads(p.read_text())
                except Exception:
                    continue
                for n in data.get("nodes", []):
                    k = n.get("kind")
                    if k == "document":
                        out["documents"] += 1
                    elif k == "chunk":
                        out["chunks"] += 1
                    elif k == "entity":
                        out["entities"] += 1
                out["relations"] += sum(1 for e in data.get("links", []) if e.get("rel") == "REL")
            for col in self._client.list_collections():  # objects pre-0.6, names after
                name = getattr(col, "name", col)
                if str(name).startswith("cognify_"):
                    c = col if hasattr(col, "count") else self._client.get_collection(str(name))
                    out["vectors"] += c.count()
        return out

    def close(self):
        pass
