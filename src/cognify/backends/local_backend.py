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
        g = (nx.node_link_graph(json.loads(p.read_text()), directed=True, edges="links")
             if p.exists() else nx.DiGraph())
        self._graphs[tenant] = g
        return g

    def _save_graph(self, tenant):
        import networkx as nx
        self._graph_path(tenant).write_text(
            json.dumps(nx.node_link_data(self._graphs[tenant], edges="links")))

    def load_document(self, doc, *, tenant, namespace, agent, chunk_vecs, extractions):
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
                    g.add_edge(eid, f"chunk::{c.id}", rel="MENTIONED_IN")
                for r in ex.relations:
                    sid, oid = n2i.get(r.subject.lower()), n2i.get(r.object.lower())
                    if sid and oid and sid != oid:
                        g.add_edge(sid, oid, rel="REL", type=r.predicate, doc_id=doc.id)
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

    def expand(self, chunk_ids, *, tenant, hops):
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
            visited, frontier, relations = set(ent_ids), set(ent_ids), []
            for _ in range(max(1, min(hops, 3))):
                nxt = set()
                for e in frontier:
                    for _, tgt, d in g.out_edges(e, data=True):
                        if d.get("rel") != "REL":
                            continue
                        relations.append({"subject": g.nodes[e].get("name"),
                                          "predicate": d.get("type"),
                                          "object": g.nodes[tgt].get("name")})
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
