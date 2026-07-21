"""
Fleet/server backend: TurboVec (vectors) + Neo4j (typed graph).

Graph model (C-prefixed labels so Cognify never collides with other graphs in the
same Neo4j; every node carries tenant + namespace):

  (:CDocument)-[:PART_OF]<-(:CChunk)  (:CEntity)-[:MENTIONED_IN]->(:CChunk)
  (:CEntity)-[:REL {type}]->(:CEntity)
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from pathlib import Path

import numpy as np

from .. import config

_DOC, _CHUNK, _ENT = "CDocument", "CChunk", "CEntity"


def _safe(t: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", t)


def _cid_int(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "little") & 0x7FFFFFFFFFFFFFFF


def _entity_id(tenant: str, name: str, etype: str) -> str:
    return f"{tenant}::{name.lower().strip()}::{etype}"


class Neo4jBackend:
    def __init__(self):
        import turbovec  # noqa: F401  (validate availability early)
        from neo4j import GraphDatabase
        c = config.neo4j_creds()
        if not c.get("uri") or not c.get("password"):
            raise RuntimeError("Set NEO4J_URI and NEO4J_PASSWORD (see .env.example)")
        self._driver = GraphDatabase.driver(c["uri"], auth=(c["user"], c["password"]))
        self._root = config.DATA_DIR / "neo4j"
        self._lock = threading.Lock()
        self._idx, self._meta = {}, {}
        self._ensure_schema()

    def _ensure_schema(self):
        with self._driver.session() as s:
            for lbl in (_DOC, _CHUNK, _ENT):
                s.run(f"CREATE INDEX {lbl.lower()}_id IF NOT EXISTS FOR (n:{lbl}) ON (n.id)")
            s.run(f"CREATE INDEX {_ENT.lower()}_tenant IF NOT EXISTS FOR (n:{_ENT}) ON (n.tenant)")
            # doc/chunk ids are content-derived and NOT tenant-prefixed, so nodes
            # are keyed (id, tenant) — two tenants ingesting the same document
            # must get separate nodes, never overwrite each other's.
            for lbl in (_DOC, _CHUNK):
                s.run(f"CREATE INDEX {lbl.lower()}_id_tenant IF NOT EXISTS "
                      f"FOR (n:{lbl}) ON (n.id, n.tenant)")

    # -- per-tenant TurboVec index ----------------------------------------
    def _tenant_dir(self, tenant: str) -> Path:
        d = self._root / _safe(tenant)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load_index(self, tenant: str):
        import turbovec
        if tenant in self._idx:
            return self._idx[tenant], self._meta[tenant]
        d = self._tenant_dir(tenant)
        ip, mp = d / "index.turbo", d / "meta.json"
        if ip.exists() and mp.exists():
            idx = turbovec.IdMapIndex.load(str(ip))
            meta = {int(k): v for k, v in json.loads(mp.read_text()).items()}
        else:
            idx, meta = turbovec.IdMapIndex(dim=config.EMBED_DIM), {}
        self._idx[tenant], self._meta[tenant] = idx, meta
        return idx, meta

    def _save_index(self, tenant: str):
        d = self._tenant_dir(tenant)
        self._idx[tenant].write(str(d / "index.turbo"))
        (d / "meta.json").write_text(json.dumps({str(k): v for k, v in self._meta[tenant].items()}))

    # -- load --------------------------------------------------------------
    def load_document(self, doc, *, tenant, namespace, agent, chunk_vecs, extractions):
        with self._lock:
            idx, meta = self._load_index(tenant)
            ids = []
            for c in doc.chunks:
                cint = _cid_int(c.id)
                try:
                    idx.remove(cint)
                except Exception:
                    pass
                ids.append(cint)
                meta[cint] = {"id": c.id, "doc_id": c.doc_id, "ord": c.ord, "heading": c.heading,
                              "text": c.text[:600], "tenant": tenant, "namespace": namespace,
                              "title": doc.title}
            if doc.chunks:
                idx.add_with_ids(np.asarray(chunk_vecs, dtype=np.float32),
                                 np.asarray(ids, dtype=np.uint64))
            # graph first — it also reports stale chunks from a previous, longer
            # version of this doc. The index is saved once, after, so a graph
            # failure leaves the persisted vectors untouched.
            stale = self._write_graph(doc, tenant, namespace, agent, extractions)
            self._drop_vectors(idx, meta, stale)
            self._save_index(tenant)

    @staticmethod
    def _drop_vectors(idx, meta, chunk_ids):
        for cid in chunk_ids:
            cint = _cid_int(cid)
            try:
                idx.remove(cint)
            except Exception:
                pass
            meta.pop(cint, None)

    def _prune_orphans(self, s, tenant) -> int:
        """Delete this tenant's entities with no remaining mention anywhere."""
        return s.run(f"MATCH (e:{_ENT} {{tenant:$t}}) WHERE NOT (e)-[:MENTIONED_IN]->() "
                     "WITH e DETACH DELETE e RETURN count(*) AS n", t=tenant).single()["n"]

    def _write_graph(self, doc, tenant, namespace, agent, extractions) -> list:
        ts = time.time()
        chunk_rows = [{"id": c.id, "doc_id": c.doc_id, "ord": c.ord, "heading": c.heading,
                       "text": c.text[:2000]} for c in doc.chunks]
        ent_rows, ment_rows, rel_rows = [], [], []
        for c in doc.chunks:
            ex = extractions.get(c.id)
            if not ex:
                continue
            n2i = {}
            for e in ex.entities:
                eid = _entity_id(tenant, e.name, e.type)
                n2i[e.name.lower()] = eid
                ent_rows.append({"id": eid, "name": e.name, "etype": e.type})
                ment_rows.append({"eid": eid, "cid": c.id})
            for r in ex.relations:
                sid, oid = n2i.get(r.subject.lower()), n2i.get(r.object.lower())
                if sid and oid and sid != oid:
                    rel_rows.append({"sid": sid, "oid": oid, "type": r.predicate, "doc_id": doc.id})
        with self._driver.session() as s:
            stale = [r["id"] for r in s.run(
                f"MATCH (c:{_CHUNK} {{tenant:$t, doc_id:$d}}) WHERE NOT c.id IN $keep "
                "RETURN c.id AS id", t=tenant, d=doc.id, keep=[c.id for c in doc.chunks])]
            if stale:
                s.run(f"MATCH (c:{_CHUNK} {{tenant:$t}}) WHERE c.id IN $ids DETACH DELETE c",
                      t=tenant, ids=stale)
            s.run(f"MERGE (d:{_DOC} {{id:$id, tenant:$t}}) SET d.namespace=$n,d.agent=$a,"
                  "d.title=$ti,d.source=$src,d.ts=$ts",
                  id=doc.id, t=tenant, n=namespace, a=agent, ti=doc.title, src=doc.source, ts=ts)
            s.run(f"UNWIND $rows AS r MERGE (c:{_CHUNK} {{id:r.id, tenant:$t}}) "
                  "SET c.namespace=$n,c.doc_id=r.doc_id,c.ord=r.ord,"
                  "c.heading=r.heading,c.text=r.text "
                  f"WITH c MATCH (d:{_DOC} {{id:$doc, tenant:$t}}) MERGE (c)-[:PART_OF]->(d)",
                  rows=chunk_rows, t=tenant, n=namespace, doc=doc.id)
            if ent_rows:
                s.run(f"UNWIND $rows AS r MERGE (e:{_ENT} {{id:r.id}}) "
                      "SET e.tenant=$t,e.namespace=$n,e.name=r.name,e.etype=r.etype",
                      rows=ent_rows, t=tenant, n=namespace)
            if ment_rows:
                s.run(f"UNWIND $rows AS r MATCH (e:{_ENT} {{id:r.eid}}) "
                      f"MATCH (c:{_CHUNK} {{id:r.cid, tenant:$t}}) MERGE (e)-[:MENTIONED_IN]->(c)",
                      rows=ment_rows, t=tenant)
            if rel_rows:
                s.run(f"UNWIND $rows AS r MATCH (a:{_ENT} {{id:r.sid}}) MATCH (b:{_ENT} {{id:r.oid}}) "
                      "MERGE (a)-[rel:REL {type:r.type}]->(b) SET rel.doc_id=r.doc_id", rows=rel_rows)
            if stale:  # a shrink can strand entities whose only mentions were stale chunks
                self._prune_orphans(s, tenant)
        return stale

    # -- search ------------------------------------------------------------
    def search(self, qvec, *, tenant, namespace, k):
        with self._lock:
            idx, meta = self._load_index(tenant)
            if not meta:
                return []
            # over-fetch 3x (min 30) so namespace filtering + dedup below can
            # still fill k results
            scores, ids = idx.search(np.asarray(qvec, dtype=np.float32), max(k * 3, 30))
        out, seen = [], set()
        for sc, uid in zip(scores[0].tolist(), ids[0].tolist()):
            row = meta.get(int(uid))
            if not row or (namespace and row.get("namespace") != namespace):
                continue
            # dedup by chunk id, not text prefix — templated corpora share long
            # boilerplate heads and prefix-dedup starves k
            if row.get("id") in seen:
                continue
            seen.add(row.get("id"))
            out.append({**row, "score": round(float(sc), 4)})
            if len(out) >= k:
                break
        return out

    # -- expand ------------------------------------------------------------
    def expand(self, chunk_ids, *, tenant, hops):
        if not chunk_ids:
            return {"entities": [], "relations": []}
        h = max(1, min(int(hops or 1), 3))  # var-length bounds can't be parameterized
        with self._driver.session() as s:
            ent = s.run(f"MATCH (e:{_ENT})-[:MENTIONED_IN]->(c:{_CHUNK}) "
                        "WHERE c.id IN $cids AND c.tenant=$t AND e.tenant=$t "
                        "RETURN DISTINCT e.name AS name,e.etype AS etype,e.id AS id LIMIT 100",
                        cids=chunk_ids, t=tenant).data()
            eids = [e["id"] for e in ent]
            rel, more = [], []
            if eids:
                rel = s.run(f"MATCH p=(a:{_ENT})-[:REL*1..{h}]->(:{_ENT}) "
                            "WHERE a.id IN $e AND a.tenant=$t "
                            "UNWIND relationships(p) AS r "
                            "RETURN DISTINCT startNode(r).name AS subject, r.type AS predicate, "
                            "endNode(r).name AS object LIMIT 200",
                            e=eids, t=tenant).data()
                if h > 1:
                    more = s.run(f"MATCH p=(a:{_ENT})-[:REL*1..{h}]->(:{_ENT}) "
                                 "WHERE a.id IN $e AND a.tenant=$t "
                                 "UNWIND nodes(p) AS n WITH DISTINCT n WHERE NOT n.id IN $e "
                                 "RETURN n.name AS name, n.etype AS etype, n.id AS id LIMIT 50",
                                 e=eids, t=tenant).data()
        return {"entities": (ent + more)[:100], "relations": rel}

    # -- delete ------------------------------------------------------------
    def delete_document(self, doc_id, *, tenant):
        """Remove a document, its chunks (graph + vectors), then prune entities
        with no remaining MENTIONED_IN edge anywhere in this tenant."""
        with self._lock:
            with self._driver.session() as s:
                cids = [r["id"] for r in s.run(
                    f"MATCH (c:{_CHUNK})-[:PART_OF]->(d:{_DOC} {{id:$id, tenant:$t}}) "
                    "RETURN c.id AS id", id=doc_id, t=tenant)]
                s.run(f"MATCH (d:{_DOC} {{id:$id, tenant:$t}}) "
                      f"OPTIONAL MATCH (c:{_CHUNK})-[:PART_OF]->(d) DETACH DELETE c, d",
                      id=doc_id, t=tenant)
                pruned = self._prune_orphans(s, tenant)
            idx, meta = self._load_index(tenant)
            self._drop_vectors(idx, meta, cids)
            self._save_index(tenant)
        return {"doc_id": doc_id, "chunks_deleted": len(cids), "entities_pruned": pruned}

    # -- stats -------------------------------------------------------------
    def stats(self, *, tenant, namespace=None):
        """Counts for a tenant. With namespace: documents/chunks are filtered
        (entities/relations stay tenant-wide — they merge across namespaces)."""
        where = "WHERE n.tenant=$t" if tenant else ""
        ns_where = where + (" AND n.namespace=$ns" if (tenant and namespace) else "")
        p = {"t": tenant} if tenant else {}
        np_ = {**p, "ns": namespace} if namespace else p
        with self._driver.session() as s:
            d = s.run(f"MATCH (n:{_DOC}) {ns_where} RETURN count(n) AS c", **np_).single()["c"]
            c = s.run(f"MATCH (n:{_CHUNK}) {ns_where} RETURN count(n) AS c", **np_).single()["c"]
            e = s.run(f"MATCH (n:{_ENT}) {where} RETURN count(n) AS c", **p).single()["c"]
            r = s.run(f"MATCH (:{_ENT})-[rel:REL]->() "
                      + ("WHERE startNode(rel).tenant=$t " if tenant else "")
                      + "RETURN count(rel) AS c", **p).single()["c"]
        out = {"documents": d, "chunks": c, "entities": e, "relations": r}
        if namespace:
            out["namespace"] = namespace
        return out

    def close(self):
        self._driver.close()
