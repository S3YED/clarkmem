"""Smoke tests. The local-backend e2e is skipped unless an LLM key is set
(it makes one cheap API call per chunk)."""
import os

import clarkmem
from clarkmem.loader import load
from clarkmem.extractor import Extraction, _parse


def test_chunking():
    doc = load("# A\n" + "word " * 800 + "\n# B\nshort tail here for a second segment.",
               is_path=False, title="t")
    assert doc.chunks and all(c.text for c in doc.chunks)
    assert len({c.id for c in doc.chunks}) == len(doc.chunks)  # unique ids


def test_inline_doc_ids_content_addressed():
    import hashlib
    from clarkmem.loader import _doc_id
    base = "x" * 600
    # two distinct notes sharing a >512-char preamble must NOT collide
    assert _doc_id("inline", base + " alpha") != _doc_id("inline", base + " omega")
    # ids for short inline text are unchanged from the pre-0.5 scheme (no
    # migration churn for existing stores)
    short = "a short inline note"
    assert _doc_id("inline", short) == hashlib.sha256(f"inline::{short}".encode()).hexdigest()[:16]


def test_extractor_caps_hostile_output():
    import json as _json
    from clarkmem.extractor import _MAX_ENTITIES, _MAX_NAME
    flood = {"entities": [{"name": f"e{i}", "type": "Concept"} for i in range(500)],
             "relations": []}
    assert len(_parse(_json.dumps(flood)).entities) == _MAX_ENTITIES
    long_name = {"entities": [{"name": "N" * 5000, "type": "Concept"}], "relations": []}
    assert len(_parse(_json.dumps(long_name)).entities[0].name) == _MAX_NAME


def test_extraction_parse():
    ex = _parse('{"entities":[{"name":"Clark","type":"Product"},{"name":"Neo4j","type":"Technology"}],'
                '"relations":[{"subject":"Clark","predicate":"USES","object":"Neo4j"}]}')
    assert isinstance(ex, Extraction)
    assert {e.name for e in ex.entities} == {"Clark", "Neo4j"}
    assert ex.relations[0].predicate == "USES"


def test_parse_drops_ungrounded_relations():
    ex = _parse('{"entities":[{"name":"Clark","type":"Product"}],'
                '"relations":[{"subject":"Clark","predicate":"USES","object":"Ghost"}]}')
    assert ex.relations == ()  # object not in entities -> dropped


def test_parallel_extraction_matches_serial(monkeypatch):
    import numpy as np
    from clarkmem import core
    from clarkmem.extractor import Entity

    def fake_extract(text, **kw):
        return Extraction(entities=(Entity(name=text[:8], type="Concept"),), relations=())

    monkeypatch.setattr(core._ex, "extract", fake_extract)

    class FakeBackend:
        def embed_texts(self, texts):
            return np.zeros((len(texts), 384), dtype=np.float32)

        def load_document(self, doc, **kw):
            self.extractions = kw["extractions"]

    text = "# A\n" + "alpha " * 700 + "\n# B\n" + "beta " * 700
    serial_be, par_be = FakeBackend(), FakeBackend()
    r1 = core.ingest(serial_be, text, is_path=False, workers=1)
    r2 = core.ingest(par_be, text, is_path=False, workers=4)
    assert r1.chunks == r2.chunks > 1
    assert r1.entities == r2.entities == r1.chunks  # one stub entity per chunk
    assert serial_be.extractions.keys() == par_be.extractions.keys()


def test_parallel_extraction_degrades_per_chunk(monkeypatch):
    import numpy as np
    from clarkmem import core
    from clarkmem.extractor import Entity

    def flaky_extract(text, **kw):
        if "beta" in text:
            raise RuntimeError("boom")
        return Extraction(entities=(Entity(name="A", type="Concept"),), relations=())

    monkeypatch.setattr(core._ex, "extract", flaky_extract)

    class FakeBackend:
        def embed_texts(self, texts):
            return np.zeros((len(texts), 384), dtype=np.float32)

        def load_document(self, doc, **kw):
            pass

    r = core.ingest(FakeBackend(), "# A\n" + "alpha " * 700 + "\n# B\n" + "beta " * 700,
                    is_path=False, workers=4)
    assert r.chunks > 1 and 0 < r.entities < r.chunks  # failures degraded, not raised


def test_fastembed_provider_normalized(monkeypatch):
    try:
        import fastembed  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("fastembed not installed")
    import numpy as np
    from clarkmem import config, core
    monkeypatch.setattr(config, "EMBED_PROVIDER", "fastembed")
    monkeypatch.setattr(core, "_model", None)
    v = core.embed(["hello world", "knowledge graphs connect facts"])
    core._model = None  # do not leak the singleton into other tests
    assert v.shape == (2, 384) and v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-3)


def test_server_auth(monkeypatch):
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        import pytest
        pytest.skip("fastapi not installed")
    from clarkmem import server

    class StubBackend:
        def stats(self, *, tenant=None, namespace=None):
            return {"documents": 0}

    monkeypatch.setattr(server, "_backend", StubBackend())
    monkeypatch.setenv("COGNIFY_API_KEY", "sekrit")
    c = TestClient(server.app)
    assert c.get("/health").status_code == 200                    # always open
    assert c.get("/stats").status_code == 401                     # no key
    assert c.get("/stats", headers={"x-api-key": "wrong"}).status_code == 401
    assert c.get("/stats", headers={"x-api-key": "sekrit"}).status_code == 200
    assert c.post("/recall", json={"query": "x"},
                  headers={"x-api-key": "wrong"}).status_code == 401
    monkeypatch.delenv("COGNIFY_API_KEY")
    assert c.get("/stats").status_code == 200                     # unset = open


def test_extractor_retries_on_429(monkeypatch):
    import time as _time
    from clarkmem import extractor

    calls = []

    class Resp:
        def __init__(self, code, content=""):
            self.status_code = code
            self._content = content

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"http {self.status_code}")

        def json(self):
            return {"choices": [{"message": {"content": self._content}}]}

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(url)
        if len(calls) == 1:
            return Resp(429)
        return Resp(200, '{"entities":[{"name":"X","type":"Concept"}],"relations":[]}')

    monkeypatch.setattr(extractor.requests, "post", fake_post)
    monkeypatch.setattr(_time, "sleep", lambda s: None)
    monkeypatch.setenv("COGNIFY_LLM_KEY", "k")
    ex = extractor.extract("some sufficiently long chunk of text to pass the length gate")
    assert len(calls) == 2 and ex.entities[0].name == "X"


def test_cache_path_is_namespaced(monkeypatch, tmp_path):
    from clarkmem import cli, config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    a = cli._cache_path("acme", "docs")
    b = cli._cache_path("acme", "mail")
    assert a != b  # same dir into a second namespace must not be cache-skipped


def _local_backend_or_skip(monkeypatch, tmp_path):
    import pytest
    pytest.importorskip("chromadb")
    from clarkmem import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    return clarkmem.get_backend("local")


def test_local_e2e_vectors_only(monkeypatch, tmp_path):
    be = _local_backend_or_skip(monkeypatch, tmp_path)
    r = clarkmem.ingest(be, "Clark uses Neo4j and TurboVec for memory.",
                       is_path=False, tenant="t", namespace="n", do_extract=False)
    assert r.chunks == 1 and not r.extracted
    res = clarkmem.recall(be, "what does Clark use?", tenant="t")
    assert res.chunks and res.chunks[0]["text"].startswith("Clark uses")
    be.close()


def _stub_extract_two_docs(monkeypatch):
    """doc about turtles -> entities A,B + A->B; doc about cheese -> B,C,D + B->C, C->D."""
    from clarkmem import core
    from clarkmem.extractor import Entity, Relation

    def fake_extract(text, **kw):
        if "turtle" in text:
            return Extraction(entities=(Entity("A", "Concept"), Entity("B", "Concept")),
                              relations=(Relation("A", "LIKES", "B"),))
        return Extraction(entities=(Entity("B", "Concept"), Entity("C", "Concept"),
                                    Entity("D", "Concept")),
                          relations=(Relation("B", "MADE_OF", "C"), Relation("C", "AGED_IN", "D")))

    monkeypatch.setattr(core._ex, "extract", fake_extract)


def test_local_multihop_expand(monkeypatch, tmp_path):
    be = _local_backend_or_skip(monkeypatch, tmp_path)
    _stub_extract_two_docs(monkeypatch)
    clarkmem.ingest(be, "quantum turtles stack in shells all the way down today",
                   is_path=False, tenant="t")
    clarkmem.ingest(be, "medieval cheese wheels ferment in stone cellars for years",
                   is_path=False, tenant="t")
    one = clarkmem.recall(be, "quantum turtles", tenant="t", k=1, hops=1)
    names1 = {e["name"] for e in one.entities}
    rels1 = {(r["subject"], r["object"]) for r in one.relations}
    assert names1 == {"A", "B"} and ("B", "C") in rels1 and ("C", "D") not in rels1
    two = clarkmem.recall(be, "quantum turtles", tenant="t", k=1, hops=2)
    names2 = {e["name"] for e in two.entities}
    rels2 = {(r["subject"], r["object"]) for r in two.relations}
    assert "C" in names2 and ("C", "D") in rels2
    be.close()


def test_local_delete_prunes_orphans(monkeypatch, tmp_path):
    be = _local_backend_or_skip(monkeypatch, tmp_path)
    _stub_extract_two_docs(monkeypatch)
    clarkmem.ingest(be, "quantum turtles stack in shells all the way down today",
                   is_path=False, tenant="t")
    r2 = clarkmem.ingest(be, "medieval cheese wheels ferment in stone cellars for years",
                        is_path=False, tenant="t")
    assert be.stats(tenant="t")["entities"] == 4  # A, B, C, D
    out = be.delete_document(r2.doc_id, tenant="t")
    assert out["chunks_deleted"] == 1 and out["entities_pruned"] == 2  # C, D gone
    s = be.stats(tenant="t")
    assert s["documents"] == 1 and s["entities"] == 2  # A, B survive (B still cited by doc1)
    assert not clarkmem.recall(be, "medieval cheese", tenant="t").chunks or \
        "cheese" not in clarkmem.recall(be, "medieval cheese", tenant="t").chunks[0]["text"]
    be.close()


def test_local_stats_namespace_filter(monkeypatch, tmp_path):
    be = _local_backend_or_skip(monkeypatch, tmp_path)
    clarkmem.ingest(be, "alpha " * 30, is_path=False, tenant="t", namespace="n1", do_extract=False)
    clarkmem.ingest(be, "beta " * 30, is_path=False, tenant="t", namespace="n2", do_extract=False)
    assert be.stats(tenant="t")["documents"] == 2
    s = be.stats(tenant="t", namespace="n1")
    assert s["documents"] == 1 and s["chunks"] == 1 and s["namespace"] == "n1"
    be.close()


def test_key_identity_and_local_shrink_cleanup(monkeypatch, tmp_path):
    be = _local_backend_or_skip(monkeypatch, tmp_path)
    _stub_extract_two_docs(monkeypatch)
    long_text = "# A\n" + "turtle " * 700 + "\n# B\n" + "cheese " * 700
    r1 = clarkmem.ingest(be, long_text, is_path=False, tenant="t", key="note")
    assert r1.chunks > 1 and be.stats(tenant="t")["entities"] == 4  # A,B + C,D
    short = "quantum turtle memo, tiny now but still all about turtles today"
    r2 = clarkmem.ingest(be, short, is_path=False, tenant="t", key="note")
    assert r1.doc_id == r2.doc_id                       # key = stable identity
    s = be.stats(tenant="t")
    assert s["documents"] == 1 and s["chunks"] == 1     # stale graph chunks gone
    assert s["entities"] == 2 and s["vectors"] == 1     # C,D pruned with them
    be.close()


def test_local_stats_all_tenants(monkeypatch, tmp_path):
    be = _local_backend_or_skip(monkeypatch, tmp_path)
    clarkmem.ingest(be, "alpha " * 30, is_path=False, tenant="t1", do_extract=False)
    clarkmem.ingest(be, "beta " * 30, is_path=False, tenant="t2", do_extract=False)
    s = be.stats(tenant=None)  # no tenant = global, same meaning as neo4j backend
    assert s["documents"] == 2 and s["chunks"] == 2 and s["vectors"] == 2
    be.close()


def test_server_clamps_and_path_gate(monkeypatch, tmp_path):
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        import pytest
        pytest.skip("fastapi not installed")
    import numpy as np
    from clarkmem import server

    class StubBackend:
        seen_k = None

        def embed_texts(self, texts):
            return np.zeros((len(texts), 384), dtype=np.float32)

        def search(self, qvec, *, tenant, namespace, k):
            StubBackend.seen_k = k
            return []

        def anchor_chunks(self, query, *, tenant, namespace, limit):
            return []

        def expand(self, chunk_ids, *, tenant, hops, include_invalidated=False):
            return {"entities": [], "relations": []}

        def load_document(self, doc, **kw):
            pass

        def invalidate_relations(self, subject, *, tenant, predicate=None, object=None):
            return 2

        def maintain(self, *, tenant):
            return {"tenant": tenant, "entities_pruned": 0}

    monkeypatch.setattr(server, "_backend", StubBackend())
    monkeypatch.delenv("COGNIFY_API_KEY", raising=False)
    c = TestClient(server.app)

    assert c.post("/invalidate", json={"subject": "Clark"}).json() == {"invalidated": 2}
    assert c.post("/invalidate", json={}).status_code == 400
    assert c.post("/maintain", json={"tenant": "t"}).status_code == 200
    assert c.post("/maintain", json={}).status_code == 400

    assert c.post("/recall", json={"query": "x", "k": 99999}).status_code == 200
    assert StubBackend.seen_k == 100                       # k clamped

    doc = tmp_path / "doc.md"
    doc.write_text("a perfectly harmless document with plenty of characters in it")
    monkeypatch.delenv("COGNIFY_INGEST_ROOT", raising=False)
    assert c.post("/ingest", json={"path": str(doc)}).status_code == 403   # gate closed
    monkeypatch.setenv("COGNIFY_INGEST_ROOT", str(tmp_path))
    assert c.post("/ingest", json={"path": str(doc), "extract": False}).status_code == 200
    assert c.post("/ingest", json={"path": "/etc/hostname"}).status_code == 403  # outside root

    monkeypatch.setenv("COGNIFY_MAX_TEXT", "100")
    assert c.post("/ingest", json={"text": "y" * 200}).status_code == 413  # size cap


def test_temporal_evidence_invalidate_revive(monkeypatch, tmp_path):
    be = _local_backend_or_skip(monkeypatch, tmp_path)
    _stub_extract_two_docs(monkeypatch)
    turtle = "quantum turtles stack in shells all the way down today"
    clarkmem.ingest(be, turtle, is_path=False, tenant="t", key="n1")
    clarkmem.ingest(be, turtle + " again", is_path=False, tenant="t", key="n2")
    rels = {(r["subject"], r["object"]): r for r in
            clarkmem.recall(be, "quantum turtles", tenant="t", k=1).relations}
    assert rels[("A", "B")]["evidence"] == 2          # two docs assert the fact

    assert clarkmem.invalidate(be, "A", tenant="t") == 1
    live = clarkmem.recall(be, "quantum turtles", tenant="t", k=1)
    assert not live.relations                          # closed facts stay hidden
    hist = clarkmem.recall(be, "quantum turtles", tenant="t", k=1, include_invalidated=True)
    assert any(r.get("invalid_at") for r in hist.relations)  # ...but not erased

    clarkmem.ingest(be, turtle + " once more", is_path=False, tenant="t", key="n3")
    revived = clarkmem.recall(be, "quantum turtles", tenant="t", k=1)
    assert any(r["subject"] == "A" for r in revived.relations)  # new evidence revives
    be.close()


def test_functional_predicate_supersedes(monkeypatch, tmp_path):
    from clarkmem import core
    from clarkmem.extractor import Entity, Relation
    be = _local_backend_or_skip(monkeypatch, tmp_path)
    monkeypatch.setenv("COGNIFY_FUNCTIONAL_PREDICATES", "WORKS_AT")

    def fake_extract(text, **kw):
        org = "Acme" if "acme" in text else "Globex"
        return Extraction(entities=(Entity("Sam", "Person"), Entity(org, "Organization")),
                          relations=(Relation("Sam", "WORKS_AT", org),))

    monkeypatch.setattr(core._ex, "extract", fake_extract)
    clarkmem.ingest(be, "sam joined acme years ago and stayed a long time",
                   is_path=False, tenant="t")
    clarkmem.ingest(be, "sam moved over to globex just this spring season",
                   is_path=False, tenant="t")
    rels = clarkmem.recall(be, "where does sam work", tenant="t", k=8).relations
    objs = {r["object"] for r in rels if r["subject"] == "Sam"}
    assert objs == {"Globex"}                          # old employer closed, not shown
    hist = clarkmem.recall(be, "where does sam work", tenant="t", k=8,
                          include_invalidated=True).relations
    assert {"Acme", "Globex"} <= {r["object"] for r in hist if r["subject"] == "Sam"}
    be.close()


def test_anchor_chunks_and_hybrid_fusion(monkeypatch, tmp_path):
    from clarkmem import core
    from clarkmem.extractor import Entity
    be = _local_backend_or_skip(monkeypatch, tmp_path)

    def fake_extract(text, **kw):
        return Extraction(entities=(Entity("Zebrastripe Observatory", "Location"),),
                          relations=())

    monkeypatch.setattr(core._ex, "extract", fake_extract)
    clarkmem.ingest(be, "the dome atop the hill hosts nightly stargazing sessions",
                   is_path=False, tenant="t")
    hits = be.anchor_chunks("history of the Zebrastripe Observatory building",
                            tenant="t", namespace=None, limit=5)
    assert hits and hits[0]["anchor"] == "Zebrastripe Observatory"
    assert "stargazing" in hits[0]["text"]

    # rrf fuses distinct vector + anchor lists, both survive
    fused = core._rrf([[{"id": "x", "text": "vec"}], [{"id": "y", "text": "anc"}]], k=8)
    assert {r["id"] for r in fused} == {"x", "y"}
    be.close()


def test_hybrid_recall_namespace_filter(monkeypatch, tmp_path):
    """Regression (v1.0.0): namespaced recall in the DEFAULT hybrid mode must
    filter both lanes, not fail — the neo4j anchor Cypher was invalid whenever
    a namespace was set, 500-ing every namespaced /recall."""
    from clarkmem import core
    from clarkmem.extractor import Entity
    be = _local_backend_or_skip(monkeypatch, tmp_path)

    def fake_extract(text, **kw):  # same entity in both namespaces
        return Extraction(entities=(Entity("Kelpwatch Station", "Location"),), relations=())

    monkeypatch.setattr(core._ex, "extract", fake_extract)
    clarkmem.ingest(be, "the reef sensors at the station log kelp density hourly",
                   is_path=False, tenant="t", namespace="ocean")
    clarkmem.ingest(be, "alpine weather balloons report wind shear at dawn",
                   is_path=False, tenant="t", namespace="sky")
    res = clarkmem.recall(be, "Kelpwatch Station kelp readings", tenant="t", namespace="ocean")
    assert res.chunks and all(c["namespace"] == "ocean" for c in res.chunks)
    assert any("reef sensors" in c["text"] for c in res.chunks)
    other = clarkmem.recall(be, "Kelpwatch Station kelp readings", tenant="t", namespace="sky")
    assert all(c["namespace"] == "sky" for c in other.chunks)  # anchor lane filtered too
    be.close()


def test_recall_survives_anchor_failure():
    """A broken anchor lane must degrade to vector-only results, never kill the
    whole recall (how the v1.0.0 neo4j bug became a total namespaced-recall
    outage instead of a soft one)."""
    import numpy as np

    class Backend:
        def embed_texts(self, texts):
            return np.zeros((len(texts), 384), dtype=np.float32)

        def search(self, qvec, *, tenant, namespace, k):
            return [{"id": "c1", "text": "vector hit", "namespace": namespace or ""}]

        def anchor_chunks(self, query, *, tenant, namespace, limit):
            raise RuntimeError("boom")

        def expand(self, chunk_ids, *, tenant, hops, include_invalidated=False):
            return {"entities": [], "relations": []}

    res = clarkmem.recall(Backend(), "q", tenant="t", namespace="n")
    assert [c["id"] for c in res.chunks] == ["c1"]


def test_local_maintain_heals(monkeypatch, tmp_path):
    be = _local_backend_or_skip(monkeypatch, tmp_path)
    _stub_extract_two_docs(monkeypatch)
    r1 = clarkmem.ingest(be, "quantum turtles stack in shells all the way down today",
                        is_path=False, tenant="t")
    clarkmem.ingest(be, "medieval cheese wheels ferment in stone cellars for years",
                   is_path=False, tenant="t")
    g = be._load_graph("t")
    g.remove_node(f"doc::{r1.doc_id}")                 # simulate historical drift
    rep = be.maintain(tenant="t")
    assert rep["dangling_chunks_removed"] == 1 and rep["vectors_removed"] == 1
    assert rep["entities_pruned"] == 1                 # A orphaned; B survives via doc2
    s = be.stats(tenant="t")
    assert s["documents"] == 1 and s["chunks"] == 1 and s["vectors"] == 1
    be.close()


def test_local_e2e():
    if not (os.environ.get("COGNIFY_LLM_KEY") or os.environ.get("OPENROUTER_API_KEY")):
        import pytest
        pytest.skip("no LLM key set")
    import tempfile

    from clarkmem import config
    config.DATA_DIR = __import__("pathlib").Path(tempfile.mkdtemp())
    be = clarkmem.get_backend("local")
    r = clarkmem.ingest(be, "Clark uses Neo4j and TurboVec for memory.",
                       is_path=False, tenant="t", namespace="n")
    assert r.chunks == 1
    res = clarkmem.recall(be, "what does Clark use?", tenant="t")
    assert res.chunks
    be.close()
