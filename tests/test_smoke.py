"""Smoke tests. The local-backend e2e is skipped unless an LLM key is set
(it makes one cheap API call per chunk)."""
import os

import cognify
from cognify.loader import load
from cognify.extractor import Extraction, _parse


def test_chunking():
    doc = load("# A\n" + "word " * 800 + "\n# B\nshort tail here for a second segment.",
               is_path=False, title="t")
    assert doc.chunks and all(c.text for c in doc.chunks)
    assert len({c.id for c in doc.chunks}) == len(doc.chunks)  # unique ids


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
    from cognify import core
    from cognify.extractor import Entity

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
    from cognify import core
    from cognify.extractor import Entity

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
    from cognify import config, core
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
    from cognify import server

    class StubBackend:
        def stats(self, *, tenant=None):
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


def test_local_e2e():
    if not (os.environ.get("COGNIFY_LLM_KEY") or os.environ.get("OPENROUTER_API_KEY")):
        import pytest
        pytest.skip("no LLM key set")
    import tempfile
    os.environ["COGNIFY_DATA_DIR"] = tempfile.mkdtemp()
    be = cognify.get_backend("local")
    r = cognify.ingest(be, "Clark uses Neo4j and TurboVec for memory.",
                       is_path=False, tenant="t", namespace="n")
    assert r.chunks == 1
    res = cognify.recall(be, "what does Clark use?", tenant="t")
    assert res.chunks
    be.close()
