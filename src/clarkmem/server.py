"""
ClarkMem HTTP server — a tiny FastAPI app exposing the engine over HTTP, so any
agent runtime (Hermes, n8n, a shell, another service) can ingest and recall
without a Python import.

Run:  clarkmem-serve              (defaults to 127.0.0.1:8799)
Env:  CLARKMEM_HOST (comma-separated for multi-bind), CLARKMEM_PORT, CLARKMEM_BACKEND,
      plus the usual LLM/Neo4j vars.

Endpoints:
  GET    /health     (always open — for watchdogs)
  POST   /ingest     {text|path, title, key, tenant, namespace, agent, extract, workers}
  POST   /recall     {query, tenant, namespace, k, hops, mode, include_invalidated}
  POST   /invalidate {subject, predicate?, object?, tenant}   fact is closed, kept
  POST   /maintain   {tenant}                                  integrity pass
  DELETE /doc?doc_id=&tenant=
  GET    /stats?tenant=&namespace=

Auth: set CLARKMEM_API_KEY to require an x-api-key header on every endpoint
except /health. Unset = open (loopback-only single-user setups).

Hardening: server-side path ingestion ({"path": ...}) is DISABLED unless
CLARKMEM_INGEST_ROOT points at a directory; only files under that root can be
read. Text bodies are capped at CLARKMEM_MAX_TEXT chars (default 2,000,000);
recall k is clamped to 100 and hops to 3.
"""
from __future__ import annotations

import os
import threading

import clarkmem
from clarkmem import config

try:
    from fastapi import FastAPI, Header, HTTPException, Query
except ImportError as e:  # pragma: no cover
    raise SystemExit("FastAPI not installed. Run: pip install 'clarkmem-kg[serve]'") from e

app = FastAPI(title="ClarkMem", version=clarkmem.__version__)
_backend = None
_backend_lock = threading.Lock()


def _be():
    global _backend
    with _backend_lock:  # two first-requests must not build two backends
        if _backend is None:
            _backend = clarkmem.get_backend(config.backend_default())
        return _backend


def _auth(key: str | None):
    """Constant-time key check when CLARKMEM_API_KEY is set; no-op when unset."""
    import hmac
    want = config.api_key()
    if want and not (key and hmac.compare_digest(key, want)):
        raise HTTPException(401, "invalid or missing x-api-key")


@app.get("/health")
def health():
    return {"status": "ok", "backend": config.backend_default(),
            "version": clarkmem.__version__}


@app.post("/ingest")
def ingest(body: dict, x_api_key: str | None = Header(None)):
    _auth(x_api_key)
    text, path = body.get("text"), body.get("path")
    if not text and not path:
        raise HTTPException(400, "provide 'text' or 'path'")
    if path:
        root = config.ingest_root()
        if not root:
            raise HTTPException(403, "server-side path ingestion is disabled; set "
                                     "CLARKMEM_INGEST_ROOT to a directory to allow it")
        real, rootr = os.path.realpath(str(path)), os.path.realpath(root)
        if real != rootr and not real.startswith(rootr + os.sep):
            raise HTTPException(403, "path is outside CLARKMEM_INGEST_ROOT")
        src = real
    else:
        max_chars = config.max_text()
        if len(str(text)) > max_chars:
            raise HTTPException(413, f"text exceeds CLARKMEM_MAX_TEXT ({max_chars} chars)")
        src = str(text)
    try:
        workers = min(int(body["workers"]), 32) if body.get("workers") else None
        r = clarkmem.ingest(_be(), src, tenant=body.get("tenant", "default"),
                           namespace=body.get("namespace", "default"),
                           agent=body.get("agent", "agent"),
                           is_path=bool(path), title=body.get("title"),
                           key=body.get("key"),
                           do_extract=body.get("extract", True), workers=workers)
        return r.__dict__
    except Exception as e:
        raise HTTPException(500, f"ingest failed: {e}")


@app.post("/recall")
def recall(body: dict, x_api_key: str | None = Header(None)):
    _auth(x_api_key)
    q = body.get("query")
    if not q:
        raise HTTPException(400, "provide 'query'")
    try:
        k, hops = int(body.get("k", 8)), int(body.get("hops", 1))  # core clamps bounds
    except (TypeError, ValueError):
        raise HTTPException(400, "k and hops must be integers")
    try:
        res = clarkmem.recall(_be(), q, tenant=body.get("tenant", "default"),
                             namespace=body.get("namespace"), k=k, hops=hops,
                             mode=body.get("mode", "hybrid"),
                             include_invalidated=bool(body.get("include_invalidated")))
        return {"query": res.query, "tenant": res.tenant, "chunks": list(res.chunks),
                "entities": list(res.entities), "relations": list(res.relations)}
    except Exception as e:
        raise HTTPException(500, f"recall failed: {e}")


@app.post("/invalidate")
def invalidate(body: dict, x_api_key: str | None = Header(None)):
    _auth(x_api_key)
    subject = body.get("subject")
    if not subject:
        raise HTTPException(400, "provide 'subject' (and optionally predicate/object)")
    try:
        n = clarkmem.invalidate(_be(), str(subject), tenant=body.get("tenant", "default"),
                               predicate=body.get("predicate"), object=body.get("object"))
        return {"invalidated": n}
    except Exception as e:
        raise HTTPException(500, f"invalidate failed: {e}")


@app.post("/maintain")
def maintain(body: dict, x_api_key: str | None = Header(None)):
    _auth(x_api_key)
    tenant = body.get("tenant")
    if not tenant:
        raise HTTPException(400, "provide 'tenant'")
    try:
        return _be().maintain(tenant=tenant)
    except Exception as e:
        raise HTTPException(500, f"maintain failed: {e}")


@app.delete("/doc")
def forget(doc_id: str = Query(...), tenant: str = Query("default"),
           x_api_key: str | None = Header(None)):
    _auth(x_api_key)
    try:
        return _be().delete_document(doc_id, tenant=tenant)
    except Exception as e:
        raise HTTPException(500, f"delete failed: {e}")


@app.get("/stats")
def stats(tenant: str = Query(None), namespace: str = Query(None),
          x_api_key: str | None = Header(None)):
    _auth(x_api_key)
    if namespace and not tenant:
        raise HTTPException(400, "a namespace filter requires a tenant")
    return _be().stats(tenant=tenant, namespace=namespace)


def main():
    import threading

    import uvicorn
    port = config.serve_port()
    # CLARKMEM_HOST may be comma-separated (e.g. "127.0.0.1,100.x.y.z") to bind
    # loopback + a VPN/tailnet IP without ever exposing 0.0.0.0.
    hosts = config.serve_hosts()
    for h in hosts[1:]:
        threading.Thread(target=lambda hh=h: uvicorn.run(app, host=hh, port=port,
                                                         log_level="warning"), daemon=True).start()
    uvicorn.run(app, host=hosts[0], port=port, log_level="info")


if __name__ == "__main__":
    main()
