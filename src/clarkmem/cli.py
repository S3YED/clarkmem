"""ClarkMem CLI: ingest documents and run hybrid recall from the terminal."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import clarkmem
from clarkmem import config


def _cache_path(tenant: str, namespace: str) -> Path:
    import re
    d = config.DATA_DIR / "cache"
    d.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", f"{tenant}--{namespace}")
    return d / f"ingest-{safe}.json"


def cmd_ingest(a):
    be = clarkmem.get_backend(a.backend)
    text = sys.stdin.read() if a.path == "-" else a.path
    r = clarkmem.ingest(be, text, tenant=a.tenant, namespace=a.namespace or "default",
                       agent=a.agent, is_path=(a.path != "-"), key=a.key or None,
                       do_extract=not a.no_extract, workers=a.workers or None)
    print(json.dumps(r.__dict__, indent=2)); be.close()


def cmd_ingest_dir(a):
    be = clarkmem.get_backend(a.backend)
    files = sorted(p for p in Path(a.path).expanduser().glob(a.glob) if p.is_file())
    if a.limit:
        files = files[:a.limit]
    cache, cpath = {}, _cache_path(a.tenant, a.namespace or "default")
    if a.cache and cpath.exists():
        try:
            cache = json.loads(cpath.read_text())
        except Exception:
            cache = {}
    print(f"ingesting {len(files)} files -> tenant={a.tenant} ns={a.namespace or 'default'} "
          f"extract={not a.no_extract} cache={a.cache}", flush=True)
    tot = {"docs": 0, "chunks": 0, "entities": 0, "relations": 0, "failed": 0, "skipped": 0}
    t0 = time.time()
    for i, p in enumerate(files):
        if a.cache:
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            if cache.get(str(p)) == h:
                tot["skipped"] += 1
                continue
        try:
            r = clarkmem.ingest(be, str(p), tenant=a.tenant, namespace=a.namespace or "default",
                               agent=a.agent, is_path=True, do_extract=not a.no_extract,
                               workers=a.workers or None)
            tot["docs"] += 1; tot["chunks"] += r.chunks
            tot["entities"] += r.entities; tot["relations"] += r.relations
            if a.cache:
                cache[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        except Exception as e:
            tot["failed"] += 1
            print(f"  [FAIL] {p.name}: {e}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(files)} | {tot['chunks']}c {tot['entities']}e "
                  f"{tot['relations']}r ({time.time()-t0:.0f}s)", flush=True)
    if a.cache:
        cpath.write_text(json.dumps(cache, indent=2))
    tot["seconds"] = round(time.time() - t0, 1)
    print("DONE:", json.dumps(tot, indent=2)); be.close()


def cmd_recall(a):
    be = clarkmem.get_backend(a.backend)
    res = clarkmem.recall(be, a.query, tenant=a.tenant, namespace=a.namespace, k=a.k, hops=a.hops,
                         mode=a.mode, include_invalidated=a.include_invalidated)
    print(json.dumps({
        "query": res.query, "tenant": res.tenant,
        "chunks": [{"score": c["score"], "heading": c.get("heading"), "text": c["text"][:200],
                    **({"anchor": c["anchor"]} if c.get("anchor") else {})}
                   for c in res.chunks],
        "entities": [f"{e['name']} ({e['etype']})" for e in res.entities],
        "relations": [f"{r['subject']} -{r['predicate']}-> {r['object']}"
                      + (f"  [x{r['evidence']}]" if r.get("evidence", 1) > 1 else "")
                      + ("  [invalidated]" if r.get("invalid_at") else "")
                      for r in res.relations],
    }, indent=2, ensure_ascii=False)); be.close()


def cmd_invalidate(a):
    be = clarkmem.get_backend(a.backend)
    n = clarkmem.invalidate(be, a.subject, tenant=a.tenant,
                           predicate=a.predicate or None, object=a.object or None)
    print(json.dumps({"invalidated": n})); be.close()


def cmd_maintain(a):
    be = clarkmem.get_backend(a.backend)
    print(json.dumps(be.maintain(tenant=a.tenant), indent=2)); be.close()


def cmd_stats(a):
    be = clarkmem.get_backend(a.backend)
    print(json.dumps(be.stats(tenant=a.tenant, namespace=a.namespace), indent=2)); be.close()


def cmd_forget(a):
    be = clarkmem.get_backend(a.backend)
    print(json.dumps(be.delete_document(a.doc_id, tenant=a.tenant), indent=2)); be.close()


def main():
    p = argparse.ArgumentParser(prog="clarkmem")
    p.add_argument("--backend", default=None, choices=["neo4j", "local"])
    p.add_argument("--tenant", default="default")
    # None so recall/stats span ALL namespaces unless one is asked for
    # (ingest falls back to 'default'), and 'default' itself stays filterable
    p.add_argument("--namespace", default=None,
                   help="ingest into / filter by a namespace (ingest default: 'default'; "
                        "recall/stats default: all namespaces)")
    p.add_argument("--agent", default="agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ingest"); s.add_argument("path"); s.add_argument("--no-extract", action="store_true"); s.add_argument("--workers", type=int, default=0); s.add_argument("--key", default="", help="stable identity for an evolving note (re-ingest updates in place)"); s.set_defaults(fn=cmd_ingest)
    s = sub.add_parser("ingest-dir"); s.add_argument("path"); s.add_argument("--glob", default="**/*.md"); s.add_argument("--limit", type=int, default=0); s.add_argument("--no-extract", action="store_true"); s.add_argument("--cache", action="store_true"); s.add_argument("--workers", type=int, default=0); s.set_defaults(fn=cmd_ingest_dir)
    s = sub.add_parser("recall"); s.add_argument("query"); s.add_argument("-k", type=int, default=8); s.add_argument("--hops", type=int, default=1); s.add_argument("--mode", default="hybrid", choices=["hybrid", "vector"]); s.add_argument("--include-invalidated", action="store_true"); s.set_defaults(fn=cmd_recall)
    s = sub.add_parser("invalidate"); s.add_argument("subject"); s.add_argument("--predicate", default=""); s.add_argument("--object", default=""); s.set_defaults(fn=cmd_invalidate)
    s = sub.add_parser("maintain"); s.set_defaults(fn=cmd_maintain)
    s = sub.add_parser("forget"); s.add_argument("doc_id"); s.set_defaults(fn=cmd_forget)
    s = sub.add_parser("stats"); s.set_defaults(fn=cmd_stats)
    a = p.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
