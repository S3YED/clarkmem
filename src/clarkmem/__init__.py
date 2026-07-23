"""
ClarkMem — a lightweight document-ingestion + typed knowledge-graph engine.

Quickstart:
    import clarkmem
    be = clarkmem.get_backend("local")          # zero external services
    clarkmem.ingest(be, "notes.md", tenant="me")
    res = clarkmem.recall(be, "what uses Neo4j?", tenant="me")
    print(res.entities, res.relations)
"""
from .core import (ingest, recall, invalidate, embed, get_model,  # noqa: F401
                   IngestResult, RecallResult)
from .backends import get_backend  # noqa: F401

__version__ = "1.0.1"
