import pandas as pd
import numpy as np

from pathlib import Path

import src.rag.retriever.bm25_retriever as bm25_mod
from src.rag.retriever.bm25_retriever import BM25Retriever


class DummyBM25:
    def __init__(self, docs):
        self.docs = docs

    def get_scores(self, tokens):
        # return ascending scores so argsort yields predictable ordering
        return np.arange(len(self.docs), dtype=float)


def test_bm25_get_index_and_query(tmp_path, monkeypatch):
    kb = "RXNORM"
    kb_dir = tmp_path / kb
    kb_dir.mkdir()
    metadata_path = kb_dir / f"{kb}_metadata.parquet"
    metadata_path.write_text("")

    df = pd.DataFrame([
        {"id": 10, "name": "Aspirin 100mg", "tty": "IN"},
        {"id": 11, "name": "Ibuprofen 200mg", "tty": "IN"},
        {"id": 12, "name": "Paracetamol 500mg", "tty": "IN"},
    ])

    # Monkeypatch pd.read_parquet and BM25Okapi
    monkeypatch.setattr(bm25_mod.pd, "read_parquet", lambda path: df)
    monkeypatch.setattr(bm25_mod, "BM25Okapi", DummyBM25)

    retriever = BM25Retriever(output=tmp_path)

    results = retriever.query("Aspirin", kb, top_k=2)

    # Expect up to 2 results and ids must be strings
    assert len(results) <= 2
    for r in results:
        assert isinstance(r.id, str)
        assert isinstance(r.score, float)
