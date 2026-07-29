from pathlib import Path

import src.rag.retriever.faiss_retriever as faiss_mod
from src.rag.retriever.faiss_retriever import FaissRetriever


class DummyKBIndex:
    def __init__(self, kb, encoder):
        self.kb = kb

    @classmethod
    def load(cls, kb, encoder):
        return cls(kb, encoder)

    def query(self, mention, top_k):
        # Return simple list of RetrievalResult-like dicts
        from src.rag.retriever.base import RetrievalResult

        return [RetrievalResult(id="1", name=mention, score=0.9, tty="")]


def test_faiss_retriever_uses_kbindex(monkeypatch):
    monkeypatch.setattr(faiss_mod, "KBIndex", DummyKBIndex)

    retriever = FaissRetriever(encoder="dummy")
    results = retriever.query("headache", kb="ICD", top_k=3)

    assert len(results) == 1
    assert results[0].name == "headache"
