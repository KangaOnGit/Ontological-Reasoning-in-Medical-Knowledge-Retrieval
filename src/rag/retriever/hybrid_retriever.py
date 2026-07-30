from src.rag.retriever.faiss_retriever import FaissRetriever
from src.rag.retriever.bm25_retriever import BM25Retriever
from src.rag.retriever.exact_alias_retriever import ExactAliasRetriever
from src.rag.encoders.text_encoder import TextEncoder
from src.rag.retriever.base import RetrievalResult

from pathlib import Path

import logging
log = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, encoder: TextEncoder, output_dir: str | Path):
        self.faiss = FaissRetriever(encoder)
        self.bm25 = BM25Retriever(output_dir)
        self.exact = ExactAliasRetriever(output_dir)

    def query(self, mention: str, kb: str, top_k: int = 5) -> list[RetrievalResult]:
        if not mention:
            return []

        try:
            faiss_results = self.faiss.query(mention, kb, top_k=top_k)
        except Exception as exc:  # pragma: no cover - defensive fallback
            log.warning("FAISS retrieval failed for %s: %s", kb, exc)
            faiss_results = []

        try:
            bm25_results = self.bm25.query(mention, kb, top_k=top_k)
        except Exception as exc:  # pragma: no cover - defensive fallback
            log.warning("BM25 retrieval failed for %s: %s", kb, exc)
            bm25_results = []

        try:
            exact_results = self.exact.query(mention, kb, top_k=top_k)
        except Exception as exc:  # pragma: no cover - defensive fallback
            log.warning("Exact alias retrieval failed for %s: %s", kb, exc)
            exact_results = []

        combined: dict[str, RetrievalResult] = {}

        def add_result(results: list[RetrievalResult], weight: float) -> None:
            for result in results:
                # entry = combined[result.id]
                entry = combined.setdefault( # Keeps first object inserted
                    result.id,
                    RetrievalResult(
                        id=result.id,
                        name=result.name,
                        score=0.0,
                        tty=result.tty,
                    ),
                )
                entry.score += result.score * weight

        add_result(faiss_results, 0.5)
        add_result(bm25_results, 0.3)
        add_result(exact_results, 0.2)

        ranked = sorted(combined.values(), key=lambda item: item.score, reverse=True)
        return ranked[:top_k]
