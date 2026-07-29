from pathlib import Path

import pandas as pd
import logging
from rank_bm25 import BM25Okapi

from src.utils.config import load_config
from src.rag.retriever.utils import clean_mention
from src.rag.retriever.base import BaseRetriever, RetrievalResult

CONFIG = load_config(r"configs/RAG/indexing/faiss_indexing.yaml")

log = logging.getLogger(__name__)


class BM25Retriever(BaseRetriever):
    """BM25 index over KB aliases."""

    def __init__(
        self,
        output: Path | str = CONFIG["output"]["path"],
    ):
        self.output = Path(output)
        self.cache: dict[str, BM25Okapi] = {}
        self.metadata: dict[str, pd.DataFrame] = {}
        
    def get_index(self,
                  kb: str) -> BM25Okapi:
        if kb not in self.cache:
            log.info("Building BM25 for %s...", kb)
            
            metadata_path = self.output / kb / f"{kb}_metadata.parquet"
            if not metadata_path.exists():
                raise ValueError(
                    f"{kb}: Missing '{metadata_path.name} / {metadata_path}' — the index has no trusted "
                    "metadata."
                )
            self.metadata[kb] = pd.read_parquet(metadata_path)
            documents = [
                clean_mention(str(name), kb).split()
                for name in self.metadata[kb]["name"]]
            
            log.info("Built BM25 for %s...", kb)
            
            self.cache[kb] = BM25Okapi(documents)
            
        return self.cache[kb]
    
    def query(
        self,
        mention: str,
        kb: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:

        tokens = clean_mention(mention, kb).split()
        scores = self.get_index(kb).get_scores(tokens)
        top_idx = scores.argsort()[::-1][:top_k]
        has_tty = "tty" in self.metadata[kb].columns
        
        results = []
        for i in top_idx:
            if scores[i] <= 0:
                continue
            row = self.metadata[kb].iloc[int(i)]
            results.append(
                RetrievalResult(
                    id=str(row["id"]),
                    name=str(row["name"]),
                    score=float(scores[i]),
                    tty=str(row["tty"]) if has_tty else "",
                )
            )

        return results