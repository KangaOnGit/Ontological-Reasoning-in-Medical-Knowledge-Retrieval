from pathlib import Path
from typing import List, Tuple

import pandas as pd
from rank_bm25 import BM25Okapi

from src.utils.config import load_config
from src.rag.retrieval_methods.utils import clean_mention

CONFIG = load_config(r"configs/RAG/indexing/faiss_indexing.yaml")


class BM25Index:
    """BM25 index over KB aliases."""

    def __init__(
        self,
        name: str,
        output: Path | str = CONFIG["output"]["path"],
    ):
        self.kb = name
        output = Path(output)

        metadata_path = output / name / f"{name}_metadata.parquet"
        self.metadata = pd.read_parquet(metadata_path)

        # Normalize every alias once
        self.documents = [
            clean_mention(str(x), self.kb)
            for x in self.metadata["name"]
        ]

        self.index = BM25Okapi(self.documents)

    def query(
        self,
        mention: str,
        top_k: int = 5,
    ) -> List[Tuple[str, str, float, str]]:
        """
        Return: [(ID, Name, Scores, TTY),...]
        """
        

        tokens = clean_mention(mention, self.kb)

        scores = self.index.get_scores(tokens)

        top_idx = scores.argsort()[::-1][:top_k]

        has_tty = "tty" in self.metadata.columns

        out = []

        for i in top_idx:
            row = self.metadata.iloc[int(i)]

            tty = str(row["tty"]) if has_tty else ""

            out.append(
                (
                    str(row["id"]),
                    str(row["name"]),
                    float(scores[i]),
                    tty,
                )
            )

        return out
    
# python -m src.rag.retrieval_methods.traditional.bm25