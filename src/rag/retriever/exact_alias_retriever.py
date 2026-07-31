from pathlib import Path
from collections import defaultdict

import pandas as pd
import logging

from src.utils.config import load_config
from src.rag.retriever.utils import clean_mention
from src.rag.retriever.base import BaseRetriever, RetrievalResult

log = logging.getLogger(__name__)

CONFIG = load_config(r"configs/RAG/indexing/faiss_indexing.yaml")

class ExactAliasRetriever(BaseRetriever):

    def __init__(
        self,
        output: Path | str = CONFIG["output"]["path"],
    ):
        self.cache: dict[str, dict[str, list[RetrievalResult]]] = {}
        self.output = Path(output)

    def get_kb(
        self,
        kb: str,
    ) -> defaultdict[str, list[RetrievalResult]]:
        """
        self.cache = {
            "ICD": {
                "alias1": [RetrievalResults (1), RetrievalResults (2),...]
                ...
                },
                
            "RXNorm": {...},
            }
            
                (an alias can have multiple results due to normalization/cleaning)
        """
        if kb not in self.cache:
            metadata_path = self.output / kb / f"{kb}_metadata.parquet"

            if not metadata_path.exists():
                raise ValueError(
                    f"{kb}: Missing '{metadata_path.name} / {metadata_path}'"
                )
            metadata = pd.read_parquet(metadata_path)
            alias_dict = defaultdict(list)
            has_tty = "tty" in metadata.columns
            for row in metadata.itertuples(index=False):
                alias = clean_mention(row.name, kb)
                
                alias_dict[alias].append(
                    RetrievalResult(
                        id=str(row.id),
                        name=str(row.name),
                        score=1.0,
                        tty=str(row.tty) if has_tty else "",
                    )
                )

            self.cache[kb] = alias_dict

        return self.cache[kb]

    def query(
        self,
        mention: str,
        kb: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:

        query = clean_mention(mention, kb)
        return self.get_kb(kb).get(query, [])[:top_k]