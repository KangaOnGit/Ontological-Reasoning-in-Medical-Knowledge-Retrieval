import logging

from src.rag.indexing.faiss_indexing import KBIndex
from src.rag.retriever.utils import clean_mention
from src.rag.retriever.base import BaseRetriever, RetrievalResult

log = logging.getLogger(__name__)


class FaissRetriever(BaseRetriever):

    def __init__(self, encoder):
        self.encoder = encoder
        self.cache: dict[str, KBIndex] = {}

    def get_kb(self, kb: str) -> KBIndex:
        """
        self.cache = {
            "ICD": KBIndex,
            "RXNorm": KBIndex,
            }
                (see faiss_indexing.py for info on the class)
        """
        if kb not in self.cache:
            self.cache[kb] = KBIndex.load(kb, self.encoder)
        return self.cache[kb]

    def query(
        self,
        mention: str,
        kb: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """
        Return the top-k FAISS retrieval results.
        """
        mention = clean_mention(mention, kb)
        return self.get_kb(kb).query(mention, top_k)