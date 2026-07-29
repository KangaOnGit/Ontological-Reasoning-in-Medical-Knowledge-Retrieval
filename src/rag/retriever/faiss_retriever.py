import logging

from src.rag.indexing.faiss_indexing import KBIndex
from src.rag.retriever.utils import clean_mention
from src.rag.retriever.base import BaseRetriever, RetrievalResult

log = logging.getLogger(__name__)


class FaissRetriever(BaseRetriever):

    def __init__(self, encoder):
        self.encoder = encoder
        self.indices: dict[str, KBIndex] = {}

    def get_index(self, kb: str) -> KBIndex:
        if kb not in self.indices:
            log.info("Loading %s...", kb)
            self.indices[kb] = KBIndex.load(kb, self.encoder)
        return self.indices[kb]

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
        return self.get_index(kb).query(mention, top_k)