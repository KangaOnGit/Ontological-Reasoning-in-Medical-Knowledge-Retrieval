import logging
from typing import Iterable

from src.rag.indexing.faiss_indexing import KBIndex
from src.rag.retrieval_methods.utils import clean_mention
from src.rag.retrieval_methods.base_retriever import BaseRetriever

log = logging.getLogger(__name__)

class FaissRetriever(BaseRetriever):

    def __init__(self, encoder):
        self.encoder = encoder
        self.indices = {}

    def get_index(self, kb):
        if kb not in self.indices:
            log.info("Loading %s...", kb)
            self.indices[kb] = KBIndex.load(kb, self.encoder)
        return self.indices[kb]

    def query(self, mention, kb, top_k=5) -> List[Tuple[str, str, float, str]]:
        """
        Return: [(ID, Name, Scores, TTY),...]
        """
        
        mention = clean_mention(mention, self.kb)
        return self.get_index(kb).query(mention, top_k)