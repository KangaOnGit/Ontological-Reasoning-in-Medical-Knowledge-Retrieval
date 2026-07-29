from abc import ABC, abstractmethod


class BaseRetriever(ABC):

    @abstractmethod
    def query(
        self,
        mention: str,
        kb: str,
        top_k: int = 5,
    ):
        ...