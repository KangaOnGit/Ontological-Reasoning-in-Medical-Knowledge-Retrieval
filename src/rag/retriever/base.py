from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


class BaseRetriever(ABC):

    @abstractmethod
    def query(
        self,
        mention: str,
        kb: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        ...
        
    @abstractmethod
    def get_index(
        self,
        kb: str,
    ):
        ...
        
@dataclass(slots=True)
class RetrievalResult:
    id: str
    name: str
    score: float
    tty: str = ""