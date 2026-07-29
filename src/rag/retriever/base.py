from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass(slots=True)
class RetrievalResult:
    id: str
    name: str
    score: float
    tty: str = ""

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
    def get_kb(
        self,
        kb: str,
    ):
        ...