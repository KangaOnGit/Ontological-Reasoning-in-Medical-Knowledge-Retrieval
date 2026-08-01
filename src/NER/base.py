from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class Span:
    text: str
    typ: str
    section: str = ""
    subsection: str =""
    context: str = ""
    start: int  | None = None
    end: int | None = None


class BaseNER(ABC):
    
    @abstractmethod
    def forward(self,
                ipt: str
                ) -> list[Span]:
        ...