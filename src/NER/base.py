from dataclasses import dataclass

@dataclass
class Span:
    text: str
    type: str
    section: str
    subsection: str
    context: str