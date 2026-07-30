from dataclasses import dataclass

@dataclass
class Span:
    text: str
    typ: str
    section: str
    subsection: str
    context: str