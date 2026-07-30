from dataclasses import dataclass

@dataclass
class Span:
    text: str
    section: str
    subsection: str
    context: str