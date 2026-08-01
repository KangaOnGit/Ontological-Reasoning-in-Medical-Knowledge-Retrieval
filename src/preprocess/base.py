from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    records: list[ParsedRecord]
    record_offsets: list[int]
    
@dataclass
class ParsedRecord:
    text: str
    path: list[str | None]
    start: int
    end: int