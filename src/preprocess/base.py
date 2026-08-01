from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    records: list[dict]
    record_offsets: list[int]