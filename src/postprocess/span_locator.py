from src.NER.base import Span

def locate_span_position(span: Span, records: list[dict]) -> list[int]:
    """Locate the span text in the original file content and return [start, end]."""
    needle = span.text.strip()

    for record in records:
        idx = record["text"].find(needle)

        if idx != -1:
            start = record["start"] + idx
            return [start, start + len(needle)]

    return []