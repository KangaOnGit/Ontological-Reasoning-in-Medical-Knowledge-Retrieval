from src.NER.base import Span
from src.preprocess.base import Chunk

def locate_span_position(span: Span, chunk: Chunk) -> list[int]:
    """Locate the span text in the original file content and return [start, end]."""
    
    if span.end <= span.start:
        return []

    for record, chunk_start in zip(
        chunk.records,
        chunk.record_offsets,
    ):
        chunk_end = chunk_start + len(record["text"])

        if (
            span.start >= chunk_start
            and span.end <= chunk_end
        ):
            local_start = span.start - chunk_start
            local_end = span.end - chunk_start

            return [
                record["start"] + local_start,
                record["start"] + local_end,
            ]

    return []