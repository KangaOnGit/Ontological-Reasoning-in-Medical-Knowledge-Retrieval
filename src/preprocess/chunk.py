from collections import defaultdict
from src.preprocess.base import Chunk

def build_chunks(results: list[dict]) -> list[Chunk]:
    groups = defaultdict(list)

    for item in results:
        groups[tuple(item["path"])].append(item)

    chunks: list[Chunk] = []

    for (section, subsection), records in groups.items():
        lines: list[str] = []
        record_offsets: list[int] = []

        if section:
            lines.append(f"Section: {section}")

        if subsection:
            lines.append(f"Subsection: {subsection}")

        if lines:
            lines.append("")

        for record in records:
            # Offset where this record begins in the chunk
            chunk_so_far = "\n".join(lines)
            record_offsets.append(
                len(chunk_so_far) + (1 if lines else 0)
            )

            lines.append(record["text"])

        chunks.append(
            Chunk(
                text="\n".join(lines),
                records=records,
                record_offsets=record_offsets,
            )
        )

    return chunks