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

        curr_offset = 0
        
        if section:
            line = f"Section: {section}"
            lines.append(line)
            curr_offset += len(line) + 1  # newline

        if subsection:
            line = f"Subsection: {subsection}"
            lines.append(line)
            curr_offset += len(line) + 1  # newline

        if lines:
            lines.append("")
            curr_offset += 1  # blank line ("\n")

        for record in records:
            record_offsets.append(current_offset)

            line = record["text"]
            lines.append(line)
            current_offset += len(line) + 1  # newline

        chunks.append(
            Chunk(
                text="\n".join(lines),
                records=records,
                record_offsets=record_offsets,
            )
        )

    return chunks