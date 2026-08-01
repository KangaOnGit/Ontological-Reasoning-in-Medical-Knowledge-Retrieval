from collections import defaultdict
from src.preprocess.base import Chunk, ParsedRecord

def build_chunks(results: list[ParsedRecord]) -> list[Chunk]:
    groups = defaultdict(list)

    for item in results:
        # group ParsedRecords with same path
        groups[tuple(item.path)].append(item)

    chunks: list[Chunk] = []

    for (section, subsection), records in groups.items():
        lines: list[str] = []
        record_offsets: list[int] = []

        curr_offset = 0
        
        if section:
            line = f"Section: {section}"
            lines.append(line)
            curr_offset += len(line) + 1  # account for '\n' inserted by "\n".join(...)

        if subsection:
            line = f"Subsection: {subsection}"
            lines.append(line)
            curr_offset += len(line) + 1  # account for '\n' inserted by "\n".join(...)

        if lines:
            lines.append("")
            curr_offset += 1  # blank line ("\n")

        for record in records:
            record_offsets.append(curr_offset)

            line = record.text
            lines.append(line)
            curr_offset += len(line) + 1  # newline

        chunks.append(
            Chunk(
                text="\n".join(lines),
                records=records,
                record_offsets=record_offsets,
            )
        )

    return chunks