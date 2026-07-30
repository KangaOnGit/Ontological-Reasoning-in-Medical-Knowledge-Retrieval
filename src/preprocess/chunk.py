from collections import defaultdict
from src.preprocess.base import Chunk

def build_chunks(results: list[dict]) -> list[Chunk]:
    groups = defaultdict(list)

    for item in results:
        groups[tuple(item["path"])].append(item)

    chunks = []

    for (section, subsection), records in groups.items():
        lines = []

        if section:
            lines.append(f"Section: {section}")

        if subsection:
            lines.append(f"Subsection: {subsection}")

        if lines:
            lines.append("")

        lines.extend(record["text"] for record in records)

        chunks.append(
            Chunk(
                text="\n".join(lines),
                records=records,
            )
        )

    return chunks