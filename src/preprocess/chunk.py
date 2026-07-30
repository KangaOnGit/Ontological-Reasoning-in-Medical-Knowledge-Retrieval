from collections import defaultdict


def build_chunks(results: list[dict]) -> list[str]:
    """Group parsed text by (section, subsection) into LLM-ready chunks."""

    groups = defaultdict(list)

    for item in results:
        groups[tuple(item["path"])].append(item["text"])

    chunks = []

    for (section, subsection), texts in groups.items():
        lines = []

        if section:
            lines.append(f"Section: {section}")

        if subsection:
            lines.append(f"Subsection: {subsection}")

        if lines:
            lines.append("")

        lines.extend(texts)

        chunks.append("\n".join(lines))

    return chunks