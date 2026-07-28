from collections import defaultdict

def chunk(results):
    """
    Build one LLM-ready string for each (section, subsection).

    Returns:
        List[str]
    """

    groups = defaultdict(list)

    # Group by (section, subsection)
    for item in results:
        groups[tuple(item["path"])].append(item["text"])

    chunks = []

    for (section, subsection), texts in groups.items():

        chunk = []

        if section:
            chunk.append(f"Section: {section}")

        if subsection:
            chunk.append(f"Subsection: {subsection}")

        chunk.append("")  # blank line
        chunk.extend(texts)

        chunks.append("\n".join(chunk))

    return chunks