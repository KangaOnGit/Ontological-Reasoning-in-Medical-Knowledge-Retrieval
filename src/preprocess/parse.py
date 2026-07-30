import re
from pathlib import Path

SECTION_RE = re.compile(r"^\s*\d+\.\s*(.+)$")
ITEM_RE = re.compile(r"^\s*-\s+")


def classify(line: str) -> str:
    """Classify a line in the clinical note."""
    line = line.strip()

    if not line:
        return "EMPTY"

    if SECTION_RE.match(line):
        return "SECTION"

    if ITEM_RE.match(line):
        return "ITEM"

    if ":" in line:
        return "SUBSECTION"

    return "TEXT"


def add_result(
    results: list[dict],
    text: str,
    section: str | None,
    subsection: str | None,
) -> None:
    """Append a parsed text span."""
    results.append(
        {
            "text": text.strip(),
            "path": [section, subsection],
        }
    )


def next_nonempty_type(lines: list[str], start: int) -> str | None:
    """Return the type of the next non-empty line."""
    for line in lines[start + 1 :]:
        line = line.strip()
        if line:
            return classify(line)
    return None


def parse_lines(lines: list[str]) -> list[dict]:
    """Parse already-split lines."""

    current_section = None
    current_subsection = None
    results = []

    for i, raw in enumerate(lines):
        line = raw.strip()

        if not line:
            continue

        typ = classify(line)

        if typ == "SECTION":
            current_section = line
            current_subsection = None

        elif typ == "SUBSECTION":
            title, *rest = line.split(":", 1)
            current_subsection = title.strip()

            if rest and rest[0].strip():
                add_result(
                    results,
                    rest[0],
                    current_section,
                    current_subsection,
                )

        elif typ == "ITEM":
            add_result(
                results,
                line.lstrip("-").strip(),
                current_section,
                current_subsection,
            )

        else:
            if next_nonempty_type(lines, i) == "ITEM":
                current_subsection = line
            else:
                add_result(
                    results,
                    line,
                    current_section,
                    current_subsection,
                )

    return results

def parse(
    filename: str | Path | None = None,
    text: str | None = None,
) -> list[dict]:
    """
    Parse either a text file or a raw text string.

    Exactly one of `filename` or `text` must be provided.
    """
    if (filename is None) == (text is None):
        raise ValueError("Provide exactly one of 'filename' or 'text'.")

    if filename is not None:
        lines = Path(filename).read_text(encoding="utf-8").splitlines()
    else:
        lines = text.splitlines()

    return parse_lines(lines)