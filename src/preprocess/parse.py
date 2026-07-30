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
    start: int,
    end: int,
) -> None:
    """Append a parsed text span."""
    results.append(
        {
            "text": text,
            "path": [section, subsection],
            "start": start,
            "end": end,
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
    """Parse already-split lines while preserving character offsets."""

    current_section = None
    current_subsection = None
    results = []

    offset = 0

    for i, raw in enumerate(lines):
        raw_no_newline = raw.rstrip("\r\n")
        line = raw_no_newline.strip()

        # Character offsets of the stripped text
        leading = len(raw_no_newline) - len(raw_no_newline.lstrip())
        trailing = len(raw_no_newline) - len(raw_no_newline.rstrip())

        start = offset + leading
        end = offset + len(raw_no_newline) - trailing

        offset += len(raw)

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
                value = rest[0].strip()

                colon_pos = raw_no_newline.find(":")
                value_start = offset - len(raw) + colon_pos + 1

                while (
                    value_start < offset
                    and raw_no_newline[value_start - (offset - len(raw))].isspace()
                ):
                    value_start += 1

                add_result(
                    results,
                    value,
                    current_section,
                    current_subsection,
                    value_start,
                    value_start + len(value),
                )

        elif typ == "ITEM":
            item = line.lstrip("-").strip()

            item_start = raw_no_newline.find(item) + (offset - len(raw))

            add_result(
                results,
                item,
                current_section,
                current_subsection,
                item_start,
                item_start + len(item),
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
                    start,
                    end,
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
        text = Path(filename).read_text(encoding="utf-8")

    lines = text.splitlines(keepends=True)

    return parse_lines(lines)