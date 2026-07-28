import re
from pathlib import Path
from pprint import pprint

SECTION_RE = re.compile(r'^\s*\d+\.\s*(.+)$')
ITEM_RE = re.compile(r'^\s*-\s+')


def classify(line):
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

def add_result(results, text, section, subsection):
    """Append a parsed text span."""
    results.append({
        "text": text.strip(),
        "path": [section, subsection]
    })


def next_nonempty_type(lines, start):
    """Return the type of the next non-empty line."""
    for j in range(start + 1, len(lines)):
        nxt = lines[j].strip()
        if nxt:
            return classify(nxt)
    return None


def parse(filename):

    with open(filename, encoding="utf8") as f:
        lines = [l.rstrip() for l in f]

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
                    current_subsection
                )

        elif typ == "ITEM":

            add_result(
                results,
                line.lstrip("-"),
                current_section,
                current_subsection
            )

        else:

            next_type = next_nonempty_type(lines, i)

            if next_type == "ITEM":
                current_subsection = line
            else:
                add_result(
                    results,
                    line,
                    current_section,
                    current_subsection
                )

    return results
