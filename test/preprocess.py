from pathlib import Path

from src.preprocess.chunk import build_chunks
from src.preprocess.parse import parse


def print_structure(results, raw_text: str):
    """Pretty-print parsed document with offsets."""

    last_section = object()
    last_subsection = object()

    for item in results:
        section, subsection = item["path"]

        if section != last_section:
            print(f"\n=== {section} ===")
            last_section = section
            last_subsection = object()

        if subsection != last_subsection:
            if subsection:
                print(f"\n  [{subsection}]")
            last_subsection = subsection

        start = item["start"]
        end = item["end"]

        recovered = raw_text[start:end]

        ok = "✓" if recovered == item["text"] else "✗"

        print(
            f"{ok} [{start}, {end}] "
            f"{item['text']!r}"
        )

        if recovered != item["text"]:
            print(f"recovered: {recovered!r}")

def test_file():
    print("=" * 80)
    print("PARSER TEST")
    print("=" * 80)

    file = Path("data/Round 1/P2/6.txt")

    raw_text = file.read_text(encoding="utf-8")
    results = parse(filename=file)

    print_structure(results, raw_text)

    print("\n" + "=" * 80)
    print("CHUNK TEST")
    print("=" * 80)

    chunks = build_chunks(results)

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n{'@'*80}")
        print(f"Chunk {i}/{len(chunks)}")
        print("-"*80)
        print(chunk.text)


def test_text():
    print("\n" + "=" * 80)
    print("TEXT TEST")
    print("=" * 80)

    raw_text = """
    1. Khám bệnh

    Triệu chứng
    - Đau ngực
    - Khó thở

    Tiền sử:
    Tăng huyết áp nhiều năm.
    """

    results = parse(text=raw_text)

    print_structure(results, raw_text)


if __name__ == "__main__":
    try:
        test_file()
        test_text()

    except Exception as e:
        print(f"Error: {e}")