from pathlib import Path

from src.preprocess.chunk import build_chunks
from src.preprocess.parser import parse


def print_structure(results):
    """Pretty-print the parsed document structure."""
    last_section = object()
    last_subsection = object()

    for item in results:
        section, subsection = item["path"]

        if section != last_section:
            print(f"\n=== {section} ===")
            last_section = section
            last_subsection = object()

        if subsection != last_subsection:
            if subsection is not None:
                print(f"\n  [{subsection}]")
            last_subsection = subsection

        print(f"    • {item['text']}")


def test_file():
    print("=" * 80)
    print("PARSER TEST (FILE)")
    print("=" * 80)

    file = Path("data/Round 1/P2/6.txt")

    results = parse(filename=file)

    print_structure(results)

    print("\n" + "=" * 80)
    print("CHUNK TEST")
    print("=" * 80)

    chunks = build_chunks(results)

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n{'@' * 80}")
        print(f"Chunk {i}/{len(chunks)}")
        print("-" * 80)
        print(chunk)


def test_text():
    print("\n" + "=" * 80)
    print("PARSER TEST (TEXT)")
    print("=" * 80)

    text = """
    1. Khám bệnh

    Triệu chứng
    - Đau ngực
    - Khó thở

    Tiền sử:
    Tăng huyết áp nhiều năm.
    """

    results = parse(text=text)

    print_structure(results)


if __name__ == "__main__":
    try:
        test_file()
        test_text()

    except Exception as e:
        print(f"Error: {e}")