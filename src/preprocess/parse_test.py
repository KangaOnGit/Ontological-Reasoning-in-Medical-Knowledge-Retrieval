from pathlib import Path

from postprocess.searcher import search
from parser import parse
from chunker import chunk


def print_structure(results):
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

if __name__ == "__main__":
    try:
        
        print("="*80)
        print(f"!!! PARSER TEST !!!")
        
        results = parse(r"data\1st Round Data P2\input P2\40.txt")
        print_structure(results)
        
        print("="*80)
        print(f"!!! SEARCHER TEST !!!")
        match_phrase = search.trace_phrase(results, "01 lần 15")
        print(match_phrase)
    
        print("-"*80)
        match_word = search.trace_word(results, "aspirin")
        print(match_word)
        
        print("="*80)
        print(f"!!! CHUNKER TEST !!!")
        chunks = chunk(results)
        for chunk in chunks:
            print("@" * 80)
            print(chunk)
        print("-"*20)
        print(chunks[0])
    except Exception as e:
        print(f"Error: {e}")