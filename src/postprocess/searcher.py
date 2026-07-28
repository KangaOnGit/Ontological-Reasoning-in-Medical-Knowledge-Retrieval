import re

class Searcher:
    
    def __init__(self):
        pass
    
    @staticmethod
    def _print_match(match):
        print("=" * 80)
        print(f"Match #{match['index']}")
        print(f"Section    : {match['section']}")
        print(f"Subsection : {match['subsection']}")
        print(f"Text       : {match['text']}")
    
    
    @staticmethod
    def _build_match(index, item):
        return {
            "index": index,
            "section": item["path"][0],
            "subsection": item["path"][1],
            "text": item["text"],
        }
        
    @staticmethod
    def trace_phrase(results, phrase, verbose=True):
        """
        Search for a phrase in the parsed results.
        
        Returns:
            [
                {
                    "section": ...,
                    "subsection": ...,
                    "text": ...,
                    "index": ...
                },
                ...
            ]
        """
        matches = []

        for i, item in enumerate(results):
            if phrase.lower() in item["text"].lower():
                match = Searcher._build_match(i, item)

                matches.append(match)

                if verbose:
                    Searcher._print_match(match)

        if not matches and verbose:
            print(f"No span containing '{phrase}' found.")
        return matches
    
    @staticmethod
    def trace_word(results, word, verbose=True):
        """
        Search for a standalone word in the parsed results.

        Returns:
            [
                {
                    "section": ...,
                    "subsection": ...,
                    "text": ...,
                    "index": ...
                },
                ...
            ]
        """
        word = word.lower()
        matches = []

        for i, item in enumerate(results):
            tokens = re.findall(r"[\w-]+", item["text"].lower(), flags=re.UNICODE)
            if word in tokens:
                match = Searcher._build_match(i, item)

                matches.append(match)

                if verbose:
                    Searcher._print_match(match)
                    
        if not matches and verbose:
            print(f"No standalone word '{word}' found.")
            
        return matches
    
        
search = Searcher()