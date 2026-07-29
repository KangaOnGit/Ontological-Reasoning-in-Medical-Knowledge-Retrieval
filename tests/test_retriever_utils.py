import pytest

from src.rag.retriever.utils import clean_mention


def test_basic_normalization():
    # lowercasing and whitespace collapse
    assert clean_mention("  A  B   C ", "ICD") == "a b c"
    assert clean_mention("  Mixed CASE   and   SPACES ", "ICD") == "mixed case and spaces"


def test_rxnorm_decimal_drug_and_unit_handling():
    # decimal comma -> dot, drug tokens removed, unit handling (current implementation)
    inp = "Take 5,5mg PO qd"
    out = clean_mention(inp, "RXNORM")
    # current implementation normalizes to lowercase, converts 5,5 -> 5.5
    # and removes 'PO' and 'qd', yielding 'take 5.5mg'
    assert out == "take 5.5mg"


def test_decimal_only():
    assert clean_mention("1,5", "RXNORM") == "1.5"
