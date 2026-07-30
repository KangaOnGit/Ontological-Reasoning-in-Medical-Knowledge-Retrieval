from __future__ import annotations
import re
from src.utils.config import load_config
from src.NER.base import Span

def normalize_rule_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

ASSERTION_RULES = {
    label: [normalize_rule_text(p) for p in phrases]
    for label, phrases in load_config("configs/assertion.yaml")["labels"].items()
}

def rule_based_assertion(span: Span) -> list[str]:
    """Infer assertion labels from subsection first, then section/context."""
    candidates = []
    
    haystack = normalize_rule_text(
        " ".join(
            part for part in (span.subsection, span.section, span.context, span.text)
            if part
        )
    )

    for label, phrases in ASSERTION_RULES.items():
        if any(phrase in haystack for phrase in phrases):
            candidates.append(label)

    return candidates