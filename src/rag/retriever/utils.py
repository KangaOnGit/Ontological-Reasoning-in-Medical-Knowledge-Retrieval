import re
import unicodedata

DRUG = re.compile(
    r"\b(po|iv|im|sc|sl|pr|tid|bid|qd|qid|qhs|qam|qpm|prn|q\d+h(:prn)?|daily|twice|once|"
    r"uống|tiêm|truyền|lần|ngày|viên|tab|caps?)\b",
    re.IGNORECASE,
)

DECIMAL_COMMA = re.compile(r"(\d),(\d)")
UNIT_SPACING = re.compile(
    r"\s*(mg|mcg|µg|ug|g|ml|iu|meq|mmol)\b",
    re.IGNORECASE,
)


def clean_mention(mention: str, kb: str) -> str:
    mention = unicodedata.normalize("NFKC", str(mention))
    mention = mention.lower()
    mention = " ".join(mention.split())

    kb = kb.upper()

    if kb == "RXNORM":
        mention = DRUG.sub(" ", mention)
        mention = DECIMAL_COMMA.sub(r"\1.\2", mention)
        mention = UNIT_SPACING.sub(r"\1 \2", mention)

    return " ".join(mention.split())