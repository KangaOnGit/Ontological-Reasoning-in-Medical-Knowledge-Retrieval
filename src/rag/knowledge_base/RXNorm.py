import csv as csv
import pandas as pd

from pathlib import Path
from src.utils.config import load_config

CONFIG = load_config(r"configs/RAG/kb_sources/RXNorm.yaml")


def default_raw():
    """
    Preferably preprocess .RRF file else .csv
    """
    return (
        CONFIG["input"]["path_RRF"]
        if CONFIG["input"]["path_RRF"]
        else CONFIG["input"]["path_CSV"]
    )

def read_raw(path: Path) -> pd.DataFrame:
    """
    RRF Format:
        38|ENG||||||829|829|38||RXNORM|BN|38|Parlodel||N|4096|
        44|ENG||||||12251526|12251526|44||RXNORM|IN|44|mesna||N|4096|
        
    Columns separated by "|"
    Last Column is always empty, separated by "|"
        Splits into: 38, ENG, , , , , , 829, 829, 38, RXNORM, BN, 38, Parlodel, , N, 4096, (empty)
    Some may contains quotes -> QUOTE_NONE
    Have trailing 0's -> make them into strings
    
    If not .RRF -> .csv
        Just need to read the csv
    """
    if path.suffix.lower() == ".rrf":
        df = pd.read_csv(
            path, sep="|", header=None, names=CONFIG["data"]["keep_col"] + ["_"],
            dtype=str, keep_default_na=False, encoding="utf-8",
            quoting=csv.QUOTE_NONE, engine="c", on_bad_lines="skip",
        )
        return df[CONFIG["data"]["keep_col"]] # Keep the extra empty column
    
    # CSV form (has header, possibly a UTF-8 BOM on the first column name)
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def build(raw: Path = None,
             out: Path = CONFIG["output"]["path"],
             synonyms: bool = True) -> pd.DataFrame:
    raw = raw or default_raw()
    df = read_raw(raw)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Column aliases / defaults so the lighter RxNorm Current Prescribable Content
    # layout also parses. The full RXNCONSO has every column, so this is a no-op
    # for the release used to produce the submitted results.
    if "str" not in df.columns and "name" in df.columns:
        df = df.rename(columns={"name": "str"})
    if "rxcui" not in df.columns and "rxnorm_cui" in df.columns:
        df = df.rename(columns={"rxnorm_cui": "rxcui"})
    for col, default in (("lat", "ENG"), ("sab", "RXNORM"), ("suppress", "")):
        if col not in df.columns:
            df[col] = default

    base = (
        (df["lat"] == "ENG")
        & df["sab"].isin(CONFIG["data"]["keep_sab"])
        & (df["suppress"].str.upper() != "Y")
    )
    kept = df.loc[base & df["tty"].isin(CONFIG["data"]["keep_tty"]),
                  ["rxcui", "str", "tty", "sab"]].rename(columns={"str": "name"})
    kept["alias_source"] = "RXNORM:" + kept["tty"]
    kept["is_synonym"] = False
    kept["original_name"] = kept["name"]

    if synonyms:
        rxcui2tty = dict(zip(kept["rxcui"], kept["tty"]))
        syn_mask = base & df["tty"].isin(CONFIG["data"]["syn_tty"]) & df["rxcui"].isin(rxcui2tty)
        syn = df.loc[syn_mask, ["rxcui", "str", "sab"]].rename(columns={"str": "name"})
        syn["tty"] = syn["rxcui"].map(rxcui2tty)
        syn["alias_source"] = "RXNORM:SYN"
        syn["is_synonym"] = True
        syn["original_name"] = syn["name"]
        kept = pd.concat([kept, syn], ignore_index=True)

    kept["name"] = kept["name"].str.strip()
    kept = kept[kept["name"].str.len() > 0]
    kept = kept.drop_duplicates(subset=["rxcui", "name", "tty"]).reset_index(drop=True)
    kept = kept.rename(columns={"rxcui": "code"})[["code", "name", "tty", "alias_source",
                                                   "is_synonym", "original_name"]]
    out.parent.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(out, index=False)
    print(f"RXNorm: {len(kept):,} aliases / {kept['code'].nunique():,} rxcui/code -> {out}")
    print(kept["tty"].value_counts().to_string())
    return kept