
import pandas as pd

from pathlib import Path
from src.utils.config import load_config

CONFIG = load_config(r"configs/RAG/kb_sources/ICD.yaml")

COVID_CODES = [
    ("U07.1", "COVID-19, vi rút được xác định", "COVID-19, virus identified"),
    ("U07.2", "COVID-19, vi rút không được xác định", "COVID-19, virus not identified"),
]


def dot_code(code: str) -> str:
    """Normalize a dotless ICD-10 code to dotted form (A001 -> A00.1)."""
    if pd.isna(code):
        return ""

    code = str(code).strip().upper().replace(".", "")
    return code[:3] + ("." + code[3:] if len(code) > 3 else "")


def default_raw() -> Path:
    path = Path(CONFIG["input"]["path"])

    if not path.is_file():
        raise FileNotFoundError(path)

    if path.suffix.lower() not in {".xlsx", ".xls"}:
        raise ValueError(f"Expected an Excel file, got {path.suffix}")

    return path


def clean_name(s: str) -> str:
    """Normalise punctuation/whitespace but preserve the original diacritics."""
    return " ".join(str(s).replace("・", " ").split()).strip(" ,;:")


def load(path: Path) -> pd.DataFrame:
    """
    Parse the TT06-2026 xlsx into long-form (code, name, alias_source) rows.

    Locates the header row by the ``MÃ BỆNH`` cell, then reads the four spec
    columns directly (robust to column reordering). The sheet carries both
    3-char category rows (MÃ BỆNH == MÃ NHÓM) and leaf rows; we emit a category
    alias from the category columns and a leaf alias from the leaf columns, so a
    short mention like ``bệnh tả`` hits category A00 while ``bệnh tả do ...``
    hits the leaf.
    """
    
    raw = pd.read_excel(path, sheet_name=0, header=None, dtype=str).fillna("")
    header_row = None
    for i in range(min(15, len(raw))):
        if any(str(x).strip().upper() == "MÃ BỆNH" for x in raw.iloc[i]):
            header_row = i
            break
    if header_row is None:
        raise ValueError(f"Could not find 'MÃ BỆNH' header in {path}")
    header = [str(x).strip() for x in raw.iloc[header_row]]
    body = raw.iloc[header_row + 1:].reset_index(drop=True)
    body.columns = header

    header_map = {c.strip().upper(): c for c in header}

    c_cat_code = header_map.get(CONFIG["data"]["cat_code"])
    c_cat_name = header_map.get(CONFIG["data"]["cat_name"])

    c_cat_code = header_map.get(CONFIG["data"]["cat_code"])
    c_cat_name = header_map.get(CONFIG["data"]["cat_name"])
    c_leaf_code = header_map.get(CONFIG["data"]["leaf_code"])
    c_leaf_name = header_map.get(CONFIG["data"]["leaf_name"])
    
    if c_leaf_code is None or c_leaf_name is None:
        raise ValueError(f"Missing MÃ BỆNH / TÊN BỆNH in {path}")

    rows = []
    seen = set()

    def add(code, name, src):
        if pd.isna(name):
            return

        name = clean_name(name)
        if not name:
            return

        code = dot_code(code)
        if len(code.replace(".", "")) < 3:
            return

        key = (code, name.lower())
        if key in seen:
            return

        seen.add(key)
        rows.append({
            "code": code,
            "name": name,
            "alias_source": src,
            "tty": "",
            "is_synonym": False,
            "original_name": name,
        })

    for _, r in body.iterrows():
        add(r[c_leaf_code], r[c_leaf_name], "TT06:LEAF")
        if c_cat_code is not None and c_cat_name is not None:
            add(r[c_cat_code], r[c_cat_name], "TT06:CAT3")

    # COVID (QĐ 98) — Vietnamese surface only
    for code, name_vi, _name_en in COVID_CODES:
        add(code, name_vi, "QD98-COVID:VI")
    return pd.DataFrame(rows)


def build(raw: Path | None = None) -> pd.DataFrame:
    """Build from official TT06-2026 excel file"""
    raw = raw or default_raw()

    df = load(raw).drop_duplicates(subset=["code", "name"]).reset_index(drop=True)
    raw.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(raw, index=False)
    print(f"ICD Terms: {len(df):,} aliases / {df['code'].nunique():,} codes -> {raw}")
    return df
