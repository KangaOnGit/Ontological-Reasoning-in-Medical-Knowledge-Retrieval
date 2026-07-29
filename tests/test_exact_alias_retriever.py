import pandas as pd
from pathlib import Path

from src.rag.retriever.exact_alias_retriever import ExactAliasRetriever


def test_exact_alias_get_index_and_query(tmp_path, monkeypatch):
    kb = "ICD"
    # create fake metadata parquet file path to satisfy exists()
    out_dir = tmp_path
    kb_dir = out_dir / kb
    kb_dir.mkdir()
    metadata_path = kb_dir / f"{kb}_metadata.parquet"
    metadata_path.write_text("")

    # create a pandas DataFrame resembling metadata
    df = pd.DataFrame([
        {"id": 1, "name": "Headache", "tty": "PT"},
        {"id": 2, "name": "Severe Headache", "tty": "SY"},
    ])

    # monkeypatch read_parquet to return our DataFrame
    import src.rag.retriever.exact_alias_retriever as exact_mod

    monkeypatch.setattr(exact_mod.pd, "read_parquet", lambda path: df)

    retriever = ExactAliasRetriever(output=out_dir)

    index = retriever.get_index(kb)
    # cleaned alias for 'Headache' should be present
    assert "headache" in index

    # query should return RetrievalResult entries
    results = retriever.query("Headache", kb, top_k=5)
    assert len(results) >= 1
    assert results[0].name == "Headache"
