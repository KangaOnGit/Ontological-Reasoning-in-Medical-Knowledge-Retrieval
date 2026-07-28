import json
import logging

from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import pandas as pd

from src.utils.config import load_config
from src.rag.knowledge_base.text_encoder import TextEncoder

CONFIG = load_config(r"configs/KB/indexing.yaml")

log = logging.getLogger(__name__)

class KBIndex:
    """A FAISS inner-product index over one KB's term names."""

    def __init__(self, kb: str, index,
                 metadata: pd.DataFrame,
                 encoder: Optional[TextEncoder] = None):
        self.kb = kb
        self.index = index
        self.metadata = metadata  # columns: id, name
        self.encoder = encoder

    def save(self, name: str, encoder_meta: dict,
                   output: Path | str = CONFIG["output"]["path"],
                   ) -> None:
        """
        Write:
            [name].faiss (Vector)
            [name]_meta.parquet (Metadata)
            [name].index.json (Index Information)
            
        [encoder_meta]: Model / pooling / query_format /
                    embedding_dim / kb_parquet_sha256 / number_of_vectors
                        for load-time checks.
        """
        output = Path(output) / name
        
        output.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(output / f"{name}.faiss"))
        self.metadata.to_parquet(output / f"{name}_metadata.parquet", index=False)
        
        meta = dict(encoder_meta)
        meta.setdefault("embedding_dim", self.index.d)
        meta.setdefault("number_of_vectors", int(self.index.ntotal))
        meta.setdefault("meta_rows", int(len(self.metadata)))
        meta["kb"] = self.kb
        (output / f"{name}.index.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=4),
            encoding="utf-8")

    @staticmethod
    def load(name: str, encoder, output: Path | str = CONFIG["output"]["path"],
                   check_dim: bool = True) -> "KBIndex":
        """
        Load the KB and Validate it.
        Returns KBIndex Class
        """
        output = Path(output)

        faiss_path = output / f"{name}.faiss"
        metadata_path = output / f"{name}_metadata.parquet"
        idx_metadata_path = output / f"{name}.index.json"
        if not idx_metadata_path.exists():
            raise ValueError(
                f"{name}: Missing '{idx_metadata_path.name}' — the index has no trusted "
                "metadata."
            )
        info = json.loads(idx_metadata_path.read_text())
        index = faiss.read_index(str(faiss_path))
        metadata = pd.read_parquet(metadata_path)

        enc_dim = getattr(getattr(encoder, "model", None), "config", None)
        enc_dim = getattr(enc_dim, "hidden_size", None)
        checks = []
        
        # Validate
        if check_dim:
            if enc_dim is not None and enc_dim != index.d:
                checks.append(f"encoder hidden_size={enc_dim} != index dim={index.d}")
            if info.get("embedding_dim") != index.d:
                checks.append(f"metadata dim {info.get('embedding_dim')} != index dim {index.d}")
            elif enc_dim is not None and info.get("embedding_dim") != enc_dim:
                checks.append(f"metadata dim {info.get('embedding_dim')} != encoder dim {enc_dim}")
        if info.get("number_of_vectors") != index.ntotal:
            checks.append(f"number_of_vectors={info.get('number_of_vectors')} != ntotal={index.ntotal}")
        if "meta_rows" in info and int(info["meta_rows"]) != len(metadata):
            checks.append(f"meta_rows={info['meta_rows']} != loaded {len(metadata)}")
        # source-parquet checksum (only if the recorded path is still present)
        src = info.get("kb_parquet")
        if src and Path(src).exists() and info.get("kb_parquet_sha256"):
            if sha256_file(Path(src)) != info["kb_parquet_sha256"]:
                checks.append("kb_parquet_sha256 mismatch (source parquet changed)")
        # encoder contract (pooling / query_format) if the encoder declares it
        for fld in ("pooling", "query_format"):
            enc_val = getattr(encoder, fld, None)
            if enc_val is not None and info.get(fld) is not None and info[fld] != enc_val:
                checks.append(f"{fld}: index='{info[fld]}' != encoder='{enc_val}'")
        if checks:
            raise ValueError(
                f"{name}: index integrity check failed:\n  - " + "\n  - ".join(checks)
                + "\nRebuild this index with the matching encoder (kb.build_v2)."
            )
        return KBIndex(info.get("kb", name), index, metadata, encoder)

    def query(self, mention: str, top_k: int = 10) -> List[Tuple[str, str, float, str]]:
        """
        Return: [(id, name, score, tty),...] best matches for a mention.
        """
        
        if self.encoder is None:
            raise RuntimeError("KBIndex has no encoder; construct/load with one")

        q = self.encoder.encode_queries([mention])
        scores, idx = self.index.search(q, min(top_k, self.index.ntotal))
        has_tty = "tty" in self.metadata.columns
        out = []
        for j, s in zip(idx[0], scores[0]):
            if j < 0:
                continue
            row = self.metadata.iloc[int(j)]
            tty = str(row["tty"]) if has_tty else ""
            out.append((str(row["id"]), str(row["name"]), float(s), tty))
        return out

def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(kb: str, encoder:TextEncoder,
          tag: str, parquet_path: Path,
          output: Path | str = CONFIG["output"]["path"],
          limit: int | None = None,) -> "KBIndex":
    """
    Build an index from any parquets and save it with metadata
    """
    
    df = pd.read_parquet(parquet_path)
    if limit is not None:
        df = df.head(limit)
        
    for candidate in ("id", "code", "rxcui"):
        if candidate in df.columns:
            id_col = candidate
            break
    else:
        raise ValueError(f"No id-like column found in parquet {parquet_path}."
                         "Expected one of ['id','code','rxcui']")
    keep = ["id", "name"]
    for c in [
        "tty",
        "alias_source",
        "is_synonym",
        "original_name"
    ]:
        if c in df.columns:
            keep.append(c)
            
    metadata = df.rename(columns={id_col: "id"})[keep].reset_index(drop=True)
    log.info("[%s/%s] embedding %d aliases", kb, tag, len(metadata))
    
    # Embed all of the names
    embedding = encoder.encode_documents(
        metadata["name"].tolist(),
        verbose=True,)
        
    index = faiss.IndexFlatIP(embedding.shape[1])
    index.add(embedding)
    ix = KBIndex(kb, index, metadata, encoder) # Wrapper
    
    enc_meta = {
        "encoder_model": encoder.model_name,
        "pooling": encoder.pooling,
        "query_format": encoder.query_format,
        "embedding_dim": encoder.embedding_dim,
        "kb_parquet": parquet_path.name,
        "kb_parquet_sha256": sha256_file(parquet_path),
        "number_of_vectors": int(index.ntotal),
    }
    ix.save(tag, enc_meta, output)
    log.info("[%s/%s] wrote %s.faiss (%d vectors)", kb, tag, tag, index.ntotal)
    return ix