import argparse
import logging
from pathlib import Path

from src.utils.config import load_config
from rag.indexing.faiss_indexing import build
from rag.encoders.text_encoder import TextEncoder

CONFIG = load_config(r"configs/KB/indexing.yaml")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build FAISS indexes for all knowledge bases."
    )

    parser.add_argument(
        "--encoder",
        default=CONFIG["model"]["name"],
        choices=CONFIG["encoders"].keys(),
        help="Encoder to use."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only build the first N entries (for debugging)."
    )
        
    return parser.parse_args()


def main():
    args = parse_args()

    model_name = CONFIG["encoders"][args.encoder]["link"]
    encoder = TextEncoder(model_name=model_name)
    
    sources = CONFIG["input"]["sources"]

    for kb, parquet_path in sources.items():
        logging.info("Building %s...", kb)

        build(
            kb=kb,
            encoder=encoder,
            tag=kb,
            parquet_path=Path(parquet_path),
            limit = args.limit,
        )

    logging.info("Finished building all knowledge base indexes.")


if __name__ == "__main__":
    main()
    
# python -m scripts.build_rag.build_index.build_faiss_index [--args]