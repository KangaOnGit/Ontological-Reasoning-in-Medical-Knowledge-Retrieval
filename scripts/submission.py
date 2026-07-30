import argparse
import logging
from pathlib import Path
import os

from __future__ import annotations

# Model
from src.NER.model import NERmodel
from src.NER.base import Span

# Configs
from src.utils.config import load_config

# Preprocess
from src.preprocess.parse import parse
from src.preprocess.chunk import build_chunks

CONFIG_NER = load_config("configs/NER.yaml")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Build Medical NER model")

    parser.add_argument(
        "--model",
        type=str,
        default=CONFIG_NER["model"]["name"][0],
        choices=CONFIG_NER["LLM"].keys(),
        help="Model alias defined in configs/NER.yaml",
    )

    parser.add_argument(
        "--prompt_path",
        type=str,
        default="configs/prompt/span_extraction.jinja",
        help="Path to Jinja prompt template",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.05,
    )

    return parser.parse_args()
        

def main():
    args = parse_args()

    ner_model = NERmodel(
        model_name=CONFIG_NER["LLM"][args.model]["link"],
        prompt_path=args.prompt_path,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    inference(ner_model = ner_model)
    
    log.info("Finished building Medical NER model.")



CONFIG_INFER = load_config(r"configs/infer.yaml")

def inference(
    ner_model,
    cls_model,
    data_dir: Path = Path(CONFIG_INFER['data']['eval']['path']),
):
    files = sorted(data_dir.glob("*.txt"))

    for i, f in enumerate(os.listdir(data_dir)):
        parsed_f = parse(os.path.join(data_dir, f))
        chunks = build_chunks(parsed_f)
        
        Span_list = ner_model.forward(chunks)
        
        for sp in Span_list:
            
    return

if __name__ == "__main__":
    main()