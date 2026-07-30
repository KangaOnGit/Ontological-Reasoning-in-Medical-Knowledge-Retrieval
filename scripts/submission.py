from __future__ import annotations

import argparse
from argparse import BooleanOptionalAction

import logging

from src.utils.config import load_config
from src.inference.pipeline import run_inference

CONFIG_NER = load_config("configs/NER.yaml")
CONFIG_SUBMISSION = load_config("configs/submission.yaml")
ENTITY_TO_KB = {
    "CHẨN_ĐOÁN": "ICD10",
    "THUỐC": "RXNorm",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the medical inference pipeline")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=CONFIG_SUBMISSION["data"]["eval"]["path"],
        help="Directory containing input .txt files",
    )
    parser.add_argument(
        "--ner_model",
        type=str,
        default=CONFIG_NER["model"]["name"][0],
        choices=CONFIG_NER["LLM"].keys(),
        help="Model alias defined in configs/NER.yaml",
    )
    
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="configs/prompt/span_extraction.jinja",
        help="Path to the Jinja prompt template",
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
    parser.add_argument(
        "--output_dir",
        type=str,
        default=CONFIG_SUBMISSION["output"]["path"],
        help="Directory that will receive per-file zip archives",
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_inference(args)
    log.info("Finished inference pipeline with %d extracted spans", len(results))


if __name__ == "__main__":
    main()
