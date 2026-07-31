from __future__ import annotations

import argparse
import logging

from src.inference.pipeline import InferencePipeline
from src.utils.config import load_config

CONFIG_NER = load_config("configs/NER.yaml")
CONFIG_SUBMISSION = load_config("configs/submission.yaml")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the medical inference pipeline"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default=CONFIG_SUBMISSION["data"]["eval"]["path"],
        help="Directory containing input .txt files",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=CONFIG_SUBMISSION["output"]["path"],
        help="Directory that will receive the submission ZIP",
    )

    parser.add_argument(
        "--ner_model",
        type=str,
        default=CONFIG_NER["model"]["default"],
        choices=CONFIG_NER["LLM"].keys(),
        help="NER model defined in configs/NER.yaml",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = InferencePipeline(
        ner_model=args.ner_model,
    )

    results = pipeline.run_submission(
        input_dir=args.input_dir,
        output_dir=args.output_dir,)
    
    log.info(
        "Finished inference pipeline with %d files", len(results),
        )


if __name__ == "__main__":
    main()