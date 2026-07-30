import argparse
import logging

from src.NER.model import NERmodel
from src.utils.config import load_config

CONFIG = load_config("configs/NER.yaml")

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
        default=CONFIG["model"]["name"][0],
        choices=CONFIG["LLM"].keys(),
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
        model_name=CONFIG["LLM"][args.model]["link"],
        prompt_path=args.prompt_path,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    log.info("Finished building Medical NER model.")


if __name__ == "__main__":
    main()