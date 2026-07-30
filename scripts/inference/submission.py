from __future__ import annotations

import argparse
import json
import logging
import zipfile
from pathlib import Path
from typing import Any

from src.NER.model import NERmodel
from src.assertion.classifier import rule_based_assertion
from src.preprocess.chunk import build_chunks
from src.preprocess.parse import parse
from src.rag.encoders.text_encoder import TextEncoder
from src.rag.retriever.hybrid_retriever import HybridRetriever
from src.utils.config import load_config
from src.postprocess.span_locator import locate_span_position

CONFIG_NER = load_config("configs/NER.yaml")
CONFIG_INFER = load_config("configs/inference.yaml")
CONFIG_RAG = load_config("configs/RAG/indexing/faiss_indexing.yaml")
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
        default=CONFIG_INFER["data"]["eval"]["path"],
        help="Directory containing input .txt files",
    )
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
        default=None,
        help="Directory that will receive per-file zip archives",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save a combined JSON summary",
    )
    return parser.parse_args()


def run_inference(args: argparse.Namespace) -> list[dict[str, Any]]:
    ner_model = NERmodel(
        model_name=CONFIG_NER["LLM"][args.model]["link"],
        prompt_path=args.prompt_path,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    encoder = TextEncoder(
        model_name=CONFIG_RAG["encoders"][CONFIG_RAG["model"]["name"]]["link"],
        device="auto",
        max_length=32,
        batch_size=128,
    )
    retriever = HybridRetriever(encoder, CONFIG_RAG["output"]["path"])

    input_dir = Path(args.input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is None and args.output_json:
        output_dir = Path(args.output_json).parent
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []

    for path in sorted(input_dir.glob("*.txt")):
        log.info("Processing %s", path.name)

        raw_text = path.read_text(encoding="utf-8")
        parsed_results = parse(path)
        chunks = build_chunks(parsed_results)

        file_records: list[dict[str, Any]] = []

        for chunk in chunks:
            spans = ner_model.forward(chunk)

            for span in spans:
                if not span.typ or not span.text:
                    continue

                assertion = rule_based_assertion(span)
                candidates: list[str] = []
                kb = ENTITY_TO_KB.get(span.typ)
                
                if kb:
                    retrieval_results = retriever.query(span.text, kb, top_k=5)
                    candidates = [
                        item.id for item in results if item.id
                    ]
                    candidates = list(
                        dict.fromkeys(item.id for item in retrieval_results if item.id)
                        )

                position = locate_span_position(span, chunk.records)
                file_records.append(
                    {
                        "text": span.text,
                        "type": span.typ,
                        "candidates": candidates,
                        "assertions": assertion,
                        "position": position,
                    }
                )

        if output_dir is not None:
            archive_path = output_dir / f"{path.stem}.zip"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for idx, record in enumerate(file_records, start=1):
                    zf.writestr(f"{idx}.json", json.dumps(record, ensure_ascii=False, indent=2))
            log.info("Wrote %d records to %s", len(file_records), archive_path)

        results.extend(file_records)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("Saved %d combined results to %s", len(results), output_path)

    return results


def main() -> None:
    args = parse_args()
    results = run_inference(args)
    log.info("Finished inference pipeline with %d extracted spans", len(results))


if __name__ == "__main__":
    main()
