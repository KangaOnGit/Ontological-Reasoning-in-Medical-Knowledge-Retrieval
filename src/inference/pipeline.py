from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from src.NER.model import NERmodel
from src.assertion.classifier import rule_based_assertion
from src.inference.writer import write_json, write_submission_zip
from src.postprocess.span_locator import locate_span_position
from src.preprocess.chunk import build_chunks
from src.preprocess.parse import parse
from src.rag.encoders.text_encoder import TextEncoder
from src.rag.retriever.hybrid_retriever import HybridRetriever
from src.utils.config import load_config

CONFIG_NER = load_config("configs/NER.yaml")
CONFIG_INFER = load_config("configs/inference.yaml")
CONFIG_RAG = load_config("configs/RAG/indexing/faiss_indexing.yaml")

ENTITY_TO_KB = {
    "CHẨN_ĐOÁN": "ICD10",
    "THUỐC": "RXNorm",
}

log = logging.getLogger(__name__)


def run_inference(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    ner_model = NERmodel(
        model_name=CONFIG_NER["LLM"][args.ner_model]["link"],
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

    retriever = HybridRetriever(
        encoder,
        CONFIG_RAG["output"]["path"],
    )

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is None and args.output_json:
        output_dir = Path(args.output_json).parent

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    submission_files: dict[str, list[dict[str, Any]]] = {}

    for path in sorted(input_dir.glob("*.txt")):
        
        log.info("Processing %s", path.name)

        parsed = parse(path)
        chunks = build_chunks(parsed)

        file_records: list[dict[str, Any]] = []

        for chunk in chunks:
            spans = ner_model.forward(chunk.text)

            for span in spans:
                if not span.text or not span.typ:
                    continue

                assertion = rule_based_assertion(span)

                candidates: list[str] = []

                kb = ENTITY_TO_KB.get(span.typ)
                if kb:
                    retrieval_results = retriever.query(
                        span.text,
                        kb,
                        top_k=5,
                    )
                    
                    # Remove dupes
                    candidates = list(
                        dict.fromkeys(
                            item.id
                            for item in retrieval_results
                            if item.id
                        )
                    )

                file_records.append(
                    {
                        "text": span.text,
                        "type": span.typ,
                        "candidates": candidates,
                        "assertions": assertion,
                        "position": locate_span_position(
                            span,
                            chunk.records,
                        ),
                    }
                )

        submission_files[path.stem] = file_records

        log.info(
            "Collected %d records from %s",
            len(file_records),
            path.name,
        )
            
    if output_dir:
        zip_path = write_submission_zip(
            submission_files,
            output_dir / "submission.zip",
        )

        log.info(
            "Wrote submission ZIP to %s",
            zip_path,
        )

    return submission_files