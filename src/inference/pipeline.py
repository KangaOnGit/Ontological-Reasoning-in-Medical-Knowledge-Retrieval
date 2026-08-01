from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from collections import defaultdict
from src.NER.builder import build_ner
from src.assertion.classifier import rule_based_assertion
from src.inference.writer import write_submission_zip
from src.postprocess.span_locator import locate_span_position
from src.preprocess.chunk import build_chunks
from src.preprocess.parse import parse
from src.rag.encoders.text_encoder import TextEncoder
from src.rag.retriever.hybrid_retriever import HybridRetriever
from src.utils.config import load_config

CONFIG_RAG = load_config("configs/RAG/indexing/faiss_indexing.yaml")

ENTITY_TO_KB = {
    "CHẨN_ĐOÁN": "ICD10",
    "THUỐC": "RXNorm",
}

log = logging.getLogger(__name__)


class InferencePipeline:

    def __init__(self,
                ner_type: str,
                ner_model: str,
                 ):
        # ------
        log.info(
            "Loading %s model '%s'...",
            ner_type,
            ner_model,
            )
        self.ner_model = build_ner(
            ner_type=ner_type,
            ner_model=ner_model,
        )
        log.info(
            "Loaded %s (%s) successfully.",
            ner_model,
            ner_type,
        )
        # ------
        encoder_cfg = CONFIG_RAG["encoders"][CONFIG_RAG["model"]["name"]]
        log.info("Loading %s...", encoder_cfg["link"])
        self.encoder = TextEncoder(
            model_name=encoder_cfg["link"],
            device="auto",
            max_length=32,
            batch_size=128,
        )
        log.info("Loaded %s successfully.", encoder_cfg["link"])
        # ------
        log.info("Loading HybridRetriever...")
        self.retriever = HybridRetriever(
            self.encoder,
            CONFIG_RAG["output"]["path"],
        )
        log.info("Loaded HybridRetriever successfully.")

    def _run_parsed(
        self,
        parsed: list[dict],
        file_name: str = "input",
        ) -> list[dict[str, Any]]:

        cache: dict[tuple[str, str], list[str]] = {}
        log.info("Processing %s", file_name)
        
        chunks = build_chunks(parsed)

        file_records: list[dict[str, Any]] = []

        for chunk in chunks:
            spans = self.ner_model.forward(chunk)

            for span in spans:
                if (
                    not span.text
                    or not span.typ
                    or span.typ == "UNKNOWN"
                ):
                    continue

                log.debug("Processing span: %s", span.text)

                assertion = rule_based_assertion(span)

                candidates: list[str] = []

                if span.typ in ENTITY_TO_KB:
                    key = (span.text.strip().lower(), span.typ)
                    if key in cache:
                        candidates = cache[key]
                    else:
                        retrieval_results = self.retriever.query(
                            span.text,
                            ENTITY_TO_KB[span.typ],
                            top_k=5,
                        )

                        candidates = list(
                            dict.fromkeys(
                                item.id
                                for item in retrieval_results
                                if item.id
                            )
                        )
                        cache[key] = candidates

                file_records.append(
                    {
                        "text": span.text,
                        "type": span.typ,
                        "candidates": candidates,
                        "assertions": assertion,
                        "position": locate_span_position(
                            span,
                            chunk
                        ),
                    }
                )

        log.info(
            "Collected %d records from %s",
            len(file_records),
            file_name,
        )

        return file_records

    def run_text(
        self,
        text: str,
    ) -> dict[str, list[dict[str, Any]]]:

        parsed = parse(text=text)
        
        return {
            "input": self._run_parsed(
                parsed,
                file_name="input",
            )
        }

    def run_file(
        self,
        path: str | Path,
    ) -> dict[str, list[dict[str, Any]]]:

        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(path)

        if not path.is_file():
            raise ValueError(f"{path} is not a file.")

        if path.suffix.lower() != ".txt":
            raise ValueError(
                "Only .txt files are supported."
            )
            
        parsed = parse(filename = path)
        return {
            path.stem: self._run_parsed(
                parsed,
                file_name=path.name,
            )
        }

    def run_submission(
        self,
        input_dir: str | Path,
        output_dir: str | Path | None = None,
    ) -> dict[str, list[dict[str, Any]]]:

        input_dir = Path(input_dir)

        if not input_dir.exists():
            raise FileNotFoundError(
                f"Input directory does not exist: {input_dir}"
            )

        submission_files: dict[str, list[dict[str, Any]]] = {}

        for path in sorted(input_dir.glob("*.txt")):
            parsed = parse(filename=path)
            submission_files[path.stem] = self._run_parsed(
                parsed,
                file_name=path.name,
                )

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            zip_path = write_submission_zip(
                submission_files,
                output_dir / "submission.zip",
            )

            log.info(
                "Wrote submission ZIP to %s",
                zip_path,
            )

        return submission_files