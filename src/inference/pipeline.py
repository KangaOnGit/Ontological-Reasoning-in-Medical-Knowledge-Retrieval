from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from src.NER.model import NERmodel
from src.assertion.classifier import rule_based_assertion
from src.inference.writer import write_submission_zip
from src.postprocess.span_locator import locate_span_position
from src.preprocess.chunk import build_chunks
from src.preprocess.parse import parse
from src.rag.encoders.text_encoder import TextEncoder
from src.rag.retriever.hybrid_retriever import HybridRetriever
from src.utils.config import load_config

CONFIG_NER = load_config("configs/NER.yaml")
CONFIG_RAG = load_config("configs/RAG/indexing/faiss_indexing.yaml")

ENTITY_TO_KB = {
    "CHẨN_ĐOÁN": "ICD10",
    "THUỐC": "RXNorm",
}

log = logging.getLogger(__name__)


class InferencePipeline:

    def __init__(self,
                 ner_model: str
                 ):
        # ------
        model_cfg = CONFIG_NER["LLM"][ner_model]
        log.info("Loading %s...", model_cfg["link"])
        self.ner_model = NERmodel(
            model_name=model_cfg["link"],
            prompt_path=model_cfg["prompt_path"],
            max_new_tokens=model_cfg["max_new_tokens"],
            repetition_penalty=model_cfg["repetition_penalty"],
        )
        log.info("Loaded %s successfully.", model_cfg["link"])
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

    def _run_single_file(
        self,
        path: Path,
    ) -> list[dict[str, Any]]:

        log.info("Processing %s", path.name)

        parsed = parse(path)
        chunks = build_chunks(parsed)

        file_records: list[dict[str, Any]] = []

        for chunk in chunks:
            spans = self.ner_model.forward(chunk.text)

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

                kb = ENTITY_TO_KB.get(span.typ)

                if kb:
                    retrieval_results = self.retriever.query(
                        span.text,
                        kb,
                        top_k=5,
                    )

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

        log.info(
            "Collected %d records from %s",
            len(file_records),
            path.name,
        )

        return file_records

    def run_text(
        self,
        text: str,
    ) -> dict[str, list[dict[str, Any]]]:

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            encoding="utf-8",
        ) as handle:
            handle.write(text)
            temp_path = Path(handle.name)

        try:
            return {
                "input": self._run_single_file(temp_path)
            }
        finally:
            temp_path.unlink(missing_ok=True)

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

        return {
            path.stem: self._run_single_file(path)
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
            submission_files[path.stem] = self._run_single_file(path)

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