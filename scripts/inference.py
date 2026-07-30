from __future__ import annotations

import argparse
import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Any

from src.NER.base import Span
from src.NER.model import NERmodel
from src.assertation.model import assert_cls
from src.preprocess.chunk import build_chunks
from src.preprocess.parse import parse
from src.rag.encoders.text_encoder import TextEncoder
from src.rag.retriever.base import RetrievalResult
from src.rag.retriever.bm25_retriever import BM25Retriever
from src.rag.retriever.exact_alias_retriever import ExactAliasRetriever
from src.rag.retriever.faiss_retriever import FaissRetriever
from src.utils.config import load_config

CONFIG_NER = load_config("configs/NER.yaml")
CONFIG_INFER = load_config("configs/infer.yaml")
CONFIG_RAG = load_config("configs/RAG/indexing/faiss_indexing.yaml")

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


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def rule_based_assertion(span: Span) -> list[str]:
    """Infer assertion labels from subsection first, then section/context."""
    candidates = []

    if span.subsection:
        haystack = normalize_text(span.subsection)
    else:
        haystack = normalize_text(" ".join(
            part for part in [span.section, span.context, span.text] if part
        ))

    for assertion_name in ("isHistorical", "isNegated", "isFamily"):
        if any(phrase in haystack for phrase in assert_cls[assertion_name]):
            candidates.append(assertion_name)

    return candidates


class HybridRetriever:
    """Combine FAISS, BM25, and exact alias matching into a single ranked list."""

    def __init__(self, kb: str, output_dir: str | Path):
        self.kb = kb
        self.output_dir = Path(output_dir)

    def _build(self, mention: str, top_k: int = 5) -> list[RetrievalResult]:
        raise NotImplementedError


class HybridKnowledgeRetriever:
    def __init__(self, encoder: TextEncoder, output_dir: str | Path):
        self.faiss = FaissRetriever(encoder)
        self.bm25 = BM25Retriever(output_dir)
        self.exact = ExactAliasRetriever(output_dir)

    def query(self, mention: str, kb: str, top_k: int = 5) -> list[RetrievalResult]:
        if not mention:
            return []

        try:
            faiss_results = self.faiss.query(mention, kb, top_k=top_k)
        except Exception as exc:  # pragma: no cover - defensive fallback
            log.warning("FAISS retrieval failed for %s: %s", kb, exc)
            faiss_results = []

        try:
            bm25_results = self.bm25.query(mention, kb, top_k=top_k)
        except Exception as exc:  # pragma: no cover - defensive fallback
            log.warning("BM25 retrieval failed for %s: %s", kb, exc)
            bm25_results = []

        try:
            exact_results = self.exact.query(mention, kb, top_k=top_k)
        except Exception as exc:  # pragma: no cover - defensive fallback
            log.warning("Exact alias retrieval failed for %s: %s", kb, exc)
            exact_results = []

        combined: dict[str, RetrievalResult] = {}

        def add_result(results: list[RetrievalResult], weight: float) -> None:
            for result in results:
                entry = combined.setdefault(
                    result.id,
                    RetrievalResult(
                        id=result.id,
                        name=result.name,
                        score=0.0,
                        tty=result.tty,
                    ),
                )
                entry.score += result.score * weight

        add_result(faiss_results, 0.5)
        add_result(bm25_results, 0.3)
        add_result(exact_results, 0.2)

        ranked = sorted(combined.values(), key=lambda item: item.score, reverse=True)
        return ranked[:top_k]


def locate_span_position(raw_text: str, span: Span) -> list[int]:
    """Locate the span text in the original file content and return [start, end]."""
    if not span.text:
        return []

    needle = span.text.strip()
    if not needle:
        return []

    for candidate in (needle, re.sub(r"\s+", " ", needle)):
        if not candidate:
            continue
        if candidate in raw_text:
            start = raw_text.index(candidate)
            return [start, start + len(candidate)]

    return []


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
    retriever = HybridKnowledgeRetriever(encoder, CONFIG_RAG["output"]["path"])

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
                map_to_kb: dict[str, str] = {
                    "CHẨN_ĐOÁN": "ICD10",
                    "THUỐC": "RXNorm"
                }
                if span.typ in map_to_kb:
                    results = retriever.query(span.text, map_to_kb[span.typ], top_k=5)
                    candidates = [
                        item.id for item in results if item.id
                    ]
                    candidates = list(dict.fromkeys(candidates))

                position = locate_span_position(raw_text, span)
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
