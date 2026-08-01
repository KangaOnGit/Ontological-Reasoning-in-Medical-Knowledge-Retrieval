# src/NER/factory.py

from src.ner.base import BaseNER
from src.ner.llm import LLM_NER
from src.ner.gliner import GLiNER_NER
from src.utils.config import load_config

CONFIG_NER = load_config("configs/NER.yaml")

def build_ner(
    ner_type: str,
    ner_model: str,
) -> BaseNER:

    cfg = CONFIG_NER[ner_type][ner_model]

    if ner_type == "LLM":
        return LLM_NER(
            model_name = cfg["link"]
        )

    if ner_type == "GLINER":
        return GLiNER_NER(
            model_name = cfg["link"],
            threshold = cfg["threshold"]
        )

    raise ValueError(f"Unknown NER backend: {ner_type}")