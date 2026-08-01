from gliner import GLiNER
from src.NER.base import BaseNER, Span
from typing import Dict
import logging

DEFAULT_LABEL_MAP: Dict[str, str] = {
    "symptom": "TRIỆU_CHỨNG",
    "disease or diagnosis": "CHẨN_ĐOÁN",
    "medication or drug": "THUỐC",
    "medical test or lab name": "TÊN_XÉT_NGHIỆM",
    "test result or measurement value": "KẾT_QUẢ_XÉT_NGHIỆM",
}

log = logging.getLogger(__name__)

class GLiNER_NER(BaseNER):

    def __init__(self,
                 model_name: str,
                 threshold: float = 0.5,
                 label_map: dict[str, str] | None = None
                 ):
        log.info(
            "Model=%s | Threshold=%.2f",
            model_name,
            threshold,
            )
        
        self.label_map = label_map or DEFAULT_LABEL_MAP
        self.threshold = threshold
        
        log.info("Loading model...")
        self.model = GLiNER.from_pretrained(model_name)

    def forward(self,
                text: str
                ) -> list[Span]:

        entities = self.model.predict_entities(
            text,
            labels=list(self.label_map.keys()),
            threshold = self.threshold
        )

        spans: list[Span] = []

        for ent in entities:
            typ = self.label_map.get(ent["label"])
            if not typ:
                log.warning("Skipping Unknown Label %s", ent["label"])
                continue
            spans.append(
                Span(
                    text=ent["text"],
                    typ=typ,
                    context=text[max(0, ent["start"]-30):ent["end"]+30],
                    start = ent["start"],
                    end = ent["end"]
                )
            )

        return spans