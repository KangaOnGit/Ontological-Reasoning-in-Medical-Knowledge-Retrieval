from gliner import GLiNER
from src.NER.base import BaseNER, Span
from src.preprocess.base import Chunk
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
                chunk: Chunk
                ) -> list[Span]:
        
        spans: list[Span] = []
        chunk_text = chunk.text
        
        tokenizer = self.model.data_processor.transformer_tokenizer
        tokens = tokenizer(
            chunk_text,
            add_special_tokens=False,
            return_offsets_mapping=True
        )

        input_ids = tokens["input_ids"]
        offsets = tokens["offset_mapping"]
        max_tokens = 384

        for i in range(0, len(input_ids), max_tokens):
            
            token_start = i
            token_end = min(i + max_tokens, len(input_ids))

            char_start = offsets[token_start][0]
            char_end = offsets[token_end - 1][1]

            text = chunk_text[char_start:char_end]
            
            entities = self.model.predict_entities(
                text,
                labels=list(self.label_map.keys()),
                threshold = self.threshold
            )

            for ent in entities:
                typ = self.label_map.get(ent["label"])
                if not typ:
                    log.warning("Skipping Unknown Label %s", ent["label"])
                    continue
                spans.append(
                    Span(
                        text=ent["text"],
                        typ=typ,
                        
                        # since grouped records have same path
                        subsection=chunk.records[0].path[1],
                        section=chunk.records[0].path[0],
                        
                        context=text[max(0, ent["start"]-30):ent["end"]+30],
                        start=ent["start"],
                        end=ent["end"]
                    )
                )

        return spans