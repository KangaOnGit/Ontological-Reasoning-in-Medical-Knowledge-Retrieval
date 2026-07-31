import logging
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from src.utils.config import HF_TOKEN

log = logging.getLogger(__name__)

class TextEncoder:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_length: int = 32,
        batch_size: int = 256,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        # Metadata used by the index
        self.model_name = model_name
        self.pooling = "cls"
        self.query_format = ""

        self.max_length = max_length
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        self.model = (
            AutoModel.from_pretrained(model_name, token=HF_TOKEN)
            .to(self.device)
            .eval()
        )

    @property
    def embedding_dim(self) -> int:
        return self.model.config.hidden_size

    def encode(
        self,
        texts: Sequence[str],
        verbose: bool = False,
    ) -> np.ndarray:

        vectors = []
        n = len(texts)

        for i in range(0, n, self.batch_size):
            batch = list(texts[i:i + self.batch_size])

            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                output = self.model(**enc)

            embeddings = output.last_hidden_state[:, 0, :] # [CLS] token
            embeddings = torch.nn.functional.normalize(
                embeddings,
                p=2,
                dim=1,
            )

            vectors.append(
                embeddings.cpu().numpy().astype(np.float32)
            )

            if verbose and (i // self.batch_size) % 20 == 0:
                log.info("%d/%d", min(i + self.batch_size, n), n)

        if vectors:
            return np.vstack(vectors)

        return np.empty((0, self.embedding_dim), dtype=np.float32)

    # ------------------------------------------------------------------
    # Wrapper methods for compatibility with other embedding models
    # ------------------------------------------------------------------

    def encode_documents(
        self,
        texts: Sequence[str],
        verbose: bool = False,
    ) -> np.ndarray:
        return self.encode(texts, verbose)

    def encode_queries(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        return self.encode(texts)