"""Thin wrapper around sentence-transformers with sane defaults for this corpus."""

from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from . import config


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    def __init__(self, model_name: str = config.MODEL_NAME,
                 max_seq_length: int = config.MAX_SEQ_LENGTH,
                 device: str | None = None):
        self.model_name = model_name
        self.device = device or pick_device()
        self.model = SentenceTransformer(model_name, device=self.device)
        # Shorter sequences trade a sliver of accuracy for a large speedup, and
        # award text puts the useful signal first anyway.
        self.model.max_seq_length = max_seq_length
        self.max_seq_length = max_seq_length
        # sentence-transformers 5 renamed this; keep working on older releases.
        measure = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension
        self.dimension = measure()

    def encode(self, texts, batch_size: int = config.BATCH_SIZE,
               progress: bool = False) -> np.ndarray:
        return self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0].tolist()
