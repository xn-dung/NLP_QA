import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

