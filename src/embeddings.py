# src/embeddings.py
"""Build the embedding model used for the vector index (one place, so ingest and retrieval always match).

Calls sentence-transformers directly instead of langchain_huggingface.HuggingFaceEmbeddings:
older versions of that wrapper can't give queries and documents different settings,
and instruction-tuned embedders (Qwen3-Embedding) need a task instruction on queries only.
For models without an instruction (MiniLM) the vectors are identical to the wrapper's,
so existing indexes keep working.
"""
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from config import EMBEDDING_MODEL, QUERY_INSTRUCTIONS


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str, query_prefix: str = "", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        import os
        import torch

        device = os.getenv("EMBEDDING_DEVICE") or (
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=device)
        print(f"Embedding model {model_name} on {self.model.device}")

    def _encode(self, texts, progress=False):
        vectors = self.model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True,
                                    show_progress_bar=progress)
        return [list(map(float, v)) for v in vectors]

    def embed_documents(self, texts):
        texts = list(texts)
        return self._encode(texts, progress=len(texts) > 1000)  # progress bar when building an index

    def embed_query(self, text):
        return self._encode([self.query_prefix + text])[0]


@lru_cache(maxsize=4)
def get_embeddings(model: str = EMBEDDING_MODEL):
    prefix = f"Instruct: {QUERY_INSTRUCTIONS[model]}\nQuery: " if model in QUERY_INSTRUCTIONS else ""
    return SentenceTransformerEmbeddings(model, query_prefix=prefix)
