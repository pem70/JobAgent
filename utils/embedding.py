from __future__ import annotations

import hashlib
import os
from typing import Optional

import numpy as np
from openai import OpenAI
from config import load_env


def _deterministic_fallback_embedding(text: str, dim: int = 1536) -> np.ndarray:
    """
    Deterministic fallback vector when embedding API is unavailable.
    This keeps the pipeline operable for local/dev tests.
    """
    seed_bytes = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    seed = int.from_bytes(seed_bytes[:8], byteorder="big", signed=False)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)


def get_embedding(text: str, client: Optional[OpenAI] = None) -> np.ndarray:
    """
    Call OpenAI text-embedding-3-small and return shape (1536,) float32.
    Truncates text to approx 32k chars.
    Falls back to deterministic local embedding if API call fails.
    """
    safe_text = (text or "").strip()[:32000]
    if not safe_text:
        safe_text = "empty"

    load_env()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    embedding_client = client or OpenAI(api_key=api_key or None)
    try:
        response = embedding_client.embeddings.create(
            model="text-embedding-3-small",
            input=safe_text,
        )
        vector = response.data[0].embedding
        return np.array(vector, dtype=np.float32)
    except Exception:
        return _deterministic_fallback_embedding(safe_text)


def get_embeddings_batch(texts: list[str], client: Optional[OpenAI] = None) -> list[np.ndarray]:
    """
    Batch embedding API call using OpenAI input=list[str].
    Returns embeddings aligned to input order.
    Falls back per item to deterministic embedding on API failure.
    """
    if not texts:
        return []

    safe_texts = [(text or "").strip()[:32000] or "empty" for text in texts]

    load_env()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    embedding_client = client or OpenAI(api_key=api_key or None)
    try:
        response = embedding_client.embeddings.create(
            model="text-embedding-3-small",
            input=safe_texts,
        )
        vectors_by_index = {
            int(item.index): np.array(item.embedding, dtype=np.float32)
            for item in response.data
        }
        return [vectors_by_index[i] for i in range(len(safe_texts))]
    except Exception:
        return [_deterministic_fallback_embedding(text) for text in safe_texts]
