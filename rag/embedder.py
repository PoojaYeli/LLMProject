"""
Turn text chunks into embedding vectors using a local Hugging Face model.

Model: BAAI/bge-base-en-v1.5
An embedding is a list of numbers that represents the meaning of text.
Similar chunks get similar numbers, which makes search work later.
"""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Load .env from rag/ or the parent project folder
load_dotenv()
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Load the embedding model once and reuse it.
    The first run downloads the model; later runs are faster.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """
    Convert each text chunk into an embedding vector.

    Steps:
    1. Load the BGE model (cached after first use)
    2. Encode all chunks locally on your machine
    3. Return one vector (list of floats) per chunk
    """
    if not chunks:
        return []

    model = get_embedding_model()

    vectors = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return vectors.tolist()
