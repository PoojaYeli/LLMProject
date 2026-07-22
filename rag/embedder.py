"""
Turn text into embedding vectors using a local Hugging Face model.

Model: BAAI/bge-base-en-v1.5
"""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

RAG_DIR = Path(__file__).resolve().parent

load_dotenv()
load_dotenv(RAG_DIR / ".env")
load_dotenv(RAG_DIR.parent / ".env")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

# BGE uses this prefix for search queries (not for document chunks)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the embedding model once and reuse it."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Convert document chunks into embedding vectors."""
    if not chunks:
        return []

    model = get_embedding_model()
    vectors = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_query(question: str) -> list[float]:
    """Convert a user question into an embedding vector for search."""
    model = get_embedding_model()
    prefixed_question = QUERY_PREFIX + question.strip()
    vector = model.encode(
        prefixed_question,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vector.tolist()
