"""
Store chunk embeddings in ChromaDB.

ChromaDB keeps three things together for each chunk:
- the text (document)
- the embedding vector
- metadata (source name, chunk number)
"""

import os
import uuid
from functools import lru_cache
from pathlib import Path

import chromadb
from dotenv import load_dotenv

load_dotenv()
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DEFAULT_PERSIST_DIR = Path(__file__).resolve().parent / "chroma_data"
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(DEFAULT_PERSIST_DIR))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")


@lru_cache(maxsize=1)
def get_collection():
    """Open ChromaDB on disk and get (or create) the collection."""
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def store_chunks(
    chunks: list[str],
    embeddings: list[list[float]],
    source: str,
) -> int:
    """
    Save chunks and their embeddings to ChromaDB.

    Steps:
    1. Open the ChromaDB collection
    2. Create a unique ID for each chunk
    3. Save the text, vector, and metadata together
    4. Return how many chunks were stored
    """
    if not chunks or not embeddings:
        return 0

    if len(chunks) != len(embeddings):
        raise ValueError("Number of chunks and embeddings must match.")

    collection = get_collection()

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {"source": source, "chunk_index": index}
        for index in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(chunks)


def get_stored_chunk_count() -> int:
    """Return total number of chunks currently stored in ChromaDB."""
    return get_collection().count()
