"""
Semantic search over stored chunks in ChromaDB.
"""

from embedder import embed_query
from vector_store import get_collection

DEFAULT_TOP_K = 3


def warm_up() -> None:
    """Pre-load the model and ChromaDB before the first search."""
    from embedder import warm_up_model

    warm_up_model()
    get_collection().count()


def search_chunks(question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Find the most relevant chunks for a user question.

    Steps:
    1. Convert the question into an embedding
    2. Search ChromaDB for similar chunk vectors
    3. Return the best matching chunks with metadata
    """
    collection = get_collection()
    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    query_embedding = embed_query(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, total_chunks),
        include=["documents", "metadatas", "distances"],
    )

    matches = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for text, metadata, distance in zip(documents, metadatas, distances):
        matches.append(
            {
                "text": text,
                "source": metadata["source"],
                "chunk_index": metadata["chunk_index"],
                "distance": distance,
            }
        )

    return matches
