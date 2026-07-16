"""
Split long text into smaller chunks for RAG.

Each chunk is a piece of text small enough to embed and search later.
Neighboring chunks overlap a little so context is not lost at boundaries.
"""

# Default settings (same idea as in .env.example)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping chunks.

    Example with chunk_size=1000 and overlap=200:
        Chunk 1: characters 0-1000
        Chunk 2: characters 800-1800
        Chunk 3: characters 1600-2600
        ...
    """
    text = text.strip()
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Prefer breaking at a space so we don't cut words in half
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start:
                end = space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        # Move forward, but go back a little for overlap
        start = max(end - chunk_overlap, start + 1)

    return chunks
