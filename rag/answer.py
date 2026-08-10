"""
Generate an answer using OpenAI and the top retrieved chunks.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

RAG_DIR = Path(__file__).resolve().parent

load_dotenv()
load_dotenv(RAG_DIR / ".env")
load_dotenv(RAG_DIR.parent / ".env")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
# OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def get_openai_client() -> OpenAI:
    """Create an OpenAI client using the API key from .env."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Add it to your .env file.")
    return OpenAI(api_key=api_key)


def build_context(chunks: list[dict]) -> str:
    """Combine the top retrieved chunks into one context string."""
    parts = []

    for index, chunk in enumerate(chunks, start=1):
        parts.append(
            f"Chunk {index} (source: {chunk['source']}):\n{chunk['text']}"
        )

    return "\n\n".join(parts)


def generate_answer(question: str, chunks: list[dict]) -> str:
    """
    Send the question and top chunks to OpenAI and return the answer.

    Steps:
    1. Combine the 3 retrieved chunks into context
    2. Ask OpenAI to answer using only that context
    3. Return the answer text
    """
    client = get_openai_client()
    context = build_context(chunks)

    prompt = f"""Answer the question using only the context below.
If the context does not contain the answer, say "I don't know based on the uploaded documents."

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()
