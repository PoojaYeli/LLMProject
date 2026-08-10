"""
RAG app UI.

Run with: streamlit run app.py
"""

import streamlit as st
from pathlib import Path
from urllib.parse import urlparse

from answer import OPENAI_MODEL, generate_answer
from chunker import CHUNK_OVERLAP, CHUNK_SIZE, chunk_text
from embedder import EMBEDDING_MODEL, embed_chunks
from file_reader import read_uploaded_file, read_webpage
from search import search_chunks, warm_up
from vector_store import COLLECTION_NAME, PERSIST_DIR, get_stored_chunk_count, store_chunks

# Allowed file extensions (webpage = HTML files)
ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx", ".html", ".htm", ".mhtml"}

EXTENSION_LABELS = {
    ".pdf": "PDF",
    ".doc": "DOC",
    ".docx": "DOCX",
    ".html": "Webpage (HTML)",
    ".htm": "Webpage (HTML)",
    ".mhtml": "Webpage (MHTML)",
}

PREVIEW_CHAR_LIMIT = 3000


def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def is_valid_webpage_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def show_allowed_types() -> None:
    st.caption("Allowed types: PDF, DOC, DOCX, or Webpage (HTML file or URL)")


def show_text_preview(text: str, source_label: str) -> None:
    """Show how much text was read and a short preview."""
    char_count = len(text)
    word_count = len(text.split())

    st.success(f"Read **{word_count:,} words** ({char_count:,} characters) from {source_label}")

    if char_count == 0:
        st.warning("The file was accepted, but no text could be extracted.")
        return

    preview = text[:PREVIEW_CHAR_LIMIT]
    st.text_area(
        "Extracted text preview",
        value=preview,
        height=300,
        disabled=True,
    )

    if char_count > PREVIEW_CHAR_LIMIT:
        st.caption(f"Showing first {PREVIEW_CHAR_LIMIT:,} characters.")


def show_chunks(chunks: list[str]) -> None:
    """Show how many chunks were created and a preview of each one."""
    st.subheader("Chunks")
    st.success(
        f"Created **{len(chunks)} chunks** "
        f"(size: {CHUNK_SIZE} chars, overlap: {CHUNK_OVERLAP} chars)"
    )

    if not chunks:
        st.warning("No chunks were created because the text is empty.")
        return

    for index, chunk in enumerate(chunks, start=1):
        with st.expander(f"Chunk {index} ({len(chunk)} characters)"):
            st.text(chunk)


def show_embeddings(embeddings: list[list[float]]) -> None:
    """Show how many embeddings were created and a small preview."""
    st.subheader("Embeddings")

    if not embeddings:
        st.warning("No embeddings were created because there are no chunks.")
        return

    dimensions = len(embeddings[0])
    st.success(
        f"Created **{len(embeddings)} embeddings** "
        f"using `{EMBEDDING_MODEL}` ({dimensions} numbers per chunk)"
    )
    st.caption(
        "Each chunk is now a vector of numbers. "
        "Similar text gets similar vectors, which enables semantic search."
    )

    preview_values = [round(value, 4) for value in embeddings[0][:8]]
    st.text(f"Chunk 1 embedding preview (first 8 values): {preview_values}")


def show_storage_result(stored_count: int) -> None:
    """Show that vectors were saved to ChromaDB."""
    st.subheader("Vector Storage")
    total_count = get_stored_chunk_count()
    st.success(
        f"Stored **{stored_count} chunks** in ChromaDB "
        f"(collection: `{COLLECTION_NAME}`, total stored: **{total_count}**)"
    )
    st.caption(f"Database folder: `{PERSIST_DIR}`")


def process_document(text: str, source_label: str) -> None:
    """Run chunking, embedding, and storage on extracted text."""
    show_text_preview(text, source_label)

    with st.spinner("Chunking text..."):
        chunks = chunk_text(text)

    show_chunks(chunks)

    if not chunks:
        return

    with st.spinner("Creating embeddings..."):
        embeddings = embed_chunks(chunks)

    show_embeddings(embeddings)

    with st.spinner("Saving to ChromaDB..."):
        stored_count = store_chunks(chunks, embeddings, source_label)

    show_storage_result(stored_count)


def handle_question(question: str) -> None:
    """Handle a user question: search ChromaDB and show relevant chunks."""
    question = question.strip()

    if not question:
        st.warning("Please enter a question.")
        return

    if get_stored_chunk_count() == 0:
        st.warning("No documents in the knowledge base yet. Upload a document first.")
        return

    st.info(f"Your question: **{question}**")

    with st.spinner("Searching for relevant chunks..."):
        matches = search_chunks(question)

    if not matches:
        st.warning("No relevant chunks found.")
        return

    st.success(f"Found **{len(matches)} relevant chunks**")

    try:
        with st.spinner(f"Generating answer with OpenAI ({OPENAI_MODEL})..."):
            answer = generate_answer(question, matches)
    except Exception as error:
        st.error(f"Could not generate answer: {error}")
        return

    st.subheader("Answer")
    st.write(answer)

    st.caption("Retrieved chunks used for this answer:")
    for index, match in enumerate(matches, start=1):
        with st.expander(
            f"Match {index} | source: {match['source']} | chunk: {match['chunk_index']}"
        ):
            st.text(match["text"])
            st.caption(f"Distance score: {match['distance']:.4f} (lower is more similar)")

st.set_page_config(page_title="RAG App", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def _warm_up_retrieval() -> bool:
    """Load the embedding model once when the app starts."""
    loading = st.empty()
    loading.info("Downloading the embeder model and loading the chroma DB...")
    warm_up()
    loading.empty()
    return True


_warm_up_retrieval()

st.title("RAG Knowledge Base")

upload_col, question_col = st.columns(2)

with upload_col:
    st.subheader("Upload a Document")
    st.write("Upload a file or paste a webpage URL to add it to your knowledge base.")
    show_allowed_types()

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "doc", "docx", "html", "htm", "mhtml"],
        help="Supported: PDF, Word (.doc/.docx), or saved webpage (.html)",
    )

    webpage_url = st.text_input(
        "Or paste a webpage URL",
        placeholder="https://example.com/article",
    )

    if st.button("Submit", type="primary", key="submit_upload"):
        if uploaded_file is not None:
            ext = get_file_extension(uploaded_file.name)

            if ext not in ALLOWED_EXTENSIONS:
                st.error(
                    f"Unsupported file type `{ext or '(none)'}`. "
                    f"Please upload a PDF, DOC, DOCX, or webpage (HTML) file."
                )
            else:
                file_type = EXTENSION_LABELS.get(ext, ext)

                try:
                    with st.spinner(f"Reading {file_type} file..."):
                        text = read_uploaded_file(uploaded_file, ext)

                    st.info(f"File: **{uploaded_file.name}** ({file_type})")
                    process_document(text, uploaded_file.name)

                except Exception as error:
                    st.error(f"Could not read the file: {error}")

        elif webpage_url.strip():
            if not is_valid_webpage_url(webpage_url):
                st.error("Invalid webpage URL. Use a full link starting with http:// or https://")
            else:
                url = webpage_url.strip()

                try:
                    with st.spinner("Downloading webpage..."):
                        text = read_webpage(url)

                    st.info(f"Webpage: **{url}**")
                    process_document(text, url)

                except Exception as error:
                    st.error(f"Could not read the webpage: {error}")

        else:
            st.warning("Please upload a file or enter a webpage URL.")

with question_col:
    st.subheader("Ask a Question")
    st.write("Ask a question about the documents you uploaded.")

    question = st.text_input(
        "Your question",
        placeholder="What is the refund policy?",
    )

    if st.button("Ask", type="primary", key="ask_question"):
        handle_question(question)
