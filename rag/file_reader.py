"""
Read plain text from uploaded documents and webpages.
Each function handles one file type to keep the logic easy to follow.
Reads table contents and not images.
"""

import email
import re
from email import policy

import requests
from bs4 import BeautifulSoup
from docx import Document
from olefile import OleFileIO
from pypdf import PdfReader


def read_pdf(file_obj) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(file_obj)
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())

    return "\n\n".join(pages)


def read_docx(file_obj) -> str:
    """Extract text from a Word .docx file."""
    document = Document(file_obj)
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    return "\n\n".join(paragraphs)


def read_doc(file_obj) -> str:
    """
    Extract text from a legacy Word .doc file.
    Works best for simple documents. For complex files, save as .docx.
    """
    file_obj.seek(0)

    if not OleFileIO.isOleFile(file_obj):
        raise ValueError("This is not a valid .doc file. Try saving it as .docx.")

    file_obj.seek(0)
    ole = OleFileIO(file_obj)

    if not ole.exists("WordDocument"):
        raise ValueError("Could not find text inside this .doc file. Try saving it as .docx.")

    data = ole.openstream("WordDocument").read()
    chunks = re.findall(r"[\x20-\x7E\r\n\t]{4,}", data.decode("latin-1", errors="ignore"))
    text = "\n".join(chunk.strip() for chunk in chunks if chunk.strip())

    if not text.strip():
        raise ValueError("Could not extract text from this .doc file. Try saving it as .docx.")

    return text


def _html_to_text(html: str) -> str:
    """Turn HTML into plain text."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    return soup.get_text(separator="\n", strip=True)


def read_html(file_obj) -> str:
    """Extract text from a saved HTML webpage file."""
    file_obj.seek(0)
    html = file_obj.read()

    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="ignore")

    return _html_to_text(html)


def read_mhtml(file_obj) -> str:
    """Extract text from a saved .mhtml webpage file."""
    file_obj.seek(0)
    message = email.message_from_binary_file(file_obj, policy=policy.default)

    for part in message.walk():
        if part.get_content_type() == "text/html":
            return _html_to_text(part.get_content())

    raise ValueError("No HTML content found in this MHTML file.")


def read_webpage(url: str) -> str:
    """Download a webpage and extract its text."""
    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"},
    )
    response.raise_for_status()
    return _html_to_text(response.text)


def read_uploaded_file(file_obj, extension: str) -> str:
    """
    Pick the right reader based on the file extension
    and return the document text.
    """
    readers = {
        ".pdf": read_pdf,
        ".docx": read_docx,
        ".doc": read_doc,
        ".html": read_html,
        ".htm": read_html,
        ".mhtml": read_mhtml,
    }

    reader = readers.get(extension)
    if reader is None:
        raise ValueError(f"No reader available for {extension} files.")

    file_obj.seek(0)
    return reader(file_obj)
