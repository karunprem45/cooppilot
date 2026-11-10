"""Turn an uploaded JD or resume (PDF or plain text) into clean text."""
import io
import re

from pypdf import PdfReader


def extract_text(data: bytes, filename: str = "") -> str:
    if filename.lower().endswith(".pdf") or data[:4] == b"%PDF":
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = data.decode("utf-8", errors="ignore")
    return clean_text(text)


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
