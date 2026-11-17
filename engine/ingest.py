"""Turn uploaded files into clean text and overlapping, paragraph-aware chunks."""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from pathlib import PurePath
from xml.etree import ElementTree

TEXT_EXTS = {".txt", ".md", ".markdown", ".csv", ".json", ".log", ".yaml", ".yml"}
SUPPORTED_EXTS = TEXT_EXTS | {".pdf", ".docx"}


class IngestError(ValueError):
    """Raised for files the engine cannot or will not read."""


def safe_filename(name: str) -> str:
    base = PurePath(name.replace("\\", "/")).name
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .")
    return base[:200] or "untitled.txt"


def extract_text(filename: str, data: bytes) -> tuple[str, str]:
    """Return (kind, text) for a file. Raises IngestError for unsupported or empty files."""
    ext = PurePath(filename).suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise IngestError(f"Unsupported file type {ext or '(none)'}. Use: {', '.join(sorted(SUPPORTED_EXTS))}")
    if ext == ".pdf":
        text, kind = _pdf_text(data), "pdf"
    elif ext == ".docx":
        text, kind = _docx_text(data), "docx"
    else:
        text, kind = data.decode("utf-8", errors="replace"), ext.lstrip(".")
    text = normalize(text)
    if not text.strip():
        raise IngestError(f"{filename} has no readable text")
    return kind, text


def _pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:  # pypdf raises many exception types for bad files
        raise IngestError(f"Could not read PDF: {exc}") from exc


def _docx_text(data: bytes) -> str:
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            root = ElementTree.fromstring(zf.read("word/document.xml"))
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as exc:
        raise IngestError(f"Could not read DOCX: {exc}") from exc
    paragraphs = ["".join(t.text or "" for t in p.iter(f"{ns}t")) for p in root.iter(f"{ns}p")]
    return "\n\n".join(paragraphs)


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[-])")


def _units(text: str, max_words: int) -> list[str]:
    """Paragraphs, with any paragraph longer than max_words split into sentences."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para.split()) <= max_words:
            units.append(para)
            continue
        # Too long: split by lines first (keeps bullets and headings intact), then sentences.
        for line in filter(None, (ln.strip() for ln in para.split("\n"))):
            if len(line.split()) <= max_words:
                units.append(line)
                continue
            for sentence in _SENTENCE.split(line):
                words = sentence.split()
                # A single huge "sentence" (e.g. a table dump) is cut into fixed windows.
                for i in range(0, len(words), max_words):
                    units.append(" ".join(words[i : i + max_words]))
    return units


def chunk(text: str, max_words: int = 120, overlap_words: int = 30) -> list[str]:
    """Greedy, paragraph-aware chunking with a small overlap so facts are not cut in half."""
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for unit in _units(text, max_words):
        n = len(unit.split())
        if current and count + n > max_words:
            chunks.append("\n\n".join(current))
            carry: list[str] = []
            carried = 0
            for prev in reversed(current[1:]):  # never carry the whole chunk forward
                w = len(prev.split())
                if carried + w > overlap_words:
                    break
                carry.insert(0, prev)
                carried += w
            current, count = carry, carried
        current.append(unit)
        count += n
    if current:
        last = "\n\n".join(current)
        if not chunks or last != chunks[-1]:
            chunks.append(last)
    return chunks
