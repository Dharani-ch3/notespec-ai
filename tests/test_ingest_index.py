import io
import zipfile

import pytest

from engine.index import HashingEmbedder, HybridIndex, tokenize
from engine.ingest import IngestError, chunk, extract_text, safe_filename


def test_chunks_respect_size_and_keep_bullets():
    text = "# Notes\n\n" + "\n".join(f"- Item {i} must be tracked carefully by staff." for i in range(60))
    pieces = chunk(text, max_words=50, overlap_words=10)
    assert len(pieces) > 3
    assert all(len(p.split()) <= 60 for p in pieces)
    assert all(not p.startswith("# Notes - Item") for p in pieces)  # headings stay on their own line


def test_overlap_repeats_the_tail_of_the_previous_chunk():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 20 for i in range(6))
    pieces = chunk(text, max_words=50, overlap_words=25)
    assert pieces[1].split("\n\n")[0] in pieces[0]


def test_unsupported_and_empty_files_are_rejected():
    with pytest.raises(IngestError):
        extract_text("virus.exe", b"MZ")
    with pytest.raises(IngestError):
        extract_text("empty.md", b"   \n ")


def test_docx_text_is_extracted():
    xml = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
        "<w:p><w:r><w:t>Nurses must flag urgent cases.</w:t></w:r></w:p></w:body></w:document>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", xml)
    kind, text = extract_text("notes.docx", buf.getvalue())
    assert kind == "docx" and "flag urgent" in text


def test_safe_filename_strips_paths():
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("C:\\Users\\x\\notes<1>.md") == "notes_1_.md"


def test_hybrid_search_ranks_keyword_and_paraphrase_matches():
    chunks = [
        {"id": "a#0", "filename": "a.md", "text": "Check-in must keep working offline when the Wi-Fi drops."},
        {"id": "a#1", "filename": "a.md", "text": "The manager wants a weekly report of wait times."},
        {"id": "a#2", "filename": "a.md", "text": "Staff sign in with Google SSO accounts."},
    ]
    index = HybridIndex(chunks, HashingEmbedder())
    hits = index.search("offline wifi", k=3)
    assert hits[0].chunk_id == "a#0"
    assert hits[0].score == 1.0 and hits[0].bm25_rank == 1


def test_tokenize_stems_and_drops_stopwords():
    assert tokenize("The nurses are tracking appointments") == ["nurse", "track", "appointment"]
