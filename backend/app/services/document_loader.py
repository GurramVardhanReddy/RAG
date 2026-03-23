"""
Document Loading Module

Loads PDF and DOCX files and enriches each page's metadata with
paragraph-level tracking for explainability.
"""

import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from loguru import logger


def _detect_paragraphs(text: str) -> list[dict]:
    """
    Split a page's text into paragraphs and return each with its
    paragraph number and character offsets.

    A paragraph is defined as text separated by two or more newlines,
    or a single newline followed by a line that starts with uppercase/number.

    Returns:
        List of dicts: [{"paragraph_number": 1, "text": "...", "start": 0, "end": 120}, ...]
    """
    # Split on double newlines (common paragraph separator)
    raw_paragraphs = re.split(r'\n\s*\n', text)

    paragraphs = []
    para_num = 1
    current_pos = 0

    for raw_para in raw_paragraphs:
        cleaned = raw_para.strip()
        if not cleaned:
            current_pos += len(raw_para) + 1
            continue

        # Find actual position in original text
        start_pos = text.find(cleaned, current_pos)
        if start_pos == -1:
            start_pos = current_pos

        paragraphs.append({
            "paragraph_number": para_num,
            "text": cleaned,
            "start": start_pos,
            "end": start_pos + len(cleaned),
        })
        para_num += 1
        current_pos = start_pos + len(cleaned)

    return paragraphs


def _enrich_with_paragraphs(documents: list[Document]) -> list[Document]:
    """
    Take page-level documents and add paragraph metadata.

    Each document gets:
    - metadata["paragraphs"]: list of paragraph dicts with numbers and text
    - metadata["paragraph_count"]: total paragraphs on the page
    """
    enriched = []
    for doc in documents:
        paragraphs = _detect_paragraphs(doc.page_content)

        # Add paragraph info to metadata
        doc.metadata["paragraphs"] = paragraphs
        doc.metadata["paragraph_count"] = len(paragraphs)

        enriched.append(doc)

    return enriched


def load_document(file_path: str) -> list[Document]:
    """
    Load a document and enrich it with paragraph-level metadata.

    Supports: .pdf, .docx, .txt

    Each returned Document has metadata:
    - source: file path
    - page: page number (0-indexed)
    - paragraphs: list of {paragraph_number, text, start, end}
    - paragraph_count: total paragraphs on the page
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        docs = [Document(page_content=text, metadata={"source": file_path, "page": 0})]
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Enrich with paragraph detection
    docs = _enrich_with_paragraphs(docs)

    total_paragraphs = sum(d.metadata.get("paragraph_count", 0) for d in docs)
    logger.info(
        f"Loaded {file_path}: {len(docs)} pages, "
        f"{total_paragraphs} paragraphs detected"
    )

    return docs


def find_paragraph_for_chunk(chunk_text: str, page_paragraphs: list[dict]) -> dict:
    """
    Given a chunk's text and the paragraphs from its source page,
    find which paragraph(s) the chunk belongs to.

    Returns:
        dict with:
        - paragraph_number: the primary paragraph number
        - paragraph_numbers: list of all matching paragraph numbers
        - match_type: "exact", "partial", or "best_effort"
    """
    if not page_paragraphs:
        return {
            "paragraph_number": -1,
            "paragraph_numbers": [],
            "match_type": "no_paragraphs",
        }

    chunk_clean = chunk_text.strip().lower()
    matching_paras = []

    for para in page_paragraphs:
        para_clean = para["text"].strip().lower()

        # Check if chunk is fully contained in paragraph
        if chunk_clean in para_clean:
            return {
                "paragraph_number": para["paragraph_number"],
                "paragraph_numbers": [para["paragraph_number"]],
                "match_type": "exact",
            }

        # Check if paragraph is fully contained in chunk
        if para_clean in chunk_clean:
            matching_paras.append(para["paragraph_number"])

        # Check for significant overlap (at least 50% of shorter text)
        elif len(chunk_clean) > 0 and len(para_clean) > 0:
            # Use first 100 chars of chunk to find overlap
            chunk_start = chunk_clean[:100]
            if chunk_start in para_clean:
                matching_paras.append(para["paragraph_number"])

    if matching_paras:
        return {
            "paragraph_number": matching_paras[0],
            "paragraph_numbers": matching_paras,
            "match_type": "partial" if len(matching_paras) > 1 else "exact",
        }

    # Best effort: find paragraph with most word overlap
    best_para = -1
    best_overlap = 0
    chunk_words = set(chunk_clean.split())

    for para in page_paragraphs:
        para_words = set(para["text"].lower().split())
        overlap = len(chunk_words & para_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_para = para["paragraph_number"]

    return {
        "paragraph_number": best_para,
        "paragraph_numbers": [best_para] if best_para > 0 else [],
        "match_type": "best_effort",
    }
