"""
Tests for the RAG chunking module.
"""

import pytest
from langchain_core.documents import Document
from backend.rag.chunking.chunker import recursive_chunk


class TestRecursiveChunking:
    """Tests for recursive character chunking."""

    def test_basic_chunking(self):
        """Verify that documents get split into chunks."""
        docs = [
            Document(page_content="A" * 1000, metadata={"page": 0, "source": "test.pdf"}),
        ]
        chunks = recursive_chunk(docs, chunk_size=200, chunk_overlap=20)

        assert len(chunks) > 1, "Document should be split into multiple chunks"

    def test_metadata_preserved(self):
        """Verify that metadata is carried through to chunks."""
        docs = [
            Document(page_content="A" * 1000, metadata={"page": 5, "source": "test.pdf"}),
        ]
        chunks = recursive_chunk(docs, chunk_size=200, chunk_overlap=20)

        for chunk in chunks:
            assert chunk.metadata["source"] == "test.pdf"
            assert chunk.metadata["page"] == 5

    def test_empty_input(self):
        """Verify empty input returns empty output."""
        chunks = recursive_chunk([], chunk_size=200, chunk_overlap=20)
        assert chunks == []

    def test_small_document_no_split(self):
        """A document smaller than chunk_size should not be split."""
        docs = [
            Document(page_content="Short text", metadata={"page": 0}),
        ]
        chunks = recursive_chunk(docs, chunk_size=500, chunk_overlap=50)

        assert len(chunks) == 1
        assert chunks[0].page_content == "Short text"
