"""
Document Chunking Module

Supports multiple chunking strategies:
- Recursive character splitting (fast, simple)
- Semantic chunking (slower, better quality — groups by meaning)
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from loguru import logger


def recursive_chunk(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """
    Split documents using recursive character splitting.
    Fast and reliable — good default for most use cases.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Recursive chunking: {len(documents)} docs → {len(chunks)} chunks")
    return chunks


def semantic_chunk(
    documents: list[Document],
    embedding_model,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float = 70.0,
) -> list[Document]:
    """
    Split documents based on semantic similarity between sentences.
    
    How it works:
    1. Splits text into sentences
    2. Embeds each sentence
    3. Compares adjacent sentence embeddings
    4. Splits where the semantic similarity drops significantly
    
    This produces chunks where all sentences in a chunk are about the 
    same topic — much better for retrieval quality.
    
    Args:
        documents: List of Document objects to chunk
        embedding_model: The embedding model (same one used for retrieval)
        breakpoint_threshold_type: How to determine split points
            - "percentile": Split at the Xth percentile of dissimilarity (default)
            - "standard_deviation": Split when gap exceeds X standard deviations
            - "interquartile": Split using IQR method
        breakpoint_threshold_amount: Threshold value for the chosen method
            - For "percentile": 70.0 means split at top 30% biggest gaps
            - Lower = more chunks, Higher = fewer chunks
    """
    splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
    )
    chunks = splitter.split_documents(documents)
    
    # Log chunk statistics
    chunk_lengths = [len(c.page_content) for c in chunks]
    avg_len = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
    
    logger.info(
        f"Semantic chunking: {len(documents)} docs → {len(chunks)} chunks "
        f"(avg size: {avg_len:.0f} chars, min: {min(chunk_lengths, default=0)}, "
        f"max: {max(chunk_lengths, default=0)})"
    )
    return chunks