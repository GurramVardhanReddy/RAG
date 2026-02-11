"""
Document Loading Service

Handles loading documents from various file formats.
"""

from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from loguru import logger


SUPPORTED_LOADERS = {
    ".pdf": PyPDFLoader,
    # ".docx": Add docx loader when needed
    # ".txt": Add text loader when needed
}


def load_document(file_path: str) -> list[Document]:
    """
    Load a document from file path.
    Automatically selects the correct loader based on file extension.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = path.suffix.lower()
    loader_class = SUPPORTED_LOADERS.get(extension)

    if loader_class is None:
        raise ValueError(f"Unsupported file type: {extension}. Supported: {list(SUPPORTED_LOADERS.keys())}")

    loader = loader_class(str(path))
    documents = loader.load()

    logger.info(f"Loaded '{path.name}': {len(documents)} pages")
    return documents
