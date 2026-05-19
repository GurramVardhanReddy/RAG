"""
Multimodal Document Processor

Extracts tables, graphs, and images from PDF documents using
the 'unstructured' library, generates brief but comprehensive
summaries via GPT-4o, and prepares them for embedding into
Qdrant alongside regular text chunks.

This transforms the pipeline from text-only RAG → multimodal RAG,
following the approach described in Joshi et al. (ICAAIC 2024).
"""

import os
import base64
import uuid
from typing import Optional
from openai import OpenAI
from langchain_core.documents import Document
from loguru import logger


# ── Summary Prompts ──────────────────────────────────────────────

TABLE_SUMMARY_PROMPT = """You are a financial document analyst. Analyze this table and provide a BRIEF but COMPLETE summary (3-5 sentences max).

Include:
- What the table shows (purpose/title)
- Key data points, values, and categories
- Any notable patterns, totals, or comparisons

Be factual and specific. Include actual numbers and labels from the table.
Do NOT add opinions or external context. Output plain text only."""

IMAGE_SUMMARY_PROMPT = """You are a financial document analyst. Analyze this image/graph/chart and provide a BRIEF but COMPLETE summary (3-5 sentences max).

Include:
- What type of visual this is (bar chart, flow diagram, pie chart, etc.)
- What it represents (title, axes, categories)
- Key data points, trends, or relationships shown
- Any notable peaks, dips, or outliers

Be factual and specific. Include actual numbers and labels visible in the image.
Do NOT add opinions or external context. Output plain text only."""


class MultimodalProcessor:
    """
    Extracts and summarizes visual elements (tables, images, graphs)
    from PDF documents for inclusion in the RAG vector store.
    """

    def __init__(self, openai_api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        logger.info(f"MultimodalProcessor initialized (model: {model})")

    # ── Core Processing ──────────────────────────────────────────

    def process_pdf(
        self,
        file_path: str,
        output_dir: str = "data/extracted_visuals",
    ) -> list[Document]:
        """
        Extract all visual elements from a PDF and generate summaries.

        Args:
            file_path: Path to the PDF file
            output_dir: Directory to save extracted images

        Returns:
            List of LangChain Document objects, each containing a
            summary as page_content and rich metadata for explainability.
        """
        from unstructured.partition.pdf import partition_pdf

        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Extracting visual elements from: {os.path.basename(file_path)}")

        # Use unstructured's hi_res strategy for layout detection
        elements = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=True,
            extract_image_block_output_dir=output_dir,
            extract_image_block_types=["Image", "Table"],
        )

        visual_documents = []
        table_counter = {}   # per-page table numbering
        image_counter = {}   # per-page image numbering

        for el in elements:
            page_num = getattr(el.metadata, "page_number", 0) or 0

            # ── Tables ───────────────────────────────────────────
            if el.category == "Table":
                table_counter[page_num] = table_counter.get(page_num, 0) + 1
                table_num = table_counter[page_num]

                logger.info(f"  [TABLE] Page {page_num}, Table #{table_num}")

                table_html = getattr(el.metadata, "text_as_html", None)
                image_path = getattr(el.metadata, "image_path", None)

                summary = self._summarize_table(table_html, image_path)

                if summary:
                    # Prefix summary with context for better retrieval
                    content = (
                        f"[Table {table_num} on Page {page_num}] "
                        f"{summary}"
                    )
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": file_path,
                            "page": page_num,
                            "content_type": "table",
                            "table_number": table_num,
                            "paragraph_number": -1,
                            "has_html": bool(table_html),
                            "image_path": image_path or "",
                        },
                    )
                    visual_documents.append(doc)

            # ── Images / Graphs ──────────────────────────────────
            elif el.category == "Image":
                image_counter[page_num] = image_counter.get(page_num, 0) + 1
                fig_num = image_counter[page_num]

                image_path = getattr(el.metadata, "image_path", None)
                if not image_path or not os.path.exists(image_path):
                    logger.warning(f"  [IMAGE] Page {page_num} — no file, skipping")
                    continue

                logger.info(f"  [IMAGE] Page {page_num}, Figure #{fig_num}")

                summary = self._summarize_image(image_path)

                if summary:
                    content = (
                        f"[Figure {fig_num} on Page {page_num}] "
                        f"{summary}"
                    )
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": file_path,
                            "page": page_num,
                            "content_type": "image",
                            "figure_number": fig_num,
                            "paragraph_number": -1,
                            "image_path": image_path,
                        },
                    )
                    visual_documents.append(doc)

        logger.info(
            f"Multimodal extraction complete: "
            f"{sum(table_counter.values())} tables, "
            f"{sum(image_counter.values())} images → "
            f"{len(visual_documents)} summaries generated"
        )
        return visual_documents

    # ── GPT-4o Summarization ─────────────────────────────────────

    def _summarize_table(
        self,
        table_html: Optional[str],
        image_path: Optional[str] = None,
    ) -> Optional[str]:
        """Generate a brief summary of a table using GPT-4o."""
        try:
            # Prefer image-based analysis (captures formatting/context)
            if image_path and os.path.exists(image_path):
                return self._call_vision(IMAGE_SUMMARY_PROMPT, image_path)

            # Fall back to HTML-based analysis
            if table_html:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    max_tokens=300,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"{TABLE_SUMMARY_PROMPT}\n\n"
                            f"Table HTML:\n{table_html}"
                        ),
                    }],
                )
                return response.choices[0].message.content.strip()

            return None

        except Exception as e:
            logger.error(f"Table summarization failed: {e}")
            return None

    def _summarize_image(self, image_path: str) -> Optional[str]:
        """Generate a brief summary of an image/graph using GPT-4o vision."""
        try:
            return self._call_vision(IMAGE_SUMMARY_PROMPT, image_path)
        except Exception as e:
            logger.error(f"Image summarization failed: {e}")
            return None

    def _call_vision(self, prompt: str, image_path: str) -> str:
        """Send an image to GPT-4o vision and get a text summary."""
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        # Detect MIME type from extension
        ext = os.path.splitext(image_path)[1].lower()
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(ext, "image/png")

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64}",
                            "detail": "high",
                        },
                    },
                ],
            }],
        )
        return response.choices[0].message.content.strip()


# ── Integration Helper ───────────────────────────────────────────

def extract_and_embed_visuals(
    file_path: str,
    document_id: str,
    openai_api_key: str,
    embedding_model,
    vector_store,
    sparse_encoder=None,
    output_dir: str = "data/extracted_visuals",
) -> dict:
    """
    End-to-end helper: extract visuals → summarize → embed → store.

    Call this from your upload endpoint AFTER processing text chunks.

    Args:
        file_path: Path to the uploaded PDF
        document_id: Same document_id used for text chunks
        openai_api_key: OpenAI API key from settings
        embedding_model: The same SentenceTransformer embedding model
        vector_store: Your VectorStoreService instance
        sparse_encoder: Your SPLADE sparse encoder (from app.state.sparse_encoder)
        output_dir: Where to save extracted image files

    Returns:
        Dict with stats: {"tables": N, "images": N, "chunks_added": N}
    """
    # Only process PDFs (images don't exist in .txt or .docx via this method)
    ext = os.path.splitext(file_path)[1].lower()
    if ext != ".pdf":
        logger.info(f"Skipping multimodal extraction for {ext} file")
        return {"tables": 0, "images": 0, "chunks_added": 0}

    processor = MultimodalProcessor(openai_api_key)

    # 1. Extract and summarize
    visual_docs = processor.process_pdf(file_path, output_dir)

    if not visual_docs:
        logger.info("No visual elements found in document")
        return {"tables": 0, "images": 0, "chunks_added": 0}

    # 2. Generate DENSE embeddings for the summaries
    summary_texts = [doc.page_content for doc in visual_docs]
    dense_vectors = embedding_model.embed_documents(summary_texts)

    # 3. Generate SPARSE embeddings (SPLADE) for the summaries
    sparse_vectors = None
    if sparse_encoder is not None:
        try:
            sparse_vectors = sparse_encoder.encode_sparse(summary_texts)
            logger.info(f"Generated {len(sparse_vectors)} sparse vectors for visual summaries")
        except Exception as e:
            logger.warning(f"Sparse encoding for visuals failed: {e}")

    # 4. Store in Qdrant with NAMED vectors (matching hybrid collection schema)
    from qdrant_client.models import PointStruct, SparseVector

    points = []
    for i, (doc, dense_vec) in enumerate(zip(visual_docs, dense_vectors)):
        # Build named vector dict — must match your collection's vector config
        vector_data = {"dense": dense_vec}

        # Add sparse vector if available
        if sparse_vectors and i < len(sparse_vectors):
            sv = sparse_vectors[i]
            # Handle both dict format and SparseVector format
            if isinstance(sv, dict):
                vector_data["sparse"] = SparseVector(
                    indices=sv.get("indices", []),
                    values=sv.get("values", []),
                )
            else:
                vector_data["sparse"] = sv

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector_data,
            payload={
                "text": doc.page_content,
                "page_number": doc.metadata.get("page", -1),
                "paragraph_number": doc.metadata.get("paragraph_number", -1),
                "paragraph_numbers": [],
                "paragraph_match_type": "visual_element",
                "content_type": doc.metadata.get("content_type", "visual"),
                "table_number": doc.metadata.get("table_number", -1),
                "figure_number": doc.metadata.get("figure_number", -1),
                "source": doc.metadata.get("source", ""),
                "document_id": document_id,
                "chunk_index": -(i + 1),  # Negative index to distinguish from text
                "image_path": doc.metadata.get("image_path", ""),
            },
        )
        points.append(point)

    # Upload to Qdrant
    vector_store.client.upsert(
        collection_name=vector_store.collection_name,
        points=points,
    )

    # Count by type
    n_tables = sum(1 for d in visual_docs if d.metadata.get("content_type") == "table")
    n_images = sum(1 for d in visual_docs if d.metadata.get("content_type") == "image")

    logger.info(
        f"Multimodal indexing complete: {n_tables} tables + {n_images} images "
        f"= {len(points)} visual chunks added to Qdrant"
    )

    return {
        "tables": n_tables,
        "images": n_images,
        "chunks_added": len(points),
    }