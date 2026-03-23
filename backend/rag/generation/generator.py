"""
Generation Module

Handles LLM-based answer generation from retrieved context.
Includes page and paragraph number attribution for regulatory compliance.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from loguru import logger


# RAG prompt designed for UCITS compliance with page + paragraph attribution
RAG_PROMPT_TEMPLATE = """You are an expert financial compliance assistant specializing in UCITS regulations.
Your task is to answer questions from the Section 10 UCITS Application Form using the provided document context.

Rules:
1. Extract the answer from the context, even if the exact wording differs from the question.
2. For Yes/No questions, answer Yes or No first, then provide the supporting evidence.
3. For "State..." or "Disclose..." questions, extract the specific information requested.
4. If the information is partially available, provide what you can find and note what's missing.
5. Always cite which source you used by referencing [Source X] (Page Y, Para Z).
6. Only say "Information not found in the provided document" if the information is genuinely absent from ALL provided context.

Context:
{context}

Question:
{question}

Answer:"""


def build_context(
    search_results: list[dict],
) -> str:
    """
    Format retrieved chunks into a numbered context string
    with page AND paragraph numbers for provenance tracking.

    Each source shows:
    [Source 1] (Page 3, Para 2, relevance: 0.87):
    <chunk text>
    """
    context_parts = []
    for i, result in enumerate(search_results):
        page = result.get("page_number", "?")
        para = result.get("paragraph_number", "?")
        score = result.get("score", 0.0)

        # Build location string
        location_parts = []
        if page is not None and page >= 0:
            location_parts.append(f"Page {page + 1}")  # 0-indexed to 1-indexed
        if para is not None and para > 0:
            location_parts.append(f"Para {para}")
        location_parts.append(f"relevance: {score:.4f}")
        location_str = ", ".join(location_parts)

        context_parts.append(
            f"[Source {i + 1}] ({location_str}):\n{result['text']}"
        )
    return "\n\n".join(context_parts)


def generate_answer(
    question: str,
    search_results: list[dict],
    model_name: str = "gpt-4o",
    temperature: float = 0,
    api_key: str = "",
) -> str:
    """
    Generate an answer using the LLM with retrieved context.

    Args:
        question: The user's question
        search_results: List of dicts from vector_store.search()
            Each has: text, page_number, paragraph_number, score
        model_name: OpenAI model to use
        temperature: LLM temperature (0 for factual)
        api_key: OpenAI API key

    Returns:
        Generated answer string with source citations
    """
    # Build the context string with page + paragraph numbers
    context = build_context(search_results)

    # Setup the LLM chain
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
    )
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()

    # Generate
    logger.info(f"Generating answer with {model_name} ({len(search_results)} sources)")
    response = chain.invoke({
        "context": context,
        "question": question,
    })

    return response
