"""
Generation Module

Handles LLM-based answer generation from retrieved context.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from loguru import logger


# Default RAG prompt — designed to encourage grounded, source-attributed answers
RAG_PROMPT_TEMPLATE = RAG_PROMPT_TEMPLATE = """You are an expert financial compliance assistant specializing in UCITS regulations.
Your task is to answer questions from the Section 10 UCITS Application Form using the provided document context.

Rules:
1. Extract the answer from the context, even if the exact wording differs from the question.
2. For Yes/No questions, answer Yes or No first, then provide the supporting evidence.
3. For "State..." or "Disclose..." questions, extract the specific information requested.
4. If the information is partially available, provide what you can find and note what's missing.
5. Always cite which part of the context supports your answer using [Source X].
6. Only say "not found" if the information is genuinely absent from ALL provided context.

Context:
{context}

Question:
{question}

Answer:"""


def build_context(documents: list[Document], scores: list[float] = None) -> str:
    """
    Format retrieved documents into a numbered context string.
    This makes it easy for the LLM to reference specific sources.
    """
    context_parts = []
    for i, doc in enumerate(documents):
        page = doc.metadata.get("page", "unknown")
        score_info = f" (relevance: {scores[i]:.4f})" if scores else ""
        context_parts.append(
            f"[Source {i + 1}] (Page {page}{score_info}):\n{doc.page_content}"
        )
    return "\n\n".join(context_parts)


def generate_answer(
    question: str,
    documents: list[Document],
    scores: list[float] = None,
    model_name: str = "gpt-4o",
    temperature: float = 0,
    api_key: str = "",
) -> str:
    """
    Generate an answer using the LLM with retrieved context.
    """
    # Build the context string
    context = build_context(documents, scores)

    # Setup the chain
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
    )
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()

    # Generate
    logger.info(f"Generating answer with {model_name} ({len(documents)} sources)")
    response = chain.invoke({
        "context": context,
        "question": question,
    })

    return response
