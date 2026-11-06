"""
Generation Module

Handles LLM-based answer generation from retrieved context.
Supports two prompt modes:
    - "domain" (default): UCITS compliance with page + paragraph attribution
    - "general": General-purpose QA for standard benchmark evaluation

References:
    - Lewis et al., "RAG for Knowledge-Intensive NLP Tasks" (NeurIPS 2020)
    - Salemi & Zamani, "Evaluating Retrieval Quality in RAG" (SIGIR 2024)
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from loguru import logger


# ============================================================
# PROMPT TEMPLATES
# ============================================================

# Domain-specific: UCITS compliance with page + paragraph attribution
DOMAIN_PROMPT_TEMPLATE = """You are a UCITS compliance expert reviewing a fund supplement document.
Answer the question using ONLY the provided context.

Some context chunks are text from the document. Others are summaries of tables, charts, or images
(prefixed with [Table X on Page Y] or [Figure X on Page Y]). Treat both equally as valid sources.

Key UCITS domain knowledge to apply when interpreting context:
- A fund described as a "UCITS ETF" is by definition authorised for public sale within the EU/EEA.
- Shares "listed for trading on one or more stock exchanges" and "fully transferable to investors" implies public promotion within the EU under UCITS regulations.
- For Yes/No compliance questions, answer Yes or No first, then cite the supporting evidence.
- Always cite [Source X] (Page Y, Para Z) for every claim.
- Only say "not found" if the information is genuinely absent from ALL provided context.

Context:
{context}

Question: {question}

Answer:"""


# General-purpose: For standard benchmark evaluation (NQ, HotpotQA, FEVER, etc.)
GENERAL_PROMPT_TEMPLATE = """Answer the question using ONLY the provided context.
Be concise and direct — provide the shortest correct answer possible.
If the context does not contain enough information to answer, say "Information not found".

Context:
{context}

Question:
{question}

Answer:"""


# Fact verification: For FEVER-style claims
# VERIFICATION_PROMPT_TEMPLATE = """You are evaluating whether a claim is supported by the provided evidence.
# Based ONLY on the context below, classify the claim as one of:
# - SUPPORTS (the context confirms the claim)
# - REFUTES (the context contradicts the claim)
# - NOT ENOUGH INFO (the context does not contain sufficient information)

# Respond with ONLY the label (SUPPORTS, REFUTES, or NOT ENOUGH INFO), followed by a brief explanation.

# Context:
# {context}

# Claim:
# {question}

# Verdict:"""
VERIFICATION_PROMPT_TEMPLATE = """You are a fact-checking system. Based ONLY on the context below, classify the claim.

Respond with EXACTLY ONE of these labels and nothing else:
SUPPORTS
REFUTES
NOT ENOUGH INFO

Context:
{context}

Claim:
{question}

Label:"""

# BoolQ: Yes/No questions — output True or False only
BOOLQ_PROMPT_TEMPLATE = """Answer the yes/no question using ONLY the provided context.
Respond with EXACTLY one word: True or False.
If the context does not contain enough information, respond with: False

Context:
{context}

Question:
{question}

Answer:"""


# Map prompt mode names to templates
PROMPT_TEMPLATES = {
    "domain": DOMAIN_PROMPT_TEMPLATE,
    "general": GENERAL_PROMPT_TEMPLATE,
    "verification": VERIFICATION_PROMPT_TEMPLATE,
    "boolq": BOOLQ_PROMPT_TEMPLATE,
}


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
    prompt_mode: str = "domain",
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
        prompt_mode: Which prompt template to use:
            - "domain": UCITS compliance (default, used by main app)
            - "general": Concise QA for benchmarks (NQ, HotpotQA, TriviaQA)
            - "verification": Claim verification for FEVER

    Returns:
        Generated answer string with source citations
    """
    # Build the context string with page + paragraph numbers
    context = build_context(search_results)

    # Select prompt template
    template = PROMPT_TEMPLATES.get(prompt_mode, DOMAIN_PROMPT_TEMPLATE)

    # Setup the LLM chain
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
    )
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    # Generate
    logger.info(f"Generating answer with {model_name} ({len(search_results)} sources, mode={prompt_mode})")
    response = chain.invoke({
        "context": context,
        "question": question,
    })

    return response
