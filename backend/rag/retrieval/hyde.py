"""
HyDE — Hypothetical Document Embeddings

Instead of embedding the raw query, HyDE asks an LLM to generate a
hypothetical document passage that *would* answer the query, then
embeds that passage for retrieval.

Why it works:
    Queries and documents live in different embedding spaces — a question
    like "Does the fund promote EU sales?" is far from a document passage
    like "Shares are fully transferable and listed on stock exchanges."
    A hypothesis like "The fund promotes its shares to EU public investors
    through stock exchange listings" bridges that gap.

Key benefit for hybrid search:
    The hypothesis contains relevant keywords ("promote", "EU", "investors"),
    so SPLADE now matches the *correct* chunks instead of false positives
    like the MSCI "PROMOTED BY" disclaimer.

Reference:
    Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance
    Labels" (ACL 2023). https://arxiv.org/abs/2212.10496
"""

from openai import OpenAI
from loguru import logger


UCITS_HYPOTHESIS_PROMPT = """You are a UCITS fund document expert.

Given a compliance question about a fund supplement, write a short passage
(2-4 sentences) that would appear verbatim in a UCITS fund supplement
document and directly answers the question.

Rules:
- Write as actual document text, NOT as a question answer
- Use formal financial/legal language typical of fund prospectuses
- Include specific terms from the question (EU, promote, shares, etc.)
- Keep it concise — 2-4 sentences maximum

Question: {query}

Hypothetical document passage:"""


def generate_hypothesis(
    query: str,
    api_key: str,
    model: str = "gpt-4o",
    temperature: float = 0.5,
) -> str:
    """
    Generate a hypothetical document passage that would answer the query.

    Args:
        query: The user's question
        api_key: OpenAI API key
        model: LLM to use for hypothesis generation
        temperature: Higher = more varied hypotheses

    Returns:
        A short passage written as if from the source document
    """
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": UCITS_HYPOTHESIS_PROMPT.format(query=query),
        }],
    )

    hypothesis = response.choices[0].message.content.strip()
    logger.info(f"HyDE hypothesis: {hypothesis[:100]}...")
    return hypothesis
