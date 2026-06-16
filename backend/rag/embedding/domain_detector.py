"""
Domain Detector — Automatic Document Domain Classification with XAI
====================================================================
Identifies the domain of an uploaded document and selects the appropriate
domain-specific embedding model. Also produces XAI explanations for WHY
the domain was selected (token attribution via keyword importance scoring).

NOVEL RESEARCH CONTRIBUTION:
    Automatic domain-adaptive embedding selection in a RAG pipeline.
    No existing RAG framework dynamically routes to domain-specific
    pre-trained embedding models based on document content at upload time.

Domain → Embedding Model Mapping:
    Finance/Regulatory  → nickmuchi/sec-bert-base         (SEC filings, regulatory docs)
    Legal               → nlpaueb/legal-bert-base-uncased (EU/US legal corpora)
    Medical/Biomedical  → dmis-lab/biobert-base-cased-v1.2 (PubMed/clinical)
    Scientific/Academic → allenai/scibert_scivocab_uncased (arXiv/scientific papers)
    Code/Software       → microsoft/codebert-base          (code + docstrings)
    E-commerce/Product  → sentence-transformers/multi-qa-mpnet-base-dot-v1
    General (fallback)  → BAAI/bge-m3                     (multilingual, general)

XAI Method:
    Keyword Attribution — a lightweight, interpretable technique inspired by
    LIME/SHAP. Each word in the document sample is scored by how strongly it
    signals a particular domain based on a domain keyword lexicon.
    The top contributing keywords are returned as the explanation for the
    domain decision, making the classification transparent and auditable.

References:
    - Ribeiro et al. (2016) — LIME (model-agnostic local explanations)
    - Lundberg & Lee (2017) — SHAP (unified feature attribution)
    - Adadi & Berrada (2018) — Survey on XAI (Peeking Inside the Black Box)
"""

import re
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger


# ============================================================
# DOMAIN CONFIGURATION
# ============================================================

DOMAIN_MODEL_MAP: dict[str, str] = {
    # All models verified as sentence-transformers compatible
    # (fine-tuned for embedding/retrieval, NOT classification)
    "Finance/Regulatory":  "FinLang/finance-embeddings-investopedia",          # Fine-tuned on BGE + Investopedia finance data, 768d
    "Legal":               "Stern5497/sbert-legal-xlm-roberta-base",           # SBERT on legal corpora, XLM-RoBERTa backbone, 768d
    "Medical/Biomedical":  "NeuML/pubmedbert-base-embeddings",                 # PubMedBERT fine-tuned for PubMed retrieval, 768d
    "Scientific/Academic": "allenai/scibert_scivocab_uncased",                 # SciBERT trained on scientific papers, 768d
    "Code/Software":       "jinaai/jina-embeddings-v2-base-code",              # 150M+ code QA pairs, 30 languages, 768d (needs trust_remote_code=True)
    "E-commerce/Product":  "sentence-transformers/multi-qa-mpnet-base-dot-v1", # Trained on QA retrieval pairs, 768d
    "General":             "BAAI/bge-base-en-v1.5",                            # Strong general retrieval baseline, 768d
}

DOMAIN_DIMENSIONS: dict[str, int] = {
    "Finance/Regulatory": 768,
    "Legal":              768,
    "Medical/Biomedical": 768,
    "Scientific/Academic":768,
    "Code/Software":      768,
    "E-commerce/Product": 768,
    "General":            768,
}

# Domain keyword lexicons — used for both GPT-4o prompt grounding
# and XAI keyword attribution scoring.
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "Finance/Regulatory": [
        "ucits", "fund", "prospectus", "supplement", "sub-fund", "nav",
        "redemption", "subscription", "kiid", "aifmd", "mifid", "sec",
        "regulatory", "compliance", "portfolio", "investment", "equity",
        "bond", "dividend", "fiscal", "audit", "financial", "revenue",
        "asset", "liability", "hedge", "derivative", "etf", "management fee",
        "custodian", "depository", "aum", "annual report", "balance sheet",
        "income statement", "cash flow", "shareholder", "prospectus",
    ],
    "Legal": [
        "plaintiff", "defendant", "jurisdiction", "statute", "legislation",
        "regulation", "clause", "provision", "contract", "agreement",
        "liability", "indemnify", "arbitration", "litigation", "tort",
        "breach", "damages", "court", "judgment", "precedent", "affidavit",
        "deposition", "counsel", "attorney", "barrister", "solicitor",
        "hereby", "whereas", "notwithstanding", "pursuant", "hereinafter",
        "subpoena", "injunction", "warrant", "appeal", "testimony",
    ],
    "Medical/Biomedical": [
        "patient", "clinical", "diagnosis", "treatment", "therapy",
        "disease", "symptom", "medication", "drug", "trial", "cohort",
        "biomarker", "genome", "protein", "cell", "tissue", "organ",
        "pathology", "oncology", "surgery", "pharmacology", "dose",
        "randomized", "placebo", "adverse", "efficacy", "mrna", "antibody",
        "receptor", "enzyme", "mutation", "biopsy", "radiology", "ehr",
        "icd", "cpt", "emr", "hospital", "physician", "nurse",
    ],
    "Scientific/Academic": [
        "abstract", "methodology", "hypothesis", "experiment", "dataset",
        "results", "conclusion", "literature", "citation", "peer-reviewed",
        "journal", "arxiv", "doi", "algorithm", "model", "baseline",
        "evaluation", "benchmark", "accuracy", "precision", "recall",
        "ablation", "neural", "transformer", "bert", "gpt", "llm",
        "embedding", "retrieval", "rag", "nlp", "machine learning",
        "deep learning", "research", "study", "analysis", "findings",
    ],
    "Code/Software": [
        "function", "class", "method", "variable", "import", "module",
        "library", "api", "endpoint", "request", "response", "database",
        "query", "sql", "python", "javascript", "java", "typescript",
        "docker", "kubernetes", "git", "repository", "branch", "commit",
        "bug", "exception", "error", "stack", "thread", "async", "await",
        "http", "rest", "graphql", "json", "xml", "yaml", "config",
        "deployment", "ci/cd", "test", "unit test", "integration",
    ],
    "E-commerce/Product": [
        "product", "price", "cart", "checkout", "order", "shipping",
        "delivery", "return", "refund", "customer", "review", "rating",
        "inventory", "sku", "catalog", "category", "brand", "seller",
        "buyer", "marketplace", "payment", "discount", "coupon", "promo",
        "wishlist", "recommendation", "upsell", "cross-sell", "fulfillment",
        "warehouse", "logistics", "tracking", "purchase", "subscription",
    ],
}


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class KeywordMatch:
    """A single keyword that contributed to the domain decision."""
    word: str
    domain: str
    importance_score: float     # 0.0–1.0 (normalised TF-IDF-style weight)
    frequency: int              # How many times it appeared in the sample
    context_snippet: str = ""   # Up to 80 chars of surrounding context


@dataclass
class DomainDetectionResult:
    """
    Full output of the domain detection step, including XAI attribution.
    Stored in the document registry and returned in the UploadResponse.
    """
    # ── Decision ──
    detected_domain: str = "General"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024
    confidence: float = 0.0         # 0.0–1.0

    # ── Domain scores (all 7 domains) ──
    domain_scores: dict = field(default_factory=dict)   # {domain: score}
    domain_probabilities: dict = field(default_factory=dict)  # softmax-normalised

    # ── XAI Attribution ──
    top_keywords: list = field(default_factory=list)    # List[KeywordMatch]
    xai_method: str = "keyword_attribution"
    xai_explanation: str = ""   # Human-readable explanation sentence

    # ── GPT-4o meta (if used) ──
    gpt_used: bool = False
    gpt_reasoning: str = ""

    # ── Text sample used for detection ──
    text_sample_chars: int = 0


# ============================================================
# CORE DETECTOR
# ============================================================

class DomainDetector:
    """
    Two-stage domain detector with XAI attribution.

    Stage 1 — Keyword Attribution (fast, always runs):
        Scores each domain by counting weighted keyword hits in a
        500-word sample of the document. Produces per-keyword importance
        scores for the XAI explanation.

    Stage 2 — GPT-4o Verification (optional, runs when Stage 1 is uncertain):
        When max keyword score is below the confidence threshold, GPT-4o
        classifies the text. Its reasoning is stored and shown to the user.

    The XAI output tells the user EXACTLY which words triggered the domain
    decision, making the embedding model selection fully auditable.
    """

    SAMPLE_WORDS = 500          # Words to extract for classification
    CONFIDENCE_THRESHOLD = 0.35 # Below this → escalate to GPT-4o
    MIN_KEYWORD_SCORE = 1e-6

    def __init__(self, openai_api_key: str = ""):
        self.openai_api_key = openai_api_key
        self._gpt_client = None

    def _get_gpt_client(self):
        """Lazy-load OpenAI client."""
        if self._gpt_client is None and self.openai_api_key:
            try:
                from openai import OpenAI
                self._gpt_client = OpenAI(api_key=self.openai_api_key)
            except Exception as e:
                logger.warning(f"Could not initialise OpenAI client: {e}")
        return self._gpt_client

    # ----------------------------------------------------------
    # PUBLIC API
    # ----------------------------------------------------------

    def detect(self, text: str) -> DomainDetectionResult:
        """
        Detect the domain of a document from its text content.

        Args:
            text: Raw text extracted from the uploaded document.

        Returns:
            DomainDetectionResult with domain, model, confidence, and XAI data.
        """
        result = DomainDetectionResult()

        # 1. Extract a representative sample
        sample = self._extract_sample(text)
        result.text_sample_chars = len(sample)

        # 2. Keyword attribution (Stage 1)
        keyword_scores, keyword_matches = self._keyword_attribution(sample)
        result.domain_scores = keyword_scores

        # Softmax-normalise scores to probabilities
        result.domain_probabilities = self._softmax(keyword_scores)

        # 3. Pick best domain
        best_domain = max(keyword_scores, key=keyword_scores.get)
        best_score = keyword_scores[best_domain]
        normalised_confidence = result.domain_probabilities[best_domain]

        # 4. If confidence is low, escalate to GPT-4o
        if normalised_confidence < self.CONFIDENCE_THRESHOLD:
            logger.info(
                f"Low keyword confidence ({normalised_confidence:.3f}) — "
                f"escalating to GPT-4o for domain classification"
            )
            gpt_domain, gpt_reasoning = self._gpt_classify(sample)
            if gpt_domain:
                best_domain = gpt_domain
                result.gpt_used = True
                result.gpt_reasoning = gpt_reasoning
                # Boost GPT-chosen domain score for recalculation
                keyword_scores[best_domain] = max(keyword_scores.values()) * 1.5
                result.domain_probabilities = self._softmax(keyword_scores)
                normalised_confidence = result.domain_probabilities[best_domain]

        # 5. Fill result
        result.detected_domain = best_domain
        result.embedding_model = DOMAIN_MODEL_MAP[best_domain]
        result.embedding_dimension = DOMAIN_DIMENSIONS[best_domain]
        result.confidence = round(normalised_confidence, 4)

        # 6. XAI: top keywords that drove the decision
        domain_keywords = [m for m in keyword_matches if m.domain == best_domain]
        domain_keywords.sort(key=lambda m: m.importance_score, reverse=True)
        result.top_keywords = domain_keywords[:15]  # Top 15

        # 7. Human-readable XAI explanation
        result.xai_explanation = self._build_explanation(result)

        logger.info(
            f"Domain detected: '{best_domain}' "
            f"(confidence={result.confidence:.3f}, "
            f"model={result.embedding_model}, "
            f"gpt_used={result.gpt_used})"
        )

        return result

    # ----------------------------------------------------------
    # STAGE 1: KEYWORD ATTRIBUTION (XAI core)
    # ----------------------------------------------------------

    def _extract_sample(self, text: str) -> str:
        """
        Extract a representative word sample from the document.
        Takes words from beginning, middle, and end to avoid bias
        from introductory boilerplate.
        """
        words = text.split()
        n = len(words)
        if n <= self.SAMPLE_WORDS:
            return text

        third = self.SAMPLE_WORDS // 3
        start = words[:third]
        mid_start = n // 2 - third // 2
        middle = words[mid_start: mid_start + third]
        end = words[-(self.SAMPLE_WORDS - 2 * third):]

        return " ".join(start + middle + end)

    def _keyword_attribution(
        self, sample: str
    ) -> tuple[dict[str, float], list[KeywordMatch]]:
        """
        Score each domain by keyword frequency-weighted attribution.

        Scoring:
            - TF: how often the keyword appears in the sample
            - IDF-like weight: keywords unique to fewer domains score higher
            - Final score = sum(TF × IDF_weight) for all keywords in domain

        Returns:
            domain_scores: {domain: float} raw aggregated scores
            keyword_matches: List[KeywordMatch] for XAI attribution
        """
        sample_lower = sample.lower()
        word_counts = Counter(re.findall(r"\b\w[\w\-/]*\b", sample_lower))
        total_words = max(sum(word_counts.values()), 1)

        # Count how many domains each keyword appears in (for IDF)
        keyword_domain_count: dict[str, int] = {}
        for kws in DOMAIN_KEYWORDS.values():
            for kw in kws:
                keyword_domain_count[kw] = keyword_domain_count.get(kw, 0) + 1

        domain_scores: dict[str, float] = {d: 0.0 for d in DOMAIN_KEYWORDS}
        all_matches: list[KeywordMatch] = []

        for domain, keywords in DOMAIN_KEYWORDS.items():
            for kw in keywords:
                # Count occurrences (support multi-word keywords)
                if " " in kw:
                    freq = sample_lower.count(kw)
                else:
                    freq = word_counts.get(kw, 0)

                if freq == 0:
                    continue

                # TF (log-normalised)
                tf = math.log(1 + freq)

                # IDF-like: keyword appearing in only 1 domain is more informative
                n_domains = keyword_domain_count.get(kw, 1)
                idf = math.log(len(DOMAIN_KEYWORDS) / n_domains)

                score = tf * idf
                domain_scores[domain] += score

                # Find context snippet
                idx = sample_lower.find(kw)
                snippet = ""
                if idx >= 0:
                    start = max(0, idx - 30)
                    end = min(len(sample_lower), idx + len(kw) + 50)
                    snippet = "..." + sample[start:end].strip() + "..."

                all_matches.append(KeywordMatch(
                    word=kw,
                    domain=domain,
                    importance_score=round(score, 6),
                    frequency=freq,
                    context_snippet=snippet[:120],
                ))

        # Normalise importance scores within each domain to 0–1 range
        for domain in DOMAIN_KEYWORDS:
            domain_hits = [m for m in all_matches if m.domain == domain]
            if domain_hits:
                max_score = max(m.importance_score for m in domain_hits)
                if max_score > 0:
                    for m in domain_hits:
                        m.importance_score = round(m.importance_score / max_score, 4)

        return domain_scores, all_matches

    def _softmax(self, scores: dict[str, float]) -> dict[str, float]:
        """Convert raw scores to probabilities via softmax."""
        values = list(scores.values())
        max_v = max(values) if values else 0
        exp_values = [math.exp(v - max_v) for v in values]
        total = sum(exp_values)
        if total == 0:
            n = len(scores)
            return {k: 1.0 / n for k in scores}
        return {
            k: round(e / total, 6)
            for k, e in zip(scores.keys(), exp_values)
        }

    # ----------------------------------------------------------
    # STAGE 2: GPT-4o VERIFICATION
    # ----------------------------------------------------------

    def _gpt_classify(self, sample: str) -> tuple[Optional[str], str]:
        """
        Use GPT-4o to classify the domain when keyword confidence is low.
        Returns (domain_name, reasoning) or (None, "") on failure.
        """
        client = self._get_gpt_client()
        if client is None:
            logger.warning("GPT-4o not available — using keyword result as-is")
            return None, ""

        domain_list = "\n".join(f"- {d}" for d in DOMAIN_MODEL_MAP.keys())
        prompt = f"""You are a document domain classifier. Given a text sample, identify its domain.

Available domains:
{domain_list}

Instructions:
1. Read the text sample carefully.
2. Choose EXACTLY ONE domain from the list above.
3. Respond in this exact JSON format (no markdown, no extra text):
{{"domain": "<domain name>", "reasoning": "<one sentence explaining key indicators>"}}

Text sample:
---
{sample[:1500]}
---"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content.strip()

            import json
            # Strip markdown code fences if present
            content = re.sub(r"```json|```", "", content).strip()
            parsed = json.loads(content)
            domain = parsed.get("domain", "").strip()
            reasoning = parsed.get("reasoning", "").strip()

            if domain in DOMAIN_MODEL_MAP:
                logger.info(f"GPT-4o classified domain as '{domain}': {reasoning}")
                return domain, reasoning
            else:
                logger.warning(f"GPT-4o returned unknown domain: '{domain}'")
                return None, ""

        except Exception as e:
            logger.warning(f"GPT-4o domain classification failed: {e}")
            return None, ""

    # ----------------------------------------------------------
    # XAI EXPLANATION BUILDER
    # ----------------------------------------------------------

    def _build_explanation(self, result: DomainDetectionResult) -> str:
        """
        Build a human-readable explanation of WHY the domain was selected.
        This is the XAI output shown in the UI and stored for audit.
        """
        top_words = [m.word for m in result.top_keywords[:5]]
        confidence_pct = round(result.confidence * 100, 1)

        if result.gpt_used:
            base = (
                f"Domain classified as '{result.detected_domain}' "
                f"(confidence: {confidence_pct}%) using GPT-4o verification. "
                f"GPT-4o reasoning: {result.gpt_reasoning} "
            )
        else:
            base = (
                f"Domain classified as '{result.detected_domain}' "
                f"(confidence: {confidence_pct}%) via keyword attribution. "
            )

        if top_words:
            base += (
                f"Top triggering keywords: {', '.join(top_words)}. "
            )

        base += (
            f"Embedding model selected: '{result.embedding_model}' "
            f"(vector dimension: {result.embedding_dimension}d). "
            f"XAI method: keyword TF-IDF attribution across {len(DOMAIN_KEYWORDS)} domain lexicons."
        )

        return base


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def detect_domain(text: str, openai_api_key: str = "") -> DomainDetectionResult:
    """
    Convenience wrapper — detect domain from document text.

    Args:
        text: Raw document text (full or partial)
        openai_api_key: Optional — enables GPT-4o fallback for low-confidence cases

    Returns:
        DomainDetectionResult with domain, model, confidence, and XAI attribution
    """
    detector = DomainDetector(openai_api_key=openai_api_key)
    return detector.detect(text)