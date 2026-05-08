"""
embedding_matching.py — Hybrid Resume Retrieval Engine
======================================================
Contains two retrieval approaches:

1. HybridResumeRetriever (NEW — recommended)
   - Loads pre-built FAISS + BM25 indexes from disk
   - Performs dual-tower retrieval: semantic (FAISS) + keyword (BM25)
   - Fuses results using Reciprocal Rank Fusion (RRF)

2. ResumeJDMatcher (LEGACY — preserved for backward compatibility)
   - Original on-the-fly semantic matching
   - Deprecated: use HybridResumeRetriever for production workloads
"""

import os
import pickle
import string
import time
from functools import wraps
import numpy as np
import faiss
from rank_bm25 import BM25Okapi

def retry_on_rate_limit(max_retries=3, base_delay=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    err_str = str(e)
                    if ("429" in err_str or "503" in err_str) and attempt < max_retries - 1:
                        time.sleep(delay)
                        delay *= 2
                    else:
                        raise e
        return wrapper
    return decorator


# ===========================================================================
# NEW: High-Performance Hybrid Retrieval Engine
# ===========================================================================

class HybridResumeRetriever:
    """
    Two-tower hybrid retrieval: Semantic (FAISS) + Keyword (BM25),
    fused via Reciprocal Rank Fusion (RRF).

    Requires pre-built indexes from build_index.py:
      - index_store/faiss.index   (FAISS IndexFlatIP)
      - index_store/bm25.pkl      (BM25Okapi object)
      - index_store/metadata.pkl  (resume JSONs + token docs)
    """

    # RRF damping constant. Higher k reduces the influence of high-ranked
    # documents, making the fusion more balanced. k=60 is the standard
    # value from the original RRF paper (Cormack et al., 2009).
    RRF_K = 60

    def __init__(self, api_key: str, index_dir: str = None):
        """
        Load all pre-built indexes into memory.

        Args:
            api_key: Gemini API key for generating JD embeddings at query time.
            index_dir: Path to the directory containing faiss.index, bm25.pkl,
                       and metadata.pkl. Defaults to <project_root>/index_store.
        """
        from google import genai

        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.embedding_model = "gemini-embedding-001"

        # Resolve index directory
        if index_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            index_dir = os.path.join(project_root, "index_store")

        self.index_dir = index_dir

        # ------------------------------------------------------------------
        # Load FAISS index (vector store for semantic search)
        # ------------------------------------------------------------------
        faiss_path = os.path.join(index_dir, "faiss.index")
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(
                f"FAISS index not found at {faiss_path}. "
                f"Run build_index.py first to create it."
            )
        self.faiss_index = faiss.read_index(faiss_path)

        # ------------------------------------------------------------------
        # Load BM25 index (keyword search engine)
        # ------------------------------------------------------------------
        bm25_path = os.path.join(index_dir, "bm25.pkl")
        if not os.path.exists(bm25_path):
            raise FileNotFoundError(
                f"BM25 index not found at {bm25_path}. "
                f"Run build_index.py first to create it."
            )
        with open(bm25_path, "rb") as f:
            self.bm25: BM25Okapi = pickle.load(f)

        # ------------------------------------------------------------------
        # Load metadata (candidate_id, resume_json, token_doc per resume)
        # ------------------------------------------------------------------
        metadata_path = os.path.join(index_dir, "metadata.pkl")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata not found at {metadata_path}. "
                f"Run build_index.py first to create it."
            )
        with open(metadata_path, "rb") as f:
            self.metadata: list = pickle.load(f)

        # Sanity check: index sizes must match
        assert self.faiss_index.ntotal == len(self.metadata), (
            f"FAISS index has {self.faiss_index.ntotal} vectors but metadata "
            f"has {len(self.metadata)} entries. Rebuild indexes with build_index.py."
        )

        print(f"[HybridResumeRetriever] Loaded {len(self.metadata)} resumes from {index_dir}")

    @retry_on_rate_limit(max_retries=4, base_delay=2)
    def generate_embedding(self, text: str) -> list:
        """Generate embedding vector for a text string using gemini-embedding-001."""
        result = self.client.models.embed_content(
            model=self.embedding_model,
            contents=[text],
        )
        return result.embeddings[0].values

    def prepare_text_for_embedding(self, parsed_json: dict) -> str:
        """
        Convert a parsed resume/JD JSON into a flat text string for embedding.
        Handles both resume and JD schemas gracefully.

        This method is identical to ResumeJDMatcher.prepare_text_for_embedding()
        to ensure consistency between indexing and querying.
        """
        parts = []

        # Skills (resume) or required_skills (JD)
        skills = (
            parsed_json.get("skills")
            or parsed_json.get("required_skills")
            or []
        )
        if isinstance(skills, list):
            parts.append(" ".join(skills))

        # Nice-to-have skills (JD only)
        nice_to_have = parsed_json.get("nice_to_have_skills", [])
        if isinstance(nice_to_have, list):
            parts.append(" ".join(nice_to_have))

        # Education
        education = parsed_json.get("education")
        if isinstance(education, list):
            for edu in education:
                parts.append(" ".join(str(v) for v in edu.values() if v))
        elif isinstance(education, str):
            parts.append(education)

        # Work experience
        experience = (
            parsed_json.get("experience")
            or parsed_json.get("work_experience")
        )
        if isinstance(experience, list):
            for exp in experience:
                parts.append(" ".join(str(v) for v in exp.values() if v))

        # Projects
        projects = parsed_json.get("projects")
        if isinstance(projects, list):
            for proj in projects:
                parts.append(" ".join(str(v) for v in proj.values() if v))

        # Responsibilities (JD only)
        responsibilities = parsed_json.get("responsibilities")
        if isinstance(responsibilities, list):
            parts.append(" ".join(responsibilities))

        # Job title (JD only)
        job_title = parsed_json.get("job_title")
        if job_title:
            parts.append(job_title)

        # Location
        location = parsed_json.get("location")
        if location:
            parts.append(location)

        return " ".join(parts)

    def _tokenize(self, text: str) -> list:
        """
        Tokenize text for BM25 querying: lowercase, strip punctuation, split.
        MUST be identical to tokenize_text() in build_index.py.
        """
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text.split()

    def hybrid_search(self, jd_json: dict, top_k: int = 30) -> list:
        """
        Perform hybrid retrieval: FAISS semantic search + BM25 keyword search,
        fused via Reciprocal Rank Fusion (RRF).

        Args:
            jd_json: Parsed Job Description JSON (from jd_api_integration.py).
            top_k: Number of top results to return after RRF fusion.

        Returns:
            List of (rrf_score, resume_json) tuples, sorted descending by RRF score.
            The resume_json is the full parsed JSON from the index.
        """
        n_resumes = len(self.metadata)
        if n_resumes == 0:
            return []

        # ==================================================================
        # TOWER 1: Semantic Search via FAISS
        # ==================================================================
        # 1a. Prepare JD text and generate a single embedding vector
        jd_text = self.prepare_text_for_embedding(jd_json)
        jd_embedding = np.array(self.generate_embedding(jd_text), dtype=np.float32).reshape(1, -1)

        # 1b. L2-normalize the query vector (index vectors are already normalized)
        #     so that inner product = cosine similarity
        faiss.normalize_L2(jd_embedding)

        # 1c. Search FAISS for ALL resumes, ranked by cosine similarity
        #     distances: shape (1, n_resumes) — similarity scores (higher = better)
        #     indices:   shape (1, n_resumes) — resume indexes in the FAISS index
        sem_distances, sem_indices = self.faiss_index.search(jd_embedding, n_resumes)

        # 1d. Build semantic rank mapping: resume_index → rank (1-indexed)
        #     FAISS returns results sorted by similarity (highest first),
        #     so position 0 = rank 1 (best), position 1 = rank 2, etc.
        semantic_rank = {}
        for rank_0indexed, resume_idx in enumerate(sem_indices[0]):
            semantic_rank[int(resume_idx)] = rank_0indexed + 1  # 1-indexed

        # ==================================================================
        # TOWER 2: Keyword Search via BM25
        # ==================================================================
        # 2a. Tokenize JD text using the same tokenizer as build_index.py
        jd_tokens = self._tokenize(jd_text)

        # 2b. Get BM25 scores for all documents
        bm25_scores = self.bm25.get_scores(jd_tokens)

        # 2c. Build BM25 rank mapping: resume_index → rank (1-indexed)
        #     Sort by BM25 score descending; position 0 = rank 1 (best)
        bm25_sorted_indices = np.argsort(bm25_scores)[::-1]
        bm25_rank = {}
        for rank_0indexed, resume_idx in enumerate(bm25_sorted_indices):
            bm25_rank[int(resume_idx)] = rank_0indexed + 1  # 1-indexed

        # ==================================================================
        # FUSION: Reciprocal Rank Fusion (RRF)
        # ==================================================================
        # Formula: RRF_Score(doc) = 1/(k + sem_rank) + 1/(k + bm25_rank)
        #
        # Where:
        #   - k = 60 (damping constant from the original RRF paper)
        #   - sem_rank  = 1-indexed rank from FAISS cosine similarity search
        #   - bm25_rank = 1-indexed rank from BM25 keyword search
        #
        # Intuition: A document that ranks highly in BOTH towers gets a
        # higher combined score. The 1/(k+rank) formula ensures that:
        #   1. Top-ranked documents contribute more than low-ranked ones
        #   2. The damping constant k prevents any single high rank from
        #      dominating (e.g., rank 1 in one tower can't overpower
        #      rank 100 in the other)
        #
        # Example:
        #   Doc A: sem_rank=3, bm25_rank=1
        #     RRF = 1/(60+3) + 1/(60+1) = 0.01587 + 0.01639 = 0.03226
        #   Doc B: sem_rank=1, bm25_rank=15
        #     RRF = 1/(60+1) + 1/(60+15) = 0.01639 + 0.01333 = 0.02972
        #   → Doc A wins because it's strong in BOTH towers.
        # ==================================================================
        k = self.RRF_K
        rrf_scores = []

        for idx in range(n_resumes):
            s_rank = semantic_rank.get(idx, n_resumes)  # Fallback to worst rank
            b_rank = bm25_rank.get(idx, n_resumes)

            rrf_score = (1.0 / (k + s_rank)) + (1.0 / (k + b_rank))

            rrf_scores.append((rrf_score, idx))

        # Sort by RRF score descending (highest combined score first)
        rrf_scores.sort(reverse=True, key=lambda x: x[0])

        # Return top_k results as (rrf_score, resume_json) tuples
        top_results = []
        for rrf_score, idx in rrf_scores[:top_k]:
            resume_json = self.metadata[idx]["resume_json"]
            top_results.append((rrf_score, resume_json))

        return top_results


# ===========================================================================
# LEGACY: Original on-the-fly semantic matcher (DEPRECATED)
# ===========================================================================
# Preserved for backward compatibility with existing test scripts.
# For production, use HybridResumeRetriever instead.

class ResumeJDMatcher:
    """
    [DEPRECATED] Original semantic matcher that generates embeddings on-the-fly.
    Use HybridResumeRetriever for production workloads.
    """
    def __init__(self, api_key: str):
        from google import genai
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.embedding_model = "gemini-embedding-001"

    @retry_on_rate_limit(max_retries=4, base_delay=2)
    def generate_embedding(self, text: str) -> list:
        result = self.client.models.embed_content(
            model=self.embedding_model,
            contents=[text],
        )
        return result.embeddings[0].values

    def prepare_text_for_embedding(self, parsed_json: dict) -> str:
        parts = []

        skills = (
            parsed_json.get("skills")
            or parsed_json.get("required_skills")
            or []
        )
        if isinstance(skills, list):
            parts.append(" ".join(skills))

        # Nice-to-have skills (JD)
        nice_to_have = parsed_json.get("nice_to_have_skills", [])
        if isinstance(nice_to_have, list):
            parts.append(" ".join(nice_to_have))

        # Education
        education = parsed_json.get("education")
        if isinstance(education, list):
            for edu in education:
                parts.append(" ".join(str(v) for v in edu.values() if v))
        elif isinstance(education, str):
            parts.append(education)

        # Work experience
        experience = (
            parsed_json.get("experience")
            or parsed_json.get("work_experience")
        )
        if isinstance(experience, list):
            for exp in experience:
                parts.append(" ".join(str(v) for v in exp.values() if v))

        # Projects
        projects = parsed_json.get("projects")
        if isinstance(projects, list):
            for proj in projects:
                parts.append(" ".join(str(v) for v in proj.values() if v))

        # Responsibilities (JD)
        responsibilities = parsed_json.get("responsibilities")
        if isinstance(responsibilities, list):
            parts.append(" ".join(responsibilities))

        # Job title
        job_title = parsed_json.get("job_title")
        if job_title:
            parts.append(job_title)

        # Location
        location = parsed_json.get("location")
        if location:
            parts.append(location)

        return " ".join(parts)

    def semantic_match(self, jd_json: dict, resumes_json: list, top_k: int = 1):
        jd_text = self.prepare_text_for_embedding(jd_json)
        jd_embedding = self.generate_embedding(jd_text)

        scored_resumes = []
        for resume_json in resumes_json:
            resume_text = self.prepare_text_for_embedding(resume_json)
            resume_embedding = self.generate_embedding(resume_text)
            score = cosine_similarity(jd_embedding, resume_embedding)
            scored_resumes.append((score, resume_json))

        # Sort by score (descending) and take top_k
        scored_resumes.sort(reverse=True, key=lambda x: x[0])
        top_resumes = scored_resumes[:top_k]
        return jd_json, top_resumes


def cosine_similarity(vec1, vec2):
    """Cosine similarity between two vectors. Kept for reference/testing."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
