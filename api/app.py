"""
app.py — FastAPI Resume-JD Fit Scoring API
============================================
Uses the HybridResumeRetriever (FAISS + BM25 + RRF) for blazing-fast
resume retrieval. Resumes are pre-indexed offline via build_index.py.

Endpoints:
  POST /evaluate_resumes/  — Upload a JD, get top-K scored candidates
  GET  /index_status        — Check if indexes are loaded, resume count
  GET  /                    — Health check
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional
import os
import json
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Import pipeline modules
# ---------------------------------------------------------------------------
from resume_model.jd_api_integration import call_jd_gemini_api
from resume_model.embedding_matching import HybridResumeRetriever
from resume_model.llm_fit_scorer import call_llm_fit_scorer
from resume_model.text_extract import extract_text_and_links

# ---------------------------------------------------------------------------
# Load environment and configuration
# ---------------------------------------------------------------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Index directory (created by build_index.py)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# app.py lives in api/, so project root is one level up
INDEX_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "index_store")

# ---------------------------------------------------------------------------
# Initialize FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Resume-JD Hybrid Fit Scoring API",
    description="Hybrid search (Semantic + Keyword) with RRF fusion for resume-job matching.",
    version="2.0.0",
)

# ---------------------------------------------------------------------------
# Load HybridResumeRetriever at startup (once, not per-request)
# ---------------------------------------------------------------------------
retriever: Optional[HybridResumeRetriever] = None


@app.on_event("startup")
async def load_retriever():
    """
    Load pre-built FAISS + BM25 indexes into memory on server startup.
    This makes every subsequent query near-instant (no disk I/O per request).
    """
    global retriever
    try:
        retriever = HybridResumeRetriever(api_key=api_key, index_dir=INDEX_DIR)
        print(f"[STARTUP] HybridResumeRetriever loaded from {INDEX_DIR}")
    except FileNotFoundError as e:
        print(f"[WARNING] Could not load indexes: {e}")
        print(f"[WARNING] Run build_index.py first, then restart the server.")
        retriever = None


# ===========================================================================
# Endpoints
# ===========================================================================

@app.post("/evaluate_resumes/")
async def evaluate_resumes(
    jd_file: UploadFile = File(...),
    top_k: int = Form(30),
):
    """
    Upload a Job Description file and get ranked resume matches.

    The endpoint:
    1. Parses the JD via Gemini (jd_api_integration.py — unchanged)
    2. Runs hybrid retrieval: FAISS semantic + BM25 keyword, fused with RRF
    3. Passes top-K results to LLM fit scorer (llm_fit_scorer.py — unchanged)

    Accepts .txt and .pdf JD files.

    Args:
        jd_file: Job description file (.txt or .pdf)
        top_k: Number of top candidates to retrieve and score (default: 30)
    """
    # ------------------------------------------------------------------
    # Guard: ensure indexes are loaded
    # ------------------------------------------------------------------
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Indexes not loaded. Run build_index.py first, then restart the server."
        )

    # ------------------------------------------------------------------
    # Step 1: Extract and parse JD text
    # ------------------------------------------------------------------
    filename = jd_file.filename.lower()
    content = await jd_file.read()

    if filename.endswith(".txt"):
        jd_text = content.decode("utf-8")
    elif filename.endswith(".pdf"):
        # Save to temp, extract text, clean up
        temp_path = os.path.join(PROJECT_ROOT, f"_temp_jd_{jd_file.filename}")
        try:
            with open(temp_path, "wb") as f:
                f.write(content)
            jd_text, _ = extract_text_and_links(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported JD file format. Use .txt or .pdf"
        )

    # Parse JD into structured JSON via Gemini (prompt logic untouched)
    jd_json_str = call_jd_gemini_api(jd_text, api_key)
    jd_json = json.loads(jd_json_str)

    # ------------------------------------------------------------------
    # Step 2: Hybrid retrieval — FAISS + BM25 + RRF fusion
    #   - Generates ONE embedding for the JD (single API call)
    #   - Queries pre-built FAISS index (milliseconds)
    #   - Queries pre-built BM25 index (milliseconds)
    #   - Fuses ranks via RRF formula
    # ------------------------------------------------------------------
    top_results = retriever.hybrid_search(jd_json, top_k=top_k)

    # Extract just the resume JSONs for the LLM scorer
    top_resume_jsons = [resume_json for _, resume_json in top_results]

    # ------------------------------------------------------------------
    # Step 3: LLM fit scoring (prompt logic untouched)
    #   Passes top-K resume JSONs + JD JSON to Gemini for comparative ranking
    # ------------------------------------------------------------------
    llm_result = call_llm_fit_scorer(top_resume_jsons, jd_json, api_key)

    # ------------------------------------------------------------------
    # Step 4: Return results
    # ------------------------------------------------------------------
    if isinstance(llm_result, list):
        return llm_result
    else:
        return {"raw_result": llm_result}


@app.get("/index_status")
def index_status():
    """
    Check index health: is the retriever loaded, how many resumes are indexed.
    Useful for admin dashboards and monitoring.
    """
    if retriever is None:
        return {
            "status": "not_loaded",
            "message": "Indexes not loaded. Run build_index.py first.",
            "resume_count": 0,
            "index_dir": INDEX_DIR,
        }

    return {
        "status": "loaded",
        "resume_count": len(retriever.metadata),
        "faiss_vectors": retriever.faiss_index.ntotal,
        "index_dir": retriever.index_dir,
    }


@app.get("/")
def read_root():
    return {"message": "Resume-JD Hybrid Fit Scoring API is running (v2.0 — FAISS + BM25 + RRF)."}
