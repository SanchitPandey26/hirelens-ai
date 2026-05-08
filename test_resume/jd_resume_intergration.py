"""
jd_resume_integration.py — Test script for hybrid JD-Resume matching.
Uses HybridResumeRetriever (FAISS + BM25 + RRF) from pre-built indexes.
"""
import os
import json
from dotenv import load_dotenv
from resume_model.jd_api_integration import call_jd_gemini_api
from resume_model.embedding_matching import HybridResumeRetriever

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


def process_jd(jd_file: str, api_key: str) -> dict:
    """Parse a JD text file into structured JSON via Gemini."""
    with open(jd_file, "r", encoding="utf-8") as f:
        jd_text = f.read()
    jd_json_str = call_jd_gemini_api(jd_text, api_key)
    return json.loads(jd_json_str)


def main(jd_file: str, top_k: int = 30):
    """
    Run hybrid retrieval against the pre-built index.
    No resume processing needed at runtime — everything is pre-indexed.
    """
    # Parse JD
    jd_json = process_jd(jd_file, api_key)
    print("Parsed JD JSON:")
    print(json.dumps(jd_json, indent=2, ensure_ascii=False))

    # Hybrid search: FAISS + BM25 + RRF
    retriever = HybridResumeRetriever(api_key)
    top_results = retriever.hybrid_search(jd_json, top_k=top_k)

    print(f"\nTop {len(top_results)} matched resumes (by RRF score):")
    for i, (rrf_score, resume_json) in enumerate(top_results):
        candidate_id = resume_json.get("candidate_id", "unknown")
        print(f"\n--- Rank {i+1} | RRF Score: {rrf_score:.5f} | Candidate: {candidate_id} ---")
        print(json.dumps(resume_json, indent=2, ensure_ascii=False))

    return jd_json, [r[1] for r in top_results]


if __name__ == "__main__":
    jd_file = "resume_data/JD.txt"
    main(jd_file, top_k=5)
