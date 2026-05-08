"""
llm_fit_test.py — End-to-end test: Hybrid retrieval → LLM fit scoring.
Uses HybridResumeRetriever (FAISS + BM25 + RRF) + LLM comparative scorer.
"""
import os
import json
from dotenv import load_dotenv
from resume_model.jd_api_integration import call_jd_gemini_api
from resume_model.embedding_matching import HybridResumeRetriever
from resume_model.llm_fit_scorer import call_llm_fit_scorer

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
    Full pipeline test:
    1. Parse JD
    2. Hybrid search (FAISS + BM25 + RRF) from pre-built indexes
    3. LLM fit scoring on top-K candidates
    """
    # Step 1: Parse JD
    jd_json = process_jd(jd_file, api_key)
    print("Parsed JD JSON:")
    print(json.dumps(jd_json, indent=2, ensure_ascii=False))

    # Step 2: Hybrid retrieval — near-instant from pre-built indexes
    retriever = HybridResumeRetriever(api_key)
    top_results = retriever.hybrid_search(jd_json, top_k=top_k)
    top_resume_jsons = [resume_json for _, resume_json in top_results]

    print(f"\n[INFO] Retrieved {len(top_resume_jsons)} candidates via hybrid search")
    for i, (rrf_score, resume_json) in enumerate(top_results):
        candidate_id = resume_json.get("candidate_id", "unknown")
        print(f"  Rank {i+1}: {candidate_id} (RRF: {rrf_score:.5f})")

    # Step 3: LLM fit scoring (unchanged prompt logic)
    print("\n[INFO] Running LLM comparative fit scoring...")
    llm_result = call_llm_fit_scorer(top_resume_jsons, jd_json, api_key)

    # Print results
    if isinstance(llm_result, list):
        print(f"\n--- LLM Fit Scores ({len(llm_result)} candidates) ---\n")
        for item in llm_result:
            print(json.dumps(item, ensure_ascii=False))
    else:
        print(llm_result)


if __name__ == "__main__":
    jd_file = "resume_data/JD.txt"
    main(jd_file, top_k=30)
