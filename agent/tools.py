import os
import json
from typing import List, Dict, Any
from resume_model.jd_api_integration import call_jd_gemini_api
from resume_model.text_extract import extract_text_and_links
from resume_model.resume_api_integration import call_gemini_api
from resume_model.linkedin_parser import parse_linkedin_json
from resume_model.embedding_matching import HybridResumeRetriever
from resume_model.llm_fit_scorer import call_llm_fit_scorer

def parse_jd(jd_text: str, api_key: str) -> Dict[str, Any]:
    res_str = call_jd_gemini_api(jd_text, api_key)
    return json.loads(res_str)

def parse_resume_file(file_path: str, api_key: str) -> Dict[str, Any]:
    if file_path.lower().endswith(".json"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return parse_linkedin_json(content, os.path.basename(file_path))
    else:
        text, _ = extract_text_and_links(file_path)
        res_str = call_gemini_api(text, os.path.basename(file_path), api_key)
        return json.loads(res_str)

def search_talent_pool(jd_json: Dict[str, Any], api_key: str, index_dir: str, top_k: int = 10) -> List[Dict]:
    retriever = HybridResumeRetriever(api_key=api_key, index_dir=index_dir)
    results = retriever.hybrid_search(jd_json, top_k=top_k)
    return [res_json for _, res_json in results]

def score_candidates(resume_jsons: List[Dict], jd_json: Dict[str, Any], api_key: str) -> List[Dict]:
    if not resume_jsons:
        return []
    return call_llm_fit_scorer(resume_jsons, jd_json, api_key)
