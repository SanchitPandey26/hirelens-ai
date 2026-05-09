import os
from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from agent.tools import parse_jd, parse_resume_file, search_talent_pool, score_candidates
from langgraph.checkpoint.memory import MemorySaver

def parse_jd_node(state: AgentState):
    api_key = os.getenv("GEMINI_API_KEY")
    try:
        jd_json = parse_jd(state.get("jd_text", ""), api_key)
        return {"jd_json": jd_json}
    except Exception as e:
        return {"error": f"JD Parse Error: {e}"}

def ingest_resumes_node(state: AgentState):
    api_key = os.getenv("GEMINI_API_KEY")
    paths = state.get("uploaded_file_paths", [])
    resumes = []
    for p in paths:
        try:
            resumes.append(parse_resume_file(p, api_key))
        except Exception as e:
            print(f"Failed to parse {p}: {e}")
    return {"uploaded_resumes": resumes, "all_candidates": resumes}

def search_talent_pool_node(state: AgentState):
    api_key = os.getenv("GEMINI_API_KEY")
    index_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "index_store")
    
    jd_json = state.get("jd_json", {})
    if not jd_json:
        return {"error": "Missing JD JSON for talent search"}
        
    try:
        talent_resumes = search_talent_pool(jd_json, api_key, index_dir, top_k=5)
        existing_ids = {c.get("candidate_id") for c in state.get("all_candidates", [])}
        new_candidates = [r for r in talent_resumes if r.get("candidate_id") not in existing_ids]
        
        all_cands = state.get("all_candidates", []) + new_candidates
        return {
            "talent_pool_resumes": talent_resumes,
            "all_candidates": all_cands,
            "search_triggered": True
        }
    except Exception as e:
        print(f"Talent pool search failed: {e}")
        return {"search_triggered": True}

def score_candidates_node(state: AgentState):
    api_key = os.getenv("GEMINI_API_KEY")
    jd_json = state.get("jd_json", {})
    
    all_candidates = state.get("all_candidates", [])
    scored_candidates = state.get("scored_candidates", [])
    
    # If scored_candidates is None due to uninitialized state
    if scored_candidates is None:
        scored_candidates = []
        
    scored_ids = {c.get("candidate_id") for c in scored_candidates}
    
    unscored = [c for c in all_candidates if c.get("candidate_id") not in scored_ids]
    
    if not unscored:
        return {}
        
    try:
        new_scored = score_candidates(unscored, jd_json, api_key)
        merged_scored = scored_candidates + new_scored
        
        merged_scored.sort(key=lambda x: x.get("weighted_total", 0), reverse=True)
        has_strong_fit = any(c.get("weighted_total", 0) >= 7.0 for c in merged_scored)
        
        return {
            "scored_candidates": merged_scored,
            "has_strong_fit": has_strong_fit
        }
    except Exception as e:
        return {"error": f"Scoring failed: {e}"}

def check_strong_fit(state: AgentState):
    if state.get("has_strong_fit", False):
        return "human_review"
    if not state.get("search_triggered", False):
        return "search_talent_pool"
    return "human_review"

def human_review_node(state: AgentState):
    pass

def apply_overrides_node(state: AgentState):
    return state

def create_agent_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("parse_jd", parse_jd_node)
    workflow.add_node("ingest_resumes", ingest_resumes_node)
    workflow.add_node("score_candidates", score_candidates_node)
    workflow.add_node("search_talent_pool", search_talent_pool_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("apply_overrides", apply_overrides_node)
    
    workflow.add_edge(START, "parse_jd")
    workflow.add_edge("parse_jd", "ingest_resumes")
    workflow.add_edge("ingest_resumes", "score_candidates")
    
    workflow.add_conditional_edges("score_candidates", check_strong_fit, {
        "human_review": "human_review",
        "search_talent_pool": "search_talent_pool"
    })
    
    workflow.add_edge("search_talent_pool", "score_candidates")
    workflow.add_edge("human_review", "apply_overrides")
    workflow.add_edge("apply_overrides", END)
    
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer, interrupt_before=["human_review"])
