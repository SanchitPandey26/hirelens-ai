from typing import TypedDict, List, Dict, Optional, Any

class AgentState(TypedDict):
    jd_text: str                      # Raw JD text
    jd_json: Dict[str, Any]           # Parsed JD requirements
    uploaded_file_paths: List[str]    # Paths to user uploads
    uploaded_resumes: List[Dict]      # Resumes from user uploads (PDF/DOCX/LinkedIn JSON)
    talent_pool_resumes: List[Dict]   # Resumes from vector DB search
    all_candidates: List[Dict]        # Merged candidate list
    scored_candidates: List[Dict]     # After rubric scoring
    has_strong_fit: bool              # Controls talent pool trigger
    search_triggered: bool            # Whether talent pool was used
    human_overrides: List[Dict]       # HIL override log
    audit_log: List[Dict]             # Full audit trail
    error: Optional[str]              # Error state
