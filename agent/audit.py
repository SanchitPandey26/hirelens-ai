import os
import json
from datetime import datetime

def get_audit_file(thread_id: str) -> str:
    os.makedirs("audit_logs", exist_ok=True)
    return os.path.join("audit_logs", f"audit_{thread_id}.json")

def load_audit_log(thread_id: str) -> list:
    filepath = get_audit_file(thread_id)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("audit_log", [])
    return []

def log_override(thread_id: str, candidate_id: str, dimension: str, original_score: int, original_justification: str, new_score: int, reason: str, overridden_by: str = "HR_user"):
    filepath = get_audit_file(thread_id)
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "candidate_id": candidate_id,
        "action": "score_override",
        "dimension": dimension,
        "original_score": original_score,
        "original_justification": original_justification,
        "new_score": new_score,
        "reason": reason,
        "overridden_by": overridden_by
    }
    
    current_log = load_audit_log(thread_id)
    current_log.append(log_entry)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({"audit_log": current_log}, f, indent=2)

def log_flag(thread_id: str, candidate_id: str, reason: str, flagged_by: str = "HR_user"):
    filepath = get_audit_file(thread_id)
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "candidate_id": candidate_id,
        "action": "flag_candidate",
        "reason": reason,
        "flagged_by": flagged_by
    }
    
    current_log = load_audit_log(thread_id)
    current_log.append(log_entry)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({"audit_log": current_log}, f, indent=2)
