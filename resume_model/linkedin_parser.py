import json
from typing import Dict, Any

def parse_linkedin_json(json_content: str, filename: str) -> Dict[str, Any]:
    """
    Parses a manually exported LinkedIn JSON file into the standard candidate schema.
    """
    try:
        data = json.loads(json_content)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format for LinkedIn data.")

    # Normalize into standard schema
    skills = data.get("skills", [])
    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(',')]
    
    education = []
    for edu in data.get("education", []):
        if isinstance(edu, dict):
            education.append({
                "degree": edu.get("degree", ""),
                "institution": edu.get("school", ""),
                "year": edu.get("year", ""),
                "grade": edu.get("grade", "")
            })
        elif isinstance(edu, str):
            education.append({"degree": edu, "institution": "", "year": "", "grade": ""})

    work_experience = []
    for pos in data.get("positions", []):
        if isinstance(pos, dict):
            duration = pos.get("duration", "")
            if not duration and "start_date" in pos:
                duration = f"{pos.get('start_date', '')} - {pos.get('end_date', 'Present')}"
                
            work_experience.append({
                "role": pos.get("title", ""),
                "duration": duration,
                "description": pos.get("description", "")
            })

    projects = []
    for proj in data.get("projects", []):
        if isinstance(proj, dict):
            projects.append({
                "title": proj.get("title", ""),
                "description": proj.get("description", "")
            })

    achievements = []
    for cert in data.get("certifications", []):
        if isinstance(cert, dict):
            achievements.append(f"{cert.get('name', '')} ({cert.get('authority', '')})")
        elif isinstance(cert, str):
            achievements.append(cert)

    # The returned schema matches resume_api_integration.py output exactly
    return {
        "candidate_id": filename,
        "skills": skills,
        "education": education,
        "work_experience": work_experience,
        "projects": projects,
        "achievements": achievements
    }
