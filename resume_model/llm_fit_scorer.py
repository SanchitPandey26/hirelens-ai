import os
import json
import time
from functools import wraps
from google import genai
from google.genai import types

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

@retry_on_rate_limit(max_retries=4, base_delay=2)
def call_llm_fit_scorer(resume_jsons: list, jd_json: dict, api_key: str) -> list:
    client = genai.Client(api_key=api_key)
    model = "gemini-3.1-flash-lite-preview"

    prompt_text = f"""You are an expert HR recruiter and talent evaluator.
You are given a job description and multiple anonymized candidate resumes.
Evaluate each candidate against the job description strictly using the 5-dimension rubric below.
Return a comparative ranked analysis for ONLY the top 10 candidates. Ignore the rest.

For each of the top 10 candidates, provide exactly the following in the JSON schema:
- candidate_id: from the resume
- dimensions: an object containing 5 specific dimensions:
    - skills_match (Weight: 0.30): Score 0-10.
    - experience_relevance (Weight: 0.25): Score 0-10.
    - education_certs (Weight: 0.15): Score 0-10.
    - project_portfolio (Weight: 0.20): Score 0-10.
    - communication_quality (Weight: 0.10): Score 0-10.
  For each dimension, also provide a "justification" (exactly 1 sentence).
- weighted_total: Calculate the exact weighted total (out of 10.0) based on the scores and weights above.
- recommendation: Map the weighted_total to exactly one of: "Hire" (>= 7.0), "Maybe" (5.0 - 6.9), or "No Hire" (< 5.0).
- summary: exactly 2 sentences explaining the overall evaluation.

Scoring Rubric per Dimension:
| Dimension | 0-4 (Poor) | 5-7 (Average) | 8-10 (Excellent) |
|---|---|---|---|
| Skills Match | <50% skills match | 50-70% skills match | >85% skills match |
| Experience Relevance | Unrelated domain | Adjacent domain | Exact domain & seniority |
| Education & Certs | Below minimum | Meets minimum | Exceeds + extra certs |
| Project / Portfolio | No evidence | 1-2 generic projects | Strong relevant portfolio |
| Communication Quality | Poor structure/grammar | Adequate clarity | Crisp, structured, impactful |

Rules:
- Score candidates relative to each other.
- Calculate the weighted_total precisely.
- Sort the final output by weighted_total, highest first.

Job Description:
{json.dumps(jd_json, ensure_ascii=False)}

Candidate Resumes:
{json.dumps(resume_jsons, ensure_ascii=False)}"""

    dimension_schema = genai.types.Schema(
        type=genai.types.Type.OBJECT,
        required=["score", "weight", "justification"],
        properties={
            "score": genai.types.Schema(type=genai.types.Type.INTEGER),
            "weight": genai.types.Schema(type=genai.types.Type.NUMBER),
            "justification": genai.types.Schema(type=genai.types.Type.STRING),
        }
    )

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt_text),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=0.1,
        thinking_config=types.ThinkingConfig(
            thinking_level="MINIMAL",
        ),
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.ARRAY,
            items=genai.types.Schema(
                type=genai.types.Type.OBJECT,
                required=["candidate_id", "dimensions", "weighted_total", "recommendation", "summary"],
                properties={
                    "candidate_id": genai.types.Schema(type=genai.types.Type.STRING),
                    "dimensions": genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        required=["skills_match", "experience_relevance", "education_certs", "project_portfolio", "communication_quality"],
                        properties={
                            "skills_match": dimension_schema,
                            "experience_relevance": dimension_schema,
                            "education_certs": dimension_schema,
                            "project_portfolio": dimension_schema,
                            "communication_quality": dimension_schema,
                        }
                    ),
                    "weighted_total": genai.types.Schema(type=genai.types.Type.NUMBER),
                    "recommendation": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        enum=["Hire", "Maybe", "No Hire"],
                    ),
                    "summary": genai.types.Schema(type=genai.types.Type.STRING),
                },
            ),
        ),
        system_instruction=[
            types.Part.from_text(text="""You are an expert HR recruiter and talent evaluator.
Your job is to comparatively evaluate multiple anonymized candidate resumes against a job description using a strict 5-dimension rubric.
Always return output as a valid JSON array of ONLY the top 10 candidates, one object per candidate, sorted by weighted_total in descending order.
Do not include any commentary, explanation, or markdown — only the raw JSON."""),
        ],
    )

    response_text = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        response_text += chunk.text

    try:
        return json.loads(response_text)
    except Exception:
        return []