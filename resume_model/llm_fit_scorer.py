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

    prompt_text = f"""You are given a job description and multiple anonymized candidate resumes.
Evaluate each candidate against the job description and return a comparative ranked analysis for ONLY the top 10 candidates. Ignore the rest.

For each of the top 10 candidates return:
- candidate_id: from the resume
- score: integer from 0 to 100 based on overall fit
- key_strengths: exactly 2 to 3 short points on what makes this candidate strong for this role
- key_gaps: exactly 2 to 3 short points on what this candidate is missing for this role
- recommendation: exactly one of "Strong Fit", "Moderate Fit", or "Not Fit"
- summary: 2 sentence explanation justifying the score

Scoring guide — apply strictly and consistently:
- 80 to 100 → Strong Fit: meets most required skills, relevant experience, strong alignment
- 50 to 79 → Moderate Fit: meets some required skills, partial alignment, has gaps but shows potential
- 0 to 49 → Not Fit: missing most required skills, significant gaps, poor alignment

Comparative rules — these are critical:
- Score candidates relative to each other, not in isolation
- No two candidates should have the same score
- A candidate with more relevant experience should always score higher than one with less, even if both meet the skill requirements
- Recommendation must always be consistent with score — 80+ is always Strong Fit, 50-79 is always Moderate Fit, below 50 is always Not Fit
- Sort the final output by score, highest first

Job Description:
{json.dumps(jd_json, ensure_ascii=False)}

Candidate Resumes:
{json.dumps(resume_jsons, ensure_ascii=False)}"""

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
                required=["candidate_id", "score", "key_strengths", "key_gaps", "recommendation", "summary"],
                properties={
                    "candidate_id": genai.types.Schema(
                        type=genai.types.Type.STRING,
                    ),
                    "score": genai.types.Schema(
                        type=genai.types.Type.INTEGER,
                    ),
                    "key_strengths": genai.types.Schema(
                        type=genai.types.Type.ARRAY,
                        items=genai.types.Schema(
                            type=genai.types.Type.STRING,
                        ),
                    ),
                    "key_gaps": genai.types.Schema(
                        type=genai.types.Type.ARRAY,
                        items=genai.types.Schema(
                            type=genai.types.Type.STRING,
                        ),
                    ),
                    "recommendation": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        enum=["Strong Fit", "Moderate Fit", "Not Fit"],
                    ),
                    "summary": genai.types.Schema(
                        type=genai.types.Type.STRING,
                    ),
                },
            ),
        ),
        system_instruction=[
            types.Part.from_text(text="""You are an expert HR recruiter and talent evaluator.
Your job is to comparatively evaluate multiple anonymized candidate resumes against a job description and rank them.
You must evaluate all candidates relative to each other — not in isolation. This means if multiple candidates have the same skill, differentiate them based on depth, relevance, and context of that skill.
Always return output as a valid JSON array of ONLY the top 10 candidates, one object per candidate, sorted by score in descending order.
Be objective, concise, and consistent across all candidates.
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