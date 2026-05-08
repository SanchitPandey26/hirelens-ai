import os
from google import genai
from google.genai import types


def call_gemini_api(resume_text: str, filename: str, api_key: str) -> dict:
    client = genai.Client(api_key=api_key)
    model = "gemini-3.1-flash-lite-preview"

    prompt_text = f"""Extract the following fields from the resume below and return them as a single JSON object.

Fields to extract:
- candidate_id: use the filename provided
- skills: list of all skills — explicitly listed AND reasonably inferred from projects and experience. Include tools, technologies, frameworks, languages, methodologies.
- education: list of objects with fields — degree, institution, year, grade (if available)
- work_experience: list of objects with fields — role, duration, description
- projects: list of objects with fields — title, description
- achievements: list of awards, honors, competitions only — do NOT include grades here if already in education

Rules:
- DO NOT include name, email, phone, LinkedIn, GitHub, or any contact info
- For skills, infer from project descriptions and work experience even if not explicitly listed
- Do not include soft skills like "communication" or "teamwork" unless explicitly stated
- Do not include generic domain knowledge like "Finance" or "Healthcare" as a skill

Filename: {filename}

Resume Text:
{resume_text}"""

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
            type=genai.types.Type.OBJECT,
            required=["candidate_id", "skills", "education", "work_experience", "projects", "achievements"],
            properties={
                "candidate_id": genai.types.Schema(
                    type=genai.types.Type.STRING,
                ),
                "skills": genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=genai.types.Schema(
                        type=genai.types.Type.STRING,
                    ),
                ),
                "education": genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        properties={
                            "degree": genai.types.Schema(type=genai.types.Type.STRING),
                            "institution": genai.types.Schema(type=genai.types.Type.STRING),
                            "year": genai.types.Schema(type=genai.types.Type.STRING),
                            "grade": genai.types.Schema(type=genai.types.Type.STRING),
                        },
                    ),
                ),
                "work_experience": genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        properties={
                            "role": genai.types.Schema(type=genai.types.Type.STRING),
                            "duration": genai.types.Schema(type=genai.types.Type.STRING),
                            "description": genai.types.Schema(type=genai.types.Type.STRING),
                        },
                    ),
                ),
                "projects": genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        properties={
                            "title": genai.types.Schema(type=genai.types.Type.STRING),
                            "description": genai.types.Schema(type=genai.types.Type.STRING),
                        },
                    ),
                ),
                "achievements": genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=genai.types.Schema(
                        type=genai.types.Type.STRING,
                    ),
                ),
            },
        ),
        system_instruction=[
            types.Part.from_text(text="""You are an expert HR assistant and resume parsing specialist.
Your job is to extract structured, anonymized information from resumes.
You must remove all personally identifiable information (PII) such as name, email, phone number, LinkedIn URL, GitHub URL, or any other contact details.
Always return output as a valid JSON object following the requested schema exactly.
Be exhaustive and accurate when extracting skills — infer skills from projects and experience, not just the skills section.
If a field is missing or not found, return null for strings and empty array [] for lists.
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

    return response_text