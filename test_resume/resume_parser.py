"""
resume_parser.py — Test script for parsing a single resume via Gemini.
"""
import os
from dotenv import load_dotenv
from resume_model.text_extract import extract_text_and_links
from resume_model.resume_api_integration import call_gemini_api

# Load the API key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Path to your resume file
resume_file = "resume_data/1.pdf"  # or .docx

# Extract text and links
resume_text, resume_links = extract_text_and_links(resume_file)

# Call Gemini API — filename becomes candidate_id in the parsed JSON
json_output = call_gemini_api(resume_text, os.path.basename(resume_file), api_key)
print(json_output)
