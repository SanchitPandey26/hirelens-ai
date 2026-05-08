import os
from dotenv import load_dotenv
from Resume_Screening.resume_model.jd_api_integration import call_jd_gemini_api

# Load API key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Read JD text from file
jd_file_path = "../resume_data/JD.txt"
try:
    with open(jd_file_path, "r", encoding="utf-8") as f:
        jd_text = f.read()
except FileNotFoundError:
    print(f"File {jd_file_path} not found.")
    jd_text = None

if jd_text:
    json_output = call_jd_gemini_api(jd_text, api_key)
    print(json_output)
else:
    print("No job description text to parse.")
