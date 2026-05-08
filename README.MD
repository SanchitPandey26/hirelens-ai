# Resume–JD Matching Pipeline

A modern, modular AI pipeline for parsing, matching, and evaluating resumes against job descriptions using Google Gemini LLM, FAISS, BM25, and Streamlit. The system features an interactive Streamlit frontend and is also wrapped as a FastAPI service for easy integration, batch processing, and cloud deployment. It utilizes a pre-built hybrid vector database for blazing-fast retrieval.

---

## 🚀 Features

- **Hybrid Search Retrieval**: FAISS semantic + BM25 keyword, fused with RRF (Reciprocal Rank Fusion).
- 📄 **Resume & JD Parsing** (PDF/DOCX & unstructured text)
- 🧑‍💼 **LLM Fit Scoring** (fit score, matched/missing skills, summary)
- 🖥️ **Interactive UI** (Streamlit frontend without requiring manual resume uploads per run)
- 📦 **Batch & API Support** (multiple resumes per call)
- ☁️ **Cloud-Ready** (Google Cloud, Azure, AWS, or local)

---

## Tech Stack

- [Streamlit](https://streamlit.io/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [FAISS](https://faiss.ai/)
- [Rank-BM25](https://pypi.org/project/rank-bm25/)
- [Google Gemini API](https://ai.google.dev/)
- [PyMuPDF](https://pymupdf.readthedocs.io/)
- [python-docx](https://python-docx.readthedocs.io/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)

---

## Getting Started

### 1. Clone the Repository
```commandline
Will upload this on github and then update it.
```

### 2. Install Dependencies

```commandline
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```dotenv
GEMINI_API_KEY=your-google-gemini-api-key-here
```

### 4. Add Your Data & Build Index

1. Place all candidate resumes (`.pdf` or `.docx`) in the `resume_data/` folder.
2. Build the FAISS and BM25 indexes offline by running:
```commandline
python build_index.py
```
*This will parse all resumes via Gemini and create the hybrid indexes in the `index_store/` directory.*

---

## Running the Streamlit Frontend

```commandline
streamlit run streamlit_app.py
```

- Upload a Job Description (`.txt` or `.pdf`).
- The app instantly retrieves and ranks top candidate resumes from the pre-built vector database.
- View detailed LLM Fit Scoring directly in the browser.

---

## Running the FastAPI Server

```commandline
uvicorn api.app:app --reload
```

- The API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Interactive API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Index Status Check**: [http://127.0.0.1:8000/index_status](http://127.0.0.1:8000/index_status)

---

## Using the API

### Endpoint

- **POST** `/evaluate_resumes/`

### Parameters

- `jd_file`: Upload the JD as a `.txt` or `.pdf` file.
- `top_k`: Number of top resumes to retrieve and score (default: 30).

### Example Using Swagger UI

1. Go to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
2. Click "Try it out" on `/evaluate_resumes/`
3. Upload your JD file, set `top_k`, and execute.

### Example Using cURL

```commandline
curl -X 'POST' \
  'http://127.0.0.1:8000/evaluate_resumes/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'jd_file=@resumes/JD.txt' \
  -F 'top_k=30'
```

---

## Project Structure

```text
Internship/
├── resume_model/
│   ├── text_extract.py
│   ├── resume_api_integration.py
│   ├── jd_api_integration.py
│   ├── embedding_matching.py
│   └── llm_fit_scorer.py
├── api/
│   └── app.py         # FastAPI app
├── resume_data/       # Folder for bulk candidate resumes
├── index_store/       # FAISS & BM25 pre-built indexes
├── build_index.py     # Script to build vector indices
├── streamlit_app.py   # Streamlit Interactive UI
├── requirements.txt
└── .env
```

---

## Output

- Returns a list of JSON objects, each with:
  - Candidate name, email, phone
  - Fit score (0–10)
  - Matched and missing skills
  - Evaluation summary

---

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## License

*This project is licensed under the [MIT License](./LICENSE) © 2025 Sanchit Pandey.*

You are free to use, modify, and distribute this software with attribution.
See the LICENSE file for details.
---

## Contact

For questions or demo requests, contact [sanchit.pdy@gmail.com].

---

**Happy coding! 🚀**
