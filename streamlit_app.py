import streamlit as st
import json
import os
import tempfile
from html import escape
from dotenv import load_dotenv
from resume_model.embedding_matching import HybridResumeRetriever
from resume_model.jd_api_integration import call_jd_gemini_api
from resume_model.llm_fit_scorer import call_llm_fit_scorer

load_dotenv()

@st.cache_resource
def get_retriever(api_key):
    # Retrieve pre-built DB from index_store directory
    index_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index_store")
    return HybridResumeRetriever(api_key=api_key, index_dir=index_dir)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0a0f;
    color: #e8e6f0;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 0%, #1a0f2e 0%, #0a0a0f 50%),
                radial-gradient(ellipse at 80% 100%, #0f1a2e 0%, transparent 50%);
    background-color: #0a0a0f;
}

[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }

h1, h2, h3 {
    font-family: 'Syne', sans-serif;
    letter-spacing: -0.02em;
}

.hero {
    text-align: center;
    padding: 3.5rem 1rem 2.5rem;
}

.hero-badge {
    display: inline-block;
    background: rgba(139, 92, 246, 0.15);
    border: 1px solid rgba(139, 92, 246, 0.3);
    color: #a78bfa;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 100px;
    margin-bottom: 1.5rem;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.5rem, 5vw, 4rem);
    font-weight: 800;
    line-height: 1.05;
    background: linear-gradient(135deg, #ffffff 0%, #a78bfa 50%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 1rem;
}

.hero-sub {
    font-size: 1.05rem;
    color: #9ca3af;
    font-weight: 300;
    max-width: 480px;
    margin: 0 auto !important;
    text-align: center !important;
    line-height: 1.6;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(139,92,246,0.3), transparent);
    margin: 0.5rem 0 2.5rem;
}

.upload-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 0.5rem;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
    transition: border-color 0.2s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(139,92,246,0.4) !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: 1.5px dashed rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(139,92,246,0.5) !important;
    background: rgba(139,92,246,0.04) !important;
}

[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2.5rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 24px rgba(124, 58, 237, 0.3) !important;
}

[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 32px rgba(124, 58, 237, 0.45) !important;
}

[data-testid="stButton"] > button:active {
    transform: translateY(0px) !important;
}

[data-testid="stInfo"] {
    background: rgba(99, 102, 241, 0.1) !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    border-radius: 10px !important;
    color: #a5b4fc !important;
}

[data-testid="stSuccess"] {
    background: rgba(16, 185, 129, 0.1) !important;
    border: 1px solid rgba(16, 185, 129, 0.25) !important;
    border-radius: 10px !important;
    color: #6ee7b7 !important;
}

[data-testid="stError"] {
    background: rgba(239, 68, 68, 0.1) !important;
    border: 1px solid rgba(239, 68, 68, 0.25) !important;
    border-radius: 10px !important;
    color: #fca5a5 !important;
}

[data-testid="stWarning"] {
    background: rgba(245, 158, 11, 0.1) !important;
    border: 1px solid rgba(245, 158, 11, 0.25) !important;
    border-radius: 10px !important;
    color: #fcd34d !important;
}

[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #7c3aed, #60a5fa) !important;
    border-radius: 100px !important;
}

.results-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #f3f4f6;
    margin-bottom: 0.25rem;
}

.results-sub {
    font-size: 0.9rem;
    color: #6b7280;
    margin-bottom: 2rem;
}

.candidate-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s, transform 0.2s;
    position: relative;
    overflow: hidden;
}

.candidate-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 16px 16px 0 0;
}

.candidate-card.strong::before { background: linear-gradient(90deg, #10b981, #34d399); }
.candidate-card.moderate::before { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.candidate-card.not-fit::before { background: linear-gradient(90deg, #ef4444, #f87171); }

.candidate-card:hover {
    border-color: rgba(139,92,246,0.25);
    transform: translateY(-1px);
}

.rank-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2rem;
    height: 2rem;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    margin-right: 0.75rem;
    flex-shrink: 0;
}

.rank-1 { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
.rank-2 { background: rgba(156,163,175,0.15); color: #9ca3af; border: 1px solid rgba(156,163,175,0.3); }
.rank-3 { background: rgba(180,120,60,0.15); color: #b4783c; border: 1px solid rgba(180,120,60,0.3); }
.rank-other { background: rgba(255,255,255,0.05); color: #6b7280; border: 1px solid rgba(255,255,255,0.1); }

.score-block { text-align: center; }

.score-number {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1;
}

.score-strong { color: #10b981; }
.score-moderate { color: #f59e0b; }
.score-notfit { color: #ef4444; }

.score-label {
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b7280;
    margin-top: 0.2rem;
}

.rec-pill {
    display: inline-block;
    padding: 0.25rem 0.85rem;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-top: 0.4rem;
}

.rec-strong { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.25); }
.rec-moderate { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.25); }
.rec-notfit { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.25); }

.candidate-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #f3f4f6;
}

.candidate-file {
    font-size: 0.78rem;
    color: #4b5563;
    margin-top: 0.1rem;
}

.summary-text {
    font-size: 0.875rem;
    color: #9ca3af;
    line-height: 1.6;
    margin-top: 0.75rem;
    padding-top: 0.75rem;
    border-top: 1px solid rgba(255,255,255,0.05);
}

.tag-section-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

.tag-strengths { color: #10b981; }
.tag-gaps { color: #f87171; }

.tag {
    display: block;
    padding: 0.35rem 0.7rem;
    border-radius: 6px;
    font-size: 0.8rem;
    margin: 0.25rem 0;
    line-height: 1.5;
}

.tag-s { background: rgba(16,185,129,0.08); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.15); }
.tag-g { background: rgba(239,68,68,0.08); color: #fca5a5; border: 1px solid rgba(239,68,68,0.15); }

.stats-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 2rem;
}

.stat-chip {
    flex: 1;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    text-align: center;
}

.stat-chip-number {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
}

.stat-chip-label {
    font-size: 0.7rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.1rem;
}

.strong-num { color: #10b981; }
.moderate-num { color: #f59e0b; }
.notfit-num { color: #ef4444; }

[data-testid="stExpander"] {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}

[data-testid="stExpander"] summary {
    color: #9ca3af !important;
    font-size: 0.85rem !important;
}

.jd-row {
    display: flex;
    gap: 0.6rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    align-items: flex-start;
}

.jd-row:last-child { border-bottom: none; }

.jd-key {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b7280;
    min-width: 130px;
    padding-top: 0.15rem;
    flex-shrink: 0;
}

.jd-val {
    font-size: 0.85rem;
    color: #d1d5db;
    line-height: 1.5;
}

.jd-skill-tag {
    display: inline-block;
    background: rgba(139,92,246,0.1);
    border: 1px solid rgba(139,92,246,0.2);
    color: #a78bfa;
    font-size: 0.75rem;
    padding: 0.15rem 0.6rem;
    border-radius: 5px;
    margin: 0.15rem 0.15rem 0.15rem 0;
}

.jd-nice-tag {
    display: inline-block;
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.15);
    color: #818cf8;
    font-size: 0.75rem;
    padding: 0.15rem 0.6rem;
    border-radius: 5px;
    margin: 0.15rem 0.15rem 0.15rem 0;
}

.jd-resp-item {
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
    margin-bottom: 0.3rem;
    font-size: 0.85rem;
    color: #d1d5db;
}

.jd-resp-dot {
    color: #7c3aed;
    margin-top: 0.1rem;
    flex-shrink: 0;
}
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────

def get_score_class(score):
    if score >= 80: return "strong", "score-strong", "rec-strong", "strong-num"
    if score >= 50: return "moderate", "score-moderate", "rec-moderate", "moderate-num"
    return "not-fit", "score-notfit", "rec-notfit", "notfit-num"

def get_rank_class(rank):
    if rank == 1: return "rank-1"
    if rank == 2: return "rank-2"
    if rank == 3: return "rank-3"
    return "rank-other"

def clean_candidate_id(candidate_id):
    return os.path.splitext(candidate_id)[0].replace("-", " ").replace("_", " ").title()

# FIX: defensively parse list fields that LLM may return as JSON strings
def parse_list_field(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [value]
    return []


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ AI Powered</div>
    <h1 class="hero-title">Resume Screening<br>Reimagined</h1>
    <p class="hero-sub">Upload a job description to automatically search and rank top matches from our vector database.</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)


# ── Input section ─────────────────────────────────────────────────────────────
col_jd, col_gap, col_sliders = st.columns([1, 0.08, 1])

with col_jd:
    st.markdown('<div class="upload-label">Job Description</div>', unsafe_allow_html=True)
    jd_file = st.file_uploader(
        "Upload JD",
        type=["txt"],
        key="jd_upload",
        label_visibility="collapsed",
        help="Upload the job description as a .txt file"
    )
    if jd_file:
        st.success(f"✓ {jd_file.name} uploaded")

with col_sliders:
    st.markdown('<div class="upload-label">Retrieval Settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Top Candidates to Retrieve", min_value=5, max_value=50, value=10, step=5)

st.markdown("<br>", unsafe_allow_html=True)

# ── Analyze button ────────────────────────────────────────────────────────────
_, btn_col, _ = st.columns([1, 1.2, 1])
with btn_col:
    analyze_clicked = st.button("✦ Analyze Candidates", use_container_width=True)

if analyze_clicked:
    if not jd_file:
        st.warning("Please upload a Job Description file.")
        st.stop()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found. Please set it in your .env file or Streamlit secrets.")
        st.stop()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    progress = st.progress(0)
    status = st.empty()

    try:
        retriever = get_retriever(api_key)

        status.info("⟳ Parsing job description...")
        jd_text = jd_file.read().decode("utf-8")
        jd_json_str = call_jd_gemini_api(jd_text, api_key)
        jd_json = json.loads(jd_json_str)
        progress.progress(20)

        status.info("⟳ Searching vector database for top candidates...")
        top_results = retriever.hybrid_search(jd_json, top_k=top_k)
        
        parsed_resumes = [resume_json for _, resume_json in top_results]
        progress.progress(60)

        status.info("⟳ Running comparative analysis across top candidates...")
        results = call_llm_fit_scorer(parsed_resumes, jd_json, api_key)
        progress.progress(100)

        if not results:
            status.error("No results returned. Please check your API key and try again.")
            st.stop()

        # FIX: persist results across reruns caused by filter dropdown
        st.session_state["results"] = results
        st.session_state["jd_json"] = jd_json

        status.success(f"✓ Analysis complete — {len(results)} candidates ranked")

    except Exception as e:
        status.error(f"Something went wrong: {str(e)}")
        st.stop()


# ── Results — outside if block so filter reruns don't wipe it ────────────────
if "results" in st.session_state:
    results = st.session_state["results"]
    jd_json = st.session_state["jd_json"]

    st.markdown("<br>", unsafe_allow_html=True)

    strong  = sum(1 for r in results if r.get("recommendation") == "Strong Fit")
    moderate = sum(1 for r in results if r.get("recommendation") == "Moderate Fit")
    not_fit  = sum(1 for r in results if r.get("recommendation") == "Not Fit")

    st.markdown(f"""
    <div class="results-header">Candidate Rankings</div>
    <div class="results-sub">{len(results)} candidates evaluated against <em>{escape(jd_json.get('job_title', 'the role'))}</em></div>
    <div class="stats-row">
        <div class="stat-chip">
            <div class="stat-chip-number">{len(results)}</div>
            <div class="stat-chip-label">Total Evaluated</div>
        </div>
        <div class="stat-chip">
            <div class="stat-chip-number strong-num">{strong}</div>
            <div class="stat-chip-label">Strong Fit</div>
        </div>
        <div class="stat-chip">
            <div class="stat-chip-number moderate-num">{moderate}</div>
            <div class="stat-chip-label">Moderate Fit</div>
        </div>
        <div class="stat-chip">
            <div class="stat-chip-number notfit-num">{not_fit}</div>
            <div class="stat-chip-label">Not Fit</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    filter_col, _ = st.columns([1.2, 2])
    with filter_col:
        filter_option = st.selectbox(
            "Filter by recommendation",
            ["All", "Strong Fit", "Moderate Fit", "Not Fit"],
            label_visibility="collapsed"
        )

    filtered = results if filter_option == "All" else [
        r for r in results if r.get("recommendation") == filter_option
    ]

    if not filtered:
        st.info(f"No candidates with recommendation: {filter_option}")
    else:
        for rank, candidate in enumerate(filtered, 1):
            score = candidate.get("score", 0)
            rec   = candidate.get("recommendation", "")
            card_class, score_class, rec_class, _ = get_score_class(score)
            rank_class = get_rank_class(rank)
            name   = clean_candidate_id(candidate.get("candidate_id", "Unknown"))
            file_id = candidate.get("candidate_id", "")

            # FIX: use parse_list_field so these always come back as plain Python lists
            strengths = parse_list_field(candidate.get("key_strengths", []))
            gaps      = parse_list_field(candidate.get("key_gaps", []))
            summary   = candidate.get("summary", "")
            if not isinstance(summary, str):
                summary = str(summary)

            # Build strengths as bullet items
            strengths_items = "".join(
                f'<div class="jd-resp-item"><span class="jd-resp-dot" style="color:#10b981;">▸</span><span style="color:#6ee7b7;">{escape(str(s))}</span></div>'
                for s in strengths
            ) or '<span style="color:#4b5563">—</span>'

            # Build gaps as bullet items
            gaps_items = "".join(
                f'<div class="jd-resp-item"><span class="jd-resp-dot" style="color:#ef4444;">▸</span><span style="color:#fca5a5;">{escape(str(g))}</span></div>'
                for g in gaps
            ) or '<span style="color:#4b5563">—</span>'

            # Build card HTML without indentation to prevent Streamlit
            # from treating it as a markdown code block
            card_html = (
                f'<div class="candidate-card {card_class}">'
                f'<div style="display:flex;align-items:flex-start;gap:1rem;">'
                f'<div style="display:flex;align-items:center;flex:1;min-width:0;">'
                f'<div class="rank-badge {rank_class}">#{rank}</div>'
                f'<div style="min-width:0;">'
                f'<div class="candidate-name">{escape(name)}</div>'
                f'<div class="candidate-file">{escape(file_id)}</div>'
                f'</div></div>'
                f'<div class="score-block" style="flex-shrink:0;">'
                f'<div class="score-number {score_class}">{score}</div>'
                f'<div class="score-label">Score</div>'
                f'<div class="rec-pill {rec_class}">{escape(rec)}</div>'
                f'</div></div>'
                f'<div style="padding:0.5rem 0;margin-top:1rem;border-top:1px solid rgba(255,255,255,0.05);">'
                f'<div class="jd-row">'
                f'<div class="jd-key" style="color:#10b981;">Key Strengths</div>'
                f'<div class="jd-val">{strengths_items}</div>'
                f'</div>'
                f'<div class="jd-row">'
                f'<div class="jd-key" style="color:#f87171;">Key Gaps</div>'
                f'<div class="jd-val">{gaps_items}</div>'
                f'</div>'
                f'<div class="jd-row">'
                f'<div class="jd-key">Summary</div>'
                f'<div class="jd-val">{escape(summary)}</div>'
                f'</div>'
                f'</div></div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

    # ── JD expander ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("View parsed job description"):
        req_skills      = jd_json.get("required_skills", [])
        nice_skills     = jd_json.get("nice_to_have_skills", [])
        responsibilities = jd_json.get("responsibilities", [])

        req_tags   = "".join(f'<span class="jd-skill-tag">{escape(s)}</span>' for s in req_skills)   or "<span style='color:#4b5563'>—</span>"
        nice_tags  = "".join(f'<span class="jd-nice-tag">{escape(s)}</span>'  for s in nice_skills)  or "<span style='color:#4b5563'>—</span>"
        resp_items = "".join(
            f'<div class="jd-resp-item"><span class="jd-resp-dot">▸</span><span>{escape(r)}</span></div>'
            for r in responsibilities
        ) or "<span style='color:#4b5563'>—</span>"

        st.markdown(f"""
        <div style="padding: 0.5rem 0;">
            <div class="jd-row">
                <div class="jd-key">Role</div>
                <div class="jd-val">{escape(jd_json.get('job_title', '—'))}</div>
            </div>
            <div class="jd-row">
                <div class="jd-key">Experience</div>
                <div class="jd-val">{escape(jd_json.get('experience_level', '—'))}</div>
            </div>
            <div class="jd-row">
                <div class="jd-key">Education</div>
                <div class="jd-val">{escape(jd_json.get('education', '—'))}</div>
            </div>
            <div class="jd-row">
                <div class="jd-key">Required Skills</div>
                <div class="jd-val">{req_tags}</div>
            </div>
            <div class="jd-row">
                <div class="jd-key">Nice to Have</div>
                <div class="jd-val">{nice_tags}</div>
            </div>
            <div class="jd-row">
                <div class="jd-key">Responsibilities</div>
                <div class="jd-val">{resp_items}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)