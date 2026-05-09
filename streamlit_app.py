import streamlit as st
import json
import os
import tempfile
import uuid
from datetime import datetime
from html import escape
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader

from agent.graph import create_agent_graph
from agent.audit import log_override
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener (Agentic)",
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

h1, h2, h3 { font-family: 'Syne', sans-serif; letter-spacing: -0.02em; }

.hero { text-align: center; padding: 3.5rem 1rem 2.5rem; }
.hero-badge {
    display: inline-block; background: rgba(139, 92, 246, 0.15);
    border: 1px solid rgba(139, 92, 246, 0.3); color: #a78bfa;
    font-size: 0.75rem; font-weight: 500; letter-spacing: 0.12em;
    text-transform: uppercase; padding: 0.35rem 1rem; border-radius: 100px;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-size: clamp(2.5rem, 5vw, 4rem); font-weight: 800; line-height: 1.05;
    background: linear-gradient(135deg, #ffffff 0%, #a78bfa 50%, #60a5fa 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 1rem;
}
.hero-sub { color: #9ca3af; max-width: 100%; margin: 0 auto; text-align: center; white-space: nowrap; }
.divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(139,92,246,0.3), transparent); margin: 0.5rem 0 2.5rem; }

.candidate-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 1.5rem 1.75rem; margin-bottom: 1rem;
    position: relative; overflow: hidden;
}
.candidate-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 2px; border-radius: 16px 16px 0 0;
}
.candidate-card.hire::before { background: linear-gradient(90deg, #10b981, #34d399); }
.candidate-card.maybe::before { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.candidate-card.nohire::before { background: linear-gradient(90deg, #ef4444, #f87171); }

.score-number { font-family: 'Syne', sans-serif; font-size: 2.2rem; font-weight: 800; line-height: 1; }
.score-hire { color: #10b981; }
.score-maybe { color: #f59e0b; }
.score-nohire { color: #ef4444; }

.rec-pill { display: inline-block; padding: 0.25rem 0.85rem; border-radius: 100px; font-size: 0.75rem; font-weight: 600; margin-top: 0.4rem; }
.rec-hire { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.25); }
.rec-maybe { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.25); }
.rec-nohire { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.25); }

.dim-row { display: flex; align-items: center; margin-bottom: 0.5rem; }
.dim-label { width: 150px; font-size: 0.85rem; color: #d1d5db; }
.dim-score { width: 40px; font-weight: bold; color: #a78bfa; }
.dim-just { flex: 1; font-size: 0.8rem; color: #9ca3af; font-style: italic; }

</style>
""", unsafe_allow_html=True)

# ── Init Agent ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_agent():
    return create_agent_graph()

agent = get_agent()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# ── Helper functions ──────────────────────────────────────────────────────────

def get_rec_class(rec):
    if rec == "Hire": return "hire", "score-hire", "rec-hire"
    if rec == "Maybe": return "maybe", "score-maybe", "rec-maybe"
    return "nohire", "score-nohire", "rec-nohire"

def generate_report(candidates):
    env = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), 'templates')))
    template = env.get_template('report.html')
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return template.render(candidates=candidates, date=date_str)

# ── UI Layout ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ LangGraph Agent</div>
    <h1 class="hero-title">Intelligent HR Screener</h1>
    <p class="hero-sub">Upload JD and Candidate profiles. The agent parses, scores via 5-dimension rubric, and allows human override.</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    jd_file = st.file_uploader("1. Upload JD (.txt)", type=["txt"])
    
with col2:
    resume_files = st.file_uploader("2. Upload Resumes (.pdf, .docx)", type=["pdf", "docx"], accept_multiple_files=True)
    
with col3:
    linkedin_files = st.file_uploader("3. Upload LinkedIn (.json)", type=["json"], accept_multiple_files=True)

st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 1, 1])

if btn_col.button("✦ Start Agentic Screening", use_container_width=True):
    if not jd_file:
        st.warning("Please upload a Job Description.")
        st.stop()
        
    # Removed check for uploaded resumes to allow autonomous talent pool search
        
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY is not set.")
        st.stop()
        
    with st.spinner("Agent is analyzing inputs..."):
        # Save files to temp
        temp_dir = tempfile.mkdtemp()
        paths = []
        all_files = (resume_files or []) + (linkedin_files or [])
        for f in all_files:
            path = os.path.join(temp_dir, f.name)
            with open(path, "wb") as out:
                out.write(f.read())
            paths.append(path)
            
        jd_text = jd_file.read().decode("utf-8")
        
        initial_state = {
            "jd_text": jd_text,
            "uploaded_file_paths": paths,
            "search_triggered": False,
            "all_candidates": [],
            "scored_candidates": []
        }
        
        import time
        # Invoke agent
        for event in agent.stream(initial_state, config):
            for k, v in event.items():
                st.write(f"Agent Action: **{k}** complete.")
                if v and "error" in v:
                    st.error(v["error"])
                    st.stop()
                
                if k == "score_candidates":
                    current_state = agent.get_state(config).values
                    # If no strong fit and search hasn't been triggered yet, it means we are about to search the talent pool
                    if not current_state.get("has_strong_fit", False) and not current_state.get("search_triggered", False):
                        st.info("No good fits found in the given pool of candidates, hence searching internal talent pool.", icon="🔍")
        
        st.session_state.agent_state = agent.get_state(config).values
        st.success("Agent paused for Human Review.")

# ── HIL Review UI ─────────────────────────────────────────────────────────────

if "agent_state" in st.session_state:
    state = st.session_state.agent_state
    scored = state.get("scored_candidates", [])
    
    st.markdown("## Human-in-the-Loop Review")
    st.info("Review the AI's 5-dimension rubric scores. You can override any score based on human intuition or out-of-band context.")
    
    # Render candidates
    for i, c in enumerate(scored):
        rec = c.get("recommendation", "No Hire")
        card_cls, score_cls, rec_cls = get_rec_class(rec)
        
        dims = c.get("dimensions", {})
        
        st.markdown(f"""
        <div class="candidate-card {card_cls}">
            <div style="display:flex; justify-content: space-between;">
                <div>
                    <h2>{escape(c.get('candidate_id'))}</h2>
                    <div class="rec-pill {rec_cls}">{rec}</div>
                </div>
                <div style="text-align: right;">
                    <div class="score-number {score_cls}">{c.get('weighted_total', 0):.2f} / 10</div>
                    <div style="font-size:0.8rem; color:#9ca3af;">Weighted Total</div>
                </div>
            </div>
            <p style="color:#d1d5db; margin-top: 10px;">{escape(c.get('summary', ''))}</p>
            <div style="margin-top: 15px;">
                <div class="dim-row"><div class="dim-label">Skills (30%)</div><div class="dim-score">{dims.get('skills_match', {}).get('score', 0)}</div><div class="dim-just">{escape(dims.get('skills_match', {}).get('justification', ''))}</div></div>
                <div class="dim-row"><div class="dim-label">Experience (25%)</div><div class="dim-score">{dims.get('experience_relevance', {}).get('score', 0)}</div><div class="dim-just">{escape(dims.get('experience_relevance', {}).get('justification', ''))}</div></div>
                <div class="dim-row"><div class="dim-label">Education (15%)</div><div class="dim-score">{dims.get('education_certs', {}).get('score', 0)}</div><div class="dim-just">{escape(dims.get('education_certs', {}).get('justification', ''))}</div></div>
                <div class="dim-row"><div class="dim-label">Projects (20%)</div><div class="dim-score">{dims.get('project_portfolio', {}).get('score', 0)}</div><div class="dim-just">{escape(dims.get('project_portfolio', {}).get('justification', ''))}</div></div>
                <div class="dim-row"><div class="dim-label">Communication (10%)</div><div class="dim-score">{dims.get('communication_quality', {}).get('score', 0)}</div><div class="dim-just">{escape(dims.get('communication_quality', {}).get('justification', ''))}</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander(f"Override Scores for {c.get('candidate_id')}"):
            with st.form(key=f"form_{i}"):
                colA, colB, colC = st.columns(3)
                s1 = colA.slider("Skills Match", 0, 10, dims.get('skills_match', {}).get('score', 0), key=f"s1_{i}")
                s2 = colB.slider("Experience", 0, 10, dims.get('experience_relevance', {}).get('score', 0), key=f"s2_{i}")
                s3 = colC.slider("Education", 0, 10, dims.get('education_certs', {}).get('score', 0), key=f"s3_{i}")
                
                colD, colE, _ = st.columns(3)
                s4 = colD.slider("Projects", 0, 10, dims.get('project_portfolio', {}).get('score', 0), key=f"s4_{i}")
                s5 = colE.slider("Communication", 0, 10, dims.get('communication_quality', {}).get('score', 0), key=f"s5_{i}")
                
                reason = st.text_input("Reason for override (required if changing)", key=f"reason_{i}")
                submit = st.form_submit_button("Apply Override")
                
                if submit:
                    if not reason:
                        st.error("Please provide a reason for the override to maintain audit trail.")
                    else:
                        # Log and update locally
                        log_override(st.session_state.thread_id, c.get('candidate_id'), "skills_match", dims['skills_match']['score'], dims['skills_match']['justification'], s1, reason)
                        log_override(st.session_state.thread_id, c.get('candidate_id'), "experience", dims['experience_relevance']['score'], dims['experience_relevance']['justification'], s2, reason)
                        
                        dims['skills_match']['score'] = s1
                        dims['experience_relevance']['score'] = s2
                        dims['education_certs']['score'] = s3
                        dims['project_portfolio']['score'] = s4
                        dims['communication_quality']['score'] = s5
                        
                        # Recalculate
                        w_tot = (s1*0.3) + (s2*0.25) + (s3*0.15) + (s4*0.2) + (s5*0.1)
                        c['weighted_total'] = w_tot
                        if w_tot >= 7.0: c['recommendation'] = "Hire"
                        elif w_tot >= 5.0: c['recommendation'] = "Maybe"
                        else: c['recommendation'] = "No Hire"
                        
                        # Update state
                        st.session_state.agent_state["scored_candidates"][i] = c
                        agent.update_state(config, {"scored_candidates": st.session_state.agent_state["scored_candidates"]})
                        st.toast("Overrides applied and logged!", icon="✅")
                        import time
                        time.sleep(1)
                        st.rerun()

    st.markdown("---")
    _, btn2, _ = st.columns([1,1,1])
    if btn2.button("Finalize & Generate Report", type="primary", use_container_width=True):
        with st.spinner("Generating HTML Report..."):
            # Resume graph (it will go to END)
            agent.invoke(Command(resume="continue"), config)
            
            # Generate Report
            html_content = generate_report(st.session_state.agent_state["scored_candidates"])
            st.session_state.report_html = html_content
            st.success("Report generated!")
            
if "report_html" in st.session_state:
    st.download_button(
        label="Download HTML Report",
        data=st.session_state.report_html,
        file_name="candidate_shortlist.html",
        mime="text/html",
        use_container_width=True
    )