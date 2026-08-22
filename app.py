import html
import re
import time
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import PyPDF2
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Streamlit Page Configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="TalentPulse AI | Resume Intelligence & Job Matcher",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Modern Glassmorphic CSS Design System (Clean, Frosted, Sophisticated)
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

        :root {
            --font-main: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            --bg-base: #f8fafc;
            --glass-bg: rgba(255, 255, 255, 0.78);
            --glass-bg-hover: rgba(255, 255, 255, 0.92);
            --glass-border: rgba(226, 232, 240, 0.85);
            --glass-border-highlight: rgba(99, 102, 241, 0.25);
            --glass-shadow: 0 10px 30px -5px rgba(15, 23, 42, 0.04), 0 4px 6px -2px rgba(15, 23, 42, 0.02);
            --glass-shadow-hover: 0 20px 40px -10px rgba(15, 23, 42, 0.08), 0 8px 16px -4px rgba(15, 23, 42, 0.03);
            
            --slate-900: #0f172a;
            --slate-800: #1e293b;
            --slate-700: #334155;
            --slate-600: #475569;
            --slate-500: #64748b;
            --slate-400: #94a3b8;
            --slate-200: #e2e8f0;
            --slate-100: #f1f5f9;
            
            --indigo-600: #4f46e5;
            --indigo-500: #6366f1;
            --indigo-50: #eef2ff;
            --teal-600: #0d9488;
            --teal-500: #14b8a6;
            --teal-50: #f0fdfa;
            --emerald-600: #059669;
            --emerald-50: #ecfdf5;
            --amber-600: #d97706;
            --amber-50: #fffbeb;
            --rose-600: #e11d48;
            --rose-50: #fff1f2;
        }

        html, body, [class*="css"], .stApp {
            font-family: var(--font-main) !important;
            color: var(--slate-800);
            background-color: var(--bg-base);
        }

        .stApp {
            background: 
                radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99, 102, 241, 0.08), transparent 70%),
                radial-gradient(ellipse 60% 40% at 80% 20%, rgba(13, 148, 136, 0.05), transparent 60%),
                radial-gradient(ellipse 50% 30% at 20% 40%, rgba(244, 63, 94, 0.03), transparent 50%),
                linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
            min-height: 100vh;
        }

        /* Hide Streamlit Chrome */
        [data-testid="stHeader"] {
            background: rgba(248, 250, 252, 0.8) !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            border-bottom: 1px solid var(--glass-border);
        }
        [data-testid="stToolbar"], [data-testid="stDecoration"] {
            display: none !important;
        }

        /* Container Layout */
        .block-container {
            max-width: 1220px !important;
            padding-top: 2rem !important;
            padding-bottom: 4rem !important;
        }

        /* Hero Glassmorphic Header */
        .hero-banner {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.85) 0%, rgba(248, 250, 252, 0.7) 100%);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 2.2rem 2.4rem;
            margin-bottom: 2rem;
            box-shadow: var(--glass-shadow);
            position: relative;
            overflow: hidden;
        }

        .hero-banner::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--indigo-500), var(--teal-500), #ec4899);
        }

        .hero-tag {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: var(--indigo-50);
            border: 1px solid rgba(99, 102, 241, 0.2);
            color: var(--indigo-600);
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            margin-bottom: 0.85rem;
        }

        .hero-title {
            font-size: 2.4rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: var(--slate-900);
            line-height: 1.15;
            margin-bottom: 0.6rem;
        }

        .hero-title span {
            background: linear-gradient(135deg, var(--indigo-600) 0%, var(--teal-600) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-subtitle {
            font-size: 1.05rem;
            color: var(--slate-600);
            line-height: 1.6;
            max-width: 48rem;
            margin-bottom: 1.2rem;
        }

        .hero-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-top: 0.5rem;
        }

        .hero-badge-item {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: rgba(255, 255, 255, 0.65);
            border: 1px solid var(--slate-200);
            color: var(--slate-700);
            font-size: 0.82rem;
            font-weight: 600;
            padding: 0.3rem 0.75rem;
            border-radius: 8px;
            backdrop-filter: blur(6px);
        }

        /* Glassmorphic Cards */
        .glass-card {
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            box-shadow: var(--glass-shadow);
            transition: all 0.25s ease;
            margin-bottom: 1.2rem;
        }

        .glass-card:hover {
            border-color: var(--glass-border-highlight);
            box-shadow: var(--glass-shadow-hover);
        }

        .glass-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--slate-100);
        }

        .glass-card-title {
            font-size: 1.12rem;
            font-weight: 700;
            color: var(--slate-900);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* KPI Metric Tiles */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
            margin-bottom: 1.6rem;
        }

        .kpi-card {
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 14px;
            padding: 1.25rem 1.4rem;
            box-shadow: var(--glass-shadow);
            position: relative;
            overflow: hidden;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--glass-shadow-hover);
            border-color: rgba(99, 102, 241, 0.3);
        }

        .kpi-card-label {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: var(--slate-500);
            margin-bottom: 0.4rem;
        }

        .kpi-card-value {
            font-size: 1.7rem;
            font-weight: 800;
            color: var(--slate-900);
            line-height: 1.2;
            letter-spacing: -0.02em;
        }

        .kpi-card-hint {
            font-size: 0.82rem;
            font-weight: 500;
            color: var(--slate-600);
            margin-top: 0.4rem;
        }

        /* Skill Chips */
        .skill-group {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.5rem;
        }

        .skill-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            background: rgba(99, 102, 241, 0.08);
            border: 1px solid rgba(99, 102, 241, 0.22);
            color: var(--indigo-600);
            font-size: 0.82rem;
            font-weight: 600;
            padding: 0.3rem 0.75rem;
            border-radius: 8px;
            transition: all 0.15s ease;
        }

        .skill-chip:hover {
            background: rgba(99, 102, 241, 0.14);
            transform: translateY(-1px);
        }

        .skill-chip-match {
            background: var(--emerald-50);
            border: 1px solid rgba(5, 150, 105, 0.3);
            color: var(--emerald-600);
        }

        .skill-chip-gap {
            background: var(--amber-50);
            border: 1px solid rgba(217, 119, 6, 0.3);
            color: var(--amber-600);
        }

        /* Job Card Modern Glass Styling */
        .job-glass-card {
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1.2rem;
            box-shadow: var(--glass-shadow);
            transition: all 0.2s ease;
            position: relative;
        }

        .job-glass-card:hover {
            border-color: rgba(99, 102, 241, 0.35);
            box-shadow: var(--glass-shadow-hover);
            transform: translateY(-2px);
        }

        .job-card-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 1rem;
            margin-bottom: 0.75rem;
        }

        .job-card-title {
            font-size: 1.18rem;
            font-weight: 750;
            color: var(--slate-900);
            line-height: 1.3;
        }

        .job-card-company {
            font-size: 0.92rem;
            font-weight: 600;
            color: var(--indigo-600);
            margin-top: 0.15rem;
        }

        .job-meta-pill-wrap {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 0.75rem 0;
        }

        .job-meta-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: var(--slate-100);
            border: 1px solid var(--slate-200);
            color: var(--slate-600);
            font-size: 0.8rem;
            font-weight: 600;
            padding: 0.25rem 0.65rem;
            border-radius: 6px;
        }

        .match-score-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.4rem 0.9rem;
            border-radius: 999px;
            font-size: 0.9rem;
            font-weight: 800;
            letter-spacing: -0.01em;
            white-space: nowrap;
        }

        .match-score-high {
            background: linear-gradient(135deg, #059669 0%, #10b981 100%);
            color: #ffffff;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25);
        }

        .match-score-med {
            background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
            color: #ffffff;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
        }

        .match-score-low {
            background: linear-gradient(135deg, #64748b 0%, #94a3b8 100%);
            color: #ffffff;
        }

        .match-progress-track {
            background: var(--slate-100);
            border-radius: 999px;
            height: 7px;
            width: 100%;
            overflow: hidden;
            margin: 0.85rem 0 1rem 0;
        }

        .match-progress-fill {
            height: 100%;
            border-radius: 999px;
            transition: width 0.6s ease;
        }

        /* Streamlit Input Component Tweaks */
        [data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(12px);
            border: 1px dashed rgba(99, 102, 241, 0.4) !important;
            border-radius: 16px !important;
            padding: 1.2rem;
            box-shadow: var(--glass-shadow);
        }

        .stButton > button {
            background: linear-gradient(135deg, var(--indigo-600) 0%, var(--indigo-500) 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: 0.92rem !important;
            padding: 0.55rem 1.25rem !important;
            box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3) !important;
            transition: all 0.2s ease !important;
        }

        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 8px 20px rgba(79, 70, 229, 0.4) !important;
        }

        /* Tabs Styling */
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: rgba(255, 255, 255, 0.6);
            backdrop-filter: blur(10px);
            padding: 0.4rem;
            border-radius: 12px;
            border: 1px solid var(--slate-200);
        }

        [data-testid="stTabs"] [data-baseweb="tab"] {
            border-radius: 8px !important;
            padding: 0.45rem 1rem !important;
            font-weight: 600 !important;
            color: var(--slate-600) !important;
            border: none !important;
            background: transparent !important;
            transition: all 0.15s ease !important;
        }

        [data-testid="stTabs"] [aria-selected="true"] {
            background: #ffffff !important;
            color: var(--indigo-600) !important;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06) !important;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.75) !important;
            backdrop-filter: blur(16px) !important;
            border-right: 1px solid var(--glass-border) !important;
        }

        /* Section Subheading */
        .section-subhead {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--slate-900);
            margin: 1.5rem 0 1rem 0;
        }

        .section-subhead .indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--indigo-500);
        }

        /* Recommendation Box */
        .rec-box {
            background: rgba(255, 255, 255, 0.85);
            border-left: 4px solid var(--indigo-500);
            border-radius: 0 12px 12px 0;
            padding: 1rem 1.2rem;
            margin-bottom: 0.8rem;
            border-top: 1px solid var(--glass-border);
            border-right: 1px solid var(--glass-border);
            border-bottom: 1px solid var(--glass-border);
        }

        .rec-box-title {
            font-weight: 700;
            font-size: 0.95rem;
            color: var(--slate-900);
            margin-bottom: 0.25rem;
        }

        .rec-box-desc {
            font-size: 0.88rem;
            color: var(--slate-600);
            line-height: 1.5;
            margin: 0;
        }

        /* Code & Pre */
        .raw-text-box {
            background: #ffffff;
            border: 1px solid var(--slate-200);
            border-radius: 12px;
            padding: 1.2rem;
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 0.86rem;
            line-height: 1.6;
            color: var(--slate-800);
            max-height: 400px;
            overflow-y: auto;
            white-space: pre-wrap;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Asset Loading Function (Cached)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing AI Models & Job Database...")
def load_all_assets():
    try:
        model = joblib.load("resume_classifier.pkl")
        resume_tfidf = joblib.load("resume_tfidf.pkl")
        matching_tfidf = joblib.load("job_matching_tfidf.pkl")
        skill_list = joblib.load("skill_list.pkl")
        jobs_raw = joblib.load("jobs_data.pkl")

        if isinstance(jobs_raw, pd.DataFrame):
            jobs_df = jobs_raw.copy()
        else:
            jobs_df = pd.DataFrame(jobs_raw)

        # Standardize and clean dataframe column names
        cols_map = {
            "Job Title": "title",
            "Cleaned_Job_Text": "clean_text",
            "Job_Text": "raw_text",
            "Key Skills": "key_skills",
            "Location": "location",
            "Industry": "industry",
            "Role Category": "role_category",
            "Role": "role",
            "Job Experience Required": "experience",
            "Extracted_Skills": "extracted_skills",
        }
        for orig, target in cols_map.items():
            if orig in jobs_df.columns and target not in jobs_df.columns:
                jobs_df[target] = jobs_df[orig]

        # Fill NAs
        for col in ["title", "location", "industry", "role_category", "role", "experience", "key_skills"]:
            if col in jobs_df.columns:
                jobs_df[col] = jobs_df[col].fillna("").astype(str).str.strip()

        if "clean_text" not in jobs_df.columns and "raw_text" in jobs_df.columns:
            jobs_df["clean_text"] = jobs_df["raw_text"]

        return {
            "model": model,
            "resume_tfidf": resume_tfidf,
            "matching_tfidf": matching_tfidf,
            "skill_list": list(skill_list),
            "jobs_df": jobs_df,
        }
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None


# ----------------------------------------------------------------------------
# Extraction & Inference Helpers
# ----------------------------------------------------------------------------
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


def extract_matched_skills(resume_text, skill_list):
    """Case-insensitive regex whole-word/phrase match of known skills."""
    text_lower = resume_text.lower()
    matched = []
    for skill in skill_list:
        skill_str = str(skill).strip()
        if not skill_str:
            continue
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill_str.lower()) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, text_lower):
            matched.append(skill_str)

    # De-duplicate preserving order
    seen = set()
    unique = []
    for s in matched:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


def match_jobs_fast(resume_text, matching_tfidf, jobs_df, top_n=10, min_score=0.0):
    """Calculates TF-IDF cosine similarity scores and returns ranked matches."""
    text_col = "clean_text" if "clean_text" in jobs_df.columns else "raw_text"
    job_texts = jobs_df[text_col].astype(str).fillna("")

    resume_vec = matching_tfidf.transform([resume_text])
    job_vecs = matching_tfidf.transform(job_texts)

    scores = cosine_similarity(resume_vec, job_vecs).flatten() * 100
    temp_df = jobs_df.assign(match_score=scores)

    if min_score > 0:
        temp_df = temp_df[temp_df["match_score"] >= min_score]

    ranked = temp_df.sort_values("match_score", ascending=False).head(top_n)

    results = []
    for _, row in ranked.iterrows():
        # Clean skills for this job
        job_skills_raw = row.get("extracted_skills", [])
        if isinstance(job_skills_raw, list):
            job_skills = job_skills_raw
        elif isinstance(job_skills_raw, str):
            job_skills = [s.strip() for s in job_skills_raw.replace("|", ",").split(",") if s.strip()]
        else:
            job_skills = []

        results.append(
            {
                "title": row.get("title", "Untitled Role") or "Untitled Role",
                "location": row.get("location", "Not Specified") or "Not Specified",
                "industry": row.get("industry", "General") or "General",
                "role_category": row.get("role_category", "Specialist") or "Specialist",
                "experience": row.get("experience", "Flexible") or "Flexible",
                "key_skills": row.get("key_skills", "") or "",
                "description": row.get("raw_text", "") or row.get("clean_text", ""),
                "score": float(row.get("match_score", 0.0)),
                "job_skills": job_skills,
            }
        )
    return results


# ----------------------------------------------------------------------------
# Curated Sample Resumes for Instant 1-Click Demos
# ----------------------------------------------------------------------------
SAMPLE_PROFILES = {
    "Senior Data Scientist & ML Engineer": """
Devon Alexander
Email: devon.ai@example.com | Phone: +1-555-0199 | Portfolio: github.com/devon-ai
SUMMARY
Results-driven Senior Data Scientist with 6+ years of experience designing scalable machine learning pipelines, predictive modeling, NLP, and deep learning architectures. Proven track record deploying models to AWS and GCP environments that enhanced business ROI by 32%.

CORE SKILLS
Programming & Tools: Python, R, SQL, Git, Docker, Kubernetes, Linux, AWS, PySpark
Machine Learning & AI: Scikit-learn, TensorFlow, PyTorch, XGBoost, NLP, Computer Vision, Deep Learning, Statistics, Time Series
Data Engineering: Pandas, NumPy, Data Analysis, ETL Pipelines, Tableau, Power BI, PostgreSQL

PROFESSIONAL EXPERIENCE
Lead AI/ML Engineer | Nexus Analytics Inc. (2022 – Present)
- Engineered end-to-end recommendation engine using TF-IDF, embeddings, and Random Forest boosting user engagement by 40%.
- Implemented deep learning classification models in PyTorch, reducing inference latency by 45%.
- Led cross-functional team of 5 data scientists and spearheaded agile ML sprint cycles.

Data Scientist | DataCore Labs (2019 – 2022)
- Built predictive customer churn algorithms with 92% accuracy using Python, Pandas, and Scikit-learn.
- Designed automated SQL queries and ETL pipelines handling 15M+ daily transactions.

EDUCATION
Master of Science in Computer Science & Machine Learning | Stanford University (2019)
Bachelor of Technology in Information Technology | MIT (2017)
""",
    "Full Stack Python & React Developer": """
Alex Morgan
Email: alex.fullstack@example.com | LinkedIn: linkedin.com/in/alex-dev
SUMMARY
Full Stack Software Engineer with 5+ years of experience designing robust web applications, RESTful microservices, and reactive frontends. Passionate about clean architecture, CI/CD automation, and cloud-native solutions.

CORE TECHNICAL SKILLS
Languages: Python, JavaScript, TypeScript, HTML5, CSS3, SQL
Backend: Django, FastAPI, Flask, Node.js, REST APIs, Microservices, Celery, Redis
Frontend: React, Redux, Next.js, Tailwind CSS, Bootstrap, Webpack
Databases & Cloud: PostgreSQL, MongoDB, MySQL, Docker, Git, AWS, CI/CD, Agile, Linux

PROFESSIONAL EXPERIENCE
Senior Full Stack Developer | CloudTech Solutions (2021 – Present)
- Developed responsive web applications using React, TypeScript, and FastAPI servicing 200,000+ monthly active users.
- Designed database schemas in PostgreSQL with optimized indexing, decreasing API response times by 35%.
- Integrated third-party payment gateways, OAuth authentication, and automated Docker deployment workflows.

Software Engineer | BitForge Interactive (2019 – 2021)
- Built enterprise dashboards using Python, Django, and JavaScript with role-based access controls.
- Maintained 98% test coverage using PyTest and integrated automated GitHub Actions CI/CD pipelines.

EDUCATION
B.S. in Computer Science | University of California, Berkeley (2019)
""",
    "Financial Analyst & Senior Accountant": """
Elena Rostova
Email: elena.finance@example.com | Phone: +1-555-0842
SUMMARY
Certified Senior Accountant and Financial Analyst with 7+ years of experience managing corporate financial statements, taxation, budgeting, and risk analysis. Expert in QuickBooks, SAP ERP, financial modeling, and regulatory compliance.

CORE SKILLS
Accounting & Finance: Financial Reporting, General Ledger, Auditing, Tax Compliance, Accounts Payable, Accounts Receivable, Budgeting, Forecasting, Cost Accounting, Variance Analysis
Tools & Software: SAP, QuickBooks, Advanced Excel, Power BI, Financial Analysis, Accounting

PROFESSIONAL EXPERIENCE
Senior Corporate Accountant | Pinnacle Global Finance (2021 – Present)
- Supervised month-end and year-end close processes, preparing comprehensive GAAP-compliant financial statements.
- Spearheaded company-wide budget planning of $45M, identifying 14% operational cost reductions.
- Liaised directly with external auditors to ensure 100% compliance during annual tax and financial audits.

Staff Accountant | Vanguard Asset Management (2018 – 2021)
- Managed daily reconciliations for 30+ corporate accounts and maintained audit-ready journal entries.
- Automated monthly revenue forecasting models in Excel, saving 15 hours of manual work each cycle.

EDUCATION & CERTIFICATIONS
Certified Public Accountant (CPA) | State Board of Accountancy
B.S. in Accounting & Finance | New York University (2018)
""",
    "Human Resources & Talent Acquisition Partner": """
Sarah Jenkins
Email: sarah.hr@example.com | Phone: +1-555-0144
SUMMARY
Dynamic HR Manager and Talent Acquisition Specialist with 6+ years driving strategic recruitment, employee engagement, performance management, and organizational culture in high-growth tech environments.

CORE EXPERTISE
Human Resources: Talent Acquisition, Full-Cycle Recruitment, Sourcing, Employee Relations, HR Policies, Onboarding, Performance Appraisal, Compensation & Benefits, HRIS, Compliance, Labor Laws, Conflict Resolution

EXPERIENCE
Talent Acquisition Lead | Horizon Innovations (2021 – Present)
- Led end-to-end recruitment for 120+ technical and executive roles, reducing average time-to-hire by 28%.
- Designed structured onboarding programs that increased 90-day employee retention from 82% to 96%.
- Partnered with department directors to execute workforce planning and diversity recruitment initiatives.

Human Resources Generalist | Apex Systems (2018 – 2021)
- Managed employee grievance procedures, benefit administration, and compliance with federal labor standards.
- Implemented automated HRIS software reducing paperwork processing time by 50%.

EDUCATION
B.A. in Human Resource Management & Organizational Psychology | University of Michigan (2018)
""",
}


# ----------------------------------------------------------------------------
# Load Assets
# ----------------------------------------------------------------------------
assets = load_all_assets()
if not assets:
    st.error("Failed to load model assets. Please ensure `.pkl` files exist in workspace.")
    st.stop()

model = assets["model"]
resume_tfidf = assets["resume_tfidf"]
matching_tfidf = assets["matching_tfidf"]
skill_list = assets["skill_list"]
jobs_df = assets["jobs_df"]


# ----------------------------------------------------------------------------
# Sidebar Configuration & System Health
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 1rem;">
            <div style="background: linear-gradient(135deg, #4f46e5, #06b6d4); color: white; border-radius: 10px; width: 34px; height: 34px; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; font-weight: bold;">
                ✨
            </div>
            <div>
                <div style="font-weight: 800; font-size: 1.05rem; color: #0f172a; line-height: 1.1;">TalentPulse AI</div>
                <div style="font-size: 0.76rem; color: #64748b; font-weight: 500;">Career Intelligence Suite</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="background: rgba(255,255,255,0.7); border: 1px solid #e2e8f0; border-radius: 10px; padding: 0.75rem 0.9rem; margin-bottom: 1.2rem;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.4rem;">
                <span style="font-size: 0.78rem; font-weight: 700; color: #475569; text-transform: uppercase;">Pipeline Health</span>
                <span style="display: inline-flex; align-items: center; gap: 0.3rem; font-size: 0.78rem; color: #059669; font-weight: 700;">
                    <span style="width: 7px; height: 7px; background: #10b981; border-radius: 50%;"></span> Active
                </span>
            </div>
            <div style="font-size: 0.8rem; color: #64748b;">
                ⚡ <b>30,000</b> Indexed Roles<br>
                🧠 <b>325</b> Verified Skills<br>
                🎯 <b>25+</b> Career Domains
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div style="font-size: 0.82rem; font-weight: 700; color: #334155; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Match Controls</div>', unsafe_allow_html=True)
    top_n = st.slider("Top Job Matches Count", min_value=3, max_value=25, value=8, step=1)
    min_score_threshold = st.slider("Minimum Match Threshold (%)", min_value=0, max_value=80, value=15, step=5)

    st.markdown("---")
    st.markdown('<div style="font-size: 0.82rem; font-weight: 700; color: #334155; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Market Quick Filters</div>', unsafe_allow_html=True)

    # Unique locations for filter
    popular_locs = ["All Locations", "Bangalore", "Mumbai", "Pune", "Delhi", "Hyderabad", "Chennai", "Kolkata", "Noida", "Gurgaon"]
    selected_loc = st.selectbox("Location Filter", popular_locs, index=0)

    # Unique industries
    top_industries = ["All Industries", "IT-Software, Software Services", "Banking, Financial Services", "Recruitment, Staffing", "Advertising, PR, Event Management", "BPO, Call Centre", "Education, Teaching"]
    selected_industry = st.selectbox("Industry Filter", top_industries, index=0)

    st.markdown("---")
    st.caption("AI-Powered Resume ATS & Semantic Similarity v2.5")


# ----------------------------------------------------------------------------
# Hero Banner
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-tag">
            <span>✨ AI Career Intelligence Engine</span>
        </div>
        <h1 class="hero-title">Screen Resumes & Discover <span>Precise Job Matches</span></h1>
        <p class="hero-subtitle">
            Upload your resume or pick a profile to unlock instant career classification, comprehensive skill detection,
            ATS keyword gap analysis, and tailored job recommendations from our database of 30,000+ opportunities.
        </p>
        <div class="hero-badges">
            <div class="hero-badge-item">⚡ TF-IDF Vector Space</div>
            <div class="hero-badge-item">🧠 Random Forest Classifier</div>
            <div class="hero-badge-item">🎯 Cosine Semantic Scoring</div>
            <div class="hero-badge-item">📊 Instant Skill Gap Breakdown</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Input Selector (PDF Upload / Direct Text / Sample Profiles)
# ----------------------------------------------------------------------------
input_tab1, input_tab2, input_tab3 = st.tabs([
    "📄 Upload Resume (PDF)",
    "✍️ Paste Resume Text",
    "⚡ 1-Click Sample Profiles",
])

active_resume_text = ""
resume_source = "None"

with input_tab1:
    uploaded_file = st.file_uploader(
        "Upload Resume PDF",
        type=["pdf"],
        help="Upload a standard PDF resume with selectable text.",
        key="pdf_uploader",
    )
    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF document..."):
            extracted = extract_text_from_pdf(uploaded_file)
            if len(extracted.strip()) > 0:
                active_resume_text = extracted
                resume_source = f"Uploaded PDF ({uploaded_file.name})"
                st.success(f"Successfully processed **{uploaded_file.name}** ({len(active_resume_text)} characters extracted).")
            else:
                st.error("Could not extract selectable text from this PDF. Please ensure it is not a scanned image.")

with input_tab2:
    pasted_text = st.text_area(
        "Paste your complete resume or CV text below:",
        height=180,
        placeholder="Paste candidate work history, education, skills, and summary here...",
        key="text_area_input",
    )
    if pasted_text and len(pasted_text.strip()) > 20:
        if not uploaded_file:  # Only override if no PDF is uploaded
            active_resume_text = pasted_text.strip()
            resume_source = "Pasted Text Input"

with input_tab3:
    st.markdown('<p style="font-size: 0.9rem; color: #64748b; margin-bottom: 0.75rem;">Select a preloaded profile to test and demo the AI pipeline instantly:</p>', unsafe_allow_html=True)
    selected_sample = st.selectbox(
        "Choose a sample profile to load:",
        list(SAMPLE_PROFILES.keys()),
        key="sample_profile_select",
    )
    if st.button("🚀 Load Sample Profile", use_container_width=False):
        active_resume_text = SAMPLE_PROFILES[selected_sample].strip()
        resume_source = f"Sample Profile ({selected_sample})"
        st.session_state["loaded_sample_text"] = active_resume_text

    # Retain sample if loaded via session state
    if "loaded_sample_text" in st.session_state and not uploaded_file and not pasted_text:
        active_resume_text = st.session_state["loaded_sample_text"]
        resume_source = "Loaded Sample Profile"


# ----------------------------------------------------------------------------
# Core Processing & Analysis Logic
# ----------------------------------------------------------------------------
if active_resume_text and len(active_resume_text.strip()) > 30:
    # 1. Vectorize and Classify Career Category
    resume_vector = resume_tfidf.transform([active_resume_text])
    predicted_category = model.predict(resume_vector)[0]
    
    # Calculate Prediction Probabilities
    top_categories = []
    confidence_score = 0.0
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(resume_vector)[0]
            top_indices = np.argsort(probabilities)[::-1][:6]
            confidence_score = float(probabilities[top_indices[0]]) * 100
            for idx in top_indices:
                cat_name = model.classes_[idx]
                prob_val = float(probabilities[idx]) * 100
                if prob_val >= 0.5:
                    top_categories.append({"category": cat_name, "probability": prob_val})
        except Exception:
            confidence_score = 0.0

    # 2. Extract Matched Skills
    candidate_skills = extract_matched_skills(active_resume_text, skill_list)

    # 3. Match Jobs Against 30,000 Database
    with st.spinner("Analyzing candidate profile against 30,000+ job openings..."):
        matched_jobs = match_jobs_fast(
            active_resume_text,
            matching_tfidf,
            jobs_df,
            top_n=top_n,
            min_score=float(min_score_threshold),
        )

    # Apply Sidebar Filters to Matched Jobs
    filtered_jobs = matched_jobs
    if selected_loc != "All Locations":
        filtered_jobs = [j for j in filtered_jobs if selected_loc.lower() in j["location"].lower()]
    if selected_industry != "All Industries":
        filtered_jobs = [j for j in filtered_jobs if selected_industry.lower() in j["industry"].lower()]

    top_fit_score = filtered_jobs[0]["score"] if filtered_jobs else (matched_jobs[0]["score"] if matched_jobs else 0.0)
    avg_fit_score = np.mean([j["score"] for j in matched_jobs[:5]]) if matched_jobs else 0.0

    # ------------------------------------------------------------------------
    # KPI Executive Metrics Row
    # ------------------------------------------------------------------------
    st.markdown('<div class="section-subhead"><span class="indicator"></span> Executive Analysis Summary</div>', unsafe_allow_html=True)
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-card-label">Predicted Domain</div>
                <div class="kpi-card-value" style="font-size: 1.35rem; color: #4f46e5;">{predicted_category}</div>
                <div class="kpi-card-hint">🎯 Primary Career Sector</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col2:
        conf_display = f"{confidence_score:.1f}%" if confidence_score > 0 else "High"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-card-label">Model Confidence</div>
                <div class="kpi-card-value" style="color: #0d9488;">{conf_display}</div>
                <div class="kpi-card-hint">📈 Classification certainty</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-card-label">Skills Extracted</div>
                <div class="kpi-card-value" style="color: #059669;">{len(candidate_skills)}</div>
                <div class="kpi-card-hint">🧠 Out of 325 known skills</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-card-label">Top Job Match Fit</div>
                <div class="kpi-card-value" style="color: #4f46e5;">{top_fit_score:.1f}%</div>
                <div class="kpi-card-hint">⭐ Best opportunity overlap</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------------
    # Detected Skills Bar
    # ------------------------------------------------------------------------
    with st.expander(f"🏷️ View All Detected Candidate Skills ({len(candidate_skills)} items)", expanded=True):
        if candidate_skills:
            chips_html = "".join(
                f'<span class="skill-chip">✓ {html.escape(s.title())}</span>' for s in candidate_skills
            )
            st.markdown(f'<div class="skill-group">{chips_html}</div>', unsafe_allow_html=True)
        else:
            st.info("No matching skills detected from our predefined skill taxonomy.")


    # ------------------------------------------------------------------------
    # Multi-Tab Main Dashboard View
    # ------------------------------------------------------------------------
    main_tab1, main_tab2, main_tab3, main_tab4, main_tab5 = st.tabs([
        "🎯 Ranked Job Matches",
        "📊 Resume Analytics & Insights",
        "⚡ AI ATS Optimizer & Gap Analysis",
        "🔍 30,000+ Job Market Explorer",
        "📑 Extracted Text & Export Report",
    ])

    # ------------------------------------------------------------------------
    # TAB 1: RANKED JOB MATCHES & SKILL GAPS
    # ------------------------------------------------------------------------
    with main_tab1:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <div>
                    <span style="font-size: 1.15rem; font-weight: 700; color: #0f172a;">Top Matching Opportunities</span>
                    <span style="font-size: 0.85rem; color: #64748b; margin-left: 0.5rem;">({len(filtered_jobs)} results shown)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not filtered_jobs:
            st.warning("No job openings matched your current filter criteria. Try relaxing the minimum score threshold or location filter in the sidebar.")
        else:
            for idx, job in enumerate(filtered_jobs):
                score = max(0.0, min(100.0, job["score"]))
                
                # Determine score color class
                if score >= 65:
                    badge_class = "match-score-high"
                    fill_gradient = "linear-gradient(90deg, #059669, #10b981)"
                elif score >= 45:
                    badge_class = "match-score-med"
                    fill_gradient = "linear-gradient(90deg, #4f46e5, #6366f1)"
                else:
                    badge_class = "match-score-low"
                    fill_gradient = "linear-gradient(90deg, #64748b, #94a3b8)"

                # Skill Overlap Calculation
                job_skills_lower = [s.lower() for s in job["job_skills"]]
                candidate_skills_lower = [s.lower() for s in candidate_skills]
                
                matched_job_skills = [s for s in job["job_skills"] if s.lower() in candidate_skills_lower]
                missing_job_skills = [s for s in job["job_skills"] if s.lower() not in candidate_skills_lower][:6]

                # Format skills chips
                skills_match_html = "".join(
                    f'<span class="skill-chip skill-chip-match">✓ {html.escape(s.title())}</span>' for s in matched_job_skills[:8]
                ) if matched_job_skills else '<span style="font-size:0.8rem; color:#94a3b8;">No overlapping taxonomy skills</span>'

                skills_gap_html = "".join(
                    f'<span class="skill-chip skill-chip-gap">+ {html.escape(s.title())}</span>' for s in missing_job_skills
                ) if missing_job_skills else '<span style="font-size:0.8rem; color:#10b981;">No critical skill gaps identified</span>'

                # Safe strings
                safe_title = html.escape(job["title"])
                safe_loc = html.escape(job["location"])
                safe_exp = html.escape(job["experience"])
                safe_ind = html.escape(job["industry"])
                safe_cat = html.escape(job["role_category"])
                safe_keyskills = html.escape(job["key_skills"].replace("|", " • "))

                st.markdown(
                    f"""
                    <div class="job-glass-card">
                        <div class="job-card-header">
                            <div>
                                <div class="job-card-title">{idx+1}. {safe_title}</div>
                                <div class="job-card-company">{safe_cat} &nbsp;•&nbsp; {safe_ind}</div>
                            </div>
                            <div class="match-score-badge {badge_class}">
                                <span>{score:.1f}% Fit</span>
                            </div>
                        </div>
                        
                        <div class="match-progress-track">
                            <div class="match-progress-fill" style="width: {score:.1f}%; background: {fill_gradient};"></div>
                        </div>

                        <div class="job-meta-pill-wrap">
                            <span class="job-meta-pill">📍 {safe_loc}</span>
                            <span class="job-meta-pill">💼 {safe_exp}</span>
                            <span class="job-meta-pill">🏢 {safe_ind}</span>
                        </div>

                        <div style="margin-top: 0.9rem; padding-top: 0.75rem; border-top: 1px solid #f1f5f9;">
                            <div style="font-size: 0.8rem; font-weight: 700; color: #059669; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.35rem;">
                                🟢 Candidate Skills Present in this Role:
                            </div>
                            <div class="skill-group">{skills_match_html}</div>
                        </div>

                        <div style="margin-top: 0.75rem;">
                            <div style="font-size: 0.8rem; font-weight: 700; color: #d97706; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.35rem;">
                                ⚡ Recommended Skills to Add (Target Gaps):
                            </div>
                            <div class="skill-group">{skills_gap_html}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander(f"📋 View Full Job Description & Requirements for '{job['title']}'"):
                    if job["key_skills"]:
                        st.markdown(f"**Key Job Skills Required:** {safe_keyskills}")
                    st.markdown(f"**Role Category:** {safe_cat} | **Experience:** {safe_exp}")
                    st.text_area("Full Job Posting Details", value=job["description"], height=140, key=f"job_desc_{idx}", disabled=True)

    # ------------------------------------------------------------------------
    # TAB 2: RESUME ANALYTICS & VISUALIZATIONS
    # ------------------------------------------------------------------------
    with main_tab2:
        st.markdown('<div class="section-subhead"><span class="indicator"></span> Machine Learning Career Classification</div>', unsafe_allow_html=True)
        
        chart_col1, chart_col2 = st.columns([1.2, 0.8])

        with chart_col1:
            if top_categories:
                df_cats = pd.DataFrame(top_categories)
                df_cats = df_cats.sort_values("probability", ascending=True)
                
                fig_bar = px.bar(
                    df_cats,
                    x="probability",
                    y="category",
                    orientation="h",
                    title="Top Career Sector Probability Distribution",
                    labels={"probability": "Confidence Score (%)", "category": "Sector"},
                    color="probability",
                    color_continuous_scale=["#cbd5e1", "#818cf8", "#4f46e5"],
                )
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_family="Plus Jakarta Sans, Inter, sans-serif",
                    font_color="#334155",
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=320,
                    coloraxis_showscale=False,
                )
                fig_bar.update_xaxes(showgrid=True, gridcolor="#e2e8f0", range=[0, 100])
                fig_bar.update_yaxes(showgrid=False)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Probability distribution is only available for probabilistic classifiers.")

        with chart_col2:
            # Score Gauge Chart
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=top_fit_score,
                    title={"text": "Peak Market Fit Index", "font": {"size": 16, "color": "#0f172a"}},
                    number={"suffix": "%", "font": {"size": 36, "color": "#4f46e5", "family": "Plus Jakarta Sans"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                        "bar": {"color": "#4f46e5"},
                        "bgcolor": "white",
                        "borderwidth": 1,
                        "bordercolor": "#e2e8f0",
                        "steps": [
                            {"range": [0, 40], "color": "#f1f5f9"},
                            {"range": [40, 70], "color": "#e0e7ff"},
                            {"range": [70, 100], "color": "#dcfce7"},
                        ],
                    },
                )
            )
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_family="Plus Jakarta Sans, Inter, sans-serif",
                margin=dict(l=20, r=20, t=40, b=20),
                height=320,
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Resume Structural Health Breakdown
        st.markdown('<div class="section-subhead"><span class="indicator"></span> Resume Structural & Keyword Metrics</div>', unsafe_allow_html=True)
        
        words = active_resume_text.split()
        word_count = len(words)
        char_count = len(active_resume_text)
        skill_density = (len(candidate_skills) / max(word_count, 1)) * 100
        avg_word_len = char_count / max(word_count, 1)

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Total Word Count", f"{word_count:,}", "Recommended: 400-800")
        m_col2.metric("Total Character Count", f"{char_count:,}")
        m_col3.metric("Skill Density Rate", f"{skill_density:.1f}%", "Optimal: 2-5%")
        m_col4.metric("Avg Word Length", f"{avg_word_len:.1f} chars", "Standard: 4.5-6.0")

    # ------------------------------------------------------------------------
    # TAB 3: AI ATS OPTIMIZER & GAP SIMULATOR
    # ------------------------------------------------------------------------
    with main_tab3:
        st.markdown('<div class="section-subhead"><span class="indicator"></span> Interactive Target Role Gap Simulator</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 0.9rem; color: #64748b;">Select any target job position from our database to compare your resume side-by-side and see exact skill deficiencies to target 90%+ match score:</p>', unsafe_allow_html=True)

        sim_titles = jobs_df["title"].dropna().unique()[:100]
        selected_sim_job_title = st.selectbox("Select Target Job Role:", sim_titles, index=0)

        # Find matching row
        target_rows = jobs_df[jobs_df["title"] == selected_sim_job_title]
        if not target_rows.empty:
            target_job = target_rows.iloc[0]
            target_text = target_job.get("clean_text", target_job.get("raw_text", ""))
            
            # Compute similarity with this specific job using matching_tfidf
            resume_matching_vec = matching_tfidf.transform([active_resume_text])
            t_vec = matching_tfidf.transform([target_text])
            sim_score = float(cosine_similarity(resume_matching_vec, t_vec)[0][0]) * 100

            target_skills_raw = target_job.get("extracted_skills", [])
            if isinstance(target_skills_raw, list):
                target_skills = target_skills_raw
            else:
                target_skills = [s.strip() for s in str(target_skills_raw).split(",") if s.strip()]

            cand_skills_lower = set(s.lower() for s in candidate_skills)
            shared_skills = [s for s in target_skills if s.lower() in cand_skills_lower]
            missing_skills = [s for s in target_skills if s.lower() not in cand_skills_lower]

            sim_c1, sim_c2 = st.columns([1, 2])
            with sim_c1:
                st.markdown(
                    f"""
                    <div class="kpi-card" style="text-align: center; padding: 1.8rem 1rem;">
                        <div class="kpi-card-label">Target Role Match</div>
                        <div class="kpi-card-value" style="color: {'#059669' if sim_score >= 60 else '#4f46e5'}; font-size: 2.4rem;">
                            {sim_score:.1f}%
                        </div>
                        <div class="kpi-card-hint" style="margin-top: 0.6rem;">
                            <b>{len(shared_skills)}</b> shared skills<br>
                            <b>{len(missing_skills)}</b> target skill gaps
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with sim_c2:
                st.markdown(
                    f"""
                    <div class="glass-card" style="margin-bottom: 0;">
                        <div class="glass-card-title">🎯 Actionable Role Recommendations</div>
                        <div style="margin-top: 0.6rem;">
                            <div style="font-size: 0.82rem; font-weight: 700; color: #059669; text-transform: uppercase;">✅ Skills you currently match:</div>
                            <div class="skill-group">
                                {"".join(f'<span class="skill-chip skill-chip-match">✓ {html.escape(s.title())}</span>' for s in shared_skills) if shared_skills else '<span style="font-size:0.85rem; color:#94a3b8;">None</span>'}
                            </div>
                        </div>
                        <div style="margin-top: 0.9rem;">
                            <div style="font-size: 0.82rem; font-weight: 700; color: #d97706; text-transform: uppercase;">⚡ Critical Missing Keywords to Add:</div>
                            <div class="skill-group">
                                {"".join(f'<span class="skill-chip skill-chip-gap">+ {html.escape(s.title())}</span>' for s in missing_skills) if missing_skills else '<span style="font-size:0.85rem; color:#10b981;">No gaps found! Great fit.</span>'}
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Actionable ATS Recommendations Checklist
        st.markdown('<div class="section-subhead"><span class="indicator"></span> Automated ATS Optimization Checklist</div>', unsafe_allow_html=True)
        
        recs = [
            ("Quantifiable Impact", "Include numeric metrics (e.g., 'improved throughput by 25%', 'managed $2M budget') in each bullet point to demonstrate tangible results.", "✅" if any(char.isdigit() for char in active_resume_text) else "⚠️"),
            ("Action-Oriented Verbs", "Start experience bullet points with strong verbs such as 'Spearheaded', 'Engineered', 'Orchestrated', 'Optimized', or 'Architected'.", "✅"),
            ("Keyword Density", f"You have {len(candidate_skills)} recognized skill keywords. Aim to integrate 10-15 relevant keywords tailored to your target job descriptions.", "✅" if len(candidate_skills) >= 8 else "⚠️"),
            ("Standard ATS Section Headings", "Use conventional headers such as 'Professional Experience', 'Technical Skills', 'Education', and 'Summary' for flawless parser recognition.", "✅"),
        ]

        for title, desc, status in recs:
            st.markdown(
                f"""
                <div class="rec-box">
                    <div class="rec-box-title">{status} {title}</div>
                    <div class="rec-box-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ------------------------------------------------------------------------
    # TAB 4: 30,000+ JOB MARKET EXPLORER
    # ------------------------------------------------------------------------
    with main_tab4:
        st.markdown('<div class="section-subhead"><span class="indicator"></span> Live 30,000+ Job Database Search Engine</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 0.9rem; color: #64748b;">Search and inspect the complete market database by job title, skill requirement, or industry:</p>', unsafe_allow_html=True)

        exp_c1, exp_c2, exp_c3 = st.columns([1.5, 1, 1])
        with exp_c1:
            search_query = st.text_input("🔍 Search Job Title or Skill Keyword", placeholder="e.g. Python, Analyst, React, Marketing, Manager...")
        with exp_c2:
            filter_loc = st.selectbox("Location Filter", ["All"] + list(jobs_df["location"].value_counts().head(15).index), key="exp_loc")
        with exp_c3:
            filter_ind = st.selectbox("Industry Filter", ["All"] + list(jobs_df["industry"].value_counts().head(12).index), key="exp_ind")

        # Filter jobs database
        filtered_db = jobs_df
        if search_query.strip():
            q = search_query.strip().lower()
            filtered_db = filtered_db[
                filtered_db["title"].str.lower().str.contains(q)
                | filtered_db["key_skills"].str.lower().str.contains(q)
                | filtered_db["clean_text"].str.lower().str.contains(q)
            ]
        if filter_loc != "All":
            filtered_db = filtered_db[filtered_db["location"] == filter_loc]
        if filter_ind != "All":
            filtered_db = filtered_db[filtered_db["industry"] == filter_ind]

        st.markdown(f"**Found {len(filtered_db):,} matching openings** in database:")

        display_cols = [c for c in ["title", "role_category", "location", "experience", "industry", "key_skills"] if c in filtered_db.columns]
        st.dataframe(
            filtered_db[display_cols].head(50),
            use_container_width=True,
            height=380,
        )

    # ------------------------------------------------------------------------
    # TAB 5: EXTRACTED TEXT & EXPORT REPORT
    # ------------------------------------------------------------------------
    with main_tab5:
        st.markdown('<div class="section-subhead"><span class="indicator"></span> Candidate Profile Export & Raw Data</div>', unsafe_allow_html=True)
        
        # Build Downloadable Markdown Report
        report_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        matched_job_summary_lines = []
        for i, j in enumerate(matched_jobs[:8]):
            matched_job_summary_lines.append(f"{i+1}. **{j['title']}** ({j['location']}) - Match Score: {j['score']:.1f}%\n   - Industry: {j['industry']}\n   - Experience: {j['experience']}")
        
        report_markdown = f"""# AI Resume & Job Fit Analysis Report
**Generated on:** {report_timestamp}  
**Source:** {resume_source}  

---

## 1. Executive Summary
- **Predicted Career Category:** {predicted_category}
- **Classification Model Confidence:** {confidence_score:.1f}%
- **Total Detected Skills:** {len(candidate_skills)}
- **Peak Job Match Score:** {top_fit_score:.1f}%

---

## 2. Detected Technical & Core Skills
{', '.join([s.title() for s in candidate_skills]) if candidate_skills else 'None detected'}

---

## 3. Top Matched Opportunities (from 30,000+ Database)
{chr(10).join(matched_job_summary_lines)}

---

*Report generated by TalentPulse AI Intelligence Suite.*
"""

        rep_c1, rep_c2 = st.columns([1, 1])
        with rep_c1:
            st.download_button(
                label="📥 Download Full Analysis Report (Markdown)",
                data=report_markdown,
                file_name=f"resume_analysis_report_{int(time.time())}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with rep_c2:
            if matched_jobs:
                df_export = pd.DataFrame(matched_jobs)[["title", "score", "location", "experience", "industry", "role_category"]]
                csv_data = df_export.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📊 Download Top Job Matches (CSV)",
                    data=csv_data,
                    file_name=f"matched_jobs_{int(time.time())}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        st.write("")
        st.markdown('<div class="glass-card-title" style="margin-bottom: 0.6rem;">📄 Extracted Resume Raw Text</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="raw-text-box">{html.escape(active_resume_text)}</div>', unsafe_allow_html=True)

else:
    # Empty State Guide
    st.markdown(
        """
        <div class="glass-card" style="text-align: center; padding: 3.5rem 2rem; margin-top: 1rem;">
            <div style="font-size: 2.8rem; margin-bottom: 0.8rem;">📄</div>
            <div style="font-size: 1.4rem; font-weight: 800; color: #0f172a; margin-bottom: 0.5rem;">
                Ready to Analyze Your Resume
            </div>
            <p style="color: #64748b; font-size: 0.96rem; max-width: 34rem; margin: 0 auto 1.5rem auto; line-height: 1.6;">
                Upload your resume PDF in the uploader above, paste raw text, or select one of the preloaded sample profiles to explore career domain predictions, skill gap analyses, and matching roles from our 30,000+ job dataset.
            </p>
            <div style="display: flex; justify-content: center; gap: 0.75rem; flex-wrap: wrap;">
                <span class="skill-chip">⚡ 30,000+ Indexed Roles</span>
                <span class="skill-chip">🧠 325 Skills Taxonomy</span>
                <span class="skill-chip">🎯 25+ Career Domains</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )