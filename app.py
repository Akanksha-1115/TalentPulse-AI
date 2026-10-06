"""
TalentPulse AI – Resume Based Job Recommendation System
File: app.py
Description: Streamlit ML Web Application for resume parsing, technical skill extraction,
             TF-IDF vector space modeling, cosine similarity, and multi-factor job recommendation.
Academic Project: Final Year BTech Computer Science & Engineering (CSE) Mini-Project
"""

import os
import sys
import re
import pandas as pd
import numpy as np
import streamlit as st
import joblib

# ----------------------------------------------------------------------
# 1. DYNAMIC RELATIVE PATH CONFIGURATION
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.pdf_parser import extract_text_from_pdf, parse_resume_metadata
from utils.preprocessing import clean_text, extract_candidate_skills, detect_candidate_domain
from utils.recommender import load_ml_assets, recommend_jobs, PRIMARY_CATEGORIES

# ----------------------------------------------------------------------
# 2. STREAMLIT PAGE CONFIGURATION
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="TalentPulse AI - Resume Job Recommendation System",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# 3. PROFESSIONAL, CLEAN, STUDENT-BUILT BTECH UI STYLES
# ----------------------------------------------------------------------
st.markdown("""
<style>
    /* Clean base typography */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #1f2937;
    }

    /* Top Header Container */
    .tp-header {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 20px 24px;
        margin-bottom: 22px;
        border-top: 4px solid #2563eb;
    }
    .tp-title {
        font-size: 26px;
        font-weight: 700;
        color: #1e3a8a;
        margin: 0 0 6px 0;
        letter-spacing: -0.3px;
    }
    .tp-subtitle {
        font-size: 15px;
        color: #4b5563;
        margin: 0 0 10px 0;
    }
    .tp-badge-banner {
        display: inline-block;
        background-color: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        border-radius: 4px;
        padding: 4px 10px;
        font-size: 12px;
        font-weight: 600;
    }

    /* Section Cards */
    .tp-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 18px 22px;
        margin-bottom: 18px;
    }
    .tp-card-header {
        font-size: 18px;
        font-weight: 600;
        color: #111827;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #f3f4f6;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* Skill badges */
    .skill-badge {
        display: inline-block;
        background-color: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
        border-radius: 4px;
        padding: 3px 8px;
        margin: 3px 4px 3px 0;
        font-size: 12px;
        font-weight: 500;
    }
    .missing-skill-badge {
        display: inline-block;
        background-color: #fefce8;
        color: #854d0e;
        border: 1px solid #fef08a;
        border-radius: 4px;
        padding: 3px 8px;
        margin: 3px 4px 3px 0;
        font-size: 12px;
        font-weight: 500;
    }
    .general-badge {
        display: inline-block;
        background-color: #f3f4f6;
        color: #374151;
        border: 1px solid #e5e7eb;
        border-radius: 4px;
        padding: 2px 7px;
        margin: 2px 4px 2px 0;
        font-size: 11px;
    }

    /* Job Recommendation Card */
    .job-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 18px 20px;
        margin-bottom: 16px;
        transition: border-color 0.15s ease-in-out;
    }
    .job-card:hover {
        border-color: #93c5fd;
    }
    .job-title-text {
        font-size: 17px;
        font-weight: 600;
        color: #1e3a8a;
        margin-bottom: 4px;
    }
    .job-meta-row {
        font-size: 13px;
        color: #4b5563;
        margin-bottom: 10px;
    }
    .match-pill {
        display: inline-block;
        font-size: 14px;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 4px;
    }
    .match-high {
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
    }
    .match-med {
        background-color: #fef9c3;
        color: #a16207;
        border: 1px solid #fde047;
    }
    .match-low {
        background-color: #f3f4f6;
        color: #4b5563;
        border: 1px solid #e5e7eb;
    }

    /* Explanation section */
    .explanation-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 3px solid #3b82f6;
        border-radius: 4px;
        padding: 10px 14px;
        margin-top: 10px;
        font-size: 13px;
        color: #334155;
    }

    /* Footer */
    .tp-footer {
        text-align: center;
        color: #6b7280;
        font-size: 12px;
        margin-top: 40px;
        padding-top: 16px;
        border-top: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# 4. LOAD CACHED ML ARTIFACTS
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading verified job dataset & ML models...")
def get_ml_system():
    return load_ml_assets(BASE_DIR)

df_jobs, vectorizer, job_vectors, load_errors = get_ml_system()

# ----------------------------------------------------------------------
# 5. SIDEBAR: FILTERS, SETTINGS & PRESET SAMPLES
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 💼 TalentPulse AI")
    st.caption("College BTech CSE Mini-Project | TF-IDF & Cosine Similarity Job Recommendation Engine")
    st.markdown("---")

    st.markdown("#### 🔍 Job Search Filters")

    # Categories list
    all_categories = ["All"]
    if df_jobs is not None:
        cats = sorted(df_jobs["category"].dropna().unique().tolist())
        all_categories.extend(cats)
    
    selected_category = st.selectbox("Job Category:", all_categories, index=0)

    # Locations
    all_locations = ["All", "Bengaluru", "Hyderabad", "Pune", "Chennai", "Noida", "Mumbai", "Remote", "USA"]
    selected_location = st.selectbox("Location:", all_locations, index=0)

    # Experience level
    all_experience = ["All", "Internship", "Fresher", "Entry Level", "Junior", "Associate"]
    selected_experience = st.selectbox("Experience Level:", all_experience, index=0)

    # Employment Type
    selected_emp_type = st.selectbox("Employment Type:", ["All", "Full-time", "Internship"], index=0)

    st.markdown("---")
    st.markdown("#### ⚙️ Recommendation Settings")
    top_n_val = st.slider("Number of Recommendations:", min_value=3, max_value=15, value=5, step=1)
    min_score_val = st.slider("Minimum Match Score (%):", min_value=0, max_value=85, value=25, step=5)

    st.markdown("---")
    st.markdown("#### 📄 Quick Test Resumes")
    sample_selection = st.selectbox(
        "Load Pre-built Resume:",
        [
            "Upload My Own Resume",
            "Sample 1: BTech CSE / AI & ML Student",
            "Sample 2: BTech Mechanical Engineering",
            "Sample 3: Non-Engineering (MBA Marketing & HR)"
        ]
    )

    st.markdown("---")
    with st.expander("ℹ️ About Dataset & Provenance"):
        st.write(
            "**Data Source:** Official O*NET 31.0 Database (U.S. Department of Labor, CC BY 4.0) "
            "and documented public engineering benchmarks.\n\n"
            "**Distribution:** ~66% Engineering & Technology across 17 categories, "
            "~34% Allied corporate and secondary domains."
        )

# Sample resumes text dictionary for quick testing
SAMPLE_RESUME_TEXTS = {
    "Sample 1: BTech CSE / AI & ML Student": (
        "RAHUL VERMA\n"
        "Email: rahul.verma@example.com | Phone: +91 98765 43210 | Bengaluru, India\n"
        "GitHub: github.com/rahulverma | LinkedIn: linkedin.com/in/rahulverma\n\n"
        "EDUCATION:\n"
        "Bachelor of Technology in Computer Science & Engineering (2022 - 2026)\n"
        "Vellore Institute of Technology | CGPA: 8.9 / 10.0\n"
        "Relevant Coursework: Data Structures & Algorithms, Database Management Systems, "
        "Machine Learning, Deep Learning, Operating Systems, Computer Networks.\n\n"
        "TECHNICAL SKILLS:\n"
        "Programming Languages: Python, SQL, C++, Java, JavaScript\n"
        "Frameworks & Libraries: PyTorch, TensorFlow, Scikit-Learn, Pandas, NumPy, FastAPI, Flask\n"
        "Cloud & DevOps: Git, Docker, Kubernetes, Linux, AWS (EC2, S3)\n"
        "Core Concepts: Machine Learning, NLP, Computer Vision, REST APIs, Microservices\n\n"
        "PROJECTS:\n"
        "1. TalentPulse AI: Resume Based Job Recommendation System using TF-IDF and Cosine Similarity.\n"
        "2. Vision-Based Defect Detection System (PyTorch, OpenCV, FastAPI) deployed on AWS EC2.\n\n"
        "EXPERIENCE:\n"
        "Machine Learning Intern (Summer 2025) - Built automated feature pipelines and model inference endpoints."
    ),
    "Sample 2: BTech Mechanical Engineering": (
        "ADITYA RAO\n"
        "Email: aditya.rao@example.com | Phone: +91 91234 56789 | Pune, India\n"
        "EDUCATION:\n"
        "Bachelor of Technology in Mechanical Engineering (2021 - 2025)\n"
        "Coursework: Thermodynamics, Fluid Mechanics, Strength of Materials, Machine Design, CAD/CAM.\n\n"
        "TECHNICAL SKILLS:\n"
        "CAD / 3D Modeling: AutoCAD, SolidWorks, CATIA, Creo\n"
        "Engineering Simulation: ANSYS (Structural & Thermal FEA), CFD, MATLAB, Simulink\n"
        "Manufacturing & Design: GD&T, CNC Machining, Mechatronics, PLC Programming, HVAC\n\n"
        "PROJECTS:\n"
        "1. Finite Element Analysis of Formula Student Race Car Chassis (ANSYS, SolidWorks).\n"
        "2. Design and Thermal Optimization of Automotive Heat Exchanger in CFD.\n\n"
        "EXPERIENCE:\n"
        "Mechanical Design Intern at Robert Bosch Engineering (6 months) - 3D CAD modeling and tolerance stack-up."
    ),
    "Sample 3: Non-Engineering (MBA Marketing & HR)": (
        "PRIYA SHARMA\n"
        "Email: priya.sharma@example.com | Phone: +91 99887 76655 | Mumbai, India\n\n"
        "EDUCATION:\n"
        "Master of Business Administration (MBA) in Marketing & HR (2023 - 2025)\n"
        "Bachelor of Commerce (B.Com) - Honors\n\n"
        "PROFESSIONAL SKILLS:\n"
        "Marketing: Digital Marketing, Search Engine Optimization (SEO), Social Media Campaigns, Market Research\n"
        "Human Resources: Talent Acquisition, Campus Recruitment, Employee Onboarding, HR Operations\n"
        "Tools & Analytics: Microsoft Excel (VLOOKUP, Pivot Tables), Google Analytics, HubSpot CRM, PowerPoint\n\n"
        "EXPERIENCE:\n"
        "HR & Marketing Associate (2024 - Present) - Managed campus recruitment drives and inbound lead generation."
    )
}

# ----------------------------------------------------------------------
# 6. MAIN APPLICATION HEADER
# ----------------------------------------------------------------------
st.markdown("""
<div class="tp-header">
    <div class="tp-title">TalentPulse AI – Resume Based Job Recommendation System</div>
    <div class="tp-subtitle">
        An intelligent machine learning system matching candidate resumes with verified technical and engineering
        job profiles using TF-IDF vector space modeling, cosine similarity, and skill overlap scoring.
    </div>
    <span class="tp-badge-banner">BTech CSE Mini-Project • Real O*NET 31.0 Database • Content-Based ML Engine</span>
</div>
""", unsafe_allow_html=True)

# Stop if models or dataset failed to load
if load_errors:
    st.error("⚠️ Critical System Error: Required ML artifacts could not be loaded:")
    for err in load_errors:
        st.write(f"- {err}")
    st.info("Tip: Run `python prepare_dataset.py` from the project directory to rebuild all datasets and models.")
    st.stop()

# ----------------------------------------------------------------------
# 7. SECTION 1: UPLOAD RESUME
# ----------------------------------------------------------------------
st.markdown('<div class="tp-card">', unsafe_allow_html=True)
st.markdown('<div class="tp-card-header"><span>Step 1: Resume Submission</span><span>PDF or TXT</span></div>', unsafe_allow_html=True)

resume_text = ""
resume_source_name = ""

col_upload, col_preview = st.columns([1, 1], gap="medium")

with col_upload:
    if sample_selection != "Upload My Own Resume":
        st.info(f"Loaded: **{sample_selection}**")
        resume_text = SAMPLE_RESUME_TEXTS[sample_selection]
        resume_source_name = sample_selection
    else:
        uploaded_file = st.file_uploader(
            "Upload candidate resume (PDF or TXT format):",
            type=["pdf", "txt"],
            help="Upload your digital resume PDF or plain text resume file."
        )

        if uploaded_file is not None:
            resume_source_name = uploaded_file.name
            file_bytes = uploaded_file.read()

            if uploaded_file.name.lower().endswith(".pdf"):
                extracted_txt, parse_err = extract_text_from_pdf(file_bytes)
                if parse_err:
                    st.error(f"❌ Resume Extraction Error: {parse_err}")
                else:
                    resume_text = extracted_txt
                    st.success(f"✓ PDF extracted successfully ({len(resume_text.split())} words detected)")
            else:
                try:
                    resume_text = file_bytes.decode("utf-8")
                    st.success(f"✓ TXT file loaded successfully ({len(resume_text.split())} words)")
                except Exception as e:
                    st.error(f"❌ Text File Decoding Error: {str(e)}")

with col_preview:
    if resume_text:
        st.markdown("**Extracted Resume Text Preview:**")
        preview_snip = resume_text[:600] + ("..." if len(resume_text) > 600 else "")
        st.text_area("Resume Content", value=preview_snip, height=130, disabled=True, label_visibility="collapsed")
    else:
        st.markdown("**Awaiting Resume Input:**")
        st.caption("Upload a resume PDF on the left or select a sample resume from the sidebar to begin recommendation.")

st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# 8. SECTION 2 & 3: RESUME SUMMARY & DETECTED SKILLS
# ----------------------------------------------------------------------
if resume_text:
    metadata = parse_resume_metadata(resume_text)
    detected_skills = extract_candidate_skills(resume_text)
    detected_domain = detect_candidate_domain(resume_text, detected_skills)

    domain_display_names = {
        "cs_it_data": "Computer Science / Information Technology / AI & Data",
        "mechanical": "Mechanical Engineering & Automation",
        "civil": "Civil & Structural Engineering",
        "electrical_ece": "Electronics / Electrical / Embedded Systems",
        "non_engineering": "Allied Non-Engineering (Business / Management / Arts)"
    }

    col_sum, col_sk = st.columns([1, 1], gap="medium")

    with col_sum:
        st.markdown('<div class="tp-card">', unsafe_allow_html=True)
        st.markdown('<div class="tp-card-header"><span>Step 2: Resume Summary</span><span>Extracted Details</span></div>', unsafe_allow_html=True)
        
        st.markdown(f"**Identified Domain:** `{domain_display_names.get(detected_domain, detected_domain)}`")
        st.markdown(f"**Detected Education:** {metadata['education']}")
        st.markdown(f"**Detected Experience:** {metadata['experience']}")
        
        c_m1, c_m2, c_m3 = st.columns(3)
        c_m1.metric("Word Count", metadata["word_count"])
        c_m2.metric("Character Count", metadata["char_count"])
        c_m3.metric("Skills Found", len(detected_skills))
        
        st.markdown('</div>', unsafe_allow_html=True)

    with col_sk:
        st.markdown('<div class="tp-card">', unsafe_allow_html=True)
        st.markdown('<div class="tp-card-header"><span>Step 3: Detected Skills & Keywords</span><span>Ontology Match</span></div>', unsafe_allow_html=True)
        
        if detected_skills:
            badges_html = "".join([f'<span class="skill-badge">{s}</span>' for s in detected_skills])
            st.markdown(f'<div style="max-height: 145px; overflow-y: auto; padding: 4px 0;">{badges_html}</div>', unsafe_allow_html=True)
        else:
            st.caption("No standardized technical skills detected. The system will rely primarily on TF-IDF textual similarity.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 9. SECTION 4 & 5: RECOMMENDED JOBS & MATCH EXPLANATIONS
    # ------------------------------------------------------------------
    recommendations, profile_summary = recommend_jobs(
        raw_resume_text=resume_text,
        df_jobs=df_jobs,
        vectorizer=vectorizer,
        job_vectors=job_vectors,
        top_n=top_n_val,
        filter_category=selected_category,
        filter_location=selected_location,
        filter_experience=selected_experience,
        filter_employment_type=selected_emp_type,
        min_match_score=min_score_val
    )

    st.markdown('<div class="tp-card">', unsafe_allow_html=True)
    header_right_info = f"Showing Top {len(recommendations)} of {profile_summary.get('total_jobs_matching_filter', 0)} Matched Roles"
    st.markdown(f'<div class="tp-card-header"><span>Step 4: Recommended Jobs & Ranking</span><span>{header_right_info}</span></div>', unsafe_allow_html=True)

    if not recommendations:
        st.warning(
            "⚠️ No jobs found matching the selected filters and minimum score threshold. "
            "Try lowering the 'Minimum Match Score' slider or resetting filters in the sidebar."
        )
    else:
        for idx, job in enumerate(recommendations, 1):
            score = job["match_score"]
            pill_class = "match-high" if score >= 70 else ("match-med" if score >= 45 else "match-low")

            st.markdown(f"""
            <div class="job-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div class="job-title-text">#{idx}. {job['job_title']}</div>
                        <div class="job-meta-row">
                            <strong>🏢 Company:</strong> {job['company']} &nbsp;|&nbsp; 
                            <strong>📍 Location:</strong> {job['location']} &nbsp;|&nbsp; 
                            <strong>🏷️ Category:</strong> {job['category']} &nbsp;|&nbsp; 
                            <strong>⏱️ Type:</strong> {job['employment_type']}
                        </div>
                    </div>
                    <div>
                        <span class="match-pill {pill_class}">{score}% Match</span>
                    </div>
                </div>
                <div style="font-size: 13px; color: #374151; margin-bottom: 8px;">
                    <strong>Experience Level:</strong> {job['experience']} &nbsp;|&nbsp;
                    <strong>Industry:</strong> {job['industry']} &nbsp;|&nbsp;
                    <strong>Required Education:</strong> {job['education']}
                </div>
                <div style="font-size: 13px; color: #1f2937; margin-bottom: 10px;">
                    <strong>Description:</strong> {job['job_description'][:220]}...
                </div>
            """, unsafe_allow_html=True)

            # Match explanation section
            expl = job["explanation"]
            st.markdown(f"""
                <div class="explanation-box">
                    <strong>💡 Why this job matches:</strong><br>
                    • <strong>Matched Skills:</strong> {expl['matched_skills_text']}<br>
                    • <strong>Domain / Category Fit:</strong> {expl['domain_fit']} alignment with <em>{job['category']}</em><br>
                    • <strong>Score Breakdown:</strong> TF-IDF Similarity: {job['tfidf_score']}% (70% weight) | Skill Overlap: {job['skill_score']}% (20% weight) | Domain Relevance: {job['category_score']}% (10% weight)
                </div>
            """, unsafe_allow_html=True)

            # Missing Skills recommendations
            if job["missing_skills"]:
                missing_badges = "".join([f'<span class="missing-skill-badge">{s}</span>' for s in job["missing_skills"][:5]])
                st.markdown(f"""
                    <div style="margin-top: 8px; font-size: 12px; color: #6b7280;">
                        <strong>Recommended Skills to Learn for this Role:</strong><br>{missing_badges}
                    </div>
                """, unsafe_allow_html=True)

            # Action link: View Job
            st.markdown("<div style='margin-top: 10px;'>", unsafe_allow_html=True)
            if job["source_url"] and str(job["source_url"]).startswith("http"):
                st.markdown(
                    f"<a href='{job['source_url']}' target='_blank' style='text-decoration: none;'>"
                    f"<button style='background-color: #2563eb; color: #ffffff; border: none; border-radius: 4px; padding: 6px 14px; font-size: 13px; cursor: pointer; font-weight: 500;'>"
                    f"🔗 View Job (Official O*NET Standard)"
                    f"</button></a> "
                    f"<span style='font-size: 12px; color: #6b7280; margin-left: 10px;'>Source: {job['source']}</span>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(f"<span style='font-size: 12px; color: #6b7280;'>Source: {job['source']} (Verified standard profile)</span>", unsafe_allow_html=True)

            st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

else:
    # Landing state information
    st.info("👋 Welcome! Please upload your resume PDF on the left or select a sample resume from the sidebar to run the ML recommendation pipeline.")

# ----------------------------------------------------------------------
# 10. FOOTER: BTECH PROJECT DETAILS
# ----------------------------------------------------------------------
st.markdown("""
<div class="tp-footer">
    <strong>TalentPulse AI</strong> – Resume Based Job Recommendation System<br>
    Final Year B.Tech Computer Science & Engineering Mini-Project • Built with Streamlit, Scikit-Learn & PyMuPDF<br>
    Verified Dataset: O*NET 31.0 Database (U.S. Department of Labor) • MIT & CC BY 4.0 Open Licensing
</div>
""", unsafe_allow_html=True)
