"""
TalentPulse AI - Automated Validation & Test Suite
File: test_system.py
Description: Validates PDF parser, text preprocessing, skill extraction,
             TF-IDF recommender, and multi-factor scoring against multiple resumes.
"""

import os
import sys
import pandas as pd

# Add current directory to path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

from utils.pdf_parser import extract_text_from_pdf, parse_resume_metadata
from utils.preprocessing import clean_text, extract_candidate_skills, detect_candidate_domain
from utils.recommender import load_ml_assets, recommend_jobs

def run_tests():
    print("=" * 80)
    print("TALENTPULSE AI - SYSTEM VALIDATION SUITE")
    print("=" * 80)

    # 1. Test ML Assets Loading
    print("\n[Test 1] Testing ML Asset Loading:")
    df_jobs, vectorizer, job_vectors, errors = load_ml_assets(base_dir)
    assert not errors, f"Failed to load ML assets: {errors}"
    assert df_jobs is not None
    assert vectorizer is not None
    assert job_vectors is not None
    print(f"  - Jobs loaded: {len(df_jobs)}")
    print(f"  - Vectorizer vocab: {len(vectorizer.vocabulary_)} features")
    print(f"  - Job vectors shape: {job_vectors.shape}")

    # Check required columns
    required_cols = [
        "job_id", "job_title", "company", "location", "job_description",
        "required_skills", "preferred_skills", "education", "experience",
        "employment_type", "category", "industry", "source", "source_url", "date_collected"
    ]
    for col in required_cols:
        assert col in df_jobs.columns, f"Missing required column: {col}"
    print(f"  - All {len(required_cols)} required schema columns verified!")

    # 2. Test Resume 1: Engineering / CS & AI/ML
    print("\n[Test 2] Testing Engineering / CS & AI Resume:")
    cs_resume = """
    Rahul Verma | B.Tech in Computer Science and Engineering
    Email: rahul.verma@example.com | Phone: +91 98765 43210
    
    EDUCATION:
    Bachelor of Technology in Computer Science & Engineering (2022 - 2026)
    Relevant Coursework: Data Structures, Algorithms, DBMS, Operating Systems, Machine Learning
    
    SKILLS:
    Programming: Python, SQL, C++, Java, JavaScript
    Frameworks: PyTorch, TensorFlow, Scikit-Learn, Pandas, NumPy, FastAPI, Docker
    Cloud & Tools: Git, Linux, AWS, Docker, Kubernetes
    Concepts: Machine Learning, Deep Learning, Natural Language Processing, REST APIs
    
    PROJECTS:
    1. TalentPulse AI: Resume Based Job Recommendation System using TF-IDF and Cosine Similarity.
    2. Deep Learning Vision Classifier: PyTorch CNN model deployed with FastAPI on AWS.
    
    EXPERIENCE:
    Machine Learning Intern (Summer 2025) - Built data pipelines and model inference endpoints.
    """

    meta_cs = parse_resume_metadata(cs_resume)
    skills_cs = extract_candidate_skills(cs_resume)
    domain_cs = detect_candidate_domain(cs_resume, skills_cs)
    print(f"  - Detected Education : {meta_cs['education']}")
    print(f"  - Detected Experience: {meta_cs['experience']}")
    print(f"  - Detected Domain    : {domain_cs}")
    print(f"  - Detected Skills ({len(skills_cs)}): {', '.join(skills_cs[:8])}...")
    assert "Python" in skills_cs
    assert "Machine Learning" in skills_cs
    assert domain_cs == "cs_it_data"

    recs_cs, summary_cs = recommend_jobs(
        raw_resume_text=cs_resume,
        df_jobs=df_jobs,
        vectorizer=vectorizer,
        job_vectors=job_vectors,
        top_n=5
    )
    assert len(recs_cs) == 5, f"Expected 5 recommendations, got {len(recs_cs)}"
    print(f"  - Top Recommendation: {recs_cs[0]['job_title']} at {recs_cs[0]['company']}")
    print(f"    Category: {recs_cs[0]['category']}")
    print(f"    Match Score: {recs_cs[0]['match_score']}%")
    print(f"    Matched Skills: {recs_cs[0]['explanation']['matched_skills_text']}")
    print(f"    Source Link: {recs_cs[0]['source_url']}")
    # Ensure top job is an engineering / CS role
    assert any(tech in recs_cs[0]['category'] for tech in ["Computer", "Data", "Artificial", "Web", "Cloud", "Software", "Information"])

    # 3. Test Resume 2: Mechanical Engineering
    print("\n[Test 3] Testing Mechanical Engineering Resume:")
    mech_resume = """
    Aditya Rao | B.Tech Mechanical Engineering
    Email: aditya.rao@example.com
    
    EDUCATION:
    Bachelor of Technology in Mechanical Engineering
    Coursework: Thermodynamics, Fluid Mechanics, Machine Design, CAD/CAM
    
    TECHNICAL SKILLS:
    CAD / CAE Tools: AutoCAD, SolidWorks, CATIA, ANSYS
    Analysis: Finite Element Analysis (FEA), Computational Fluid Dynamics (CFD), MATLAB
    Core Competencies: GD&T, CNC Machining, Thermal Systems, Thermodynamics
    
    PROJECTS:
    Structural and Thermal FEA of Automotive Brake Disc using ANSYS and SolidWorks.
    Design and fabrication of Formula Student race car chassis.
    
    EXPERIENCE:
    Mechanical Engineering Intern at Automotive R&D Center (6 months).
    """

    skills_mech = extract_candidate_skills(mech_resume)
    domain_mech = detect_candidate_domain(mech_resume, skills_mech)
    print(f"  - Detected Domain: {domain_mech}")
    print(f"  - Detected Skills: {', '.join(skills_mech)}")
    assert domain_mech == "mechanical"
    assert "SolidWorks" in skills_mech or "AutoCAD" in skills_mech

    recs_mech, _ = recommend_jobs(
        raw_resume_text=mech_resume,
        df_jobs=df_jobs,
        vectorizer=vectorizer,
        job_vectors=job_vectors,
        top_n=5
    )
    print(f"  - Top Recommendation: {recs_mech[0]['job_title']} at {recs_mech[0]['company']}")
    print(f"    Category: {recs_mech[0]['category']}")
    print(f"    Match Score: {recs_mech[0]['match_score']}%")
    assert "Mechanical" in recs_mech[0]['category'] or "Automation" in recs_mech[0]['category'] or "Engineering" in recs_mech[0]['category']

    # 4. Test Resume 3: Non-Engineering (Marketing & HR)
    print("\n[Test 4] Testing Non-Engineering (Marketing / HR) Resume:")
    marketing_resume = """
    Priya Sharma | MBA Marketing & Human Resources
    Email: priya.sharma@example.com
    
    EDUCATION:
    Master of Business Administration (MBA) in Marketing & HR Management
    Bachelor of Commerce (B.Com)
    
    SKILLS:
    Digital Marketing, Search Engine Optimization (SEO), Social Media Marketing
    Recruitment, Talent Acquisition, Human Resources Management, Employee Relations
    Market Research, Content Strategy, Google Analytics, Microsoft Excel, CRM
    
    EXPERIENCE:
    Digital Marketing & HR Specialist (2 years)
    Coordinated talent acquisition campaigns, campus drives, and social media brand positioning.
    """

    skills_mkt = extract_candidate_skills(marketing_resume)
    domain_mkt = detect_candidate_domain(marketing_resume, skills_mkt)
    print(f"  - Detected Domain: {domain_mkt}")
    print(f"  - Detected Skills: {', '.join(skills_mkt)}")
    assert domain_mkt == "non_engineering"

    recs_mkt, _ = recommend_jobs(
        raw_resume_text=marketing_resume,
        df_jobs=df_jobs,
        vectorizer=vectorizer,
        job_vectors=job_vectors,
        top_n=5
    )
    print(f"  - Top Recommendation: {recs_mkt[0]['job_title']} at {recs_mkt[0]['company']}")
    print(f"    Category: {recs_mkt[0]['category']}")
    print(f"    Match Score: {recs_mkt[0]['match_score']}%")
    assert recs_mkt[0]['category'] in ["Marketing", "HR", "Business", "Operations", "Finance"]

    # 5. Verify that recommendations differ between resumes
    print("\n[Test 5] Verifying Distinct Recommendations for Different Resumes:")
    top_cs_title = recs_cs[0]['job_title']
    top_mech_title = recs_mech[0]['job_title']
    top_mkt_title = recs_mkt[0]['job_title']
    print(f"  - CS Top Job   : {top_cs_title}")
    print(f"  - Mech Top Job : {top_mech_title}")
    print(f"  - Mkt Top Job  : {top_mkt_title}")
    assert top_cs_title != top_mech_title
    assert top_cs_title != top_mkt_title
    assert top_mech_title != top_mkt_title
    print("  -> PASS: Recommendations dynamically and accurately change based on resume content!")

    # 6. Test Error Handling for Invalid / Empty PDFs
    print("\n[Test 6] Testing Edge Cases and Error Handling:")
    _, err_empty = extract_text_from_pdf(b"")
    print(f"  - Empty bytes error: {err_empty}")
    assert err_empty is not None and "empty" in err_empty.lower()

    _, err_corrupt = extract_text_from_pdf(b"%PDF-corrupted-bytes-xyz")
    print(f"  - Corrupt bytes error: {err_corrupt}")
    assert err_corrupt is not None

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED SUCCESSFULLY! THE SYSTEM IS COMPLETELY FUNCTIONAL.")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()
