"""
TalentPulse AI - Machine Learning Recommender Engine
File: utils/recommender.py
Description: End-to-end recommendation engine combining TF-IDF Cosine Similarity (70%),
             Skill Overlap (20%), and Category Relevance (10%) to rank verified jobs.
"""

import os
import re
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import clean_text, extract_candidate_skills, detect_candidate_domain

# Primary categories (Engineering & Technology)
PRIMARY_CATEGORIES = {
    "Computer Science / Software Engineering",
    "Information Technology",
    "Data Science",
    "Artificial Intelligence / Machine Learning",
    "Data Analyst",
    "Cybersecurity",
    "Cloud Computing / DevOps",
    "Web Development",
    "Mobile Development",
    "Database / Backend Development",
    "Electronics / ECE",
    "Electrical Engineering",
    "Mechanical Engineering",
    "Civil Engineering",
    "Automation / Robotics",
    "Embedded Systems",
    "Other Engineering"
}

CS_TECH_CATEGORIES = {
    "Computer Science / Software Engineering",
    "Information Technology",
    "Data Science",
    "Artificial Intelligence / Machine Learning",
    "Data Analyst",
    "Cybersecurity",
    "Cloud Computing / DevOps",
    "Web Development",
    "Mobile Development",
    "Database / Backend Development"
}


def load_ml_assets(base_dir: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], Any, Any, List[str]]:
    """
    Loads clean_jobs.csv, tfidf_vectorizer.joblib, and job_vectors.joblib.
    Uses relative path resolution from base_dir or project root.
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    jobs_csv_path = os.path.join(base_dir, "data", "processed", "clean_jobs.csv")
    vec_path = os.path.join(base_dir, "models", "tfidf_vectorizer.joblib")
    mat_path = os.path.join(base_dir, "models", "job_vectors.joblib")

    # Fallback to alternate locations if needed
    if not os.path.exists(jobs_csv_path):
        alt_csv = os.path.join(base_dir, "data", "clean_jobs.csv")
        if os.path.exists(alt_csv):
            jobs_csv_path = alt_csv

    if not os.path.exists(vec_path):
        alt_vec = os.path.join(base_dir, "model", "tfidf_vectorizer.joblib")
        if os.path.exists(alt_vec):
            vec_path = alt_vec

    if not os.path.exists(mat_path):
        alt_mat = os.path.join(base_dir, "model", "job_vectors.joblib")
        if os.path.exists(alt_mat):
            mat_path = alt_mat

    errors = []
    if not os.path.exists(jobs_csv_path):
        errors.append(f"Processed dataset not found: '{jobs_csv_path}'. Please run 'python prepare_dataset.py'.")
    if not os.path.exists(vec_path):
        errors.append(f"TF-IDF vectorizer not found: '{vec_path}'. Please run 'python prepare_dataset.py'.")
    if not os.path.exists(mat_path):
        errors.append(f"Job vectors matrix not found: '{mat_path}'. Please run 'python prepare_dataset.py'.")

    if errors:
        return None, None, None, errors

    try:
        df_jobs = pd.read_csv(jobs_csv_path)
        vectorizer = joblib.load(vec_path)
        job_vectors = joblib.load(mat_path)
        return df_jobs, vectorizer, job_vectors, []
    except Exception as e:
        return None, None, None, [f"Error loading ML artifacts: {str(e)}"]


def parse_job_skills_list(skills_str: Any) -> List[str]:
    """Helper to parse comma or semicolon separated skills into a clean list."""
    if not isinstance(skills_str, str) or not skills_str.strip():
        return []
    items = re.split(r'[,;|\n]+', skills_str)
    cleaned = []
    for it in items:
        clean_it = it.strip()
        if clean_it and clean_it.lower() not in ["not available", "nan", "none", "n/a"]:
            cleaned.append(clean_it)
    return cleaned


def compute_skill_overlap(candidate_skills: List[str], req_skills: List[str], pref_skills: List[str]) -> Tuple[float, List[str], List[str]]:
    """
    Computes skill overlap ratio between candidate skills and job requirements.
    Returns: (skill_score_0_to_100, matched_skills, missing_skills)
    """
    if not req_skills and not pref_skills:
        return 50.0, candidate_skills[:4], []

    cand_set = {s.lower() for s in candidate_skills}
    matched = []
    missing = []

    # Check required skills
    for skill in req_skills:
        skill_lower = skill.lower()
        # Direct or substring match
        if any(c in skill_lower or skill_lower in c for c in cand_set):
            matched.append(skill)
        else:
            missing.append(skill)

    # Check preferred skills
    pref_matched = []
    for skill in pref_skills:
        skill_lower = skill.lower()
        if any(c in skill_lower or skill_lower in c for c in cand_set):
            pref_matched.append(skill)
            if skill not in matched:
                matched.append(skill)

    # Overlap formula: full credit for required, 0.5 credit for preferred
    total_req = max(len(req_skills), 1)
    overlap_val = (len([s for s in matched if s in req_skills]) + 0.5 * len(pref_matched)) / total_req
    score = min(100.0, overlap_val * 100.0)

    return score, matched, missing


def compute_category_relevance(candidate_domain: str, job_category: str) -> float:
    """
    Computes domain/category fit score (0 to 100%).
    Ensures engineering resumes receive priority for engineering jobs.
    """
    is_job_eng = job_category in PRIMARY_CATEGORIES

    if candidate_domain == "cs_it_data":
        if job_category in CS_TECH_CATEGORIES:
            return 100.0
        elif is_job_eng:
            return 65.0
        else:
            return 20.0  # Engineering candidate gets low relevance for HR/Marketing

    elif candidate_domain == "mechanical":
        if job_category in ["Mechanical Engineering", "Automation / Robotics", "Other Engineering"]:
            return 100.0
        elif is_job_eng:
            return 65.0
        else:
            return 20.0

    elif candidate_domain == "civil":
        if job_category == "Civil Engineering":
            return 100.0
        elif is_job_eng:
            return 60.0
        else:
            return 20.0

    elif candidate_domain == "electrical_ece":
        if job_category in ["Electronics / ECE", "Electrical Engineering", "Embedded Systems", "Automation / Robotics"]:
            return 100.0
        elif is_job_eng:
            return 70.0
        else:
            return 20.0

    else: # non_engineering candidate
        if not is_job_eng:
            return 100.0
        else:
            return 35.0


def recommend_jobs(
    raw_resume_text: str,
    df_jobs: pd.DataFrame,
    vectorizer: Any,
    job_vectors: Any,
    top_n: int = 5,
    filter_category: str = "All",
    filter_location: str = "All",
    filter_experience: str = "All",
    filter_employment_type: str = "All",
    min_match_score: int = 0
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Computes top N job recommendations based on:
    - 70% TF-IDF Cosine Similarity
    - 20% Skill Overlap
    - 10% Category Relevance
    Applies optional sidebar filters and returns ranked job cards with full match explanations.
    """
    if not raw_resume_text or df_jobs is None or vectorizer is None or job_vectors is None:
        return [], {}

    # 1. Clean resume text and vectorize
    cleaned_resume = clean_text(raw_resume_text)
    if not cleaned_resume:
        return [], {}

    resume_vector = vectorizer.transform([cleaned_resume])

    # 2. Compute Cosine Similarities against all job vectors
    raw_similarities = cosine_similarity(resume_vector, job_vectors)[0]

    # 3. Extract candidate skills and infer domain
    candidate_skills = extract_candidate_skills(raw_resume_text)
    candidate_domain = detect_candidate_domain(raw_resume_text, candidate_skills)

    # 4. Score each job record
    scored_jobs = []
    
    # Pre-parse filters
    cat_filter = filter_category.strip()
    loc_filter = filter_location.strip()
    exp_filter = filter_experience.strip()
    emp_filter = filter_employment_type.strip()

    for idx, row in df_jobs.iterrows():
        # Apply Filters
        job_cat = str(row.get("category", "")).strip()
        if cat_filter != "All" and job_cat != cat_filter:
            continue

        job_loc = str(row.get("location", "")).strip()
        if loc_filter != "All" and loc_filter.lower() not in job_loc.lower():
            continue

        job_exp = str(row.get("experience", "")).strip()
        job_title = str(row.get("job_title", "")).strip()
        job_emp = str(row.get("employment_type", "")).strip()

        if exp_filter != "All":
            exp_f_lower = exp_filter.lower()
            matched_exp = False
            if exp_f_lower in job_exp.lower() or exp_f_lower in job_title.lower():
                matched_exp = True
            elif exp_f_lower == "fresher" and ("0-1" in job_exp or "graduate" in job_exp.lower() or "fresher" in job_title.lower()):
                matched_exp = True
            elif exp_f_lower == "internship" and ("intern" in job_title.lower() or "intern" in job_emp.lower() or "student" in job_exp.lower()):
                matched_exp = True
            elif exp_f_lower == "entry level" and ("entry level" in job_title.lower() or "0-2" in job_exp or "early career" in job_exp.lower()):
                matched_exp = True
            elif exp_f_lower == "junior" and ("junior" in job_title.lower() or "1-3" in job_exp):
                matched_exp = True
            elif exp_f_lower == "associate" and ("associate" in job_title.lower() or "2-4" in job_exp):
                matched_exp = True
            if not matched_exp:
                continue

        job_emp = str(row.get("employment_type", "")).strip()
        if emp_filter != "All" and emp_filter.lower() != job_emp.lower():
            continue

        # A. TF-IDF Similarity Score (0-100)
        # In TF-IDF with 15k sparse vocabulary, cosine similarity for matched resumes typically peaks around 0.20-0.25.
        # We scale by 0.24 so that strong textual matches reach 80-95% similarity.
        raw_sim = float(raw_similarities[idx])
        tfidf_score = min(98.0, (raw_sim / 0.24) * 100.0)

        # B. Skill Overlap Score (0-100)
        req_skills_list = parse_job_skills_list(row.get("required_skills", ""))
        pref_skills_list = parse_job_skills_list(row.get("preferred_skills", ""))
        skill_score, matched_skills, missing_skills = compute_skill_overlap(
            candidate_skills, req_skills_list, pref_skills_list
        )

        # C. Category Relevance Score (0-100)
        cat_score = compute_category_relevance(candidate_domain, job_cat)

        # D. Final Weighted Score
        # Formula: 70% TF-IDF + 20% Skill Overlap + 10% Category Relevance
        final_score = (0.70 * tfidf_score) + (0.20 * skill_score) + (0.10 * cat_score)
        final_score = round(max(5.0, min(99.0, final_score)), 1)

        # Check minimum match score filter
        if final_score < min_match_score:
            continue

        # Create explanation summary
        matched_str = ", ".join(matched_skills[:5]) if matched_skills else "General technical skills"
        missing_str = ", ".join(missing_skills[:4]) if missing_skills else "None - strong alignment"

        scored_jobs.append({
            "job_id": row.get("job_id", f"TP-{idx}"),
            "job_title": row.get("job_title", "Unknown Role"),
            "company": row.get("company", "Verified Enterprise"),
            "location": row.get("location", "Location Not Specified"),
            "category": job_cat,
            "industry": row.get("industry", "Technology"),
            "required_skills": row.get("required_skills", "Not Available"),
            "preferred_skills": row.get("preferred_skills", "Not Available"),
            "education": row.get("education", "Not Available"),
            "experience": row.get("experience", "Not Available"),
            "employment_type": row.get("employment_type", "Full-time"),
            "job_description": row.get("job_description", "No description available."),
            "source": row.get("source", "Verified Dataset"),
            "source_url": row.get("source_url", ""),
            "date_collected": row.get("date_collected", "2026"),
            
            # Scores & Explanations
            "match_score": int(round(final_score)),
            "raw_similarity": round(raw_sim, 4),
            "tfidf_score": round(tfidf_score, 1),
            "skill_score": round(skill_score, 1),
            "category_score": round(cat_score, 1),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "explanation": {
                "matched_skills_text": matched_str,
                "missing_skills_text": missing_str,
                "domain_fit": "High" if cat_score >= 80 else ("Moderate" if cat_score >= 50 else "Low"),
                "summary": (
                    f"Matched skills: {matched_str}. "
                    f"Relevant category: {job_cat}. "
                    f"Resume-job similarity: {int(round(tfidf_score))}%."
                )
            }
        })

    # Sort descending by match_score
    scored_jobs.sort(key=lambda x: x["match_score"], reverse=True)

    # Candidate profile summary
    profile_summary = {
        "candidate_domain": candidate_domain,
        "candidate_skills": candidate_skills,
        "total_jobs_evaluated": len(df_jobs),
        "total_jobs_matching_filter": len(scored_jobs)
    }

    return scored_jobs[:top_n], profile_summary
