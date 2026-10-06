"""
TalentPulse AI - PDF & Resume Text Parser
File: utils/pdf_parser.py
Description: Safe text extraction from PDF and TXT resumes with comprehensive
             error handling for empty, scanned, or corrupted files.
"""

import io
import re
from typing import Tuple, Optional, Dict, Any, List

# Safe import for PyMuPDF (prefer pymupdf, fallback to fitz)
try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None


def extract_text_from_pdf(file_bytes: bytes) -> Tuple[str, Optional[str]]:
    """
    Safely extracts digital text from uploaded PDF bytes.
    
    Returns:
        (extracted_text, error_message)
        If extraction succeeds: (text, None)
        If extraction fails: ("", error_message)
    """
    if not file_bytes or len(file_bytes) == 0:
        return "", "The uploaded file is empty (0 bytes). Please upload a valid resume."
    
    if fitz is None:
        return "", "PyMuPDF library is not installed. Please run 'pip install pymupdf'."

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        return "", f"Corrupted or invalid PDF file format: {str(e)}"

    if doc.page_count == 0:
        doc.close()
        return "", "The uploaded PDF document contains 0 pages."

    pages_text = []
    try:
        for page_idx in range(doc.page_count):
            page = doc.load_page(page_idx)
            text = page.get_text("text")
            if text and text.strip():
                pages_text.append(text.strip())
    except Exception as e:
        doc.close()
        return "", f"Error reading text from PDF pages: {str(e)}"
    finally:
        doc.close()

    full_text = "\n\n".join(pages_text).strip()

    # Handle scanned/image-only PDF
    if len(full_text) == 0:
        return "", (
            "No readable text could be extracted from this PDF. "
            "The document appears to be a scanned image or contains non-extractable elements. "
            "Please upload a standard text-based PDF or copy-paste text."
        )

    # Check for unusually short resume
    words = full_text.split()
    if len(words) < 25:
        # We still return the text, but note the brevity
        pass

    return full_text, None


def parse_resume_metadata(raw_text: str) -> Dict[str, Any]:
    """
    Extracts summary candidate information (education, experience, contact)
    directly from the parsed resume text using heuristic pattern matching.
    """
    if not raw_text:
        return {
            "education": "Not Available",
            "experience": "Not Available",
            "email": "Not specified",
            "phone": "Not specified",
            "word_count": 0,
            "char_count": 0
        }

    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    text_lower = raw_text.lower()

    # 1. Contact info
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', raw_text)
    email = email_match.group(0) if email_match else "Not specified"

    phone_match = re.search(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', raw_text)
    phone = phone_match.group(0) if phone_match else "Not specified"

    # 2. Extract Education
    edu_found = []
    edu_patterns = [
        r'(b\.?tech[^\n,]*)',
        r'(bachelor(?:\'s)?\s+(?:of\s+)?(?:technology|engineering|science|arts)?[^\n,]*)',
        r'(m\.?tech[^\n,]*)',
        r'(master(?:\'s)?\s+(?:of\s+)?(?:technology|engineering|science|business)?[^\n,]*)',
        r'(b\.?e\.?[^\n,]*)',
        r'(b\.?sc[^\n,]*)',
        r'(m\.?sc[^\n,]*)',
        r'(bca[^\n,]*)',
        r'(mca[^\n,]*)',
        r'(diploma\s+(?:in)?[^\n,]*)',
        r'(ph\.?d[^\n,]*)'
    ]
    for pat in edu_patterns:
        matches = re.findall(pat, text_lower, flags=re.I)
        for m in matches:
            clean_m = m.strip()
            if len(clean_m) > 4 and clean_m.title() not in edu_found:
                edu_found.append(clean_m.title())

    # Fallback search for lines under 'Education' header
    if not edu_found:
        for idx, line in enumerate(lines):
            if 'education' in line.lower() and idx + 1 < len(lines):
                edu_found.append(lines[idx + 1])
                break

    education_str = "; ".join(edu_found[:2]) if edu_found else "Not explicitly specified in text"

    # 3. Extract Experience
    exp_found = []
    exp_patterns = [
        r'(\d+\+?\s+years?(?:\s+of)?\s+experience[^\n,]*)',
        r'(intern(?:ship)?[^\n,]*)',
        r'((?:software|data|web|ml|developer|engineer|analyst|designer)\s+(?:intern|specialist|lead|associate|trainee)[^\n,]*)'
    ]
    for pat in exp_patterns:
        matches = re.findall(pat, text_lower, flags=re.I)
        for m in matches:
            clean_m = m.strip()
            if len(clean_m) > 4 and clean_m.title() not in exp_found:
                exp_found.append(clean_m.title())

    if not exp_found:
        for idx, line in enumerate(lines):
            if 'experience' in line.lower() and idx + 1 < len(lines):
                exp_found.append(lines[idx + 1])
                break

    experience_str = "; ".join(exp_found[:2]) if exp_found else "Fresher / Not explicitly specified"

    return {
        "education": education_str,
        "experience": experience_str,
        "email": email,
        "phone": phone,
        "word_count": len(raw_text.split()),
        "char_count": len(raw_text)
    }
