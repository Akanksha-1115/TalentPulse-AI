"""
TalentPulse AI - Text Preprocessing & Skill Extraction Pipeline
File: utils/preprocessing.py
Description: Cleans resume & job text while preserving technical keywords,
             and extracts domain-specific skills across engineering and allied disciplines.
"""

import re
from typing import List, Set, Tuple

# Comprehensive dictionary of technical and domain skills
SKILL_ONTOLOGY = {
    # Programming Languages & Core Computing
    "python": "Python",
    "java": "Java",
    "cpp": "C++",
    "c++": "C++",
    "cplusplus": "C++",
    "csharp": "C#",
    "c#": "C#",
    "c": "C",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "go": "Go",
    "golang": "Go",
    "rust": "Rust",
    "ruby": "Ruby",
    "php": "PHP",
    "swift": "Swift",
    "kotlin": "Kotlin",
    "dart": "Dart",
    "r": "R",
    "sql": "SQL",
    "matlab": "MATLAB",
    "assembly": "Assembly",
    "bash": "Bash / Shell",
    "shell": "Shell Scripting",

    # Web & Full-Stack Development
    "html": "HTML5",
    "html5": "HTML5",
    "css": "CSS3",
    "css3": "CSS3",
    "react": "React",
    "reactjs": "React",
    "angular": "Angular",
    "angularjs": "Angular",
    "vue": "Vue.js",
    "vuejs": "Vue.js",
    "nextjs": "Next.js",
    "nodejs": "Node.js",
    "express": "Express.js",
    "django": "Django",
    "flask": "Flask",
    "fastapi": "FastAPI",
    "spring": "Spring Boot",
    "spring boot": "Spring Boot",
    "rest api": "REST APIs",
    "graphql": "GraphQL",
    "microservices": "Microservices",

    # Data Science, AI & Machine Learning
    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "nlp": "Natural Language Processing (NLP)",
    "natural language processing": "Natural Language Processing (NLP)",
    "computer vision": "Computer Vision",
    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch",
    "keras": "Keras",
    "scikit-learn": "Scikit-Learn",
    "sklearn": "Scikit-Learn",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "opencv": "OpenCV",
    "data science": "Data Science",
    "data analysis": "Data Analysis",
    "statistics": "Statistical Modeling",
    "tableau": "Tableau",
    "power bi": "Power BI",
    "powerbi": "Power BI",
    "excel": "Microsoft Excel",

    # Cloud, DevOps & Infrastructure
    "aws": "Amazon Web Services (AWS)",
    "amazon web services": "Amazon Web Services (AWS)",
    "azure": "Microsoft Azure",
    "gcp": "Google Cloud Platform (GCP)",
    "google cloud": "Google Cloud Platform (GCP)",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "k8s": "Kubernetes",
    "ci/cd": "CI/CD",
    "cicd": "CI/CD",
    "jenkins": "Jenkins",
    "terraform": "Terraform",
    "git": "Git",
    "github": "GitHub",
    "linux": "Linux",
    "ansible": "Ansible",

    # Databases & Big Data
    "mysql": "MySQL",
    "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL",
    "mongodb": "MongoDB",
    "redis": "Redis",
    "sqlite": "SQLite",
    "oracle": "Oracle Database",
    "cassandra": "Cassandra",
    "hadoop": "Apache Hadoop",
    "spark": "Apache Spark",
    "kafka": "Apache Kafka",
    "snowflake": "Snowflake",

    # Computer Science Fundamentals
    "data structures": "Data Structures",
    "algorithms": "Algorithms",
    "dbms": "DBMS",
    "operating systems": "Operating Systems",
    "computer networks": "Computer Networks",
    "system design": "System Design",
    "oop": "Object-Oriented Programming (OOP)",
    "cybersecurity": "Cybersecurity",
    "penetration testing": "Penetration Testing",
    "cryptography": "Cryptography",

    # Electronics & Embedded Systems
    "embedded c": "Embedded C",
    "microcontroller": "Microcontrollers",
    "microcontrollers": "Microcontrollers",
    "arduino": "Arduino",
    "raspberry pi": "Raspberry Pi",
    "verilog": "Verilog",
    "vhdl": "VHDL",
    "fpga": "FPGA",
    "vlsi": "VLSI",
    "pcb design": "PCB Design",
    "rtos": "RTOS",
    "arm": "ARM Architecture",
    "iot": "Internet of Things (IoT)",
    "simulink": "Simulink",

    # Mechanical & Automation Engineering
    "autocad": "AutoCAD",
    "solidworks": "SolidWorks",
    "catia": "CATIA",
    "ansys": "ANSYS",
    "creo": "Creo",
    "thermodynamics": "Thermodynamics",
    "fluid mechanics": "Fluid Mechanics",
    "fea": "Finite Element Analysis (FEA)",
    "cfd": "Computational Fluid Dynamics (CFD)",
    "robotics": "Robotics",
    "ros": "Robot Operating System (ROS)",
    "plc": "PLC Programming",
    "mechatronics": "Mechatronics",
    "cnc": "CNC Machining",
    "gd&t": "GD&T",

    # Civil & Structural Engineering
    "revit": "Autodesk Revit",
    "staad pro": "STAAD.Pro",
    "staad.pro": "STAAD.Pro",
    "etabs": "ETABS",
    "structural analysis": "Structural Analysis",
    "surveying": "Surveying",
    "concrete technology": "Concrete Technology",
    "geotechnical": "Geotechnical Engineering",
    "gis": "GIS Mapping",

    # Secondary / Non-Engineering Roles
    "financial analysis": "Financial Analysis",
    "accounting": "Accounting",
    "digital marketing": "Digital Marketing",
    "seo": "Search Engine Optimization (SEO)",
    "recruitment": "Recruitment & Talent Acquisition",
    "human resources": "Human Resources Management",
    "project management": "Project Management",
    "agile": "Agile / Scrum",
    "scrum": "Agile / Scrum",
    "supply chain": "Supply Chain Management",
    "logistics": "Logistics & Operations",
    "ui/ux": "UI/UX Design",
    "figma": "Figma",
    "photoshop": "Adobe Photoshop",
    "illustrator": "Adobe Illustrator"
}


def clean_text(text: str) -> str:
    """
    Cleans freeform text while strictly preserving essential technical keywords
    and programming terms (C++, C#, .NET, Python, SQL, AWS, Docker, React, etc.).
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    stripped_lower = text.strip().lower()
    if stripped_lower in ["not available", "nan", "none", "null", "n/a", ""]:
        return ""

    # Preserve technical terms before punctuation stripping
    text = re.sub(r'\bC\+\+', 'cpp cplusplus', text, flags=re.I)
    text = re.sub(r'\bC#', 'csharp', text, flags=re.I)
    text = re.sub(r'(?:\.NET|\bDOTNET\b)', 'dotnet', text, flags=re.I)
    text = re.sub(r'\bNode\.js\b', 'nodejs', text, flags=re.I)
    text = re.sub(r'\bVue\.js\b', 'vuejs', text, flags=re.I)
    text = re.sub(r'\bReact\.js\b', 'reactjs', text, flags=re.I)
    text = re.sub(r'\bAngular\.js\b', 'angularjs', text, flags=re.I)
    text = re.sub(r'\bNext\.js\b', 'nextjs', text, flags=re.I)
    text = re.sub(r'\bCI\/CD\b', 'cicd', text, flags=re.I)
    text = re.sub(r'\bTCP\/IP\b', 'tcpip', text, flags=re.I)
    text = re.sub(r'\bPower\s*BI\b', 'powerbi', text, flags=re.I)
    text = re.sub(r'\bC\s*\/\s*C\+\+', 'c_language cpp cplusplus', text, flags=re.I)

    # Lowercase
    text = text.lower()

    # Remove URLs and emails
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)

    # Remove unwanted punctuation
    text = re.sub(r'[^a-z0-9_]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_candidate_skills(raw_text: str) -> List[str]:
    """
    Extracts high-confidence technical and domain skills from candidate resume text
    using pattern-matching against the comprehensive skill ontology.
    """
    if not raw_text:
        return []

    text_lower = " " + raw_text.lower() + " "
    # Normalized search string with padded spaces
    clean_search_text = " " + re.sub(r'[^a-z0-9+#.]', ' ', text_lower) + " "

    detected_skills = set()

    # Match each term in ontology
    for term, label in SKILL_ONTOLOGY.items():
        term_clean = term.lower()
        # Word boundary or special character boundaries
        if len(term_clean) <= 2:
            # Short tokens like C, R, Go need strict boundaries
            pattern = rf'(?:^|[\s,;/()\-]){re.escape(term_clean)}(?:$|[\s,;/()\-])'
            if re.search(pattern, text_lower):
                detected_skills.add(label)
        else:
            if f" {term_clean} " in clean_search_text or term_clean in text_lower:
                detected_skills.add(label)

    return sorted(list(detected_skills))


def detect_candidate_domain(raw_text: str, candidate_skills: List[str]) -> str:
    """
    Infers the primary domain of the candidate to inform category relevance boost.
    Returns: 'cs_it_data', 'mechanical', 'civil', 'electrical_ece', or 'non_engineering'
    """
    skills_lower = [s.lower() for s in candidate_skills]
    text_lower = raw_text.lower()

    # Mech indicators
    mech_indicators = ["autocad", "solidworks", "catia", "ansys", "thermodynamics", "mechanical", "fea", "cfd"]
    if any(m in skills_lower or m in text_lower for m in mech_indicators):
        if "mechanical" in text_lower or sum(1 for m in mech_indicators if m in skills_lower) >= 2:
            return "mechanical"

    # Civil indicators
    civil_indicators = ["staad.pro", "etabs", "revit", "surveying", "civil", "structural analysis", "concrete"]
    if any(c in skills_lower or c in text_lower for c in civil_indicators):
        if "civil" in text_lower or sum(1 for c in civil_indicators if c in skills_lower) >= 2:
            return "civil"

    # Electrical / ECE indicators
    elec_indicators = ["verilog", "vhdl", "vlsi", "microcontroller", "embedded c", "pcb design", "arm architecture", "electrical", "electronics"]
    if any(e in skills_lower or e in text_lower for e in elec_indicators):
        if "electronics" in text_lower or "electrical" in text_lower or sum(1 for e in elec_indicators if e in skills_lower) >= 2:
            return "electrical_ece"

    # CS / IT / Data indicators
    cs_indicators = ["python", "java", "c++", "sql", "machine learning", "deep learning", "react", "docker", "aws", "data science", "nlp", "html"]
    if any(c in skills_lower for c in cs_indicators) or "computer science" in text_lower or "b.tech" in text_lower or "software" in text_lower:
        return "cs_it_data"

    # Default to non-engineering if no technical indicators
    non_eng_indicators = ["accounting", "financial", "marketing", "human resources", "recruitment", "sales"]
    if any(n in text_lower for n in non_eng_indicators):
        return "non_engineering"

    return "cs_it_data" # Default engineering assumption for technical students
