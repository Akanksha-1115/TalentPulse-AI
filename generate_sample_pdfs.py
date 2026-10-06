"""
Generates test PDF resumes for TalentPulse AI evaluation
"""

import os
import pymupdf as fitz

assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "sample_resumes")
os.makedirs(assets_dir, exist_ok=True)

# 1. Engineering / CS & AI Resume
doc_cs = fitz.open()
page = doc_cs.new_page()
text_cs = """RAHUL VERMA
Email: rahul.verma@example.com | Phone: +91 98765 43210 | Bengaluru, India
GitHub: github.com/rahulverma | LinkedIn: linkedin.com/in/rahulverma

EDUCATION:
Bachelor of Technology in Computer Science & Engineering (2022 - 2026)
Vellore Institute of Technology | CGPA: 8.9 / 10.0
Relevant Coursework: Data Structures & Algorithms, Database Management Systems,
Machine Learning, Deep Learning, Operating Systems, Computer Networks.

TECHNICAL SKILLS:
Programming Languages: Python, SQL, C++, Java, JavaScript
Frameworks & Libraries: PyTorch, TensorFlow, Scikit-Learn, Pandas, NumPy, FastAPI, Flask
Cloud, DevOps & Tools: Git, GitHub, Docker, Kubernetes, Linux, AWS (EC2, S3)
Core Concepts: Machine Learning, NLP, Computer Vision, REST APIs, Microservices

PROJECTS:
1. TalentPulse AI: Resume Based Job Recommendation System
   - Implemented TF-IDF vectorization and cosine similarity across 1,195 verified jobs.
   - Built multi-factor ranking combining textual similarity, skill overlap, and category relevance.
2. Vision-Based Defect Detection System (PyTorch, OpenCV, FastAPI)
   - Trained convolutional neural network achieving 94.2% test accuracy on industrial parts.
   - Deployed containerized REST API with Docker on AWS EC2.

EXPERIENCE:
Machine Learning Engineering Intern | Summer 2025
- Built automated data cleaning pipelines in Pandas and trained baseline classification models.
- Collaborated with engineering team on deploying REST API endpoints in FastAPI.
"""
page.insert_text((50, 60), text_cs, fontsize=10, fontname="helv")
doc_cs.save(os.path.join(assets_dir, "sample_resume_cs_ai.pdf"))
doc_cs.close()

# 2. Mechanical Engineering Resume
doc_mech = fitz.open()
page_mech = doc_mech.new_page()
text_mech = """ADITYA RAO
Email: aditya.rao@example.com | Phone: +91 91234 56789 | Pune, India
LinkedIn: linkedin.com/in/adityarao

EDUCATION:
Bachelor of Technology in Mechanical Engineering (2021 - 2025)
College of Engineering Pune | CGPA: 8.6 / 10.0
Coursework: Thermodynamics, Fluid Mechanics, Strength of Materials, Machine Design, CAD/CAM.

TECHNICAL SKILLS:
CAD / 3D Modeling: AutoCAD, SolidWorks, CATIA, Creo
Engineering Simulation: ANSYS (Structural & Thermal FEA), CFD, MATLAB, Simulink
Manufacturing & Design: GD&T, CNC Machining, Mechatronics, PLC Programming, HVAC

PROJECTS:
1. Finite Element Analysis of Formula Student Race Car Chassis (ANSYS, SolidWorks)
   - Evaluated torsional rigidity and stress distribution under dynamic loading conditions.
2. Design and Thermal Optimization of Automotive Heat Exchanger
   - Designed 3D parametric model in SolidWorks; simulated fluid dynamics and heat transfer in CFD.

EXPERIENCE:
Mechanical Design Intern | Robert Bosch Engineering (Jan 2025 - Present)
- Assisted senior engineers in drafting production drawings and tolerance stack-up analysis.
- Generated 3D CAD assemblies and bill of materials (BOM) in SolidWorks.
"""
page_mech.insert_text((50, 60), text_mech, fontsize=10, fontname="helv")
doc_mech.save(os.path.join(assets_dir, "sample_resume_mechanical.pdf"))
doc_mech.close()

# 3. Non-Engineering (Marketing & HR) Resume
doc_mkt = fitz.open()
page_mkt = doc_mkt.new_page()
text_mkt = """PRIYA SHARMA
Email: priya.sharma@example.com | Phone: +91 99887 76655 | Mumbai, India

EDUCATION:
Master of Business Administration (MBA) in Marketing & HR (2023 - 2025)
Bachelor of Commerce (B.Com) - Honors

PROFESSIONAL SKILLS:
Marketing: Digital Marketing, Search Engine Optimization (SEO), Social Media Campaigns, Market Research
Human Resources: Talent Acquisition, Campus Recruitment, Employee Onboarding, HR Operations
Tools & Analytics: Microsoft Excel (VLOOKUP, Pivot Tables), Google Analytics, HubSpot CRM, PowerPoint

EXPERIENCE:
HR & Marketing Associate | Retail Enterprises (2024 - Present)
- Sourced and screened 150+ candidate resumes for sales and customer relations roles.
- Managed quarterly social media content calendar and increased inbound leads by 28%.
"""
page_mkt.insert_text((50, 60), text_mkt, fontsize=10, fontname="helv")
doc_mkt.save(os.path.join(assets_dir, "sample_resume_marketing_hr.pdf"))
doc_mkt.close()

# 4. Empty PDF (0 pages / 0 text)
doc_empty = fitz.open()
doc_empty.new_page() # blank page
doc_empty.save(os.path.join(assets_dir, "sample_empty_scanned.pdf"))
doc_empty.close()

# 5. Corrupted PDF
with open(os.path.join(assets_dir, "sample_corrupted.pdf"), "wb") as f:
    f.write(b"%PDF-1.4-Invalid-Binary-Data-Corrupted")

print(f"Generated sample resumes in: {assets_dir}")
