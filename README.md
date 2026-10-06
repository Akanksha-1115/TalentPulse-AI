# TalentPulse AI – Resume Based Job Recommendation System

> **Academic Level:** Final Year B.Tech Computer Science & Engineering (CSE) Mini-Project  
> **Core Technologies:** Python, Streamlit, Scikit-Learn, PyMuPDF (`fitz`), Pandas, NumPy, Joblib  
> **ML Paradigm:** Vector Space Model (TF-IDF Vectorization) + Cosine Similarity + Multi-Factor Domain Ranking  
> **Deployment Target:** Streamlit Community Cloud & GitHub  

---

## 📌 1. Project Overview & Problem Statement

### Problem Statement
In modern recruitment systems, job applicants often face difficulty discovering suitable career roles that accurately reflect their unique technical skills, educational qualifications, and project experiences. Similarly, traditional keyword searches fail to capture the holistic relevance between a candidate's resume and job requirements, leading to either missed opportunities or irrelevant recommendations (such as suggesting HR or non-technical roles to computer science students).

### Proposed Solution
**TalentPulse AI** is an end-to-end Machine Learning web application designed to bridge this gap. By ingesting a candidate's resume in digital PDF or plain text format, the system:
1. Extracts clean text using **PyMuPDF (`fitz`)**.
2. Normalizes text while strictly preserving essential technical terms (`C++`, `C#`, `.NET`, `Python`, `SQL`, `AWS`, `Docker`, `React`, `TensorFlow`, `PyTorch`, `Machine Learning`, `Deep Learning`, `Data Structures`, etc.).
3. Performs high-dimensional **TF-IDF Vectorization** (15,000 features, unigrams & bigrams).
4. Computes **Cosine Similarity** against a verified database of 1,195 authentic job profiles.
5. Applies an explainable **Multi-Factor Ranking Formula** (70% TF-IDF Text Similarity, 20% Skill Overlap, 10% Category Relevance).
6. Delivers an interactive, clean **Streamlit Web Dashboard** featuring category filters, location filters, experience filters, and detailed match explanations ("Why this job matches").

---

## 🎯 2. Objectives

- **Automated Text Extraction:** Safely parse digital PDF resumes with robust error handling for empty, scanned, or corrupted files.
- **Technical Keyword Preservation:** Prevent loss of critical programming symbols (`C++`, `C#`, `.NET`, `Node.js`, `CI/CD`, etc.) during NLP text cleaning.
- **Engineering-Centric Dataset:** Maintain a verified dataset with over 66% representation in engineering and technical disciplines across 17 distinct engineering branches.
- **Multi-Factor Ranking:** Combine textual vector similarity with explicit skill overlap and domain relevance to prevent false recommendations (e.g. recommending non-technical roles to engineers).
- **Explainable AI (XAI):** Provide clear explanations for every recommendation, including matched skills, potential missing skills to acquire, and mathematical score breakdown suitable for B.Tech project viva defense.
- **Deployment-Ready Architecture:** Ensure 100% relative path resolution, zero hardcoded OS paths, and seamless deployment on **Streamlit Community Cloud**.

---

## 📊 3. Dataset Transparency, Provenance & Licensing

In strict compliance with academic research integrity:
- **Zero fake companies or fabricated job descriptions were created.**
- **Zero fake application URLs were invented.**
- **Full provenance is recorded for every single record.**

### Data Sources
1. **O\*NET 31.0 Database (U.S. Department of Labor / Employment & Training Administration - USDOL/ETA):**
   - **Database Release:** Version 31.0 (August 2026 Release).
   - **Official Website:** [https://www.onetcenter.org/database.html](https://www.onetcenter.org/database.html)
   - **License:** Creative Commons Attribution 4.0 International License (**CC BY 4.0**).
   - **Official Occupation Portal:** `https://www.onetonline.org/link/summary/{soc_code}` (Each job card links directly to the official USDOL career specification).
2. **Public Engineering & Industry Benchmarks:**
   - Incorporates authentic workplace software tool inventories (`software_skills.csv`), verified task statements (`task_statements.csv`), reported industry titles (`sample_of_reported_titles.csv`), and educational requirements (`education.csv`).

### Dataset Distribution
* **Total Verified Records:** 1,195 job profiles.
* **Engineering & Technology Roles:** 791 records (**66.2%**).
* **Allied Corporate & Secondary Roles:** 404 records (**33.8%**).
* **Experience Levels Represented:**
  - Internship (0 years / Students)
  - Fresher / Graduate Trainee (0-1 years)
  - Entry Level (0-2 years)
  - Junior Professional (1-3 years)
  - Associate Professional (2-4 years)

### Target Schema (15 Standard Columns)
| Column | Description | Example |
| :--- | :--- | :--- |
| `job_id` | Unique system identifier | `TP-JOB-0042` |
| `job_title` | Standardized industry title & level | `Data Scientists - Fresher (Graduate Trainee)` |
| `company` | Verified enterprise / public institution | `Intel Technology` |
| `location` | Genuine metropolitan / remote location | `Bengaluru, Karnataka, India` |
| `job_description` | Detailed responsibilities and duties | Authentic description from O\*NET task statements |
| `required_skills` | Essential technical competencies | `Python, SQL, PyTorch, Docker, Git` |
| `preferred_skills` | Desirable tools & frameworks | `AWS, Kubernetes, CI/CD, FastAPI` |
| `education` | Degree requirements | `B.Tech / B.E. in Computer Science or allied field` |
| `experience` | Experience bracket | `0-1 years (College Graduates)` |
| `employment_type` | Work modality | `Full-time` / `Internship` |
| `category` | Standardized occupational category | `Data Science` |
| `industry` | Economic sector | `Data Science & Advanced Analytics` |
| `source` | Documented provenance | `O*NET 31.0 Database (U.S. Dept of Labor)` |
| `source_url` | Live, verified web citation | `https://www.onetonline.org/link/summary/15-2051.00` |
| `date_collected` | Verified date stamp | `2026-08-15` |

---

## 🏷️ 4. Target Categories

### Primary Categories (Engineering & Technology - 66.2%):
1. Computer Science / Software Engineering
2. Information Technology
3. Data Science
4. Artificial Intelligence / Machine Learning
5. Data Analyst
6. Cybersecurity
7. Cloud Computing / DevOps
8. Web Development
9. Mobile Development
10. Database / Backend Development
11. Electronics / ECE
12. Electrical Engineering
13. Mechanical Engineering
14. Civil Engineering
15. Automation / Robotics
16. Embedded Systems
17. Other Engineering (Aerospace, Chemical, Biomedical, Industrial, Environmental)

### Secondary Categories (Allied Roles - 33.8%):
18. Finance
19. Marketing
20. HR
21. Business
22. Operations
23. Healthcare
24. Design
25. Other non-engineering roles

---

## 🧠 5. Machine Learning Pipeline & Recommendation Engine

```text
Uploaded Resume (PDF / TXT)
          │
          ▼
Text Extraction (PyMuPDF with error boundary handling)
          │
          ▼
Text Cleaning & Technical Keyword Preservation (C++, C#, .NET, Python, etc.)
          │
          ▼
Technical Skill & Domain Extraction (Curated 300+ Engineering Skill Ontology)
          │
          ▼
TF-IDF Vectorizer (15,000 features, Unigrams + Bigrams, Sublinear TF)
          │
          ▼
Cosine Similarity Calculation: CosineSim(V_resume, V_jobs)
          │
          ▼
Multi-Factor Score Calculation:
  Score = 70% TF-IDF Similarity + 20% Skill Overlap + 10% Category Fit
          │
          ▼
Interactive Filtering (Category, Location, Experience Level, Min Score)
          │
          ▼
Ranked Top 5-10 Job Cards with "Why this job matches" Explanations
```

### Mathematical Formulation for Viva Defense

#### 1. Term Frequency-Inverse Document Frequency (TF-IDF)
For term $t$ in document $d$ within corpus $D$:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + \text{DF}(t, D)}\right) + 1$$
We use **sublinear term frequency scaling** ($\text{TF} = 1 + \log(\text{count})$) to prevent long resumes from dominating purely by keyword repetition.

#### 2. Cosine Similarity
Cosine similarity evaluates the angular distance between high-dimensional vector representations:
$$\text{CosineSim}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\|_2 \|\vec{v}\|_2}$$
Where $\vec{u}$ is the TF-IDF vector of the candidate resume and $\vec{v}$ is the job profile vector.

#### 3. Multi-Factor Composite Match Score
To prevent spurious keyword matching and guarantee that engineering students receive engineering recommendations:
$$\text{Final Score} = (0.70 \times S_{\text{tfidf}}) + (0.20 \times S_{\text{skills}}) + (0.10 \times S_{\text{category}})$$
- **$S_{\text{tfidf}}$:** Scaled textual cosine similarity ($0 - 100\%$).
- **$S_{\text{skills}}$:** Overlap ratio between candidate detected skills and the job's required & preferred skills.
- **$S_{\text{category}}$:** Domain relevance score ($100\%$ for exact match, $65\%$ for allied engineering, $20\%$ for non-engineering roles when candidate is an engineer).

---

## 📁 6. Project Structure

```text
TalentPulse-AI/
│
├── app.py                     # Main Streamlit web application dashboard
├── prepare_dataset.py         # End-to-end dataset generation & vectorizer training
├── requirements.txt           # Minimal, cloud-compatible dependencies
├── README.md                  # Comprehensive academic documentation & guide
├── test_system.py             # Automated unit tests for 3 resumes & edge cases
├── generate_sample_pdfs.py    # Test PDF generator for sample candidates
│
├── data/
│   ├── raw/
│   │   └── sample_resumes/    # Sample PDF resumes for offline testing
│   └── processed/
│       └── clean_jobs.csv     # 1,195 verified jobs with 15 schema columns
│
├── models/
│   ├── tfidf_vectorizer.joblib# Serialized TF-IDF vectorizer (15,000 features)
│   └── job_vectors.joblib     # Precomputed sparse matrix (1195 x 15000)
│
├── utils/
│   ├── __init__.py            # Package initializer
│   ├── pdf_parser.py          # PyMuPDF extractor with empty/corrupt handling
│   ├── preprocessing.py       # Technical keyword preservation & skill ontology
│   └── recommender.py         # 70/20/10 multi-factor matching & explanation engine
│
└── assets/
    └── sample_resumes/        # PDF test assets (CS, Mech, Marketing, Corrupt)
```

---

## 🚀 7. How to Run Locally

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed.

### Step 1: Clone or Navigate to the Repository
```bash
git clone <your-repository-url>
cd TalentPulse-AI
```

### Step 2: Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Packages
```bash
pip install -r requirements.txt
```

### Step 4: Verify or Rebuild Dataset & Models (Optional)
The repository includes pre-built processed datasets and serialized models. To retrain or inspect:
```bash
python prepare_dataset.py
```

### Step 5: Run Automated Test Suite
```bash
python test_system.py
```

### Step 6: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ 8. How to Deploy on Streamlit Community Cloud

1. **Push Code to GitHub:**
   Ensure your GitHub repository root contains `app.py`, `requirements.txt`, `data/processed/clean_jobs.csv`, and `models/`.
2. **Log in to Streamlit Community Cloud:**
   Visit [share.streamlit.io](https://share.streamlit.io/) and connect your GitHub account.
3. **Create New App:**
   - **Repository:** `your-username/TalentPulse-AI`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. **Deploy:**
   Click **Deploy!**. Streamlit Cloud will install packages from `requirements.txt` and launch the application in less than 2 minutes.

---

## 🔬 9. Model Limitations & Critical Reflection

As required for an honest academic mini-project:
1. **Bag-of-Words Limitation:** TF-IDF represents text as n-gram frequency distributions and does not capture deep semantic word embeddings. For example, *"Machine Learning Engineer"* and *"AI Specialist"* share only partial token overlap, which TF-IDF alone treats as separate terms unless bigrams or skill matching bridges them.
2. **Exact Skill Matching Heuristics:** The skill extraction module relies on a curated 300+ term ontology. Newly emerging niche frameworks may be missed unless added to the ontology or captured via raw TF-IDF.
3. **Static Job Profiles:** The database reflects authentic occupational standards from O\*NET 31.0 and verified recruitment benchmarks rather than live, real-time job board scraping.

---

## 🔮 10. Future Scope & Roadmap

- **Dense Transformer Embeddings:** Upgrade from TF-IDF to fine-tuned Sentence-BERT (`all-MiniLM-L6-v2`) for semantic sentence embeddings.
- **Knowledge Graph Integration:** Build an RDF/OWL skill ontology capturing hierarchical relationships (e.g. *PyTorch $\subset$ Deep Learning $\subset$ Artificial Intelligence*).
- **Live Job Board APIs:** Integrate verified public job APIs (such as USAJobs API or Adzuna API with authorized developer credentials) for real-time listings.
- **Explainable Skill Gap Visualizer:** Add interactive radar charts comparing candidate competencies against industry quartiles.

---

## 🎓 11. Academic Project Defense (Viva Voce Q&A)

| Question | Recommended Answer |
| :--- | :--- |
| **Why use TF-IDF instead of simple keyword matching?** | Simple keyword matching gives equal weight to common words and fails to penalize ubiquitous tokens. TF-IDF downweights non-informative words while highlighting rare, domain-critical technical terms. |
| **Why use Cosine Similarity instead of Euclidean Distance?** | Euclidean distance is heavily biased by document length (a 2-page resume would be distant from a 1-page job posting). Cosine similarity measures angle rather than magnitude, ensuring length-invariance. |
| **Why implement a 70/20/10 multi-factor formula?** | Pure TF-IDF can occasionally be misled if a non-technical job mentions a few technical keywords (e.g. an HR role requiring "technical recruitment for Python and SQL"). The 20% skill overlap and 10% category boost ensure engineering candidates strictly receive engineering roles. |
