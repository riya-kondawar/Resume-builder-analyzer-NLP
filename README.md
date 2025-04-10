# 📄 NLP Resume Builder, Analyzer & Parser

SmartResume Pro is an AI-powered resume builder and analyzer with an integrated chatbot for user assistance. It offers resume parsing, skill extraction, ATS compatibility checks, and automated PDF generation.

## **1️⃣ Resume Builder**
- Users input personal, educational, and professional details.
- Select from multiple professional resume templates.
- Generates an **ATS-friendly resume** with a calculated **ATS Score**.
- Allows users to download the generated resume in **PDF/DOCX format**.

## **2️⃣ Resume Upload & Analyzer**
- Users can **upload** an existing resume (PDF/DOCX).
- System extracts content using **PDF parsing**.
- Uses NLP techniques to:
  - **Parse key sections**: Name, contact details, experience, education, skills, projects, etc.
  - **Identify key skills** from the resume.
  - **Suggest relevant courses & learning resources** to improve skills.

## **3️⃣ Resume Analysis w.r.t Job Description (JD)**
- User uploads a **job description**.
- NLP model compares **resume skills vs. job requirements**.
- Provides:
  - **Match percentage** between resume & JD.
  - **Missing skills & keywords** that should be added.
  - **Suggested modifications** to improve resume for the specific job.




# Recreate venv
python -m venv venv
venv\Scripts\activate  # (Windows)

# activate virtual env
.\venv\Scripts\Activate.ps1

# Delete venv folder
rm -rf venv  # or use File Explorer

# Reinstall using fresh requirements
pip freeze > requirements.txt
pip install -r requirements.txt

# Run command 
python -m streamlit run app.py
streamlit run app.py


pip install pdfkit
pip install jinja2

## Alternative Resume websites:
1. https://www.resume-now.com/ 

