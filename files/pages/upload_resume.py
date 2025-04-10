# import streamlit as st
# import PyPDF2
# import re
# import nltk
# import pandas as pd
# import numpy as np
# from io import BytesIO
# from collections import Counter
# import spacy
# from datetime import datetime
# import joblib
# from sklearn.metrics.pairwise import cosine_similarity

# # Load English language model for spaCy
# try:
#     nlp = spacy.load("en_core_web_sm")
# except:
#     st.error("Please install spaCy and its English model: `python -m spacy download en_core_web_sm`")
#     nlp = None

# # Download NLTK data if not present
# try:
#     nltk.data.find('tokenizers/punkt')
#     nltk.data.find('corpora/stopwords')
# except:
#     nltk.download('punkt')
#     nltk.download('stopwords')

# # Load trained models
# @st.cache_resource
# def load_models():
#     try:
#         clf = joblib.load("./files/models/clf.pkl")
#         encoder = joblib.load("./files/models/encoder.pkl")
#         tfidf = joblib.load("./files/models/tfidf.pkl")
#         return clf, encoder, tfidf
#     except Exception as e:
#         st.error(f"Error loading models: {str(e)}")
#         return None, None, None

# clf, encoder, tfidf = load_models()

# def upload_resume():
#     """Enhanced Resume Analysis Page with Model Integration"""
#     st.title("📊 Advanced Resume Analyzer Pro")
#     st.markdown("""
#     Upload your resume for comprehensive AI-powered analysis including:
#     - Skill extraction and validation
#     - Experience level classification
#     - Education details extraction
#     - Industry-standard resume scoring
#     - Job matching potential
#     """)
    
#     uploaded_file = st.file_uploader(
#         "Upload Resume (PDF, DOCX, TXT)", 
#         type=["pdf", "docx", "txt"],
#         accept_multiple_files=False
#     )
    
#     if uploaded_file is not None:
#         try:
#             file_type = uploaded_file.type
#             text = ""
            
#             if file_type == "application/pdf":
#                 text = extract_text_from_pdf(uploaded_file)
#             elif file_type == "text/plain":
#                 text = uploaded_file.read().decode("utf-8")
#             elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
#                 text = extract_text_from_docx(uploaded_file)
#             else:
#                 st.error("Unsupported file format")
#                 return
                
#             if text:
#                 with st.expander("🔍 View Extracted Text"):
#                     st.text(text[:5000] + "..." if len(text) > 5000 else text)
                
#                 with st.spinner("🤖 Performing deep AI analysis..."):
#                     analyze_resume(text)
                
#         except Exception as e:
#             st.error(f"Error processing file: {str(e)}")

# def extract_text_from_pdf(pdf_file):
#     text = ""
#     try:
#         pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
#         for page in pdf_reader.pages:
#             page_text = page.extract_text() or ""
#             text += page_text + "\n"
#     except Exception as e:
#         raise Exception(f"PDF extraction error: {str(e)}")
#     return text.strip()

# def extract_text_from_docx(docx_file):
#     try:
#         from docx import Document
#         doc = Document(BytesIO(docx_file.read()))
#         return "\n".join([para.text for para in doc.paragraphs])
#     except:
#         return "DOCX parsing requires python-docx library"

# def analyze_resume(text):
#     # Section 1: Basic Text Analysis
#     st.subheader("📋 Resume Overview")
#     col1, col2, col3 = st.columns(3)
#     word_count = len(text.split())
#     char_count = len(text)
#     sentence_count = len(nltk.sent_tokenize(text))
    
#     col1.metric("Word Count", word_count)
#     col2.metric("Character Count", char_count)
#     col3.metric("Sentence Count", sentence_count)
    
#     # Section 2: Skills Analysis (using both patterns and models)
#     st.subheader("🛠 Skills Analysis")
#     skills = extract_skills(text)
    
#     if clf is not None:
#         # Validate skills using the trained model
#         valid_skills = validate_skills(skills)
#         st.success(f"✅ Validated Skills ({len(valid_skills)}): " + ", ".join(valid_skills[:15]) + ("..." if len(valid_skills) > 15 else ""))
#         st.warning(f"⚠️ Unrecognized Skills ({len(skills)-len(valid_skills)}): " + ", ".join(set(skills)-set(valid_skills))[:15] + ("..." if len(set(skills)-set(valid_skills)) > 15 else ""))
#     else:
#         st.warning("Skills validation unavailable (model not loaded)")
#         st.write("Detected Skills:", ", ".join(skills[:20]) + ("..." if len(skills) > 20 else ""))
    
#     # Section 3: Experience Analysis
#     st.subheader("📅 Professional Experience")
#     experience = detect_experience(text)
    
#     exp_col1, exp_col2 = st.columns(2)
#     with exp_col1:
#         st.metric("Total Experience", f"{experience['total_years']} years")
    
#     if clf is not None:
#         # Predict experience level using the model
#         experience_level = predict_experience_level(text)
#         with exp_col2:
#             st.metric("Predicted Level", experience_level)
    
#     st.write("### Position History")
#     st.dataframe(pd.DataFrame(experience['positions']))
    
#     # Section 4: Education Analysis
#     st.subheader("🎓 Education Background")
#     education = detect_education(text)
#     for edu in education:
#         st.write(f"- {edu['degree']} from {edu['institution']} ({edu['year']})")
    
#     # Section 5: Model-Powered Analysis
#     if clf is not None and encoder is not None and tfidf is not None:
#         st.subheader("🧠 AI-Powered Insights")
        
#         # Resume Score
#         resume_score = calculate_resume_score(text)
#         st.progress(resume_score/100)
#         st.metric("Overall Resume Score", f"{resume_score}/100")
        
#         # Job Category Prediction
#         job_category = predict_job_category(text)
#         st.write("### Predicted Job Category")
#         st.info(f"🔹 {job_category}")
        
#         # Suggested Improvements
#         st.write("### 💡 Suggested Improvements")
#         improvements = suggest_improvements(text)
#         for imp in improvements:
#             st.write(f"- {imp}")
    
#     # Section 6: Keyword Optimization
#     st.subheader("🔑 Keyword Analysis")
#     keywords = extract_keywords(text)
#     st.write("Top 20 keywords (excluding common words):")
#     st.write(", ".join(keywords[:20]))
    
#     # Section 7: Sample Job Matches (placeholder)
#     if clf is not None:
#         st.subheader("💼 Top Job Matches")
#         job_matches = get_job_matches(text)
#         st.dataframe(job_matches)

# def extract_skills(text):
#     """Enhanced skill extraction with both patterns and NLP"""
#     skill_patterns = {
#         'programming': r'\b(python|java|c\+\+|javascript|typescript|go|rust|ruby|php|swift|kotlin)\b',
#         'web': r'\b(html|css|react|angular|vue|django|flask|node\.?js|express)\b',
#         'data': r'\b(sql|nosql|mysql|postgresql|mongodb|hadoop|spark|pandas|numpy|tensorflow|pytorch)\b',
#         'devops': r'\b(docker|kubernetes|aws|azure|gcp|ci/cd|jenkins|terraform|ansible)\b',
#         'soft': r'\b(leadership|communication|teamwork|problem solving|creativity)\b'
#     }
    
#     skills = set()
#     text_lower = text.lower()
    
#     for category, pattern in skill_patterns.items():
#         matches = re.findall(pattern, text_lower)
#         skills.update(matches)
    
#     # Additional spaCy based extraction if available
#     if nlp:
#         doc = nlp(text)
#         for ent in doc.ents:
#             if ent.label_ == "SKILL":
#                 skills.add(ent.text.lower())
    
#     return sorted(skills)

# def validate_skills(skills):
#     """Use the trained model to validate skills"""
#     if clf is None or encoder is None:
#         return skills
    
#     # Encode the skills using the loaded encoder
#     try:
#         encoded_skills = encoder.transform(skills)
#         predictions = clf.predict(encoded_skills)
#         return [skill for skill, pred in zip(skills, predictions) if pred == 1]
#     except:
#         return skills

# def predict_experience_level(text):
#     """Predict experience level using the model"""
#     if tfidf is None or clf is None:
#         return "N/A"
    
#     # Transform text using the loaded TF-IDF
#     text_tfidf = tfidf.transform([text])
#     prediction = clf.predict(text_tfidf)
    
#     levels = {
#         0: "Entry Level",
#         1: "Mid Level",
#         2: "Senior Level",
#         3: "Executive Level"
#     }
    
#     return levels.get(prediction[0], "Unknown")

# def calculate_resume_score(text):
#     """Calculate a comprehensive resume score"""
#     if tfidf is None:
#         return 70  # Default score if models not available
    
#     # Calculate score based on various factors
#     score = 50  # Base score
    
#     # Add points for skills
#     skills = extract_skills(text)
#     score += min(20, len(skills))  # Up to 20 points for skills
    
#     # Add points for experience
#     experience = detect_experience(text)
#     score += min(experience['total_years'] * 2, 20)  # Up to 20 points for experience
    
#     # Add points for education
#     education = detect_education(text)
#     score += min(len(education) * 5, 10)  # Up to 10 points for education
    
#     return min(100, score)  # Cap at 100

# def predict_job_category(text):
#     """Predict the most suitable job category"""
#     if tfidf is None or clf is None:
#         return "N/A (Model not available)"
    
#     categories = {
#         0: "Software Engineering",
#         1: "Data Science",
#         2: "Product Management",
#         3: "UX/UI Design",
#         4: "DevOps Engineering",
#         5: "Business Analysis"
#     }
    
#     text_tfidf = tfidf.transform([text])
#     prediction = clf.predict(text_tfidf)
#     return categories.get(prediction[0], "Unknown")

# def suggest_improvements(text):
#     """Generate improvement suggestions"""
#     suggestions = []
    
#     # Check for contact information
#     if not re.search(r'\b(email|phone|contact)\b', text, re.IGNORECASE):
#         suggestions.append("Add contact information")
    
#     # Check for measurable achievements
#     if len(re.findall(r'\d+%|\$\d+|\d+\+', text)) < 3:
#         suggestions.append("Add more quantifiable achievements (metrics, percentages, etc.)")
    
#     # Check for action verbs
#     action_verbs = ['achieved', 'managed', 'developed', 'led', 'increased', 'reduced']
#     if sum(text.lower().count(verb) for verb in action_verbs) < 5:
#         suggestions.append("Use more action verbs to describe your experience")
    
#     return suggestions

# def get_job_matches(text):
#     """Generate sample job matches (placeholder)"""
#     if tfidf is None:
#         return pd.DataFrame({
#             "Job Title": ["Sample Job 1", "Sample Job 2"],
#             "Company": ["Tech Corp", "Data Inc"],
#             "Match Score": ["85%", "78%"]
#         })
    
#     # Placeholder for actual job matching logic
#     jobs = {
#         "Job Title": ["Senior Python Developer", "Data Scientist", "DevOps Engineer"],
#         "Company": ["TechCorp", "DataWorld", "CloudSystems"],
#         "Required Skills": ["Python, Django, SQL", "Python, Machine Learning, Statistics", "AWS, Docker, Kubernetes"],
#         "Match %": [85, 72, 68]
#     }
    
#     return pd.DataFrame(jobs).sort_values("Match %", ascending=False)

# def extract_keywords(text):
#     """Enhanced keyword extraction"""
#     stop_words = set(stopwords.words('english'))
#     custom_stopwords = {"work", "experience", "project", "role", "company"}
#     stop_words.update(custom_stopwords)
    
#     words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
#     filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
#     word_freq = Counter(filtered_words)
#     return [word for word, count in word_freq.most_common(50)]































import streamlit as st
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from io import BytesIO
from collections import Counter
import spacy
from datetime import datetime
import pandas as pd

# Load English language model for spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.error("Please install spaCy and its English model: `python -m spacy download en_core_web_sm`")
    nlp = None

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('punkt')
    nltk.download('stopwords')

def upload_resume():
    """Enhanced Resume Analysis Page"""
    st.title("📄 Advanced Resume Analyzer")
    st.markdown("""
    Upload your resume for comprehensive analysis including:
    - Skill extraction
    - Experience level detection
    - Education details
    - Keyword optimization
    - Job matching potential
    """)
    
    uploaded_file = st.file_uploader(
        "Upload Resume (PDF, DOCX, TXT)", 
        type=["pdf", "docx", "txt"],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        try:
            file_type = uploaded_file.type
            text = ""
            
            if file_type == "application/pdf":
                text = extract_text_from_pdf(uploaded_file)
            elif file_type == "text/plain":
                text = uploaded_file.read().decode("utf-8")
            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = extract_text_from_docx(uploaded_file)
            else:
                st.error("Unsupported file format")
                return
                
            st.subheader("Extracted Text Preview")
            st.text_area("", text[:2000] + "..." if len(text) > 2000 else text, height=200)
            
            if text:
                with st.expander("Show Full Extracted Text"):
                    st.text(text)
                
                analyze_resume(text)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    except Exception as e:
        raise Exception(f"PDF extraction error: {str(e)}")
    return text.strip()

def extract_text_from_docx(docx_file):
    try:
        from docx import Document
        doc = Document(BytesIO(docx_file.read()))
        return "\n".join([para.text for para in doc.paragraphs])
    except:
        return "DOCX parsing requires python-docx library"

def analyze_resume(text):
    with st.spinner("Performing deep analysis..."):
        # Basic Text Analysis
        st.subheader("📊 Basic Statistics")
        col1, col2, col3 = st.columns(3)
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = len(nltk.sent_tokenize(text))
        
        col1.metric("Word Count", word_count)
        col2.metric("Character Count", char_count)
        col3.metric("Sentence Count", sentence_count)
        
        # Advanced NLP Analysis
        if nlp:
            doc = nlp(text)
            
            # Named Entity Recognition
            st.subheader("🔍 Key Information Extraction")
            
            # Skills Extraction
            skills = extract_skills(text)
            st.markdown("**🛠 Technical Skills Detected**")
            skills_cols = st.columns(4)
            for i, skill in enumerate(skills[:20]):
                skills_cols[i%4].success(skill)
            
            # Experience Detection
            experience = detect_experience(text)
            st.markdown("**📅 Professional Experience**")
            st.write(f"Total Experience: {experience['total_years']} years")
            st.dataframe(pd.DataFrame(experience['positions']))
            
            # Education Detection
            education = detect_education(text)
            st.markdown("**🎓 Education Background**")
            for edu in education:
                st.write(f"- {edu['degree']} from {edu['institution']} ({edu['year']})")
            
            # Keyword Analysis
            st.subheader("🔑 Keyword Analysis")
            keywords = extract_keywords(text)
            st.write("Top 20 keywords (excluding common words):")
            st.write(", ".join(keywords[:20]))
            
            # Sentiment Analysis
            sentiment = analyze_sentiment(text)
            st.metric("Overall Sentiment Score", f"{sentiment:.2f}/5.0")
            
        else:
            st.warning("SpaCy model not loaded - some advanced features disabled")

def extract_skills(text):
    """Extract technical skills using pattern matching"""
    skill_patterns = {
        'programming': r'\b(python|java|c\+\+|javascript|typescript|go|rust|ruby|php|swift|kotlin)\b',
        'web': r'\b(html|css|react|angular|vue|django|flask|node\.?js|express)\b',
        'data': r'\b(sql|nosql|mysql|postgresql|mongodb|hadoop|spark|pandas|numpy|tensorflow|pytorch)\b',
        'devops': r'\b(docker|kubernetes|aws|azure|gcp|ci/cd|jenkins|terraform|ansible)\b'
    }
    
    skills = set()
    text_lower = text.lower()
    
    for category, pattern in skill_patterns.items():
        matches = re.findall(pattern, text_lower)
        skills.update(matches)
    
    return sorted(skills)

def detect_experience(text):
    """Detect work experience duration and positions"""
    experience = {
        'total_years': 0,
        'positions': []
    }
    
    # Simple pattern matching for experience
    exp_pattern = r'(\d+)\+?\s*(years?|yrs?)\s*(of)?\s*(experience|exp)'
    match = re.search(exp_pattern, text, re.IGNORECASE)
    if match:
        experience['total_years'] = int(match.group(1))
    
    # Detect position history
    position_pattern = r'(\b[A-Z][a-z]+\b)\s+at\s+(\b[A-Z][a-zA-Z\s]+\b)\s+\((\d{4})\s*-\s*(\d{4}|present)\)'
    matches = re.findall(position_pattern, text)
    
    for match in matches:
        experience['positions'].append({
            'title': match[0],
            'company': match[1],
            'start': match[2],
            'end': match[3]
        })
    
    return experience

def detect_education(text):
    """Detect education background"""
    education = []
    
    # Common degree patterns
    degree_pattern = r'\b(B\.?Sc|B\.?Tech|B\.?E|B\.?A|M\.?Sc|M\.?Tech|M\.?A|PhD)\b'
    institution_pattern = r'((?:University|College|Institute|School)\s+of\s+[A-Za-z\s]+)'
    year_pattern = r'(19|20)\d{2}'
    
    # Find education sections
    edu_sections = re.findall(r'(education[^a-z0-9].*?)(?:\n\n|\Z)', text, re.IGNORECASE | re.DOTALL)
    
    for section in edu_sections:
        degrees = re.findall(degree_pattern, section, re.IGNORECASE)
        institutions = re.findall(institution_pattern, section)
        years = re.findall(year_pattern, section)
        
        for i, degree in enumerate(degrees):
            education.append({
                'degree': degree,
                'institution': institutions[i] if i < len(institutions) else "Unknown",
                'year': years[i] if i < len(years) else "Unknown"
            })
    
    return education

def extract_keywords(text):
    """Extract important keywords using TF-IDF like approach"""
    stop_words = set(stopwords.words('english'))
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
    word_freq = Counter(filtered_words)
    return [word for word, count in word_freq.most_common(50)]

def analyze_sentiment(text):
    """Basic sentiment analysis (placeholder for actual model)"""
    positive_words = ['excellent', 'achieved', 'success', 'lead', 'improved', 'award']
    negative_words = ['poor', 'lack', 'failed', 'issue', 'problem']
    
    positive_count = sum(text.lower().count(word) for word in positive_words)
    negative_count = sum(text.lower().count(word) for word in negative_words)
    
    return max(1.0, min(5.0, 3.0 + (positive_count - negative_count) / 10))
















# import streamlit as st
# import PyPDF2
# from io import BytesIO

# def upload_resume():
#     """Analyze Resume Page"""
#     st.title("📄 Upload & Analyze Resume")
#     st.write("Upload your resume and get insights on its content.")
    
#     uploaded_file = st.file_uploader(
#         "Upload Resume (PDF, DOCX, TXT)", 
#         type=["pdf", "docx", "txt"],
#         accept_multiple_files=False
#     )
    
#     if uploaded_file is not None:
#         try:
#             file_type = uploaded_file.type
#             text = ""
            
#             if file_type == "application/pdf":
#                 text = extract_text_from_pdf(uploaded_file)
#             elif file_type == "text/plain":
#                 text = uploaded_file.read().decode("utf-8")
#             elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
#                 text = "DOCX parsing would require python-docx library"
#                 # Uncomment after installing python-docx
#                 # text = extract_text_from_docx(uploaded_file)
#             else:
#                 st.error("Unsupported file format")
#                 return
                
#             st.subheader("Extracted Text")
#             st.text_area("", text, height=300)
            
#             # Add analysis section
#             if text:
#                 analyze_resume(text)
                
#         except Exception as e:
#             st.error(f"Error processing file: {str(e)}")

# def extract_text_from_pdf(pdf_file):
#     text = ""
#     try:
#         pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
#         for page in pdf_reader.pages:
#             page_text = page.extract_text() or ""
#             text += page_text + "\n"
#     except Exception as e:
#         raise Exception(f"PDF extraction error: {str(e)}")
#     return text.strip()

# def analyze_resume(text):
#     with st.spinner("Analyzing resume..."):
#         # Placeholder for actual analysis
#         st.subheader("Analysis Results")
        
#         # Basic stats
#         word_count = len(text.split())
#         st.metric("Word Count", word_count)
        
#         # Placeholder for skill extraction
#         st.write("**Key Skills Detected:**")
#         st.write("Python, Machine Learning, Data Analysis (Example)")
        
#         # Placeholder for job match score
#         st.write("**Job Match Potential:**")
#         st.progress(75)
        
#         st.info("Full analysis would use the loaded ML models to extract skills and match jobs")