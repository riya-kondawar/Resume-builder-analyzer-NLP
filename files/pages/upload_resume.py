import streamlit as st
import PyPDF2
from io import BytesIO

def upload_resume():
    """Analyze Resume Page"""
    st.title("📄 Upload & Analyze Resume")
    st.write("Upload your resume and get insights on its content.")
    
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
                text = "DOCX parsing would require python-docx library"
                # Uncomment after installing python-docx
                # text = extract_text_from_docx(uploaded_file)
            else:
                st.error("Unsupported file format")
                return
                
            st.subheader("Extracted Text")
            st.text_area("", text, height=300)
            
            # Add analysis section
            if text:
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

def analyze_resume(text):
    with st.spinner("Analyzing resume..."):
        # Placeholder for actual analysis
        st.subheader("Analysis Results")
        
        # Basic stats
        word_count = len(text.split())
        st.metric("Word Count", word_count)
        
        # Placeholder for skill extraction
        st.write("**Key Skills Detected:**")
        st.write("Python, Machine Learning, Data Analysis (Example)")
        
        # Placeholder for job match score
        st.write("**Job Match Potential:**")
        st.progress(75)
        
        st.info("Full analysis would use the loaded ML models to extract skills and match jobs")