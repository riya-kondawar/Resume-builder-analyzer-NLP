# Run command:
# python -m streamlit run app.py

import streamlit as st
# import os
import joblib
import gc
from files.pages.resume_builder import build_resume
from files.pages.upload_resume import upload_resume
from files.pages.job_match import job_match
from files.pages.home import home
from files.pages.chatbot import chatbot
import spacy

# Initialize configuration first
st.set_page_config(
    page_title="Resume Builder & Analyzer", 
    page_icon="📄", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MODEL_PATHS = {
    'clf': "./files/models/clf.joblib",
    'encoder': "./files/models/encoder.joblib",
    'tfidf': "./files/models/tfidf.joblib"
}

@st.cache_resource
def load_spacy_model():
    """Load spaCy model with caching"""
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_models():
    # """Load all ML models with caching and error handling"""
    # models = {}
    # for name, path in MODEL_PATHS.items():
    #     try:
    #         # models[name] = joblib.load(path, mmap_mode='r')
    #         models[name] = joblib.load(path)
    #     except FileNotFoundError:
    #         st.warning(f"⚠️ Model file not found: {path}")
    #         models[name] = None
    #     except Exception as e:
    #         st.error(f"❌ Error loading {name}: {str(e)}")
    #         models[name] = None
    # return models
    """Load all ML models with caching and error handling"""
    models = {}
    for name, path in MODEL_PATHS.items():
        try:
            models[name] = joblib.load(path)  # <-- removed mmap_mode='r'
        except FileNotFoundError:
            st.warning(f"⚠️ Model file not found: {path}")
            models[name] = None
        except Exception as e:
            st.error(f"❌ Error loading {name}: {str(e)}")
            models[name] = None
    return models


def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'current_page': "🏠 Home",
        'resume_data': {
            "text": "",
            "skills": [],
            "experience": [],
            "education": []
        },
        'messages': [],
        'conversation': None,
        'last_api_call': 0,
        'user_profile': {
            "name": "",
            "career_level": "",
            "industry": "",
            "skills": [],
            "last_updated": None
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def main():
    """Main application logic"""
    # Initialize components
    nlp = load_spacy_model()
    initialize_session_state()
    
    # Load models (cached)
    models = load_models()
    
    # Navigation sidebar
    st.sidebar.title("📌 Navigation")
    page_options = [
        "🏠 Home", 
        "🛠 Build Your Resume", 
        "📄 Resume Analysis", 
        "🤖 Career Chatbot"
    ]
    
    selected_page = st.sidebar.radio(
        "Select a Page",
        options=page_options,
        index=page_options.index(st.session_state.current_page)
    )
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
        st.rerun()
    
    # Page routing
    page_functions = {
        "🏠 Home": home,
        "🛠 Build Your Resume": build_resume,
        "📄 Resume Analysis": upload_resume,
        "🤖 Career Chatbot": chatbot
    }
    
    if st.session_state.current_page in page_functions:
        page_functions[st.session_state.current_page]()
    
    # Clean up
    gc.collect()

if __name__ == "__main__":
    main()