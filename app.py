# Run command:
# python -m streamlit run app.py

import streamlit as st  # ✅ Import first
import os
import joblib
import gc
from files.pages.resume_builder import build_resume
from files.pages.upload_resume import upload_resume
from files.pages.job_match import job_match
from files.pages.home import home
from files.pages.chatbot import chatbot

# ✅ Set page config (must be the first Streamlit command)
st.set_page_config(
    page_title="Resume Builder & Analyzer", 
    page_icon="📄", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ✅ Garbage Collection
gc.collect()

# ✅ Download fonts if not present
if not os.path.exists("files/fonts/DejaVuSans.ttf"):
    import requests
    from zipfile import ZipFile

    url = "https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-fonts-ttf-2.37.zip"
    r = requests.get(url)
    
    with open("./files/fonts/dejavu-fonts.zip", "wb") as f:
        f.write(r.content)

    with ZipFile("files/fonts/dejavu-fonts.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

    os.rename("./files/fonts/dejavu-fonts-ttf-2.37/ttf/DejaVuSans.ttf", "files/fonts/DejaVuSans.ttf")
    os.rename("./files/fonts/dejavu-fonts-ttf-2.37/ttf/DejaVuSans-Bold.ttf", "files/fonts/DejaVuSans-Bold.ttf")

# ✅ Load trained models with error handling
model, model2, model3 = None, None, None

def load_models():
    global model, model2, model3
    try:
        model = joblib.load("./files/models/clf.pkl", mmap_mode='r')
        model2 = joblib.load("./files/models/encoder.pkl", mmap_mode='r')
        model3 = joblib.load("./files/models/tfidf.pkl", mmap_mode='r')
    except FileNotFoundError:
        st.warning("⚠️ Some models are missing. Resume analysis may not work correctly.")
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")

load_models()

# ✅ Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Select a Page", 
    [
        "🏠 Home", 
        "🛠 Build Your Resume", 
        "📄 Upload Resume", 
        "📊 Job Match",
        "🤖 Career Chatbot"
    ]
)

# ✅ Route to respective pages
if page == "🏠 Home":
    home()
elif page == "🛠 Build Your Resume":
    build_resume()
elif page == "📄 Upload Resume":
    upload_resume()
elif page == "📊 Job Match":
    job_match()
elif page == "🤖 Career Chatbot":
    chatbot()
