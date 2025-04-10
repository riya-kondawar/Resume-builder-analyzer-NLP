# home.py
import streamlit as st
from streamlit_extras.colored_header import colored_header

def home():
    """Home Page - Resume Analyzer & Career Companion"""
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .big-font {
            font-size:18px !important;
        }
        .feature-card {
            padding: 15px;
            border-radius: 10px;
            border: 1px dashed #e0e0e0;
            background: #779FA1B3;
            margin-bottom: 15px;
        }
        .step-card {
            text-align: center;
            padding: 20px 10px;
            margin-bottom: 20px;
            border-radius: 10px;
            border: 1px dashed #e0e0e0;
            background: #779FA153;
        }
    </style>
    """, unsafe_allow_html=True)

    # Hero Section
    colored_header(
        label="🚀 AI-Powered Resume Analyzer Pro",
        description="",
        color_name="blue-70"
    )
    
    st.markdown("""
    <div class='big-font'>
    Get <strong>instant resume feedback</strong>, <strong>skill extraction</strong>, and <strong>personalized job matching</strong> 
    powered by our advanced NLP engine.
    </div>
    """, unsafe_allow_html=True)
    
    # Layout
    col1, col2 = st.columns([2, 3])
    with col1:
        st.image(
            "https://www.resume-now.com/sapp/uploads/2024/08/resume-example-senior-financial-analyst.png", 
            width=400,
            caption="Example AI-analyzed resume"
        )
    
    with col2:
        st.markdown("""
        ### 🔥 Key Features
        
        <div class='feature-card'>
        <strong>📊 Smart Resume Analysis</strong><br>
        Extract skills, experience level, education and key qualifications automatically
        </div>
        
        <div class='feature-card'>
        <strong>🎯 Job Matching Score</strong><br>
        See how well your resume matches specific job descriptions
        </div>
        
        <div class='feature-card'>
        <strong>📈 Improvement Suggestions</strong><br>
        Get AI-powered recommendations to optimize your resume
        </div>
        
        <div class='feature-card'>
        <strong>💼 Career Path Recommendations</strong><br>
        Discover roles that match your profile
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Get Started Now →", type="primary", use_container_width=True):
            st.session_state.navigation = "📄 Upload Resume"
            st.rerun()

    # Divider
    st.divider()
    
    # How It Works Section
    st.markdown("### 📲 How It Works")
    grid_col1, grid_col2, grid_col3 = st.columns(3)

    with grid_col1:
        with st.container(border=True):
            st.image(
                "https://img.icons8.com/?size=150&id=103982&format=png&color=000000",
                width=100
            )
            st.markdown("""
            <div class='step-card'>
            <h4>Step 1</h4>
            <strong>Upload Your Resume</strong><br>
            PDF, DOCX or plain text formats supported
            </div>
            """, unsafe_allow_html=True)

    with grid_col2:
        with st.container(border=True):
            st.image(
                "https://img.icons8.com/color/150/ai.png",
                width=100
            )
            st.markdown("""
            <div class='step-card'>
            <h4>Step 2</h4>
            <strong>AI Analysis</strong><br>
            Our system extracts key information and skills
            </div>
            """, unsafe_allow_html=True)

    with grid_col3:
        with st.container(border=True):
            st.image(
                "https://img.icons8.com/color/150/job.png",
                width=100
            )
            st.markdown("""
            <div class='step-card'>
            <h4>Step 3</h4>
            <strong>Get Insights</strong><br>
            View analysis results and job matches instantly
            </div>
            """, unsafe_allow_html=True)

    # Testimonials Section (Optional)
    st.divider()
    st.markdown("### ❤️ Trusted by Job Seekers Worldwide")
    testimonial_col1, testimonial_col2 = st.columns(2)
    
    with testimonial_col1:
        with st.container(border=True):
            st.markdown("""
            ⭐⭐⭐⭐⭐  
            "Landeda job at Google after optimizing my resume with their suggestions!"
            """)
            st.caption("- Sarah K., Software Engineer")

    with testimonial_col2:
        with st.container(border=True):
            st.markdown("""
            ⭐⭐⭐⭐⭐  
            "The skill extraction helped me identify gaps in my profile I never noticed."
            """)
            st.caption("- Michael T., Marketing Director")

if __name__ == "__main__":
    home()











# import streamlit as st

# def home():
#     st.title("🏠 Welcome to Resume Builder & Analyzer")
#     st.write("This app helps you create, analyze, and match resumes with job descriptions.")

# def home():
#     """Home Page"""
#     st.title("🚀 Resume Builder & Analyzer")
#     st.write("AI-powered resume insights, skill extraction & job recommendations.")
#     st.write("This app helps you create, analyze, and match resumes with job descriptions.")

#     # Layout
#     col1, col2 = st.columns([2, 3])
#     with col1:
#         st.image("https://www.resume-now.com/sapp/uploads/2024/08/resume-example-senior-financial-analyst.png", width=400)
    
#     with col2:
#         st.markdown("""
#         ✅ **Extract Key Skills & Experience**  
#         ✅ **AI-powered Job Recommendations**  
#         ✅ **PDF & DocX Resume Support**  
#         ✅ **Instant Job Matching & Score Calculation**  
#         ✅ **Download AI-generated Resume Report**  
#         """)
    
#     if st.button("Try Now 🚀"):
#         st.session_state.navigation = "📄 Upload Resume"
#         st.experimental_rerun()

#     st.write("---")
#     st.markdown("### 🔍 How It Works?")
#     grid_col1, grid_col2, grid_col3 = st.columns(3)

#     with grid_col1:
#         st.image("https://img.icons8.com/?size=150&id=103982&format=png&color=000000")
#         st.write("**Step 1:** Upload your Resume")

#     with grid_col2:
#         st.image("https://img.icons8.com/color/150/ai.png")
#         st.write("**Step 2:** AI Analyzes Key Information")

#     with grid_col3:
#         st.image("https://img.icons8.com/color/150/job.png")
#         st.write("**Step 3:** Get Matched to Jobs!")