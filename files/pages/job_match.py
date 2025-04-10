import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import joblib
from collections import defaultdict

# Load models and data
@st.cache_resource
def load_job_data():
    try:
        # Load your trained models
        clf = joblib.load("./files/models/clf.pkl")
        encoder = joblib.load("./files/models/encoder.pkl")
        tfidf = joblib.load("./files/models/tfidf.pkl")
        
        # Sample job database (in production, replace with real database)
        jobs_db = pd.DataFrame([
            {
                "title": "Senior Data Scientist",
                "company": "Tech Innovations Inc",
                "description": "Looking for experienced data scientist with Python, ML, and cloud experience...",
                "requirements": "Python, Machine Learning, SQL, AWS, 5+ years experience",
                "skills": ["python", "machine learning", "sql", "aws", "statistics"],
                "location": "Remote",
                "salary": "$120,000 - $150,000"
            },
            {
                "title": "Machine Learning Engineer",
                "company": "AI Solutions Co",
                "description": "Seeking ML engineer to develop and deploy models...",
                "requirements": "Python, TensorFlow, PyTorch, Docker, 3+ years experience",
                "skills": ["python", "tensorflow", "pytorch", "docker", "mlops"],
                "location": "San Francisco, CA",
                "salary": "$130,000 - $160,000"
            },
            {
                "title": "Data Analyst",
                "company": "Analytics Pro",
                "description": "Data analyst needed for business insights team...",
                "requirements": "SQL, Tableau, Python, Excel, 2+ years experience",
                "skills": ["sql", "tableau", "python", "excel", "data visualization"],
                "location": "New York, NY",
                "salary": "$90,000 - $110,000"
            }
        ])
        
        return clf, encoder, tfidf, jobs_db
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None

def job_match():
    """Standalone Job Matching System"""
    st.title("🎯 Smart Job Matcher")
    st.markdown("""
    ### AI-powered career matching that understands your profile
    Enter your skills and experience to find your perfect match.
    """)
    
    # Initialize session state variables
    if 'saved_jobs' not in st.session_state:
        st.session_state.saved_jobs = []
    if 'current_job' not in st.session_state:
        st.session_state.current_job = None
    
    # User input section (replacing resume upload dependency)
    with st.expander("✍️ Enter Your Profile Details", expanded=True):
        user_skills = st.text_area("Your Skills (comma separated)", 
                                 "Python, Machine Learning, SQL, Data Analysis")
        user_experience = st.selectbox("Years of Experience", 
                                     ["0-2", "3-5", "6-10", "10+"])
        user_description = st.text_area("Brief Professional Summary",
                                      "Experienced data professional with expertise in...")
    
    # User preferences section
    with st.expander("⚙️ Set Your Preferences", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            experience_filter = st.selectbox(
                "Desired Experience Level",
                ["Any", "Entry Level", "Mid Level", "Senior", "Executive"]
            )
            location_type = st.selectbox(
                "Preferred Location Type",
                ["Any", "On-site", "Hybrid", "Remote"]
            )
        with col2:
            salary_range = st.slider(
                "Minimum Salary Expectation",
                50000, 250000, 100000, step=10000,
                format="$%d"
            )
            skill_boost = st.multiselect(
                "Skills to emphasize",
                ["Python", "Machine Learning", "SQL", "AWS", "Data Analysis"],
                default=[]
            )
    
    # Analysis section
    if st.button("🔍 Find My Best Matches", type="primary", use_container_width=True):
        with st.status("🧠 Analyzing your profile...", expanded=True) as status:
            # Create combined profile text
            profile_text = f"""
            Skills: {user_skills}
            Experience: {user_experience} years
            Summary: {user_description}
            """
            
            st.write("🛠 Processing your skills and experience...")
            time.sleep(0.5)
            
            # Load models and data
            clf, encoder, tfidf, jobs_db = load_job_data()
            
            # Step 3: Match against jobs
            st.write("🔍 Finding matching opportunities...")
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Get matches (using models if available)
            if clf is not None and tfidf is not None:
                # Transform profile text using TF-IDF
                profile_vec = tfidf.transform([profile_text])
                
                # Transform job descriptions and calculate similarity
                job_descriptions = jobs_db['description'] + " " + jobs_db['requirements']
                job_vecs = tfidf.transform(job_descriptions)
                
                # Calculate cosine similarity
                similarities = cosine_similarity(profile_vec, job_vecs)[0]
                jobs_db['match_score'] = (similarities * 100).round(1)
            else:
                # Fallback: assign random scores
                jobs_db['match_score'] = np.random.randint(70, 98, size=len(jobs_db))
            
            # Apply filters
            filtered_jobs = jobs_db.copy()
            if experience_filter != "Any":
                filtered_jobs = filtered_jobs[filtered_jobs['title'].str.contains(experience_filter, case=False)]
            if location_type != "Any":
                filtered_jobs = filtered_jobs[filtered_jobs['location'].str.contains(location_type, case=False)]
            
            # Sort by match score
            filtered_jobs = filtered_jobs.sort_values('match_score', ascending=False)
            
            status.update(label="Analysis complete!", state="complete", expanded=False)
        
        # Display results
        st.success(f"✨ Found {len(filtered_jobs)} matching jobs!")
        
        # Results tabs
        tab1, tab2 = st.tabs(["Best Matches", "All Opportunities"])
        
        with tab1:
            st.subheader("🎯 Your Top Matches")
            for _, job in filtered_jobs.head(3).iterrows():
                display_job_card(job)
        
        with tab2:
            st.subheader("📋 All Matching Jobs")
            for _, job in filtered_jobs.iterrows():
                display_job_card(job)
        
        # Skills gap analysis
        if not filtered_jobs.empty and 'skills' in filtered_jobs.columns:
            analyze_skill_gaps(user_skills, filtered_jobs)

def display_job_card(job):
    """Display job card with expandable details"""
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### {job['title']}")
            st.caption(f"🏢 {job['company']} | 🌎 {job['location']} | 💰 {job['salary']}")
        with col2:
            st.metric("Match Score", f"{job['match_score']}%")
        
        with st.expander("View Details"):
            st.markdown("#### Description")
            st.write(job['description'])
            
            st.markdown("#### Requirements")
            st.write(job['requirements'])
            
            # Action buttons
            btn1, btn2 = st.columns(2)
            with btn1:
                if st.button("⭐ Save", key=f"save_{job['title']}"):
                    if 'saved_jobs' not in st.session_state:
                        st.session_state.saved_jobs = []
                    st.session_state.saved_jobs.append(job.to_dict())
                    st.toast("Job saved to your favorites!")
            with btn2:
                if st.button("📝 Apply Now", key=f"apply_{job['title']}"):
                    st.session_state.current_job = job.to_dict()
                    st.info("Application system would launch here")

def analyze_skill_gaps(user_skills, matched_jobs):
    """Analyze missing skills across top matches"""
    st.divider()
    st.subheader("🔍 Skills Gap Analysis")
    
    # Get all required skills from top jobs
    all_required_skills = defaultdict(int)
    for skills in matched_jobs['skills'].head(5):
        for skill in skills:
            all_required_skills[skill] += 1
    
    # Get user skills
    user_skills_set = set(skill.strip().lower() for skill in user_skills.split(','))
    
    # Find missing skills
    missing_skills = [skill for skill in all_required_skills if skill not in user_skills_set]
    
    if missing_skills:
        st.warning("These skills appear in your top matches but aren't in your profile:")
        cols = st.columns(4)
        for i, skill in enumerate(sorted(missing_skills, key=lambda x: -all_required_skills[x])):
            cols[i%4].metric(skill.title(), f"in {all_required_skills[skill]} jobs")
        
        st.info("💡 Consider developing these skills to increase your matches")
    else:
        st.success("🎉 Great job! Your profile contains all key skills for your top matches")

if __name__ == "__main__":
    job_match()