import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from time import time
import backoff
from difflib import SequenceMatcher
import requests
from io import BytesIO
import PyPDF2
import docx
import json
from datetime import datetime
from packaging import version

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ Google API key not found! Check your .env file.")
    st.stop()

# ✅ Configure Google AI
genai.configure(api_key=API_KEY)

# Limited access to the API
TOPIC_CATEGORIES = {
    "Resume & Job Application": [
        "resume", "cv", "cover letter", "job application", "LinkedIn", 
        "portfolio", "personal branding", "ATS-friendly resume", 
        "resume formatting", "resume keywords"
    ],
    "Career Development": [
        "career", "career path", "career change", "career growth", 
        "career advice", "job search", "job opportunities"
    ],
    "Interview Preparation": [
        "interview", "interview preparation", "interview tips", 
        "behavioral interview", "technical interview"
    ],
    "Professional Skills": [
        "skills", "soft skills", "hard skills", "communication", 
        "leadership", "teamwork"
    ]
}

ALLOWED_TOPICS = [topic for sublist in TOPIC_CATEGORIES.values() for topic in sublist]

GREETINGS = [
    "hi", "hello", "hey", "good morning", "good afternoon", 
    "good evening", "greetings", "hi there", "hello there"
]

def init_session_state():
    print("✅ init_session_state() was called")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "last_api_call" not in st.session_state:
        st.session_state.last_api_call = 0
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {
            "name": "",
            "career_level": "",
            "industry": "",
            "skills": [],
            "last_updated": None
        }
    if "resume_data" not in st.session_state:
        st.session_state.resume_data = {
            "text": "",
            "skills": [],
            "experience": [],
            "education": []
        }
    if st.session_state.conversation is None:
        try:
            st.session_state.conversation = genai.GenerativeModel("gemini-1.5-flash").start_chat(history=[])
        except Exception as e:
            if "118" in str(e):
                st.session_state.conversation = None
            else:
                st.session_state.conversation = None


# Text extraction 
def extract_text_from_pdf(file):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(file.read()))
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"PDF extraction error: {str(e)}")
    return text.strip()

def extract_text_from_docx(file):
    try:
        doc = docx.Document(BytesIO(file.read()))
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"DOCX extraction error: {str(e)}")
        return ""

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def is_relevant(question):
    """Enhanced relevance checking with fuzzy matching"""
    question_lower = question.lower()
    
    if any(greeting == question_lower for greeting in GREETINGS):
        return True
        
    for topic in ALLOWED_TOPICS:
        if topic in question_lower:
            return True
            
    # Fuzzy matching for similar terms
    for word in question_lower.split():
        for topic in ALLOWED_TOPICS:
            if similar(word, topic) > 0.7:  # 70% similarity threshold
                return True
                
    return False

# ✅ API Call with rate limiting and retries
@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def safe_api_call(prompt):
    """Make API call with rate limiting"""
    current_time = time()
    
    # Rate limiting (1 call per second)
    if current_time - st.session_state.last_api_call < 1:
        time.sleep(1 - (current_time - st.session_state.last_api_call))
    
    if st.session_state.conversation is None:
        try:
            st.session_state.conversation = genai.GenerativeModel("gemini-1.5-flash").start_chat(history=[])
        except Exception as e:
            if "118" in str(e):
                st.session_state.conversation = None  # Silent fail
            else:
                st.warning("⚠️ Gemini model couldn't be loaded. Try again later.")
                st.stop()

    # Make the API call
    response = st.session_state.conversation.send_message(prompt)
    st.session_state.last_api_call = time()
    return response

# ✅ Suggested follow-up questions
def get_suggested_followups(response_text):
    """Generate relevant follow-up questions"""
    detected_topics = []
    for category, topics in TOPIC_CATEGORIES.items():
        for topic in topics:
            if topic in response_text.lower():
                detected_topics.append((category, topic))
    
    if not detected_topics:
        return []
    
    # Sort by relevance (most mentioned topics first)
    detected_topics.sort(key=lambda x: response_text.lower().count(x[1]), reverse=True)
    
    suggestions = []
    for category, topic in detected_topics[:2]:  # Top 2 most relevant topics
        if category == "Resume & Job Application":
            suggestions.extend([
                "How should I format my resume?",
                "What are the most important resume keywords?",
                "How long should my resume be?"
            ])
        elif category == "Interview Preparation":
            suggestions.extend([
                "What are common interview questions?",
                "How should I answer behavioral questions?",
                "What should I wear to an interview?"
            ])
    
    return list(set(suggestions))[:3]  # Return max 3 unique suggestions

# ✅ Resume analysis functions
def analyze_resume(text):
    """Extract key information from resume text"""
    if not text:
        return {}
    
    # Simple extraction
    skills = []
    experience = []
    education = []
    
    # Basic keyword matching
    skill_keywords = ["skill", "ability", "expertise", "proficient"]
    exp_keywords = ["experience", "work", "job", "employment"]
    edu_keywords = ["education", "degree", "university", "college"]
    
    for line in text.split('\n'):
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in skill_keywords):
            skills.append(line.strip())
        elif any(keyword in line_lower for keyword in exp_keywords):
            experience.append(line.strip())
        elif any(keyword in line_lower for keyword in edu_keywords):
            education.append(line.strip())
    
    return {
        "skills": skills[:10],  # Limit to top 10
        "experience": experience[:5],
        "education": education[:3]
    }

# ✅ Main chatbot function
def chatbot():
    """Career Guidance Chatbot with enhanced features"""
    init_session_state()
    
    st.title("💬 Career Guidance Chatbot")
    st.markdown("""
    Ask me about:
    - Resume writing and optimization
    - Job search strategies
    - Interview preparation
    - Career development
    """)
    
    # File uploader for resume analysis
    uploaded_file = st.file_uploader(
        "📄 Upload your resume for personalized advice (PDF/DOCX)", 
        type=["pdf", "docx"],
        key="resume_uploader"
    )
    
    if uploaded_file and not st.session_state.resume_data.get("text", ""):
        with st.spinner("Analyzing your resume..."):
            if uploaded_file.type == "application/pdf":
                text = extract_text_from_pdf(uploaded_file)
            else:
                text = extract_text_from_docx(uploaded_file)
            
            if text:
                st.session_state.resume_data["text"] = text
                analysis = analyze_resume(text)
                st.session_state.resume_data.update(analysis)
                st.success("Resume analyzed successfully!")
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Suggested questions for new users
    if not st.session_state.messages:
        st.markdown("**Try asking:**")
        cols = st.columns(3)
        sample_questions = [
            "How do I make my resume ATS-friendly?",
            "What are common interview questions for software engineers?",
            "How can I improve my LinkedIn profile?"
        ]
        for i, question in enumerate(sample_questions):
            if cols[i].button(question):
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()
    
    # Chat input
    user_input = st.chat_input("Type your career question here...")
    
    if user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Prepare context for the AI
        context = f"""
        You are a professional career advisor helping with job search, resume writing, 
        interview preparation, and career development. Be concise but helpful.
        
        User profile: {json.dumps(st.session_state.user_profile, indent=2)}
        Resume data: {json.dumps({k: v for k, v in st.session_state.resume_data.items() if k != 'text'}, indent=2)}
        """
        
        # Generate response
        with st.spinner("Thinking..."):
            try:
                if user_input.lower() in GREETINGS:
                    bot_response = "Hello! I'm your career assistant. How can I help you with your job search or career questions today?"
                elif "analyze my resume" in user_input.lower() and st.session_state.resume_data["text"]:
                    bot_response = f"Based on your resume:\n\n"
                    bot_response += f"- **Skills**: {', '.join(st.session_state.resume_data['skills'][:5])}\n"
                    bot_response += f"- **Experience**: {st.session_state.resume_data['experience'][0] if st.session_state.resume_data['experience'] else 'Not specified'}\n"
                    bot_response += f"- **Education**: {st.session_state.resume_data['education'][0] if st.session_state.resume_data['education'] else 'Not specified'}\n\n"
                    bot_response += "How would you like me to help you improve your resume?"
                elif is_relevant(user_input):
                    response = safe_api_call(context + "\n\nQuestion: " + user_input)
                    bot_response = response.text
                else:
                    bot_response = "I specialize in career-related topics. Please ask about resume writing, job search, or interview preparation."
            except Exception as e:
                bot_response = f"⚠️ Error: {str(e)}"
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        
        with st.chat_message("assistant"):
            st.markdown(bot_response)
            
            # Show suggested follow-ups
            suggestions = get_suggested_followups(bot_response)
            if suggestions:
                st.markdown("**You might ask:**")
                for suggestion in suggestions:
                    if st.button(suggestion, key=f"suggestion_{suggestion[:20]}"):
                        st.session_state.messages.append({"role": "user", "content": suggestion})
                        st.rerun()
        
        # Feedback mechanism
        if not bot_response.startswith("⚠️"):
            st.markdown("---")
            cols = st.columns(2)
            if cols[0].button("👍 Helpful"):
                log_feedback(user_input, bot_response, True)
                st.success("Thanks for your feedback!")
            if cols[1].button("👎 Not Helpful"):
                feedback = st.text_input("How could I improve?", key="feedback_input")
                if feedback:
                    log_feedback(user_input, bot_response, False, feedback)
                    st.success("Thanks for helping me improve!")

def log_feedback(question, response, was_helpful, improvement_suggestion=""):
    """Log user feedback for improvement"""
    timestamp = datetime.now().isoformat()
    feedback = {
        "timestamp": timestamp,
        "question": question,
        "response": response,
        "helpful": was_helpful,
        "suggestion": improvement_suggestion
    }
    
    try:
        with open("feedback_log.json", "a") as f:
            f.write(json.dumps(feedback) + "\n")
    except Exception as e:
        st.error(f"Couldn't save feedback: {str(e)}")

# For testing purposes
if __name__ == "__main__":
    chatbot()

