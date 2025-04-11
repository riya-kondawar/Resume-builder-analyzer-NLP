import streamlit as st
from fpdf.fpdf import FPDF  
import os
from datetime import datetime
from PIL import Image
from pathlib import Path
import io
import base64
import re

# from files.pages.templates.classic import render_classic_resume
# from files.pages.templates.modern import render_modern_resume
# from files.pages.templates.minimal import render_minimal_resume 

class UnicodePDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add DejaVuSans font that supports Unicode
        self.add_font('DejaVu', '', 'files/fonts/DejaVuSans.ttf', uni=True)
        self.add_font('DejaVu', 'B', 'files/fonts/DejaVuSans-Bold.ttf', uni=True)
        self.set_font('DejaVu', '', 12)

def build_resume():
    """Build Resume"""
    st.title("Resume Builder")
    st.markdown("### 📄 Create a Professional Resume like this")

    image_url = "https://raw.githubusercontent.com/riya-kondawar/Resume-builder-analyzer-NLP/main/assets/test-pg1.png"
    st.image(image_url, caption="Example of a generated resume", use_container_width=True)
    
    st.write("Fill out the form below to generate your resume.")


    # Initialize session state
    if "resume_data" not in st.session_state:
        st.session_state.resume_data = {
            "name": "",
            "email": "",
            "phone": "",
            "education": "",
            "experience": "",
            "skills": "",
            "summary": "",
            "linkedin": "",
            "portfolio": "",
            "address": "",
            "certifications": "",
            "projects": "",
            "achievements": "",
            "languages": "",
            "volunteer": "",
            "hobbies": "",
            "references": "",
            "photo_path": ""
        }

    st.header("🛠 Build Your Resume")

    # Photo upload with validation
    photo = st.file_uploader("Upload Profile Photo (Optional)", type=["jpg", "jpeg", "png"])
    photo_path = ""
    
    if photo:
        try:
            st.image(photo, width=150, caption="Profile Photo")
            # Save photo temporarily
            photo_path = f"temp_photo_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            img = Image.open(photo)
            img.save(photo_path)
            st.session_state.resume_data["photo_path"] = photo_path
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")

    # Input fields with better organization
    with st.expander("Personal Information", expanded=True):
        name = st.text_input("Full Name*", value=st.session_state.resume_data.get("name", ""))
        email = st.text_input("Email Address*", value=st.session_state.resume_data.get("email", ""))
        # email validation
        if email and not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            st.warning("⚠️ Please enter a valid email address.")

        phone = st.text_input("Phone Number", value=st.session_state.resume_data["phone"])
        linkedin = st.text_input("LinkedIn Profile", value=st.session_state.resume_data["linkedin"])
        portfolio = st.text_input("Portfolio/Website", value=st.session_state.resume_data["portfolio"])
        address = st.text_area("Address", value=st.session_state.resume_data["address"])
    
    with st.expander("Professional Details"):
        summary = st.text_area("Professional Summary*", value=st.session_state.resume_data["summary"])
        experience = st.text_area("Work Experience*", value=st.session_state.resume_data["experience"])
        education = st.text_area("Education*", value=st.session_state.resume_data["education"])
        skills = st.text_area("Skills* (comma-separated)", value=st.session_state.resume_data["skills"])
    
    with st.expander("Additional Information"):
        certifications = st.text_area("Certifications & Courses", value=st.session_state.resume_data["certifications"])
        projects = st.text_area("Projects", value=st.session_state.resume_data["projects"])
        achievements = st.text_area("Achievements & Awards", value=st.session_state.resume_data["achievements"])
        languages = st.text_area("Languages", value=st.session_state.resume_data["languages"])
        volunteer = st.text_area("Volunteer Experience", value=st.session_state.resume_data["volunteer"])
        hobbies = st.text_area("Hobbies & Interests", value=st.session_state.resume_data["hobbies"])
        references = st.text_area("References", value=st.session_state.resume_data["references"])

    # Update session state
    st.session_state.resume_data.update({
        "name": name,
        "email": email,
        "phone": phone,
        "linkedin": linkedin,
        "portfolio": portfolio,
        "address": address,
        "summary": summary,
        "education": education,
        "experience": experience,
        "skills": skills,
        "certifications": certifications,
        "projects": projects,
        "achievements": achievements,
        "languages": languages,
        "volunteer": volunteer,
        "hobbies": hobbies,
        "references": references
    })

    # Resume Template Selection
    template = st.selectbox("Choose Resume Template", ["Classic", "Modern", "Minimal"])
    
    # Generate Resume PDF with validation
    if st.button("Generate Resume PDF"):
        if not name or not email or not summary or not experience or not education or not skills:
            st.error("Please fill in all required fields (marked with *)")
        else:
            pdf_file = None  # Initialize variable to avoid UnboundLocalError
            try:
                pdf_file = generate_resume_pdf(st.session_state.resume_data, template)
                st.success("✅ Resume Generated Successfully!")
                
                with open(pdf_file, "rb") as f:
                    st.download_button(
                        "📥 Download Resume", 
                        data=f, 
                        file_name=f"resume_{name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf", 
                        # file_name = f"Resume_{data['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
                        mime="application/pdf"
                    )
                
                # Clean up temporary files
                if pdf_file and os.path.exists(pdf_file):
                    os.remove(pdf_file)
                try:
                    if st.session_state.resume_data.get("photo_path") and os.path.exists(st.session_state.resume_data["photo_path"]):
                        os.remove(st.session_state.resume_data["photo_path"])
                except Exception as e:
                    st.warning(f"Could not remove photo: {e}")

            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
                if pdf_file and os.path.exists(pdf_file):
                    os.remove(pdf_file)

def generate_resume_pdf(data, template):
    pdf = UnicodePDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Header with photo (if exists)
    if data.get("photo_path") and os.path.exists(data["photo_path"]):
        # Place photo on the right side
        pdf.image(data["photo_path"], x=160, y=10, w=37.6, h=47)
    
    # Name as title
    pdf.set_font('DejaVu', 'B', 32)
    pdf.cell(0, 10, data["name"], ln=True)
    pdf.ln(5)
    
    # Contact 
    pdf.set_font('DejaVu', '', 10)
    contact_info = []
    if data.get("email"): contact_info.append(data["email"])
    if data.get("phone"): contact_info.append(data["phone"])
    if data.get("linkedin"): contact_info.append(data["linkedin"])
    if data.get("portfolio"): contact_info.append(data["portfolio"])
    if data.get("address"): contact_info.append(data["address"])
    
    pdf.cell(0, 5, " | ".join(contact_info), ln=True)
    pdf.ln(10)
    
    # # Horizontal line
    # pdf.set_draw_color(200, 200, 200)
    # pdf.cell(0, 0, "", ln=True, border="T")
    # pdf.ln(10)

    # Horizontal line
    # Get PDF width
    # Horizontal line
    pdf_width = pdf.w
    margin = pdf.l_margin
    start_x = margin
    y = pdf.get_y()  # Current vertical position

    # ✅ Conditional line length
    if data.get("photo_path") and os.path.exists(data["photo_path"]):
        line_width = pdf_width * 0.70  # shorter line if photo exists
    else:
        line_width = pdf_width - 2 * margin  # full width line

    end_x = start_x + line_width

    # Set line color and draw line
    pdf.set_draw_color(200, 200, 200)
    pdf.line(start_x, y, end_x, y)
    pdf.ln(10)

    # Helper function for sections
    def add_section(title, content, is_list=False):
        if content and content.strip():
            # Section title
            pdf.set_font('DejaVu', 'B', 12)
            pdf.set_text_color(50, 50, 150)  # Dark blue color
            pdf.cell(0, 8, title.upper(), ln=True)
            pdf.set_text_color(0, 0, 0)  # Black color
            
            # Section content
            pdf.set_font('DejaVu', '', 10)
            if is_list:
                items = [item.strip() for item in content.split(",")]
                for item in items:
                    if item:  # Skip empty items
                        pdf.cell(10)  # Indent
                        pdf.cell(0, 6, "• " + item, ln=True)
            else:
                # Replace any special dashes with regular hyphens
                clean_content = content.replace('–', '-').replace('—', '-')
                pdf.multi_cell(0, 6, clean_content)
            pdf.ln(4)
    
    # Add sections based on template
    add_section("Professional Summary", data.get("summary", ""))
    
    if template == "Classic":
        add_section("Work Experience", data.get("experience", ""))
        add_section("Education", data.get("education", ""))
        add_section("Skills", data.get("skills", ""), is_list=True)
        add_section("Certifications", data.get("certifications", ""))
        add_section("Projects", data.get("projects", ""))
        add_section("Achievements", data.get("achievements", ""))
    elif template == "Modern":
        add_section("Experience", data.get("experience", ""))
        add_section("Education", data.get("education", ""))
        add_section("Technical Skills", data.get("skills", ""), is_list=True)
        if data.get("projects"): add_section("Key Projects", data.get("projects", ""))
        if data.get("achievements"): add_section("Achievements", data.get("achievements", ""))
    else:  # Minimal
        add_section("Experience", data.get("experience", ""))
        add_section("Education", data.get("education", ""))
        add_section("Skills", data.get("skills", ""), is_list=True)
    
    # Additional sections for all templates if data exists
    if data.get("languages"): add_section("Languages", data.get("languages", ""), is_list=True)
    if data.get("volunteer"): add_section("Volunteer Experience", data.get("volunteer", ""))
    if data.get("hobbies"): add_section("Interests", data.get("hobbies", ""), is_list=True)
    if data.get("references"): add_section("References", data.get("references", ""))

    # Save PDF to temporary file
    pdf_file = f"temp_resume_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    pdf.output(pdf_file)
    return pdf_file

