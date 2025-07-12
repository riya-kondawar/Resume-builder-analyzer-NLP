# import streamlit as st
# from fpdf.fpdf import FPDF  
# import os
# from datetime import datetime
# from PIL import Image
# from pathlib import Path
# import io
# import base64
# import re

# class UnicodePDF(FPDF):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # Add DejaVuSans font that supports Unicode
#         self.add_font('DejaVu', '', 'files/fonts/DejaVuSans.ttf', uni=True)
#         self.add_font('DejaVu', 'B', 'files/fonts/DejaVuSans-Bold.ttf', uni=True)
#         self.set_font('DejaVu', '', 12)

# def build_resume():
#     """Build Resume"""
#     st.title("Resume Builder")
#     st.markdown("### 📄 Create a Professional Resume like this")

#     image_url = "https://raw.githubusercontent.com/riya-kondawar/Resume-builder-analyzer-NLP/main/assets/test-pg1.png"
#     st.image(image_url, caption="Example of a generated resume", use_container_width=True)
    
#     st.write("Fill out the form below to generate your resume.")

#     # Initialize session state FIRST
#     if "resume_data" not in st.session_state:
#         st.session_state.resume_data = {
#             "name": "",
#             "email": "",
#             "phone": "",
#             "education": "",
#             "experience": "",
#             "skills": "",
#             "summary": "",
#             "linkedin": "",
#             "portfolio": "",
#             "address": "",
#             "certifications": "",
#             "projects": "",
#             "achievements": "",
#             "languages": "",
#             "volunteer": "",
#             "hobbies": "",
#             "references": "",
#             "photo_path": ""
#         }

#     st.header("🛠 Build Your Resume")

#     # Photo upload with validation
#     photo = st.file_uploader("Upload Profile Photo (Optional)", type=["jpg", "jpeg", "png"])
#     photo_path = ""
    
#     if photo:
#         try:
#             st.image(photo, width=150, caption="Profile Photo")
#             # Save photo temporarily
#             photo_path = f"temp_photo_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
#             img = Image.open(photo)
#             img.save(photo_path)
#             st.session_state.resume_data["photo_path"] = photo_path
#         except Exception as e:
#             st.error(f"Error loading image: {str(e)}")

#     # Input fields with better organization
#     with st.expander("Personal Information", expanded=True):
#         name = st.text_input("Full Name*", value=st.session_state.resume_data.get("name", ""))
#         email = st.text_input("Email Address*", value=st.session_state.resume_data.get("email", ""))
#         # email validation
#         if email and not re.match(r"[^@]+@[^@]+\.[^@]+", email):
#             st.warning("⚠️ Please enter a valid email address.")

#         phone = st.text_input("Phone Number", value=st.session_state.resume_data.get("phone", ""))        
#         linkedin = st.text_input("LinkedIn Profile", value=st.session_state.resume_data.get("linkedin", ""))

#         # linkedin = st.text_input("LinkedIn Profile", value=st.session_state.resume_data["linkedin"])
#         portfolio = st.text_input("Portfolio/Website", value=st.session_state.resume_data.get("portfolio", ""))
#         address = st.text_area("Address", value=st.session_state.resume_data.get("address", ""))
    
#     with st.expander("Professional Details"):
#         summary = st.text_area("Professional Summary*", value=st.session_state.resume_data.get("summary", ""))
#         experience = st.text_area("Work Experience*", value=st.session_state.resume_data.get("experience", ""))
#         education = st.text_area("Education*", value=st.session_state.resume_data.get("education", ""))
#         skills = st.text_area("Skills* (comma-separated)", value=st.session_state.resume_data.get("skills", ""))
    
#     with st.expander("Additional Information"):
#         certifications = st.text_area("Certifications & Courses", value=st.session_state.resume_data.get("certifications", ""))
#         projects = st.text_area("Projects", value=st.session_state.resume_data.get("projects", ""))
#         achievements = st.text_area("Achievements & Awards", value=st.session_state.resume_data.get("achievements", ""))
#         languages = st.text_area("Languages", value=st.session_state.resume_data.get("languages", ""))
#         volunteer = st.text_area("Volunteer Experience", value=st.session_state.resume_data.get("volunteer", ""))
#         hobbies = st.text_area("Hobbies & Interests", value=st.session_state.resume_data.get("hobbies", ""))
#         references = st.text_area("References", value=st.session_state.resume_data.get("references", ""))

#     # Update session state
#     st.session_state.resume_data.update({
#         "name": name,
#         "email": email,
#         "phone": phone,
#         "linkedin": linkedin,
#         "portfolio": portfolio,
#         "address": address,
#         "summary": summary,
#         "education": education,
#         "experience": experience,
#         "skills": skills,
#         "certifications": certifications,
#         "projects": projects,
#         "achievements": achievements,
#         "languages": languages,
#         "volunteer": volunteer,
#         "hobbies": hobbies,
#         "references": references
#     })

#     # Resume Template Selection
#     template = st.selectbox("Choose Resume Template", ["Classic", "Modern", "Minimal"])
    
#     # Generate Resume PDF with validation
#     if st.button("Generate Resume PDF"):
#         if not name or not email or not summary or not experience or not education or not skills:
#             st.error("Please fill in all required fields (marked with *)")
#         else:
#             pdf_file = None  # Initialize variable to avoid UnboundLocalError
#             try:
#                 pdf_file = generate_resume_pdf(st.session_state.resume_data, template)
#                 st.success("✅ Resume Generated Successfully!")
                
#                 with open(pdf_file, "rb") as f:
#                     st.download_button(
#                         "📥 Download Resume", 
#                         data=f, 
#                         file_name=f"resume_{name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf", 
#                         # file_name = f"Resume_{data['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
#                         mime="application/pdf"
#                     )
                
#                 # Clean up temporary files
#                 if pdf_file and os.path.exists(pdf_file):
#                     os.remove(pdf_file)
#                 try:
#                     if st.session_state.resume_data.get("photo_path") and os.path.exists(st.session_state.resume_data["photo_path"]):
#                         os.remove(st.session_state.resume_data["photo_path"])
#                 except Exception as e:
#                     st.warning(f"Could not remove photo: {e}")

#             except Exception as e:
#                 st.error(f"Error generating PDF: {str(e)}")
#                 if pdf_file and os.path.exists(pdf_file):
#                     os.remove(pdf_file)

# def generate_resume_pdf(data, template):
#     pdf = UnicodePDF()
#     pdf.add_page()
#     pdf.set_auto_page_break(auto=True, margin=15)
    
#     # Header with photo (if exists)
#     if data.get("photo_path") and os.path.exists(data["photo_path"]):
#         # Place photo on the right side
#         pdf.image(data["photo_path"], x=160, y=10, w=37.6, h=47)
    
#     # Name as title
#     pdf.set_font('DejaVu', 'B', 32)
#     pdf.cell(0, 10, data["name"], ln=True)
#     pdf.ln(5)
    
#     # Contact 
#     pdf.set_font('DejaVu', '', 10)
#     contact_info = []
#     if data.get("email"): contact_info.append(data["email"])
#     if data.get("phone"): contact_info.append(data["phone"])
#     if data.get("linkedin"): contact_info.append(data["linkedin"])
#     if data.get("portfolio"): contact_info.append(data["portfolio"])
#     if data.get("address"): contact_info.append(data["address"])
    
#     pdf.cell(0, 5, " | ".join(contact_info), ln=True)
#     pdf.ln(10)
    
#     # # Horizontal line
#     # pdf.set_draw_color(200, 200, 200)
#     # pdf.cell(0, 0, "", ln=True, border="T")
#     # pdf.ln(10)

#     # Horizontal line
#     # Get PDF width
#     # Horizontal line
#     pdf_width = pdf.w
#     margin = pdf.l_margin
#     start_x = margin
#     y = pdf.get_y()  # Current vertical position

#     # ✅ Conditional line length
#     if data.get("photo_path") and os.path.exists(data["photo_path"]):
#         line_width = pdf_width * 0.70  # shorter line if photo exists
#     else:
#         line_width = pdf_width - 2 * margin  # full width line

#     end_x = start_x + line_width

#     # Set line color and draw line
#     pdf.set_draw_color(200, 200, 200)
#     pdf.line(start_x, y, end_x, y)
#     pdf.ln(10)

#     # Helper function for sections
#     def add_section(title, content, is_list=False):
#         if content and content.strip():
#             # Section title
#             pdf.set_font('DejaVu', 'B', 12)
#             pdf.set_text_color(50, 50, 150)  # Dark blue color
#             pdf.cell(0, 8, title.upper(), ln=True)
#             pdf.set_text_color(0, 0, 0)  # Black color
            
#             # Section content
#             pdf.set_font('DejaVu', '', 10)
#             if is_list:
#                 items = [item.strip() for item in content.split(",")]
#                 for item in items:
#                     if item:  # Skip empty items
#                         pdf.cell(10)  # Indent
#                         pdf.cell(0, 6, "• " + item, ln=True)
#             else:
#                 # Replace any special dashes with regular hyphens
#                 clean_content = content.replace('–', '-').replace('—', '-')
#                 pdf.multi_cell(0, 6, clean_content)
#             pdf.ln(4)
    
#     # Add sections based on template
#     add_section("Professional Summary", data.get("summary", ""))
    
#     if template == "Classic":
#         add_section("Work Experience", data.get("experience", ""))
#         add_section("Education", data.get("education", ""))
#         add_section("Skills", data.get("skills", ""), is_list=True)
#         add_section("Certifications", data.get("certifications", ""))
#         add_section("Projects", data.get("projects", ""))
#         add_section("Achievements", data.get("achievements", ""))
#     elif template == "Modern":
#         add_section("Experience", data.get("experience", ""))
#         add_section("Education", data.get("education", ""))
#         add_section("Technical Skills", data.get("skills", ""), is_list=True)
#         if data.get("projects"): add_section("Key Projects", data.get("projects", ""))
#         if data.get("achievements"): add_section("Achievements", data.get("achievements", ""))
#     else:  # Minimal
#         add_section("Experience", data.get("experience", ""))
#         add_section("Education", data.get("education", ""))
#         add_section("Skills", data.get("skills", ""), is_list=True)
    
#     # Additional sections for all templates if data exists
#     if data.get("languages"): add_section("Languages", data.get("languages", ""), is_list=True)
#     if data.get("volunteer"): add_section("Volunteer Experience", data.get("volunteer", ""))
#     if data.get("hobbies"): add_section("Interests", data.get("hobbies", ""), is_list=True)
#     if data.get("references"): add_section("References", data.get("references", ""))

#     # Save PDF to temporary file
#     pdf_file = f"temp_resume_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
#     pdf.output(pdf_file)
#     return pdf_file





import streamlit as st
from fpdf import FPDF
import os
from datetime import datetime
from PIL import Image
import re
import tempfile
import base64
import contextlib

class UnicodePDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_font('helvetica', '', 12)  # Use built-in font

def handle_image_upload(uploaded_file):
    """Process uploaded image with proper format handling"""
    if not uploaded_file:
        return None
    
    try:
        img = Image.open(uploaded_file)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        # Create temp file with proper extension
        fd, temp_path = tempfile.mkstemp(suffix='.jpg')
        os.close(fd)  # Close the file descriptor immediately
        img.save(temp_path, quality=85)
        return temp_path
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def initialize_resume_data():
    """Ensure all resume data fields exist with empty defaults"""
    default_data = {
        "name": "", "email": "", "phone": "", "education": "",
        "experience": "", "skills": "", "summary": "", "linkedin": "",
        "portfolio": "", "address": "", "certifications": "", "projects": "",
        "achievements": "", "languages": "", "volunteer": "", "hobbies": "",
        "references": "", "photo_path": ""
    }
    
    if "resume_data" not in st.session_state:
        st.session_state.resume_data = default_data.copy()
    else:
        for key in default_data:
            if key not in st.session_state.resume_data:
                st.session_state.resume_data[key] = default_data[key]

def safe_remove_file(path):
    """Safely remove a file with error handling"""
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception as e:
            st.warning(f"Couldn't remove temporary file {path}: {str(e)}")

def generate_pdf(data, template):
    """Generate PDF from resume data with proper file handling"""
    pdf = UnicodePDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Header with photo
    if data.get("photo_path") and os.path.exists(data["photo_path"]):
        pdf.image(data["photo_path"], x=160, y=10, w=37.6, h=47)
    
    # Name and contact info
    pdf.set_font('helvetica', 'B', 24)
    pdf.cell(0, 10, data["name"], ln=True)
    pdf.set_font('helvetica', '', 10)
    
    contact_info = []
    for field in ["email", "phone", "linkedin", "portfolio", "address"]:
        if data.get(field):
            contact_info.append(data[field])
    pdf.cell(0, 5, " | ".join(contact_info), ln=True)
    pdf.ln(10)
    
    # Horizontal line
    pdf.set_draw_color(200, 200, 200)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(10)
    
    # Helper function for sections
    def add_section(title, content, is_list=False):
        if content:
            pdf.set_font('helvetica', 'B', 12)
            pdf.set_text_color(50, 50, 150)
            pdf.cell(0, 8, title.upper(), ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font('helvetica', '', 10)
            
            if is_list:
                for item in [x.strip() for x in content.split(",") if x.strip()]:
                    pdf.cell(10)
                    pdf.cell(0, 6, "- " + item, ln=True)  # Using hyphen instead of bullet
            else:
                pdf.multi_cell(0, 6, content.replace("–", "-"))
            pdf.ln(4)
    
    # Core sections
    add_section("Professional Summary", data.get("summary", ""))
    add_section("Work Experience", data.get("experience", ""))
    add_section("Education", data.get("education", ""))
    add_section("Skills", data.get("skills", ""), is_list=True)
    
    # Template-specific sections
    if template == "Classic":
        add_section("Certifications", data.get("certifications", ""))
        add_section("Projects", data.get("projects", ""))
        add_section("Achievements", data.get("achievements", ""))
    elif template == "Modern":
        if data.get("projects"): 
            add_section("Key Projects", data.get("projects", ""))
        if data.get("achievements"): 
            add_section("Notable Achievements", data.get("achievements", ""))
    
    # Optional sections
    if data.get("languages"): 
        add_section("Languages", data.get("languages", ""), is_list=True)
    if data.get("volunteer"): 
        add_section("Volunteer Experience", data.get("volunteer", ""))
    if data.get("hobbies"): 
        add_section("Interests", data.get("hobbies", ""), is_list=True)
    if data.get("references"): 
        add_section("References", data.get("references", ""))

    # Save to temp file with proper file handling
    fd, pdf_path = tempfile.mkstemp(suffix='.pdf')
    os.close(fd)  # Close the file descriptor immediately
    pdf.output(pdf_path)
    return pdf_path

def build_resume():
    """Main resume builder interface"""
    st.title("Resume Builder")
    st.markdown("### 📄 Create a Professional Resume")
    
    initialize_resume_data()
    
    image_url = "https://raw.githubusercontent.com/riya-kondawar/Resume-builder-analyzer-NLP/main/assets/test-pg1.png"
    st.image(image_url, caption="Example of a generated resume", use_container_width=True)
    # st.image("https://via.placeholder.com/800x600?text=Resume+Example",
    #         caption="Example Resume", use_container_width=True)
    
    # Photo upload
    photo = st.file_uploader("Upload Profile Photo (Optional)", 
                           type=["jpg", "jpeg", "png"])
    if photo:
        st.image(photo, width=150, caption="Your Photo")
        st.session_state.resume_data["photo_path"] = handle_image_upload(photo)

    # Input sections
    with st.expander("Personal Information", expanded=True):
        cols = st.columns(2)
        with cols[0]:
            st.session_state.resume_data["name"] = st.text_input(
                "Full Name*", 
                st.session_state.resume_data["name"]
            )
            st.session_state.resume_data["email"] = st.text_input(
                "Email*", 
                st.session_state.resume_data["email"]
            )
            st.session_state.resume_data["phone"] = st.text_input(
                "Phone", 
                st.session_state.resume_data["phone"]
            )
        with cols[1]:
            st.session_state.resume_data["linkedin"] = st.text_input(
                "LinkedIn", 
                st.session_state.resume_data["linkedin"]
            )
            st.session_state.resume_data["portfolio"] = st.text_input(
                "Portfolio", 
                st.session_state.resume_data["portfolio"]
            )
        st.session_state.resume_data["address"] = st.text_area(
            "Address", 
            st.session_state.resume_data["address"]
        )
        
        if (st.session_state.resume_data["email"] and 
            not re.match(r"[^@]+@[^@]+\.[^@]+", st.session_state.resume_data["email"])):
            st.warning("Please enter a valid email address")

    with st.expander("Professional Details"):
        st.session_state.resume_data["summary"] = st.text_area(
            "Professional Summary*", 
            st.session_state.resume_data["summary"],
            height=100
        )
        st.session_state.resume_data["experience"] = st.text_area(
            "Work Experience*", 
            st.session_state.resume_data["experience"],
            height=150
        )
        st.session_state.resume_data["education"] = st.text_area(
            "Education*", 
            st.session_state.resume_data["education"],
            height=100
        )
        st.session_state.resume_data["skills"] = st.text_area(
            "Skills* (comma separated)", 
            st.session_state.resume_data["skills"],
            height=100
        )

    with st.expander("Additional Information"):
        cols = st.columns(2)
        with cols[0]:
            st.session_state.resume_data["certifications"] = st.text_area(
                "Certifications", 
                st.session_state.resume_data["certifications"]
            )
            st.session_state.resume_data["projects"] = st.text_area(
                "Projects", 
                st.session_state.resume_data["projects"]
            )
        with cols[1]:
            st.session_state.resume_data["achievements"] = st.text_area(
                "Achievements", 
                st.session_state.resume_data["achievements"]
            )
            st.session_state.resume_data["languages"] = st.text_area(
                "Languages", 
                st.session_state.resume_data["languages"]
            )
        st.session_state.resume_data["volunteer"] = st.text_area(
            "Volunteer Experience", 
            st.session_state.resume_data["volunteer"]
        )
        st.session_state.resume_data["hobbies"] = st.text_area(
            "Hobbies & Interests", 
            st.session_state.resume_data["hobbies"]
        )
        st.session_state.resume_data["references"] = st.text_area(
            "References", 
            st.session_state.resume_data["references"]
        )

    # Template selection
    template = st.selectbox("Choose Template", ["Classic", "Modern", "Minimal"])

    # Generate PDF button
    if st.button("✨ Generate PDF Resume"):
        required_fields = ["name", "email", "summary", "experience", "education", "skills"]
        missing_fields = [field for field in required_fields 
                         if not st.session_state.resume_data[field]]
        
        if missing_fields:
            st.error(f"Please fill all required fields (*): {', '.join(missing_fields)}")
        else:
            with st.spinner("Generating your resume..."):
                try:
                    pdf_path = generate_pdf(st.session_state.resume_data, template)
                    
                    # Read PDF content and create download link
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    
                    b64 = base64.b64encode(pdf_bytes).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="resume_{st.session_state.resume_data["name"].replace(" ", "_")}.pdf">Download Resume</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    # Clean up temporary files
                    safe_remove_file(pdf_path)
                    safe_remove_file(st.session_state.resume_data.get("photo_path"))
                    
                    st.success("✅ Resume generated successfully!")
                except Exception as e:
                    st.error(f"Failed to generate PDF: {str(e)}")
                    safe_remove_file(st.session_state.resume_data.get("photo_path"))

# Clean up any remaining temp files when the script ends
import atexit
atexit.register(lambda: safe_remove_file(st.session_state.get("resume_data", {}).get("photo_path")))