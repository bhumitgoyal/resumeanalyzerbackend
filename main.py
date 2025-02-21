from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
import tempfile
import pdfplumber
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

openai_api_key = os.getenv("OPENAI_API_KEY")
chat_model = ChatOpenAI(model="gpt-4-0613", temperature=0.7, api_key=openai_api_key)

# Set max file size (e.g., 5MB)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB


def extract_text(pdf_file):
    """Extract text from a PDF using PyPDFLoader, with pdfplumber fallback."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            pdf_file.save(temp_pdf.name)

            text = ""
            try:
                loader = PyPDFLoader(temp_pdf.name)
                pages = loader.load()
                text = "\n".join([page.page_content for page in pages])
            except Exception:
                pass  # PyPDFLoader failed, trying pdfplumber

            if not text.strip():
                with pdfplumber.open(temp_pdf.name) as pdf:
                    text = "\n".join([page.extract_text() or "" for page in pdf.pages[:10]])  # Limit pages

        os.remove(temp_pdf.name)

        if not text.strip():
            return None, "Extracted text is empty. Ensure the PDF is not scanned."

        return text, None
    except Exception as e:
        return None, str(e)


def analyze_resume(resume_text, job_application=""):
    """Analyze resume using GPT-4 and return a structured evaluation as JSON."""
    try:
        messages = [
            SystemMessage(content="""You are an experienced HR professional specializing in resume analysis.
            Your task is to evaluate resumes and provide structured feedback in valid JSON format. **Only return the JSON output, without any additional text.** 
            
            ### **Evaluation Criteria:**  
            1. **Strengths**: Notable skills, achievements, and unique aspects.  
            2. **Areas for Improvement**: Weak points, missing details, formatting issues.  
            3. **Missing Skills & Qualifications**: Compare with job requirements if provided.  
            4. **Technical & Soft Skills Advice**: Relevant certifications or skills.  
            5. **Formatting & Readability**: Clarity, conciseness, and structure.  
            6. **Overall Score**: Provide a score out of 10 based on industry standards.
            
            ### **Response Format (JSON)**:
            ```json
            {
              "strengths": ["List of strong points"],
              "areas_to_improve": ["List of weaknesses"],
              "missing_skills": ["List of missing skills"],
              "suggested_enhancements": ["List of suggestions"],
              "overall_score": 7
            }
            ```
            Only return valid JSON output, no additional explanations.
            """),
            HumanMessage(content=f"""### Candidate Resume Analysis
            Resume Content: {resume_text}
            
            Job Application (if provided): {job_application}
            
            Provide a structured response in valid JSON format only.""")
        ]
        
        response = chat_model.invoke(messages).content

        try:
            return json.loads(response), None
        except json.JSONDecodeError:
            return None, f"Invalid JSON response: {response}"
    except Exception as e:
        return None, str(e)


@app.route('/upload', methods=['POST'])
def upload_resume():
    """API Endpoint to upload and analyze resumes."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        job_application = request.form.get('job_application', "")

        resume_text, error = extract_text(file)
        if error:
            return jsonify({"error": error}), 400

        analysis_result, error = analyze_resume(resume_text, job_application)
        if error:
            return jsonify({"error": error}), 500

        return jsonify(analysis_result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run()
