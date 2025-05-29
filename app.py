import streamlit as st
import os
from langchain_litellm import ChatLiteLLM
from langchain.prompts import ChatPromptTemplate
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
import time
import warnings
import PyPDF2
import docx
from io import BytesIO

# Set USER_AGENT to fix warnings
os.environ['USER_AGENT'] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Suppress warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="Medical Q&A Assistant",
    page_icon="🏥",
    layout="wide"
)

def setup_llm():
    """Setup LiteLLM with medical expert system prompt"""
    return ChatLiteLLM(
        model=st.secrets["LITELLM_MODEL"],
        api_base=st.secrets["LITELLM_BASE_URL"],
        api_key=st.secrets["LITELLM_API_KEY"],
        temperature=0.1
    )

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def search_medical_web(query, max_results=5):
    """Search for current medical information using DuckDuckGo"""
    try:
        with DDGS() as ddgs:
            # Search trusted medical sites
            search_query = f"{query} site:mayoclinic.org OR site:webmd.org OR site:healthline.com OR site:medlineplus.gov OR site:nih.gov OR site:who.int"
            
            results = []
            for result in ddgs.text(search_query, max_results=max_results):
                results.append({
                    'title': result.get('title', ''),
                    'body': result.get('body', ''),
                    'href': result.get('href', '')
                })
            
            return results
    except Exception as e:
        return []

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_docx(docx_file):
    """Extract text from uploaded DOCX file"""
    try:
        doc = docx.Document(BytesIO(docx_file.read()))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return ""

def process_uploaded_files(uploaded_files):
    """Process uploaded knowledge base files"""
    knowledge_content = ""
    
    for uploaded_file in uploaded_files:
        file_type = uploaded_file.type
        
        if file_type == "application/pdf":
            content = extract_text_from_pdf(uploaded_file)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            content = extract_text_from_docx(uploaded_file)
        elif file_type == "text/plain":
            content = str(uploaded_file.read(), "utf-8")
        else:
            st.warning(f"Unsupported file type: {uploaded_file.name}")
            continue
        
        if content:
            knowledge_content += f"\n--- Content from {uploaded_file.name} ---\n{content}\n"
    
    return knowledge_content

def create_medical_expert_prompt(user_type, search_results=None, custom_knowledge=None):
    """Create system prompt that makes LLM think like a medical expert"""
    
    if user_type == "Healthcare Professional":
        system_prompt = """You are a highly experienced medical doctor and clinical expert with extensive knowledge across all medical specialties. You have access to the latest medical research, clinical guidelines, and evidence-based practices.

Your expertise includes:
- Internal Medicine, Cardiology, Endocrinology, Neurology, Psychiatry, Oncology
- Emergency Medicine, Surgery, Pediatrics, Geriatrics, Infectious Diseases
- Pharmacology, Pathology, Radiology, Laboratory Medicine
- Current clinical guidelines (AHA, ACC, ADA, WHO, CDC, NICE)
- Latest medical research and evidence-based medicine

When responding to healthcare professionals:
1. Provide detailed clinical information with differential diagnoses
2. Include relevant clinical guidelines and evidence levels
3. Mention contraindications, drug interactions, and monitoring parameters
4. Use appropriate medical terminology
5. Suggest further workup or specialist referral when appropriate
6. Always emphasize clinical judgment and patient-specific factors

Remember: Support clinical decision-making but never replace direct patient assessment."""

    else:
        system_prompt = """You are a compassionate and knowledgeable medical doctor who specializes in patient education and communication. You excel at explaining complex medical concepts in simple, understandable terms while maintaining medical accuracy.

Your approach with patients:
1. Use clear, non-technical language that patients can understand
2. Provide practical, actionable health guidance
3. Explain when and why to seek medical care
4. Address common concerns and misconceptions
5. Emphasize the importance of professional medical consultation
6. Be empathetic and supportive while being medically accurate

Always include appropriate disclaimers about the need for professional medical evaluation and never attempt to diagnose or prescribe treatments."""

    # Add custom knowledge if available
    if custom_knowledge:
        system_prompt += f"\n\nAdditional Knowledge Base Available:\n{custom_knowledge[:2000]}..."

    # Add search context if available
    if search_results:
        context_info = "\n\nCurrent Medical Information from Trusted Sources:\n"
        for i, result in enumerate(search_results[:3], 1):
            context_info += f"{i}. {result.get('title', 'Medical Information')}: {result.get('body', '')}\n"
        system_prompt += context_info

    return system_prompt

def generate_medical_response(question, user_type, enable_web_search=True, custom_knowledge=None):
    """Generate medical response using expert system prompt and available knowledge"""
    start_time = time.time()
    
    # Search for current medical information
    search_results = []
    if enable_web_search:
        with st.spinner("🔍 Searching current medical literature..."):
            search_results = search_medical_web(question)
    
    # Setup LLM with medical expert system prompt
    llm = setup_llm()
    system_prompt = create_medical_expert_prompt(user_type, search_results, custom_knowledge)
    
    # Create the prompt template
    if user_type == "Healthcare Professional":
        human_template = """Medical Query: {question}

As a medical expert, provide a comprehensive clinical response that includes:
1. Clinical assessment and differential diagnoses
2. Evidence-based recommendations and guidelines
3. Monitoring parameters and contraindications
4. When to consider specialist referral

Maintain professional medical standards while supporting clinical decision-making."""

    else:
        human_template = """Health Question: {question}

As a medical doctor, provide a clear and helpful response that includes:
1. Easy-to-understand explanation
2. Practical health guidance
3. When to seek medical care
4. Important safety considerations

Use simple language while maintaining medical accuracy. Always emphasize the importance of consulting healthcare professionals for proper medical care."""

    # Create chat prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_template)
    ])
    
    # Generate response
    with st.spinner("🧠 Analyzing medical information..."):
        try:
            chain = prompt | llm
            response = chain.invoke({"question": question})
            
            end_time = time.time()
            response_time = round((end_time - start_time) * 1000)
            
            return response.content, len(search_results), response_time, bool(custom_knowledge)
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return None, 0, 0, False

def main():
    # Header
    st.title("🏥 Medical Q&A Assistant")
    st.markdown("*AI-powered medical expert system with optional knowledge base*")
    
    # Sidebar for settings and knowledge base
    with st.sidebar:
        st.header("👤 User Information")
        
        user_type = st.selectbox(
            "I am a:",
            ["Patient/General Public", "Healthcare Professional"],
            help="This helps tailor the response appropriately"
        )
        
        st.markdown("---")
        st.header("📚 Knowledge Base (Optional)")
        
        uploaded_files = st.file_uploader(
            "Upload your medical documents:",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload medical documents, research papers, or guidelines to enhance responses"
        )
        
        custom_knowledge = None
        if uploaded_files:
            with st.spinner("📄 Processing uploaded documents..."):
                custom_knowledge = process_uploaded_files(uploaded_files)
                if custom_knowledge:
                    st.success(f"✅ Processed {len(uploaded_files)} document(s)")
                    st.info(f"Knowledge base: {len(custom_knowledge.split())} words")
        
        st.markdown("---")
        st.header("⚙️ Search Settings")
        
        enable_web_search = st.toggle(
            "🌐 Enable Web Search",
            value=True,
            help="Search current medical information from trusted websites"
        )
        
        if enable_web_search:
            st.success("✅ Web search enabled")
            st.caption("Sources: Mayo Clinic, WebMD, Healthline, MedlinePlus, NIH, WHO")
        else:
            st.info("📚 Using medical expert knowledge only")
        
        st.markdown("---")
        st.markdown("### ⚠️ Important Notice")
        st.error(
            "🚨 **EMERGENCY:** For medical emergencies, call emergency services immediately. "
            "This AI assistant provides general information only and does not replace professional medical care."
        )
    
    # Main interface
    st.markdown("### 💬 Ask Your Medical Question")
    
    question = st.text_area(
        "Enter your medical question:",
        placeholder="e.g., What are the latest treatment guidelines for type 2 diabetes management?",
        height=100
    )
    
    if st.button("🔍 Get Medical Information", type="primary"):
        if not question.strip():
            st.warning("Please enter a medical question.")
            return
        
        # Generate response
        result, web_sources, response_time, has_custom_kb = generate_medical_response(
            question, user_type, enable_web_search, custom_knowledge
        )
        
        if result:
            # Display response
            st.markdown("### 📋 Medical Information")
            
            # Response metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⚡ Response Time", f"{response_time}ms")
            with col2:
                if web_sources > 0:
                    st.metric("🌐 Web Sources", web_sources)
                else:
                    st.metric("🧠 Expert Knowledge", "✓")
            with col3:
                if has_custom_kb:
                    st.metric("📚 Custom KB", "✓")
                else:
                    st.metric("📚 Custom KB", "None")
            
            # Main answer
            st.markdown("**Medical Response:**")
            st.write(result)
            
            # Information sources
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Information Sources:**")
                sources = []
                if has_custom_kb:
                    sources.append("📚 Your uploaded documents")
                if web_sources > 0:
                    sources.append(f"🌐 {web_sources} current medical websites")
                sources.append("🧠 Medical expert knowledge")
                
                for source in sources:
                    st.success(source)
                
            with col2:
                st.markdown("**👤 Response Type:**")
                st.info(f"Tailored for: {user_type}")
                
                if user_type == "Healthcare Professional":
                    st.caption("Clinical-grade response with detailed medical information")
                else:
                    st.caption("Patient-friendly explanation with clear guidance")
            
            # Final comprehensive disclaimer
            st.markdown("---")
            if user_type == "Healthcare Professional":
                st.warning(
                    "⚠️ **CLINICAL DISCLAIMER:** This AI assistant provides medical information to support clinical decision-making. "
                    "It does not replace clinical judgment, direct patient assessment, or consultation with specialists. "
                    "Always consider individual patient factors, contraindications, and current clinical guidelines. "
                    "Verify critical information with authoritative medical sources."
                )
            else:
                st.error(
                    "⚠️ **MEDICAL DISCLAIMER:** This AI assistant provides general medical information for educational purposes only. "
                    "It does not constitute professional medical advice, diagnosis, or treatment recommendations. "
                    "Always consult qualified healthcare professionals for medical concerns, treatment decisions, and health management. "
                    "In case of medical emergencies, contact emergency services immediately."
                )

if __name__ == "__main__":
    main()
