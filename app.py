import streamlit as st
import os
import tempfile
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import time

# Configure page
st.set_page_config(
    page_title="Medical Q&A Assistant", 
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None

def initialize_llm():
    """Initialize LiteLLM with configuration"""
    try:
        api_key = st.secrets["LITELLM_API_KEY"]
        base_url = st.secrets["LITELLM_BASE_URL"] 
        model = st.secrets["LITELLM_MODEL"]
        
        os.environ['LITELLM_API_KEY'] = api_key
        os.environ['LITELLM_BASE_URL'] = base_url
        
        llm = ChatLiteLLM(
            model=model,
            api_base=base_url,
            api_key=api_key,
            temperature=0.1,
            max_tokens=1000
        )
        
        return llm
    except Exception as e:
        st.error(f"Failed to initialize LLM: {str(e)}")
        return None

@st.cache_resource
def load_embeddings():
    """Load sentence transformer embeddings"""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_pdf(uploaded_file):
    """Process uploaded PDF and create knowledge base"""
    try:
        with st.spinner("Processing PDF document..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Load PDF
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create embeddings and vector store
            embeddings = load_embeddings()
            knowledge_base = FAISS.from_documents(chunks, embeddings)
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            return knowledge_base
            
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def classify_user_context(user_type, urgency):
    """Classify user context and urgency level"""
    context_map = {
        "Healthcare Professional": "medical_professional",
        "Patient": "patient",
        "Student": "student"
    }
    
    urgency_map = {
        "Low": "routine_inquiry",
        "Medium": "standard_consultation", 
        "High": "urgent_medical_concern"
    }
    
    return context_map.get(user_type, "patient"), urgency_map.get(urgency, "routine_inquiry")

def create_medical_prompt(user_context, urgency_level, has_knowledge_base=False):
    """Create context-aware medical prompt"""
    
    if user_context == "medical_professional":
        base_prompt = """You are a medical information assistant designed to support healthcare professionals. 
        Provide evidence-based, detailed medical information with clinical context."""
    else:
        base_prompt = """You are a medical information assistant designed to provide general health information to patients.
        Use clear, understandable language. Always emphasize the importance of consulting healthcare professionals."""
    
    if urgency_level == "urgent_medical_concern":
        urgency_note = "\n\nIMPORTANT: This appears to be an urgent medical concern. Strongly recommend immediate medical consultation."
    else:
        urgency_note = ""
    
    if has_knowledge_base:
        context_instruction = """
        Answer the question based on the provided context from the uploaded document. 
        If the context doesn't contain relevant information, clearly state that the information is not available in the document.
        
        Context: {context}
        
        Question: {question}"""
    else:
        context_instruction = """
        Answer the medical question based on your general medical knowledge. 
        Provide accurate, evidence-based information.
        
        Question: {question}"""
    
    full_prompt = base_prompt + urgency_note + "\n\n" + context_instruction
    
    return ChatPromptTemplate.from_template(full_prompt)

def generate_response(query, user_type, urgency):
    """Generate response with or without knowledge base"""
    
    llm = initialize_llm()
    if not llm:
        return None
    
    user_context, urgency_level = classify_user_context(user_type, urgency)
    
    try:
        if st.session_state.knowledge_base:
            # Use knowledge base for retrieval
            prompt_template = create_medical_prompt(user_context, urgency_level, has_knowledge_base=True)
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state.knowledge_base.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": prompt_template}
            )
            
            response = qa_chain.run(query)
            
        else:
            # Use LLM directly without knowledge base
            prompt_template = create_medical_prompt(user_context, urgency_level, has_knowledge_base=False)
            formatted_prompt = prompt_template.format_messages(question=query)
            response = llm.invoke(formatted_prompt)
            response = response.content
        
        return response
        
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None

def add_medical_disclaimers(urgency_level):
    """Add appropriate medical disclaimers"""
    
    base_disclaimer = """
    **⚠️ Medical Disclaimer:**
    This information is for educational purposes only and should not replace professional medical advice. 
    Always consult with qualified healthcare professionals for medical concerns.
    """
    
    if urgency_level == "urgent_medical_concern":
        urgent_disclaimer = """
        **🚨 URGENT NOTICE:**
        This appears to be a potentially urgent medical situation. 
        Please seek immediate medical attention or contact emergency services.
        """
        return urgent_disclaimer + base_disclaimer
    
    return base_disclaimer

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🏥 Medical Q&A Assistant")
    st.markdown("*Chat with medical documents or get general medical information*")
    
    # Sidebar for document upload and user context
    with st.sidebar:
        st.header("📄 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Upload a medical document (PDF)",
            type="pdf",
            help="Upload a medical document to chat with its content"
        )
        
        if uploaded_file is not None:
            if st.session_state.uploaded_file_name != uploaded_file.name:
                st.session_state.knowledge_base = process_pdf(uploaded_file)
                st.session_state.uploaded_file_name = uploaded_file.name
                
                if st.session_state.knowledge_base:
                    st.success(f"✅ Document '{uploaded_file.name}' processed successfully!")
                else:
                    st.error("❌ Failed to process document")
        
        if st.button("Clear Document"):
            st.session_state.knowledge_base = None
            st.session_state.uploaded_file_name = None
            st.success("Document cleared!")
        
        st.markdown("---")
        st.header("User Context")
        
        user_type = st.selectbox(
            "I am a:",
            ["Patient", "Healthcare Professional", "Student"],
            help="This helps tailor the response appropriately"
        )
        
        urgency = st.selectbox(
            "Urgency Level:",
            ["Low", "Medium", "High"],
            help="How urgent is your medical question?"
        )
        
        # Show current status
        st.markdown("---")
        st.header("Current Status")
        if st.session_state.knowledge_base:
            st.success(f"📄 Document loaded: {st.session_state.uploaded_file_name}")
            st.info("Answers will be based on the uploaded document")
        else:
            st.info("💡 No document loaded - answers will be based on general medical knowledge")
    
    # Main chat interface
    st.subheader("💬 Ask Your Medical Question")
    
    query = st.text_area(
        "Enter your medical question:",
        height=100,
        placeholder="e.g., What are the symptoms of diabetes? How does aspirin work? What does this test result mean?"
    )
    
    if st.button("Get Answer", type="primary") and query.strip():
        
        with st.spinner("🔍 Generating response..."):
            response = generate_response(query, user_type, urgency)
            
            if response:
                st.markdown("---")
                st.subheader("📋 Answer")
                
                # Display response
                st.write(response)
                
                # Add disclaimers
                user_context, urgency_level = classify_user_context(user_type, urgency)
                disclaimer = add_medical_disclaimers(urgency_level)
                st.markdown(disclaimer)
                
                # Show source information
                if st.session_state.knowledge_base:
                    st.info(f"💡 Answer based on uploaded document: {st.session_state.uploaded_file_name}")
                else:
                    st.info("💡 Answer based on general medical knowledge")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <small>Medical Q&A Assistant | Upload documents for specific information or ask general medical questions</small>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
