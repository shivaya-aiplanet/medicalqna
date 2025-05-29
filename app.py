import streamlit as st
import os
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Langchain imports
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

# Page configuration
st.set_page_config(
    page_title="🏥 MedAssist AI - Your Medical Q&A Companion",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #A23B72;
        --accent-color: #F18F01;
        --success-color: #C73E1D;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: #E8F4FD;
        font-size: 1.2rem;
        margin: 0;
    }
    
    /* Chat interface styling */
    .chat-container {
        background: linear-gradient(145deg, #f0f8ff, #ffffff);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
        float: right;
        clear: both;
        max-width: 80%;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #ffecd2, #fcb69f);
        color: #333;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(252, 182, 159, 0.3);
        float: left;
        clear: both;
        max-width: 80%;
    }
    
    /* Metrics cards */
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f8fafc);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border-left: 4px solid var(--primary-color);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #64748b;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        margin-bottom: 1rem;
    }
    
    /* Status indicators */
    .status-ready {
        background: linear-gradient(135deg, #56ab2f, #a8e6cf);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        text-align: center;
    }
    
    .status-processing {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        text-align: center;
    }
    
    /* File upload area */
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(145deg, #f8fafc, #ffffff);
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #764ba2;
        background: linear-gradient(145deg, #ffffff, #f8fafc);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Animation keyframes */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

class MedicalQAApp:
    def __init__(self):
        self.initialize_session_state()
        self.setup_azure_llm()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'vectorstore' not in st.session_state:
            st.session_state.vectorstore = None
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
        if 'total_chunks' not in st.session_state:
            st.session_state.total_chunks = 0
        if 'document_stats' not in st.session_state:
            st.session_state.document_stats = {}
            
    def setup_azure_llm(self):
        """Setup Azure OpenAI LLM"""
        try:
            self.llm = AzureChatOpenAI(
                azure_deployment=st.secrets["AZURE_DEPLOYMENT"],
                azure_endpoint=st.secrets["AZURE_ENDPOINT"],
                api_key=st.secrets["AZURE_API_KEY"],
                api_version=st.secrets["AZURE_API_VERSION"],
                temperature=0.3,
                max_tokens=1000,
            )
        except Exception as e:
            st.error(f"❌ Error setting up Azure OpenAI: {str(e)}")
            st.stop()
    
    def create_enhanced_prompt(self):
        """Create an enhanced medical Q&A prompt template"""
        template = """
        You are MedAssist AI, a highly knowledgeable medical AI assistant specializing in providing accurate, 
        evidence-based health information. You have access to medical literature and documents to answer questions.

        ## Your Role & Responsibilities:
        - Provide clear, accurate medical information based on the provided context
        - Use medical terminology appropriately while ensuring explanations are understandable
        - Always emphasize when professional medical consultation is necessary
        - Cite relevant information from the provided medical documents
        - Maintain a professional, empathetic, and supportive tone

        ## Guidelines:
        1. **Accuracy First**: Base your responses on the provided medical context
        2. **Safety Disclaimer**: Always remind users that this is for informational purposes only
        3. **Professional Referral**: Encourage consultation with healthcare providers for diagnosis/treatment
        4. **Clear Structure**: Organize responses with headings and bullet points when helpful
        5. **Evidence-Based**: Reference the medical literature when possible

        ## Context from Medical Documents:
        {context}

        ## User Question:
        {question}

        ## Response Format:
        Provide a comprehensive answer that includes:
        - Direct answer to the question
        - Relevant medical details from the context
        - Important considerations or warnings
        - Recommendation to consult healthcare professionals

        **Important Disclaimer**: This information is for educational purposes only and should not replace professional medical advice, diagnosis, or treatment.

        Answer:
        """
        
        return ChatPromptTemplate.from_template(template)
    
    def process_documents(self, uploaded_files):
        """Process uploaded PDF documents with progress tracking"""
        all_docs = []
        
        # Create progress containers
        progress_container = st.container()
        
        with progress_container:
            st.markdown("### 📄 Processing Documents...")
            overall_progress = st.progress(0)
            status_text = st.empty()
            
            # Process each uploaded file
            total_files = len(uploaded_files)
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.markdown(f"**Processing:** {uploaded_file.name}")
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                
                try:
                    # Load PDF
                    loader = PyPDFLoader(tmp_file_path)
                    docs = loader.load()
                    all_docs.extend(docs)
                    
                    # Update progress
                    progress = (idx + 1) / total_files
                    overall_progress.progress(progress)
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                finally:
                    # Clean up temp file
                    os.unlink(tmp_file_path)
            
            status_text.markdown("**Splitting documents into chunks...**")
            
            # Text splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunks = text_splitter.split_documents(all_docs)
            st.session_state.total_chunks = len(chunks)
            
            # Document statistics
            st.session_state.document_stats = {
                'total_documents': len(all_docs),
                'total_chunks': len(chunks),
                'total_files': total_files,
                'processing_time': time.time()
            }
            
            status_text.markdown("**Creating embeddings and vector store...**")
            
            # Create embeddings and vector store
            with st.spinner("🧠 Creating intelligent embeddings..."):
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
                
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=None
                )
                
                st.session_state.vectorstore = vectorstore
                st.session_state.processing_complete = True
            
            overall_progress.progress(1.0)
            status_text.markdown("✅ **Processing Complete!**")
            
        return True
    
    def create_rag_chain(self):
        """Create the RAG chain for question answering"""
        if st.session_state.vectorstore is None:
            return None
            
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        prompt = self.create_enhanced_prompt()
        
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def render_header(self):
        """Render the main header"""
        st.markdown("""
        <div class="main-header">
            <h1>🏥 MedAssist AI</h1>
            <p>Your Intelligent Medical Q&A Companion - Powered by Advanced AI</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with controls and information"""
        with st.sidebar:
            st.markdown("""
            <div class="sidebar-content">
                <h2>🏥 MedAssist Control Panel</h2>
                <p>Upload medical documents and get intelligent answers to your health questions.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Document Upload Section
            st.markdown("### 📁 Document Upload")
            uploaded_files = st.file_uploader(
                "Upload Medical PDFs",
                type=['pdf'],
                accept_multiple_files=True,
                help="Upload medical documents, research papers, or health guides in PDF format."
            )
            
            if uploaded_files and not st.session_state.processing_complete:
                if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                    with st.spinner("Processing your medical documents..."):
                        self.process_documents(uploaded_files)
                        st.rerun()
            
            # System Status
            st.markdown("### 📊 System Status")
            if st.session_state.processing_complete:
                st.markdown('<div class="status-ready">🟢 Ready for Questions</div>', 
                           unsafe_allow_html=True)
                
                # Display statistics
                stats = st.session_state.document_stats
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📄 Documents", stats.get('total_documents', 0))
                with col2:
                    st.metric("🧩 Chunks", stats.get('total_chunks', 0))
                    
            else:
                st.markdown('<div class="status-processing">⏳ Waiting for Documents</div>', 
                           unsafe_allow_html=True)
            
            # Chat History Management
            st.markdown("### 💬 Chat Management")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
            with col2:
                chat_count = len(st.session_state.chat_history)
                st.metric("Messages", chat_count)
            
            # Quick Actions
            st.markdown("### ⚡ Quick Questions")
            quick_questions = [
                "What is hypertension?",
                "Symptoms of heart disease",
                "How to prevent diabetes?",
                "What is cholesterol?",
                "Signs of a heart attack"
            ]
            
            for question in quick_questions:
                if st.button(f"💡 {question}", key=f"quick_{question}", use_container_width=True):
                    st.session_state.current_question = question
                    st.rerun()
    
    def render_chat_interface(self):
        """Render the main chat interface"""
        st.markdown("### 💬 Ask Your Medical Questions")
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                st.markdown(f"""
                <div class="user-message">
                    <strong>🙋‍♀️ You:</strong><br>{question}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="assistant-message">
                    <strong>🏥 MedAssist AI:</strong><br>{answer}
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Question input
        question = st.text_area(
            "Enter your medical question:",
            height=100,
            placeholder="e.g., What are the symptoms of high blood pressure?",
            key="question_input"
        )
        
        # Handle quick questions
        if hasattr(st.session_state, 'current_question'):
            question = st.session_state.current_question
            del st.session_state.current_question
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            ask_button = st.button("🤖 Ask MedAssist AI", type="primary", use_container_width=True)
        with col2:
            example_button = st.button("📋 Example Questions", use_container_width=True)
        with col3:
            help_button = st.button("❓ Help", use_container_width=True)
        
        if example_button:
            st.info("""
            **Example Medical Questions:**
            - What are the risk factors for cardiovascular disease?
            - How can I lower my cholesterol naturally?
            - What is the difference between Type 1 and Type 2 diabetes?
            - What are the warning signs of a stroke?
            - How does exercise benefit heart health?
            """)
        
        if help_button:
            st.info("""
            **How to use MedAssist AI:**
            1. Upload medical PDF documents using the sidebar
            2. Wait for processing to complete
            3. Ask specific medical questions
            4. Get evidence-based answers from your documents
            
            **Remember:** This is for educational purposes only. Always consult healthcare professionals for medical advice.
            """)
        
        # Process question
        if ask_button and question and st.session_state.processing_complete:
            with st.spinner("🧠 MedAssist AI is analyzing your question..."):
                try:
                    rag_chain = self.create_rag_chain()
                    if rag_chain:
                        response = rag_chain.invoke(question)
                        
                        # Add to chat history
                        st.session_state.chat_history.append((question, response))
                        
                        # Clear the input
                        st.session_state.question_input = ""
                        st.rerun()
                    else:
                        st.error("❌ Unable to create the Q&A system. Please try again.")
                except Exception as e:
                    st.error(f"❌ Error processing question: {str(e)}")
        
        elif ask_button and question and not st.session_state.processing_complete:
            st.warning("⚠️ Please upload and process medical documents first!")
        
        elif ask_button and not question:
            st.warning("⚠️ Please enter a question!")
    
    def render_analytics_dashboard(self):
        """Render analytics and insights dashboard"""
        if not st.session_state.chat_history:
            return
            
        st.markdown("### 📊 Chat Analytics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Questions Asked</div>
            </div>
            """.format(len(st.session_state.chat_history)), unsafe_allow_html=True)
        
        with col2:
            avg_length = sum(len(q) for q, _ in st.session_state.chat_history) / len(st.session_state.chat_history)
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Avg Question Length</div>
            </div>
            """.format(int(avg_length)), unsafe_allow_html=True)
        
        with col3:
            if st.session_state.document_stats:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Documents Processed</div>
                </div>
                """.format(st.session_state.document_stats.get('total_files', 0)), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Knowledge Chunks</div>
            </div>
            """.format(st.session_state.total_chunks), unsafe_allow_html=True)
    
    def run(self):
        """Main application runner"""
        self.render_header()
        self.render_sidebar()
        
        # Main content area
        col1, col2 = st.columns([3, 1])
        
        with col1:
            self.render_chat_interface()
        
        with col2:
            self.render_analytics_dashboard()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #64748b; padding: 2rem;">
            <p><strong>🏥 MedAssist AI</strong> - Your trusted medical information companion</p>
            <p style="font-size: 0.9rem;">
                ⚠️ <strong>Medical Disclaimer:</strong> This application provides general health information for educational purposes only. 
                It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician 
                or other qualified health provider with any questions you may have regarding a medical condition.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    app = MedicalQAApp()
    app.run()
