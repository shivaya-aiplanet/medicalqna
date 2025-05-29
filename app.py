import streamlit as st
import os
import tempfile
import uuid
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
import time

# Configure page
st.set_page_config(
    page_title="Medical Q&A Assistant", 
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'collection_name' not in st.session_state:
    st.session_state.collection_name = None

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
def initialize_embeddings():
    """Initialize Azure OpenAI embeddings"""
    try:
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=st.secrets["AZURE_DEPLOYMENT"],
            openai_api_version=st.secrets["OPENAI_API_VERSION"],
            azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
            api_key=st.secrets["OPENAI_API_KEY"]
        )
        return embeddings
    except Exception as e:
        st.error(f"Failed to initialize embeddings: {str(e)}")
        return None

@st.cache_resource
def initialize_qdrant_client():
    """Initialize Qdrant client"""
    try:
        client = QdrantClient(
            url=st.secrets["QDRANT_URL"],
            api_key=st.secrets["QDRANT_API_KEY"],
            prefer_grpc=True
        )
        return client
    except Exception as e:
        st.error(f"Failed to initialize Qdrant client: {str(e)}")
        return None

def process_pdf_with_qdrant(uploaded_file):
    """Process uploaded PDF and create Qdrant vector store"""
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
            
            # Initialize embeddings and Qdrant client
            embeddings = initialize_embeddings()
            qdrant_client = initialize_qdrant_client()
            
            if not embeddings or not qdrant_client:
                return None, None
            
            # Create unique collection name
            collection_name = f"medical_docs_{uuid.uuid4().hex[:8]}"
            
            # Create vector store
            vector_store = QdrantVectorStore.from_documents(
                documents=chunks,
                embedding=embeddings,
                url=st.secrets["QDRANT_URL"],
                api_key=st.secrets["QDRANT_API_KEY"],
                collection_name=collection_name,
                prefer_grpc=True
            )
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            return vector_store, collection_name
            
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None, None

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
        if st.session_state.vector_store:
            # Use Qdrant vector store for retrieval
            prompt_template = create_medical_prompt(user_context, urgency_level, has_knowledge_base=True)
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 3}),
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

def clear_vector_store():
    """Clear the current vector store and collection"""
    try:
        if st.session_state.collection_name:
            qdrant_client = initialize_qdrant_client()
            if qdrant_client:
                # Delete the collection from Qdrant
                qdrant_client.delete_collection(st.session_state.collection_name)
        
        # Clear session state
        st.session_state.vector_store = None
        st.session_state.uploaded_file_name = None
        st.session_state.collection_name = None
        
    except Exception as e:
        st.error(f"Error clearing vector store: {str(e)}")

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🏥 Medical Q&A Assistant")
    st.markdown("*Chat with medical documents using OpenAI embeddings and Qdrant vector database*")
    
    # Sidebar for document upload and user context
    with st.sidebar:
        st.header("📄 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Upload a medical document (PDF)",
            type="pdf",
            help="Upload a medical document to chat with its content using Qdrant vector search"
        )
        
        if uploaded_file is not None:
            if st.session_state.uploaded_file_name != uploaded_file.name:
                # Clear previous vector store
                if st.session_state.vector_store:
                    clear_vector_store()
                
                # Process new document
                vector_store, collection_name = process_pdf_with_qdrant(uploaded_file)
                
                if vector_store and collection_name:
                    st.session_state.vector_store = vector_store
                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.session_state.collection_name = collection_name
                    st.success(f"✅ Document '{uploaded_file.name}' processed successfully!")
                    st.info(f"📊 Collection: {collection_name}")
                else:
                    st.error("❌ Failed to process document")
        
        if st.button("Clear Document"):
            clear_vector_store()
            st.success("Document and vector store cleared!")
            st.rerun()
        
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
        if st.session_state.vector_store:
            st.success(f"📄 Document: {st.session_state.uploaded_file_name}")
            st.success(f"🔍 Vector DB: Qdrant Cloud")
            st.success(f"🧠 Embeddings: Azure OpenAI")
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
                if st.session_state.vector_store:
                    st.info(f"💡 Answer based on uploaded document: {st.session_state.uploaded_file_name}")
                    st.info(f"🔍 Powered by: Qdrant Vector Search + Azure OpenAI Embeddings")
                else:
                    st.info("💡 Answer based on general medical knowledge")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <small>Medical Q&A Assistant | Powered by OpenAI Embeddings + Qdrant Vector Database</small>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
