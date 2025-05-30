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
if 'messages' not in st.session_state:
    st.session_state.messages = []

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
        with st.spinner("📄 Reading and analyzing your document... This may take a moment."):
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

def get_style_instruction(style):
    """Get style instruction based on user selection"""
    style_map = {
        "Concise": "Provide a brief, concise answer focusing on key points only. Keep the response short and direct.",
        "Moderate": "Provide a balanced response with adequate detail and explanation while maintaining clarity.",
        "Prolonged": "Provide a comprehensive, detailed response with thorough explanations, background information, and examples where appropriate."
    }
    return style_map.get(style, style_map["Moderate"])

def create_medical_prompt(user_context, urgency_level, style, has_knowledge_base=False):
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
    
    # Add style instruction
    style_instruction = f"\n\nResponse Style: {get_style_instruction(style)}"
    
    if has_knowledge_base:
        context_instruction = f"""
        Answer the question based on the provided context from the uploaded document. 
        If the context doesn't contain relevant information, clearly state that the information is not available in the document.
        
        Document Context: {{context}}
        
        Question: {{question}}"""
    else:
        context_instruction = f"""
        Answer the medical question based on your general medical knowledge. 
        Provide accurate, evidence-based information.
        
        Question: {{question}}"""
    
    full_prompt = base_prompt + urgency_note + style_instruction + "\n\n" + context_instruction
    
    return ChatPromptTemplate.from_template(full_prompt)

def generate_response(query, user_type, urgency, style):
    """Generate response with or without knowledge base"""
    
    llm = initialize_llm()
    if not llm:
        return None
    
    user_context, urgency_level = classify_user_context(user_type, urgency)
    
    try:
        if st.session_state.vector_store:
            # Use Qdrant vector store for retrieval
            prompt_template = create_medical_prompt(user_context, urgency_level, style, has_knowledge_base=True)
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": prompt_template}
            )
            
            response = qa_chain.run(query)
            
        else:
            # Use LLM directly without knowledge base
            prompt_template = create_medical_prompt(user_context, urgency_level, style, has_knowledge_base=False)
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
        st.session_state.messages = []
        
    except Exception as e:
        st.error(f"Error clearing vector store: {str(e)}")

def get_sample_questions():
    """Get sample questions based on document availability"""
    if st.session_state.vector_store:
        return [
            "What is the main topic discussed in this document?",
            "Can you summarize the key findings or recommendations?",
            "What are the important symptoms or conditions mentioned?",
            "Are there any treatment options or medications discussed?"
        ]
    else:
        return [
            "What are the common symptoms of diabetes?",
            "How can I maintain good heart health?",
            "What should I know about high blood pressure?",
            "When should I see a doctor for persistent headaches?"
        ]

def process_message_response(query, user_type, urgency, style):
    """Process and return message response data"""
    response = generate_response(query, user_type, urgency, style)
    if response:
        user_context, urgency_level = classify_user_context(user_type, urgency)
        disclaimer = add_medical_disclaimers(urgency_level)
        
        # Combine response with disclaimer
        full_response = response + "\n\n" + disclaimer
        
        # Add source information
        if st.session_state.vector_store:
            source_info = f"\n\n💡 Answer based on your document: {st.session_state.uploaded_file_name}"
        else:
            source_info = "\n\n💡 Answer based on general medical knowledge"
        
        full_response += source_info
        
        return {
            "role": "assistant",
            "content": full_response
        }
    return None

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🏥 Medical Q&A Assistant")
    st.markdown("*Get answers to your medical questions with AI-powered assistance*")
    
    # Sidebar configuration
    with st.sidebar:
        # 1. Document Upload
        st.header("📄 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Upload a medical document (PDF)",
            type="pdf",
            help="Upload a medical document to get specific answers from its content"
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
                    st.success(f"✅ Document '{uploaded_file.name}' is ready for questions!")
                else:
                    st.error("❌ Failed to process document")
        
        if st.button("Clear Document"):
            clear_vector_store()
            st.success("Document cleared! Starting fresh.")
            st.rerun()
        
        st.markdown("---")
        
        # 2. Current Status
        st.header("📊 Current Status")
        if st.session_state.vector_store:
            st.success(f"📄 Document: {st.session_state.uploaded_file_name}")
            st.info("Answers will be based on your uploaded document")
        else:
            st.info("💡 No document loaded - answers will be based on general medical knowledge")
        
        st.markdown("---")
        
        # 3. Response Style
        st.header("🎨 Response Style")
        style = st.radio(
            "Choose response length:",
            ["Concise", "Moderate", "Prolonged"],
            index=1,  # Default to Moderate
            help="Select how detailed you want the responses to be"
        )
        
        st.markdown("---")
        
        # 4. User Context
        st.header("👤 User Context")
        
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
    
    # Main chat interface
    # Sample questions
    sample_questions = get_sample_questions()
    if sample_questions:
        st.subheader("💡 Sample Questions")
        cols = st.columns(2)
        for i, question in enumerate(sample_questions):
            with cols[i % 2]:
                if st.button(f"📊 {question}", key=f"sample_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": question})
                    with st.spinner("Processing..."):
                        response_data = process_message_response(question, user_type, urgency, style)
                        if response_data:
                            st.session_state.messages.append(response_data)
                        st.rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_data = process_message_response(prompt, user_type, urgency, style)
                if response_data:
                    st.session_state.messages.append(response_data)
                    st.markdown(response_data["content"])
                else:
                    error_msg = "Sorry, I encountered an error while processing your question. Please try again."
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.markdown(error_msg)

if __name__ == "__main__":
    main()
