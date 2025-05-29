import streamlit as st
import os
from langchain_community.chat_models import ChatLiteLLM
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models
import time

# Page config
st.set_page_config(
    page_title="Medical Q&A Assistant",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

def setup_llm():
    """Setup LiteLLM with API credentials"""
    return ChatLiteLLM(
        model=st.secrets["LITELLM_MODEL"],
        api_base=st.secrets["LITELLM_BASE_URL"],
        api_key=st.secrets["LITELLM_API_KEY"],
        temperature=0.1
    )

def setup_embeddings():
    """Setup HuggingFace embeddings"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def create_medical_knowledge_base():
    """Create a basic medical knowledge base"""
    medical_docs = [
        "Fever is a temporary increase in body temperature, often due to an illness. Normal body temperature is around 98.6°F (37°C). A fever is generally considered when temperature reaches 100.4°F (38°C) or higher.",
        "Headaches can be caused by stress, dehydration, lack of sleep, or underlying medical conditions. Most headaches are not serious, but persistent or severe headaches should be evaluated by a healthcare professional.",
        "High blood pressure (hypertension) is when blood pressure readings are consistently 130/80 mmHg or higher. It's often called the 'silent killer' because it usually has no symptoms.",
        "Diabetes is a group of metabolic disorders characterized by high blood sugar levels. Type 1 diabetes is autoimmune, while Type 2 diabetes is often related to lifestyle factors.",
        "Chest pain can range from minor to life-threatening. Cardiac chest pain may indicate heart attack and requires immediate medical attention. Other causes include muscle strain, acid reflux, or anxiety.",
        "Shortness of breath (dyspnea) can be caused by heart conditions, lung diseases, anxiety, or physical exertion. Sudden onset of severe shortness of breath requires immediate medical evaluation.",
        "Antibiotics are medications that fight bacterial infections. They do not work against viral infections like the common cold or flu. Overuse can lead to antibiotic resistance.",
        "Aspirin is used for pain relief, fever reduction, and blood thinning. Low-dose aspirin may be prescribed for heart disease prevention, but should only be taken under medical supervision.",
        "Depression is a mental health condition characterized by persistent sadness, loss of interest, and other symptoms that interfere with daily life. It's treatable with therapy, medication, or both.",
        "Hypertension medications include ACE inhibitors, beta-blockers, diuretics, and calcium channel blockers. Each works differently to lower blood pressure and may have different side effects."
    ]
    
    documents = [Document(page_content=doc, metadata={"source": "medical_knowledge"}) for doc in medical_docs]
    return documents

def setup_vector_store():
    """Setup Qdrant vector store with medical knowledge"""
    with st.spinner("🔄 Setting up medical knowledge base..."):
        try:
            # Setup embeddings
            embeddings = setup_embeddings()
            
            # Create medical documents
            documents = create_medical_knowledge_base()
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            splits = text_splitter.split_documents(documents)
            
            # Setup Qdrant client (in-memory for simplicity)
            client = QdrantClient(":memory:")
            
            # Create collection
            client.create_collection(
                collection_name="medical_knowledge",
                vectors_config=models.VectorParams(
                    size=384,  # MiniLM embedding size
                    distance=models.Distance.COSINE
                )
            )
            
            # Create vector store
            vector_store = Qdrant(
                client=client,
                collection_name="medical_knowledge",
                embeddings=embeddings
            )
            
            # Add documents
            vector_store.add_documents(splits)
            
            return vector_store
            
        except Exception as e:
            st.error(f"Error setting up vector store: {str(e)}")
            return None

def classify_user_context(user_type, urgency):
    """Classify user context and urgency"""
    context = {
        "user_type": user_type,
        "urgency": urgency,
        "is_professional": user_type == "Healthcare Professional"
    }
    return context

def create_qa_chain(vector_store, user_context):
    """Create QA chain with medical prompt template"""
    llm = setup_llm()
    
    # Create prompt template based on user context
    if user_context["is_professional"]:
        template = """You are a medical AI assistant providing information to healthcare professionals.
        
Context: {context}

Question: {question}

Provide a detailed, evidence-based response including:
1. Clinical information and differential diagnoses where relevant
2. Current medical guidelines and recommendations
3. Potential complications or considerations
4. Confidence level in your response

Remember: This is for informational purposes only and should not replace clinical judgment.

Answer:"""
    else:
        template = """You are a medical AI assistant providing information to patients and the general public.

Context: {context}

Question: {question}

Provide a clear, understandable response including:
1. Simple explanation of the condition or topic
2. General recommendations and when to seek medical care
3. Important safety information
4. Confidence level in your response

IMPORTANT DISCLAIMER: This information is for educational purposes only and does not constitute medical advice. Always consult with a qualified healthcare professional for proper diagnosis and treatment.

Answer:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return qa_chain

def generate_medical_response(question, user_context, qa_chain):
    """Generate evidence-based medical response"""
    with st.spinner("🔍 Searching medical knowledge base..."):
        time.sleep(1)  # Simulate processing
        
    with st.spinner("🧠 Analyzing medical information..."):
        try:
            result = qa_chain({"query": question})
            return result
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return None

def main():
    # Header
    st.title("🏥 Medical Q&A Assistant")
    st.markdown("*Providing evidence-based medical information with appropriate disclaimers*")
    
    # Sidebar for user context
    with st.sidebar:
        st.header("👤 User Information")
        
        user_type = st.selectbox(
            "I am a:",
            ["Patient/General Public", "Healthcare Professional"],
            help="This helps tailor the response appropriately"
        )
        
        urgency = st.selectbox(
            "Urgency Level:",
            ["General Information", "Moderate Concern", "High Priority"],
            help="Indicates the urgency of your medical question"
        )
        
        st.markdown("---")
        st.markdown("### ⚠️ Important Notice")
        st.warning(
            "This AI assistant provides general medical information only. "
            "For emergencies, call emergency services immediately. "
            "Always consult healthcare professionals for medical advice."
        )
    
    # Initialize vector store if not exists
    if st.session_state.vector_store is None:
        with st.spinner("🚀 Initializing medical knowledge base..."):
            st.session_state.vector_store = setup_vector_store()
            
        if st.session_state.vector_store is None:
            st.error("Failed to initialize the medical knowledge base. Please refresh the page.")
            return
    
    # Main interface
    st.markdown("### 💬 Ask Your Medical Question")
    
    question = st.text_area(
        "Enter your medical question:",
        placeholder="e.g., What are the symptoms of high blood pressure?",
        height=100
    )
    
    if st.button("🔍 Get Medical Information", type="primary"):
        if not question.strip():
            st.warning("Please enter a medical question.")
            return
            
        # Classify user context
        user_context = classify_user_context(user_type, urgency)
        
        # Create QA chain
        if st.session_state.qa_chain is None:
            with st.spinner("⚙️ Setting up medical analysis system..."):
                st.session_state.qa_chain = create_qa_chain(
                    st.session_state.vector_store, 
                    user_context
                )
        
        # Generate response
        result = generate_medical_response(
            question, 
            user_context, 
            st.session_state.qa_chain
        )
        
        if result:
            # Display response
            st.markdown("### 📋 Medical Information")
            
            # Main answer
            st.markdown("**Response:**")
            st.write(result['result'])
            
            # Confidence and reliability
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Reliability Level:**")
                st.info("Based on general medical knowledge")
                
            with col2:
                st.markdown("**👤 Response Type:**")
                st.info(f"Tailored for: {user_type}")
            
            # Source information
            if result.get('source_documents'):
                with st.expander("📚 Knowledge Sources"):
                    for i, doc in enumerate(result['source_documents']):
                        st.write(f"**Source {i+1}:** {doc.page_content[:200]}...")
            
            # Related topics
            st.markdown("---")
            st.markdown("**🔗 Related Topics to Explore:**")
            related_topics = [
                "Symptoms and diagnosis",
                "Treatment options",
                "Prevention strategies",
                "When to seek medical care"
            ]
            
            cols = st.columns(len(related_topics))
            for i, topic in enumerate(related_topics):
                with cols[i]:
                    if st.button(topic, key=f"related_{i}"):
                        st.info(f"Consider asking about: {topic.lower()} related to your condition")
            
            # Final disclaimer
            st.markdown("---")
            st.error(
                "⚠️ **MEDICAL DISCLAIMER:** This information is for educational purposes only. "
                "It does not constitute medical advice, diagnosis, or treatment. "
                "Always consult with qualified healthcare professionals for medical concerns."
            )

if __name__ == "__main__":
    main()
