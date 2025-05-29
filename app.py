import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from litellm import completion
from datetime import datetime
import pandas as pd
import os

# Page config
st.set_page_config(
    page_title="MedAssist AI - Medical Q&A",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for modern UI with fixed visibility
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family= Angstrom:wght@300;400;600&display=swap');

    * {
        font-family: 'Poppins', sans-serif;
    }

    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
    }

    /* Header styling */
    .header {
        background: linear-gradient(90deg, #6b48ff 0%, #00ddeb 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin-bottom: 2rem;
    }

    .header h1 {
        font-size: 2.5rem;
        font-weight: 600;
        margin: 0;
    }

    .header p {
        font-size: 1.1rem;
        font-weight: 300;
        margin: 0.5rem 0 0;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #6b48ff 0%, #3b2a99 100%);
        border-radius: 12px;
        padding: 1rem;
    }

    .css-1d391kg h3 {
        color: white;
        font-weight: 600;
    }

    .css-1d391kg select {
        background: #ffffff;
        border-radius: 8px;
        padding: 0.5rem;
        border: none;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        color: #333;
    }

    /* Main content styling */
    .content-box {
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
    }

    .content-box h3 {
        font-weight: 600;
        color: #333;
        margin-bottom: 1rem;
    }

    /* Response box - transparent background, black text */
    .response-box {
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e1e5e9;
        margin: 1rem 0;
        background: rgba(255, 255, 255, 0.9);
        color: #333;
    }

    /* Disclaimer */
    .disclaimer {
        background: #fff5f5;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #fed7d7;
        color: #721c24;
        font-weight: 400;
    }

    /* Query history item - transparent background, white text */
    .history-item {
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #6b48ff;
        background: rgba(255, 255, 255, 0.1);
        color: #ffffff;
    }

    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #d1d5db;
        padding: 0.75rem;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        color: #333; /* Ensure input text is dark */
    }

    /* Button styling */
    .stButton > button {
        background: #6b48ff;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: #5439cc;
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }

    /* Remove Streamlit's default branding */
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

class MedicalRAGSystem:
    def __init__(self):
        # Set litellm environment variables
        os.environ["LITELLM_API_KEY"] = st.secrets["LITELLM_API_KEY"]
        os.environ["LITELLM_BASE_URL"] = st.secrets["LITELLM_BASE_URL"]
        self.model = st.secrets["LITELLM_MODEL"]
        # Initialize Hugging Face embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Initialize vector store
        self.vector_store = self._initialize_vector_store()

    def _initialize_knowledge_base(self):
        """Simple medical knowledge base as text"""
        knowledge = """
        Fever: Elevated body temperature, often indicating infection or inflammation.
        Headache: Pain in the head or upper neck, may be due to stress, dehydration, or other causes.
        Chest Pain: Discomfort in the chest, potentially indicating cardiac or pulmonary issues.
        Shortness of Breath: Difficulty breathing, may be related to respiratory or cardiac conditions.
        Antibiotics: Medications used to treat bacterial infections.
        Analgesics: Pain-relieving medications like ibuprofen or acetaminophen.
        Warfarin: Blood thinner with multiple drug interactions.
        Aspirin: May interact with blood thinners and cause side effects.
        """
        return knowledge

    def _initialize_vector_store(self):
        """Split text and create FAISS vector store"""
        text = self._initialize_knowledge_base()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_text(text)
        return FAISS.from_texts(chunks, self.embeddings)

    def retrieve_relevant_info(self, query: str, top_k: int = 3):
        """Retrieve relevant documents from vector store"""
        docs = self.vector_store.similarity_search(query, k=top_k)
        return [doc.page_content for doc in docs]

    def generate_response(self, query: str, user_type: str):
        """Generate response using RAG with litellm"""
        # Retrieve relevant information
        relevant_docs = self.retrieve_relevant_info(query)
        context = "\n".join(relevant_docs)

        # Create prompt
        system_prompt = f"""You are MedAssist AI, a medical information assistant. Provide accurate, evidence-based answers in {'technical' if user_type == 'Healthcare Professional' else 'simple, clear'} language. Use the following context:

{context}

Always include a disclaimer to consult a healthcare professional. Answer the query: {query}"""
        
        try:
            with st.spinner("Processing your query..."):
                response = completion(
                    model=self.model,
                    api_base=st.secrets["LITELLM_BASE_URL"],
                    api_key=st.secrets["LITELLM_API_KEY"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}
                    ],
                    temperature=0.5,
                    max_tokens=1000
                )
            answer = response.choices[0].message.content
            disclaimer = "⚠️ **Disclaimer**: This information is for educational purposes only. Always consult a healthcare professional for diagnosis and treatment."
            return {"answer": answer, "disclaimer": disclaimer}
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return {"error": str(e)}

def initialize_session_state():
    """Initialize session state"""
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = MedicalRAGSystem()

def main():
    initialize_session_state()

    # Header
    st.markdown("""
    <div class="header">
        <h1>🩺 MedAssist AI</h1>
        <p>Your Modern Medical Information Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 👤 User Profile")
        user_type = st.selectbox(
            "Select your profile:",
            ["Patient/General Public", "Healthcare Professional"],
            help="Customizes response language"
        )

        st.markdown("### 📋 Query History")
        if st.button("Clear History"):
            st.session_state.query_history = []
            st.rerun()

        if st.session_state.query_history:
            for query_data in reversed(st.session_state.query_history[-3:]):
                st.markdown(f"""
                <div class="history-item">
                    <small>{query_data['timestamp']}</small><br>
                    <strong>{query_data['query'][:50]}...</strong>
                </div>
                """, unsafe_allow_html=True)

    # Main content
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.markdown("### 💬 Ask a Medical Question")
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., What are the symptoms of hypertension?"
    )

    if st.button("Get Answer"):
        if query.strip():
            response = st.session_state.rag_system.generate_response(query, user_type)
            if "error" not in response:
                st.markdown(f"""
                <div class="response-box">
                    <h3>Response</h3>
                    {response['answer']}
                </div>
                <div class="disclaimer">
                    {response['disclaimer']}
                </div>
                """, unsafe_allow_html=True)

                # Save to history
                st.session_state.query_history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "query": query,
                    "response": response['answer'][:100] + "..."
                })
        else:
            st.warning("Please enter a question.")
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
