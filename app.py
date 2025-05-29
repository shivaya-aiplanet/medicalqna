import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from openai import AzureOpenAI
from datetime import datetime
import pandas as pd

# Page config
st.set_page_config(
    page_title="MedAssist AI - Medical Q&A",
    page_icon="🩺",
    layout="wide"
)

# Basic CSS for UI
st.markdown("""
<style>
    .header {
        background: #667eea;
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
    }
    .response-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e1e5e9;
        margin: 1rem 0;
    }
    .disclaimer {
        background: #fff5f5;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #fed7d7;
    }
</style>
""", unsafe_allow_html=True)

class MedicalRAGSystem:
    def __init__(self):
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=st.secrets["AZURE_API_KEY"],
            api_version=st.secrets["AZURE_API_VERSION"],
            azure_endpoint=st.secrets["AZURE_ENDPOINT"]
        )
        # Initialize Hugging Face embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Initialize knowledge base and vector store
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
        """Generate response using RAG"""
        # Retrieve relevant information
        relevant_docs = self.retrieve_relevant_info(query)
        context = "\n".join(relevant_docs)

        # Create prompt
        system_prompt = f"""You are MedAssist AI, a medical information assistant. Provide accurate, evidence-based answers in {'technical' if user_type == 'Healthcare Professional' else 'simple, clear'} language. Use the following context:

{context}

Always include a disclaimer to consult a healthcare professional. Answer the query: {query}"""
        
        try:
            with st.spinner("Processing your query..."):
                response = self.client.chat.completions.create(
                    model=st.secrets["MODEL_NAME"],
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
        <p>Medical Information Assistant</p>
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
                <div style="background: #f8f9fa; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0;">
                    <small>{query_data['timestamp']}</small><br>
                    <strong>{query_data['query'][:50]}...</strong>
                </div>
                """, unsafe_allow_html=True)

    # Main content
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

if __name__ == "__main__":
    main()
