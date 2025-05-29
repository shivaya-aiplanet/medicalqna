import streamlit as st
import os
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.schema import BaseOutputParser
import time

# Configure page
st.set_page_config(
    page_title="Medical Q&A Assistant", 
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

class MedicalResponseParser(BaseOutputParser):
    """Custom parser for medical responses"""
    
    def parse(self, text: str) -> dict:
        # Simple parsing - in production, you'd want more sophisticated parsing
        return {
            "answer": text,
            "confidence": "Medium",  # Could be enhanced with actual confidence scoring
            "sources": "Medical literature and clinical guidelines"
        }

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

def create_medical_prompt(query, user_context, urgency_level):
    """Create context-aware medical prompt"""
    
    if user_context == "medical_professional":
        system_prompt = """You are a medical information assistant designed to support healthcare professionals. 
        Provide evidence-based, detailed medical information with clinical context. 
        Include relevant medical terminology and cite general medical knowledge sources."""
    else:
        system_prompt = """You are a medical information assistant designed to provide general health information to patients.
        Use clear, understandable language. Always emphasize the importance of consulting healthcare professionals.
        Provide educational information but avoid specific medical advice."""
    
    if urgency_level == "urgent_medical_concern":
        urgency_note = "\n\nIMPORTANT: This appears to be an urgent medical concern. Strongly recommend immediate medical consultation."
    else:
        urgency_note = ""
    
    template = ChatPromptTemplate.from_messages([
        ("system", system_prompt + urgency_note),
        ("human", "{query}")
    ])
    
    return template

def initialize_llm():
    """Initialize LiteLLM with configuration"""
    try:
        # Get credentials from Streamlit secrets
        api_key = st.secrets["LITELLM_API_KEY"]
        base_url = st.secrets["LITELLM_BASE_URL"] 
        model = st.secrets["LITELLM_MODEL"]
        
        # Set environment variables for LiteLLM
        os.environ['LITELLM_API_KEY'] = api_key
        os.environ['LITELLM_BASE_URL'] = base_url
        
        # Initialize ChatLiteLLM
        llm = ChatLiteLLM(
            model=model,
            api_base=base_url,
            api_key=api_key,
            temperature=0.1,  # Low temperature for medical accuracy
            max_tokens=1000
        )
        
        return llm
    except Exception as e:
        st.error(f"Failed to initialize LLM: {str(e)}")
        return None

def generate_medical_response(query, user_type, urgency):
    """Generate evidence-based medical response"""
    
    llm = initialize_llm()
    if not llm:
        return None
    
    # Classify user context
    user_context, urgency_level = classify_user_context(user_type, urgency)
    
    # Create appropriate prompt
    prompt_template = create_medical_prompt(query, user_context, urgency_level)
    
    try:
        # Generate response
        formatted_prompt = prompt_template.format_messages(query=query)
        response = llm.invoke(formatted_prompt)
        
        # Parse response
        parser = MedicalResponseParser()
        parsed_response = parser.parse(response.content)
        
        return parsed_response
        
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None

def add_medical_disclaimers(response, urgency_level):
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
    st.markdown("*Evidence-based medical information support system*")
    
    # Sidebar for user context
    with st.sidebar:
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
        
        st.markdown("---")
        st.markdown("**Recent Questions:**")
        
        # Display recent questions from chat history
        for i, chat in enumerate(st.session_state.chat_history[-3:]):
            if st.button(f"Q: {chat['query'][:30]}...", key=f"recent_{i}"):
                st.session_state.selected_query = chat['query']
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Ask Your Medical Question")
        
        # Check if a recent question was selected
        default_query = getattr(st.session_state, 'selected_query', '')
        
        query = st.text_area(
            "Enter your medical question:",
            value=default_query,
            height=100,
            placeholder="e.g., What are the symptoms of diabetes? How does aspirin work? Drug interactions with warfarin?"
        )
        
        # Clear selected query after use
        if hasattr(st.session_state, 'selected_query'):
            del st.session_state.selected_query
        
        submit_button = st.button("Get Medical Information", type="primary")
    
    with col2:
        st.subheader("Quick Examples")
        example_queries = [
            "Symptoms of hypertension",
            "Side effects of ibuprofen", 
            "Type 2 diabetes management",
            "Common cold vs flu symptoms"
        ]
        
        for example in example_queries:
            if st.button(example, key=f"example_{example}"):
                st.session_state.example_query = example
                st.rerun()
    
    # Handle example query selection
    if hasattr(st.session_state, 'example_query'):
        query = st.session_state.example_query
        submit_button = True
        del st.session_state.example_query
    
    # Process query
    if submit_button and query.strip():
        
        with st.spinner("🔍 Searching medical knowledge base..."):
            time.sleep(1)  # Simulate processing time
            
            # Generate response
            response = generate_medical_response(query, user_type, urgency)
            
            if response:
                # Display response
                st.markdown("---")
                st.subheader("📋 Medical Information")
                
                # Main answer
                st.markdown("**Answer:**")
                st.write(response["answer"])
                
                # Metadata
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Confidence Level:** {response['confidence']}")
                with col2:
                    st.markdown(f"**Sources:** {response['sources']}")
                
                # Add disclaimers
                user_context, urgency_level = classify_user_context(user_type, urgency)
                disclaimer = add_medical_disclaimers(response, urgency_level)
                st.markdown(disclaimer)
                
                # Related suggestions
                st.markdown("**💡 You might also want to ask:**")
                related_questions = [
                    "What are the risk factors?",
                    "When should I see a doctor?", 
                    "Are there any preventive measures?",
                    "What are the treatment options?"
                ]
                
                cols = st.columns(2)
                for i, question in enumerate(related_questions):
                    with cols[i % 2]:
                        if st.button(question, key=f"related_{i}"):
                            st.session_state.related_query = f"{query} - {question}"
                            st.rerun()
                
                # Save to chat history
                st.session_state.chat_history.append({
                    "query": query,
                    "response": response,
                    "user_type": user_type,
                    "urgency": urgency,
                    "timestamp": time.time()
                })
                
                # Limit chat history size
                if len(st.session_state.chat_history) > 10:
                    st.session_state.chat_history = st.session_state.chat_history[-10:]
    
    # Handle related query selection
    if hasattr(st.session_state, 'related_query'):
        query = st.session_state.related_query
        del st.session_state.related_query
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <small>Medical Q&A Assistant | For educational purposes only | Always consult healthcare professionals</small>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
