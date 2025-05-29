import streamlit as st
import openai
from openai import AzureOpenAI
import pandas as pd
from datetime import datetime
import time
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from typing import List, Dict, Any
import re
import torch

# Page config
st.set_page_config(
    page_title="MedAssist AI - Medical Q&A Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .user-type-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .medical-response {
        background: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #e1e5e9;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .confidence-high {
        background: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    
    .confidence-medium {
        background: #fff3cd;
        color: #856404;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
    }
    
    .confidence-low {
        background: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
    
    .disclaimer-box {
        background: #fff5f5;
        border: 2px solid #fed7d7;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .sidebar-section {
        background: #f7fafc;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .query-history {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

class MedicalKnowledgeBase:
    def __init__(self):
        try:
            # Check if CUDA is available, otherwise use CPU
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")
            # Fallback to CPU if there's an error
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.knowledge_base = self._initialize_medical_knowledge()
        
    def _initialize_medical_knowledge(self):
        """Initialize with basic medical knowledge"""
        return {
            "symptoms": {
                "fever": "Elevated body temperature, often indicating infection or inflammation",
                "headache": "Pain in the head or upper neck region",
                "chest_pain": "Discomfort in chest area, may indicate cardiac or pulmonary issues",
                "shortness_of_breath": "Difficulty breathing, may indicate respiratory or cardiac problems"
            },
            "treatments": {
                "antibiotics": "Medications used to treat bacterial infections",
                "analgesics": "Pain-relieving medications",
                "anti_inflammatory": "Medications that reduce inflammation"
            },
            "drug_interactions": {
                "warfarin": "Blood thinner with multiple drug interactions",
                "aspirin": "May interact with blood thinners and certain medications"
            }
        }
    
    def search_knowledge(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search medical knowledge base"""
        results = []
        query_lower = query.lower()
        
        for category, items in self.knowledge_base.items():
            for item, description in items.items():
                if any(word in item.lower() or word in description.lower() 
                      for word in query_lower.split()):
                    results.append({
                        "category": category,
                        "item": item,
                        "description": description,
                        "relevance": 0.8
                    })
        
        return results[:top_k]

class MedicalQASystem:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=st.secrets["AZURE_API_KEY"],
            api_version=st.secrets["AZURE_API_VERSION"],
            azure_endpoint=st.secrets["AZURE_ENDPOINT"]
        )
        self.knowledge_base = MedicalKnowledgeBase()
        
    def classify_query(self, query: str) -> Dict[str, Any]:
        """Classify medical query type and urgency"""
        urgency_keywords = {
            "emergency": ["chest pain", "difficulty breathing", "severe pain", "bleeding", "unconscious"],
            "urgent": ["fever", "infection", "pain", "rash"],
            "routine": ["information", "general", "advice", "prevention"]
        }
        
        query_lower = query.lower()
        urgency = "routine"
        
        for level, keywords in urgency_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                urgency = level
                break
                
        category = "general"
        if any(word in query_lower for word in ["symptom", "pain", "fever", "headache"]):
            category = "symptoms"
        elif any(word in query_lower for word in ["treatment", "medication", "drug"]):
            category = "treatment"
        elif any(word in query_lower for word in ["interaction", "side effect"]):
            category = "drug_interaction"
            
        return {
            "category": category,
            "urgency": urgency,
            "confidence": 0.7
        }
    
    def generate_medical_response(self, query: str, user_type: str, context: Dict) -> Dict[str, Any]:
        """Generate comprehensive medical response"""
        
        # Search knowledge base
        relevant_info = self.knowledge_base.search_knowledge(query)
        
        # Create context-aware prompt
        system_prompt = self._create_system_prompt(user_type, context)
        user_prompt = self._create_user_prompt(query, relevant_info, context)
        
        try:
            with st.spinner("🔍 Analyzing medical query and searching knowledge base..."):
                response = self.client.chat.completions.create(
                    model=st.secrets["MODEL_NAME"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1500
                )
            
            content = response.choices[0].message.content
            
            # Extract structured response
            structured_response = self._parse_response(content)
            
            return {
                "answer": structured_response.get("answer", content),
                "confidence": structured_response.get("confidence", "medium"),
                "sources": structured_response.get("sources", []),
                "related_topics": structured_response.get("related_topics", []),
                "disclaimer": self._get_disclaimer(context["urgency"]),
                "follow_up_questions": structured_response.get("follow_up", [])
            }
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return {"error": str(e)}
    
    def _create_system_prompt(self, user_type: str, context: Dict) -> str:
        """Create context-aware system prompt"""
        base_prompt = """You are MedAssist AI, a medical information assistant designed to provide evidence-based, reliable medical information. 

Your role is to:
1. Provide accurate, evidence-based medical information
2. Cite reliable medical sources when possible
3. Indicate confidence levels in your responses
4. Suggest follow-up questions for comprehensive care
5. Always recommend professional medical consultation for diagnosis and treatment

Response Format:
- Provide clear, structured medical information
- Include confidence level (high/medium/low)
- List relevant sources or medical guidelines
- Suggest related topics to explore
- Provide appropriate follow-up questions

Safety Guidelines:
- Never provide specific diagnostic conclusions
- Always recommend professional medical consultation
- Indicate when immediate medical attention may be needed
- Be clear about limitations of AI medical advice"""

        if user_type == "Healthcare Professional":
            base_prompt += "\n\nUser Context: Healthcare Professional - You may use more technical terminology and provide detailed clinical information."
        else:
            base_prompt += "\n\nUser Context: Patient/General Public - Use clear, accessible language and focus on general health education."
            
        return base_prompt
    
    def _create_user_prompt(self, query: str, relevant_info: List, context: Dict) -> str:
        """Create user prompt with context"""
        prompt = f"""Medical Query: {query}

Query Classification:
- Category: {context.get('category', 'general')}
- Urgency: {context.get('urgency', 'routine')}

Relevant Knowledge Base Information:
"""
        
        for info in relevant_info:
            prompt += f"- {info['category']}: {info['item']} - {info['description']}\n"
        
        prompt += """
Please provide a comprehensive medical response including:
1. Evidence-based answer to the query
2. Confidence level assessment
3. Relevant medical sources or guidelines
4. Related topics for further exploration
5. Appropriate follow-up questions
6. Any necessary warnings or recommendations for professional consultation
"""
        
        return prompt
    
    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse structured response from LLM"""
        # Simple parsing - in production, use more sophisticated NLP
        lines = content.split('\n')
        
        response = {
            "answer": content,
            "confidence": "medium",
            "sources": [],
            "related_topics": [],
            "follow_up": []
        }
        
        # Extract confidence if mentioned
        content_lower = content.lower()
        if "high confidence" in content_lower or "very confident" in content_lower:
            response["confidence"] = "high"
        elif "low confidence" in content_lower or "uncertain" in content_lower:
            response["confidence"] = "low"
            
        return response
    
    def _get_disclaimer(self, urgency: str) -> str:
        """Get appropriate disclaimer based on urgency"""
        base_disclaimer = "⚠️ **Medical Disclaimer**: This information is for educational purposes only and should not replace professional medical advice, diagnosis, or treatment."
        
        if urgency == "emergency":
            return f"🚨 **URGENT**: {base_disclaimer} If this is a medical emergency, please contact emergency services immediately."
        elif urgency == "urgent":
            return f"⚡ **IMPORTANT**: {base_disclaimer} Please consult with a healthcare provider promptly."
        else:
            return f"{base_disclaimer} Always consult with qualified healthcare professionals for medical concerns."

def initialize_session_state():
    """Initialize session state variables"""
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = MedicalQASystem()
    if 'total_queries' not in st.session_state:
        st.session_state.total_queries = 0

def main():
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🩺 MedAssist AI</h1>
        <p>Your AI-Powered Medical Information Assistant</p>
        <p><em>Evidence-based medical information at your fingertips</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 👤 User Profile")
        
        user_type = st.selectbox(
            "Select your profile:",
            ["Patient/General Public", "Healthcare Professional", "Medical Student"],
            help="This helps customize the response complexity and terminology"
        )
        
        st.markdown("### 📊 Query Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{st.session_state.total_queries}</h3>
                <p>Total Queries</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(st.session_state.query_history)}</h3>
                <p>Session Queries</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        if st.button("🔄 Clear History"):
            st.session_state.query_history = []
            st.rerun()
            
        if st.button("📁 Export History"):
            if st.session_state.query_history:
                history_df = pd.DataFrame(st.session_state.query_history)
                csv = history_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"medical_qa_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        # Recent Queries
        if st.session_state.query_history:
            st.markdown("### 📋 Recent Queries")
            for i, query_data in enumerate(reversed(st.session_state.query_history[-5:])):
                st.markdown(f"""
                <div class="query-history">
                    <small>{query_data['timestamp']}</small><br>
                    <strong>{query_data['query'][:50]}...</strong>
                </div>
                """, unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 Ask Your Medical Question")
        
        # Query input
        query = st.text_area(
            "Enter your medical question:",
            placeholder="e.g., What are the common symptoms of hypertension?",
            height=100,
            help="Be as specific as possible for better results"
        )
        
        # Additional context
        with st.expander("➕ Additional Context (Optional)"):
            age_range = st.selectbox(
                "Age Range:",
                ["Not specified", "0-12", "13-17", "18-30", "31-50", "51-70", "70+"]
            )
            
            urgency = st.selectbox(
                "Urgency Level:",
                ["Routine Information", "Moderate Concern", "Urgent Concern"],
                help="This helps prioritize and format the response appropriately"
            )
            
            additional_context = st.text_area(
                "Additional Context:",
                placeholder="Any relevant medical history, current medications, or specific concerns...",
                height=60
            )
        
        # Submit button
        if st.button("🔍 Get Medical Information", type="primary"):
            if query.strip():
                # Process query
                context = {
                    "user_type": user_type,
                    "age_range": age_range,
                    "urgency": urgency.lower().replace(" ", "_"),
                    "additional_context": additional_context
                }
                
                # Classify query
                classification = st.session_state.qa_system.classify_query(query)
                context.update(classification)
                
                # Generate response
                with st.spinner("🧠 Processing your medical query..."):
                    response = st.session_state.qa_system.generate_medical_response(
                        query, user_type, context
                    )
                
                if "error" not in response:
                    # Display response
                    st.markdown(f"""
                    <div class="medical-response">
                        <h3>📋 Medical Information Response</h3>
                        {response['answer']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence indicator
                    confidence = response.get('confidence', 'medium')
                    confidence_class = f"confidence-{confidence}"
                    confidence_emoji = {"high": "✅", "medium": "⚠️", "low": "❌"}
                    
                    st.markdown(f"""
                    <div class="{confidence_class}">
                        {confidence_emoji[confidence]} <strong>Confidence Level:</strong> {confidence.title()}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Related topics
                    if response.get('related_topics'):
                        st.markdown("**🔗 Related Topics:**")
                        for topic in response['related_topics']:
                            st.markdown(f"• {topic}")
                    
                    # Follow-up questions
                    if response.get('follow_up_questions'):
                        st.markdown("**❓ Suggested Follow-up Questions:**")
                        for question in response['follow_up_questions']:
                            if st.button(f"💭 {question}", key=f"followup_{hash(question)}"):
                                st.session_state.temp_query = question
                    
                    # Disclaimer
                    st.markdown(f"""
                    <div class="disclaimer-box">
                        {response['disclaimer']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Save to history
                    query_data = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "query": query,
                        "user_type": user_type,
                        "response": response['answer'][:200] + "...",
                        "confidence": confidence
                    }
                    st.session_state.query_history.append(query_data)
                    st.session_state.total_queries += 1
                
            else:
                st.warning("⚠️ Please enter a medical question to get started.")
    
    with col2:
        st.markdown("### 📚 Medical Resources")
        
        st.markdown("""
        <div class="user-type-card">
            <h4>🏥 Emergency Resources</h4>
            <p><strong>Emergency:</strong> 911 (US)</p>
            <p><strong>Poison Control:</strong> 1-800-222-1222</p>
            <p><strong>Crisis Hotline:</strong> 988</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="user-type-card">
            <h4>🔬 Reliable Medical Sources</h4>
            <ul>
                <li>Mayo Clinic</li>
                <li>WebMD</li>
                <li>MedlinePlus</li>
                <li>CDC Guidelines</li>
                <li>WHO Resources</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="user-type-card">
            <h4>💡 Tips for Better Queries</h4>
            <ul>
                <li>Be specific about symptoms</li>
                <li>Include duration and severity</li>
                <li>Mention relevant medical history</li>
                <li>Ask focused questions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🩺 <strong>MedAssist AI</strong> - Built for healthcare education and information</p>
        <p><em>Always consult healthcare professionals for medical advice, diagnosis, and treatment</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
