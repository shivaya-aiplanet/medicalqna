import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from litellm import completion
from datetime import datetime
import pandas as pd
import os
import time
import json

# Page config
st.set_page_config(
    page_title="MedAssist AI - Your Caring Medical Companion",
    page_icon="🩺",
    layout="wide"
)

# Enhanced CSS for modern chatbot UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }

    /* Header styling */
    .chat-header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 1.5rem 2rem;
        border-radius: 20px;
        color: #2d3748;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .chat-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .chat-header p {
        font-size: 1rem;
        font-weight: 400;
        margin: 0.5rem 0 0;
        color: #718096;
    }

    /* Chat container */
    .chat-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        max-height: 500px;
        overflow-y: auto;
    }

    /* Message styling */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 0.5rem 0 0.5rem auto;
        max-width: 80%;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        animation: slideInRight 0.3s ease-out;
    }

    .bot-message {
        background: rgba(247, 250, 252, 0.9);
        color: #2d3748;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 0.5rem auto 0.5rem 0;
        max-width: 85%;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        animation: slideInLeft 0.3s ease-out;
    }

    .system-message {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 15px;
        margin: 0.5rem auto;
        text-align: center;
        font-size: 0.9rem;
        box-shadow: 0 4px 15px rgba(72, 187, 120, 0.3);
        animation: fadeIn 0.5s ease-out;
    }

    .typing-indicator {
        background: rgba(247, 250, 252, 0.9);
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 0.5rem auto 0.5rem 0;
        max-width: 85%;
        border-left: 4px solid #667eea;
        animation: pulse 1.5s infinite;
    }

    .typing-dots {
        display: inline-block;
    }

    .typing-dots span {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #667eea;
        margin: 0 2px;
        animation: typing 1.4s infinite ease-in-out;
    }

    .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
    .typing-dots span:nth-child(2) { animation-delay: -0.16s; }
    .typing-dots span:nth-child(3) { animation-delay: 0s; }

    @keyframes typing {
        0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
        40% { transform: scale(1.2); opacity: 1; }
    }

    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }

    @keyframes slideInLeft {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse {
        0%, 100% { opacity: 0.8; }
        50% { opacity: 1; }
    }

    /* Input area styling */
    .input-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .sidebar-header {
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 1rem;
        text-align: center;
    }

    .mood-indicator {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
        color: white;
    }

    .mood-emoji {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }

    .conversation-stats {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        color: white;
    }

    .stat-item {
        display: flex;
        justify-content: space-between;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }

    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 15px;
        border: 2px solid rgba(102, 126, 234, 0.3);
        padding: 1rem;
        font-size: 1rem;
        background: rgba(255, 255, 255, 0.9);
        transition: all 0.3s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    /* Disclaimer styling */
    .disclaimer {
        background: rgba(254, 178, 178, 0.2);
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 1rem;
        border-radius: 15px;
        color: #991b1b;
        font-size: 0.9rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }

    /* Remove default elements */
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 6px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(102, 126, 234, 0.5);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(102, 126, 234, 0.7);
    }
</style>
""", unsafe_allow_html=True)

class MedicalChatbot:
    def __init__(self):
        # Set litellm environment variables
        os.environ["LITELLM_API_KEY"] = st.secrets["LITELLM_API_KEY"]
        os.environ["LITELLM_BASE_URL"] = st.secrets["LITELLM_BASE_URL"]
        self.model = st.secrets["LITELLM_MODEL"]
        
        # Initialize Hugging Face embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize vector store with enhanced medical knowledge
        self.vector_store = self._initialize_vector_store()
        
        # Conversation context and personality
        self.personality = {
            "empathetic": True,
            "supportive": True,
            "professional": True,
            "caring": True
        }

    def _initialize_medical_knowledge(self):
        """Enhanced medical knowledge base with more comprehensive information"""
        knowledge = """
        Common Symptoms and Their Meanings:
        
        Fever: Elevated body temperature above 100.4°F (38°C), often indicating infection, inflammation, or immune response. Can be accompanied by chills, sweating, headache, and fatigue. Usually managed with rest, fluids, and fever reducers like acetaminophen or ibuprofen.
        
        Headache: Pain in the head or upper neck region. Types include tension headaches (most common), migraines (severe, often with nausea and light sensitivity), cluster headaches, and secondary headaches. Triggers can include stress, dehydration, certain foods, lack of sleep, or underlying conditions.
        
        Chest Pain: Discomfort or pain in the chest area. Can range from sharp, stabbing pain to dull aches. Cardiac causes include heart attack, angina, or pericarditis. Non-cardiac causes include muscle strain, anxiety, acid reflux, or lung conditions. Severe chest pain requires immediate medical attention.
        
        Shortness of Breath (Dyspnea): Difficulty breathing or feeling breathless. Can be acute (sudden) or chronic (long-term). Causes include asthma, pneumonia, heart conditions, anxiety, or physical exertion. Severe breathing difficulties require immediate medical care.
        
        Abdominal Pain: Pain in the stomach area. Can be cramping, sharp, dull, or burning. Common causes include indigestion, gas, constipation, gastroenteritis, appendicitis, or gynecological issues. Location and type of pain can help determine the cause.
        
        Nausea and Vomiting: Feeling sick to the stomach and potentially vomiting. Causes include food poisoning, viral infections, motion sickness, pregnancy, medications, or underlying medical conditions.
        
        Fatigue: Persistent tiredness or lack of energy. Can be physical, mental, or both. Causes include lack of sleep, stress, poor nutrition, dehydration, infections, or chronic conditions like anemia or thyroid disorders.
        
        Common Medications and Information:
        
        Acetaminophen (Tylenol): Pain reliever and fever reducer. Generally safe when used as directed. Maximum daily dose is 3000-4000mg for adults. Can cause liver damage if overdosed or combined with alcohol.
        
        Ibuprofen (Advil, Motrin): Anti-inflammatory pain reliever. Reduces pain, fever, and inflammation. Can cause stomach upset, so take with food. Avoid if you have kidney problems or stomach ulcers.
        
        Aspirin: Pain reliever, fever reducer, and blood thinner. Low-dose aspirin is often prescribed for heart protection. Can interact with blood thinners and cause stomach bleeding.
        
        Antibiotics: Medications that fight bacterial infections. Only effective against bacteria, not viruses. Must be taken as prescribed, even if feeling better. Common types include amoxicillin, azithromycin, and ciprofloxacin.
        
        Antacids: Neutralize stomach acid to relieve heartburn and indigestion. Examples include Tums, Rolaids, and Maalox. Provide quick but temporary relief.
        
        Preventive Care and Wellness:
        
        Regular Exercise: Aim for 150 minutes of moderate aerobic activity weekly. Benefits include improved cardiovascular health, stronger bones, better mood, and disease prevention.
        
        Balanced Nutrition: Include fruits, vegetables, whole grains, lean proteins, and healthy fats. Stay hydrated with adequate water intake. Limit processed foods, excessive sugar, and sodium.
        
        Sleep Hygiene: Adults need 7-9 hours of quality sleep nightly. Maintain consistent sleep schedule, create comfortable sleep environment, and avoid screens before bedtime.
        
        Stress Management: Chronic stress affects physical and mental health. Practice relaxation techniques, regular exercise, adequate sleep, and social connections. Consider professional help if needed.
        
        Regular Check-ups: Annual physical exams, dental cleanings, eye exams, and age-appropriate screenings help detect problems early when they're most treatable.
        
        Mental Health Awareness:
        
        Anxiety: Excessive worry or fear that interferes with daily life. Symptoms include restlessness, rapid heartbeat, sweating, and difficulty concentrating. Treatment options include therapy, medication, and lifestyle changes.
        
        Depression: Persistent feelings of sadness, hopelessness, or loss of interest in activities. Can affect sleep, appetite, energy, and concentration. Professional help is available and effective.
        
        Emergency Warning Signs:
        
        Seek immediate medical attention for: chest pain with shortness of breath, severe abdominal pain, difficulty breathing, signs of stroke (facial drooping, arm weakness, speech difficulties), severe allergic reactions, or any symptom that feels life-threatening.
        
        When to See a Doctor:
        
        Persistent symptoms lasting more than a few days, worsening symptoms, high fever, severe pain, unusual changes in your body, or when you're concerned about your health. Trust your instincts about your body.
        """
        return knowledge

    def _initialize_vector_store(self):
        """Split text and create FAISS vector store with enhanced knowledge"""
        text = self._initialize_medical_knowledge()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        chunks = text_splitter.split_text(text)
        return FAISS.from_texts(chunks, self.embeddings)

    def retrieve_relevant_info(self, query: str, conversation_history: list, top_k: int = 4):
        """Retrieve relevant documents considering conversation context"""
        # Combine current query with recent conversation context
        context_query = query
        if conversation_history:
            recent_messages = conversation_history[-3:]  # Last 3 exchanges
            context_text = " ".join([msg['content'] for msg in recent_messages])
            context_query = f"{context_text} {query}"
        
        docs = self.vector_store.similarity_search(context_query, k=top_k)
        return [doc.page_content for doc in docs]

    def analyze_user_mood(self, message: str):
        """Simple mood analysis based on keywords"""
        worry_words = ['worried', 'scared', 'anxious', 'afraid', 'concerned', 'nervous', 'panic']
        pain_words = ['hurt', 'pain', 'ache', 'sore', 'painful', 'burning', 'stabbing']
        sad_words = ['sad', 'depressed', 'down', 'upset', 'crying', 'hopeless']
        
        message_lower = message.lower()
        
        if any(word in message_lower for word in worry_words):
            return {"mood": "worried", "emoji": "😟", "response_tone": "reassuring"}
        elif any(word in message_lower for word in pain_words):
            return {"mood": "in_pain", "emoji": "😣", "response_tone": "caring"}
        elif any(word in message_lower for word in sad_words):
            return {"mood": "sad", "emoji": "😢", "response_tone": "supportive"}
        else:
            return {"mood": "neutral", "emoji": "🙂", "response_tone": "friendly"}

    def generate_empathetic_response(self, query: str, user_type: str, conversation_history: list, user_mood: dict):
        """Generate empathetic response using RAG with conversation memory"""
        # Retrieve relevant information
        relevant_docs = self.retrieve_relevant_info(query, conversation_history)
        context = "\n".join(relevant_docs)
        
        # Build conversation context
        conversation_context = ""
        if conversation_history:
            recent_history = conversation_history[-4:]  # Last 4 exchanges
            conversation_context = "\n".join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in recent_history
            ])

        # Create empathetic system prompt based on mood
        empathy_instructions = {
            "worried": "The user seems worried or anxious. Be extra reassuring, acknowledge their concerns, and provide calm, supportive guidance.",
            "in_pain": "The user appears to be in pain or discomfort. Show compassion, validate their experience, and provide helpful information with care.",
            "sad": "The user seems sad or down. Be particularly supportive, encouraging, and gentle in your response.",
            "neutral": "Maintain a warm, caring, and professional tone while being helpful and informative."
        }

        system_prompt = f"""You are MedAssist AI, a caring and empathetic medical information assistant. You have a warm, compassionate personality and truly care about helping people with their health concerns.

PERSONALITY TRAITS:
- Deeply empathetic and understanding
- Warm and caring in tone
- Professional yet personable
- Supportive and reassuring
- Never dismissive of concerns
- Acknowledges emotions and validates feelings

USER CONTEXT:
- User type: {user_type}
- Current mood: {user_mood['mood']} {user_mood['emoji']}
- Response approach: {empathy_instructions[user_mood['mood']]}

CONVERSATION HISTORY:
{conversation_context}

MEDICAL KNOWLEDGE CONTEXT:
{context}

INSTRUCTIONS:
1. Always acknowledge the user's feelings and concerns first
2. Provide {'detailed, technical' if user_type == 'Healthcare Professional' else 'clear, easy-to-understand'} information
3. Be encouraging and supportive
4. Use appropriate emotional language and empathy
5. Remember previous parts of the conversation
6. Always end with care and support
7. Include relevant medical information from the context
8. Always include appropriate disclaimers about consulting healthcare professionals

Current user message: {query}

Respond with empathy, care, and helpful medical information."""
        
        try:
            response = completion(
                model=self.model,
                api_base=st.secrets["LITELLM_BASE_URL"],
                api_key=st.secrets["LITELLM_API_KEY"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,  # Slightly higher for more empathetic responses
                max_tokens=1200
            )
            
            answer = response.choices[0].message.content
            return {"answer": answer, "mood_detected": user_mood}
            
        except Exception as e:
            return {"error": str(e)}

def initialize_session_state():
    """Initialize session state for chatbot"""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = MedicalChatbot()
    if 'user_mood_history' not in st.session_state:
        st.session_state.user_mood_history = []
    if 'conversation_started' not in st.session_state:
        st.session_state.conversation_started = False
    if 'typing' not in st.session_state:
        st.session_state.typing = False

def display_message(role: str, content: str, timestamp: str = None, mood: dict = None):
    """Display a message in the chat interface"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%H:%M")
    
    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            {content}
            <div style="font-size: 0.8em; opacity: 0.7; margin-top: 0.5rem;">
                {timestamp} {mood['emoji'] if mood else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif role == "assistant":
        st.markdown(f"""
        <div class="bot-message">
            {content}
            <div style="font-size: 0.8em; opacity: 0.7; margin-top: 0.5rem;">
                🩺 MedAssist • {timestamp}
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif role == "system":
        st.markdown(f"""
        <div class="system-message">
            {content}
        </div>
        """, unsafe_allow_html=True)

def display_typing_indicator():
    """Display typing indicator"""
    st.markdown("""
    <div class="typing-indicator">
        <div class="typing-dots">
            <span></span>
            <span></span>
            <span></span>
        </div>
        <span style="margin-left: 10px; color: #667eea;">MedAssist is thinking...</span>
    </div>
    """, unsafe_allow_html=True)

def get_conversation_stats():
    """Calculate conversation statistics"""
    total_messages = len(st.session_state.conversation_history)
    user_messages = len([msg for msg in st.session_state.conversation_history if msg['role'] == 'user'])
    
    mood_counts = {}
    for mood_data in st.session_state.user_mood_history:
        mood = mood_data['mood']
        mood_counts[mood] = mood_counts.get(mood, 0) + 1
    
    dominant_mood = max(mood_counts.items(), key=lambda x: x[1])[0] if mood_counts else "neutral"
    
    return {
        "total_messages": total_messages,
        "user_messages": user_messages,
        "dominant_mood": dominant_mood,
        "session_start": st.session_state.get('session_start', datetime.now())
    }

def main():
    initialize_session_state()
    
    # Initialize session start time
    if 'session_start' not in st.session_state:
        st.session_state.session_start = datetime.now()

    # Header
    st.markdown("""
    <div class="chat-header">
        <h1>🩺 MedAssist AI</h1>
        <p>Your Caring Medical Companion - Here to listen, understand, and help 💙</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar with user profile and stats
    with st.sidebar:
        st.markdown('<div class="sidebar-header">👤 Your Profile</div>', unsafe_allow_html=True)
        
        user_type = st.selectbox(
            "I am a:",
            ["Patient/General Public", "Healthcare Professional"],
            help="This helps me tailor my responses to your level of medical knowledge"
        )

        # Conversation stats
        stats = get_conversation_stats()
        st.markdown(f"""
        <div class="conversation-stats">
            <h4 style="color: white; text-align: center; margin-bottom: 1rem;">📊 Chat Session</h4>
            <div class="stat-item">
                <span>Messages:</span>
                <span>{stats['total_messages']}</span>
            </div>
            <div class="stat-item">
                <span>Your questions:</span>
                <span>{stats['user_messages']}</span>
            </div>
            <div class="stat-item">
                <span>Session time:</span>
                <span>{int((datetime.now() - stats['session_start']).total_seconds() / 60)}m</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Current mood indicator
        if st.session_state.user_mood_history:
            latest_mood = st.session_state.user_mood_history[-1]
            mood_descriptions = {
                "worried": "You seem concerned - I'm here to help ease your worries",
                "in_pain": "I understand you're uncomfortable - let's work through this",
                "sad": "I can sense you're feeling down - you're not alone",
                "neutral": "Great to chat with you today!"
            }
            
            st.markdown(f"""
            <div class="mood-indicator">
                <div class="mood-emoji">{latest_mood['emoji']}</div>
                <div style="font-size: 0.9rem;">{mood_descriptions[latest_mood['mood']]}</div>
            </div>
            """, unsafe_allow_html=True)

        # Control buttons
        if st.button("🔄 New Conversation"):
            st.session_state.conversation_history = []
            st.session_state.user_mood_history = []
            st.session_state.conversation_started = False
            st.session_state.session_start = datetime.now()
            st.rerun()

        if st.button("💾 Save Chat History"):
            if st.session_state.conversation_history:
                chat_data = {
                    "timestamp": datetime.now().isoformat(),
                    "user_type": user_type,
                    "conversation": st.session_state.conversation_history,
                    "mood_history": st.session_state.user_mood_history
                }
                st.download_button(
                    label="📱 Download Chat",
                    data=json.dumps(chat_data, indent=2),
                    file_name=f"medassist_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

    # Main chat interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Welcome message for new users
    if not st.session_state.conversation_started:
        display_message("system", "💙 Welcome! I'm MedAssist AI, your caring medical companion. I'm here to listen to your health concerns and provide helpful information with empathy and understanding. How are you feeling today?")
        st.session_state.conversation_started = True

    # Display conversation history
    for i, message in enumerate(st.session_state.conversation_history):
        mood_data = st.session_state.user_mood_history[i//2] if message['role'] == 'user' and i//2 < len(st.session_state.user_mood_history) else None
        display_message(
            message['role'], 
            message['content'], 
            message.get('timestamp', ''), 
            mood_data
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Input area
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.markdown("### 💬 Share your health concerns, symptoms, or questions:")
    
    # Create columns for input and button
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Message MedAssist...",
            placeholder="e.g., I've been having headaches for 3 days and I'm worried...",
            key="user_message_input",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send 💙", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Process user input
    if send_button and user_input.strip():
        # Analyze user mood
        user_mood = st.session_state.chatbot.analyze_user_mood(user_input)
        st.session_state.user_mood_history.append(user_mood)
        
        # Add user message to conversation
        user_message = {
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%H:%M")
        }
        st.session_state.conversation_history.append(user_message)
        
        # Display typing indicator
        st.session_state.typing = True
        st.rerun()

    # Show typing indicator when processing
    if st.session_state.typing:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        display_typing_indicator()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Generate response
        response = st.session_state.chatbot.generate_empathetic_response(
            st.session_state.conversation_history[-1]['content'],
            user_type,
            st.session_state.conversation_history[:-1],  # Exclude current message
            st.session_state.user_mood_history[-1]
        )
        
        st.session_state.typing = False
        
        if "error" not in response:
            # Add assistant response to conversation
            assistant_message = {
                "role": "assistant",
                "content": response['answer'],
                "timestamp": datetime.now().strftime("%H:%M")
            }
            st.session_state.conversation_history.append(assistant_message)
            
            # Clear input and refresh
            st.session_state.user_message_input = ""
            st.rerun()
        else:
            st.error(f"I'm sorry, I encountered an issue: {response['error']}. Please try again.")
            st.session_state.typing = False

    # Quick action buttons for common concerns
    st.markdown("### 🚀 Quick Help Options:")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🤒 Fever & Symptoms"):
            quick_message = "I have a fever and I'm not feeling well. What should I know about fever and when should I be concerned?"
            st.session_state.conversation_history.append({
                "role": "user",
                "content": quick_message,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            st.session_state.user_mood_history.append({"mood": "worried", "emoji": "😟", "response_tone": "reassuring"})
            st.session_state.typing = True
            st.rerun()
    
    with col2:
        if st.button("💊 Medication Info"):
            quick_message = "I need information about medications and their effects. Can you help me understand what I should know?"
            st.session_state.conversation_history.append({
                "role": "user",
                "content": quick_message,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            st.session_state.user_mood_history.append({"mood": "neutral", "emoji": "🙂", "response_tone": "friendly"})
            st.session_state.typing = True
            st.rerun()
    
    with col3:
        if st.button("😰 Anxiety & Stress"):
            quick_message = "I've been feeling anxious and stressed lately. Can you help me understand these feelings and what might help?"
            st.session_state.conversation_history.append({
                "role": "user",
                "content": quick_message,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            st.session_state.user_mood_history.append({"mood": "worried", "emoji": "😟", "response_tone": "reassuring"})
            st.session_state.typing = True
            st.rerun()
    
    with col4:
        if st.button("🏥 When to See Doctor"):
            quick_message = "I'm not sure if my symptoms are serious enough to see a doctor. Can you help me understand when I should seek medical care?"
            st.session_state.conversation_history.append({
                "role": "user",
                "content": quick_message,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            st.session_state.user_mood_history.append({"mood": "worried", "emoji": "😟", "response_tone": "reassuring"})
            st.session_state.typing = True
            st.rerun()

    # Emergency notice
    st.markdown("""
    <div class="disclaimer">
        <h4 style="color: #991b1b; margin: 0 0 0.5rem 0;">🚨 Important Medical Disclaimer</h4>
        <p style="margin: 0; font-size: 0.9rem;">
            <strong>Emergency situations:</strong> If you're experiencing severe chest pain, difficulty breathing, 
            signs of stroke, severe bleeding, or any life-threatening emergency, please call emergency services 
            immediately (911 in the US) or go to your nearest emergency room.
        </p>
        <br>
        <p style="margin: 0; font-size: 0.9rem;">
            <strong>Medical advice:</strong> I provide general health information and emotional support, but I cannot 
            replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare 
            providers for medical concerns and before making health decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Helpful resources section
    with st.expander("📚 Helpful Health Resources"):
        st.markdown("""
        **🆘 Emergency Services:**
        - **Emergency:** 911 (US), 999 (UK), 112 (EU)
        - **Poison Control:** 1-800-222-1222 (US)
        - **Crisis Text Line:** Text HOME to 741741
        
        **🔍 Trusted Medical Resources:**
        - **Mayo Clinic:** mayoclinic.org
        - **WebMD:** webmd.com  
        - **MedlinePlus:** medlineplus.gov
        - **CDC:** cdc.gov
        
        **🧠 Mental Health Resources:**
        - **National Suicide Prevention Lifeline:** 988
        - **Crisis Text Line:** Text HOME to 741741
        - **Mental Health America:** mhanational.org
        
        **💡 Remember:** I'm here to provide information and support, but these resources can offer additional help when you need it most.
        """)

    # Footer with care message
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: rgba(255, 255, 255, 0.8); font-size: 0.9rem;">
        💙 Take care of yourself - your health and wellbeing matter. I'm here whenever you need support. 💙
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
