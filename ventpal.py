# SQLite monkey-patch for ChromaDB compatibility
import sys
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import streamlit as st
import time
import hashlib
import os
import random
import re
import unicodedata
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from huggingface_hub import InferenceClient

# Page configuration
st.set_page_config(
    page_title="VentPal - Mental Health Support",
    page_icon="💨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .crisis-alert {
        background-color: #ffebee;
        border: 2px solid #f44336;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .emotion-indicator {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
    .source-citation {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
        margin-top: 0.5rem;
    }
    .typing-indicator {
        display: inline-block;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .skill-badge {
        display: inline-block;
        background: #e8f5e8;
        color: #2e7d32;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.7rem;
        margin-top: 0.5rem;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# --- GDPR Settings ---
GDPR_COMPLIANT = True  # Session-based data only

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "request_count" not in st.session_state:
    st.session_state.request_count = 0
if "last_reset" not in st.session_state:
    st.session_state.last_reset = time.time()
if "user_id" not in st.session_state:
    st.session_state.user_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = ""
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "session_goals" not in st.session_state:
    st.session_state.session_goals = ""
if "session_start_time" not in st.session_state:
    st.session_state.session_start_time = time.time()
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        max_token_limit=2000  # Limit memory to prevent token overflow
    )
if "recent_cues" not in st.session_state:
    st.session_state.recent_cues = []
if "used_skills" not in st.session_state:
    st.session_state.used_skills = []
if "last_skill" not in st.session_state:
    st.session_state.last_skill = ""
if "conversation_backup" not in st.session_state:
    st.session_state.conversation_backup = []

# Set deterministic random seed for consistent micro-cue ordering
random.seed(st.session_state.user_id)

# Configuration - now from secrets where possible
HUGGINGFACE_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY", "")
MODEL_NAME = st.secrets.get("MODEL_NAME", "meta-llama/Llama-4-Scout-17B-16E-Instruct")
EMBEDDING_MODEL = st.secrets.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VECTOR_DB_PATH = st.secrets.get("VECTOR_DB_PATH", "vector_db")
MAX_REQUESTS_PER_HOUR = st.secrets.get("MAX_REQUESTS_PER_HOUR", 50)

# Enhanced micro-cue library (50 items, grouped by use-case)
CUE_GROUPS = {
    "start": [
        "Hi there.", "Nice to meet you.", "Glad you're here.", "Welcome."
    ],
    "heavy_emotion": [
        "I'm really hearing you.", "That sounds so tough.", "I can feel how heavy that is.", 
        "That must be exhausting.", "I'm sorry it's hurting."
    ],
    "mild_distress": [
        "I'm here with you.", "Tell me more.", "I'm listening.", "Take your time.", 
        "Go ahead, I'm all ears."
    ],
    "positive_progress": [
        "That's a big step!", "I love that.", "Nice work.", "That's encouraging.", 
        "You're making progress."
    ],
    "user_reflects": [
        "That's an important realisation.", "That makes sense.", "I see what you mean.", 
        "Good observation.", "That's a useful way to look at it."
    ],
    "user_stuck": [
        "Let's slow it down.", "One step at a time.", "We can figure this out together.", 
        "Let's unpack that.", "We can sit with it for a moment."
    ],
    "offering_strategy": [
        "Here's something that might help.", "Want to try a quick exercise?", 
        "Can I share a technique?", "Would you like a tool for that?", 
        "We could test a small idea."
    ],
    "encouraging": [
        "You've got this.", "I'm rooting for you.", "You're not alone in this.", 
        "I believe in your ability.", "You can handle this."
    ],
    "crisis_safety": [
        "Your safety matters most.", "I'm concerned about you.", "Let's keep you safe first."
    ],
    "neutral": [
        "Mmm-hmm.", "Uh-huh.", "I hear you."
    ]
}

# Enhanced CBT Micro-skills with regex patterns for tracking
CBT_SKILLS = {
    "breathing": {
        "name": "Deep Breathing",
        "description": "4-6 breathing technique",
        "patterns": [r"deep ?breath", r"4-?6 ?breath", r"breathing ?exercise", r"inhale.*exhale"]
    },
    "grounding": {
        "name": "5-4-3-2-1 Grounding",
        "description": "Sensory grounding technique",
        "patterns": [r"5-?4-?3-?2-?1", r"grounding", r"five ?senses", r"sensory.*technique"]
    },
    "thought_challenging": {
        "name": "Thought Challenging",
        "description": "Cognitive restructuring",
        "patterns": [r"thought.*challenge", r"cognitive.*restructur", r"reframe", r"alternative.*thought"]
    },
    "behavioral_activation": {
        "name": "Behavioral Activation",
        "description": "Activity scheduling",
        "patterns": [r"behavioral.*activation", r"activity.*schedul", r"small.*step", r"gradual.*exposure"]
    },
    "mindfulness": {
        "name": "Mindfulness",
        "description": "Present moment awareness",
        "patterns": [r"mindful", r"present.*moment", r"observe.*thought", r"non-?judgment"]
    },
    "progressive_relaxation": {
        "name": "Progressive Relaxation",
        "description": "Muscle relaxation technique",
        "patterns": [r"progressive.*relax", r"muscle.*tension", r"body.*scan", r"relaxation.*technique"]
    }
}

# Enhanced crisis detection patterns
CRISIS_PATTERNS = [
    r"kill.*myself", r"suicide", r"want.*die", r"end.*life", r"hurt.*myself",
    r"self.*harm", r"cut.*myself", r"overdose", r"take.*pills", r"jump.*off",
    r"hang.*myself", r"shoot.*myself", r"drown.*myself", r"burn.*myself"
]

# World-class prompt templates
SYSTEM_PROMPT = """You are VentPal, a mental health companion trained in evidence-based Cognitive Behavioural Therapy (CBT).  
Primary goal: relieve distress in a single turn while building rapport for the next turn.  
Secondary goal: teach one CBT micro-skill at a time, only if user is ready.  
You are NOT a medical professional and never prescribe or diagnose."""

STYLE_PROMPT = """• Always speak in first-person singular ("I", "me", "we can…")  
• Use contractions (you're, it's) and everyday vocabulary.  
• Keep replies ≤ 120 words → feel like chat, not essay.  
• Follow the trio:  
  1. Validation (acknowledge feeling)  
  2. Exploration (open question OR gentle reflection)  
  3. Offer (one micro-skill or tiny suggestion (optional), ≤ 40 words.  
• Insert a micro back-channel every 2–3 replies ("I'm with you…", "Mmm-hmm…")  
• Finish with exactly **one** open question unless user asked for facts only.
• Remember previous context and build on earlier conversations."""

SAFETY_PROMPT = """If user expresses self-harm or suicidal intent:
    ➊ Acknowledge the pain ("I'm really sorry you're feeling…")  
    ➋ Encourage immediate professional help (show crisis numbers)  
    ➌ Stop CBT content; wait for user response before continuing.

If you are unsure whether content is safe → ask a clarifying question first.
Never mention policy or guidelines."""

# Gentle follow-up questions
GENTLE_FOLLOW_UPS = [
    "How does that feel to you?",
    "What's your take on that?",
    "Does that resonate with you?",
    "How does that sound?",
    "What do you think about that?",
    "Does that make sense to you?",
    "How does that land with you?",
    "What's your experience with that?",
    "Does that feel right to you?",
    "How does that sit with you?"
]

# Crisis keywords and resources - UK focused
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "want to die", "end it all", "no reason to live",
    "better off dead", "hurt myself", "self-harm", "cutting", "overdose",
    "jump off", "hang myself", "shoot myself", "take pills", "bleed out"
]

CRISIS_RESOURCES = {
    "UK": {
        "Samaritans": "116 123",
        "Shout": "Text SHOUT to 85258",
        "Emergency": "999",
        "NHS 111": "111"
    }
}

# Emotion classification keywords
EMOTION_KEYWORDS = {
    "anxiety": ["anxious", "worried", "nervous", "panic", "fear", "stress", "overwhelmed", "can't breathe"],
    "depression": ["sad", "depressed", "hopeless", "worthless", "empty", "tired", "exhausted", "no energy"],
    "anger": ["angry", "furious", "irritated", "frustrated", "mad", "rage", "hate", "livid"],
    "grief": ["grief", "loss", "bereaved", "mourning", "missing", "gone", "heartbroken"],
    "joy": ["happy", "joy", "excited", "thrilled", "elated", "content", "peaceful", "good"]
}

def is_english_text(text: str) -> bool:
    """Check if text is primarily English using ASCII ratio."""
    if not text.strip():
        return True
    
    # Count non-ASCII characters
    non_ascii_count = sum(1 for char in text if ord(char) > 127)
    total_chars = len(text)
    
    # If more than 30% are non-ASCII, likely not English
    return (non_ascii_count / total_chars) <= 0.3

def sanitize_user_input(text: str) -> str:
    """Enhanced sanitization to prevent prompt injection."""
    if not text:
        return ""
    
    # Strip control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    
    # Limit length
    if len(text) > 1000:
        text = text[:1000] + "..."
    
    # Escape triple backticks to prevent nested markdown
    text = text.replace("```", "\\`\\`\\`")
    
    # Prevent suffix hijack by escaping prompt markers
    text = text.replace("</Assistant-reply>", "\\</Assistant-reply\\>")
    text = text.replace("<Assistant-reply>", "\\<Assistant-reply\\>")
    
    return text.strip()

def get_emotion(text: str) -> str:
    """Classify the primary emotion in the text."""
    if not is_english_text(text):
        return "neutral"
    
    text_lower = text.lower()
    emotion_scores = {}
    
    for emotion, keywords in EMOTION_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            emotion_scores[emotion] = score
    
    if emotion_scores:
        return max(emotion_scores, key=emotion_scores.get)
    return "neutral"

def detect_crisis(text: str) -> bool:
    """Detect crisis keywords in the text."""
    if not is_english_text(text):
        return False
    
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in CRISIS_PATTERNS)

def crisis_block(country: str = "UK") -> str:
    """Professional crisis response block for UK."""
    return (
        "🚨 I'm really sorry you're feeling like this—it sounds unbearably painful.\n\n"
        "If you feel you might act on these thoughts **right now**, please call **999** "
        "or go to your nearest A&E.\n\n"
        "If you can keep yourself safe for the next few minutes, please consider:\n"
        "• **Samaritans** 116 123 (24/7, free)\n"
        "• **Shout** – text **SHOUT** to 85258 (24/7)\n"
        "• **NHS 111** for urgent medical advice\n\n"
        "I'm here with you. Are you able to stay safe for the next few minutes?"
    )

def select_micro_cue(emotion: str, recent_cues: List[str]) -> str:
    """Select appropriate micro-cue based on emotion and avoid recent ones."""
    # Map emotions to cue groups
    emotion_to_group = {
        "crisis": "crisis_safety",
        "anxiety": "heavy_emotion",
        "depression": "heavy_emotion",
        "anger": "heavy_emotion",
        "grief": "heavy_emotion",
        "joy": "positive_progress",
        "neutral": "mild_distress"
    }
    
    group = emotion_to_group.get(emotion, "mild_distress")
    available_cues = [cue for cue in CUE_GROUPS[group] if cue not in recent_cues]
    
    # If no available cues in preferred group, expand to all groups
    if not available_cues:
        all_cues = [cue for group_cues in CUE_GROUPS.values() for cue in group_cues]
        available_cues = [cue for cue in all_cues if cue not in recent_cues]
    
    # If still no available cues, reset and use any cue
    if not available_cues:
        available_cues = [cue for group_cues in CUE_GROUPS.values() for cue in group_cues]
    
    return random.choice(available_cues)

def update_recent_cues(cue: str, recent_cues: List[str], max_recent: int = 5):
    """Update recent cues list, keeping only the last N."""
    recent_cues.append(cue)
    if len(recent_cues) > max_recent:
        recent_cues.pop(0)

def detect_skill_usage(response: str) -> Optional[str]:
    """Detect which CBT skill was offered in the response."""
    response_lower = response.lower()
    
    for skill_name, skill_info in CBT_SKILLS.items():
        for pattern in skill_info["patterns"]:
            if re.search(pattern, response_lower):
                return skill_name
    
    return None

def check_rate_limit() -> bool:
    """Check if user has exceeded rate limit."""
    current_time = time.time()
    
    # Reset counter if an hour has passed
    if current_time - st.session_state.last_reset > 3600:
        st.session_state.request_count = 0
        st.session_state.last_reset = current_time
    
    return st.session_state.request_count < MAX_REQUESTS_PER_HOUR

def increment_rate_limit():
    """Increment the request counter."""
    st.session_state.request_count += 1

def get_conversation_memory() -> str:
    """Get formatted conversation memory from LangChain buffer."""
    try:
        memory_variables = st.session_state.memory.load_memory_variables({})
        chat_history = memory_variables.get("chat_history", [])
        
        if not chat_history:
            return ""
        
        # Get memory_turns from session state, default to 3
        last_n = st.session_state.get("memory_turns", 3)
        
        # Format the conversation history (×2 because each "exchange" is one user + one assistant message)
        formatted_history = []
        for message in chat_history[-last_n*2:]:
            if isinstance(message, HumanMessage):
                formatted_history.append(f"User: {message.content}")
            elif isinstance(message, AIMessage):
                formatted_history.append(f"Assistant: {message.content}")
        
        return "\n".join(formatted_history)
    except Exception as e:
        st.error(f"Error loading memory: {str(e)}")
        return ""

def update_memory(user_message: str, assistant_message: str):
    """Update the conversation memory."""
    try:
        st.session_state.memory.chat_memory.add_user_message(user_message)
        st.session_state.memory.chat_memory.add_ai_message(assistant_message)
    except Exception as e:
        st.error(f"Error updating memory: {str(e)}")

def ensure_follow_up_question(response: str) -> str:
    """Ensure the response ends with a follow-up question."""
    response = response.strip()
    
    # If it already ends with a question, return as is
    if response.endswith('?'):
        return response
    
    # If it ends with a period, replace with a question
    if response.endswith('.'):
        response = response[:-1] + '?'
        return response
    
    # If it doesn't end with punctuation, add a question
    if not response.endswith(('.', '!', '?')):
        response += '?'
        return response
    
    # If it ends with exclamation, add a question after
    if response.endswith('!'):
        response += ' What do you think?'
        return response
    
    # Default fallback - add a gentle follow-up
    response += ' How does that sound to you?'
    return response

@st.cache_resource
def get_hf_client():
    """Cached HuggingFace client to reuse HTTPS connections."""
    return InferenceClient(
        provider="auto",
        api_key=HUGGINGFACE_API_KEY,
    )

def polish_chunks_with_llama(raw_chunks: List[str], user_query: str) -> str:
    """Use Llama to polish and summarize relevant chunks."""
    if not raw_chunks:
        return ""
    
    # Skip polishing for ≤ 2 chunks to save latency
    if len(raw_chunks) <= 2:
        return "\n\n".join(raw_chunks)
    
    try:
        # Combine chunks and create polishing prompt
        combined_chunks = "\n\n".join(raw_chunks)
        
        polish_prompt = f"""You are a mental health expert. Below are relevant CBT techniques and strategies from our knowledge base. 

**User's current situation:** {user_query}

**Raw knowledge chunks:**
{combined_chunks}

**Task:** Polish and summarize the most relevant information for this user's situation. 
- Keep it concise (≤150 words)
- Focus on practical, actionable advice
- Use warm, supportive language
- Ensure it flows naturally
- Remove any redundant or irrelevant information

**Polished response:**"""

        # Use cached client to polish
        client = get_hf_client()
        
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": polish_prompt}],
            max_tokens=150,  # Reduced from 300 to save latency
            temperature=0.3  # Lower temperature for more focused polishing
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        st.error(f"Error polishing chunks: {str(e)}")
        # Fallback to simple concatenation
        return "\n\n".join(raw_chunks[:3])  # Just use first 3 chunks

@st.cache_resource(show_spinner=False)
def create_vector_store():
    """Load existing vector store with your PDF chunks."""
    try:
        # Late import to speed up cold start
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        # Load your existing vector store
        if os.path.exists(VECTOR_DB_PATH):
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            vectorstore = Chroma(
                persist_directory=VECTOR_DB_PATH,
                embedding_function=embeddings
            )
            st.success(f"✅ Loaded existing vector store from {VECTOR_DB_PATH}")
            return vectorstore
        else:
            st.error(f"❌ Vector database not found at {VECTOR_DB_PATH}")
            st.info("Please ensure your vector_db directory is copied to the project root")
            # Stop the app if vector store is missing
            st.stop()
        
    except Exception as e:
        # Show detailed error in dev mode
        if st.secrets.get("ENV", "production") == "development":
            st.exception(e)
        else:
            st.error(f"Error loading vector store: {str(e)}")
        st.stop()

def get_relevant_context(query: str, vectorstore, score_threshold: float = 0.45) -> tuple[str, List[str]]:
    """Retrieve and polish relevant context from the vector store.
    
    Note: Chroma returns distance scores (0 = identical, higher = less similar).
    We filter by score <= (1.0 - threshold) to keep similar documents.
    Higher threshold = stricter filtering = fewer but more relevant docs.
    """
    try:
        # Get raw chunks with scores
        results_with_scores = vectorstore.similarity_search_with_score(query, k=10)
        
        # Filter by score threshold (Chroma distance: 0 = identical, higher = less similar)
        # Convert threshold to distance: lower distance = higher similarity
        distance_threshold = 1.0 - score_threshold
        filtered_results = [
            (doc, score) for doc, score in results_with_scores 
            if score <= distance_threshold
        ]
        
        # Extract raw chunks and source titles
        raw_chunks = []
        source_titles = []
        
        for doc, score in filtered_results[:5]:  # Get more chunks for polishing
            content = doc.page_content
            if content.strip():
                raw_chunks.append(content)
                source_titles.append(doc.metadata.get("title", "CBT Resource"))
        
        if not raw_chunks:
            return "", []
        
        # Check if chunk polishing is enabled
        if st.session_state.get("polish_chunks", True):
            # Polish chunks with Llama
            with st.spinner("Polishing context..."):
                polished_context = polish_chunks_with_llama(raw_chunks, query)
        else:
            # Use simple concatenation
            polished_context = "\n\n".join(raw_chunks[:3])
        
        return polished_context, source_titles[:3]  # Return top 3 sources
        
    except Exception as e:
        st.error(f"Error retrieving context: {str(e)}")
        return "", []

def build_therapist_prompt(user_msg: str, context: str = "", temperature: float = 0.7) -> str:
    """Build a world-class therapist prompt with conversation memory."""
    emotion = get_emotion(user_msg)
    micro_cue = select_micro_cue(emotion, st.session_state.recent_cues)
    used_skills_str = ", ".join(st.session_state.used_skills) if st.session_state.used_skills else "none"
    follow_up = random.choice(GENTLE_FOLLOW_UPS)
    
    # Get conversation memory from LangChain buffer
    conversation_memory = get_conversation_memory()
    
    # Wrap user message in triple backticks for extra safety against markdown injection
    safe_user_msg = f"```\n{user_msg}\n```"
    
    prompt = f"""{SYSTEM_PROMPT}

{STYLE_PROMPT}

{SAFETY_PROMPT}

**Conversation History (last {st.session_state.get('memory_turns', 3)} exchanges):**
{conversation_memory if conversation_memory else 'This is the start of our conversation.'}

**Latest user message:**  
{safe_user_msg}

**Detected emotion(s):** {emotion}

**Tiny empathy opener:** {micro_cue}

**Used CBT skills this session:** {used_skills_str}

**Gentle follow-up question:** {follow_up}

**Context snippets from CBT corpus:**  
{context}

<Assistant-reply> :=  
1. Validation: mirror the emotion in ≤ 15 words.  
2. Exploration: open question OR gentle reflection, ≤ 20 words.  
3. Offer: one micro-skill or tiny suggestion (optional), ≤ 40 words.  
4. End with open question → keep the dialogue flowing.

CRITICAL: Your response MUST end with a question mark (?). Use the gentle follow-up question if needed.

Remember: Keep total response ≤ 120 words. Sound human, not robotic. Build on previous context."""

    return prompt

def generate_response(prompt: str, context: str = "", temperature: float = 0.7, max_tokens: int = 150) -> str:
    """Generate response using HuggingFace InferenceClient with therapist-quality prompts and retry logic."""
    if not HUGGINGFACE_API_KEY:
        return "I apologize, but I'm currently unable to generate responses due to missing API configuration."
    
    max_retries = 2
    base_wait = 3
    
    for attempt in range(max_retries + 1):
        try:
            # Use cached client
            client = get_hf_client()
            
            # Build the therapist-quality prompt
            full_prompt = build_therapist_prompt(prompt, context, temperature)
            
            # Use the same approach as your working Colab code
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": full_prompt
                            }
                        ]
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            response = completion.choices[0].message.content
            
            # Ensure it ends with a follow-up question
            response = ensure_follow_up_question(response)
            
            # Update recent cues
            emotion = get_emotion(prompt)
            micro_cue = select_micro_cue(emotion, st.session_state.recent_cues)
            update_recent_cues(micro_cue, st.session_state.recent_cues)
            
            # Detect skill usage
            skill_used = detect_skill_usage(response)
            if skill_used:
                st.session_state.last_skill = skill_used
                if skill_used not in st.session_state.used_skills:
                    st.session_state.used_skills.append(skill_used)
            
            return response
            
        except Exception as e:
            if attempt < max_retries:
                wait_time = base_wait * (2 ** attempt)  # Exponential backoff: 3s, 6s
                time.sleep(wait_time)
                continue
            else:
                return f"I apologize, but I'm having trouble connecting right now. Error: {str(e)}"

def get_user_conversation(user_id: str) -> List[Dict[str, str]]:
    """Get conversation history for a user (session-based for GDPR compliance)."""
    return st.session_state.messages

def save_user_conversation(user_id: str, conversation: List[Dict[str, str]]):
    """Save conversation history (session-based for GDPR compliance)."""
    st.session_state.messages = conversation

def stream_chunks(text: str, chunk_size: int = 3) -> str:
    """Stream text in small chunks for typing effect."""
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size]) + " "
        time.sleep(0.05)

def show_disclaimer():
    """Show disclaimer once per session."""
    if "disclaimer_shown" not in st.session_state:
        st.session_state.disclaimer_shown = True
        
        st.markdown("""
        <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 10px; padding: 1rem; margin: 1rem 0;">
        <h4>⚠️ Important Disclaimer</h4>
        <p><strong>VentPal is not a substitute for professional mental health care.</strong></p>
        <p>This chatbot provides general support and information based on Cognitive Behavioral Therapy principles. 
        It cannot diagnose, treat, or provide medical advice.</p>
        <p><strong>If you're in crisis:</strong> Call 999 (UK emergency) or contact Samaritans at 116 123 (24/7, free).</p>
        <p>By continuing, you acknowledge that this is for informational purposes only.</p>
        </div>
        """, unsafe_allow_html=True)

# Main application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>💨 VentPal - Mental Health Support</h1>
        <p>Your empathetic CBT-informed companion</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show disclaimer
    show_disclaimer()
    
    # GDPR Notice
    if GDPR_COMPLIANT:
        st.sidebar.info("🔒 **Privacy Notice:** Your conversation is stored only in this session and will be automatically deleted when you close this tab for your privacy.")
    
    # Initialize vector store - FIXED: removed problematic truth test
    with st.spinner("Loading mental health resources..."):
        vectorstore = create_vector_store()
    
    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # User info
        if not st.session_state.user_name:
            st.session_state.user_name = st.text_input("What should I call you?", placeholder="Your name")
        
        if st.session_state.user_name:
            st.success(f"Hello, {st.session_state.user_name}! 💨")
        
        # Session goals
        if not st.session_state.session_goals:
            st.session_state.session_goals = st.text_area(
                "What would you like to work on today?", 
                placeholder="e.g., managing anxiety, improving mood, coping with stress",
                height=100
            )
        
        # Configuration options
        st.subheader("🔧 Configuration")
        
        temperature = st.slider(
            "Creativity Level", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.7, 
            step=0.1,
            help="Higher values make responses more creative, lower values more focused"
        )
        
        max_tokens = st.slider(
            "Response Length", 
            min_value=50, 
            max_value=200, 
            value=150, 
            step=10,
            help="Maximum number of words in responses"
        )
        
        memory_turns = st.slider(
            "Conversation Memory", 
            min_value=1, 
            max_value=5, 
            value=3, 
            step=1,
            help="Number of previous exchanges to remember"
        )
        
        score_threshold = st.slider(
            "Content Relevance Filter", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.45, 
            step=0.05,
            help="Higher values show only the most relevant CBT content"
        )
        
        # New setting for chunk polishing
        polish_chunks = st.checkbox(
            "Polish Knowledge Chunks", 
            value=True,
            help="Use Llama to polish and summarize retrieved chunks for better relevance"
        )
        
        # Store memory_turns in session state for use in get_conversation_memory()
        st.session_state.memory_turns = memory_turns
        st.session_state.polish_chunks = polish_chunks
        
        # Rate limiting info
        st.subheader("📊 Usage")
        st.metric("Requests Today", st.session_state.request_count)
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation", type="secondary"):
            # Backup current conversation
            st.session_state.conversation_backup = st.session_state.messages.copy()
            st.session_state.messages = []
            st.session_state.memory.clear()
            st.session_state.used_skills = []
            st.session_state.recent_cues = []
            st.session_state.last_skill = ""
            st.toast("Conversation cleared! 💬", icon="🗑️")
        
        # Undo clear (if backup exists)
        if st.session_state.conversation_backup and st.button("↩️ Undo Clear", type="secondary"):
            st.session_state.messages = st.session_state.conversation_backup.copy()
            st.session_state.conversation_backup = []
            st.toast("Conversation restored! ✅", icon="↩️")
        
        st.divider()
        st.markdown("**Rate Limit:** " + 
                    f"{st.session_state.request_count}/{MAX_REQUESTS_PER_HOUR} requests this hour")
        
        if st.button("Reset Rate Limit"):
            st.session_state.request_count = 0
            st.session_state.last_reset = time.time()
            st.success("Rate limit reset!")
            st.rerun()
    
    # Get user name if not set
    if not st.session_state.user_name:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown("**Hello!** I'm **VentPal**, your mental health companion.\n\nWhat should I call you?")
        
        name_input = st.text_input("Your name:", key="name_input")
        if name_input:
            st.session_state.user_name = sanitize_user_input(name_input)
            st.rerun()
        return
    
    # Show welcome message if first visit
    conversation = get_user_conversation(st.session_state.user_id)
    if not conversation:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(f"**Hello {st.session_state.user_name}!** I'm **VentPal**, your mental health companion.\n\nYou can talk to me about how you're feeling. This space is private and safe.")
    
    # Display chat history
    for message in conversation:
        with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
            st.markdown(message["content"])
            
            # Show emotion indicator for user messages
            if message["role"] == "user" and "emotion" in message:
                emotion = message["emotion"]
                emotion_colors = {
                    "anxiety": "#ff9800",
                    "depression": "#2196f3", 
                    "anger": "#f44336",
                    "grief": "#9c27b0",
                    "joy": "#4caf50",
                    "neutral": "#9e9e9e"
                }
                color = emotion_colors.get(emotion, "#9e9e9e")
                st.markdown(f'<span class="emotion-indicator" style="background-color: {color}; color: white;">{emotion.title()}</span>', 
                           unsafe_allow_html=True)
            
            # Show skill badge for assistant messages
            if message["role"] == "assistant" and "skill_used" in message:
                skill_used = message["skill_used"]
                if skill_used:
                    st.markdown(f'<div class="skill-badge">💡 {CBT_SKILLS[skill_used]["name"]}</div>', unsafe_allow_html=True)
            
            # Show source citations for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                sources = message["sources"]
                if sources:
                    st.markdown(f'<div class="source-citation">Sources: {", ".join(sources)}</div>', 
                               unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input(f"How are you feeling today, {st.session_state.user_name}?"):
        # Sanitize user input
        sanitized_prompt = sanitize_user_input(prompt)
        
        # Check rate limit
        if not check_rate_limit():
            st.error("You've reached the hourly rate limit. Please wait before making more requests.")
            st.stop()
        
        # Add user message to chat
        with st.chat_message("user", avatar="👤"):
            st.markdown(sanitized_prompt)
        
        # Detect emotion and crisis
        emotion = get_emotion(sanitized_prompt)
        is_crisis = detect_crisis(sanitized_prompt)
        
        # Add user message to conversation
        conversation.append({
            "role": "user",
            "content": sanitized_prompt,
            "emotion": emotion,
            "timestamp": datetime.now().isoformat()
        })
        
        # Crisis detection and response
        if is_crisis:
            with st.chat_message("assistant", avatar="🤖"):
                crisis_response = crisis_block()
                st.markdown(crisis_response)
                
                conversation.append({
                    "role": "assistant", 
                    "content": crisis_response,
                    "crisis_alert": True,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Update memory even for crisis responses
                update_memory(sanitized_prompt, crisis_response)
        else:
            # Generate AI response with typing indicator
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("VentPal is thinking..."):
                    # Get relevant context (now with Llama polishing)
                    context, source_titles = get_relevant_context(sanitized_prompt, vectorstore, score_threshold)
                    
                    # Generate response
                    response = generate_response(sanitized_prompt, context, temperature, max_tokens)
                    
                    # Detect skill usage
                    skill_used = detect_skill_usage(response)
                    
                    # Display response
                    st.markdown(response)
                    
                    # Show skill badge if skill was used
                    if skill_used:
                        st.markdown(f'<div class="skill-badge">💡 {CBT_SKILLS[skill_used]["name"]}</div>', unsafe_allow_html=True)
                    
                    # Add assistant message to conversation
                    conversation.append({
                        "role": "assistant",
                        "content": response,
                        "skill_used": skill_used,
                        "sources": source_titles if context else [],
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Update LangChain memory
                    update_memory(sanitized_prompt, response)
        
        # Increment rate limit
        increment_rate_limit()
        
        # Save conversation
        save_user_conversation(st.session_state.user_id, conversation)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>🤖 VentPal - Powered by CBT & AI | Session data is private and temporary</p>
    <p>For crisis support: Samaritans 116 123 | Shout 85258 | NHS 111</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
