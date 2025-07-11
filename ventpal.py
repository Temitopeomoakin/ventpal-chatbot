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
    page_icon="",
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
    .skill-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.7rem;
        background-color: #e8f5e8;
        color: #2e7d32;
        margin-top: 0.5rem;
        border: 1px solid #c8e6c9;
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
        max_token_limit=1024  # Reduced for Llama tokenizer compatibility
    )
if "recent_cues" not in st.session_state:
    st.session_state.recent_cues = []
if "recent_skills" not in st.session_state:
    st.session_state.recent_skills = []
if "conversation_backup" not in st.session_state:
    st.session_state.conversation_backup = None
if "last_skill_offered" not in st.session_state:
    st.session_state.last_skill_offered = None

# Configuration - now from secrets where possible
HUGGINGFACE_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY", "")
MODEL_NAME = st.secrets.get("MODEL_NAME", "meta-llama/Llama-4-Scout-17B-16E-Instruct")
EMBEDDING_MODEL = st.secrets.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VECTOR_DB_PATH = st.secrets.get("VECTOR_DB_PATH", "vector_db")
MAX_REQUESTS_PER_HOUR = st.secrets.get("MAX_REQUESTS_PER_HOUR", 50)

# World-class prompt templates
SYSTEM_PROMPT = """You are VentPal, a mental health companion trained in evidence-based Cognitive Behavioural Therapy (CBT).  
Primary goal: relieve distress in a single turn while building rapport for the next turn.  
Secondary goal: teach one CBT micro-skill at a time, only if user is ready.  
You are NOT a medical professional and never prescribe or diagnose."""

# Micro-cue library (50 items, grouped by use-case)
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
    "insight_reflection": [
        "That's an important realisation.", "That makes sense.", "I see what you mean.", 
        "Good observation.", "That's a useful way to look at it."
    ],
    "feeling_stuck": [
        "Let's slow it down.", "One step at a time.", "We can figure this out together.", 
        "Let's unpack that.", "We can sit with it for a moment."
    ],
    "asking_strategy": [
        "Here's something that might help.", "Want to try a quick exercise?", 
        "Can I share a technique?", "Would you like a tool for that?", 
        "We could test a small idea."
    ],
    "encouraging_action": [
        "You've got this.", "I'm rooting for you.", "You're not alone in this.", 
        "I believe in your ability.", "You can handle this."
    ],
    "crisis_safety": [
        "Your safety matters most.", "I'm concerned about you.", "Let's keep you safe first."
    ],
    "neutral_filler": [
        "Mmm-hmm.", "Uh-huh.", "I hear you."
    ]
}

# Flatten for fallback
ALL_CUES = [cue for group in CUE_GROUPS.values() for cue in group]

STYLE_PROMPT = """### VENTPAL STYLE GUIDE (v3)

You are "VentPal", an empathetic CBT-informed companion. Speak in first-person singular with contractions; total reply ≤120 words.

**Required structure each non-crisis turn**

1  **Validation** (≤15 words)  
   • Paraphrase the user's feeling; include *one* micro-cue from MICRO_CUE_PLACEHOLDER.

2  **Exploration** (≤20 words)  
   • Ask **exactly one** open question that logically follows the conversation flow.  
   • This question ends the message (only one "?" in the whole reply).

3  **Offer** (optional, ≤40 words)  
   • If appropriate, offer **one** CBT micro-skill not in USED_SKILLS_PLACEHOLDER.  
   • Mention it concretely, e.g. "Would you like to try a 5-4-3-2-1 grounding?"

**Crisis override**

If the latest user message shows self-harm intent, replace the above with the UK crisis block:

🚨 I'm really sorry you're feeling like this—it sounds unbearably painful.
If you feel you might act on these thoughts right now, please call 999 or go to A&E.
If you can keep yourself safe for a few minutes, you could also contact:
• Samaritans 116 123 (24/7)
• Shout – text SHOUT to 85258
• NHS 111 for urgent advice

Are you able to stay safe for the next few minutes?

…and **stop**; wait for user response before continuing.

**Diversity rules**

* Do **not** reuse the same micro-cue or the same CBT skill you used in the last two replies.  
* Vary sentence openings; avoid starting two consecutive messages with "I'm".

**Safety rules**

* No diagnosis, no medication advice, no policy talk, no links.  
* Hide internal labels (e.g., "Neutral", "Anxiety") from the user."""

SAFETY_PROMPT = """If user expresses self-harm or suicidal intent:
    ➊ Acknowledge the pain ("I'm really sorry you're feeling like this...")  
    ➋ Show UK crisis resources (999, Samaritans 116 123, Shout 85258, NHS 111)  
    ➌ Ask safety check: "Are you able to stay safe for the next few minutes?"
    ➍ Stop CBT content; wait for user response before continuing.

If you are unsure whether content is safe → ask a clarifying question first.
Never mention policy or guidelines."""

# CBT micro-skills library with regex patterns for better tracking
CBT_SKILLS = {
    "grounding": "Would you like to try a 5-4-3-2-1 grounding exercise with me?",
    "breathing": "A slow 4-second inhale, 1-second pause, 6-second exhale can calm the body quickly.",
    "thought_label": "Sometimes naming a thought 'just a thought' can shrink its power—wanna give it a go?",
    "suds_scale": "On a 0–10 scale, how strong is the feeling right now?",
    "behavioral_activation": "Choosing one small activity you normally enjoy—even if mood says no—can nudge things.",
    "cognitive_restructuring": "What's the evidence for and against that thought?",
    "mindfulness": "Can you notice what's happening in your body right now, without trying to change it?"
}

# Skill patterns for better tracking (handles paraphrasing)
SKILL_PATTERNS = {
    "breathing": [r"deep ?breath", r"4-?6 ?breath", r"inhale.*exhale", r"breathing.*exercise"],
    "grounding": [r"5-?4-?3-?2-?1", r"grounding.*exercise", r"senses.*exercise"],
    "thought_label": [r"just a thought", r"naming.*thought", r"thought.*label"],
    "suds_scale": [r"0-?10.*scale", r"suds", r"how strong.*feeling"],
    "behavioral_activation": [r"small activity", r"enjoy.*mood", r"behavioral.*activation"],
    "cognitive_restructuring": [r"evidence.*against", r"cognitive.*restructuring", r"thought.*evidence"],
    "mindfulness": [r"notice.*body", r"mindfulness", r"without.*change"]
}

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
    """Check if text is primarily English using simple heuristic."""
    if not text:
        return True
    
    # Count non-ASCII characters
    non_ascii_count = sum(1 for char in text if ord(char) > 127)
    total_chars = len(text)
    
    # If more than 30% non-ASCII, likely not English
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
    # Skip emotion detection for non-English text
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
    # Skip crisis detection for non-English text
    if not is_english_text(text):
        return False
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CRISIS_KEYWORDS)

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

def select_micro_cue(emotion: str) -> str:
    """Select contextually appropriate micro-cue, avoiding recent ones."""
    recent_cues = st.session_state.get("recent_cues", [])
    
    # Select cue group based on emotion - improved for positive emotions
    if emotion in ["anxiety", "depression", "anger", "grief"]:
        cue_pool = CUE_GROUPS["heavy_emotion"] + CUE_GROUPS["mild_distress"]
    elif emotion == "joy":
        # For joy, prefer positive progress cues only
        cue_pool = CUE_GROUPS["positive_progress"]
    else:
        cue_pool = CUE_GROUPS["mild_distress"] + CUE_GROUPS["neutral_filler"]
    
    # Filter out recently used cues (increased to 5 for better variety)
    available_cues = [cue for cue in cue_pool if cue not in recent_cues]
    
    # Fallback to all cues if none available
    if not available_cues:
        available_cues = [cue for cue in ALL_CUES if cue not in recent_cues]
    
    # If still none, reset and use any cue
    if not available_cues:
        available_cues = ALL_CUES
    
    selected_cue = random.choice(available_cues)
    
    # Update tracking
    recent_cues.append(selected_cue)
    st.session_state.recent_cues = recent_cues[-5:]  # Keep last 5 for better variety
    
    return selected_cue

def update_skill_tracking(response: str) -> Tuple[str, Optional[str]]:
    """Track which micro-skills have been offered this session using regex patterns."""
    recent_skills = st.session_state.get("recent_skills", [])
    detected_skill = None
    
    # Check if a skill was offered using regex patterns
    for skill_name, patterns in SKILL_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, response.lower()):
                if skill_name not in recent_skills:
                    recent_skills.append(skill_name)
                    detected_skill = skill_name
                    break  # Only track one skill per response
    
    # Keep only last 5 skills
    st.session_state.recent_skills = recent_skills[-5:]
    
    return response, detected_skill

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

@st.cache_resource(max_entries=1, show_spinner=False)
def get_vectorstore(path: str = VECTOR_DB_PATH):
    """Cached vector store to share across sessions."""
    try:
        # Late import to speed up cold start
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        # Load your existing vector store
        if os.path.exists(path):
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            vectorstore = Chroma(
                persist_directory=path,
                embedding_function=embeddings
            )
            st.success(f"✅ Loaded existing vector store from {path}")
            return vectorstore
        else:
            st.error(f"❌ Vector database not found at {path}")
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

def polish_chunks_with_llama(raw_chunks: List[str], user_query: str) -> Tuple[str, List[str]]:
    """Use Llama to polish and summarize relevant chunks. Returns (polished_text, source_titles)."""
    if not raw_chunks:
        return "", []
    
    # Skip polishing for ≤ 2 chunks to save latency
    if len(raw_chunks) <= 2:
        return "\n\n".join(raw_chunks), []
    
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
            temperature=0.3,  # Lower temperature for more focused polishing
            timeout=30  # Add timeout for graceful handling
        )
        
        return completion.choices[0].message.content, []
        
    except Exception as e:
        st.error(f"Error polishing chunks: {str(e)}")
        # Fallback to simple concatenation
        return "\n\n".join(raw_chunks[:3]), []  # Just use first 3 chunks

def get_relevant_context(query: str, vectorstore, score_threshold: float = 0.45) -> tuple[str, List[str]]:
    """Retrieve and polish relevant context from the vector store."""
    try:
        # Get raw chunks with scores - keep top-k and let polish step filter
        results_with_scores = vectorstore.similarity_search_with_score(query, k=10)
        
        # Extract raw chunks and source titles
        raw_chunks = []
        source_titles = []
        
        for doc, score in results_with_scores[:5]:  # Get top 5 for polishing
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
                polished_context, _ = polish_chunks_with_llama(raw_chunks, query)
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
    
    # Select contextually appropriate micro-cue
    micro_cue = select_micro_cue(emotion)
    
    # Get used skills for this session
    recent_skills = st.session_state.get("recent_skills", [])
    used_skills_str = ",".join(recent_skills) if recent_skills else "none"
    
    # Get conversation memory from LangChain buffer
    conversation_memory = get_conversation_memory()
    
    # Wrap user message in triple backticks for extra safety against markdown injection
    safe_user_msg = f"```\n{user_msg}\n```"
    
    # Use string replacement instead of format() to avoid brace issues
    style_prompt = STYLE_PROMPT.replace('MICRO_CUE_PLACEHOLDER', micro_cue).replace('USED_SKILLS_PLACEHOLDER', used_skills_str)
    
    prompt = f"""{SYSTEM_PROMPT}

{style_prompt}

{SAFETY_PROMPT}

**Conversation History (last {st.session_state.get('memory_turns', 3)} exchanges):**
{conversation_memory if conversation_memory else 'This is the start of our conversation.'}

**Latest user message:**  
{safe_user_msg}

**Detected emotion(s):** {emotion}

**Context snippets from CBT corpus:**  
{context}

Remember: Keep total response ≤ 120 words. Sound human, not robotic. Build on previous context."""

    return prompt

def generate_response(prompt: str, context: str = "", temperature: float = 0.7, max_tokens: int = 150) -> str:
    """Generate response using HuggingFace client with proper error handling."""
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
            
            # Use synchronous call (simpler and more reliable)
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
                temperature=temperature,
                timeout=30  # Add timeout for graceful handling
            )
            
            response = completion.choices[0].message.content
            
            # Ensure it ends with a follow-up question
            response = ensure_follow_up_question(response)
            
            return response
            
        except Exception as e:
            if attempt < max_retries:
                wait_time = base_wait * (2 ** attempt)
                time.sleep(wait_time)
                continue
            else:
                return f"I apologize, but I'm having trouble connecting right now. Please try again in a moment."

def get_user_conversation(user_id: str) -> List[Dict[str, str]]:
    """Get conversation history for a user (session-based for GDPR compliance)."""
    return st.session_state.messages

def save_user_conversation(user_id: str, conversation: List[Dict[str, str]]):
    """Save conversation history (session-based for GDPR compliance)."""
    st.session_state.messages = conversation

def clear_conversation_with_backup():
    """Clear conversation with backup for undo functionality."""
    # Store backup before clearing
    st.session_state.conversation_backup = st.session_state.messages.copy()
    
    # Clear everything
    st.session_state.messages = []
    st.session_state.conversation_summary = ""
    st.session_state.memory.clear()
    st.session_state.recent_cues = []
    st.session_state.recent_skills = []
    st.session_state.last_skill_offered = None
    
    # Show undo toast with auto-close
    st.success("Conversation cleared successfully!")
    st.toast("Conversation cleared – Undo available for 5 seconds", icon="️", auto_close=True)

def undo_clear_conversation():
    """Restore conversation from backup."""
    if st.session_state.conversation_backup:
        st.session_state.messages = st.session_state.conversation_backup
        st.session_state.conversation_backup = None
        st.success("Conversation restored!")
        st.rerun()

# Main application
def main():
    # Set deterministic random seed for consistent cue ordering
    random.seed(st.session_state.user_id)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>💨 VentPal</h1>
        <p>Your AI Mental Health Companion</p>
    </div>
    """, unsafe_allow_html=True)
    
    # GDPR Notice
    if GDPR_COMPLIANT:
        st.sidebar.info("🔒 **Privacy Notice:** Your conversation is stored only in this session and will be automatically deleted when you close this tab for your privacy.")
    
    # Initialize vector store using cached function
    with st.spinner("Loading mental health resources..."):
        vectorstore = get_vectorstore()
    
    # Sidebar settings with tooltips
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Response Creativity with tooltip
        st.markdown('<div class="tooltip">Response Creativity ℹ️<span class="tooltiptext">Lower = more focused responses, Higher = more creative variations</span></div>', unsafe_allow_html=True)
        temperature = st.slider("", 0.1, 1.0, 0.7, 0.1, 
                               help="Lower = more focused, Higher = more creative")
        
        # Max Response Length with tooltip
        st.markdown('<div class="tooltip">Max Response Length ℹ️<span class="tooltiptext">Maximum number of words in VentPal\'s responses</span></div>', unsafe_allow_html=True)
        max_tokens = st.slider("", 50, 200, 150, 25,
                              help="Maximum length of responses")
        
        # Conversation Memory with tooltip
        st.markdown('<div class="tooltip">Conversation Memory ℹ️<span class="tooltiptext">How many previous exchanges to remember for context</span></div>', unsafe_allow_html=True)
        memory_turns = st.slider("", 1, 5, 3, 1,
                                help="Number of previous exchanges to remember")
        
        # Relevance Threshold with tooltip
        st.markdown('<div class="tooltip">Relevance Threshold ℹ️<span class="tooltiptext">Higher = stricter filtering = fewer but more relevant knowledge chunks</span></div>', unsafe_allow_html=True)
        score_threshold = st.slider("", 0.1, 0.9, 0.45, 0.05,
                                   help="Higher = stricter filtering = fewer but more relevant docs")
        
        # New setting for chunk polishing
        polish_chunks = st.checkbox("Polish Knowledge Chunks", value=True,
                                   help="Use Llama to polish and summarize retrieved chunks for better relevance")
        
        # Reset UI if polishing setting changes
        if "previous_polish_setting" not in st.session_state:
            st.session_state.previous_polish_setting = polish_chunks
        elif st.session_state.previous_polish_setting != polish_chunks:
            st.session_state.previous_polish_setting = polish_chunks
            st.experimental_rerun()
        
        st.session_state.polish_chunks = polish_chunks
        
        # Store memory_turns in session state for use in get_conversation_memory()
        st.session_state.memory_turns = memory_turns
        
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
    
    # Disclaimer once per session
    if "disclaimer" not in st.session_state:
        st.info("⚠️ **Important:** This chatbot provides CBT-based information and support. "
                "It is **not** a substitute for professional mental health care. "
                "If you're in crisis, please contact emergency services or a crisis hotline.")
        st.session_state.disclaimer = True
    
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
            if message["role"] == "assistant" and "skill_offered" in message:
                skill_name = message["skill_offered"]
                if skill_name:
                    st.markdown(f'<div class="skill-badge">💡 {skill_name.replace("_", " ").title()}</div>', 
                               unsafe_allow_html=True)
            
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
            # Generate AI response with typing indicator and elapsed time
            with st.chat_message("assistant", avatar="🤖"):
                start_time = time.time()
                with st.spinner("VentPal is thinking..."):
                    # Get relevant context (now with Llama polishing)
                    context, source_titles = get_relevant_context(sanitized_prompt, vectorstore, score_threshold)
                    
                    # Generate response
                    response = generate_response(sanitized_prompt, context, temperature, max_tokens)
                    
                    # Update skill tracking and get detected skill
                    response, detected_skill = update_skill_tracking(response)
                    
                    # Display response (removed streaming for now)
                    st.markdown(response)
                    
                    # Add assistant message to conversation
                    conversation.append({
                        "role": "assistant",
                        "content": response,
                        "sources": source_titles if context else [],
                        "skill_offered": detected_skill,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Update LangChain memory
                    update_memory(sanitized_prompt, response)
                
                # Show elapsed time
                elapsed = time.time() - start_time
                if elapsed > 2:  # Only show if it took more than 2 seconds
                    st.caption(f"Response generated in {elapsed:.1f}s")
        
        # Increment rate limit
        increment_rate_limit()
        
        # Save conversation
        save_user_conversation(st.session_state.user_id, conversation)
    
    # Sidebar actions
    with st.sidebar:
        st.divider()
        
        # Clear conversation with backup
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear"):
                clear_conversation_with_backup()
                st.rerun()
        
        with col2:
            if st.button("↩️ Undo") and st.session_state.conversation_backup:
                undo_clear_conversation()

if __name__ == "__main__":
    main()
