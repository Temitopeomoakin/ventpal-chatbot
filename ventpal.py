# ventpal.py — VentPal with Natural Therapy Flow + Fixed Response Integration
# --------------------------------------------------------------------------
# Greeting → Emotion Check → Exploration → Consent → RAG with proper technique integration

import os, sys, time, json, re, random, hashlib
from typing import List, Tuple, Dict, Optional
from datetime import datetime

# SQLite shim for Chroma
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except:
    pass
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import requests
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

# FIXED IMPORTS - Use same combination as your working code
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from huggingface_hub import InferenceClient

# Add reranking import - ONLY ADDITION
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

# ================================ Configuration ================================
st.set_page_config(
    page_title="VentPal", 
    page_icon="💨", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state
if "messages" not in st.session_state: 
    st.session_state.messages = []
if "request_count" not in st.session_state: 
    st.session_state.request_count = 0
if "last_reset" not in st.session_state: 
    st.session_state.last_reset = time.time()
if "user_id" not in st.session_state: 
    st.session_state.user_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, max_token_limit=2000)
if "conversation_stage" not in st.session_state:
    st.session_state.conversation_stage = "greeting"  # greeting → exploration → consent → support
if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = "neutral"
if "awaiting_consent" not in st.session_state:
    st.session_state.awaiting_consent = False
if "classifier_history" not in st.session_state:
    st.session_state.classifier_history = []
if "retrieval_history" not in st.session_state:
    st.session_state.retrieval_history = []
if "ablation_mode" not in st.session_state:
    st.session_state.ablation_mode = "full_system"  # full_system, no_classifier, no_rag, baseline

# Load secrets - UPDATED to match your actual secrets
HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
MODEL_NAME = st.secrets["MODEL_NAME"]
FALLBACK_MODEL = st.secrets["FALLBACK_MODEL"]
EMBEDDING_MODEL = st.secrets["EMBEDDING_MODEL"]
VECTOR_DB_PATH = st.secrets["VECTOR_DB_PATH"]
COLLECTION_NAME = st.secrets["COLLECTION_NAME"]
CLASSIFIER_URL = st.secrets["CLASSIFIER_URL"]
CLASSIFIER_AUTH = st.secrets["CLASSIFIER_AUTH"]
MAX_REQUESTS_PER_HOUR = int(st.secrets["MAX_REQUESTS_PER_HOUR"])

# Severity settings
SEVERITY_ALERT_P = float(st.secrets["SEVERITY_ALERT_P"])
SEVERITY_HIGH_LABELS = st.secrets["SEVERITY_HIGH_LABELS"]
SEVERITY_ALERT_CONTAINS = st.secrets["SEVERITY_ALERT_CONTAINS"]

# RAG settings - ADDED reranking settings
MIN_EXCHANGES_BEFORE_RAG = int(st.secrets["MIN_EXCHANGES_BEFORE_RAG"])
ENABLE_RERANK = st.secrets["ENABLE_RERANK"].lower() == "true"
RERANK_MODEL_NAME = st.secrets["RERANK_MODEL_NAME"]
RERANK_CANDIDATES = int(st.secrets["RERANK_CANDIDATES"])
RERANK_TOP_K = int(st.secrets["RERANK_TOP_K"])

# Other settings
CLASSIFIER_POLICY = st.secrets["CLASSIFIER_POLICY"]
CLASSIFIER_MODE = st.secrets["CLASSIFIER_MODE"]
HF_PROVIDER = st.secrets["HF_PROVIDER"]
ASSUME_YES = st.secrets["ASSUME_YES"].lower() == "true"
SKILL_GRACE_TURNS = int(st.secrets["SKILL_GRACE_TURNS"])
SKILL_COOLDOWN_TURNS = int(st.secrets["SKILL_COOLDOWN_TURNS"])

# Set HF token
if HUGGINGFACE_API_KEY:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_API_KEY
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HUGGINGFACE_API_KEY

# ================================ UI Styling ================================
st.markdown("""
<style>
.main-header{
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 12px;
    color: #fff;
    text-align: center;
    margin-bottom: 1rem;
}
.skill-badge{
    display: inline-block;
    background: #e8f5e8;
    color: #2e7d32;
    padding: .2rem .5rem;
    border-radius: 12px;
    font-size: .75rem;
    margin-top: .5rem;
}
.footer-notice{
    background: #fff8e1;
    border: 1px solid #ffeaa7;
    border-radius: 12px;
    padding: .9rem;
    margin-top: 1rem;
    color: #6b5e00;
    font-size: .92rem;
}
.status-chip{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 12px;
    margin-left: 6px;
}
.ok{background: #e8f5e9; color: #1b5e20; border: 1px solid #a5d6a7;}
.warn{background: #fff8e1; color: #6b5e00; border: 1px solid #ffe082;}
.err{background: #ffebee; color: #b71c1c; border: 1px solid #ef9a9a;}
.ablation-mode{background: #e3f2fd; padding: 8px; border-radius: 6px; margin: 5px 0; font-size: 12px;}
</style>
""", unsafe_allow_html=True)

DISCLAIMER = (
    "**Important:** VentPal offers supportive conversation using CBT, DBT and journaling techniques. "
    "It isn't medical care and I'm **not a therapist**. I can't diagnose, treat, or prescribe. "
    "If you're in crisis or might act on thoughts of harming yourself or others, call **999**, "
    "or contact **Samaritans (116 123)** or **Shout (text SHOUT to 85258)** right away."
)

def footer_notice():
    st.markdown(f"<div class='footer-notice'>{DISCLAIMER}</div>", unsafe_allow_html=True)

def chip(text: str, kind: str = "ok"):
    cls = {"ok": "ok", "warn": "warn", "err": "err"}.get(kind, "ok")
    st.markdown(f'<span class="status-chip {cls}">{text}</span>', unsafe_allow_html=True)

# ================================ Test Scenarios for Thesis ================================
TEST_SCENARIOS = {
    "Depression Scenario": [
        "I am not good",
        "I feel really depressed and can't get out of bed",
        "Yes, please help me",
        "I would tell them to stay strong and that things will get better"
    ],
    "Anxiety Scenario": [
        "I'm not doing well",
        "I've been having panic attacks at work",
        "Yes, I'd like some techniques",
        "I would listen to them and try to calm them down"
    ],
    "Stress Scenario": [
        "Not great today",
        "I'm feeling overwhelmed with everything",
        "Sure, that would be helpful",
        "I would give them emotional support and try to infuse positivity"
    ],
    "Resistant User": [
        "I'm fine",
        "Just tired I guess",
        "No, I don't need help"
    ]
}

# ================================ Crisis Detection ================================
SUICIDE_PATTERNS = [
    r"\bsuicid(?:e|al)\b", r"\bi\s*want\s*to\s*die\b", r"\bend\s*my\s*life\b",
    r"\bhurt\s*myself\b", r"\bself[-\s]?harm\b", r"\boverdose\b",
    r"\bkill\s*myself\b", r"\bkms\b", r"\b(end|ending)\s+it\b"
]
CRISIS_REGEX = re.compile("|".join(SUICIDE_PATTERNS), re.IGNORECASE)

def detect_crisis_regex(text: str) -> bool:
    return bool(CRISIS_REGEX.search(text or ""))

CRISIS_RESOURCES_UK = (
    "• **Samaritans** 116 123 (24/7, free)\n"
    "• **Shout** – text **SHOUT** to 85258 (24/7)\n"
    "• **Emergency**: 999\n"
    "• **NHS 111** for urgent advice"
)

def crisis_block() -> str:
    return (
        "🚨 I'm really sorry you're feeling like this—it sounds unbearably painful.\n\n"
        "If you feel you might act on these thoughts **right now**, please call **999** or go to A&E.\n\n"
        f"{CRISIS_RESOURCES_UK}\n\n"
        "I'm here with you. Are you able to stay safe for the next few minutes?"
    )

# ================================ Classifier with Ablation ================================
def classify(text: str, use_classifier: bool = True) -> Optional[Dict]:
    """Classifier with ablation study support"""
    if not use_classifier or st.session_state.ablation_mode in ["no_classifier", "baseline"]:
        # Return neutral classifications for ablation
        return {
            "emotion": {"top": {"label": "neutral", "conf": 0.5}},
            "intent": {"top": {"label": "support", "conf": 0.5}},
            "severity": {"top": {"label": "green", "conf": 0.5}},
            "topic": {"top": {"label": "general", "conf": 0.5}},
            "_ablation": True
        }
    
    if not CLASSIFIER_URL:
        return None
    
    try:
        headers = {"Content-Type": "application/json"}
        if CLASSIFIER_AUTH:
            headers["Authorization"] = f"Bearer {CLASSIFIER_AUTH}"
        
        t0 = time.time()
        r = requests.post(f"{CLASSIFIER_URL}/classify", headers=headers, json={"text": text}, timeout=12)
        r.raise_for_status()
        out = r.json()
        out["_latency_ms"] = int((time.time() - t0) * 1000)
        
        # Track classification
        st.session_state.classifier_history.append({
            "timestamp": datetime.now().isoformat(),
            "ablation_mode": st.session_state.ablation_mode,
            "emotion": out.get("emotion", {}).get("top", {}).get("label"),
            "intent": out.get("intent", {}).get("top", {}).get("label"),
            "severity": out.get("severity", {}).get("top", {}).get("label"),
            "topic": out.get("topic", {}).get("top", {}).get("label"),
            "latency_ms": out["_latency_ms"]
        })
        
        return out
        
    except Exception as e:
        st.sidebar.error(f"Classifier error: {e}")
        return None

def _top(head: Optional[Dict], default_label="unknown", default_conf: float = 0.0) -> Tuple[str, float]:
    top = (head or {}).get("top") or {}
    return str(top.get("label", default_label)).lower(), float(top.get("conf", default_conf))

# ================================ RAG with Ablation ================================
@st.cache_resource(show_spinner=False)
def create_vector_store() -> Chroma:
    if not os.path.exists(VECTOR_DB_PATH):
        st.error(f"❌ Vector DB not found at `{VECTOR_DB_PATH}`.")
        st.stop()
    
    try:
        # FIXED: Use same import combination as your working code
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        
        test_docs = vectorstore.similarity_search("stress", k=1)
        if test_docs:
            st.sidebar.success(f"✅ Vector DB loaded")
        
        return vectorstore
        
    except Exception as e:
        st.error(f"Failed to init vector store: {e}")
        st.stop()

# Add reranker loading - ONLY ADDITION
@st.cache_resource(show_spinner=False)
def load_reranker():
    """Load reranker model"""
    if not ENABLE_RERANK or CrossEncoder is None:
        return None
    try:
        return CrossEncoder(RERANK_MODEL_NAME)
    except:
        return None

def get_relevant_context(query: str, vectorstore: Chroma, use_rag: bool = True) -> Tuple[str, List[str]]:
    """RAG with ablation study support - UPDATED with reranking"""
    if not use_rag or st.session_state.ablation_mode in ["no_rag", "baseline"]:
        # Track attempted retrieval for ablation
        st.session_state.retrieval_history.append({
            "timestamp": datetime.now().isoformat(),
            "ablation_mode": st.session_state.ablation_mode,
            "query": query[:100],
            "docs_retrieved": 0,
            "docs_used": 0,
            "_ablation": True
        })
        return "", []
    
    try:
        # Get more candidates for reranking
        k = RERANK_CANDIDATES if ENABLE_RERANK else 3
        relevant_docs = vectorstore.similarity_search(query, k=k)
        
        if not relevant_docs:
            return "", []
        
        # Rerank if enabled - ONLY ADDITION
        if ENABLE_RERANK:
            reranker = load_reranker()
            if reranker:
                try:
                    pairs = [[query, doc.page_content] for doc in relevant_docs]
                    scores = reranker.predict(pairs)
                    scored_docs = list(zip(relevant_docs, scores))
                    scored_docs.sort(key=lambda x: x[1], reverse=True)
                    relevant_docs = [doc for doc, score in scored_docs[:RERANK_TOP_K]]
                except:
                    relevant_docs = relevant_docs[:RERANK_TOP_K]
            else:
                relevant_docs = relevant_docs[:RERANK_TOP_K]
        else:
            relevant_docs = relevant_docs[:3]
        
        chunks = []
        titles = []
        
        for doc in relevant_docs:
            content = doc.page_content.strip()
            if content and len(content) > 100:
                chunks.append(content[:1200] + ("…" if len(content) > 1200 else ""))
                metadata = doc.metadata or {}
                title = metadata.get("title") or metadata.get("source") or "CBT/DBT Guide"
                titles.append(title)
        
        # Track retrieval
        st.session_state.retrieval_history.append({
            "timestamp": datetime.now().isoformat(),
            "ablation_mode": st.session_state.ablation_mode,
            "query": query[:100],
            "docs_retrieved": k,
            "docs_used": len(chunks)
        })
        
        context = "\n\n".join(chunks)
        return context, titles
        
    except Exception as e:
        st.sidebar.warning(f"Retrieval error: {e}")
        return "", []

# ================================ LLM ================================
@st.cache_resource
def hf_client_primary() -> InferenceClient:
    return InferenceClient(model=MODEL_NAME, token=HUGGINGFACE_API_KEY, timeout=120)

@st.cache_resource
def hf_client_fallback() -> InferenceClient:
    return InferenceClient(model=FALLBACK_MODEL, token=HUGGINGFACE_API_KEY, timeout=120)

def generate_therapy_response(user_text: str, stage: str, emotion: str, context: str = "") -> str:
    """Generate response based on therapy session stage with FIXED integration"""
    
    # Get conversation memory
    memory = []
    for msg in st.session_state.messages[-4:]:
        role = "User" if msg["role"] == "user" else "Therapist"
        memory.append(f"{role}: {msg['content'][:100]}")
    memory_str = "\n".join(memory) if memory else "Start of session"
    
    # Adjust prompts based on ablation mode
    if st.session_state.ablation_mode == "baseline":
        # Minimal system prompt for baseline
        system_prompt = "You are a helpful mental health support chatbot. Be supportive and ask questions."
        user_prompt = f"User said: {user_text}\n\nRespond supportively."
        
    elif st.session_state.ablation_mode == "no_classifier":
        # No emotion awareness
        emotion = "neutral"
        system_prompt = f"""You are VentPal, providing mental health support. 
Stage: {stage}. Respond appropriately for this stage without emotion-specific guidance."""
        user_prompt = f"Session context: {memory_str}\nCurrent message: {user_text}\n\nRespond for {stage} stage."
        
    elif st.session_state.ablation_mode == "no_rag":
        # No context provided
        context = ""
        system_prompt = f"""You are VentPal, providing mental health support.
Stage: {stage}. User emotion: {emotion}. Provide support without external knowledge."""
        user_prompt = f"Session context: {memory_str}\nCurrent message: {user_text}\n\nRespond for {stage} stage."
        
    else:  # full_system
        # Full system with all components
        if stage == "greeting":
            system_prompt = """You are VentPal, a warm and professional mental health support companion. 
This is the start of a therapy session. Be welcoming, establish rapport, and gently check in on their wellbeing.
Keep responses to 2-3 sentences maximum. Always end with an open question about how they're doing."""
            user_prompt = f"The user just said: '{user_text}'\n\nRespond warmly and ask how they're doing today."
            
        elif stage == "exploration":
            emotion_guidance = {
                "anxiety": "They're feeling anxious. Be gentle and explore what's causing the worry.",
                "depression": "They're feeling depressed. Show empathy and gently explore their feelings.",
                "stress": "They're feeling stressed. Acknowledge the pressure and explore the sources.",
                "anger": "They're feeling angry. Validate their feelings and explore what's underneath.",
                "sadness": "They're feeling sad. Show compassion and gently explore what's happening."
            }.get(emotion, "Explore their feelings with curiosity and compassion.")
            
            system_prompt = f"""You are VentPal, providing empathetic mental health support. 
The user has shared they're not doing well. Your role is to:
1. Validate their feelings with empathy
2. Gently explore what's causing these feelings
3. Ask open-ended questions to understand better
4. Keep responses warm and supportive (2-3 sentences max)

Emotional context: {emotion_guidance}"""
            
            user_prompt = f"""Session context: {memory_str}
Current message: {user_text}
Respond with empathy and ask what's causing them to feel this way. Be curious but gentle."""
            
        elif stage == "consent":
            system_prompt = """You are VentPal. The user has shared what's bothering them. 
Now offer to share some helpful techniques, but ask for their permission first.
Be warm and explain that you have some evidence-based strategies that might help."""
            
            user_prompt = f"""Session context: {memory_str}
Current message: {user_text}
The user has explained their situation. Now offer to share some helpful CBT/DBT techniques, 
but ask for their consent first. Say something like "I have some techniques that might help with [their issue]. 
Would you like me to share them with you?" Keep it warm and collaborative."""
            
        else:  # support stage - FIXED INTEGRATION
            system_prompt = f"""You are VentPal, providing evidence-based mental health support.
CRITICAL: You MUST use the provided CBT/DBT techniques in your response.

Your structure:
1. Acknowledge what the user just shared (1 sentence)
2. Connect their insight to therapeutic concepts (1 sentence)
3. Provide ONE specific, actionable technique from the context (2-3 sentences)
4. Ask ONE focused question to help them apply it (1 sentence)

User emotion: {emotion}
Keep responses under 120 words. Be warm but practical."""
            
            user_prompt = f"""Conversation history:
{memory_str}

User's current message: {user_text}

CBT/DBT Techniques to integrate:
{context if context else "Focus on basic stress management and self-compassion techniques."}

RESPONSE STRUCTURE:
1. Acknowledge: "That's [insight about their response]"
2. Connect: "This shows [therapeutic insight about self-compassion/helping others]"
3. Technique: "[ONE specific technique from the context above]"
4. Application: "How could you try this [specific situation/timeframe]?"

If they mentioned helping others with positivity/support, connect this to self-compassion.
Use the actual techniques from the context - don't make up generic advice."""
    
    # Generate response
    try:
        completion = hf_client_primary().chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return completion.choices[0].message.content
    except:
        # Enhanced fallback for support stage
        if stage == "support" and context:
            # Extract a technique from context if possible
            context_lines = context.split('\n')
            technique_line = ""
            for line in context_lines:
                if any(word in line.lower() for word in ['breathe', 'technique', 'try', 'practice', 'step']):
                    technique_line = line.strip()[:100]
                    break
            
            if "positivity" in user_text.lower() or "support" in user_text.lower():
                return f"""That's such a compassionate way to support others - now let's apply that same kindness to yourself.

{technique_line if technique_line else "Try the 'STOP' technique: Stop what you're doing, Take a breath, Observe your thoughts and feelings, and Proceed with intention."}

When you notice stress building up today, how could you give yourself the same emotional support you'd give a friend?"""
            else:
                return f"""I hear what you're sharing. Let me offer a technique that might help:

{technique_line if technique_line else "When feeling overwhelmed, try grounding yourself by naming 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste."}

How do you think this might work for your situation?"""
        
        # Other fallbacks
        fallbacks = {
            "greeting": f"Hello {st.session_state.user_name}! I'm VentPal, and I'm here to support you today. How are you doing?",
            "exploration": "I hear that you're struggling right now. Can you tell me a bit more about what's been weighing on your mind?",
            "consent": "I understand what you're going through. I have some techniques that might help with this. Would you like me to share them with you?",
            "support": "Thank you for sharing that insight. Let's work together to apply some helpful strategies."
        }
        return fallbacks.get(stage, "I'm here to support you. What would be most helpful right now?")

def check_for_consent(text: str) -> bool:
    """Check if user has given consent for techniques"""
    consent_words = ["yes", "yeah", "sure", "okay", "ok", "please", "help", "share"]
    text_lower = text.lower().strip()
    return any(word in text_lower for word in consent_words) and "no" not in text_lower

def rate_ok() -> bool:
    now = time.time()
    if now - st.session_state.last_reset > 3600:
        st.session_state.request_count = 0
        st.session_state.last_reset = now
    return st.session_state.request_count < MAX_REQUESTS_PER_HOUR

def rate_inc():
    st.session_state.request_count += 1

# ================================ Main App ================================
def main():
    # Original header
    st.markdown(
        """<div class="main-header">
            <h1>💨 VentPal</h1>
            <p>Gentle support with CBT, DBT, and journaling.</p>
        </div>""",
        unsafe_allow_html=True
    )
    
    if not HUGGINGFACE_API_KEY:
        st.error("Missing HUGGINGFACE_API_KEY in secrets.")
        st.stop()
    
    # Initialize components
    with st.spinner("Connecting to knowledge base…"):
        vectorstore = create_vector_store()
    
    # Sidebar with thesis controls
    with st.sidebar:
        st.header("🔬 Thesis Controls")
        
        # Ablation study controls
        st.subheader("Ablation Study")
        ablation_options = {
            "full_system": "Full System (All Components)",
            "no_classifier": "No Classifier (Neutral Emotions)",
            "no_rag": "No RAG (LLM Only)",
            "baseline": "Baseline (Minimal System)"
        }
        
        selected_mode = st.selectbox(
            "System Configuration:",
            options=list(ablation_options.keys()),
            format_func=lambda x: ablation_options[x],
            index=0
        )
        
        if selected_mode != st.session_state.ablation_mode:
            st.session_state.ablation_mode = selected_mode
            st.info(f"Switched to: {ablation_options[selected_mode]}")
        
        # Show current mode
        st.markdown(f"<div class='ablation-mode'>Mode: {ablation_options[st.session_state.ablation_mode]}</div>", 
                   unsafe_allow_html=True)
        
        # Test scenarios
        st.subheader("Test Scenarios")
        scenario = st.selectbox("Load test scenario:", ["None"] + list(TEST_SCENARIOS.keys()))
        
        if st.button("Load Scenario") and scenario != "None":
            st.session_state.test_mode = True
            st.session_state.test_messages = TEST_SCENARIOS[scenario].copy()
            st.info(f"Loaded: {scenario}")
        
        # Component status
        st.subheader("🧩 System Status")
        
        # Show which components are active based on ablation mode
        classifier_active = st.session_state.ablation_mode not in ["no_classifier", "baseline"]
        rag_active = st.session_state.ablation_mode not in ["no_rag", "baseline"]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Classifier")
            chip("active ✓" if classifier_active else "disabled", "ok" if classifier_active else "warn")
        with col2:
            st.write("RAG")
            chip("active ✓" if rag_active else "disabled", "ok" if rag_active else "warn")
        
        # Session metrics
        st.subheader("📊 Session Metrics")
        st.metric("Stage", st.session_state.conversation_stage.title())
        st.metric("Current Emotion", st.session_state.current_emotion.title())
        st.metric("Messages", len(st.session_state.messages))
        st.metric("Ablation Mode", ablation_options[st.session_state.ablation_mode])
        
        if st.session_state.classifier_history:
            st.subheader("🔭 Last Classification")
            last = st.session_state.classifier_history[-1]
            if not last.get("_ablation"):
                st.caption(f"Emotion: {last['emotion']}")
                st.caption(f"Intent: {last['intent']}")
                st.caption(f"Severity: {last['severity']}")
                st.caption(f"Topic: {last['topic']}")
            else:
                st.caption("Classification disabled (ablation)")
        
        # Export thesis data
        if st.button("📥 Export Thesis Data"):
            thesis_data = {
                "session_info": {
                    "session_id": st.session_state.user_id,
                    "ablation_mode": st.session_state.ablation_mode,
                    "total_messages": len(st.session_state.messages),
                    "conversation_stage": st.session_state.conversation_stage
                },
                "classifier_history": st.session_state.classifier_history,
                "retrieval_history": st.session_state.retrieval_history,
                "conversation": [
                    {"role": m["role"], "content": m["content"], "metadata": m.get("metadata", {})}
                    for m in st.session_state.messages
                ]
            }
            
            st.download_button(
                "Download Thesis Data",
                json.dumps(thesis_data, indent=2),
                f"ventpal_thesis_{st.session_state.ablation_mode}_{st.session_state.user_id}.json",
                "application/json"
            )
        
        if st.button("🔄 New Session"):
            for key in ["messages", "conversation_stage", "current_emotion", "awaiting_consent"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Name gate
    name = st.text_input("What should I call you?", value=st.session_state.get("user_name", ""), placeholder="Your name")
    st.session_state.user_name = (name or "").strip()
    
    if not st.session_state.user_name:
        st.markdown("Enter a name to start your session.")
        st.markdown("---")
        footer_notice()
        return
    
    # Initial greeting if no messages
    if len(st.session_state.messages) == 0:
        greeting = f"Hello {st.session_state.user_name}! I'm VentPal, and I'm here to support you today. How are you doing?"
        st.session_state.messages.append({"role": "assistant", "content": greeting})
        st.session_state.conversation_stage = "greeting"
    
    # Display conversation
    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar=("🤖" if m["role"] == "assistant" else "👤")):
            st.markdown(m["content"])
            if m.get("skill_used"):
                st.markdown(f"<div class='skill-badge'>💡 {m['skill_used']}</div>", unsafe_allow_html=True)
            if m.get("sources"):
                st.caption("Sources: " + ", ".join(m["sources"]))
    
    # Handle test mode
    if hasattr(st.session_state, 'test_mode') and st.session_state.test_mode:
        if hasattr(st.session_state, 'test_messages') and st.session_state.test_messages:
            test_input = st.session_state.test_messages.pop(0)
            if not st.session_state.test_messages:
                st.session_state.test_mode = False
            process_user_input(test_input, vectorstore)
            time.sleep(2)  # Pause between test messages
            st.rerun()
    
    # Chat input
    user_text = st.chat_input(f"Share what's on your mind, {st.session_state.user_name}...")
    
    if not user_text:
        st.markdown("---")
        footer_notice()
        return
    
    if not rate_ok():
        st.error("You've hit the hourly limit. Try again later.")
        st.stop()
    
    process_user_input(user_text, vectorstore)

def process_user_input(user_text: str, vectorstore):
    """Process user input through the therapy flow with FIXED integration"""
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_text)
    
    # Crisis check first
    if detect_crisis_regex(user_text):
        msg = crisis_block()
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg)
        
        st.session_state.messages.extend([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": msg, "crisis_alert": True}
        ])
        
        rate_inc()
        st.markdown("---")
        footer_notice()
        return
    
    # Classify user input (respects ablation mode)
    use_classifier = st.session_state.ablation_mode not in ["no_classifier", "baseline"]
    clf = classify(user_text, use_classifier)
    
    emotion = "neutral"
    if clf and not clf.get("_ablation"):
        emo_lbl, emo_conf = _top(clf.get("emotion"), "neutral", 0.0)
        if emo_conf > 0.5:
            emotion = emo_lbl
            st.session_state.current_emotion = emotion
    
    # Determine conversation flow
    current_stage = st.session_state.conversation_stage
    
    # Stage transitions with FIXED support stage
    if current_stage == "greeting":
        st.session_state.conversation_stage = "exploration"
        reply = generate_therapy_response(user_text, "exploration", emotion)
        skill_used = None
        sources = []
        
    elif current_stage == "exploration":
        st.session_state.conversation_stage = "consent"
        st.session_state.awaiting_consent = True
        reply = generate_therapy_response(user_text, "consent", emotion)
        skill_used = None
        sources = []
        
    elif current_stage == "consent" and st.session_state.awaiting_consent:
        if check_for_consent(user_text):
            st.session_state.conversation_stage = "support"
            st.session_state.awaiting_consent = False
            
            # Use RAG (respects ablation mode)
            use_rag = st.session_state.ablation_mode not in ["no_rag", "baseline"]
            context, titles = get_relevant_context(user_text, vectorstore, use_rag)
            reply = generate_therapy_response(user_text, "support", emotion, context)
            
            skill_used = "CBT/DBT/Journaling" if context else None
            sources = titles if titles else []
        else:
            st.session_state.conversation_stage = "support"
            st.session_state.awaiting_consent = False
            reply = "That's completely fine. I'm still here to listen and support you. What would feel most helpful right now?"
            skill_used = None
            sources = []
    
    else:  # support stage - FIXED to always use RAG when available
        use_rag = st.session_state.ablation_mode not in ["no_rag", "baseline"]
        context, titles = get_relevant_context(user_text, vectorstore, use_rag)
        reply = generate_therapy_response(user_text, "support", emotion, context)
        skill_used = "CBT/DBT/Journaling" if context else None
        sources = titles if titles else []
    
    # Display response
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(reply)
        if skill_used:
            st.markdown(f"<div class='skill-badge'>💡 {skill_used}</div>", unsafe_allow_html=True)
        if sources:
            st.caption("Sources: " + ", ".join(set(sources)))
    
    # Save messages
    st.session_state.messages.extend([
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": reply, 
         "skill_used": skill_used,
         "sources": sources,
         "metadata": {
             "ablation_mode": st.session_state.ablation_mode,
             "emotion": emotion,
             "stage": current_stage
         }}
    ])
    
    rate_inc()
    st.markdown("---")
    footer_notice()

if __name__ == "__main__":
    main()
