# ventpal.py — VentPal with SECURE Configuration
# ----------------------------------------------------
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

# EXACT same imports as your working Colab
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma  # Keep the OLD import that works

from huggingface_hub import InferenceClient

# ================================ SECURE Configuration ================================
st.set_page_config(
    page_title="VentPal", 
    page_icon="💨", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
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
    st.session_state.conversation_stage = "greeting"
if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = "neutral"
if "awaiting_consent" not in st.session_state:
    st.session_state.awaiting_consent = False
if "classifier_history" not in st.session_state:
    st.session_state.classifier_history = []
if "retrieval_history" not in st.session_state:
    st.session_state.retrieval_history = []
if "ablation_mode" not in st.session_state:
    st.session_state.ablation_mode = "full_system"

# CORRECTED: Just reference the secrets keys - no default values needed
HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
MODEL_NAME = st.secrets["MODEL_NAME"]
EMBEDDING_MODEL = st.secrets["EMBEDDING_MODEL"]
FALLBACK_MODEL = st.secrets["FALLBACK_MODEL"]
VECTOR_DB_PATH = st.secrets["VECTOR_DB_PATH"]
COLLECTION_NAME = st.secrets["COLLECTION_NAME"]
MAX_REQUESTS_PER_HOUR = int(st.secrets["MAX_REQUESTS_PER_HOUR"])

# Classifier settings
CLASSIFIER_URL = st.secrets["CLASSIFIER_URL"]
CLASSIFIER_AUTH = st.secrets["CLASSIFIER_AUTH"]
CLASSIFIER_POLICY = st.secrets["CLASSIFIER_POLICY"]
CLASSIFIER_MODE = st.secrets["CLASSIFIER_MODE"]

# Severity settings
SEVERITY_ALERT_P = float(st.secrets["SEVERITY_ALERT_P"])
SEVERITY_HIGH_LABELS = st.secrets["SEVERITY_HIGH_LABELS"].split(",")
SEVERITY_ALERT_CONTAINS = st.secrets["SEVERITY_ALERT_CONTAINS"].split(",")

# RAG settings
MIN_EXCHANGES_BEFORE_RAG = int(st.secrets["MIN_EXCHANGES_BEFORE_RAG"])
ENABLE_RERANK = st.secrets["ENABLE_RERANK"].lower() == "true"
RERANK_MODEL_NAME = st.secrets["RERANK_MODEL_NAME"]
RERANK_CANDIDATES = int(st.secrets["RERANK_CANDIDATES"])
RERANK_TOP_K = int(st.secrets["RERANK_TOP_K"])

# Other settings
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
.debug-info{background: #f5f5f5; padding: 8px; border-radius: 4px; font-family: monospace; font-size: 11px; margin: 5px 0;}
.severity-alert{background: #ffebee; border: 2px solid #f44336; padding: 12px; border-radius: 8px; margin: 8px 0;}
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
    "Crisis Scenario": [
        "I'm not doing well",
        "I feel like ending it all",
        "Yes, I need help"
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

def check_severity_alert(classification: Dict) -> bool:
    """Check if severity classification triggers alert"""
    if not classification:
        return False
    
    severity_data = classification.get("severity", {})
    if not severity_data:
        return False
    
    top_pred = severity_data.get("top", {})
    if not top_pred:
        return False
    
    label = str(top_pred.get("label", "")).lower()
    conf = float(top_pred.get("conf", 0.0))
    
    # Check confidence threshold
    if conf >= SEVERITY_ALERT_P:
        # Check if label contains alert keywords
        return any(alert_word in label for alert_word in SEVERITY_ALERT_CONTAINS)
    
    return False

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

def severity_alert_block(severity_label: str, confidence: float) -> str:
    return (
        f"⚠️ I notice you might be going through something particularly difficult right now.\n\n"
        f"If you're feeling unsafe or in crisis, please don't hesitate to reach out for immediate support:\n\n"
        f"{CRISIS_RESOURCES_UK}\n\n"
        f"I'm here to support you through this. How are you feeling right now?"
    )

# ================================ Enhanced Classifier with Severity Alerts ================================
def classify(text: str, use_classifier: bool = True) -> Optional[Dict]:
    """Enhanced classifier with severity alerting"""
    if not use_classifier or st.session_state.ablation_mode in ["no_classifier", "baseline"]:
        return {
            "emotion": {"top": {"label": "neutral", "conf": 0.5}},
            "intent": {"top": {"label": "support", "conf": 0.5}},
            "severity": {"top": {"label": "green", "conf": 0.5}},
            "topic": {"top": {"label": "general", "conf": 0.5}},
            "_ablation": True
        }
    
    if not CLASSIFIER_URL or CLASSIFIER_POLICY == "never":
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
        
        # Safe classification tracking
        classification_entry = {
            "timestamp": datetime.now().isoformat(),
            "ablation_mode": st.session_state.ablation_mode,
            "latency_ms": out["_latency_ms"]
        }
        
        # Safely extract classifications
        for task in ["emotion", "intent", "severity", "topic"]:
            if task in out and out[task] and isinstance(out[task], dict):
                top_pred = out[task].get("top", {})
                if isinstance(top_pred, dict) and "label" in top_pred:
                    classification_entry[task] = top_pred["label"]
                    if task == "severity" and "conf" in top_pred:
                        classification_entry["severity_conf"] = top_pred["conf"]
        
        st.session_state.classifier_history.append(classification_entry)
        return out
        
    except Exception as e:
        st.sidebar.error(f"Classifier error: {str(e)[:50]}")
        return None

def _top(head: Optional[Dict], default_label="unknown", default_conf: float = 0.0) -> Tuple[str, float]:
    if not head or not isinstance(head, dict):
        return default_label, default_conf
    top = head.get("top")
    if not top or not isinstance(top, dict):
        return default_label, default_conf
    label = str(top.get("label", default_label)).lower()
    conf = float(top.get("conf", default_conf))
    return label, conf

# ================================ Enhanced RAG with Reranking ================================
@st.cache_resource(show_spinner=False)
def create_vector_store() -> Optional[Chroma]:
    """Create vector store using EXACT same setup as working Colab"""
    
    with st.sidebar:
        st.markdown(f"<div class='debug-info'>Vector DB: {VECTOR_DB_PATH}<br>Collection: {COLLECTION_NAME}<br>Embedding: {EMBEDDING_MODEL}<br>Rerank: {ENABLE_RERANK}</div>", unsafe_allow_html=True)
    
    if not os.path.exists(VECTOR_DB_PATH):
        st.sidebar.error(f"❌ Vector DB directory not found: `{VECTOR_DB_PATH}`")
        return None
    
    try:
        st.sidebar.info("🔄 Loading embedding model...")
        # EXACT same setup as Colab
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        st.sidebar.info("🔄 Connecting to vector database...")
        # EXACT same setup as Colab - using the OLD import
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        
        st.sidebar.info("🔄 Testing database connection...")
        # Test with same query as Colab
        test_docs = vectorstore.similarity_search("anxiety", k=2)
        
        if test_docs:
            st.sidebar.success(f"✅ Vector DB loaded successfully")
            st.sidebar.info(f"📊 Found {len(test_docs)} test documents")
            st.sidebar.info(f"🤖 Embedding: {EMBEDDING_MODEL}")
            if ENABLE_RERANK:
                st.sidebar.info(f"🔄 Reranking: {RERANK_MODEL_NAME}")
        else:
            st.sidebar.warning("⚠️ No documents found in test query")
        
        return vectorstore
        
    except Exception as e:
        error_msg = str(e)
        st.sidebar.error(f"❌ Vector store error: {error_msg[:100]}")
        
        # Specific error guidance
        if "no such column" in error_msg.lower():
            st.sidebar.error("🔧 **SOLUTION:**")
            st.sidebar.code("pip install langchain==0.1.20")
            st.sidebar.info("Version mismatch detected. Use same LangChain version as Colab.")
        
        return None

def get_relevant_context(query: str, vectorstore: Optional[Chroma], use_rag: bool = True) -> Tuple[str, List[str]]:
    """Enhanced RAG with reranking support"""
    if not use_rag or st.session_state.ablation_mode in ["no_rag", "baseline"]:
        st.session_state.retrieval_history.append({
            "timestamp": datetime.now().isoformat(),
            "ablation_mode": st.session_state.ablation_mode,
            "query": query[:100],
            "docs_retrieved": 0,
            "docs_used": 0,
            "_ablation": True
        })
        return "", []
    
    if not vectorstore:
        return "", []
    
    # Check minimum exchanges threshold
    if len(st.session_state.messages) < MIN_EXCHANGES_BEFORE_RAG * 2:  # *2 for user+assistant pairs
        return "", []
    
    try:
        # Retrieve more candidates if reranking is enabled
        k = RERANK_CANDIDATES if ENABLE_RERANK else 3
        relevant_docs = vectorstore.similarity_search(query, k=k)
        
        if not relevant_docs:
            return "", []
        
        # Apply reranking if enabled (simplified version)
        if ENABLE_RERANK and len(relevant_docs) > RERANK_TOP_K:
            # Simple reranking based on content length and keyword matching
            query_words = set(query.lower().split())
            scored_docs = []
            
            for doc in relevant_docs:
                content = doc.page_content.lower()
                # Simple scoring: keyword overlap + content quality
                keyword_score = sum(1 for word in query_words if word in content)
                length_score = min(len(content) / 1000, 1.0)  # Normalize length
                total_score = keyword_score + length_score
                scored_docs.append((total_score, doc))
            
            # Sort by score and take top K
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            relevant_docs = [doc for _, doc in scored_docs[:RERANK_TOP_K]]
        
        chunks = []
        titles = []
        
        for doc in relevant_docs:
            content = doc.page_content.strip()
            if content and len(content) > 100:
                chunks.append(content[:1200] + ("…" if len(content) > 1200 else ""))
                metadata = doc.metadata or {}
                title = metadata.get("source") or metadata.get("chunk_id") or "CBT/DBT Guide"
                titles.append(title)
        
        # Track retrieval
        st.session_state.retrieval_history.append({
            "timestamp": datetime.now().isoformat(),
            "ablation_mode": st.session_state.ablation_mode,
            "query": query[:100],
            "docs_retrieved": len(relevant_docs),
            "docs_used": len(chunks),
            "reranked": ENABLE_RERANK
        })
        
        context = "\n\n".join(chunks)
        
        if chunks:
            st.sidebar.info(f"📚 Retrieved {len(chunks)} chunks" + (f" (reranked from {k})" if ENABLE_RERANK else ""))
        
        return context, titles
        
    except Exception as e:
        st.sidebar.warning(f"Retrieval error: {str(e)[:50]}")
        return "", []

# ================================ LLM ================================
@st.cache_resource
def hf_client_primary() -> InferenceClient:
    return InferenceClient(model=MODEL_NAME, token=HUGGINGFACE_API_KEY, timeout=120)

@st.cache_resource
def hf_client_fallback() -> InferenceClient:
    return InferenceClient(model=FALLBACK_MODEL, token=HUGGINGFACE_API_KEY, timeout=120)

def generate_therapy_response(user_text: str, stage: str, emotion: str, context: str = "") -> str:
    """Generate response based on therapy session stage"""
    
    # Get conversation memory
    memory = []
    for msg in st.session_state.messages[-4:]:
        role = "User" if msg["role"] == "user" else "Therapist"
        memory.append(f"{role}: {msg['content'][:100]}")
    memory_str = "\n".join(memory) if memory else "Start of session"
    
    # Adjust prompts based on ablation mode
    if st.session_state.ablation_mode == "baseline":
        system_prompt = "You are a helpful mental health support chatbot. Be supportive and ask questions."
        user_prompt = f"User said: {user_text}\n\nRespond supportively."
        
    elif st.session_state.ablation_mode == "no_classifier":
        emotion = "neutral"
        system_prompt = f"""You are VentPal, providing mental health support. 
Stage: {stage}. Respond appropriately for this stage without emotion-specific guidance."""
        user_prompt = f"Session context: {memory_str}\nCurrent message: {user_text}\n\nRespond for {stage} stage."
        
    elif st.session_state.ablation_mode == "no_rag":
        context = ""
        system_prompt = f"""You are VentPal, providing mental health support.
Stage: {stage}. User emotion: {emotion}. Provide support without external knowledge."""
        user_prompt = f"Session context: {memory_str}\nCurrent message: {user_text}\n\nRespond for {stage} stage."
        
    else:  # full_system
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
            
        else:  # support stage
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
        user_name = st.session_state.get("user_name", "there")
        fallbacks = {
            "greeting": f"Hello {user_name}! I'm VentPal, and I'm here to support you today. How are you doing?",
            "exploration": "I hear that you're struggling right now. Can you tell me a bit more about what's been weighing on your mind?",
            "consent": "I understand what you're going through. I have some techniques that might help with this. Would you like me to share them with you?",
            "support": "Thank you for sharing that insight. Let's work together to apply some helpful strategies."
        }
        return fallbacks.get(stage, "I'm here to support you. What would be most helpful right now?")

def check_for_consent(text: str) -> bool:
    """Enhanced consent detection"""
    if not text:
        return False
        
    text_lower = text.lower().strip()
    
    # If ASSUME_YES is enabled, be more permissive
    if ASSUME_YES:
        # Strong no indicators
        if any(phrase in text_lower for phrase in ["no", "not now", "not interested", "don't want", "no thanks"]):
            return False
        # Otherwise assume yes unless explicitly no
        return True
    
    # Original strict consent checking
    consent_words = ["yes", "yeah", "sure", "okay", "ok", "please", "help", "share"]
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
    st.markdown(
        """<div class="main-header">
            <h1>💨 VentPal</h1>
            <p>Gentle support with CBT, DBT, and journaling techniques</p>
        </div>""",
        unsafe_allow_html=True
    )
    
    if not HUGGINGFACE_API_KEY:
        st.error("❌ Missing HUGGINGFACE_API_KEY in secrets. Please add your HuggingFace token to secrets.toml")
        st.stop()
    
    # Initialize components
    with st.spinner("🔄 Connecting to knowledge base..."):
        vectorstore = create_vector_store()
    
    # Sidebar with thesis controls
    with st.sidebar:
        st.header("🔬 Thesis Controls")
        
        # System configuration display
        st.subheader("🧩 System Status")
        classifier_active = st.session_state.ablation_mode not in ["no_classifier", "baseline"]
        rag_active = st.session_state.ablation_mode not in ["no_rag", "baseline"] and vectorstore is not None
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Classifier")
            chip("active ✓" if classifier_active else "disabled", "ok" if classifier_active else "warn")
        with col2:
            st.write("RAG")
            chip("active ✓" if rag_active else "disabled", "ok" if rag_active else "warn")
        
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
            st.success(f"✓ Switched to: {ablation_options[selected_mode]}")
        
        # Show current mode
        st.markdown(f"<div class='ablation-mode'>Active: {ablation_options[st.session_state.ablation_mode]}</div>", 
                   unsafe_allow_html=True)
        
        # Test scenarios
        st.subheader("Test Scenarios")
        scenario = st.selectbox("Load test scenario:", ["None"] + list(TEST_SCENARIOS.keys()))
        
        if st.button("🎯 Load Scenario") and scenario != "None":
            st.session_state.test_mode = True
            st.session_state.test_messages = TEST_SCENARIOS[scenario].copy()
            st.success(f"✓ Loaded: {scenario}")
        
        # Session metrics
        st.subheader("📊 Session Metrics")
        st.metric("Stage", st.session_state.conversation_stage.title())
        st.metric("Current Emotion", st.session_state.current_emotion.title())
        st.metric("Messages", len(st.session_state.messages))
        st.metric("Requests Used", f"{st.session_state.request_count}/{MAX_REQUESTS_PER_HOUR}")
        
        # Enhanced classification display
        if st.session_state.classifier_history:
            st.subheader("🔭 Last Classification")
            last = st.session_state.classifier_history[-1]
            if not last.get("_ablation"):
                for task in ["emotion", "intent", "severity", "topic"]:
                    if task in last:
                        value = last[task]
                        if task == "severity" and "severity_conf" in last:
                            conf = last["severity_conf"]
                            st.caption(f"{task.title()}: {value} ({conf:.2f})")
                        else:
                            st.caption(f"{task.title()}: {value}")
                if "latency_ms" in last:
                    st.caption(f"Latency: {last['latency_ms']}ms")
            else:
                st.caption("Classification disabled (ablation mode)")
        
        # Retrieval history
        if st.session_state.retrieval_history:
            st.subheader("📚 Last Retrieval")
            last_retrieval = st.session_state.retrieval_history[-1]
            if not last_retrieval.get("_ablation"):
                st.caption(f"Retrieved: {last_retrieval.get('docs_used', 0)} chunks")
                if last_retrieval.get("reranked"):
                    st.caption(f"Reranked from {last_retrieval.get('docs_retrieved', 0)}")
            else:
                st.caption("RAG disabled (ablation mode)")
        
        # Export thesis data
        if st.button("📥 Export Thesis Data"):
            thesis_data = {
                "session_info": {
                    "session_id": st.session_state.user_id,
                    "ablation_mode": st.session_state.ablation_mode,
                    "total_messages": len(st.session_state.messages),
                    "conversation_stage": st.session_state.conversation_stage,
                    "vector_store_available": vectorstore is not None,
                    "classifier_available": bool(CLASSIFIER_URL),
                    "rerank_enabled": ENABLE_RERANK,
                    "assume_yes": ASSUME_YES
                },
                "classifier_history": st.session_state.classifier_history,
                "retrieval_history": st.session_state.retrieval_history,
                "conversation": [
                    {"role": m["role"], "content": m["content"], "metadata": m.get("metadata", {})}
                    for m in st.session_state.messages
                ]
            }
            
            st.download_button(
                "📄 Download Data",
                json.dumps(thesis_data, indent=2),
                f"ventpal_thesis_{st.session_state.ablation_mode}_{st.session_state.user_id}.json",
                "application/json"
            )
        
        if st.button("🔄 New Session"):
            for key in ["messages", "conversation_stage", "current_emotion", "awaiting_consent", "classifier_history", "retrieval_history"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Name gate
    name = st.text_input("👋 What should I call you?", 
                        value=st.session_state.get("user_name", ""), 
                        placeholder="Enter your name to start")
    st.session_state.user_name = (name or "").strip()
    
    if not st.session_state.user_name:
        st.info("👆 Please enter your name to begin your VentPal session.")
        st.markdown("---")
        footer_notice()
        return
    
    # Initial greeting
    if len(st.session_state.messages) == 0:
        greeting = f"Hello {st.session_state.user_name}! I'm VentPal, and I'm here to support you today. How are you doing?"
        st.session_state.messages.append({
            "role": "assistant", 
            "content": greeting,
            "metadata": {"stage": "greeting", "emotion": "neutral"}
        })
        st.session_state.conversation_stage = "greeting"
    
    # Display conversation
    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"], avatar=("🤖" if m["role"] == "assistant" else "👤")):
            st.markdown(m["content"])
            if m.get("skill_used"):
                st.markdown(f"<div class='skill-badge'>💡 {m['skill_used']}</div>", unsafe_allow_html=True)
            if m.get("sources"):
                unique_sources = list(set(m["sources"]))
                st.caption("📚 Sources: " + ", ".join(unique_sources))
            if m.get("crisis_alert"):
                st.error("🚨 Crisis resources provided above")
            if m.get("severity_alert"):
                st.markdown("<div class='severity-alert'>⚠️ High severity detected - support resources available</div>", unsafe_allow_html=True)
    
    # Handle test mode
    if hasattr(st.session_state, 'test_mode') and st.session_state.test_mode:
        if hasattr(st.session_state, 'test_messages') and st.session_state.test_messages:
            test_input = st.session_state.test_messages.pop(0)
            if not st.session_state.test_messages:
                st.session_state.test_mode = False
            process_user_input(test_input, vectorstore)
            time.sleep(1)
            st.rerun()
    
    # Chat input
    user_text = st.chat_input(f"💬 Share what's on your mind, {st.session_state.user_name}...")
    
    if not user_text:
        st.markdown("---")
        footer_notice()
        return
    
    if not rate_ok():
        st.error(f"⏱️ You've reached the hourly limit ({MAX_REQUESTS_PER_HOUR} requests). Please try again in an hour.")
        st.stop()
    
    process_user_input(user_text, vectorstore)

def process_user_input(user_text: str, vectorstore: Optional[Chroma]):
    """Enhanced user input processing with severity alerts"""
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_text)
    
    # Crisis check first (regex-based)
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
    
    # Classify user input
    use_classifier = st.session_state.ablation_mode not in ["no_classifier", "baseline"]
    clf = classify(user_text, use_classifier)
    
    # Check for severity alerts
    severity_triggered = False
    if clf and not clf.get("_ablation") and check_severity_alert(clf):
        severity_data = clf.get("severity", {}).get("top", {})
        severity_label = severity_data.get("label", "unknown")
        severity_conf = severity_data.get("conf", 0.0)
        
        msg = severity_alert_block(severity_label, severity_conf)
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg)
        
        st.session_state.messages.extend([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": msg, "severity_alert": True}
        ])
        
        rate_inc()
        st.markdown("---")
        footer_notice()
        return
    
    # Extract emotion
    emotion = "neutral"
    if clf and not clf.get("_ablation"):
        emo_lbl, emo_conf = _top(clf.get("emotion"), "neutral", 0.0)
        if emo_conf > 0.3:
            emotion = emo_lbl
            st.session_state.current_emotion = emotion
    
    # Conversation flow
    current_stage = st.session_state.conversation_stage
    
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
            
            # Use RAG
            use_rag = st.session_state.ablation_mode not in ["no_rag", "baseline"]
            context, titles = get_relevant_context(user_text, vectorstore, use_rag)
            reply = generate_therapy_response(user_text, "support", emotion, context)
            
            skill_used = "CBT/DBT Techniques" if context else None
            sources = titles if titles else []
        else:
            st.session_state.conversation_stage = "support"
            st.session_state.awaiting_consent = False
            reply = "That's completely okay. I'm still here to listen and support you however feels most comfortable. What would be most helpful for you right now?"
            skill_used = None
            sources = []
    
    else:  # support stage
        use_rag = st.session_state.ablation_mode not in ["no_rag", "baseline"]
        context, titles = get_relevant_context(user_text, vectorstore, use_rag)
        reply = generate_therapy_response(user_text, "support", emotion, context)
        skill_used = "CBT/DBT Techniques" if context else None
        sources = titles if titles else []
    
    # Display response
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(reply)
        if skill_used:
            st.markdown(f"<div class='skill-badge'>💡 {skill_used}</div>", unsafe_allow_html=True)
        if sources:
            unique_sources = list(set(sources))
            st.caption("📚 Sources: " + ", ".join(unique_sources))
    
    # Save messages
    st.session_state.messages.extend([
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": reply, 
         "skill_used": skill_used,
         "sources": sources,
         "metadata": {
             "ablation_mode": st.session_state.ablation_mode,
             "emotion": emotion,
             "stage": current_stage,
             "timestamp": datetime.now().isoformat()
         }}
    ])
    
    rate_inc()
    st.markdown("---")
    footer_notice()

if __name__ == "__main__":
    main()
