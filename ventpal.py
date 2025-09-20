# ventpal.py — VentPal with Fixed Vector Database Configuration
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

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma  

from huggingface_hub import InferenceClient

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

# Load secrets
HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
MODEL_NAME = st.secrets["MODEL_NAME"]
FALLBACK_MODEL = st.secrets["FALLBACK_MODEL"]
EMBEDDING_MODEL = st.secrets["EMBEDDING_MODEL"]
VECTOR_DB_PATH = st.secrets["VECTOR_DB_PATH"]
COLLECTION_NAME = st.secrets["COLLECTION_NAME"]
CLASSIFIER_URL = st.secrets["CLASSIFIER_URL"]
CLASSIFIER_AUTH = st.secrets["CLASSIFIER_AUTH"]
MAX_REQUESTS_PER_HOUR = int(st.secrets["MAX_REQUESTS_PER_HOUR"])

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

# ================================ Classifier ================================
def classify(text: str, use_classifier: bool = True) -> Optional[Dict]:
    """Classifier with ablation study support"""
    if not use_classifier or st.session_state.ablation_mode in ["no_classifier", "baseline"]:
        return {
            "emotion": {"top": {"label": "neutral", "conf": 0.5}},
            "intent": {"top": {"label": "support", "conf": 0.5}},
            "severity": {"top": {"label": "green", "conf": 0.5}},
            "topic": {"top": {"label": "general", "conf": 0.5}},
            "_ablation": True
        }
    
    if not CLASSIFIER_URL:
        st.sidebar.info("🔧 Classifier URL not configured - using neutral classifications")
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
        
        classification_entry = {
            "timestamp": datetime.now().isoformat(),
            "ablation_mode": st.session_state.ablation_mode,
            "latency_ms": out["_latency_ms"]
        }
        
        for task in ["emotion", "intent", "severity", "topic"]:
            if task in out and out[task] and isinstance(out[task], dict):
                top_pred = out[task].get("top", {})
                if isinstance(top_pred, dict) and "label" in top_pred:
                    classification_entry[task] = top_pred["label"]
        
        st.session_state.classifier_history.append(classification_entry)
        return out
        
    except requests.exceptions.Timeout:
        st.sidebar.warning("⏱️ Classifier timeout - using neutral classification")
        return None
    except requests.exceptions.ConnectionError:
        st.sidebar.warning("🔌 Classifier connection failed - using neutral classification")
        return None
    except Exception as e:
        st.sidebar.error(f"❌ Classifier error: {str(e)[:100]}")
        return None

def _top(head: Optional[Dict], default_label="unknown", default_conf: float = 0.0) -> Tuple[str, float]:
    """Safer extraction of top prediction"""
    if not head or not isinstance(head, dict):
        return default_label, default_conf
    
    top = head.get("top")
    if not top or not isinstance(top, dict):
        return default_label, default_conf
    
    label = str(top.get("label", default_label)).lower()
    conf = float(top.get("conf", default_conf))
    return label, conf

# ================================ Vector Store ================================
@st.cache_resource(show_spinner=False)
def create_vector_store() -> Optional[Chroma]:
    """Create or load vector store"""
    
    with st.sidebar:
        st.markdown(f"<div class='debug-info'>Vector DB Path: {VECTOR_DB_PATH}<br>Collection: {COLLECTION_NAME}<br>Embedding Model: {EMBEDDING_MODEL}</div>", unsafe_allow_html=True)
    
    # DEBUG: Check where we are and what files exist
    if not os.path.exists(VECTOR_DB_PATH):
        st.sidebar.warning(f"⚠️ {VECTOR_DB_PATH} not found in current directory")
        st.sidebar.info(f"Current directory: {os.getcwd()}")
        
        # Check if it exists in other common locations
        possible_paths = [
            os.path.join(os.getcwd(), "vector_db_new"),
            "/mount/src/ventpal-chatbot/vector_db_new",
            "../vector_db_new"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                st.sidebar.success(f"✅ Found database at: {path}")
                st.sidebar.info(f"Update your secrets.toml:")
                st.sidebar.code(f'VECTOR_DB_PATH = "{path}"')
                break
        else:
            st.sidebar.error("❌ Vector database not found in any expected location")
            return None
    
    try:
        st.sidebar.info("🔄 Loading embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        st.sidebar.info("🔄 Connecting to vector database...")
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        
        # Test the connection
        st.sidebar.info("🔄 Testing database connection...")
        test_docs = vectorstore.similarity_search("stress", k=1)
        
        if test_docs:
            st.sidebar.success(f"✅ Vector DB loaded successfully")
            st.sidebar.info(f"📊 Collection: {COLLECTION_NAME}")
            st.sidebar.info(f"🤖 Embedding: {EMBEDDING_MODEL}")
            st.sidebar.info(f"🔍 Test found {len(test_docs)} documents")
        else:
            st.sidebar.warning("⚠️ Vector DB connected but no documents found")
        
        return vectorstore
        
    except Exception as e:
        error_msg = str(e)
        st.sidebar.error(f"❌ Vector store error: {error_msg[:100]}")
        
        if "no such column" in error_msg.lower():
            st.sidebar.info("💡 This looks like a collection name mismatch. Try:")
            st.sidebar.code('COLLECTION_NAME = "cbt_docs_fresh_2024"')
        elif "no such table" in error_msg.lower():
            st.sidebar.info("💡 Database schema issue. You may need to recreate the vector store.")
        
        return None

def get_relevant_context(query: str, vectorstore: Optional[Chroma], use_rag: bool = True) -> Tuple[str, List[str]]:
    """RAG with ablation study support"""
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
        st.sidebar.warning("📚 RAG disabled - vector store not available")
        return "", []
    
    try:
        relevant_docs = vectorstore.similarity_search(query, k=3)
        
        if not relevant_docs:
            st.sidebar.info(f"🔍 No relevant documents found for: '{query[:30]}...'")
            return "", []
        
        chunks = []
        titles = []
        
        for doc in relevant_docs:
            if not hasattr(doc, 'page_content') or not doc.page_content:
                continue
                
            content = doc.page_content.strip()
            if content and len(content) > 50:
                max_length = 1000
                if len(content) > max_length:
                    content = content[:max_length] + "…"
                chunks.append(content)
                
                metadata = getattr(doc, 'metadata', {}) or {}
                title = (metadata.get("source") or 
                        metadata.get("title") or 
                        metadata.get("chunk_id") or 
                        "CBT/DBT Guide")
                titles.append(str(title))
        
        st.session_state.retrieval_history.append({
            "timestamp": datetime.now().isoformat(),
            "ablation_mode": st.session_state.ablation_mode,
            "query": query[:100],
            "docs_retrieved": len(relevant_docs),
            "docs_used": len(chunks),
            "avg_chunk_length": sum(len(c) for c in chunks) // max(len(chunks), 1)
        })
        
        context = "\n\n".join(chunks)
        
        if chunks:
            st.sidebar.info(f"📚 Retrieved {len(chunks)} relevant chunks ({len(context)} chars)")
        
        return context, titles
        
    except Exception as e:
        st.sidebar.warning(f"🔍 Retrieval error: {str(e)[:50]}")
        return "", []

# ================================ LLM ================================
@st.cache_resource
def hf_client_primary() -> InferenceClient:
    return InferenceClient(model=MODEL_NAME, token=HUGGINGFACE_API_KEY, timeout=120)

@st.cache_resource
def hf_client_fallback() -> InferenceClient:
    return InferenceClient(model=FALLBACK_MODEL, token=HUGGINGFACE_API_KEY, timeout=120)

def generate_therapy_response(user_text: str, stage: str, emotion: str, context: str = "") -> str:
    """Generate response with better prompt engineering"""
    
    # Get conversation memory
    memory = []
    for msg in st.session_state.messages[-4:]:
        role = "User" if msg["role"] == "user" else "Therapist"
        content = msg['content'][:150] + ("..." if len(msg['content']) > 150 else "")
        memory.append(f"{role}: {content}")
    memory_str = "\n".join(memory) if memory else "Start of session"
    
    # Build prompts based on ablation mode
    if st.session_state.ablation_mode == "baseline":
        system_prompt = "You are a helpful mental health support chatbot. Be supportive, empathetic, and ask helpful questions. Keep responses under 100 words."
        user_prompt = f"User said: {user_text}\n\nRespond supportively and helpfully."
        
    elif st.session_state.ablation_mode == "no_classifier":
        emotion = "neutral"
        system_prompt = f"""You are VentPal, providing mental health support. 
Current stage: {stage}. Respond appropriately for this conversation stage.
Be warm, empathetic, and professional. Keep responses under 100 words."""
        user_prompt = f"Conversation context: {memory_str}\nUser's current message: {user_text}\n\nRespond appropriately for the {stage} stage."
        
    elif st.session_state.ablation_mode == "no_rag":
        context = ""
        system_prompt = f"""You are VentPal, providing mental health support.
Stage: {stage}. User emotion: {emotion}. 
Provide support using your general knowledge of CBT and DBT techniques.
Keep responses under 100 words."""
        user_prompt = f"Conversation context: {memory_str}\nUser's message: {user_text}\n\nProvide supportive response for {stage} stage."
        
    else:  # full_system
        if stage == "greeting":
            system_prompt = """You are VentPal, a warm and professional mental health support companion. 
This is the start of a therapy session. Be welcoming, establish rapport, and gently check in.
Keep responses to 2-3 sentences. Always end with an open question about how they're doing."""
            user_prompt = f"The user just said: '{user_text}'\n\nRespond warmly and ask how they're doing today."
            
        elif stage == "exploration":
            emotion_guidance = {
                "anxiety": "They seem anxious. Be gentle and explore what's causing the worry.",
                "depression": "They seem depressed. Show empathy and gently explore their feelings.",
                "stress": "They seem stressed. Acknowledge the pressure and explore the sources.",
                "anger": "They seem angry. Validate their feelings and explore what's underneath.",
                "sadness": "They seem sad. Show compassion and gently explore what's happening."
            }.get(emotion, "Explore their feelings with curiosity and compassion.")
            
            system_prompt = f"""You are VentPal, providing empathetic mental health support. 
The user has shared they're not doing well. Your role:
1. Validate their feelings with empathy
2. Gently explore what's causing these feelings  
3. Ask open-ended questions to understand better
4. Keep responses warm and supportive (2-3 sentences)

Context: {emotion_guidance}"""
            
            user_prompt = f"""Recent conversation: {memory_str}
Current message: {user_text}

Respond with empathy and gently ask what's causing them to feel this way."""
            
        elif stage == "consent":
            system_prompt = """You are VentPal. The user has shared what's bothering them. 
Now offer to share helpful techniques, but ask for permission first.
Be warm and explain you have evidence-based strategies that might help."""
            
            user_prompt = f"""Recent conversation: {memory_str}
Current message: {user_text}

The user has explained their situation. Offer to share helpful CBT/DBT techniques, 
but ask for consent first. Be warm and collaborative."""
            
        else:  # support stage
            system_prompt = f"""You are VentPal, providing evidence-based mental health support.
IMPORTANT: Use the provided CBT/DBT techniques in your response.

Structure your response:
1. Acknowledge what they shared (1 sentence)
2. Connect to therapeutic concepts (1 sentence)  
3. Share ONE specific technique from the context (2-3 sentences)
4. Ask how they could apply it (1 sentence)

User emotion: {emotion}. Keep under 120 words. Be warm but practical."""
            
            context_summary = ""
            if context:
                context_lines = [line.strip() for line in context.split('\n') if line.strip()]
                technique_lines = []
                for line in context_lines[:10]:
                    if any(word in line.lower() for word in ['technique', 'try', 'practice', 'step', 'breathe', 'ground']):
                        technique_lines.append(line[:150])
                        if len(technique_lines) >= 2:
                            break
                context_summary = "\n".join(technique_lines) if technique_lines else context[:300]
            
            user_prompt = f"""Recent conversation: {memory_str}

User's current message: {user_text}

CBT/DBT Techniques to integrate:
{context_summary if context_summary else "Use basic stress management, grounding, and self-compassion techniques."}

Follow the response structure. If they mentioned helping others, connect this to self-compassion.
Use actual techniques from the context - don't create generic advice."""
    
    # Call LLM with fallback
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
        response = completion.choices[0].message.content
        
        if len(response.strip()) < 10:
            raise ValueError("Response too short")
            
        return response.strip()
        
    except Exception as primary_error:
        st.sidebar.warning(f"Primary model failed: {str(primary_error)[:50]}")
        
        try:
            completion = hf_client_fallback().chat.completions.create(
                model=FALLBACK_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            response = completion.choices[0].message.content
            st.sidebar.info("🔄 Using fallback model")
            return response.strip()
            
        except Exception as fallback_error:
            st.sidebar.error(f"Both models failed: {str(fallback_error)[:50]}")
            
            # Contextual fallback responses
            user_name = st.session_state.get("user_name", "there")
            
            if stage == "support" and context:
                context_lines = context.split('\n')
                technique = ""
                for line in context_lines:
                    if any(word in line.lower() for word in ['breathe', 'try', 'technique', 'practice']):
                        technique = line.strip()[:200]
                        break
                
                if "positivity" in user_text.lower() or "support" in user_text.lower():
                    return f"That's such a compassionate way to support others. Now let's apply that same kindness to yourself.\n\n{technique if technique else 'Try taking three deep breaths and asking yourself: what would I tell a good friend in this situation?'}\n\nHow could you show yourself that same compassion today?"
                else:
                    return f"I hear what you're sharing. {technique if technique else 'When feeling overwhelmed, try the 5-4-3-2-1 grounding technique: name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste.'}\n\nHow do you think this might help your situation?"
            
            fallbacks = {
                "greeting": f"Hello {user_name}! I'm VentPal, and I'm here to support you today. How are you feeling right now?",
                "exploration": "I hear that you're going through a difficult time. Can you tell me a bit more about what's been weighing on your mind lately?",
                "consent": "I understand what you're experiencing sounds really challenging. I have some evidence-based techniques that might help. Would you like me to share some strategies with you?",
                "support": "Thank you for sharing that with me. Let's work together on some strategies that might help you feel more grounded and supported."
            }
            return fallbacks.get(stage, "I'm here to support you. What would be most helpful for you right now?")

def check_for_consent(text: str) -> bool:
    """Better consent detection"""
    if not text:
        return False
        
    text_lower = text.lower().strip()
    
    strong_yes = ["yes", "yeah", "sure", "okay", "ok", "please", "help me", "share them", "that would help"]
    if any(phrase in text_lower for phrase in strong_yes):
        return True
    
    if any(phrase in text_lower for phrase in ["no", "not now", "not interested", "don't want"]):
        return False
    
    moderate_yes = ["what are", "how do", "tell me more", "sounds good", "i'd like"]
    return any(phrase in text_lower for phrase in moderate_yes)

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
    # Header
    st.markdown(
        """<div class="main-header">
            <h1>💨 VentPal</h1>
            <p>Gentle support with CBT, DBT, and journaling techniques</p>
        </div>""",
        unsafe_allow_html=True
    )
    
    if not HUGGINGFACE_API_KEY:
        st.error("❌ Missing HUGGINGFACE_API_KEY in secrets. Please add your HuggingFace token.")
        st.stop()
    
    # Initialize vector store
    vectorstore = None
    with st.spinner("🔄 Connecting to knowledge base..."):
        vectorstore = create_vector_store()
    
    if not vectorstore:
        st.warning("⚠️ Running without RAG capability. Some features may be limited.")
    
    # Sidebar controls
    with st.sidebar:
        st.header("🔬 Thesis Controls")
        
        # System status
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
        st.metric("Conversation Stage", st.session_state.conversation_stage.title())
        st.metric("Current Emotion", st.session_state.current_emotion.title())
        st.metric("Total Messages", len(st.session_state.messages))
        st.metric("Requests Used", f"{st.session_state.request_count}/{MAX_REQUESTS_PER_HOUR}")
        
        # Classification history
        if st.session_state.classifier_history:
            st.subheader("🔭 Last Classification")
            last = st.session_state.classifier_history[-1]
            if not last.get("_ablation"):
                for task in ["emotion", "intent", "severity", "topic"]:
                    if task in last:
                        st.caption(f"{task.title()}: {last[task]}")
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
                if "avg_chunk_length" in last_retrieval:
                    st.caption(f"Avg length: {last_retrieval['avg_chunk_length']} chars")
            else:
                st.caption("RAG disabled (ablation mode)")
        
        # Export data
        if st.button("📥 Export Thesis Data"):
            thesis_data = {
                "session_info": {
                    "session_id": st.session_state.user_id,
                    "ablation_mode": st.session_state.ablation_mode,
                    "total_messages": len(st.session_state.messages),
                    "conversation_stage": st.session_state.conversation_stage,
                    "vector_store_available": vectorstore is not None,
                    "classifier_available": bool(CLASSIFIER_URL)
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
    
    # Name input
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
    """Process user input with conversation flow"""
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_text)
    
    # Crisis check
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
