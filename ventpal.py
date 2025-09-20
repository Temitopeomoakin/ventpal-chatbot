
# ventpal.py — VentPal with Fixed Vector Database
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

# Use OLD imports that match your database creation
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
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
if "flow_state" not in st.session_state: 
    st.session_state.flow_state = "greeting"
if "emotion_check_done" not in st.session_state: 
    st.session_state.emotion_check_done = False
if "consent_given" not in st.session_state: 
    st.session_state.consent_given = False
if "skill_counters" not in st.session_state: 
    st.session_state.skill_counters = {}
if "last_skill_used" not in st.session_state: 
    st.session_state.last_skill_used = None
if "turn_count" not in st.session_state: 
    st.session_state.turn_count = 0
if "ablation_mode" not in st.session_state: 
    st.session_state.ablation_mode = "full"
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="response"
    )

# Load configuration from secrets
HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
MODEL_NAME = st.secrets["MODEL_NAME"]
EMBEDDING_MODEL = st.secrets["EMBEDDING_MODEL"]
FALLBACK_MODEL = st.secrets["FALLBACK_MODEL"]
VECTOR_DB_PATH = st.secrets["VECTOR_DB_PATH"]
COLLECTION_NAME = st.secrets["COLLECTION_NAME"]
MAX_REQUESTS_PER_HOUR = int(st.secrets["MAX_REQUESTS_PER_HOUR"])

# Classifier settings
CLASSIFIER_URL = st.secrets["CLASSIFIER_URL"]
CLASSIFIER_POLICY = st.secrets["CLASSIFIER_POLICY"]
CLASSIFIER_AUTH = st.secrets["CLASSIFIER_AUTH"]
CLASSIFIER_MODE = st.secrets["CLASSIFIER_MODE"]
MIN_EXCHANGES_BEFORE_RAG = int(st.secrets["MIN_EXCHANGES_BEFORE_RAG"])

# Severity settings
SEVERITY_ALERT_P = float(st.secrets["SEVERITY_ALERT_P"])
SEVERITY_HIGH_LABELS = st.secrets["SEVERITY_HIGH_LABELS"]  # Already a list
SEVERITY_ALERT_CONTAINS = st.secrets["SEVERITY_ALERT_CONTAINS"]  # Already a list

# RAG settings
ENABLE_RERANK = st.secrets["ENABLE_RERANK"].lower() == "true"
RERANK_MODEL_NAME = st.secrets["RERANK_MODEL_NAME"]
RERANK_CANDIDATES = int(st.secrets["RERANK_CANDIDATES"])
RERANK_TOP_K = int(st.secrets["RERANK_TOP_K"])

# Skill settings
ASSUME_YES = st.secrets["ASSUME_YES"].lower() == "true"
SKILL_GRACE_TURNS = int(st.secrets["SKILL_GRACE_TURNS"])
SKILL_COOLDOWN_TURNS = int(st.secrets["SKILL_COOLDOWN_TURNS"])

# HuggingFace client
client = InferenceClient(api_key=HUGGINGFACE_API_KEY)

# ================================ Styles ================================
st.markdown("""
<style>
.debug-info {
    font-size: 0.8em;
    color: #666;
    margin: 10px 0;
    padding: 5px;
    background: #f0f0f0;
    border-radius: 3px;
}
.skill-badge {
    display: inline-block;
    padding: 2px 8px;
    margin: 2px;
    border-radius: 12px;
    font-size: 0.85em;
    font-weight: 500;
}
.classification-info {
    font-size: 0.75em;
    color: #888;
    margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)

# ================================ Vector Store with Debugging ================================
@st.cache_resource(show_spinner=False)
def create_vector_store() -> Optional[Chroma]:
    """Create vector store with debugging"""
    
    with st.sidebar:
        st.markdown(f"<div class='debug-info'>Vector DB: {VECTOR_DB_PATH}<br>Collection: {COLLECTION_NAME}<br>Embedding: {EMBEDDING_MODEL}<br>Rerank: {ENABLE_RERANK}</div>", unsafe_allow_html=True)
    
    # DEBUG: Check current directory and files
    st.sidebar.info(f"Current directory: {os.getcwd()}")
    st.sidebar.info(f"Files in current dir: {os.listdir('.')[:10]}")  # Show first 10 files
    
    # Check if vector_db_new exists
    if os.path.exists(VECTOR_DB_PATH):
        st.sidebar.success(f"✓ Found {VECTOR_DB_PATH}")
        st.sidebar.info(f"Contents: {os.listdir(VECTOR_DB_PATH)[:5]}")  # Show first 5 files
        
        # Check for chroma.sqlite3
        db_file = os.path.join(VECTOR_DB_PATH, "chroma.sqlite3")
        if os.path.exists(db_file):
            st.sidebar.success(f"✓ Found database file")
            file_size = os.path.getsize(db_file) / (1024*1024)  # Size in MB
            st.sidebar.info(f"Database size: {file_size:.2f} MB")
        else:
            st.sidebar.warning(f"⚠️ No chroma.sqlite3 in {VECTOR_DB_PATH}")
    else:
        st.sidebar.error(f"✗ {VECTOR_DB_PATH} not found")
        st.sidebar.info(f"Looking for alternatives...")
        
        # Try to find it in different locations
        possible_paths = [
            "vector_db_new",
            "./vector_db_new",
            "../vector_db_new",
            "ventpal-chatbot/vector_db_new",
            "/mount/src/ventpal-chatbot/vector_db_new",
            "/app/vector_db_new"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                st.sidebar.success(f"✓ Found database at: {path}")
                st.sidebar.info(f"Update VECTOR_DB_PATH in secrets to: {path}")
                VECTOR_DB_PATH = path  # Use found path
                break
    
    try:
        st.sidebar.info("🔄 Loading embedding model...")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        st.sidebar.info("🔄 Connecting to vector database...")
        
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        
        # Test the connection
        try:
            test_results = vectorstore.similarity_search("test", k=1)
            st.sidebar.success(f"✅ Connected to vector database!")
        except:
            st.sidebar.warning("⚠️ Database connected but might be empty")
        
        return vectorstore
        
    except Exception as e:
        st.sidebar.error(f"❌ Vector store error: {str(e)}")
        return None

# ================================ Classifier ================================
def classify(text: str, use_classifier: bool = True) -> Optional[Dict]:
    """Classifier with ablation study support"""
    if not use_classifier or st.session_state.ablation_mode in ["no_classifier", "baseline"]:
        return None
    
    if not CLASSIFIER_URL:
        return None
    
    try:
        response = requests.post(
            CLASSIFIER_URL,
            json={"text": text},
            headers={"Authorization": CLASSIFIER_AUTH} if CLASSIFIER_AUTH else {},
            timeout=5
        )
        
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

# ================================ LLM Call ================================
def llm_call(prompt: str, system_prompt: str = None) -> str:
    """Call LLM with HuggingFace"""
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Add memory context
    if st.session_state.memory and st.session_state.memory.chat_memory:
        for msg in st.session_state.memory.chat_memory.messages[-6:]:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
    
    messages.append({"role": "user", "content": prompt})
    
    try:
        response = client.chat_completion(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=256,
            temperature=0.7
        )
        return response.choices[0].message.content
    except:
        # Fallback
        try:
            response = client.chat_completion(
                model=FALLBACK_MODEL,
                messages=messages,
                max_tokens=256,
                temperature=0.7
            )
            return response.choices[0].message.content
        except:
            return "I apologize, but I'm having trouble connecting right now. Please try again."

# ================================ RAG Search ================================
def search_rag(query: str, k: int = 4) -> List[Document]:
    """Search vector database for relevant documents"""
    if st.session_state.ablation_mode in ["no_rag", "baseline"]:
        return []
    
    vectorstore = create_vector_store()
    if not vectorstore:
        return []
    
    try:
        results = vectorstore.similarity_search(query, k=k)
        return results
    except:
        return []

# ================================ Generate Response ================================
def generate_response(user_input: str) -> str:
    """Generate response with optional RAG and classification"""
    
    # Classify input
    classification = classify(user_input)
    
    # Determine if RAG should be used
    use_rag = False
    if st.session_state.ablation_mode not in ["no_rag", "baseline"]:
        if len(st.session_state.messages) >= MIN_EXCHANGES_BEFORE_RAG * 2:
            use_rag = True
    
    # Build prompt
    prompt = user_input
    system_prompt = "You are VentPal, a supportive mental health companion trained in CBT techniques."
    
    # Add RAG context if applicable
    if use_rag:
        docs = search_rag(user_input, k=RERANK_TOP_K if ENABLE_RERANK else 4)
        if docs:
            context = "\n\n".join([doc.page_content for doc in docs[:3]])
            system_prompt += f"\n\nUse these CBT resources to inform your response:\n{context}"
    
    # Add classification context if available
    if classification:
        emotion = classification.get("emotion_label", "neutral")
        severity = classification.get("severity_label", "low")
        system_prompt += f"\n\nUser's emotional state: {emotion} (severity: {severity})"
    
    # Generate response
    response = llm_call(prompt, system_prompt)
    
    # Store in memory
    st.session_state.memory.chat_memory.add_user_message(user_input)
    st.session_state.memory.chat_memory.add_ai_message(response)
    
    return response

# ================================ Main Interface ================================
def main():
    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🔬 Thesis Controls")
        
        # System status
        st.markdown("#### 🧩 System Status")
        col1, col2 = st.columns(2)
        
        with col1:
            classifier_active = st.session_state.ablation_mode not in ["no_classifier", "baseline"]
            status = "active ✓" if classifier_active else "disabled"
            st.markdown(f"**Classifier**<br>{status}", unsafe_allow_html=True)
        
        with col2:
            rag_active = st.session_state.ablation_mode not in ["no_rag", "baseline"]
            status = "active ✓" if rag_active else "disabled"
            st.markdown(f"**RAG**<br>{status}", unsafe_allow_html=True)
        
        # Ablation study controls
        st.markdown("#### Ablation Study")
        ablation_mode = st.selectbox(
            "System Configuration:",
            ["full", "no_classifier", "no_rag", "baseline"],
            format_func=lambda x: {
                "full": "Full System (All Components)",
                "no_classifier": "Without Classifier",
                "no_rag": "Without RAG",
                "baseline": "Baseline (No Components)"
            }[x]
        )
        
        if ablation_mode != st.session_state.ablation_mode:
            st.session_state.ablation_mode = ablation_mode
            st.rerun()
        
        # Initialize vector store
        create_vector_store()
    
    # Main chat interface
    st.title("💨 VentPal - Mental Health Support Chatbot")
    
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("How are you feeling today?"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt)
                st.markdown(response)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Update turn count
        st.session_state.turn_count += 1

if __name__ == "__main__":
    main()
