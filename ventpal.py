# ventpal.py — VentPal with Direct Implementations (No LangChain)
# ----------------------------------------------------

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

# Direct imports - no langchain
import chromadb
from sentence_transformers import SentenceTransformer
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
if "memory" not in st.session_state:
    st.session_state.memory = []  # Simple list of messages
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

# Initialize HuggingFace client
client = InferenceClient(
    api_key=HUGGINGFACE_API_KEY
)

# ================================ Direct ChromaDB Implementation ================================
@st.cache_resource(show_spinner=False)
def create_vector_store():
    """Create vector store using chromadb directly"""
    
    with st.sidebar:
        st.markdown(f"<div class='debug-info'>Vector DB: {VECTOR_DB_PATH}<br>Collection: {COLLECTION_NAME}<br>Embedding: {EMBEDDING_MODEL}<br>Rerank: {ENABLE_RERANK}</div>", unsafe_allow_html=True)
    
    if not os.path.exists(VECTOR_DB_PATH):
        st.sidebar.error(f"❌ Vector database not found at: {VECTOR_DB_PATH}")
        return None
    
    try:
        st.sidebar.info("🔄 Loading embedding model...")
        
        # Load embedding model directly
        embedder = SentenceTransformer(EMBEDDING_MODEL)
        
        st.sidebar.info("🔄 Connecting to vector database...")
        
        # Connect directly to chromadb
        chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        
        # Try to get the collection - try both names
        collection = None
        for name in [COLLECTION_NAME, f"my_{COLLECTION_NAME}"]:
            try:
                collection = chroma_client.get_collection(name=name)
                st.sidebar.success(f"✅ Connected to collection: {name}")
                break
            except:
                continue
        
        if not collection:
            st.sidebar.error("❌ Could not find collection in database")
            return None
        
        # Return both the collection and embedder
        return collection, embedder
        
    except Exception as e:
        st.sidebar.error(f"❌ Vector store error: {str(e)}")
        st.sidebar.info("""
        🔧 SOLUTION:
        
        The vector database needs to be accessible.
        Check that the path and collection name match.
        """)
        return None

# ================================ RAG Search Function ================================
def search_documents(query: str, k: int = 4):
    """Search for relevant documents"""
    
    result = create_vector_store()
    if not result:
        return []
    
    collection, embedder = result
    
    try:
        # Embed the query
        query_embedding = embedder.encode(query).tolist()
        
        # Search in ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        # Format results
        documents = []
        if results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                documents.append({
                    'content': doc,
                    'metadata': metadata
                })
        
        return documents
        
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

# ================================ LLM Call Function ================================
def llm_call(prompt: str, use_rag: bool = False) -> str:
    """Call LLM with or without RAG context"""
    
    # Add RAG context if needed
    if use_rag:
        docs = search_documents(prompt, k=RERANK_TOP_K if ENABLE_RERANK else 4)
        if docs:
            context = "\n\n".join([d['content'] for d in docs])
            prompt = f"""Based on the following CBT resources:

{context}

User question: {prompt}

Please provide a helpful response based on the CBT techniques described above:"""
    
    # Add conversation memory
    if st.session_state.memory:
        memory_context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.memory[-4:]])
        prompt = f"Previous conversation:\n{memory_context}\n\nCurrent message: {prompt}"
    
    try:
        # Call HuggingFace
        response = client.chat_completion(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        # Fallback model
        try:
            response = client.chat_completion(
                model=FALLBACK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.7
            )
            return response.choices[0].message.content
        except:
            return f"I apologize, but I'm having trouble connecting right now. Please try again in a moment."

# ================================ Classifier ================================
def classify(text: str) -> Optional[Dict]:
    """Call the classifier service"""
    
    if st.session_state.ablation_mode == "no_classifier":
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

# ================================ Main Chat Interface ================================
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔬 Thesis Controls")
        
        # Ablation controls
        st.markdown("#### 🧩 System Status")
        col1, col2 = st.columns(2)
        with col1:
            classifier_status = "active ✓" if st.session_state.ablation_mode != "no_classifier" else "disabled"
            st.markdown(f"**Classifier**<br>{classifier_status}", unsafe_allow_html=True)
        with col2:
            rag_status = "active ✓" if st.session_state.ablation_mode != "no_rag" else "disabled"
            st.markdown(f"**RAG**<br>{rag_status}", unsafe_allow_html=True)
        
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
    
    # Main chat interface
    st.title("💨 VentPal - Mental Health Support")
    
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("How are you feeling today?"):
        # Add to messages
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.memory.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Classify if enabled
        classification = None
        if st.session_state.ablation_mode not in ["no_classifier", "baseline"]:
            classification = classify(prompt)
        
        # Determine if RAG should be used
        use_rag = False
        if st.session_state.ablation_mode not in ["no_rag", "baseline"]:
            if len(st.session_state.messages) >= MIN_EXCHANGES_BEFORE_RAG * 2:
                use_rag = True
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = llm_call(prompt, use_rag=use_rag)
                st.markdown(response)
        
        # Add to messages
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.memory.append({"role": "assistant", "content": response})
        
        # Keep memory limited
        if len(st.session_state.memory) > 10:
            st.session_state.memory = st.session_state.memory[-10:]

if __name__ == "__main__":
    main()
