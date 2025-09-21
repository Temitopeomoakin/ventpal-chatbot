# ventpal.py — VentPal with Natural Therapy Flow + High-Quality RAG with Reranking
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

# EXACT same imports as your working code
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from huggingface_hub import InferenceClient

# Import for reranking
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

# Severity settings
SEVERITY_ALERT_P = float(st.secrets["SEVERITY_ALERT_P"])
SEVERITY_HIGH_LABELS = st.secrets["SEVERITY_HIGH_LABELS"]
SEVERITY_ALERT_CONTAINS = st.secrets["SEVERITY_ALERT_CONTAINS"]

# RAG settings
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

# ================================ CSS Styling ================================
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; text-align: center; margin-bottom: 1rem; }
    .chat-message { padding: 1rem; margin: 0.5rem 0; border-radius: 0.5rem; }
    .user-message { background-color: #e3f2fd; }
    .assistant-message { background-color: #f3e5f5; }
    .debug-info { font-size: 0.8rem; background: #f0f2f6; padding: 0.5rem; border-radius: 0.3rem; margin: 0.5rem 0; }
    .source-info { font-size: 0.75rem; color: #666; margin-top: 0.5rem; }
    .alert-box { background: #fff3cd; border: 1px solid #ffeaa7; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; }
    .success-box { background: #d4edda; border: 1px solid #c3e6cb; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ================================ Vector Store ================================
@st.cache_resource(show_spinner=False)
def create_vector_store() -> Optional[Chroma]:
    """Create vector store"""
    
    with st.sidebar:
        st.markdown(f"<div class='debug-info'>Vector DB: {VECTOR_DB_PATH}<br>Collection: {COLLECTION_NAME}<br>Embedding: {EMBEDDING_MODEL}<br>Rerank: {ENABLE_RERANK}</div>", unsafe_allow_html=True)
    
    if not os.path.exists(VECTOR_DB_PATH):
        st.sidebar.error(f"❌ Vector database not found at: {VECTOR_DB_PATH}")
        return None
    
    try:
        with st.sidebar:
            st.info("🔄 Loading embedding model...")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        with st.sidebar:
            st.info("🔄 Connecting to vector database...")
        
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        
        # Test the connection
        test_results = vectorstore.similarity_search("test", k=1)
        
        with st.sidebar:
            st.success("✅ Vector store loaded successfully!")
            st.info(f"📄 Test query returned {len(test_results)} results")
        
        return vectorstore
        
    except Exception as e:
        with st.sidebar:
            st.error(f"❌ Vector store error: {str(e)}")
        return None

# ================================ Reranking ================================
@st.cache_resource(show_spinner=False)
def load_reranker():
    """Load cross-encoder reranking model"""
    if not ENABLE_RERANK or CrossEncoder is None:
        return None
    
    try:
        with st.sidebar:
            st.info("🔄 Loading reranker model...")
        
        reranker = CrossEncoder(RERANK_MODEL_NAME)
        
        with st.sidebar:
            st.success("✅ Reranker loaded successfully!")
        
        return reranker
        
    except Exception as e:
        with st.sidebar:
            st.warning(f"⚠️ Reranker failed to load: {str(e)}")
        return None

def rerank_documents(query: str, documents: List, reranker) -> List:
    """Rerank documents using cross-encoder"""
    if not reranker or not documents:
        return documents
    
    try:
        # Prepare query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Get relevance scores
        scores = reranker.predict(pairs)
        
        # Sort documents by relevance score (descending)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k documents
        return [doc for doc, score in scored_docs[:RERANK_TOP_K]]
        
    except Exception as e:
        st.error(f"Reranking error: {str(e)}")
        return documents[:RERANK_TOP_K]

# ================================ RAG Functions ================================
def get_relevant_context(query: str, vectorstore: Chroma, reranker=None, use_rag: bool = True) -> Tuple[str, List[str]]:
    """Get relevant context from vector store with optional reranking"""
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
    
    try:
        # Retrieve more candidates for reranking
        k = RERANK_CANDIDATES if ENABLE_RERANK else RERANK_TOP_K
        relevant_docs = vectorstore.similarity_search(query, k=k)
        
        if not relevant_docs:
            st.session_state.retrieval_history.append({
                "timestamp": datetime.now().isoformat(),
                "query": query[:100],
                "docs_retrieved": 0,
                "docs_used": 0
            })
            return "", []
        
        # Rerank if enabled
        if ENABLE_RERANK and reranker:
            relevant_docs = rerank_documents(query, relevant_docs, reranker)
        else:
            relevant_docs = relevant_docs[:RERANK_TOP_K]
        
        chunks = []
        titles = []
        
        for doc in relevant_docs:
            content = doc.page_content.strip()
            if content and len(content) > 50:
                chunks.append(content[:1200] + ("…" if len(content) > 1200 else ""))
                metadata = doc.metadata
                source = metadata.get('source', 'Unknown')
                titles.append(source)
        
        context = "\n\n".join(chunks) if chunks else ""
        
        # Log retrieval
        st.session_state.retrieval_history.append({
            "timestamp": datetime.now().isoformat(),
            "query": query[:100],
            "docs_retrieved": k,
            "docs_used": len(chunks),
            "sources": titles[:3],
            "reranked": ENABLE_RERANK and reranker is not None
        })
        
        return context, titles
        
    except Exception as e:
        st.error(f"RAG Error: {str(e)}")
        st.session_state.retrieval_history.append({
            "timestamp": datetime.now().isoformat(),
            "query": query[:100],
            "docs_retrieved": 0,
            "docs_used": 0,
            "error": str(e)
        })
        return "", []

# ================================ LLM Functions ================================
def call_llm(prompt: str, model: str = None) -> str:
    """Call LLM with rate limiting and error handling"""
    if st.session_state.request_count >= MAX_REQUESTS_PER_HOUR:
        return "I've reached my hourly limit. Please try again later."
    
    try:
        client = InferenceClient(api_key=HUGGINGFACE_API_KEY)
        
        completion = client.chat.completions.create(
            model=model or MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7
        )
        
        st.session_state.request_count += 1
        return completion.choices[0].message.content.strip()
        
    except Exception as e:
        if model != FALLBACK_MODEL:
            return call_llm(prompt, FALLBACK_MODEL)
        return f"I'm having technical difficulties. Please try again."

# ================================ Classifier ================================
def classify(text: str, use_classifier: bool = True) -> Optional[Dict]:
    """Classify text for emotion, severity, topic, intent"""
    if not use_classifier or st.session_state.ablation_mode in ["no_classifier", "baseline"]:
        st.session_state.classifier_history.append({
            "timestamp": datetime.now().isoformat(),
            "ablation_mode": st.session_state.ablation_mode,
            "text": text[:100],
            "_ablation": True
        })
        return None
    
    try:
        response = requests.post(
            f"{CLASSIFIER_URL}/classify",
            json={"text": text},
            headers={"Authorization": f"Bearer {CLASSIFIER_AUTH}"} if CLASSIFIER_AUTH else {},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            st.session_state.classifier_history.append({
                "timestamp": datetime.now().isoformat(),
                "text": text[:100],
                "emotion": result.get("emotion"),
                "severity": result.get("severity"),
                "topic": result.get("topic"),
                "intent": result.get("intent")
            })
            return result
        
    except Exception as e:
        st.session_state.classifier_history.append({
            "timestamp": datetime.now().isoformat(),
            "text": text[:100],
            "error": str(e)
        })
    
    return None

# ================================ Conversation Flow ================================
def determine_conversation_stage(user_input: str, classification: Optional[Dict] = None) -> str:
    """Determine conversation stage based on input and classification"""
    user_lower = user_input.lower().strip()
    
    # Check for consent responses
    if st.session_state.awaiting_consent:
        if any(word in user_lower for word in ["yes", "sure", "okay", "ok", "please", "help"]):
            st.session_state.awaiting_consent = False
            return "support"
        elif any(word in user_lower for word in ["no", "not now", "maybe later"]):
            st.session_state.awaiting_consent = False
            return "exploration"
    
    # Initial greeting
    if st.session_state.conversation_stage == "greeting":
        return "exploration"
    
    # Check if enough exchanges for RAG
    if len(st.session_state.messages) >= MIN_EXCHANGES_BEFORE_RAG * 2:
        if not st.session_state.awaiting_consent and st.session_state.conversation_stage != "support":
            return "consent"
    
    return st.session_state.conversation_stage

def generate_response(user_input: str, context: str, sources: List[str], classification: Optional[Dict] = None) -> str:
    """Generate contextual response based on conversation stage"""
    
    stage = determine_conversation_stage(user_input, classification)
    st.session_state.conversation_stage = stage
    
    if stage == "exploration":
        prompt = f"""You are VentPal, a compassionate mental health support chatbot. 

The user said: "{user_input}"

Respond with empathy and ask gentle follow-up questions to understand their situation better. Keep it conversational and supportive. Don't offer techniques yet - just listen and explore."""

    elif stage == "consent":
        st.session_state.awaiting_consent = True
        prompt = f"""You are VentPal, a mental health support chatbot.

The user said: "{user_input}"

You've been talking for a bit and want to offer helpful techniques. Acknowledge their feelings and ask if they'd like you to share some evidence-based techniques that might help. Be warm and non-pressuring."""

    else:  # support stage
        if context:
            prompt = f"""You are VentPal, a mental health support chatbot trained on CBT resources.

The user said: "{user_input}"

Based on these CBT resources:
{context}

Provide a helpful, specific response using the CBT techniques and information provided. Be supportive and practical. Reference specific techniques from the resources."""
        else:
            prompt = f"""You are VentPal, a compassionate mental health support chatbot.

The user said: "{user_input}"

Provide supportive guidance using general mental health best practices. Be empathetic and helpful."""
    
    return call_llm(prompt)

# ================================ Main Processing ================================
def process_user_input(user_text: str, vectorstore: Optional[Chroma], reranker=None):
    """Process user input and generate response"""
    
    # Classify input
    use_classifier = st.session_state.ablation_mode not in ["no_classifier", "baseline"]
    classification = classify(user_text, use_classifier)
    
    # Get context for support stage
    use_rag = (st.session_state.ablation_mode not in ["no_rag", "baseline"] and 
               st.session_state.conversation_stage == "support" and 
               vectorstore is not None)
    
    context, sources = get_relevant_context(user_text, vectorstore, reranker, use_rag) if vectorstore else ("", [])
    
    # Generate response
    response = generate_response(user_text, context, sources, classification)
    
    # Add to memory
    st.session_state.memory.chat_memory.add_user_message(user_text)
    st.session_state.memory.chat_memory.add_ai_message(response)
    
    # Display response
    with st.chat_message("assistant"):
        st.write(response)
        
        # Show sources if available
        if sources and context:
            with st.expander("📚 Sources"):
                for source in sources[:3]:
                    st.write(f"• {source}")

# ================================ Sidebar ================================
def render_sidebar():
    """Render sidebar with controls and debug info"""
    st.sidebar.title("🔬 Thesis Controls")
    
    # System Status
    st.sidebar.subheader("🧩 System Status")
    classifier_status = "active ✓" if st.session_state.ablation_mode not in ["no_classifier", "baseline"] else "disabled"
    rag_status = "active ✓" if st.session_state.ablation_mode not in ["no_rag", "baseline"] else "disabled"
    
    st.sidebar.write(f"**Classifier:** {classifier_status}")
    st.sidebar.write(f"**RAG:** {rag_status}")
    
    # Ablation Study
    st.sidebar.subheader("Ablation Study")
    ablation_options = {
        "full_system": "Full System (All Components)",
        "no_classifier": "No Classifier (RAG Only)",
        "no_rag": "No RAG (Classifier Only)", 
        "baseline": "Baseline (No AI Components)"
    }
    
    current_mode = st.sidebar.selectbox(
        "System Configuration:",
        options=list(ablation_options.keys()),
        format_func=lambda x: ablation_options[x],
        index=list(ablation_options.keys()).index(st.session_state.ablation_mode)
    )
    
    if current_mode != st.session_state.ablation_mode:
        st.session_state.ablation_mode = current_mode
        st.rerun()
    
    st.sidebar.write(f"**Active:** {ablation_options[st.session_state.ablation_mode]}")
    
    # Debug info
    if st.sidebar.checkbox("Show Debug Info"):
        st.sidebar.subheader("Debug Information")
        st.sidebar.write(f"**Stage:** {st.session_state.conversation_stage}")
        st.sidebar.write(f"**Messages:** {len(st.session_state.messages)}")
        st.sidebar.write(f"**Requests:** {st.session_state.request_count}/{MAX_REQUESTS_PER_HOUR}")
        
        if st.session_state.retrieval_history:
            last_retrieval = st.session_state.retrieval_history[-1]
            st.sidebar.write(f"**Last RAG:** {last_retrieval.get('docs_used', 0)} docs")
            if last_retrieval.get('reranked'):
                st.sidebar.write("**Reranked:** ✓")
        
        if st.session_state.classifier_history:
            last_classification = st.session_state.classifier_history[-1]
            if 'emotion' in last_classification:
                st.sidebar.write(f"**Last Emotion:** {last_classification.get('emotion', 'N/A')}")

# ================================ Main App ================================
def main():
    """Main application"""
    
    # Rate limiting reset
    if time.time() - st.session_state.last_reset > 3600:
        st.session_state.request_count = 0
        st.session_state.last_reset = time.time()
    
    # Render sidebar
    render_sidebar()
    
    # Initialize vector store and reranker
    vectorstore = create_vector_store()
    reranker = load_reranker() if ENABLE_RERANK else None
    
    # Main chat interface
    st.title("💨 VentPal - Mental Health Support Chatbot")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if user_input := st.chat_input("Share what's on your mind..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        # Process and respond
        process_user_input(user_input, vectorstore, reranker)
        
        # Add assistant message to history
        if st.session_state.memory.chat_memory.messages:
            last_ai_message = st.session_state.memory.chat_memory.messages[-1]
            if hasattr(last_ai_message, 'content'):
                st.session_state.messages.append({"role": "assistant", "content": last_ai_message.content})

if __name__ == "__main__":
    main()
