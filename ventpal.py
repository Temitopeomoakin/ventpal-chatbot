# ────────────────────────────────────────────────────────────────────────────────
# VentPal (Streamlit) — remote classifier service + non-repetitive LLM responses
# ────────────────────────────────────────────────────────────────────────────────

# SQLite monkey-patch for ChromaDB compatibility
import sys
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import re, time, json, hashlib, random
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
import streamlit as st
import requests

from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError

# ────────────────────────────────────────────────────────────────────────────────
# Page config & CSS
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VentPal - Mental Health Support",
    page_icon="💨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 1rem;}
    .chat-message {padding: 1rem; border-radius: 10px; margin: 0.5rem 0;}
    .user-message {background-color: #e3f2fd; border-left: 4px solid #2196f3;}
    .assistant-message {background-color: #f3e5f5; border-left: 4px solid #9c27b0;}
    .crisis-alert {background-color: #ffebee; border: 2px solid #f44336; border-radius: 10px; padding: 1rem; margin: 1rem 0;}
    .skill-badge {display: inline-block; background: #e8f5e8; color: #2e7d32; padding: 0.2rem 0.5rem; border-radius: 12px; font-size: 0.7rem; margin-top: 0.5rem;}
    .source-citation {font-size: 0.8rem; color: #666; font-style: italic; margin-top: 0.5rem;}
    .footer-notice {background:#fff8e1; border:1px solid #ffeaa7; border-radius:10px; padding:0.75rem; margin-top:1rem; color:#6b5e00; font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# Session state init
# ────────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state: st.session_state.messages = []
if "request_count" not in st.session_state: st.session_state.request_count = 0
if "last_reset" not in st.session_state: st.session_state.last_reset = time.time()
if "user_id" not in st.session_state: st.session_state.user_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
if "conversation_summary" not in st.session_state: st.session_state.conversation_summary = ""
if "user_name" not in st.session_state: st.session_state.user_name = ""
if "session_goals" not in st.session_state: st.session_state.session_goals = ""
if "session_start_time" not in st.session_state: st.session_state.session_start_time = time.time()
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, max_token_limit=2000)
if "recent_cues" not in st.session_state: st.session_state.recent_cues = []
if "used_skills" not in st.session_state: st.session_state.used_skills = []
if "last_question_stem" not in st.session_state: st.session_state.last_question_stem = ""
random.seed(st.session_state.user_id)
GDPR_COMPLIANT = True

# ────────────────────────────────────────────────────────────────────────────────
# Config via secrets
# ────────────────────────────────────────────────────────────────────────────────
HUGGINGFACE_API_KEY   = st.secrets.get("HUGGINGFACE_API_KEY", "")
MODEL_NAME            = st.secrets.get("MODEL_NAME", "meta-llama/Llama-4-Scout-17B-16E-Instruct")
FALLBACK_MODEL        = st.secrets.get("FALLBACK_MODEL", "HuggingFaceH4/zephyr-7b-beta")
HF_PROVIDER           = st.secrets.get("HF_PROVIDER", "serverless")  # avoid auto provider lookup
EMBEDDING_MODEL       = st.secrets.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VECTOR_DB_PATH        = st.secrets.get("VECTOR_DB_PATH", "vector_db")
MAX_REQUESTS_PER_HOUR = int(st.secrets.get("MAX_REQUESTS_PER_HOUR", "50"))

# Remote classifier microservice (Cloud Run)
CLASSIFIER_URL        = st.secrets.get("CLASSIFIER_URL", "").rstrip("/")  # e.g., https://ventpal-classifier-service-xyz.a.run.app
CLASSIFIER_POLICY     = st.secrets.get("CLASSIFIER_POLICY", "always").lower()  # "off" | "first_turn" | "always"
CLASSIFIER_AUTH       = st.secrets.get("CLASSIFIER_AUTH", "")  # optional Bearer token

# Severity alert settings
def _parse_list_secret(key: str, default: List[str]) -> List[str]:
    val = st.secrets.get(key, None)
    if val is None: return default
    try:
        parsed = json.loads(val) if isinstance(val, str) else list(val)
        return [str(x).lower() for x in parsed]
    except Exception:
        return default

SEVERITY_ALERT_P        = float(st.secrets.get("SEVERITY_ALERT_P", "0.60"))
SEVERITY_HIGH_LABELS    = set(_parse_list_secret("SEVERITY_HIGH_LABELS", ["red","crisis","severe","very_high","urgent"]))
SEVERITY_ALERT_CONTAINS = _parse_list_secret("SEVERITY_ALERT_CONTAINS", ["red","crisis","severe","high","urgent"])

# ────────────────────────────────────────────────────────────────────────────────
# Heuristics & safety
# ────────────────────────────────────────────────────────────────────────────────
CUE_GROUPS = {
    "start": ["I’m glad you’re here.","Thanks for telling me.","You’re not alone here.","We can take it slow."],
    "heavy_emotion": ["That sounds really heavy.","I can hear the strain.","That’s a lot to carry.","I’m sorry it hurts this much.","That must be exhausting."],
    "mild_distress": ["I’m listening.","Take your time.","Say more if you can.","I’m with you.","We can unpack this."],
    "positive_progress": ["That’s a real step.","I’m proud of you.","That’s encouraging.","Nice progress.","That took courage."],
    "user_reflects": ["That insight matters.","That makes sense.","I see your point.","Good noticing.","That’s understandable."],
    "user_stuck": ["Let’s slow it down.","One small step at a time.","We can map it out.","We’ll figure it out together.","Let’s sit with it."],
    "offering_strategy": ["A small tool might help.","Want a quick technique?","Could I share a tiny skill?","There’s a brief exercise.","We could try something simple."],
    "crisis_safety": ["Your safety comes first.","I’m concerned about you.","Let’s keep you safe.","You matter to me."],
    "neutral": ["Mmm-hmm.","I hear you.","Go on.","Okay."]
}
CBT_SKILLS = {
    "breathing":{"name":"Deep Breathing","patterns":[r"deep ?breath",r"breathing ?exercise",r"inhale.*exhale",r"box ?breath"]},
    "grounding":{"name":"5-4-3-2-1 Grounding","patterns":[r"5-?4-?3-?2-?1",r"grounding",r"five ?senses",r"sensory.*technique"]},
    "thought_challenging":{"name":"Thought Challenging","patterns":[r"thought.*challenge",r"cognitive.*restructur",r"reframe",r"alternative.*thought"]},
    "behavioral_activation":{"name":"Behavioral Activation","patterns":[r"behavioral.*activation",r"activity.*schedul",r"small.*step",r"gradual.*exposure"]},
    "mindfulness":{"name":"Mindfulness","patterns":[r"mindful",r"present.*moment",r"observe.*thought",r"non-?judgment"]},
    "progressive_relaxation":{"name":"Progressive Relaxation","patterns":[r"progressive.*relax",r"muscle.*tension",r"body.*scan",r"relaxation.*technique"]}
}
BANNED_STOCK = {
    "i’m here with you": ["You’re not on your own.","I’m with you here.","We can face this together."],
    "tell me more": ["Could you share a little more?","What else feels important?","What’s standing out most?"],
    "that sounds tough": ["That sounds really hard.","That’s heavy to carry.","That’s a lot to handle."],
}

CRISIS_PATTERNS = [
    r"kill.*myself", r"suicid(e|al)", r"want.*die", r"end.*life", r"hurt.*myself",
    r"self.*harm", r"cut.*myself", r"overdose", r"take.*pills", r"jump.*off",
    r"hang.*myself", r"shoot.*myself", r"drown.*myself", r"burn.*myself"
]

SYSTEM_PROMPT = """
You are VentPal, a warm, CBT-informed companion. You are not a clinician and never diagnose or prescribe.

Goals (in order):
1) Reduce distress this turn.
2) Build rapport.
3) Offer at most one CBT micro-skill, only if the user seems ready (ask permission in ≤7 words).

Style guardrails:
• 70–120 words; natural, conversational English; no clinical jargon.
• Start with 1 specific validation tied to the user’s words (≤15 words).
• Then gentle exploration (≤25 words).
• (Optional) One micro-skill in everyday language (≤35 words) + ask permission.
• End with exactly one open question (rotate stems; avoid repeating the last question form).
• Do NOT reuse the same empathy opener or micro-skill used in the last 3 turns.
• Avoid generic filler like “I’m here with you” unless there’s crisis.
• Never say “As an AI…”.

Safety:
If any self-harm intent or imminent risk:
• Short, compassionate acknowledgement.
• Provide crisis resources briefly.
• Ask a direct safety question (e.g., “Are you able to stay safe right now?”).
• Do not include CBT skills or RAG content in that message.
"""

SAFETY_PROMPT = "If self-harm risk appears, respond only with the safety flow described above."
CRISIS_RESOURCES_UK = ("• **Samaritans** 116 123 (24/7, free)\n"
                       "• **Shout** – text **SHOUT** to 85258 (24/7)\n"
                       "• **Emergency**: 999\n"
                       "• **NHS 111** for urgent advice")

# ────────────────────────────────────────────────────────────────────────────────
# Remote classifier client
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def classifier_headers():
    h = {"Content-Type": "application/json"}
    if CLASSIFIER_AUTH:
        h["Authorization"] = f"Bearer {CLASSIFIER_AUTH}"
    return h

@st.cache_resource(show_spinner=False)
def check_classifier_health() -> Dict:
    if not CLASSIFIER_URL:
        return {"online": False, "reason": "no_url"}
    try:
        r = requests.get(f"{CLASSIFIER_URL}/health", headers=classifier_headers(), timeout=6)
        if r.ok:
            data = r.json()
            return {"online": True, **data}
        return {"online": False, "status": r.status_code}
    except Exception as e:
        return {"online": False, "error": str(e)}

def classify_via_service(text: str) -> Optional[Dict]:
    if not CLASSIFIER_URL:
        return None
    try:
        r = requests.post(f"{CLASSIFIER_URL}/classify",
                          headers=classifier_headers(),
                          json={"text": text},
                          timeout=12)
        if r.status_code == 503:
            return None
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

# ────────────────────────────────────────────────────────────────────────────────
# Utilities & LLM generation
# ────────────────────────────────────────────────────────────────────────────────
def is_english_text(text: str) -> bool:
    if not text.strip(): return True
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return (non_ascii / max(1, len(text))) <= 0.3

def sanitize_user_input(text: str) -> str:
    if not text: return ""
    text = ''.join(ch for ch in text if ord(ch) >= 32 or ch in '\n\r\t')
    if len(text) > 1000: text = text[:1000] + "..."
    text = text.replace("```", "\\`\\`\\`")
    return text.strip()

def detect_crisis_regex(text: str) -> bool:
    if not is_english_text(text): return False
    tl = text.lower()
    return any(re.search(p, tl) for p in CRISIS_PATTERNS)

def crisis_block() -> str:
    return ("🚨 I’m really sorry you’re feeling this way—it sounds unbearably painful.\n\n"
            "If you feel you might act on these thoughts **right now**, please call **999** or go to A&E.\n\n"
            f"{CRISIS_RESOURCES_UK}\n\n"
            "Are you able to stay safe for the next few minutes?")

def severity_triggers_alert(sev_label: str, sev_conf: float) -> bool:
    name_l = (sev_label or "").lower()
    if sev_conf >= SEVERITY_ALERT_P and (name_l in SEVERITY_HIGH_LABELS or any(tok in name_l for tok in SEVERITY_ALERT_CONTAINS)):
        return True
    return False

def select_micro_cue(inferred_emotion: str, recent: List[str]) -> str:
    emotion_to_group = {"crisis":"crisis_safety","anxiety":"heavy_emotion","depression":"heavy_emotion",
                        "anger":"heavy_emotion","grief":"heavy_emotion","joy":"positive_progress","neutral":"mild_distress"}
    group = emotion_to_group.get(inferred_emotion, "mild_distress")
    pool = [c for c in CUE_GROUPS[group] if c.lower() not in {x.lower() for x in recent}]
    if not pool:
        pool = [c for g in CUE_GROUPS.values() for c in g]
    choice = random.choice(pool)
    return choice

def update_recent_cues(cue: str, recent: List[str], max_recent: int = 3):
    if not cue: return
    recent.append(cue)
    if len(recent) > max_recent: recent.pop(0)

def detect_skill_usage(response: str) -> Optional[str]:
    rl = response.lower()
    for name, info in CBT_SKILLS.items():
        for pat in info["patterns"]:
            if re.search(pat, rl): return name
    return None

def get_question_stem(s: str) -> str:
    q = s.strip().split("?")[-2] if "?" in s.strip() else ""
    if not q: return ""
    q = q.strip()
    first = q.split(" ", 1)[0].lower() if q else ""
    return first

def enforce_banned_phrase_variety(text: str) -> str:
    out = text
    lower = out.lower()
    for key, alts in BANNED_STOCK.items():
        if key in lower:
            # allow once every 3 turns: if used in last 2 assistant msgs, replace
            last_two = [m["content"].lower() for m in st.session_state.messages if m["role"]=="assistant"][-2:]
            if any(key in x for x in last_two):
                out = re.sub(re.escape(key), random.choice(alts), out, flags=re.IGNORECASE)
                lower = out.lower()
    return out

def check_rate_limit() -> bool:
    now = time.time()
    if now - st.session_state.last_reset > 3600:
        st.session_state.request_count = 0
        st.session_state.last_reset = now
    return st.session_state.request_count < MAX_REQUESTS_PER_HOUR

def increment_rate_limit(): st.session_state.request_count += 1

def get_conversation_memory() -> str:
    try:
        mem = st.session_state.memory.load_memory_variables({}).get("chat_history", [])
        if not mem: return ""
        last_n = st.session_state.get("memory_turns", 3)
        out = []
        for m in mem[-last_n*2:]:
            if isinstance(m, HumanMessage): out.append(f"User: {m.content}")
            elif isinstance(m, AIMessage): out.append(f"Assistant: {m.content}")
        return "\n".join(out)
    except Exception:
        return ""

def update_memory(user_msg: str, assistant_msg: str):
    try:
        st.session_state.memory.chat_memory.add_user_message(user_msg)
        st.session_state.memory.chat_memory.add_ai_message(assistant_msg)
    except Exception:
        pass

# ---- Hugging Face client (provider pinned, model pre-bound) -------------------
@st.cache_resource
def get_hf_client_primary() -> InferenceClient:
    return InferenceClient(
        model=MODEL_NAME,
        token=HUGGINGFACE_API_KEY if HUGGINGFACE_API_KEY else None,
        provider=HF_PROVIDER,
        timeout=120,
    )

@st.cache_resource
def get_hf_client_fallback() -> InferenceClient:
    return InferenceClient(
        model=FALLBACK_MODEL,
        token=HUGGINGFACE_API_KEY if HUGGINGFACE_API_KEY else None,
        provider="serverless",
        timeout=120,
    )

def _chat_once(client: InferenceClient, prompt: str, max_tokens: int, temperature: float) -> str:
    try:
        completion = client.chat.completions.create(
            messages=[{"role":"system","content":[{"type":"text","text":SYSTEM_PROMPT.strip()}]},
                      {"role":"user","content":[{"type":"text","text":prompt}]}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return completion.choices[0].message.content
    except TypeError:
        # some providers may not accept system in messages; retry without
        completion = client.chat.completions.create(
            messages=[{"role":"user","content":[{"type":"text","text":SYSTEM_PROMPT.strip()+"\n\n"+prompt}]}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return completion.choices[0].message.content

def build_therapist_prompt(user_msg: str, context: str, inferred_emotion: str) -> str:
    micro_cue = select_micro_cue(inferred_emotion, st.session_state.recent_cues)
    used_skills = ", ".join(st.session_state.used_skills[-3:]) if st.session_state.used_skills else "none"
    last_q = st.session_state.last_question_stem or "none"
    memory = get_conversation_memory() or 'This is the start of our conversation.'
    safe_user = f"```\n{user_msg}\n```"
    return f"""
{SAFETY_PROMPT}

Recent empathy openers: {st.session_state.recent_cues[-3:]}
Recent micro-skills: {used_skills}
Last question stem used: {last_q}
Do not reuse the same opener or skill as above.

**Conversation (last {st.session_state.get('memory_turns', 3)} exchanges):**
{memory}

**Latest user message:**
{safe_user}

**Detected emotion(s):** {inferred_emotion}
**Tiny empathy opener to consider:** {micro_cue}

**Context (CBT knowledge):**
{context}

Instructions for this turn:
• Paraphrase for variety; avoid repeating exact phrasing from prior turns.
• If content duplicates prior guidance, return only the one most actionable tip.
• Ask permission before offering a skill.
• End with exactly one open question, using a different stem than: {last_q}.
"""

def ensure_follow_up_question(response: str) -> str:
    r = response.strip()
    if not r.endswith("?"):
        # Add one gentle open question if missing
        stems = ["How does that sound to you?", "What feels most pressing right now?", "How would you like to start?", "What part hurts the most today?"]
        # avoid repeating last stem
        choices = [s for s in stems if (s.split(" ",1)[0].lower() != (st.session_state.last_question_stem or "").lower())]
        r = r.rstrip(".! ") + " " + random.choice(choices)
    return r

def polish_chunks_with_llama(raw_chunks: List[str], user_query: str) -> str:
    if not raw_chunks: return ""
    if len(raw_chunks) <= 2: return "\n\n".join(raw_chunks)
    try:
        combined = "\n\n".join(raw_chunks)
        polish_prompt = f"""You are a mental health expert summarizer.

User situation: {user_query}

Raw knowledge:
{combined}

Task: Summarize in ≤150 words, practical and warm. Paraphrase for variety; avoid repeating prior phrasing. If content duplicates, keep only the most actionable tip.

Polished:"""
        client = get_hf_client_primary()
        try:
            return _chat_once(client, polish_prompt, max_tokens=150, temperature=0.3)
        except HfHubHTTPError:
            client_fb = get_hf_client_fallback()
            return _chat_once(client_fb, polish_prompt, max_tokens=150, temperature=0.3)
    except Exception:
        return "\n\n".join(raw_chunks[:3])

@st.cache_resource(show_spinner=False)
def create_vector_store():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        if os.path.exists(VECTOR_DB_PATH):
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            return Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
        else:
            st.error(f"❌ Vector database not found at {VECTOR_DB_PATH}")
            st.stop()
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        st.stop()

def get_relevant_context(query: str, vectorstore, score_threshold: float = 0.45) -> Tuple[str, List[str]]:
    try:
        results = vectorstore.similarity_search_with_score(query, k=10)
        distance_threshold = 1.0 - score_threshold
        filtered = [(doc, s) for doc, s in results if s <= distance_threshold]
        raw_chunks, titles = [], []
        for doc, s in filtered[:5]:
            c = (doc.page_content or "").strip()
            if c:
                raw_chunks.append(c)
                titles.append(doc.metadata.get("title", "CBT Resource"))
        if not raw_chunks: return "", []
        if st.session_state.get("polish_chunks", True):
            with st.spinner("Polishing context…"):
                txt = polish_chunks_with_llama(raw_chunks, query)
        else:
            txt = "\n\n".join(raw_chunks[:3])
        return txt, titles[:3]
    except Exception as e:
        st.error(f"Error retrieving context: {str(e)}")
        return "", []

def generate_response(user_text: str, context: str, temperature: float, max_tokens: int, inferred_emotion: str) -> str:
    if not HUGGINGFACE_API_KEY:
        return "I’m currently not configured to generate responses (missing API key)."
    client = get_hf_client_primary()
    full_prompt = build_therapist_prompt(user_text, context, inferred_emotion)

    try:
        reply = _chat_once(client, full_prompt, max_tokens=max_tokens, temperature=temperature)
    except HfHubHTTPError as e:
        msg = str(e).lower()
        if any(code in msg for code in ["401", "403", "429", "too many requests", "forbidden", "unauthorized"]):
            try:
                client_fb = get_hf_client_fallback()
                reply = _chat_once(client_fb, full_prompt, max_tokens=max_tokens, temperature=temperature)
            except Exception:
                reply = "That sounds important. Would you like to share a bit more so I can understand better?"
        else:
            reply = "I’m here and listening. What part feels most pressing right now?"
    except Exception:
        reply = "I’m here and listening. What part feels most pressing right now?"

    reply = enforce_banned_phrase_variety(reply)
    reply = ensure_follow_up_question(reply)
    # update last question stem
    st.session_state.last_question_stem = get_question_stem(reply)
    return reply

# ────────────────────────────────────────────────────────────────────────────────
# App UI
# ────────────────────────────────────────────────────────────────────────────────
def footer_notice():
    st.markdown("""
<div class="footer-notice">
<strong>Heads up:</strong> VentPal offers general CBT-informed support and isn’t a substitute for professional care.
If you’re in crisis, call 999 (UK) or contact Samaritans 116 123 / Shout 85258.
</div>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown(
        """<div class="main-header">
            <h1>💨 VentPal - Mental Health Support</h1>
            <p>Your empathetic CBT-informed companion</p>
        </div>""",
        unsafe_allow_html=True
    )

    # Sidebar privacy note
    if GDPR_COMPLIANT:
        st.sidebar.info("🔒 **Privacy:** Session-only; nothing is stored after you close the tab.")

    # Vector store
    with st.spinner("Loading CBT resources…"):
        vectorstore = create_vector_store()

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        if not st.session_state.user_name:
            st.session_state.user_name = st.text_input("What should I call you?", placeholder="Your name")
        if st.session_state.user_name:
            st.success(f"Hello, {st.session_state.user_name}! 💨")

        if not st.session_state.session_goals:
            st.session_state.session_goals = st.text_area(
                "What would you like to work on today?",
                placeholder="e.g., managing anxiety, coping with stress",
                height=100
            )

        st.subheader("🔧 Configuration")
        temperature = st.slider("Creativity", 0.1, 1.0, 0.7, 0.1)
        max_tokens = st.slider("Response length", 50, 200, 150, 10)
        memory_turns = st.slider("Conversation memory", 1, 5, 3, 1)
        score_threshold = st.slider("Content relevance", 0.1, 0.9, 0.45, 0.05)
        polish_chunks = st.checkbox("Polish Knowledge Chunks", value=True)
        st.session_state.memory_turns = memory_turns
        st.session_state.polish_chunks = polish_chunks

        st.subheader("🧠 Classifier Service")
        health = check_classifier_health()
        if CLASSIFIER_POLICY == "off":
            st.warning("Classifier disabled by policy.")
        elif not CLASSIFIER_URL:
            st.error("CLASSIFIER_URL is not set in secrets.")
        else:
            if health.get("online"):
                st.success(f"Online • vocab={health.get('vocab_size','?')} • positions={health.get('max_position_embeddings','?')}")
            else:
                st.error("Offline / unreachable")
                if "error" in health: st.caption(health["error"])

        st.subheader("📊 Usage")
        st.metric("Requests This Hour", st.session_state.request_count)
        if st.button("Reset Rate Limit"):
            st.session_state.request_count = 0
            st.session_state.last_reset = time.time()
            st.success("Rate limit reset!")

    # Ask name first
    if not st.session_state.user_name:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown("**Hello!** I'm **VentPal**. What should I call you?")
        name_input = st.text_input("Your name:", key="name_input")
        if name_input:
            st.session_state.user_name = sanitize_user_input(name_input)
            st.rerun()
        st.markdown("---")
        footer_notice()
        return

    # Show chat history
    conversation = st.session_state.messages
    for m in conversation:
        with st.chat_message(m["role"], avatar=("🤖" if m["role"]=="assistant" else "👤")):
            st.markdown(m["content"])
            if m["role"]=="assistant" and "skill_used" in m and m["skill_used"]:
                st.markdown(f'<div class="skill-badge">💡 {CBT_SKILLS[m["skill_used"]]["name"]}</div>', unsafe_allow_html=True)
            if m["role"]=="assistant" and "sources" in m and m["sources"]:
                st.markdown(f'<div class="source-citation">Sources: {", ".join(m["sources"])}</div>', unsafe_allow_html=True)

    # Input
    if prompt := st.chat_input(f"How are you feeling today, {st.session_state.user_name}?"):
        if not check_rate_limit():
            st.error("You've reached the hourly rate limit. Please wait before making more requests.")
            st.stop()

        user_text = sanitize_user_input(prompt)
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_text)

        # 1) Classifier (remote) + crisis gate
        clf = None
        if CLASSIFIER_POLICY != "off" and CLASSIFIER_URL and check_classifier_health().get("online"):
            clf = classify_via_service(user_text)

        inferred_emotion = "neutral"
        if clf is not None and "emotion" in clf:
            top_em = clf["emotion"]["top"]
            inferred_emotion = top_em["label"] if top_em.get("conf", 0) >= 0.5 else (clf["emotion"]["pred_labels"][0] if clf["emotion"]["pred_labels"] else top_em["label"])
            sev_label = clf["severity"]["top"]["label"]
            sev_conf  = clf["severity"]["top"]["conf"]
            if severity_triggers_alert(sev_label, sev_conf):
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(f"""
<div class="crisis-alert">
<strong>⚠️ I’m concerned about your safety.</strong><br>
This looks high-risk (severity: <em>{sev_label}</em>, confidence {sev_conf:.2f}).<br><br>
{crisis_block()}
</div>
""", unsafe_allow_html=True)
                conversation += [
                    {"role":"user","content":user_text,"timestamp":datetime.now().isoformat()},
                    {"role":"assistant","content":crisis_block(), "crisis_alert":True,"timestamp":datetime.now().isoformat()},
                ]
                update_memory(user_text, crisis_block())
                increment_rate_limit()
                st.session_state.messages = conversation
                with st.sidebar:
                    st.subheader("🚨 Severity Alert")
                    st.write(f"**Severity:** {sev_label} ({sev_conf:.2f})")
                    st.caption("Generation skipped to prioritize safety.")
                st.markdown("---"); footer_notice()
                return

        # Fallback regex crisis check if classifier is down
        if clf is None and detect_crisis_regex(user_text):
            crisis_response = crisis_block()
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(crisis_response)
            conversation += [
                {"role":"user","content":user_text,"timestamp":datetime.now().isoformat()},
                {"role":"assistant","content":crisis_response,"crisis_alert":True,"timestamp":datetime.now().isoformat()},
            ]
            update_memory(user_text, crisis_response)
            increment_rate_limit()
            st.session_state.messages = conversation
            st.markdown("---"); footer_notice()
            return

        # 2) RAG retrieval (only if no crisis/alert)
        context, source_titles = get_relevant_context(user_text, vectorstore, score_threshold)

        # 3) LLM
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("VentPal is thinking…"):
                reply = generate_response(user_text, context, temperature, max_tokens, inferred_emotion)
                st.markdown(reply)

                skill_used = detect_skill_usage(reply)
                if skill_used:
                    # avoid repeating same skill within 3 turns
                    if skill_used in st.session_state.used_skills[-3:]:
                        pass  # just don’t badge again if it was just used
                    else:
                        st.session_state.used_skills.append(skill_used)
                        st.markdown(f'<div class="skill-badge">💡 {CBT_SKILLS[skill_used]["name"]}</div>', unsafe_allow_html=True)

                # Track empathy opener variety (best-effort by scanning a few)
                chosen_cue = None
                for cues in CUE_GROUPS.values():
                    for c in cues:
                        if c.lower() in reply.lower():
                            chosen_cue = c
                            break
                    if chosen_cue: break
                if chosen_cue:
                    update_recent_cues(chosen_cue, st.session_state.recent_cues)

                # Save conversation
                conversation += [
                    {"role":"user","content":user_text,"timestamp":datetime.now().isoformat()},
                    {"role":"assistant","content":reply,"skill_used":skill_used,"sources": (source_titles if context else []),"timestamp":datetime.now().isoformat()},
                ]
                update_memory(user_text, reply)

        # Sidebar: show classifier snapshot for this turn (if available)
        if clf is not None:
            with st.sidebar:
                st.subheader("🧠 Classifier (this message)")
                em_top = clf["emotion"]["top"]
                st.markdown("**Emotion (multi-label)**")
                st.write(f"Top: **{em_top['label']}** ({em_top['conf']:.2f})")
                em_probs = clf["emotion"]["probs"]; em_th = clf["emotion"]["thresholds"]
                for name, p in sorted(em_probs.items(), key=lambda kv: kv[1], reverse=True)[:5]:
                    th = em_th.get(name, 0.5)
                    st.progress(min(1.0, p), text=f"{name}  p={p:.2f}  (th={th:.2f})")
                if clf["emotion"]["pred_labels"]:
                    st.caption("Predicted labels (p≥th): " + ", ".join(clf["emotion"]["pred_labels"]))
                else:
                    st.caption("Predicted labels: (none ≥ threshold)")
                st.markdown("**Severity**")
                st.write(f"{clf['severity']['top']['label']} ({clf['severity']['top']['conf']:.2f})")
                st.markdown("**Topic**")
                st.write(f"{clf['topic']['top']['label']} ({clf['topic']['top']['conf']:.2f})")
                st.markdown("**Intent**")
                st.write(f"{clf['intent']['top']['label']} ({clf['intent']['top']['conf']:.2f})")

        increment_rate_limit()
        st.session_state.messages = conversation

    # Footer
    st.markdown("---")
    footer_notice()

if __name__ == "__main__":
    main()
