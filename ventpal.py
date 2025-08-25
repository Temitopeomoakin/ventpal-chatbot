# ventpal.py — VentPal (Streamlit) with permissioned RAG + classifier-first pipeline
# - Classifier: Cloud Run /classify (Emotion/Topic/Intent/Severity) + regex safety override
# - RAG: Chroma persistent store (opens your COLLECTION_NAME) + robust retrieval (rank-first, no brittle threshold)
# - LLM: HF Inference (Scout primary → fallback), minimal repetition, exactly one open question
# - Flow: rapport → (optionally) micro-permission → RAG-backed skill (on “yes”)
# - Sidebar: shows classifier labels/confidence; RAG probe; status chips
# - Disclaimers: clear non-clinical notice (CBT/DBT/Journaling only)
# - FIX: removed caching around probe_vector_db (previous error: UnhashableParamError)

# ─────────────────────────── Boot / Env ───────────────────────────
import sys, os, time, json, re, hashlib, random
from typing import List, Tuple, Dict, Optional
from datetime import datetime

# Chroma needs pysqlite3 on some hosts
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import requests

from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError

# Seed + session
if "messages" not in st.session_state: st.session_state.messages = []
if "request_count" not in st.session_state: st.session_state.request_count = 0
if "last_reset" not in st.session_state: st.session_state.last_reset = time.time()
if "user_id" not in st.session_state: st.session_state.user_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, max_token_limit=2000)

# turn-level state for permissioned RAG + anti-repetition
if "awaiting_skill_permission" not in st.session_state: st.session_state.awaiting_skill_permission = False
if "pending_rag_query" not in st.session_state: st.session_state.pending_rag_query = ""
if "pending_topic" not in st.session_state: st.session_state.pending_topic = ""
if "last_question_stem" not in st.session_state: st.session_state.last_question_stem = ""
if "last_empathy" not in st.session_state: st.session_state.last_empathy = ""
if "last_assistant_text" not in st.session_state: st.session_state.last_assistant_text = ""
random.seed(st.session_state.user_id)
GDPR_COMPLIANT = True

# ─────────────────────────── Config / Secrets ─────────────────────
HUGGINGFACE_API_KEY   = st.secrets.get("HUGGINGFACE_API_KEY", "")
MODEL_NAME            = st.secrets.get("MODEL_NAME", "meta-llama/Llama-4-Scout-17B-16E-Instruct")
FALLBACK_MODEL        = st.secrets.get("FALLBACK_MODEL", "HuggingFaceH4/zephyr-7b-beta")
HF_PROVIDER           = st.secrets.get("HF_PROVIDER", "serverless")
EMBEDDING_MODEL       = st.secrets.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VECTOR_DB_PATH        = st.secrets.get("VECTOR_DB_PATH", "vector_db")
COLLECTION_NAME       = st.secrets.get("COLLECTION_NAME", "")  # your ingestion used "my_cbt_docs"
MAX_REQUESTS_PER_HOUR = int(st.secrets.get("MAX_REQUESTS_PER_HOUR", 50))
MIN_EXCHANGES_BEFORE_RAG = int(st.secrets.get("MIN_EXCHANGES_BEFORE_RAG", 2))  # assistant replies count

CLASSIFIER_URL        = st.secrets.get("CLASSIFIER_URL", "").rstrip("/")
CLASSIFIER_AUTH       = st.secrets.get("CLASSIFIER_AUTH", "")
CLASSIFIER_POLICY     = (st.secrets.get("CLASSIFIER_POLICY", "always") or "always").lower()

def _get_list_secret(key: str, default: List[str]) -> List[str]:
    val = st.secrets.get(key, None)
    if val is None: return [str(x).lower() for x in default]
    if isinstance(val, (list, tuple, set)): return [str(x).lower() for x in val]
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("["):
            try: return [str(x).lower() for x in json.loads(s)]
            except Exception: pass
        return [t.strip().lower() for t in s.split(",") if t.strip()]
    return [str(x).lower() for x in default]

SEVERITY_ALERT_P        = float(st.secrets.get("SEVERITY_ALERT_P", 0.60))
SEVERITY_HIGH_LABELS    = set(_get_list_secret("SEVERITY_HIGH_LABELS", ["red","crisis","severe","very_high","urgent"]))
SEVERITY_ALERT_CONTAINS = _get_list_secret("SEVERITY_ALERT_CONTAINS", ["red","crisis","severe","high","urgent"])

# Give HF a token to reduce 429s
if HUGGINGFACE_API_KEY:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_API_KEY
    os.environ["HUGGINGFACE_HUB_TOKEN"]   = HUGGINGFACE_API_KEY
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

# ─────────────────────────── UI / CSS ────────────────────────────
st.set_page_config(page_title="VentPal", page_icon="💨", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.main-header{background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);padding:1rem;border-radius:12px;color:#fff;text-align:center;margin-bottom:1rem}
.chat-message{padding:1rem;border-radius:12px;margin:.5rem 0}
.user-message{background:#e3f2fd;border-left:4px solid #2196f3}
.assistant-message{background:#f3e5f5;border-left:4px solid #9c27b0}
.crisis-alert{background:#ffebee;border:2px solid #f44336;border-radius:12px;padding:1rem;margin:1rem 0}
.skill-badge{display:inline-block;background:#e8f5e8;color:#2e7d32;padding:.2rem .5rem;border-radius:12px;font-size:.75rem;margin-top:.5rem}
.footer-notice{background:#fff8e1;border:1px solid #ffeaa7;border-radius:12px;padding:.9rem;margin-top:1rem;color:#6b5e00;font-size:.92rem}
.status-chip{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;margin-left:6px}
.ok{background:#e8f5e9;color:#1b5e20;border:1px solid #a5d6a7}
.warn{background:#fff8e1;color:#6b5e00;border:1px solid #ffe082}
.err{background:#ffebee;color:#b71c1c;border:1px solid #ef9a9a}
.small-note{font-size:12px;color:#666;margin-top:.25rem}
</style>
""", unsafe_allow_html=True)

DISCLAIMER = """
**Important:** VentPal offers supportive conversation using CBT, DBT and journaling techniques.
It isn’t medical care and I’m **not a therapist**. I can’t diagnose, treat, or prescribe.
If you’re in crisis or might act on thoughts of harming yourself or others, call **999** (UK),
or contact **Samaritans (116 123)** or **Shout (text SHOUT to 85258)** right away.
"""

def footer_notice():
    st.markdown(f"<div class='footer-notice'>{DISCLAIMER}</div>", unsafe_allow_html=True)

def chip(text: str, kind: str = "ok"):
    cls = {"ok":"ok", "warn":"warn", "err":"err"}.get(kind,"ok")
    st.markdown(f'<span class="status-chip {cls}">{text}</span>', unsafe_allow_html=True)

# ─────────────────────────── Safety / Crisis ─────────────────────
SUICIDE_PATTERNS = [
    r"\bsuicid(?:e|al)\b", r"\bi\s*want\s*to\s*die\b", r"\bend\s*my\s*life\b", r"\bhurt\s*myself\b",
    r"\bself[-\s]?harm\b", r"\boverdose\b", r"\bkill\s*myself\b", r"\bjump\s*off\b", r"\bhang\s*myself\b",
    r"\bdrown\s*myself\b", r"\bburn\s*myself\b", r"\bunalive\b", r"\bkys\b", r"\bkms\b", r"\b(end|ending)\s+it\b"
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
    return ("🚨 I’m really sorry you’re feeling like this—it sounds unbearably painful.\n\n"
            "If you feel you might act on these thoughts **right now**, please call **999** or go to A&E.\n\n"
            f"{CRISIS_RESOURCES_UK}\n\n"
            "I’m here with you. Are you able to stay safe for the next few minutes?")

def severity_triggers_alert(label: str, conf: float) -> bool:
    name = (label or "").lower()
    return (conf >= SEVERITY_ALERT_P) and (name in SEVERITY_HIGH_LABELS or any(t in name for t in SEVERITY_ALERT_CONTAINS))

# ─────────────────────────── Classifier (Cloud Run) ───────────────────────────
def _cls_headers():
    h = {"Content-Type": "application/json"}
    if CLASSIFIER_AUTH: h["Authorization"] = f"Bearer {CLASSIFIER_AUTH}"
    return h

def classifier_health() -> Optional[Dict]:
    if not CLASSIFIER_URL: return None
    try:
        r = requests.get(f"{CLASSIFIER_URL}/health", headers=_cls_headers(), timeout=6)
        r.raise_for_status(); return r.json()
    except Exception:
        return None

def classify(text: str) -> Optional[Dict]:
    if not CLASSIFIER_URL: return None
    try:
        t0 = time.time()
        r = requests.post(f"{CLASSIFIER_URL}/classify", headers=_cls_headers(), json={"text": text}, timeout=12)
        r.raise_for_status()
        out = r.json(); out["_latency_ms"] = int((time.time()-t0)*1000)
        return out
    except Exception as e:
        st.sidebar.error(f"Classifier error: {e}"); return None

def _top(head: Optional[Dict], default_label="unknown", default_conf: float = 0.0) -> Tuple[str, float]:
    top = ((head or {}).get("top") or {})
    return str(top.get("label", default_label)).lower(), float(top.get("conf", default_conf))

# ─────────────────────────── RAG (Chroma) ─────────────────────────────────────
@st.cache_resource(show_spinner=False)
def create_vector_store():
    """Open your persisted Chroma collection; tolerant to HF 429 via token/backoff."""
    from langchain_huggingface import HuggingFaceEmbeddings

    if not os.path.exists(VECTOR_DB_PATH):
        st.error(f"❌ Vector DB not found at `{VECTOR_DB_PATH}`. Ensure the folder is present in your repo.")
        st.stop()

    last_err = None
    for delay in (0.4, 1.2, 2.5):
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            kwargs = dict(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
            if COLLECTION_NAME: kwargs["collection_name"] = COLLECTION_NAME
            return Chroma(**kwargs)
        except Exception as e:
            last_err = e; time.sleep(delay)

    st.error(f"Failed to load embeddings or open Chroma. (hint: add HF token, check COLLECTION_NAME)\n\nDetails: {last_err}")
    st.stop()

# IMPORTANT: DO NOT CACHE THIS (was causing UnhashableParamError due to Chroma arg)
def probe_vector_db(vs: Chroma) -> Dict:
    """Quick rank-based probe. We DO NOT hard-filter by distance (scores can be >1.0)."""
    try:
        probes = ["panic attack", "deep breathing", "journaling prompt", "grounding", "thought challenging"]
        hits = 0; titles = set()
        for q in probes:
            res = vs.similarity_search(q, k=2)
            hits += len(res)
            for d in res:
                md = d.metadata or {}
                titles.add(md.get("title") or md.get("source") or "CBT/DBT/Journaling")
        return {"any": hits>0, "titles": list(titles)[:5], "count": hits}
    except Exception:
        return {"any": False, "titles": [], "count": 0}

def get_relevant_context(query: str, vectorstore: Chroma, max_chunks: int = 3) -> Tuple[str, List[str]]:
    """
    Rank-first selection (no brittle threshold).
    Keep top-k results (k=8) and take the first ≤ max_chunks texts.
    """
    try:
        results = vectorstore.similarity_search_with_score(query, k=8)
    except TypeError:
        docs = vectorstore.similarity_search(query, k=8)
        results = [(d, 0.0) for d in docs]

    chunks, titles = [], []
    for doc, _score in results:
        if len(chunks) >= max_chunks: break
        text = (doc.page_content or "").strip()
        if not text: continue
        chunks.append(text[:1200] + ("…" if len(text) > 1200 else ""))
        md = doc.metadata or {}
        titles.append(md.get("title") or md.get("source") or md.get("chunk_id") or "CBT/DBT/Journaling")

    return ("\n\n".join(chunks) if chunks else ""), titles

# ─────────────────────────── LLM (HF) ─────────────────────────────────────────
@st.cache_resource
def hf_client_primary() -> InferenceClient:
    return InferenceClient(model=MODEL_NAME, token=HUGGINGFACE_API_KEY or None, provider=HF_PROVIDER, timeout=120)

@st.cache_resource
def hf_client_fallback() -> InferenceClient:
    return InferenceClient(model=FALLBACK_MODEL, token=HUGGINGFACE_API_KEY or None, provider="serverless", timeout=120)

def _chat_once(client: InferenceClient, messages: List[Dict], max_tokens: int, temperature: float) -> str:
    try:
        out = client.chat.completions.create(messages=messages, max_tokens=max_tokens, temperature=temperature)
        return out.choices[0].message.content
    except Exception:
        # collapse to plain generation if provider lacks chat
        sys_prompt = ""
        user_texts = []
        for m in messages:
            role = m.get("role"); parts = m.get("content", [])
            text = " ".join([p.get("text","") for p in parts if p.get("type")=="text"])
            if role == "system": sys_prompt += text + "\n\n"
            elif role == "user": user_texts.append(text)
        prompt = (sys_prompt + "\n\n".join(user_texts)).strip()
        return client.text_generation(prompt, max_new_tokens=max_tokens, temperature=temperature, do_sample=temperature>0)

SYSTEM_PROMPT = """You are VentPal, a UK-English mental health companion using CBT, DBT and journaling.
Primary goal: reduce distress and build rapport. Teach at most one small skill only if appropriate and permitted.
You are not a clinician and never diagnose, treat, or prescribe. Avoid medication advice. Use context if provided.
Avoid repeating the exact same opener or question across turns."""

STYLE_PROMPT = """Constraints:
• Warm, plain English; ≤120 words.
• Structure: Validation → Exploration → (optional) One micro-skill (≤40 words) *only if TEACH_SKILL:true* and ask permission in ≤7 words.
• Exactly one open question at the end (no more, no less). Vary the question stem from the previous turn.
• Do not repeat generic phrases like “Thanks for sharing” or “I’m here and listening” in consecutive turns."""

SAFETY_PROMPT = """If self-harm or suicidal intent: keep it short and compassionate; list UK crisis resources; ask “Are you able to stay safe right now?” Do not include skills or RAG content in that message."""

OPEN_QUESTIONS = [
    "What feels most important to explore next?",
    "Where do you notice this most in your day?",
    "What’s one tiny step that seems doable?",
    "How does that land with you?",
    "What would feel supportive right now?",
]
def ensure_single_question(text: str) -> str:
    r = (text or "").strip()
    if not r.endswith("?"):
        choices = [q for q in OPEN_QUESTIONS if q.split(" ")[0].lower() != st.session_state.last_question_stem]
        q = random.choice(choices or OPEN_QUESTIONS)
        r = r.rstrip(".! ") + ". " + q
    stem = r.split("?")[0].strip().split(" ")[0].lower() if "?" in r else ""
    if stem: st.session_state.last_question_stem = stem
    return r

def anti_repeat_transform(text: str) -> str:
    """If the model produced the same reply twice, replace with a safe alternate phrasing."""
    t = (text or "").strip()
    if not t: return t
    last = (st.session_state.last_assistant_text or "").strip()
    if last and last.lower() == t.lower():
        alternates = [
            "I can hear how heavy that feels. What would help even a little right now?",
            "That sounds really tough to carry. Where do you notice it most in your day?",
            "Thanks for trusting me with this. What’s one tiny step that could help today?",
        ]
        t = random.choice([a for a in alternates if a.lower()!=last.lower()] or alternates)
    st.session_state.last_assistant_text = t
    return t

def build_support_only_prompt(user_text: str, memory: str, empathy_hint: str) -> List[Dict]:
    system = f"{SYSTEM_PROMPT}\n{STYLE_PROMPT}\n{SAFETY_PROMPT}"
    user = f"""CONVERSATION_MEMORY:
{memory}

USER_MESSAGE:
{user_text}

TEACH_SKILL:false
EMPATHY_SEED:{empathy_hint}
"""
    return [
        {"role":"system","content":[{"type":"text","text":system}]},
        {"role":"user","content":[{"type":"text","text":user}]}
    ]

def build_skill_prompt(user_text: str, memory: str, empathy_hint: str, context: str) -> List[Dict]:
    system = f"{SYSTEM_PROMPT}\n{STYLE_PROMPT}\n{SAFETY_PROMPT}\nUse only FACTS_FROM_RAG for any technique details."
    user = f"""CONVERSATION_MEMORY:
{memory}

USER_MESSAGE:
{user_text}

FACTS_FROM_RAG:
{context}

TEACH_SKILL:true
EMPATHY_SEED:{empathy_hint}
"""
    return [
        {"role":"system","content":[{"type":"text","text":system}]},
        {"role":"user","content":[{"type":"text","text":user}]}
    ]

# ─────────────────────────── CBT cues / helpers ───────────────────────────────
CUE_GROUPS = {
    "start":   ["I’m glad you’re here.","Thanks for opening up.","We can take this step by step.","You’re not alone here."],
    "heavy":   ["That sounds really hard.","I hear how heavy this feels.","That’s a lot to carry.","I’m sorry it hurts this much."],
    "mild":    ["I’m with you.","Take your time.","Say more if you can.","I’m listening."],
    "progress":["That’s a real step.","Nice progress.","That’s encouraging.","I notice your effort."],
}
def select_empathy(emotion: str) -> str:
    group = {
        "anger":"heavy","fear":"heavy","sadness":"heavy","grief":"heavy","anxiety":"heavy",
        "joy":"progress","neutral":"mild","disgust":"heavy","surprise":"mild","anticipation":"mild","trust":"progress",
    }.get((emotion or "neutral").lower(), "mild")
    opts = [c for c in CUE_GROUPS[group] if c != st.session_state.last_empathy] or CUE_GROUPS[group]
    choice = random.choice(opts); st.session_state.last_empathy = choice; return choice

def get_conversation_memory() -> str:
    try:
        mem = st.session_state.memory.load_memory_variables({}).get("chat_history", [])
        out = []
        for m in mem[-6:]:
            if isinstance(m, HumanMessage): out.append(f"User: {m.content}")
            elif isinstance(m, AIMessage): out.append(f"Assistant: {m.content}")
        return "\n".join(out) if out else "This is the start of our conversation."
    except Exception:
        return "This is the start of our conversation."

# Safer yes/no detectors
import re as _re
_YES_TOKENS = {"yes","yeah","yep","yup","sure","ok","okay","alright","affirmative","please"}
_YES_PHRASES = {"yes please","sounds good","go ahead","please do","that would help","i'm open","im open","i am open","i'd like that","id like that","share it","tell me","please continue"}
_NO_TOKENS  = {"no","nah","nope"}
_NO_PHRASES = {"not now","not really","rather not","maybe later","not today","not yet","skip","don’t","dont","do not","i’d rather not","id rather not","pass"}
def _tokens(t: str) -> set:
    return set(_re.findall(r"[a-z']+", (t or "").lower()))
def is_affirmative(text: str) -> bool:
    t = (text or "").lower().strip()
    toks = _tokens(t)
    if _YES_TOKENS & toks: return True
    return any(p in t for p in _YES_PHRASES)
def is_negative(text: str) -> bool:
    t = (text or "").lower().strip()
    toks = _tokens(t)
    if _NO_TOKENS & toks: return True
    return any(p in t for p in _NO_PHRASES)

# ─────────────────────────── Rate limit ───────────────────────────────────────
def check_rate_limit() -> bool:
    now = time.time()
    if now - st.session_state.last_reset > 3600:
        st.session_state.request_count = 0
        st.session_state.last_reset = now
    return st.session_state.request_count < MAX_REQUESTS_PER_HOUR
def increment_rate_limit(): st.session_state.request_count += 1

# ─────────────────────────── App ──────────────────────────────────────────────
def main():
    # Header
    st.markdown(
        """<div class="main-header">
            <h1>💨 VentPal</h1>
            <p>Gentle support with CBT, DBT, and journaling.</p>
        </div>""",
        unsafe_allow_html=True
    )

    # Required config
    if not HUGGINGFACE_API_KEY: st.error("Missing HUGGINGFACE_API_KEY in secrets."); st.stop()
    if not CLASSIFIER_URL or CLASSIFIER_POLICY == "off":
        st.error("Classifier must be enabled. Set CLASSIFIER_URL and CLASSIFIER_POLICY='always'."); st.stop()

    # Init RAG
    with st.spinner("Connecting to knowledge base…"):
        vectorstore = create_vector_store()
        probe = probe_vector_db(vectorstore)  # NOT CACHED (fixes UnhashableParamError)

    # Classifier health
    with st.spinner("Checking classifier…"):
        hc = classifier_health()
        if not hc: st.error("Classifier /health failed. Fix Cloud Run service."); st.stop()

    # Sidebar (status + controls)
    with st.sidebar:
        st.header("🧩 System status")
        st.write("Classifier"); chip("connected ✓", "ok"); st.caption(f"vocab:{hc.get('vocab_size','?')} | pos:{hc.get('max_position_embeddings','?')}")
        st.write("RAG"); chip("ready ✓" if probe["any"] else "no obvious hits", "ok" if probe["any"] else "warn")
        if probe["titles"]: st.caption("Probe: " + ", ".join(probe["titles"]))
        st.write("LLM"); chip("configured ✓", "ok")
        if GDPR_COMPLIANT: st.info("🔒 Session-only; chats aren’t stored server-side.")
        st.subheader("🔧 Settings")
        temperature = st.slider("Creativity", 0.1, 1.0, 0.7, 0.1)
        max_tokens = st.slider("Response length", 60, 220, 160, 10)
        st.subheader("📊 Usage"); st.metric("Requests this hour", st.session_state.request_count)

    # Name gate
    name = st.text_input("What should I call you?", value=st.session_state.get("user_name",""), placeholder="Your name")
    st.session_state.user_name = name.strip()
    if not st.session_state.user_name:
        st.markdown('<div class="small-note">Enter a name to start chatting.</div>', unsafe_allow_html=True)
        st.markdown("---"); footer_notice(); return

    # History
    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar=("🤖" if m["role"]=="assistant" else "👤")):
            st.markdown(m["content"])
            if m.get("skill_used"):
                st.markdown(f'<div class="skill-badge">💡 {m["skill_used"]}</div>', unsafe_allow_html=True)
            if m.get("sources"):
                st.caption("Sources: " + ", ".join(m["sources"]))

    # Input
    user_text = st.chat_input(f"How are you feeling today, {st.session_state.user_name}?")
    if not user_text:
        st.markdown("---"); footer_notice(); return
    if not check_rate_limit():
        st.error("You’ve hit the hourly limit. Try again later."); st.stop()

    with st.chat_message("user", avatar="👤"): st.markdown(user_text)

    # 1) Classify & safety
    clf = classify(user_text)
    if not clf:
        st.error("Classifier call failed; try again."); st.stop()
    emo_lbl, emo_conf = _top(clf.get("emotion"), "neutral", 0.0)
    top_lbl, top_conf = _top(clf.get("topic"), "none", 0.0)
    int_lbl, int_conf = _top(clf.get("intent"), "others", 0.0)
    sev_lbl, sev_conf = _top(clf.get("severity"), "green", 0.0)

    if detect_crisis_regex(user_text) or severity_triggers_alert(sev_lbl, sev_conf) or "suicid" in int_lbl:
        msg = crisis_block()
        with st.chat_message("assistant", avatar="🤖"): st.markdown(msg)
        st.session_state.messages += [
            {"role":"user","content":user_text,"ts":datetime.now().isoformat()},
            {"role":"assistant","content":msg,"crisis_alert":True,"ts":datetime.now().isoformat()},
        ]
        st.session_state.memory.chat_memory.add_user_message(user_text)
        st.session_state.memory.chat_memory.add_ai_message(msg)
        increment_rate_limit(); st.markdown("---"); footer_notice(); return

    # exchanges so far (assistant replies count)
    exchanges = sum(1 for m in st.session_state.messages if m["role"] == "assistant")

    # 2) Handle pending permission (from previous turn)
    if st.session_state.awaiting_skill_permission:
        if is_negative(user_text):
            empathy = select_empathy(emo_lbl if emo_conf >= 0.5 else "neutral")
            memory = get_conversation_memory()
            msgs = build_support_only_prompt(user_text, memory, empathy)
            try:
                reply = _chat_once(hf_client_primary(), msgs, max_tokens, temperature)
            except Exception:
                try: reply = _chat_once(hf_client_fallback(), msgs, max_tokens, temperature)
                except Exception: reply = "I can see this is hard. What would feel supportive right now?"
            reply = ensure_single_question(anti_repeat_transform(reply))
            with st.chat_message("assistant", avatar="🤖"): st.markdown(reply)
            st.session_state.awaiting_skill_permission = False
            st.session_state.pending_rag_query = ""; st.session_state.pending_topic = ""
            st.session_state.messages += [
                {"role":"user","content":user_text,"ts":datetime.now().isoformat()},
                {"role":"assistant","content":reply,"ts":datetime.now().isoformat()},
            ]
            st.session_state.memory.chat_memory.add_user_message(user_text)
            st.session_state.memory.chat_memory.add_ai_message(reply)
            increment_rate_limit(); st.markdown("---"); footer_notice(); return

        if is_affirmative(user_text):
            # "Yes" → run RAG now using stored query
            rag_query = st.session_state.pending_rag_query or user_text
            context, titles = get_relevant_context(rag_query, vectorstore, max_chunks=3)
            empathy = select_empathy(emo_lbl if emo_conf >= 0.5 else "neutral")
            memory = get_conversation_memory()
            msgs = build_skill_prompt(user_text, memory, empathy, context or "")
            try:
                reply = _chat_once(hf_client_primary(), msgs, max_tokens, temperature)
            except HfHubHTTPError as e:
                if any(x in str(e).lower() for x in ["401","403","429","too many requests","forbidden","unauthorized","quota"]):
                    try: reply = _chat_once(hf_client_fallback(), msgs, max_tokens, temperature)
                    except Exception: reply = "Let’s keep it simple: what would feel supportive right now?"
                else:
                    reply = "Let’s keep it simple: what would feel supportive right now?"
            except Exception:
                reply = "Let’s keep it simple: what would feel supportive right now?"

            if not context:
                reply = reply.rstrip() + "\n\n_(I couldn’t find a matching handout just now, but I can still share a tiny idea if you’d like.)_"

            reply = ensure_single_question(anti_repeat_transform(reply))
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(reply)
                if context:
                    st.markdown(f'<div class="skill-badge">💡 CBT/DBT/Journaling tip</div>', unsafe_allow_html=True)
                    st.caption("Sources: " + ", ".join(titles))

            st.session_state.awaiting_skill_permission = False
            st.session_state.pending_rag_query = ""; st.session_state.pending_topic = ""
            st.session_state.messages += [
                {"role":"user","content":user_text,"ts":datetime.now().isoformat()},
                {"role":"assistant","content":reply,"sources":titles if context else [],"skill_used":"CBT/DBT/Journaling","ts":datetime.now().isoformat()},
            ]
            st.session_state.memory.chat_memory.add_user_message(user_text)
            st.session_state.memory.chat_memory.add_ai_message(reply)
            increment_rate_limit(); st.markdown("---"); footer_notice(); return

        # Neither clearly yes nor no → keep permission pending, supportive turn
        empathy = select_empathy(emo_lbl if emo_conf >= 0.5 else "neutral")
        memory = get_conversation_memory()
        msgs = build_support_only_prompt(user_text, memory, empathy)
        try:
            reply = _chat_once(hf_client_primary(), msgs, max_tokens, temperature)
        except Exception:
            try: reply = _chat_once(hf_client_fallback(), msgs, max_tokens, temperature)
            except Exception: reply = "Thanks for sharing that. What feels most important to explore next?"
        reply = ensure_single_question(anti_repeat_transform(reply))
        with st.chat_message("assistant", avatar="🤖"): st.markdown(reply)
        st.session_state.messages += [
            {"role":"user","content":user_text,"ts":datetime.now().isoformat()},
            {"role":"assistant","content":reply,"ts":datetime.now().isoformat()},
        ]
        st.session_state.memory.chat_memory.add_user_message(user_text)
        st.session_state.memory.chat_memory.add_ai_message(reply)
        increment_rate_limit(); st.markdown("---"); footer_notice(); return

    # 3) Decide whether to propose a skill now (gate RAG with rapport & intent/topic)
    ready_for_skill = (
        exchanges >= MIN_EXCHANGES_BEFORE_RAG and
        (top_conf >= 0.55 or int_lbl in {"providing suggestions","providing information","question"}) and
        sev_lbl not in {"crisis","red"}
    )

    if ready_for_skill:
        # Ask permission only (no RAG yet). Store the intended rag query.
        rag_query = " | ".join([user_text, f"topic:{top_lbl}"]) if top_lbl not in {"none","unknown"} else user_text
        st.session_state.awaiting_skill_permission = True
        st.session_state.pending_rag_query = rag_query
        st.session_state.pending_topic = top_lbl

        empathy = select_empathy(emo_lbl if emo_conf >= 0.5 else "neutral")
        memory = get_conversation_memory()
        ask = "Would a tiny idea be okay?"  # ≤7 words
        msgs = build_support_only_prompt(user_text, memory, empathy)
        try:
            reply = _chat_once(hf_client_primary(), msgs, max_tokens, temperature)
        except Exception:
            try: reply = _chat_once(hf_client_fallback(), msgs, max_tokens, temperature)
            except Exception: reply = "That sounds really hard. Would a tiny idea be okay?"
        if not reply.strip().endswith("?"): reply = reply.rstrip(".! ") + f" {ask}"
        reply = anti_repeat_transform(reply)
        with st.chat_message("assistant", avatar="🤖"): st.markdown(reply)
        st.session_state.messages += [
            {"role":"user","content":user_text,"ts":datetime.now().isoformat()},
            {"role":"assistant","content":reply,"ts":datetime.now().isoformat()},
        ]
        st.session_state.memory.chat_memory.add_user_message(user_text)
        st.session_state.memory.chat_memory.add_ai_message(reply)
        increment_rate_limit(); st.markdown("---"); footer_notice(); return

    # 4) Not ready for skill → supportive exploration only
    empathy = select_empathy(emo_lbl if emo_conf >= 0.5 else "neutral")
    memory = get_conversation_memory()
    msgs = build_support_only_prompt(user_text, memory, empathy)
    try:
        reply = _chat_once(hf_client_primary(), msgs, max_tokens, temperature)
    except HfHubHTTPError as e:
        if any(x in str(e).lower() for x in ["401","403","429","too many requests","forbidden","unauthorized","quota"]):
            try: reply = _chat_once(hf_client_fallback(), msgs, max_tokens, temperature)
            except Exception: reply = "I hear you. What feels most important to talk about right now?"
        else:
            reply = "I hear you. What feels most important to talk about right now?"
    except Exception:
        reply = "I hear you. What feels most important to talk about right now?"
    reply = ensure_single_question(anti_repeat_transform(reply))

    with st.chat_message("assistant", avatar="🤖"): st.markdown(reply)
    st.session_state.messages += [
        {"role":"user","content":user_text,"ts":datetime.now().isoformat()},
        {"role":"assistant","content":reply,"ts":datetime.now().isoformat()},
    ]
    st.session_state.memory.chat_memory.add_user_message(user_text)
    st.session_state.memory.chat_memory.add_ai_message(reply)
    increment_rate_limit()

    # Sidebar: classifier snapshot for this turn
    with st.sidebar:
        st.subheader("🔭 This turn")
        st.caption(f"Latency: {clf.get('_latency_ms',0)} ms")
        st.write(f"Emotion: **{emo_lbl}** ({emo_conf:.2f})")
        st.write(f"Topic: **{top_lbl}** ({top_conf:.2f})")
        st.write(f"Intent: **{int_lbl}** ({int_conf:.2f})")
        st.write(f"Severity: **{sev_lbl}** ({sev_conf:.2f})")

    st.markdown("---"); footer_notice()

if __name__ == "__main__":
    main()
