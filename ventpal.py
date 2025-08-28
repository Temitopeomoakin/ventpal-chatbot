# ventpal.py — VentPal (Streamlit) — classifier-first + implicit-consent RAG + rerank + actionable bullets
# -----------------------------------------------------------------------------------------------
# Key upgrades:
# • Implicit consent: run skills/RAG by default when helpful, unless the user explicitly says “no”.
# • Provider fix: no forced provider="serverless"; let HF auto-select a working provider.
# • Retrieved-chunk transparency: expander with title, chunk_id, preview, and distance.
# • Optional CrossEncoder rerank; top-k fallback if unavailable.
# • Anti-repetition, single-question, and validation injection to avoid bland loops.
# • Safer thresholds + top-k fallback so RAG rarely looks “empty” when DB has content.

import os, sys, time, json, re, random, hashlib
from typing import List, Tuple, Dict, Optional

# ---- SQLite shim for Chroma on sandboxed hosts ----
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import requests

# LangChain memory/messages
from langchain.memory import ConversationBufferMemory
try:
    from langchain_core.messages import HumanMessage, AIMessage
except Exception:
    from langchain.schema import HumanMessage, AIMessage

# Embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except Exception:
        from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# Vector store
try:
    from langchain_chroma import Chroma
except Exception:
    try:
        from langchain_community.vectorstores import Chroma
    except Exception:
        from langchain.vectorstores import Chroma

from sentence_transformers import CrossEncoder
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError

# --------------------------------- Session / State ---------------------------------
st.set_page_config(page_title="VentPal", page_icon="💨", layout="wide", initial_sidebar_state="expanded")

if "messages" not in st.session_state: st.session_state.messages = []
if "request_count" not in st.session_state: st.session_state.request_count = 0
if "last_reset" not in st.session_state: st.session_state.last_reset = time.time()
if "user_id" not in st.session_state: st.session_state.user_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, max_token_limit=2000)
if "last_question_stem" not in st.session_state: st.session_state.last_question_stem = ""
if "last_empathy" not in st.session_state: st.session_state.last_empathy = ""
if "skill_permission_left" not in st.session_state: st.session_state.skill_permission_left = 0
if "skill_cooldown" not in st.session_state: st.session_state.skill_cooldown = 0
if "last_assistant" not in st.session_state: st.session_state.last_assistant = ""
if "recent_q_stems" not in st.session_state: st.session_state.recent_q_stems = []
if "last_retrieved" not in st.session_state: st.session_state.last_retrieved = []

random.seed(st.session_state.user_id)
GDPR_COMPLIANT = True

# --------------------------------- Config / Secrets ---------------------------------
HUGGINGFACE_API_KEY   = st.secrets.get("HUGGINGFACE_API_KEY", "")
MODEL_NAME            = st.secrets.get("MODEL_NAME", "meta-llama/Llama-4-Scout-17B-16E-Instruct")
FALLBACK_MODEL        = st.secrets.get("FALLBACK_MODEL", "HuggingFaceH4/zephyr-7b-beta")
HF_PROVIDER           = st.secrets.get("HF_PROVIDER", "").strip()  # empty → auto

EMBEDDING_MODEL       = st.secrets.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VECTOR_DB_PATH        = st.secrets.get("VECTOR_DB_PATH", "vector_db")
COLLECTION_NAME       = st.secrets.get("COLLECTION_NAME", "")

MAX_REQUESTS_PER_HOUR = int(st.secrets.get("MAX_REQUESTS_PER_HOUR", 60))
MIN_EXCHANGES_BEFORE_RAG = int(st.secrets.get("MIN_EXCHANGES_BEFORE_RAG", 1))  # earlier skills

# Optional rerank for higher quality RAG
ENABLE_RERANK         = str(st.secrets.get("ENABLE_RERANK", "true")).lower() == "true"
RERANK_MODEL_NAME     = st.secrets.get("RERANK_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATES     = int(st.secrets.get("RERANK_CANDIDATES", 12))
RERANK_TOP_K          = int(st.secrets.get("RERANK_TOP_K", 4))

CLASSIFIER_URL        = st.secrets.get("CLASSIFIER_URL", "").rstrip("/")
CLASSIFIER_POLICY     = (st.secrets.get("CLASSIFIER_POLICY", "always") or "always").lower()
CLASSIFIER_AUTH       = st.secrets.get("CLASSIFIER_AUTH", "")

ASSUME_YES            = str(st.secrets.get("ASSUME_YES", "true")).lower() == "true"  # implicit consent
SKILL_GRACE_TURNS     = int(st.secrets.get("SKILL_GRACE_TURNS", 2))
SKILL_COOLDOWN_TURNS  = int(st.secrets.get("SKILL_COOLDOWN_TURNS", 2))

def _list_secret(key: str, default: List[str]) -> List[str]:
    v = st.secrets.get(key, None)
    if v is None: return [x.lower() for x in default]
    if isinstance(v, list): return [str(x).lower() for x in v]
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("["):
            try: return [str(x).lower() for x in json.loads(s)]
            except Exception: return [x.lower() for x in default]
        return [t.strip().lower() for t in s.split(",") if t.strip()]
    return [x.lower() for x in default]

SEVERITY_ALERT_P        = float(st.secrets.get("SEVERITY_ALERT_P", 0.60))
SEVERITY_HIGH_LABELS    = set(_list_secret("SEVERITY_HIGH_LABELS", ["red","crisis","severe","very_high","urgent"]))
SEVERITY_ALERT_CONTAINS = _list_secret("SEVERITY_ALERT_CONTAINS", ["red","crisis","severe","high","urgent"])

# HF auth envs
if HUGGINGFACE_API_KEY:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_API_KEY
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HUGGINGFACE_API_KEY
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

# --------------------------------- UI / CSS ---------------------------------
st.markdown("""
<style>
.main-header{background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);padding:1rem;border-radius:12px;color:#fff;text-align:center;margin-bottom:1rem}
.skill-badge{display:inline-block;background:#e8f5e8;color:#2e7d32;padding:.2rem .5rem;border-radius:12px;font-size:.75rem;margin-top:.5rem}
.footer-notice{background:#fff8e1;border:1px solid #ffeaa7;border-radius:12px;padding:.9rem;margin-top:1rem;color:#6b5e00;font-size:.92rem}
.status-chip{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;margin-left:6px}
.ok{background:#e8f5e9;color:#1b5e20;border:1px solid #a5d6a7}
.warn{background:#fff8e1;color:#6b5e00;border:1px solid #ffe082}
.err{background:#ffebee;color:#b71c1c;border:1px solid #ef9a9a}
.small-note{font-size:12px;color:#666;margin-top:.25rem}
.chunk-box{border:1px solid #e5e7eb;border-radius:10px;padding:.6rem;margin:.3rem 0;background:#fafafa}
.chunk-title{font-weight:600}
</style>
""", unsafe_allow_html=True)

DISCLAIMER = (
    "**Important:** VentPal offers supportive conversation using CBT, DBT and journaling techniques.\n"
    "It isn’t medical care and I’m **not a therapist**. I can’t diagnose, treat, or prescribe.\n"
    "If you’re in crisis or might act on thoughts of harming yourself or others, call **999** (UK),\n"
    "or contact **Samaritans (116 123)** or **Shout (text SHOUT to 85258)** right away."
)
def footer_notice(): st.markdown(f"<div class='footer-notice'>{DISCLAIMER}</div>", unsafe_allow_html=True)
def chip(text: str, kind: str = "ok"):
    cls = {"ok":"ok","warn":"warn","err":"err"}.get(kind,"ok")
    st.markdown(f'<span class="status-chip {cls}' + f"'>{text}</span>", unsafe_allow_html=True)

# --------------------------------- Safety / Crisis ---------------------------------
import re
SUICIDE_PATTERNS = [
    r"\bsuicid(?:e|al)\b", r"\bi\s*want\s*to\s*die\b", r"\bend\s*my\s*life\b", r"\bhurt\s*myself\b",
    r"\bself[-\s]?harm\b", r"\boverdose\b", r"\bkill\s*myself\b", r"\bjump\s*off\b", r"\bhang\s*myself\b",
    r"\bdrown\s*myself\b", r"\bburn\s*myself\b", r"\bunalive\b", r"\bkys\b", r"\bkms\b", r"\b(end|ending)\s+it\b"
]
CRISIS_REGEX = re.compile("|".join(SUICIDE_PATTERNS), re.IGNORECASE)
def detect_crisis_regex(t: str) -> bool: return bool(CRISIS_REGEX.search(t or ""))

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

# --------------------------------- Classifier (Cloud Run) ---------------------------------
def _cls_headers():
    h = {"Content-Type":"application/json"}
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
        r.raise_for_status(); out = r.json(); out["_latency_ms"] = int((time.time()-t0)*1000); return out
    except Exception as e:
        st.sidebar.error(f"Classifier error: {e}"); return None
def _top(head: Optional[Dict], default_label="unknown", default_conf: float = 0.0) -> Tuple[str, float]:
    top = ((head or {}).get("top") or {})
    return str(top.get("label", default_label)).lower(), float(top.get("conf", default_conf))

# --------------------------------- RAG (Chroma via LangChain) ---------------------------------
@st.cache_resource(show_spinner=False)
def create_vector_store() -> Chroma:
    if not os.path.exists(VECTOR_DB_PATH):
        st.error(f"❌ Vector DB not found at `{VECTOR_DB_PATH}`.")
        st.stop()
    last = None
    for delay in (0.5, 1.5, 3.0):
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            kwargs = dict(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
            if COLLECTION_NAME: kwargs["collection_name"] = COLLECTION_NAME
            vs = Chroma(**kwargs)
            try: vs.similarity_search("hello", k=1)
            except Exception: pass
            return vs
        except Exception as e:
            last = e; time.sleep(delay)
    st.error(f"Failed to init embeddings/Chroma. Details: {last}")
    st.stop()

@st.cache_resource(show_spinner=False)
def get_reranker() -> Optional[CrossEncoder]:
    if not ENABLE_RERANK: return None
    try:
        return CrossEncoder(RERANK_MODEL_NAME)
    except Exception as e:
        st.sidebar.warning(f"Reranker load failed: {e}. Continuing without rerank.")
        return None

def similarity_search_plus(vs: Chroma, query: str, k: int = 8) -> List[Tuple[str, Dict, float]]:
    """Return list of (text, metadata, distance) with optional rerank. Pull wider then cut down."""
    try:
        candidates = vs.similarity_search_with_score(query, k=max(k, RERANK_CANDIDATES))
    except TypeError:
        docs = vs.similarity_search(query, k=max(k, RERANK_CANDIDATES))
        candidates = [(d, 0.0) for d in docs]
    texts = [d.page_content for d, _ in candidates]
    metas = [d.metadata or {} for d, _ in candidates]
    dists = [float(s) for _, s in candidates]
    reranker = get_reranker()
    if reranker and texts:
        pairs = [(query, t) for t in texts]
        scores = reranker.predict(pairs)  # higher is better
        order = sorted(range(len(texts)), key=lambda i: scores[i], reverse=True)
        keep = order[:max(1, min(RERANK_TOP_K, k))]
        return [(texts[i], metas[i], dists[i]) for i in keep]
    return list(zip(texts[:k], metas[:k], dists[:k]))

def get_relevant_context(query: str, vectorstore: Chroma, score_threshold: float = 0.35) -> Tuple[str, List[str], List[str]]:
    """Return joined context, titles, and raw snippet list. Falls back to top-3 if threshold filters everything."""
    distance_threshold = 1.0 - score_threshold
    results = similarity_search_plus(vectorstore, query, k=8)
    chosen = []
    for text, meta, dist in results:
        if len(chosen) < 3 and (dist <= distance_threshold or distance_threshold <= 0 or dist == 0.0):
            if text: chosen.append((text, meta, dist))
    if not chosen:  # fallback: top-3 regardless of distance
        chosen = results[:3]
    chunks, titles, raw_snips, rows = [], [], [], []
    for text, meta, dist in chosen[:3]:
        if not text: continue
        short = text[:1200] + ("…" if len(text) > 1200 else "")
        chunks.append(short); raw_snips.append(text)
        title = meta.get("title") or meta.get("source") or "CBT/DBT/Journaling"
        titles.append(title)
        rows.append({
            "title": title,
            "distance": round(float(dist), 3),
            "chunk_id": meta.get("chunk_id") or meta.get("id") or "",
            "preview": text[:260] + ("…" if len(text) > 260 else ""),
        })
    st.session_state.last_retrieved = rows  # store for UI
    return ("\n\n".join(chunks) if chunks else ""), titles, raw_snips

# --------------------------------- LLM (HF) ---------------------------------
@st.cache_resource
def hf_client_primary() -> InferenceClient:
    if HF_PROVIDER:
        return InferenceClient(model=MODEL_NAME, token=HUGGINGFACE_API_KEY or None, provider=HF_PROVIDER, timeout=120)
    return InferenceClient(model=MODEL_NAME, token=HUGGINGFACE_API_KEY or None, timeout=120)

@st.cache_resource
def hf_client_fallback() -> InferenceClient:
    if HF_PROVIDER:
        return InferenceClient(model=FALLBACK_MODEL, token=HUGGINGFACE_API_KEY or None, provider=HF_PROVIDER, timeout=120)
    return InferenceClient(model=FALLBACK_MODEL, token=HUGGINGFACE_API_KEY or None, timeout=120)

SYSTEM_PROMPT = (
    "You are VentPal, a UK-English mental health companion using CBT, DBT, and journaling techniques.\n"
    "Primary goals: relieve distress, build rapport, and teach a tiny skill when appropriate.\n"
    "Never diagnose, treat, or mention medication. Avoid medical claims.\n"
    "When FACTS_FROM_RAG are provided, use only those facts when giving steps; do not invent or contradict them.\n"
    "Be concrete and actionable: prefer verbs and short steps.\n"
)
STYLE_PROMPT = (
    "Constraints:\n"
    "• Warm, plain UK English; ≤120 words.\n"
    "• Structure: Validation → Exploration → (optional) One micro-skill (≤40 words) iff TEACH_SKILL:true.\n"
    "• Exactly one open question at the end (no more than one question mark).\n"
)
SAFETY_PROMPT = (
    "If self-harm/suicidal intent: respond briefly and compassionately with UK crisis resources and ask ‘Are you able to stay safe right now?’.\n"
    "Do not include skills or RAG content in that crisis message."
)

OPEN_QUESTIONS = [
    "What feels most important to explore next?",
    "Where do you notice this most in your day?",
    "What’s one tiny step that seems doable?",
    "How does that land with you?",
    "What would feel supportive right now?",
]

def ensure_single_question(text: str) -> str:
    r = (text or "").strip()
    q_count = r.count("?")
    if q_count == 1 and r.endswith("?"): return r
    if q_count > 1: r = r.rsplit("?", 1)[0] + "?"
    if r.endswith("?"): return r
    recent = st.session_state.recent_q_stems
    choices = [q for q in OPEN_QUESTIONS if q not in recent]
    q = random.choice(choices or OPEN_QUESTIONS)
    recent.append(q)
    if len(recent) > 3: recent.pop(0)
    return r.rstrip(".! ") + ". " + q

def anti_repeat(reply: str) -> str:
    low = reply.strip().lower()
    if st.session_state.last_assistant.strip().lower() == low:
        alt = random.choice([q for q in OPEN_QUESTIONS if q not in st.session_state.recent_q_stems] or OPEN_QUESTIONS)
        reply = re.sub(r"[^.?!]*\?$", alt, reply).strip()
        if not reply.endswith("?"): reply += "?"
    st.session_state.last_assistant = reply
    return reply

def empathy_seed(emotion: str) -> str:
    groups = {
        "anxiety":["That sounds really hard.","I hear how heavy this feels.","I’m with you."],
        "depression":["That’s a lot to carry.","I’m sorry it hurts this much.","I’m with you."],
        "stress":["That’s a lot to carry.","We can take it step by step.","I’m with you."],
        "neutral":["I’m listening.","Thanks for sharing.","I’m here with you."],
        "joy":["I notice your effort.","That’s encouraging!","Nice progress."],
    }
    arr = groups.get((emotion or "neutral").lower(), groups["neutral"])
    last = st.session_state.last_empathy
    choice = random.choice([x for x in arr if x!=last] or arr)
    st.session_state.last_empathy = choice
    return choice

def build_support_only_prompt(user_text: str, memory: str, empathy: str) -> List[Dict]:
    sys = f"{SYSTEM_PROMPT}\n{STYLE_PROMPT}\n{SAFETY_PROMPT}"
    usr = f"""CONVERSATION_MEMORY:
{memory}

USER_MESSAGE:
{user_text}

TEACH_SKILL:false
EMPATHY_SEED:{empathy}
"""
    return [
        {"role":"system","content":[{"type":"text","text":sys}]},
        {"role":"user","content":[{"type":"text","text":usr}]}
    ]

def build_skill_prompt(user_text: str, memory: str, empathy: str, context: str) -> List[Dict]:
    sys = f"{SYSTEM_PROMPT}\n{STYLE_PROMPT}\n{SAFETY_PROMPT}\nUse only the FACTS_FROM_RAG when giving steps."
    usr = f"""CONVERSATION_MEMORY:
{memory}

USER_MESSAGE:
{user_text}

FACTS_FROM_RAG:
{context}

TEACH_SKILL:true
EMPATHY_SEED:{empathy}
"""
    return [
        {"role":"system","content":[{"type":"text","text":sys}]},
        {"role":"user","content":[{"type":"text","text":usr}]}
    ]

def chat_once(client: InferenceClient, messages: List[Dict], max_tokens: int, temperature: float) -> str:
    try:
        out = client.chat.completions.create(messages=messages, max_tokens=max_tokens, temperature=temperature)
        return out.choices[0].message.content
    except Exception:
        sys_prompt = ""; user_blocks = []
        for m in messages:
            role = m.get("role"); parts = m.get("content", [])
            txt = " ".join([p.get("text","") for p in parts if p.get("type")=="text"])
            if role=="system": sys_prompt += txt + "\n\n"
            elif role=="user": user_blocks.append(txt)
        prompt = (sys_prompt + "\n\n".join(user_blocks)).strip()
        return client.text_generation(prompt, max_new_tokens=max_tokens, temperature=temperature, do_sample=temperature>0)

# -------------- Actionable bullets / “RAG actually used” enforcement --------------
VERB_HINTS = {"try","breathe","inhale","exhale","write","journal","note","notice","observe","practice",
              "schedule","plan","rate","record","challenge","reframe","identify","list","choose",
              "do","call","text","ask","take","set","ground","relax","scan","label","balance","track"}

def _good_bullet_line(ln: str) -> bool:
    s = ln.strip().lstrip("•-* ").strip()
    if len(s.split()) < 5: return False
    low = s.lower()
    if any(v in low for v in VERB_HINTS): return True
    if re.match(r"^\s*(\d+[\).:]|step\s*\d+)", s, re.I): return True
    return False

def _overlap_signal(reply: str, raw_snips: List[str]) -> int:
    rep = re.sub(r"[^a-z0-9\s]", " ", reply.lower())
    words = set([w for w in rep.split() if len(w)>3])
    signal = 0
    for s in raw_snips:
        rep2 = re.sub(r"[^a-z0-9\s]", " ", s.lower())
        cand = set([w for w in rep2.split() if len(w)>5])
        if len(words.intersection(cand)) >= 5:
            signal += 1
    return signal

def append_action_bullets(reply: str, raw_snips: List[str]) -> str:
    if not raw_snips: return reply
    base = raw_snips[0]
    lines = [ln for ln in (l.strip() for l in base.splitlines()) if ln]
    picks = []
    for ln in lines:
        if ln.startswith(("• ","- ","* ","1","2","3","4","5","Step","STEP","step")) and _good_bullet_line(ln):
            picks.append("• " + ln.lstrip("•-* ").strip())
        if len(picks) >= 3: break
    if not picks:
        for ln in lines:
            if _good_bullet_line(ln):
                picks.append("• " + ln.lstrip("•-* ").strip())
            if len(picks) >= 3: break
    if not picks:
        picks = [
            "• Try 4–6 breathing for ~2 minutes (inhale 4, exhale 6).",
            "• Use 5-4-3-2-1 grounding: 5 see, 4 feel, 3 hear, 2 smell, 1 taste."
        ]
    return (reply.rstrip() + "\n\nFrom the guide:\n" + "\n".join(picks)).strip()

def enforce_rag_usage(reply: str, raw_snips: List[str]) -> str:
    if not raw_snips: return reply
    if _overlap_signal(reply, raw_snips) >= 1:
        return reply
    return append_action_bullets(reply, raw_snips)

def ensure_validation(reply: str, empathy_line: str) -> str:
    if not re.search(r"(i (hear|see|understand)|that sounds|i’m sorry|i am sorry|i’m with you|thanks for sharing)", reply, re.I):
        reply = f"{empathy_line} " + reply.lstrip()
    return reply

# --------------------------------- Helpers ---------------------------------
def get_conversation_memory() -> str:
    try:
        mem = st.session_state.memory.load_memory_variables({}).get("chat_history", [])
        out=[]
        for m in mem[-6:]:
            if isinstance(m, HumanMessage): out.append(f"User: {m.content}")
            elif isinstance(m, AIMessage): out.append(f"Assistant: {m.content}")
        return "\n".join(out) if out else "This is the start of our conversation."
    except Exception:
        return "This is the start of our conversation."

def is_blocking_no(t: str) -> bool:
    x = (t or "").strip().lower()
    NEG = ["no","nah","nope","not now","don’t","dont","rather not","skip","no tips","no advice","just listen","no skill","stop"]
    return any(k in x for k in NEG)

HELP_TOKENS = ["help","advice","how do i","what can i do","guide","steps","technique","exercise"]
def looks_helpful(t: str) -> bool:
    low = (t or "").lower()
    return any(tok in low for tok in HELP_TOKENS)

def rate_ok() -> bool:
    now = time.time()
    if now - st.session_state.last_reset > 3600:
        st.session_state.request_count = 0; st.session_state.last_reset = now
    return st.session_state.request_count < MAX_REQUESTS_PER_HOUR
def rate_inc(): st.session_state.request_count += 1

# --------------------------------- App ---------------------------------
def main():
    st.markdown(
        """<div class="main-header">
            <h1>💨 VentPal</h1>
            <p>Gentle support with CBT, DBT, and journaling.</p>
        </div>""", unsafe_allow_html=True
    )

    if not HUGGINGFACE_API_KEY: st.error("Missing HUGGINGFACE_API_KEY in secrets."); st.stop()
    if not CLASSIFIER_URL or CLASSIFIER_POLICY == "off":
        st.error("Classifier must be enabled. Set CLASSIFIER_URL and CLASSIFIER_POLICY='always'."); st.stop()

    with st.spinner("Connecting to knowledge base…"):
        vectorstore = create_vector_store()
    probe = probe_vector_db(vectorstore)
    kb_hint = ""
    if not probe["any"]:
        kb_hint = ("No obvious results in a quick probe. Retrieval still runs each turn. "
                   "If you expect matches, check VECTOR_DB_PATH/COLLECTION_NAME or lower the relevance slider.")

    with st.spinner("Checking classifier…"):
        hc = classifier_health()
        if not hc: st.error("Classifier /health failed. Fix Cloud Run service."); st.stop()

    # Sidebar: system/controls
    with st.sidebar:
        st.header("🧩 System status")
        st.write("Classifier"); chip("connected ✓", "ok")
        st.caption(f"vocab:{hc.get('vocab_size','?')} | pos:{hc.get('max_position_embeddings','?')}")
        st.write("RAG"); chip("ready ✓", "ok")
        if probe["titles"]: st.caption("Probe: " + ", ".join(probe["titles"]))
        if kb_hint: st.warning(kb_hint)
        st.write("LLM"); chip("configured ✓", "ok")
        if GDPR_COMPLIANT: st.info("🔒 Session-only; chats aren’t stored server-side.")

        st.subheader("🔧 Settings")
        temperature = st.slider("Creativity", 0.1, 1.0, 0.7, 0.1)
        max_tokens  = st.slider("Response length", 60, 220, 160, 10)
        score_thr   = st.slider("RAG relevance", 0.2, 0.9, 0.35, 0.05)
        show_chunks = st.toggle("Show retrieved chunks", True)

        st.subheader("📊 Usage")
        st.metric("Requests this hour", st.session_state.request_count)

    # Name gate
    name = st.text_input("What should I call you?", value=st.session_state.get("user_name",""), placeholder="Your name")
    st.session_state.user_name = (name or "").strip()
    if not st.session_state.user_name:
        st.markdown('<div class="small-note">Enter a name to start chatting.</div>', unsafe_allow_html=True)
        st.markdown("---"); footer_notice(); return

    # History
    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar=("🤖" if m["role"]=="assistant" else "👤")):
            st.markdown(m["content"])
            if m.get("skill_used"): st.markdown(f"<div class='skill-badge'>💡 {m['skill_used']}</div>", unsafe_allow_html=True)
            if m.get("sources"): st.caption("Sources: " + ", ".join(m["sources"]))
            if show_chunks and m.get("retrieved"):
                with st.expander("🔍 Retrieved Chunks"):
                    for row in m["retrieved"]:
                        st.markdown(
                            f"<div class='chunk-box'><div class='chunk-title'># {row['title']} — distance: {row['distance']}</div>"
                            f"<div><code>{row['chunk_id']}</code></div>"
                            f"<div>{row['preview']}</div></div>",
                            unsafe_allow_html=True
                        )

    # Input
    user_text = st.chat_input(f"How are you feeling today, {st.session_state.user_name}?")
    if not user_text:
        st.markdown("---"); footer_notice(); return
    if not rate_ok():
        st.error("You’ve hit the hourly limit. Try again later."); st.stop()

    with st.chat_message("user", avatar="👤"): st.markdown(user_text)

    # 1) Classify & safety
    clf = classify(user_text)
    if not clf: st.error("Classifier call failed; try again."); st.stop()
    emo_lbl, emo_conf = _top(clf.get("emotion"), "neutral", 0.0)
    top_lbl, top_conf = _top(clf.get("topic"),   "none",    0.0)
    int_lbl, int_conf = _top(clf.get("intent"),  "others",  0.0)
    sev_lbl, sev_conf = _top(clf.get("severity"),"green",   0.0)

    # Sidebar: show labels
    with st.sidebar:
        st.subheader("🔭 This turn")
        st.caption(f"Emotion: {emo_lbl} ({emo_conf:.2f}) | Topic: {top_lbl} ({top_conf:.2f})")
        st.caption(f"Intent: {int_lbl} ({int_conf:.2f}) | Severity: {sev_lbl} ({sev_conf:.2f})")

    if detect_crisis_regex(user_text) or severity_triggers_alert(sev_lbl, sev_conf) or "suicid" in int_lbl:
        msg = crisis_block()
        with st.chat_message("assistant", avatar="🤖"): st.markdown(msg)
        st.session_state.messages += [{"role":"user","content":user_text},{"role":"assistant","content":msg,"crisis_alert":True}]
        st.session_state.memory.chat_memory.add_user_message(user_text)
        st.session_state.memory.chat_memory.add_ai_message(msg)
        rate_inc(); st.markdown("---"); footer_notice(); return

    # Skill cooldown decrement
    if st.session_state.skill_cooldown > 0:
        st.session_state.skill_cooldown -= 1

    # Decide whether to run RAG
    exchanges_so_far = sum(1 for m in st.session_state.messages if m["role"]=="assistant")
    ready_for_skill = (
        exchanges_so_far >= MIN_EXCHANGES_BEFORE_RAG and
        (top_conf >= 0.30 or emo_lbl in {"depression","anxiety","stress"} or looks_helpful(user_text)) and
        sev_lbl not in {"crisis","red"} and
        st.session_state.skill_cooldown == 0
    )

    def _do_support_only():
        empathy = empathy_seed(emo_lbl if emo_conf>=0.5 else "neutral")
        memory  = get_conversation_memory()
        msgs    = build_support_only_prompt(user_text, memory, empathy)
        try: reply = chat_once(hf_client_primary(), msgs, max_tokens, temperature)
        except Exception:
            try: reply = chat_once(hf_client_fallback(), msgs, max_tokens, temperature)
            except Exception: reply = "Thanks for saying that. What would feel supportive right now?"
        reply = ensure_validation(reply, empathy)
        reply = anti_repeat(ensure_single_question(reply))
        with st.chat_message("assistant", avatar="🤖"): st.markdown(reply)
        st.session_state.messages += [{"role":"user","content":user_text},{"role":"assistant","content":reply}]
        st.session_state.memory.chat_memory.add_user_message(user_text)
        st.session_state.memory.chat_memory.add_ai_message(reply)
        rate_inc(); st.markdown("---"); footer_notice();

    def _do_skill_turn():
        rag_query = f"{user_text} | topic:{top_lbl}" if top_lbl not in {"none","unknown"} else user_text
        context, titles, raw_snips = get_relevant_context(rag_query, vectorstore, score_thr)
        empathy = empathy_seed(emo_lbl if emo_conf>=0.5 else "neutral")
        memory  = get_conversation_memory()
        msgs    = build_skill_prompt(user_text, memory, empathy, context or "")
        try:
            reply = chat_once(hf_client_primary(), msgs, max_tokens, temperature)
        except HfHubHTTPError as e:
            if any(x in str(e).lower() for x in ["401","403","429","too many requests","forbidden","unauthorized","quota"]):
                try: reply = chat_once(hf_client_fallback(), msgs, max_tokens, temperature)
                except Exception: reply = "Let’s keep it simple: what would feel supportive right now?"
            else:
                reply = "Let’s keep it simple: what would feel supportive right now?"
        except Exception:
            reply = "Let’s keep it simple: what would feel supportive right now?"

        reply = ensure_validation(reply, empathy)
        reply = enforce_rag_usage(ensure_single_question(anti_repeat(reply)), raw_snips)

        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(reply)
            if context: st.markdown(f"<div class='skill-badge'>💡 Skill shared</div>", unsafe_allow_html=True)
            if show_chunks and st.session_state.last_retrieved:
                with st.expander("🔍 Retrieved Chunks"):
                    for row in st.session_state.last_retrieved:
                        st.markdown(
                            f"<div class='chunk-box'><div class='chunk-title'># {row['title']} — distance: {row['distance']}</div>"
                            f"<div><code>{row['chunk_id']}</code></div>"
                            f"<div>{row['preview']}</div></div>",
                            unsafe_allow_html=True
                        )

        st.session_state.messages += [{"role":"user","content":user_text},
                                      {"role":"assistant","content":reply,
                                       "sources":titles if context else [],
                                       "skill_used":"CBT/DBT/Journaling",
                                       "retrieved": st.session_state.last_retrieved}]
        st.session_state.memory.chat_memory.add_user_message(user_text)
        st.session_state.memory.chat_memory.add_ai_message(reply)
        rate_inc(); st.markdown("---"); footer_notice()

    # Implicit consent logic (default YES unless user says no)
    if ready_for_skill and ASSUME_YES and not is_blocking_no(user_text):
        st.session_state.skill_permission_left = SKILL_GRACE_TURNS
        _do_skill_turn(); return

    # Explicit “no” → support only and cooldown
    if is_blocking_no(user_text):
        st.session_state.skill_cooldown = SKILL_COOLDOWN_TURNS
        _do_support_only(); return

    # If previously granted grace turns, continue
    if st.session_state.skill_permission_left > 0:
        st.session_state.skill_permission_left -= 1
        _do_skill_turn(); return

    # Otherwise, if ready, still do a skill turn
    if ready_for_skill:
        _do_skill_turn(); return

    # Default: support only
    _do_support_only(); return

# ---- Utilities used above ----
def probe_vector_db(vs: Chroma) -> Dict:
    try:
        probes = ["panic attack","deep breathing","journaling prompt","grounding","thought challenging"]
        total = 0; titles=set()
        for q in probes:
            try:
                res = vs.similarity_search(q, k=3)
                pairs = [(d, 0.0) for d in res]
            except Exception:
                pairs = vs.similarity_search_with_score(q, k=3)
            total += len(pairs)
            for doc,score in pairs:
                titles.add((doc.metadata or {}).get("title", (doc.metadata or {}).get("source","CBT/DBT")))
        return {"any": total>0, "titles": list(titles)[:5], "count": total}
    except Exception:
        return {"any": False, "titles": [], "count": 0}

if __name__ == "__main__":
    main()
