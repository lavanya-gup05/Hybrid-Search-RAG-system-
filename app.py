"""Hybrid-Search RAG with Citation Verification — Streamlit front end.

Run:  streamlit run app.py

UI notes: the color system maps meaning, not decoration — teal marks
retrieval/supported, amber marks partial, coral marks unsourced/primary
actions. The gradient title is the one deliberately bold moment; everything
around it stays flat and disciplined.
"""

import html
import logging
import os
import tempfile
import time
import uuid

import pandas as pd
import streamlit as st

from src.config import CFG
from src.ingest import load_corpus
from src.retriever import HybridRetriever
from src.reranker import Reranker
from src.generator import Generator
from src.verifier import Verifier
from src.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hybrid_rag")

# Each browser tab gets its own id, so its Chroma collection and index
# directory never collide with another concurrent user's — Chroma persistence
# and st.cache_resource are process-wide, not per-session, by default.
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]
if "last_query_at" not in st.session_state:
    st.session_state.last_query_at = 0.0

st.set_page_config(page_title="Hybrid RAG + Citation Verification",
                   page_icon="🔎", layout="wide")

# ------------------------------------------------------------------ styling --

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@500;600;700&family=Work+Sans:wght@400;500;600;700&display=swap');

:root {
  --ink:      #24303D;
  --cream:    #FFF8EC;
  --coral:    #FF6B4A;
  --coral-10: #FFEDE7;
  --teal:     #1FB6A6;
  --teal-10:  #E4F8F5;
  --sun:      #FFC93C;
  --sun-10:   #FFF6DD;
  --muted:    #6B7684;
}

html, body, [class*="css"]  { font-family: 'Work Sans', sans-serif; }
h1, h2, h3, .hero-title { font-family: 'Fredoka', sans-serif; }

/* ---- hero ---- */
.hero-wrap { padding: 0.25rem 0 1.25rem 0; }
.hero-title {
  font-size: 2.7rem; font-weight: 700; line-height: 1.15; margin: 0;
  letter-spacing: -0.015em;
  background: linear-gradient(90deg, var(--coral) 0%, var(--sun) 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-sub {
  color: var(--muted); font-size: 1.04rem; line-height: 1.55;
  margin: 0.5rem 0 1rem 0; max-width: 640px;
}
.pipeline-row { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; }
.chip {
  display: inline-flex; align-items: center; gap: 0.35rem;
  padding: 0.4rem 0.9rem; border-radius: 999px; font-size: 0.86rem; font-weight: 600;
  transition: transform 0.12s ease;
}
.chip:hover { transform: translateY(-1px); }
.chip-search  { background: var(--teal-10); color: var(--teal); }
.chip-verify  { background: var(--coral-10); color: var(--coral); }
.chip-arrow   { color: var(--muted); font-size: 0.9rem; font-weight: 600; }

/* ---- section headings inside main area ---- */
.section-heading {
  font-family: 'Fredoka', sans-serif; font-weight: 600; font-size: 1.4rem;
  color: var(--ink); display: flex; align-items: center; gap: 0.45rem;
  margin: 0.2rem 0 0.5rem 0;
}

/* ---- answer card ---- */
.answer-card {
  background: #FFFFFF; border-radius: 20px; padding: 1.4rem 1.6rem;
  border-left: 6px solid var(--coral);
  box-shadow: 0 10px 30px -12px rgba(255, 107, 74, 0.25);
  margin: 0.75rem 0 1.25rem 0; font-size: 1.05rem; line-height: 1.6;
}
.answer-card s { color: var(--muted); }

/* ---- stat pills ---- */
.stat-row { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 1rem; }
.stat-pill {
  flex: 1; min-width: 130px; border-radius: 16px; padding: 0.9rem 1.1rem;
  background: #FFFFFF; transition: transform 0.12s ease, box-shadow 0.12s ease;
}
.stat-pill:hover { transform: translateY(-2px); }
.stat-num { font-family: 'Fredoka', sans-serif; font-size: 2rem; font-weight: 600; line-height: 1; }
.stat-label {
  font-size: 0.8rem; color: var(--muted); margin-top: 0.35rem; font-weight: 600;
  display: flex; align-items: center; gap: 0.3rem;
}
.stat-neutral { border-top: 5px solid var(--ink); }
.stat-neutral .stat-num { color: var(--ink); }
.stat-supported { border-top: 5px solid var(--teal); box-shadow: 0 8px 22px -14px rgba(31,182,166,0.4); }
.stat-supported .stat-num { color: var(--teal); }
.stat-partial { border-top: 5px solid var(--sun); box-shadow: 0 8px 22px -14px rgba(255,201,60,0.5); }
.stat-partial .stat-num { color: #B8860B; }
.stat-unsourced { border-top: 5px solid var(--coral); box-shadow: 0 8px 22px -14px rgba(255,107,74,0.4); }
.stat-unsourced .stat-num { color: var(--coral); }

/* ---- claim cards (citation audit) ---- */
.claim-card {
  background: #FFFFFF; border-radius: 14px; padding: 0.95rem 1.15rem;
  margin-bottom: 0.6rem; display: flex; gap: 0.9rem; align-items: flex-start;
  transition: transform 0.12s ease, box-shadow 0.12s ease;
}
.claim-card:hover { transform: translateY(-1px); box-shadow: 0 10px 22px -16px rgba(36,48,61,0.35); }
.claim-supported { border-left: 5px solid var(--teal); }
.claim-partial    { border-left: 5px solid var(--sun); }
.claim-bad        { border-left: 5px solid var(--coral); }
.verdict-badge {
  flex-shrink: 0; padding: 0.28rem 0.7rem; border-radius: 999px;
  font-size: 0.74rem; font-weight: 700; letter-spacing: 0.01em; white-space: nowrap;
}
.badge-supported { background: var(--teal-10); color: var(--teal); }
.badge-partial    { background: var(--sun-10); color: #B8860B; }
.badge-bad        { background: var(--coral-10); color: var(--coral); }
.claim-text { font-size: 0.94rem; color: var(--ink); line-height: 1.5; }
.claim-meta {
  font-size: 0.76rem; color: var(--muted); margin-top: 0.3rem;
  font-family: 'Menlo', 'Consolas', monospace;
}

/* ---- sidebar section labels ---- */
.side-label {
  font-family: 'Fredoka', sans-serif; font-weight: 600; font-size: 1rem;
  color: var(--ink); margin: 0 0 0.5rem 0; display: flex; align-items: center; gap: 0.4rem;
}

/* ---- sidebar section cards (st.container(border=True)) ---- */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {
  background: #FFFFFF !important; border: none !important; border-radius: 16px !important;
  padding: 1rem 1.1rem 1.15rem 1.1rem !important; margin-bottom: 0.85rem !important;
  box-shadow: 0 8px 20px -14px rgba(36,48,61,0.35);
}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"]:nth-of-type(1) { border-top: 4px solid var(--sun) !important; }
section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"]:nth-of-type(2) { border-top: 4px solid var(--teal) !important; }
section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"]:nth-of-type(3) { border-top: 4px solid var(--coral) !important; }
section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"]:nth-of-type(4) { border-top: 4px solid var(--teal) !important; }

/* ---- tabs as pills ---- */
button[data-baseweb="tab"] {
  border-radius: 999px !important; padding: 0.4rem 1.1rem !important;
  font-weight: 600 !important; font-size: 0.92rem !important;
}
div[data-baseweb="tab-highlight"] { background-color: var(--coral) !important; }

/* ---- expanders (retrieved passages) ---- */
div[data-testid="stExpander"] {
  border-radius: 14px !important; border-left: 5px solid var(--teal) !important;
  overflow: hidden;
}
div[data-testid="stExpander"] summary {
  font-weight: 600 !important; font-size: 0.93rem !important; color: var(--ink) !important;
}
div[data-testid="stExpander"] summary:hover { background: var(--teal-10) !important; }

/* ---- text inputs ---- */
div[data-testid="stTextInput"] input {
  border-radius: 12px !important; border: 1.5px solid #ECE3D3 !important;
  padding: 0.6rem 0.9rem !important; font-size: 0.96rem !important;
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
div[data-testid="stTextInput"] input:focus {
  border-color: var(--coral) !important; box-shadow: 0 0 0 3px var(--coral-10) !important;
}
div[data-testid="stTextInput"] label p {
  font-weight: 600 !important; font-size: 0.87rem !important; color: var(--ink) !important;
}
/* the big question box gets a slightly larger, friendlier feel */
div[data-testid="stTextInput"]:has(input[aria-label*="Ask a question"]) input {
  font-size: 1.08rem !important; padding: 0.85rem 1.1rem !important; border-radius: 14px !important;
}

/* ---- slider / toggle / radio / checkbox labels ---- */
div[data-testid="stSlider"] label p,
div[data-testid="stToggle"] label p,
div[data-testid="stRadio"] label p {
  font-weight: 600 !important; font-size: 0.87rem !important; color: var(--ink) !important;
}
div[data-testid="stCheckbox"] label p { font-size: 0.92rem !important; }

/* ---- upload button ---- */
div[data-testid="stFileUploaderDropzone"] button {
  background: var(--coral) !important; color: #FFFFFF !important; border: none !important;
  border-radius: 999px !important; font-weight: 600 !important;
}
div[data-testid="stFileUploaderDropzone"] button:hover { background: #E85A3C !important; }

/* ---- alerts (info / success) ---- */
div[data-testid="stAlertContainer"] {
  border-radius: 14px !important; border: none !important; font-weight: 500 !important;
}

/* ---- misc ---- */
div[data-testid="stFileUploaderDropzone"] { border-radius: 16px !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    from sentence_transformers import SentenceTransformer, CrossEncoder
    return SentenceTransformer(CFG.embed_model), Reranker()


@st.cache_resource(show_spinner="Indexing documents…")
def build_pipeline(session_id, file_sig, paths, api_key):
    """session_id is part of the cache key so two browser tabs never share a
    Chroma collection or a cached Pipeline instance — without it, one user's
    upload would silently rebuild the index out from under another user's
    in-flight query (HybridRetriever(rebuild=True) drops the collection)."""
    embedder, reranker = load_models()
    chunks = load_corpus(paths)
    persist_dir = os.path.join(tempfile.gettempdir(), "hybrid_rag_index", session_id)
    os.makedirs(persist_dir, exist_ok=True)
    retriever = HybridRetriever(chunks, embedder=embedder, persist_dir=persist_dir)
    return Pipeline(retriever, reranker,
                    Generator(api_key=api_key), Verifier(api_key=api_key)), len(chunks)


def verdict_class(v: str) -> tuple[str, str, str]:
    """Returns (card class, badge class, display label) for a verdict."""
    if v == "SUPPORTED":
        return "claim-supported", "badge-supported", "✓ Supported"
    if v == "PARTIAL":
        return "claim-partial", "badge-partial", "◐ Partial"
    if v == "UNCITED":
        return "claim-bad", "badge-bad", "? Uncited"
    return "claim-bad", "badge-bad", "✕ Unsupported"


# ------------------------------- sidebar -------------------------------------

with st.sidebar:
    with st.container(border=True):
        st.markdown('<div class="side-label">🔑 Setup</div>', unsafe_allow_html=True)
        # Do NOT default this field to os.getenv("GROQ_API_KEY") in a shared
        # deployment: Streamlit ships the input's default value to every
        # client's browser DOM, which would leak the server's own key to
        # every visitor. Each user pastes their own key instead. If this is
        # genuinely single-user/local, set HYBRID_RAG_PREFILL_KEY=1 to opt in.
        prefill = os.getenv("GROQ_API_KEY", "") if os.getenv("HYBRID_RAG_PREFILL_KEY") else ""
        api_key = st.text_input("Groq API key", type="password", value=prefill)

    with st.container(border=True):
        st.markdown('<div class="side-label">📄 Documents</div>', unsafe_allow_html=True)
        uploads = st.file_uploader("Upload .pdf / .txt / .md", type=["pdf", "txt", "md"],
                                   accept_multiple_files=True, label_visibility="collapsed")
        use_sample = st.checkbox("Use sample corpus", value=not uploads)

    with st.container(border=True):
        st.markdown('<div class="side-label">🎛️ Retrieval</div>', unsafe_allow_html=True)
        # Kept as local variables and passed explicitly into pipeline.run();
        # never written back to the shared CFG singleton, which is process-wide
        # and would otherwise leak one user's slider settings into another
        # user's concurrent request.
        final_top_k = st.slider("Passages sent to the LLM", 3, 10, CFG.final_top_k)
        fused_top_k = st.slider("Candidates into reranker", 5, 40, CFG.fused_top_k)

    with st.container(border=True):
        st.markdown('<div class="side-label">✅ Verification</div>', unsafe_allow_html=True)
        verify = st.toggle("Citation verification", value=True)
        mode = st.radio("Unsupported claims", ["flag", "strip"], horizontal=True,
                        help="flag = show them struck through; strip = remove them")

# ------------------------------- hero -----------------------------------------

st.markdown("""
<div class="hero-wrap">
  <div class="hero-title">🔎 Hybrid-Search RAG</div>
  <p class="hero-sub">BM25 and dense retrieval fused, reranked, and generated with
  citations — then every claim gets independently checked against its source
  before you see it.</p>
  <div class="pipeline-row">
    <span class="chip chip-search">🔤 BM25</span>
    <span class="chip-arrow">+</span>
    <span class="chip chip-search">🧠 Dense</span>
    <span class="chip-arrow">→</span>
    <span class="chip chip-search">🎯 Rerank</span>
    <span class="chip-arrow">→</span>
    <span class="chip chip-verify">✅ Verify</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------- indexing ------------------------------------

if uploads and len(uploads) > CFG.max_uploads:
    st.error(f"Too many files ({len(uploads)}). Max {CFG.max_uploads} per session.")
    st.stop()

oversized = [u.name for u in (uploads or [])
            if u.size > CFG.max_upload_mb * 1024 * 1024]
if oversized:
    st.error(f"These files exceed {CFG.max_upload_mb} MB and were rejected: "
             f"{', '.join(oversized)}")
    st.stop()

paths = []
if uploads:
    tmpdir = tempfile.mkdtemp()
    for up in uploads:
        p = os.path.join(tmpdir, up.name)
        with open(p, "wb") as f:
            f.write(up.getbuffer())
        paths.append(p)
elif use_sample:
    paths = ["data"]

if not api_key:
    st.info("Add a Groq API key in the sidebar to start. "
            "Free keys: console.groq.com")
    st.stop()
if not paths:
    st.info("Upload documents or tick the sample corpus.")
    st.stop()

sig = tuple(sorted(os.path.basename(p) for p in paths))
try:
    pipeline, n_chunks = build_pipeline(st.session_state.session_id, sig, paths, api_key)
except Exception as e:
    logger.exception("Indexing failed")
    st.error("Couldn't index those documents. Check the file formats and try again.")
    st.stop()
st.success(f"Indexed {n_chunks} chunks from {len(sig)} source(s).")

# ------------------------------- query ---------------------------------------

question = st.text_input("Ask a question about the documents",
                         placeholder="e.g. What were the drivers of margin decline?",
                         max_chars=CFG.max_question_chars)

if question:
    elapsed = time.time() - st.session_state.last_query_at
    if elapsed < CFG.session_min_interval_s:
        st.warning("You're querying faster than the app allows — please wait "
                   f"{CFG.session_min_interval_s - elapsed:.1f}s and try again.")
        st.stop()
    if len(question) > CFG.max_question_chars:
        st.error(f"Question is too long (max {CFG.max_question_chars} characters).")
        st.stop()

    st.session_state.last_query_at = time.time()
    try:
        with st.spinner("Retrieving, reranking, verifying…"):
            res = pipeline.run(question, verify=verify, mode=mode,
                               fused_top_k=fused_top_k, final_top_k=final_top_k)
    except Exception as e:
        logger.exception("Pipeline run failed")
        st.error("Something went wrong answering that question. "
                 "This has been logged — please try again in a moment.")
        st.stop()

    st.markdown('<div class="section-heading">💬 Answer</div>', unsafe_allow_html=True)
    # The answer text originates from the LLM (which in turn saw uploaded
    # document content), so it is untrusted input as far as the browser is
    # concerned. Escape it before embedding in a raw HTML div so a document
    # that tricks the model into echoing "<script>…</script>" can't execute.
    safe_answer = html.escape(res.final_answer)
    st.markdown(f'<div class="answer-card">{safe_answer}</div>',
               unsafe_allow_html=True)

    if res.metrics:
        m = res.metrics
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-pill stat-neutral"><div class="stat-num">{m['claims']}</div><div class="stat-label">📊 Claims</div></div>
          <div class="stat-pill stat-supported"><div class="stat-num">{m['SUPPORTED']}</div><div class="stat-label">✅ Supported</div></div>
          <div class="stat-pill stat-partial"><div class="stat-num">{m['PARTIAL']}</div><div class="stat-label">◐ Partial</div></div>
          <div class="stat-pill stat-unsourced"><div class="stat-num">{m['UNSUPPORTED'] + m['UNCITED']}</div><div class="stat-label">❌ Unsourced</div></div>
        </div>
        """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🩺 Citation audit", "📚 Retrieved passages", "⚙️ Trace"])

    with tab1:
        if not res.checks:
            st.caption("Verification off.")
        else:
            for c in res.checks:
                card_cls, badge_cls, label = verdict_class(c.verdict)
                cites = ", ".join(f"[{i}]" for i in c.citations) or "—"
                safe_sentence = html.escape(c.sentence)
                safe_per_citation = html.escape(str(c.per_citation))
                st.markdown(f"""
                <div class="claim-card {card_cls}">
                  <div class="verdict-badge {badge_cls}">{label}</div>
                  <div>
                    <div class="claim-text">{safe_sentence}</div>
                    <div class="claim-meta">Cites {cites} · {safe_per_citation}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        for i, (c, s) in enumerate(zip(res.contexts, res.rerank_scores), 1):
            with st.expander(f"[{i}] {c.source} · rerank score {s:.2f}"):
                st.write(c.text)
                st.caption(f"chunk {c.id} · chars {c.start}–{c.end}")

    with tab3:
        st.json({"timings_sec": res.timings,
                 "config": {"fused_top_k": fused_top_k,
                            "final_top_k": final_top_k,
                            "embed": CFG.embed_model,
                            "rerank": CFG.rerank_model,
                            "llm": CFG.llm_model}})
        with st.expander("Raw answer before verification"):
            st.write(res.raw_answer)