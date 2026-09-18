import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # --- chunking ---
    chunk_size: int = 900          # characters, not tokens (keeps deps light)
    chunk_overlap: int = 150

    # --- retrieval ---
    bm25_top_k: int = 25           # candidates from sparse
    dense_top_k: int = 25          # candidates from dense
    rrf_k: int = 60                # reciprocal rank fusion constant
    fused_top_k: int = 20          # what goes into the reranker
    final_top_k: int = 5           # what goes into the LLM prompt

    # --- models ---
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "openai/gpt-oss-120b"      
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")

    # --- verification ---
    verify: bool = True
    min_support: str = "SUPPORTED"  # sentences below this get stripped/flagged

    index_dir: str = "index"

    # --- production hardening ---
    llm_timeout_s: float = 30.0        # per-call Groq timeout; avoid a hung session
    max_question_chars: int = 500      # reject absurd inputs before they hit retrieval
    max_upload_mb: int = 20            # per-file upload cap
    max_uploads: int = 10              # per-session file count cap
    session_min_interval_s: float = 3.0  # simple per-session rate limit for the query box


CFG = Config()