# ============================================================
# config.py — MedRAGAgents Configuration
# Default LLM vendor: Google Gemini
# ============================================================

import os
from dotenv import load_dotenv

load_dotenv()

# ── Google Gemini (default) ──────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_LLM    = os.getenv("DEFAULT_LLM", "google/gemini-1.5-flash")

# ── OpenAI (optional fallback) ───────────────────────────────
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")

# ── Anthropic (optional fallback) ────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── MedRAG corpus / retrieval settings ───────────────────────
CORPUS_DIR      = os.getenv("CORPUS_DIR", "./corpus")
CORPUS_NAME     = os.getenv("CORPUS_NAME", "Textbooks")   # Textbooks | PubMed | MedCorp …
RETRIEVER_NAME  = os.getenv("RETRIEVER_NAME", "MedCPT")   # BM25 | MedCPT | RRF-2 | RRF-4
TOP_K           = int(os.getenv("TOP_K", "32"))

# ── MedAgents multi-agent settings ───────────────────────────
NUM_QUESTION_DOMAINS = int(os.getenv("NUM_QD", "5"))   # # domain experts for question
NUM_OPTION_DOMAINS   = int(os.getenv("NUM_OD", "2"))   # # domain experts for options
MAX_VOTE_ATTEMPTS    = int(os.getenv("MAX_VOTE", "3"))  # verifier voting rounds

# ── Generation defaults ───────────────────────────────────────
TEMPERATURE  = float(os.getenv("TEMPERATURE", "0.0"))
MAX_TOKENS   = int(os.getenv("MAX_TOKENS", "2048"))

# ── Paths ─────────────────────────────────────────────────────
OUTPUT_DIR        = os.getenv("OUTPUT_DIR", "./outputs")
LONG_TERM_MEM_PATH = os.getenv("LONG_TERM_MEM", "./memory/long_term_cache.json")
DATASET_DIR       = os.getenv("DATASET_DIR", "./datasets/MedQA/")
