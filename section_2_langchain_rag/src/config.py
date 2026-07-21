"""Central config: env glue and all pipeline knobs.

Reads the generic LLM_API_KEY from .env and exposes it as GOOGLE_API_KEY so
langchain-google-genai picks it up. Swap providers by changing the two lines
under "provider glue".
"""

import os
from pathlib import Path

from dotenv import load_dotenv

SECTION_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = SECTION_ROOT / "docs"
VECTORSTORE_DIR = SECTION_ROOT / "vectorstore"

load_dotenv(SECTION_ROOT / ".env")

# Provider selection: prefer Groq if a key is present (much more generous free
# tier for interactive/voice use), fall back to Gemini otherwise. This is the
# only place a provider decision is made — graph.py branches on LLM_PROVIDER.
_groq_key = os.getenv("GROQ_API_KEY", "").strip()
_gemini_key = os.getenv("LLM_API_KEY", "").strip()

if _groq_key:
    LLM_PROVIDER = "groq"
    # 8b-instant has 500K tokens/day on the free tier (vs 100K for 70b) —
    # crucial when both the voice-agent shell AND the RAG generator share
    # the same Groq quota in the bonus voice-RAG setup.
    LLM_MODEL = "llama-3.1-8b-instant"
    # keep the same env var name the openai plugin uses
    os.environ["GROQ_API_KEY"] = _groq_key
elif _gemini_key:
    LLM_PROVIDER = "gemini"
    LLM_MODEL = "gemini-2.5-flash"
    os.environ["GOOGLE_API_KEY"] = _gemini_key
else:
    raise RuntimeError(
        "No LLM key found. Set GROQ_API_KEY (preferred) or LLM_API_KEY "
        "(Gemini) in section_2_langchain_rag/.env"
    )

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 4

# FAISS L2 distance: lower = more similar. Tuned empirically; questions whose
# best-match distance is above this are treated as out-of-scope.
RELEVANCE_THRESHOLD = 1.2

REFUSAL_MESSAGE = (
    "I couldn't find information about that in my knowledge base."
)
