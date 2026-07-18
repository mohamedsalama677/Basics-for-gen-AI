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

_raw_key = os.getenv("LLM_API_KEY", "").strip()
if not _raw_key:
    raise RuntimeError(
        "LLM_API_KEY not found. Add it to section_2_langchain_rag/.env"
    )

# provider glue — one place to change when swapping vendors
os.environ["GOOGLE_API_KEY"] = _raw_key
LLM_MODEL = "gemini-2.5-flash"

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
