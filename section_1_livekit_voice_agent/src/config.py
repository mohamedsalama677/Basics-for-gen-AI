"""Loads .env and exposes provider constants.

Keeping these in one file means swapping providers or models is a single-file
change.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

SECTION_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(SECTION_ROOT / ".env")


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill in the keys."
        )
    return value


GROQ_API_KEY = _require("GROQ_API_KEY")
DEEPGRAM_API_KEY = _require("DEEPGRAM_API_KEY")
CARTESIA_API_KEY = _require("CARTESIA_API_KEY")

# LiveKit plugins read these from the environment
os.environ["DEEPGRAM_API_KEY"] = DEEPGRAM_API_KEY
os.environ["CARTESIA_API_KEY"] = CARTESIA_API_KEY

LLM_MODEL = "llama-3.1-8b-instant"   # Groq free tier: 30 RPM, 500K tokens/day (5x the 70b budget)
STT_MODEL = "nova-2-general"
TTS_VOICE = "794f9389-aac1-45b6-b726-9d9369183238"  # Cartesia "Sarah" — swap freely
