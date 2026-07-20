"""Central config for the deployment: model, paths, generation defaults.

Everything the app needs to know about which model to load and how to run it
lives here so the app.py and model.py stay focused on their jobs.
"""

import os
from pathlib import Path

MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

# The Dockerfile places the model at /app/models/<file>. In local dev the
# download_model.py script uses the same relative path.
_project_root = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", _project_root / "models"))
MODEL_PATH = MODEL_DIR / MODEL_FILE

CTX_SIZE = 2048
N_THREADS = int(os.getenv("N_THREADS", os.cpu_count() or 4))

DEFAULT_MAX_TOKENS = 128
DEFAULT_TEMPERATURE = 0.2

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
