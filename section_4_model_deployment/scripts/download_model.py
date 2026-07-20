"""Download the GGUF model from Hugging Face into ./models/.

Runs at Docker build time so the container is fully offline after build.
Also usable standalone for local dev.
"""

import sys
from pathlib import Path

# Make src/ importable so we reuse the same config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from huggingface_hub import hf_hub_download  # noqa: E402

from config import MODEL_DIR, MODEL_FILE, MODEL_REPO  # noqa: E402


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_FILE} from {MODEL_REPO} -> {MODEL_DIR}")
    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=str(MODEL_DIR),
    )
    print(f"Done: {path}")


if __name__ == "__main__":
    main()
