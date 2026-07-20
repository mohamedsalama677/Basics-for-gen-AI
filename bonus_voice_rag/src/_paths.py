"""Path glue so this bonus can import from Section 1 and Section 2 as-is.

The tricky bit: both Section 1's src/ and Section 2's src/ contain a file
named `config.py`, and Python module caching would make it impossible to
import both under the same name via bare `import config`. This module:

1. Puts Section 2's src/ on sys.path so Section 2's own internal imports
   (e.g. `graph.py` doing `from config import EMBEDDING_MODEL, ...`)
   resolve to Section 2's config.
2. Loads Section 1's config module explicitly via importlib under the
   unique name `s1_config`, so it can be imported independently without
   the collision.

Callers do:

    from _paths import s1_config
    from graph import build_graph          # ← Section 2

Notice the bonus folder itself has NO `config.py` — that's intentional,
because a `bonus_voice_rag/src/config.py` would shadow Section 2's
`config.py` at sys.path[0] and break Section 2's own imports.
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SECTION_1_SRC = REPO_ROOT / "section_1_livekit_voice_agent" / "src"
SECTION_2_SRC = REPO_ROOT / "section_2_langchain_rag" / "src"

# 1. Section 2's src goes on sys.path so its internal `from config import ...`
#    lines in graph.py resolve to Section 2's own config.
if str(SECTION_2_SRC) not in sys.path:
    sys.path.insert(0, str(SECTION_2_SRC))

# 2. Section 1's config gets loaded explicitly under a unique name so it
#    doesn't collide with Section 2's `config` in sys.modules.
_spec = importlib.util.spec_from_file_location(
    "s1_config", SECTION_1_SRC / "config.py"
)
if _spec is None or _spec.loader is None:
    raise RuntimeError(
        f"Could not locate Section 1 config at {SECTION_1_SRC / 'config.py'}"
    )
s1_config = importlib.util.module_from_spec(_spec)
sys.modules["s1_config"] = s1_config
_spec.loader.exec_module(s1_config)
