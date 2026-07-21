# Generative AI — Reference Implementations

A hands-on portfolio project covering four production-adjacent skills for
building generative-AI systems: real-time voice agents, retrieval-augmented
generation, model quantization trade-offs, and Dockerized model serving.
Each skill is a self-contained folder with its own README and its own
`run` instructions, plus a bonus section that composes two of them into a
single voice-based knowledge assistant.

Author: **Mohamed Salama** — mohamed.salama@hexifyer.com

---

## What's in each section

### [Section 1 — LiveKit Voice Agent](section_1_livekit_voice_agent/)
Real-time voice agent using the `livekit-agents` SDK. A food-delivery
support persona ("Rida" from SwiftEats) that answers order-status questions
by having the LLM call a `@function_tool`-decorated Python function
mid-conversation. Pipeline: **silero VAD → Deepgram STT → Groq LLM
(Llama-3.3-70b) → Cartesia TTS**, run in LiveKit's local `console` mode
(no LiveKit Cloud needed). Bonus (1.2) ships an `agent_swap.py` that
substitutes Google Cloud STT for Deepgram with a one-line change, proving
the pipeline is vendor-decoupled.

### [Section 2 — LangChain RAG](section_2_langchain_rag/)
Retrieval-augmented-generation pipeline built with **LangGraph** over three
in-repo markdown docs (LLMs & transformers, RAG vs fine-tuning, agentic AI).
Chunked and embedded with `sentence-transformers/all-MiniLM-L6-v2` into a
local FAISS index. The graph explicitly routes off-topic questions to a
deterministic **refusal node** based on a FAISS similarity threshold —
no LLM call happens on out-of-scope queries, so hallucination is
structurally impossible. Generation runs through Gemini 2.5 Flash with
inline `[source.md#chunk_N]` citations. Includes 3 recorded example Q&As
(two in-KB, one out-of-KB) and a half-page write-up on chunking / hybrid
search / re-rankers.

### [Section 3 — Quantization](section_3_quantization/)
Notebook comparing `Qwen2.5-1.5B-Instruct` in **fp16** vs **4-bit NF4**
(`bitsandbytes`) on the same 5 fixed prompts on a Colab T4 GPU. Reports
weights VRAM, peak VRAM, throughput (tok/s), and side-by-side output
quality — plus a half-page write-up on when to pick **GPTQ / AWQ over
bitsandbytes**, and when **GGUF over both**. Live measurements from my
run: ~62% weight-memory saving at 4-bit, ~0.61× throughput vs fp16, with
qualitative output near-parity on this model.

### [Section 4 — Model Deployment](section_4_model_deployment/)
`Qwen2.5-0.5B-Instruct` in **GGUF Q4_K_M** served behind **FastAPI** via
`llama-cpp-python`, containerized on `python:3.11-slim` — **CPU-only, no
GPU required**. Three endpoints: `GET /health`, `POST /generate` (blocking),
`POST /generate/stream` (SSE token streaming). Plus a `/docs` Swagger UI
free from FastAPI. Ships with an async load-test script that fires 10
concurrent streaming requests and reports TTFT / total latency / tokens
per second. Model choice is a direct execution of Section 3's conclusion
that GGUF is the right answer for CPU / edge deployment.

### [Bonus — Voice-based RAG Assistant](bonus_voice_rag/) *(composition demo)*
A layer on top of the four core sections to demonstrate **system-level
composition**. Combines Section 1's voice pipeline (Deepgram STT + Groq
LLM + Cartesia TTS) with Section 2's LangGraph RAG retrieval into a
single voice-based knowledge assistant ("Nova") — user speaks a
question, RAG retrieves grounded chunks, Nova speaks back the answer.
**Nothing is rewritten** — the entire bonus is under 150 lines of glue
that imports both sections as-is, which is the whole point: it's proof
that Sections 1 and 2 were written with clean enough module boundaries
to compose cleanly. The README also documents "Option B" — how to swap
the RAG generator to Section 4's local FastAPI model — as a narrative
connection between all four core sections.

---

## Repo layout

```
Basics-for-gen-AI/
├── README.md                              (this file)
├── section_1_livekit_voice_agent/         (Section 1)
│   ├── src/                               agent.py, agent_swap.py, tools.py, config.py
│   ├── transcripts/
│   ├── .env.example
│   ├── requirements.txt
│   └── README.md
├── section_2_langchain_rag/               (Section 2)
│   ├── docs/                              3 source markdowns
│   ├── src/                               config, ingest, graph, query, run_examples
│   ├── examples/qa_examples.md
│   ├── vectorstore/                       FAISS index (gitignored, built by ingest.py)
│   ├── requirements.txt
│   └── README.md
├── section_3_quantization/                (Section 3)
│   ├── quantization_tradeoff_colab.ipynb  ← the notebook
│   └── README.md
├── section_4_model_deployment/            (Section 4)
│   ├── src/                               app.py, model.py, config.py
│   ├── scripts/                           download_model.py, loadtest.py
│   ├── results/loadtest_results.md
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md
└── bonus_voice_rag/                       (Bonus — additional, not required)
    ├── src/                               _paths.py, rag_tool.py, agent.py
    └── README.md
```

---

## Running each section (fast reference)

Each section has its own conda env + `requirements.txt` so they don't
collide.

**Section 1 — voice agent (needs Groq / Deepgram / Cartesia API keys):**
```
conda create -n section1-livekit python=3.11 -y && conda activate section1-livekit
cd section_1_livekit_voice_agent
pip install -r requirements.txt
pip install livekit-plugins-openai
# fill in .env with GROQ_API_KEY, DEEPGRAM_API_KEY, CARTESIA_API_KEY
python src/agent.py console
```

**Section 2 — RAG (needs Gemini API key as `LLM_API_KEY`):**
```
conda create -n section2-rag python=3.11 -y && conda activate section2-rag
cd section_2_langchain_rag
pip install -r requirements.txt
# fill in .env with LLM_API_KEY=<gemini key>
python src/ingest.py
python src/query.py "What is self-attention?"
python src/run_examples.py     # writes examples/qa_examples.md
```

**Section 3 — quantization notebook (no local setup — Colab):**
Open `section_3_quantization/quantization_tradeoff_colab.ipynb` in Google
Colab, switch runtime to a T4 GPU, and run all cells top-to-bottom.

**Section 4 — deployment (Docker; no API keys):**
```
cd section_4_model_deployment
docker build -t section4-llm .            # ~5–10 min first time
docker run --rm -p 8000:8000 section4-llm
# Open http://localhost:8000/docs in a browser
# Or: python scripts/loadtest.py
```

**Bonus — voice-based RAG (reuses Sections 1 + 2):**
```
conda activate section1-livekit
pip install -r section_2_langchain_rag/requirements.txt      # one-time
python section_2_langchain_rag/src/ingest.py                 # one-time
python bonus_voice_rag/src/agent.py console
```

Full setup, run commands, endpoint docs and half-page write-ups live in the
per-section READMEs.

---

## Cross-section threads worth pointing out

- **Vendor decoupling.** Section 1's bonus (STT swap) and Section 2's config
  design (single-line LLM swap in `config.py`) both hit the same theme: an
  agent / RAG pipeline shouldn't be married to any one API provider.
- **Rate-limit reality.** During Section 1 I hit Gemini's free-tier RPM cap
  hard in a voice loop, swapped to Groq in a one-line change, everything
  else in the pipeline stayed identical. This is documented in the Section 1
  README as a real example of why the decoupling matters.
- **Section 3 → Section 4 handoff.** Section 3's write-up argues GGUF is
  the right answer for CPU / edge deployment. Section 4 is that argument
  put into a working `docker build && docker run`.

## Honest limitations

- **Section 1** was tested in `console` mode against local mic/speaker, not
  against LiveKit Cloud.
- **Section 2** uses a coarse FAISS L2 threshold rather than a proper
  cross-encoder re-ranker; the write-up explains what I'd change at scale.
- **Section 3** used Colab (no local GPU), which is called out both in the
  README and inside the notebook.
- **Section 4** serves a small 0.5B model on CPU. On this stack you cap at
  ~2–4 concurrent generations before latency degrades — the "50 concurrent
  users" write-up covers what would actually solve that (continuous batching
  on GPU, horizontal replicas, caching, queues).
