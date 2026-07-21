# Bonus — Voice-based RAG Assistant

> **Composition layer on top of Sections 1 and 2.**
> The four core sections (1–4) are complete on their own. This folder
> combines Section 1's voice pipeline with Section 2's RAG graph into a
> single voice-based knowledge assistant to demonstrate system-level
> composition, and to show that the earlier sections were written with
> clean enough module boundaries that they compose cleanly.

## Why it exists

Individually, Section 1 (voice) and Section 2 (RAG) each prove one
competency. Together they demonstrate *system thinking* — how those pieces
snap into a real product surface (a voice interface for asking questions
against a knowledge base). That composition is the actual output companies
buy; the individual sections are the ingredients. This folder shows both.

Hard constraint: **nothing from Sections 1 or 2 was rewritten.** Everything
here is imports. If the reuse story doesn't hold, the bonus loses its
point.

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐
│ User speaks │───▶│ STT         │───▶│ Question text            │
│ (audio in)  │    │ Deepgram    │    │ "What is self-attention?"│
└─────────────┘    └─────────────┘    └────────────┬────────────┘
                                                    │
                                                    ▼
┌─────────────────────────┐   Chunks    ┌─────────────────────────┐
│  LLM                     │◀────────────│  RAG retrieval           │
│  Groq Llama-3.3-70b      │             │  FAISS top-4 chunks      │
│  (question + chunks in)  │             │  (Section 2 graph)       │
└────────────┬────────────┘             └─────────────────────────┘
             │
             ▼
┌─────────────────────────┐    ┌─────────────┐    ┌─────────────┐
│ Answer text              │───▶│ TTS         │───▶│ User hears  │
│ (grounded on chunks)     │    │ Cartesia    │    │ (audio out) │
└────────────┬────────────┘    └─────────────┘    └─────────────┘
             │  (optional swap, documented below)
             │  Option A: hosted API (Gemini) ─── DEFAULT
             │  Option B: local LLM from Section 4 (FastAPI + GGUF)
             ▼
       (config-level choice)
```

Two LLMs are in play, on purpose:

- **The conversational LLM** — Groq's Llama-3.3-70b, running in the
  LiveKit `AgentSession`. It handles greetings, small talk, decides
  whether to call the RAG tool, and phrases the final answer. Reused
  from Section 1's `agent.py`.
- **The grounded generator** — Gemini 2.5 Flash, running inside the
  Section 2 LangGraph. It never sees the user's raw voice; it only ever
  sees the retrieved chunks + the question the conversational LLM
  extracted.

This "cheap conversational glue on top, careful grounded generator
underneath" split is the same pattern real voice-RAG products use in
production. It also cleanly delimits which LLM is allowed to hallucinate
(the shell) and which one is structurally forbidden (the grounded one).

## How the reuse actually works

Three tiny files in [`src/`](src/):

- [`src/_paths.py`](src/_paths.py) — adds Section 2's `src/` to `sys.path`
  and loads Section 1's `config.py` under a unique name to sidestep a
  collision (both sections happen to have a file called `config.py`).
- [`src/rag_tool.py`](src/rag_tool.py) — one function decorated with
  `@function_tool`. Internally calls
  [`section_2_langchain_rag/src/graph.py`](../section_2_langchain_rag/src/graph.py)'s
  `build_graph()`, invokes the compiled graph on the user's question, and
  returns the answer string.
- [`src/agent.py`](src/agent.py) — a new `KnowledgeAssistant(Agent)` with
  a knowledge-base persona; the `AgentSession` uses the *exact same*
  Deepgram STT, Groq LLM, Cartesia TTS, and silero VAD as
  [`section_1_livekit_voice_agent/src/agent.py`](../section_1_livekit_voice_agent/src/agent.py),
  wired via `s1_config`.

Together those three files are under 150 lines. Everything else — the
LangGraph, the FAISS index, the STT/TTS pipeline, the tool dispatch — is
imported from the existing sections. That is the point.

## Prerequisites

1. **Section 2's FAISS index must exist.** Build it once from the repo
   root:
   ```
   python section_2_langchain_rag/src/ingest.py
   ```
2. **Section 1's `.env`** at `section_1_livekit_voice_agent/.env` must
   contain `GROQ_API_KEY`, `DEEPGRAM_API_KEY`, `CARTESIA_API_KEY`.
3. **Section 2's `.env`** at `section_2_langchain_rag/.env` must contain
   `LLM_API_KEY=<your Gemini key>`.

Missing pieces raise a clear error at import time.

## Setup and run

Reuse Section 1's conda env, add Section 2's dependencies on top:

```
conda activate section1-livekit
pip install -r section_2_langchain_rag/requirements.txt      # one-time
python section_2_langchain_rag/src/ingest.py                 # one-time (builds FAISS)
```

Then run the voice-RAG agent in LiveKit console mode:

```
python bonus_voice_rag/src/agent.py console
```

Speak into your mic. Try:

- *"What is self-attention?"* — should trigger a tool call and speak back
  a grounded answer from `01_llms_and_transformers.md`.
- *"When would you pick RAG over fine-tuning?"* — same, grounded on
  `02_rag_vs_finetuning.md`.
- *"What's the weather in Cairo today?"* — the RAG graph's refusal branch
  fires (no relevant chunks); Nova politely says she doesn't have that
  info. No hallucination.
- *"Hi, who are you?"* — the conversational LLM answers directly, no
  tool call.

Watch the console log for `[tool-call] answer_from_knowledge_base(...)`
lines — those are proof the RAG graph is being invoked mid-conversation.

## Option B — swap in the Section-4 local LLM as the RAG generator

The default RAG generator is Gemini (Option A). To point it at Section 4's
local Qwen-0.5B FastAPI service instead — closing the loop on all four
sections — three changes are needed. I chose not to implement this in the
current build to keep scope tight, but the design is straightforward:

1. **Start Section 4's container** in a separate terminal:
   ```
   docker run --rm -p 8000:8000 section4-llm
   ```
2. **Write a small LangChain LLM adapter** for Section 4's `/generate`
   endpoint (it isn't OpenAI-compatible out of the box). Something like:
   ```python
   # section_2_langchain_rag/src/section4_llm.py  (new file)
   from langchain_core.language_models.chat_models import BaseChatModel
   import httpx

   class Section4LLM(BaseChatModel):
       base_url: str = "http://localhost:8000"

       def _generate(self, messages, stop=None, **kw):
           prompt = "\n".join(m.content for m in messages)
           r = httpx.post(f"{self.base_url}/generate",
                          json={"prompt": prompt, "max_tokens": 256},
                          timeout=60)
           text = r.json()["text"]
           # wrap in ChatGeneration + return ChatResult (~10 lines)
           ...
   ```
3. **Branch in `graph.py`** on an env var like `RAG_LLM_BACKEND`:
   ```python
   if os.getenv("RAG_LLM_BACKEND") == "local":
       from section4_llm import Section4LLM
       llm = Section4LLM()
   else:
       llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2)
   ```

Same voice front-end, same RAG graph shape, same tool call — but the
grounded generator now runs entirely on the user's laptop through the
Section 4 Docker container. That would be a compact demonstration that all
four sections were designed to fit together.

## Known limitations

- **Dependency graph is tighter here.** This folder assumes both Section 1
  and Section 2 exist and their `.env` files are populated. Missing pieces
  raise clear errors on startup, but the coupling is real by design.
- **Citations are dropped from the spoken output.** They're logged to the
  console so grounding can be verified, but reading
  "`01_llms_and_transformers.md#chunk_3`" aloud would be jarring. A
  production version would summarize sources ("*from our LLM docs*")
  instead.
- **`ingest.py` must run first.** The FAISS index isn't committed; it's
  built on-demand from `section_2_langchain_rag/docs/`.
- **Console mode only.** Same limitation as Section 1 — not tested against
  LiveKit Cloud in this project.
- **Two hosted APIs, one voice conversation.** Groq for the conversational
  LLM, Gemini for the grounded generator. If either is rate-limited, the
  session degrades. Option B (Section-4 local LLM) removes the second
  hosted dependency and would be the right production choice.
