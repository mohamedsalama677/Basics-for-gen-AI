# Section 2 — LangChain RAG Pipeline

A minimal RAG pipeline built with **LangGraph** over a three-document
knowledge base (LLMs & transformers, RAG vs fine-tuning, agentic AI).
The graph explicitly routes off-topic questions to a refusal node so the
model never hallucinates an answer when the retriever comes up empty.

## Stack

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local, free)
- **Vector store:** FAISS (local disk)
- **LLM:** Groq Llama-3.3-70b (preferred) *or* Google Gemini 2.5 Flash, auto-detected
- **Orchestration:** LangGraph

## Setup

```
conda activate section2-rag
```

The `.env` file at this section's root must contain **one** of these keys.
The code prefers Groq if both are present, because Groq's free tier is
much more generous than Gemini's free tier (Gemini 2.5 Flash caps at 5
RPM / 20 requests per day, which is easily exhausted during voice-loop
testing).

```
GROQ_API_KEY=<your Groq key>     # preferred — https://console.groq.com/
# or, fallback:
LLM_API_KEY=<your Gemini key>    # legacy — https://aistudio.google.com/apikey
```

If you use Groq, install its adapter once:

```
pip install langchain-openai
```

Provider selection is centralised in `src/config.py` — swap models or
providers by editing the `LLM_PROVIDER` / `LLM_MODEL` block there.

## Run

```
python src/ingest.py                                     # once, or after docs change
python src/query.py "What is self-attention?"            # ad-hoc query
python src/run_examples.py                               # writes examples/qa_examples.md
```

## Architecture

```
       START
         |
         v
     retrieve            (FAISS top-4 with L2 scores)
         |
         v
  grade_relevance        (best_score vs. RELEVANCE_THRESHOLD)
      /       \
 has_context   no_context
     |             |
     v             v
generate_with_ refuse_no_
  context       context
     |             |
     +---> END <---+
```

- `retrieve` — embeds the question, pulls the top-4 chunks and their L2
  distances.
- `grade_relevance` — deterministic gate: if the best chunk's FAISS L2
  distance is above `RELEVANCE_THRESHOLD` (default 1.2 in `config.py`), mark
  the query out-of-scope. Earlier iterations layered an LLM-as-judge yes/no
  on top of the threshold, but on this corpus the threshold alone cleanly
  separates in-KB (~0.6–0.9) from out-of-KB (~1.8+), so the extra LLM call
  was removed to keep the gate cheap and deterministic. See the write-up
  below for how I'd re-introduce a proper re-ranker at scale.
- `generate_with_context` — prompts Gemini with the retrieved passages and
  requires it to cite each claim inline as
  `[source.md#chunk_N]`.
- `refuse_no_context` — deterministic string, no LLM call, no
  hallucination surface.

## Example outputs

See [`examples/qa_examples.md`](examples/qa_examples.md) for the three
required Q&As (two in-KB, one out-of-KB).

## Write-up — how I'd improve retrieval on longer documents

The current setup is intentionally simple: fixed-size character chunks of
~800 characters with 120 overlap, a single dense retriever, and no
re-ranking. It works well for the three short markdown docs in `docs/`,
but I would change several things if answer quality dropped on longer
documents.

The first move would be **semantic chunking** instead of fixed-size
splits. Recursive character splitting cuts on paragraph boundaries when
it can, but on documents with dense prose or code it still slices ideas
in half. Semantic chunking groups sentences into chunks based on
embedding-similarity boundaries, which keeps a coherent argument
together. Alternatively, for structured documents with clear headings, a
markdown-header-aware splitter that respects H2/H3 as boundaries gives
similar gains for less cost. I would also increase overlap to around 200
characters on longer prose, so context near the boundaries isn't lost.

The second and often bigger win is **hybrid search**. Dense embeddings
are strong on paraphrase but weak on rare tokens — exact names, error
codes, product SKUs, or acronyms — because those tokens carry almost no
semantic signal in a shared vector space. Combining FAISS with a BM25
sparse retriever (LangChain's `EnsembleRetriever` will do this in a few
lines) recovers the keyword strengths of classical search while keeping
the paraphrase strengths of embeddings. Weighting is domain-dependent; a
70/30 dense/sparse split is a reasonable starting point.

The third change would be a **re-ranker** on top. After retrieval, I
would take the top-20 candidates and re-rank them with a cross-encoder
such as `BAAI/bge-reranker-large` or Cohere's Rerank API. Cross-encoders
see the query and candidate together and produce far more accurate
relevance scores than bi-encoders like the one used for retrieval, at
the cost of running the model 20 times per query. For a chatbot with
human-in-the-loop latency budgets, this cost is easily worth the
precision gain. Only the top-4 after re-ranking make it into the
generation prompt.

Beyond those three, I would consider **query rewriting** — a small LLM
call that expands the user's question into 2–3 paraphrases or extracts
sub-questions, then merges retrievals. This helps for questions that are
poorly phrased or that require multiple pieces of evidence. On very long
documents, adding a **parent-document retriever** pattern is often
worthwhile: index small child chunks for precise recall, but return the
parent chunk (or a larger surrounding window) to the LLM so it has enough
context to reason with.

Finally, I would set up an **evaluation harness** before tuning any of
this. Without a fixed test set of question/answer/expected-source
triples, tuning chunk size or hybrid weights becomes vibes-based. Even a
20-question hand-curated set that gets scored on retrieval recall @ k and
faithfulness of the generated answer is enough to make principled
choices.

## Known limitations

- **First-run cost.** `HuggingFaceEmbeddings` downloads
  `all-MiniLM-L6-v2` (~90MB) on first use, cached under
  `~/.cache/huggingface`. Subsequent runs are instant.
- **L2 threshold is coarse.** `RELEVANCE_THRESHOLD` in `config.py` was
  chosen by eyeballing; a re-ranker would replace this with a proper
  score.
- **Single-turn.** No conversational memory — each query is
  independent. Adding a chat-history-aware retriever is a follow-up.
- **`langchain_community.embeddings` deprecation warning.** The newer
  `langchain-huggingface` package is not installed to keep the dep list
  short. The deprecated import still works.
- **Gemini free tier rate limits** may throttle `run_examples.py`
  during rapid iteration; retry after a minute if you see a 429.
