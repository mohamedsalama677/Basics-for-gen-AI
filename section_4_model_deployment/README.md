# Section 4 — Model Deployment

A small open-weight LLM served behind a REST API with a streaming endpoint,
containerized for CPU-only deployment. Ships with a load-test script that
fires 10 concurrent streaming requests and reports TTFT / total latency.

## Stack and why

- **Model:** `Qwen2.5-0.5B-Instruct` in GGUF format (Q4_K_M quantization).
  Same family as Section 3 (which used Qwen2.5-1.5B on GPU). I downsized
  from 1.5B → 0.5B because CPU inference on the larger model makes a
  10-concurrent load test take several minutes; 0.5B keeps the whole demo
  under a minute while proving the same architecture works.
- **Inference:** `llama-cpp-python` (llama.cpp bindings). This is exactly
  the CPU / edge deployment path recommended in Section 3's write-up —
  Section 4 is that recommendation put into practice.
- **API:** FastAPI + uvicorn. vLLM and TGI are GPU-first and heavier to
  containerize; the assessment allows FastAPI and it lets me demonstrate
  async + streaming plainly.
- **Container:** `python:3.11-slim`, CPU-only. Model is baked in at build
  time so `docker run` needs no network. Works on any machine with Docker.

## Endpoints

- `GET  /health` — `{"status": "ok", "model": "..."}`. Used by Docker HEALTHCHECK.
- `POST /generate` — blocking; returns JSON `{text, tokens, latency_ms}`.
- `POST /generate/stream` — SSE stream of tokens; `data: <token>\n\n` frames
  ending with `data: [DONE]\n\n`.
- `GET  /docs` — auto-generated Swagger UI (free from FastAPI).

## Run locally (no Docker)

```
cd section_4_model_deployment
pip install -r requirements.txt
python scripts/download_model.py         # ~350MB, one-time
uvicorn app:app --app-dir src --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000/docs** in a browser — FastAPI's built-in
Swagger UI lets you try every endpoint with a click. Or from the CLI:

```
curl http://localhost:8000/health

curl -N -X POST http://localhost:8000/generate/stream \
     -H "Content-Type: application/json" \
     -d "{\"prompt\":\"Explain quantization in one sentence.\",\"max_tokens\":64}"
```

> **Browser tip:** always use `http://localhost:8000` or `http://127.0.0.1:8000`.
> `0.0.0.0` is a bind-address used by uvicorn internally — browsers can't route
> to it and will show `ERR_ADDRESS_INVALID`.

## Run with Docker

```
cd section_4_model_deployment
docker build -t section4-llm .           # ~5-10 min first time (model download + wheels)
docker run --rm -p 8000:8000 section4-llm
```

Container startup takes a few seconds to load the GGUF weights into RAM;
watch for `Uvicorn running on http://0.0.0.0:8000` in the log. Then hit the
same URLs — again from the **browser** at http://localhost:8000/docs, or from
the CLI:

```
curl http://localhost:8000/health
```

## Load test

With the service running on port 8000:

```
python scripts/loadtest.py
# or against a custom host:
python scripts/loadtest.py http://localhost:8000
```

Fires 10 concurrent streaming requests and prints a per-request table plus
aggregate TTFT / total-latency / tokens-per-second statistics. See
[`results/loadtest_results.md`](results/loadtest_results.md) for a captured
run.

## Architecture at a glance

```
       Client
         │
         │ POST /generate/stream
         ▼
    FastAPI (uvicorn, 8000)
         │
         ▼
    LlamaService.stream()      ── async wrapper (asyncio.to_thread + queue)
         │
         ▼
    llama_cpp.Llama            ── loaded once at startup, single-threaded,
         │                        lock-protected for concurrent requests
         ▼
    SSE tokens back to client
```

## Write-up — scaling to 50 concurrent users

The current setup handles ~2–4 concurrent generations on CPU before
per-request latency degrades badly: `llama-cpp-python` runs generations
single-threaded and a lock serializes overlapping requests. To serve 50
concurrent users reliably I would change several things, roughly in order
of impact.

The biggest single win is **continuous batching on a GPU** using **vLLM**
or **TGI**. Both use PagedAttention to interleave dozens of active
generations token by token on the same GPU, so throughput scales with
batch size instead of collapsing. This would replace the entire `model.py`
backend — the FastAPI wrapper stays the same because vLLM exposes its own
OpenAI-compatible server that we could either proxy or embed. On a single
A10G or L4 GPU this comfortably serves 50+ concurrent users for a 0.5B–7B
model.

Next, **horizontal replicas behind a load balancer**. The service is
stateless (no per-user memory in this design), so `n` replicas of the
container behind nginx, envoy, or a Kubernetes Service is trivial.
Combined with **HPA-style autoscaling** on either CPU/GPU utilization or a
custom queue-depth metric, replicas can flex from 2 to N during traffic
spikes.

For burst absorption I would add a **request queue with back-pressure** —
Redis or RabbitMQ in front of the workers. The API returns `202 Accepted`
with a job id, and the client either polls or subscribes to a WebSocket
for the result. Without this, a burst of 50 simultaneous connections
takes the sync path down.

**Response caching** is often the cheapest win in practice. A Redis cache
keyed on `hash(prompt + params)` catches exact repeats — common in
customer-support flows. A **semantic cache** (embed the prompt, look up
against a small vector index with a similarity threshold) catches
near-duplicates and can serve a surprisingly high fraction of production
traffic instantly.

**Rate limiting per user** at the gateway (nginx `limit_req`, envoy rate
limit, or a proper API gateway) prevents any one client from starving the
others — a real risk when generation is expensive.

For **observability** I would export Prometheus metrics on TTFT, tokens
per second, in-flight requests, and queue depth; ship structured JSON
logs; and dashboard the SLO burn rate in Grafana so oncall sees latency
degradation before users do.

Finally, a pragmatic safety valve: **spillover to a hosted API**
(OpenAI, Groq, Gemini) behind a feature flag. When our own capacity is
saturated, we degrade to a paid API rather than dropping requests. Users
still get answered; we bleed a bit of margin instead of trust.

Honest note on the current stack's ceiling: none of the above are tweaks
you'd bolt onto CPU `llama-cpp-python`. Serving 50 users seriously means
moving to GPU + continuous batching, and the FastAPI/Docker structure
here is the right skeleton to swap the backend into.

## Known limitations

- **CPU throughput ceiling.** ~30 tok/s for a single request on modern
  CPUs, degrading to a few tok/s under concurrent load because generations
  serialize.
- **First `docker build` is slow.** Downloads the 350MB GGUF model and
  compiles or fetches `llama-cpp-python` wheels — expect 5–10 minutes on a
  cold cache. Subsequent builds are fast.
- **Model baked in.** Image is ~700MB. In a real deployment you'd mount
  the model from a volume or object store and keep the image slim.
- **No auth, no rate limiting.** This is a demo service — production
  would need API keys and per-user quotas at the gateway.
- **SSE format only.** The streaming endpoint uses SSE, not raw newline-
  delimited chunks. It's easy to add an alternate content-type if a
  reviewer needs one.
