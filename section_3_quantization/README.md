# Section 3 — Quantization Trade-off

A hands-on comparison of running the **same** small open-weight LLM in two
precisions on the **same** GPU, over the **same** 5 fixed prompts:

- **fp16** — full-precision baseline
- **4-bit NF4** via `bitsandbytes` — the go-to quick-quantize path in Hugging
  Face `transformers`

All work lives in
[`quantization_tradeoff_colab.ipynb`](quantization_tradeoff_colab.ipynb).
The notebook is self-contained: open it in Google Colab, switch the runtime
to a **T4 GPU** (free tier is enough), and run top-to-bottom.

## Why a Colab notebook and not a script

The comparison needs a real CUDA GPU — `bitsandbytes` 4-bit inference is
CUDA-only. I don't have a local GPU, so the notebook targets a free Colab
T4. Reviewers who want to reproduce the numbers just open the notebook, hit
"Run all", and get their own results in ~5 minutes. Everything the graded
write-up refers to (memory, throughput, quality) is produced by cells inside
the notebook and printed inline.

## Model and setup

- **Model:** `Qwen/Qwen2.5-1.5B-Instruct` — ungated (no HF login required),
  ~1.5B parameters, fits both fp16 and 4-bit on a T4 with room to spare.
- **Prompts:** 5 fixed prompts that stress different capabilities — a
  conceptual explainer, a factual recall question, a code-generation task,
  a small reasoning puzzle, and a one-sentence summary.
- **Decoding:** greedy (`do_sample=False`), `max_new_tokens=128`. Greedy is
  chosen deliberately so the two runs are directly comparable — any output
  differences are attributable to precision, not to sampling noise.
- **Both runs share the tokenizer, prompt list, and generation settings.**
  Only the model precision changes.

## Numbers from my run (Colab T4)

| Config                     | Weights VRAM | Peak VRAM | Throughput | Load time |
| -------------------------- | ------------ | --------- | ---------- | --------- |
| fp16 (baseline)            | 3.09 GB      | 3.11 GB   | 16.3 tok/s | 39.4 s    |
| 4-bit NF4 (bitsandbytes)   | 1.16 GB      | 1.22 GB   | 10.0 tok/s | 21.7 s    |

- **Memory saving from 4-bit:** ~62% smaller weights.
- **Speed ratio (4-bit / fp16):** 0.61× — slower, because bitsandbytes
  dequantizes on the fly during every forward pass rather than shipping
  pre-packed weights with fused kernels.

Numbers vary run-to-run and by which GPU Colab assigns. Re-run for stable
values; the *ratios* are what matter.

## Quality — same 5 prompts, both versions

The notebook prints the fp16 and 4-bit outputs side by side for each prompt.
On this model and this prompt set:

- Prompts 1, 4 (reasoning) and 5 (summary) — the two versions were
  essentially equivalent in content, with minor wording differences.
- Prompt 2 (Australia capital + landmarks) — both correctly named Canberra;
  fp16 chose Sydney Opera House and the Great Barrier Reef as landmarks,
  the 4-bit version chose two Canberra landmarks (arguably a better literal
  answer). Not a regression, just a different answer.
- Prompt 3 (Fibonacci) — both produced valid code, using slightly different
  loop styles. Both ran out of `max_new_tokens` before finishing.

At Qwen-1.5B scale, NF4 through bitsandbytes keeps most of the perceived
quality. Degradation shows up more sharply on smaller models or at more
aggressive quantizations (e.g. Q3_K); this comparison intentionally sits at
a "sensible default" that mirrors real practice.

## Write-up — bitsandbytes vs. GPTQ / AWQ vs. GGUF

**bitsandbytes** (what the notebook uses). Its strength is convenience:
quantization happens *at load time* with no calibration data and no offline
step, and the same 4-bit weights can be fine-tuned via QLoRA. That makes it
my default for **experiments, research, and any workflow that touches
training / fine-tuning**. The cost is inference speed — dequantization
happens on the fly during every forward pass, so raw throughput is lower
than formats that ship pre-packed weights with fused kernels. It is also
tied to a GPU plus the PyTorch / `transformers` stack.

**GPTQ / AWQ.** These are *post-training* quantization methods that use a
small calibration dataset to quantize once, offline, into weights optimized
for fast inference kernels (ExLlama, Marlin). I'd pick them for **GPU-served
production** where you quantize once and then serve millions of requests —
the one-time calibration cost amortizes, and you get meaningfully higher
tokens/sec and better batching than bitsandbytes. Between the two, **AWQ**
tends to preserve quality slightly better at 4-bit because it protects the
most salient (activation-aware) weight channels, so I lean AWQ when quality
is sensitive. GPTQ is a fine, widely-supported alternative with strong
tooling.

**GGUF (llama.cpp / Ollama).** GGUF is the answer when the deployment
target is **not a datacenter GPU**: CPU-only servers, laptops, Apple
Silicon, or edge devices. Its k-quant variants (Q4_K_M, Q5_K_S, etc.) let
you trade size for quality flexibly, and the llama.cpp runtime runs almost
anywhere with a small memory footprint and no Python required. Raw GPU
throughput is lower than AWQ/GPTQ on a proper GPU, but for **local /
desktop apps, offline use, or "no GPU available"** it's unbeatable for
portability — which is exactly why **Section 4 of this same submission
serves a GGUF model in a CPU-only Docker container**.

**Decision in one line:** fine-tuning or quick experiments →
**bitsandbytes**; high-throughput GPU serving, quantize-once → **AWQ**
(or GPTQ); CPU / edge / local / maximum portability → **GGUF**.

## Assumptions and limitations

- **Model choice.** Qwen2.5-1.5B fits comfortably in both configs on a free
  T4. Larger models (Mistral-7B) would show a much bigger absolute memory
  gap but wouldn't both fit on free-tier hardware.
- **Throughput is single-request, greedy, batch size 1.** It measures
  per-request latency, not server throughput with batching. Real serving
  numbers would be much higher with continuous batching.
- **Quality is a 5-prompt spot check**, not a benchmark like MMLU. A
  rigorous eval would use a proper harness and dozens to hundreds of
  prompts.
- **VRAM** is measured with `torch.cuda.memory_allocated` (PyTorch's view),
  which can differ slightly from `nvidia-smi` because of the CUDA caching
  allocator.
- **No local GPU.** Everything was run on Colab. This is honest and stated
  in the notebook — anyone with a local NVIDIA GPU can reproduce by cloning
  the notebook.

## Handoff to Section 4

Section 3 argues that **GGUF is the right answer for CPU / edge
deployment**. Section 4 puts that recommendation into practice by serving
`Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` behind a FastAPI + Docker service that
runs on any machine, GPU or not. See
[`../section_4_model_deployment/README.md`](../section_4_model_deployment/README.md).
