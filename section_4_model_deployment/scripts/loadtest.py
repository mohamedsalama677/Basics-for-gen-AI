"""Async load test: 10 concurrent streaming requests.

Measures per-request:
  - TTFT   : time from send to first token byte
  - total  : time until stream completes
  - tokens : count of SSE tokens received

Usage:
    python scripts/loadtest.py                        # against http://localhost:8000
    python scripts/loadtest.py http://host:port       # against a custom URL
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from dataclasses import dataclass

import httpx

DEFAULT_URL = "http://localhost:8000"
CONCURRENCY = 10

PROMPTS = [
    "Explain quantization in one sentence.",
    "What is FastAPI and why is it useful?",
    "Give me two reasons Docker is useful for ML deployment.",
]


@dataclass
class Result:
    idx: int
    prompt: str
    ttft_ms: float
    total_ms: float
    tokens: int
    ok: bool
    error: str = ""


async def one_request(client: httpx.AsyncClient, idx: int, url: str) -> Result:
    prompt = PROMPTS[idx % len(PROMPTS)]
    payload = {"prompt": prompt, "max_tokens": 64}
    t0 = time.perf_counter()
    first_token_time: float | None = None
    tokens = 0
    try:
        async with client.stream(
            "POST", f"{url}/generate/stream", json=payload, timeout=120.0
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                tokens += 1
        t_end = time.perf_counter()
        ttft = (first_token_time - t0) * 1000 if first_token_time else float("nan")
        total = (t_end - t0) * 1000
        return Result(idx, prompt, ttft, total, tokens, ok=True)
    except Exception as exc:
        t_end = time.perf_counter()
        return Result(
            idx, prompt, float("nan"), (t_end - t0) * 1000, tokens, ok=False, error=str(exc)
        )


async def main(url: str) -> None:
    print(f"Load test: {CONCURRENCY} concurrent streams -> {url}/generate/stream\n")

    async with httpx.AsyncClient(http2=False) as client:
        # sanity: check /health first
        try:
            r = await client.get(f"{url}/health", timeout=10.0)
            print(f"/health -> {r.status_code} {r.text}\n")
        except Exception as exc:
            print(f"/health failed: {exc}\nAbort.")
            return

        overall_t0 = time.perf_counter()
        results: list[Result] = await asyncio.gather(
            *(one_request(client, i, url) for i in range(CONCURRENCY))
        )
        overall_ms = (time.perf_counter() - overall_t0) * 1000

    print(f"{'idx':>3} {'ok':>3} {'ttft_ms':>10} {'total_ms':>10} {'tokens':>7}")
    print("-" * 40)
    for r in results:
        ok = "y" if r.ok else "N"
        print(
            f"{r.idx:>3} {ok:>3} {r.ttft_ms:>10.1f} {r.total_ms:>10.1f} {r.tokens:>7}"
        )

    ok_results = [r for r in results if r.ok]
    print()
    if ok_results:
        ttfts = [r.ttft_ms for r in ok_results]
        totals = [r.total_ms for r in ok_results]
        tokens = [r.tokens for r in ok_results]
        print(f"successful requests    : {len(ok_results)}/{len(results)}")
        print(f"TTFT   ms  mean/median/p95 : {statistics.mean(ttfts):.1f} / {statistics.median(ttfts):.1f} / {_p95(ttfts):.1f}")
        print(f"Total  ms  mean/median/p95 : {statistics.mean(totals):.1f} / {statistics.median(totals):.1f} / {_p95(totals):.1f}")
        print(f"mean tokens per request     : {statistics.mean(tokens):.1f}")
        total_tokens = sum(tokens)
        print(f"aggregate tokens/sec       : {total_tokens / (overall_ms / 1000):.1f}")
    print(f"wall clock for {CONCURRENCY} concurrent : {overall_ms:.0f} ms")

    for r in results:
        if not r.ok:
            print(f"\nERROR on idx {r.idx}: {r.error}")


def _p95(values: list[float]) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(0.95 * (len(s) - 1)))))
    return s[k]


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    asyncio.run(main(url.rstrip("/")))
