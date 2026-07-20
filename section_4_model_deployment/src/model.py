"""Thin wrapper around llama_cpp.Llama.

Loads the GGUF weights once at startup and exposes both a blocking generate()
and an async streaming generator. Streaming uses asyncio.to_thread because
llama_cpp.Llama is synchronous — we don't want it blocking the event loop.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncIterator

from llama_cpp import Llama

from config import CTX_SIZE, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, MODEL_PATH, N_THREADS


class LlamaService:
    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Run: python scripts/download_model.py"
            )
        self._llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=CTX_SIZE,
            n_threads=N_THREADS,
            verbose=False,
        )
        # llama_cpp.Llama is not thread-safe for concurrent generations.
        # Serialize with a lock so overlapping requests queue cleanly.
        self._lock = threading.Lock()

    def generate(
        self,
        prompt: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> tuple[str, int]:
        with self._lock:
            resp = self._llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )
        text = resp["choices"][0]["message"]["content"]
        usage_tokens = resp.get("usage", {}).get("completion_tokens", 0)
        return text, int(usage_tokens)

    async def stream(
        self,
        prompt: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> AsyncIterator[str]:
        """Yield token chunks as they are produced.

        The llama_cpp streaming iterator is synchronous. We run it in a worker
        thread and push chunks into a queue that the async side drains.
        """
        chunk_queue: queue.Queue[str | None] = queue.Queue()

        def _producer() -> None:
            try:
                with self._lock:
                    stream = self._llm.create_chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=True,
                    )
                    for chunk in stream:
                        delta = chunk["choices"][0].get("delta", {})
                        token = delta.get("content")
                        if token:
                            chunk_queue.put(token)
            finally:
                chunk_queue.put(None)  # sentinel = end of stream

        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, _producer)

        while True:
            token = await asyncio.to_thread(chunk_queue.get)
            if token is None:
                return
            yield token
