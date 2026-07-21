"""Wraps Section 2's LangGraph RAG pipeline as a LiveKit @function_tool.

The tool is what lets a Section 1-style voice agent call into the RAG graph
mid-conversation. The Section 2 code is imported verbatim — see
`section_2_langchain_rag/src/graph.py`. Nothing is duplicated here.
"""

import asyncio
import logging
import re

import _paths  # noqa: F401  — sets up sys.path before the Section 2 import below

from graph import build_graph  # ← Section 2's LangGraph builder
from livekit.agents import function_tool

log = logging.getLogger("bonus.rag_tool")

_graph = None  # lazy-loaded on first invocation

_CITATION_RE = re.compile(r"\s*\[[^\]\s]+\.md#chunk_\d+\]")
_VOICE_MAX_CHARS = 500  # ~30 seconds of speech; long enough to be useful, short enough not to bore


def _get_graph():
    global _graph
    if _graph is None:
        log.info("Compiling Section 2 RAG graph (first tool call)")
        _graph = build_graph()
    return _graph


def _voice_polish(text: str) -> str:
    """Make the RAG answer friendlier for speech.

    Section 2's generator writes text-optimized answers with inline
    `[source.md#chunk_N]` citations. Read aloud those markers are jarring,
    and very long answers push the shell LLM to 'summarize' the tool
    result away. This helper strips markers and softly truncates.
    """
    stripped = _CITATION_RE.sub("", text).strip()
    if len(stripped) <= _VOICE_MAX_CHARS:
        return stripped

    # Truncate at the last sentence boundary within the limit.
    cut = stripped[:_VOICE_MAX_CHARS]
    boundary = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if boundary > 0:
        cut = cut[: boundary + 1]
    return cut + " Would you like more detail?"


@function_tool
async def answer_from_knowledge_base(question: str) -> str:
    """Search the knowledge base for information about LLMs and transformers,
    RAG vs fine-tuning, or agentic AI. Use this tool whenever the user asks
    a substantive or conceptual question on any of those topics.

    Call this tool AT MOST ONCE per user turn. The retrieval covers the
    full knowledge base in a single call — issuing multiple calls with
    refined queries returns overlapping chunks and only wastes rate
    limits. Trust the first result and speak it back to the user.

    Args:
        question: the user's question, in natural language. Pass it
            through roughly as the user asked it. Do not stuff it with
            extra topics or your own re-interpretations.

    Returns:
        A grounded answer synthesized from the knowledge base, cleaned up
        for speech (citation markers stripped, capped at a comfortable
        spoken length). If no relevant context is found, returns a polite
        refusal string that the voice agent should read back as-is.
    """
    graph = _get_graph()
    # LangGraph is synchronous; run it in a worker thread so the LiveKit
    # event loop isn't blocked during retrieval + generation.
    result = await asyncio.to_thread(graph.invoke, {"question": question})

    _log_retrieval(question, result)

    return _voice_polish(result["answer"])


def _log_retrieval(question: str, result: dict) -> None:
    """Print the retrieved chunks so a reviewer can see exactly what the
    RAG returned for a spoken question. Uses print() (not log.info) so it
    always shows up next to the LiveKit event stream, formatted for easy
    reading in the console."""
    scored = result.get("scored_chunks") or []
    best_score = result.get("best_score")
    has_context = result.get("has_context")

    print("\n" + "=" * 78)
    print(f"[RAG] question:      {question}")
    if best_score is not None:
        print(f"[RAG] best_score:    {best_score:.4f}  (threshold gate: has_context={has_context})")
    if not scored:
        print("[RAG] no chunks retrieved.")
        print("=" * 78 + "\n")
        return

    print(f"[RAG] {len(scored)} chunks retrieved:")
    for i, (doc, score) in enumerate(scored, start=1):
        src = doc.metadata.get("source", "?")
        cid = doc.metadata.get("chunk_id", "?")
        preview = doc.page_content.strip().replace("\n", " ")
        if len(preview) > 280:
            preview = preview[:280] + "..."
        print(f"  {i}. [{src}#chunk_{cid}]  score={score:.4f}")
        print(f"     {preview}")
    print("=" * 78 + "\n")
