"""Wraps Section 2's LangGraph RAG pipeline as a LiveKit @function_tool.

The tool is what lets a Section 1-style voice agent call into the RAG graph
mid-conversation. The Section 2 code is imported verbatim — see
`section_2_langchain_rag/src/graph.py`. Nothing is duplicated here.
"""

import asyncio
import logging

import _paths  # noqa: F401  — sets up sys.path before the Section 2 import below

from graph import build_graph  # ← Section 2's LangGraph builder
from livekit.agents import function_tool

log = logging.getLogger("bonus.rag_tool")

_graph = None  # lazy-loaded on first invocation


def _get_graph():
    global _graph
    if _graph is None:
        log.info("Compiling Section 2 RAG graph (first tool call)")
        _graph = build_graph()
    return _graph


@function_tool
async def answer_from_knowledge_base(question: str) -> str:
    """Search the knowledge base for information about LLMs and transformers,
    RAG vs fine-tuning, or agentic AI. Use this tool whenever the user asks
    a substantive or conceptual question on any of those topics.

    Args:
        question: the user's question, in natural language.

    Returns:
        A grounded answer synthesized from the knowledge base. If no
        relevant context is found, returns a polite refusal string that
        the voice agent should read back as-is.
    """
    graph = _get_graph()
    # LangGraph is synchronous; run it in a worker thread so the LiveKit
    # event loop isn't blocked during retrieval + generation.
    result = await asyncio.to_thread(graph.invoke, {"question": question})

    citations = result.get("citations") or []
    if citations:
        cite_strs = [f"{c['source']}#chunk_{c['chunk_id']}" for c in citations]
        log.info("RAG grounded on: %s", ", ".join(cite_strs))
    else:
        log.info("RAG returned no citations (out-of-KB question)")

    return result["answer"]
