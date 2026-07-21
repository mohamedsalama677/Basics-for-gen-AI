"""The RAG LangGraph.

    START -> retrieve -> grade_relevance -> (generate_with_context | refuse_no_context) -> END

Load the FAISS index once at module scope. Nodes are pure state transformers.
"""

import os
from functools import lru_cache
from typing import TypedDict

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from config import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_PROVIDER,
    REFUSAL_MESSAGE,
    RELEVANCE_THRESHOLD,
    TOP_K,
    VECTORSTORE_DIR,
)


class RAGState(TypedDict, total=False):
    question: str
    retrieved_chunks: list[Document]
    scored_chunks: list[tuple[Document, float]]
    best_score: float
    has_context: bool
    answer: str
    citations: list[dict]


@lru_cache(maxsize=1)
def _get_vectorstore() -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


@lru_cache(maxsize=1)
def _get_llm() -> BaseChatModel:
    if LLM_PROVIDER == "groq":
        # Groq exposes an OpenAI-compatible endpoint — use the openai package
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=LLM_MODEL,
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"],
            temperature=0.2,
        )
    # Gemini fallback (works if LLM_API_KEY is set in .env instead of GROQ_API_KEY)
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2)


# ---------- nodes ----------

def retrieve(state: RAGState) -> RAGState:
    store = _get_vectorstore()
    scored = store.similarity_search_with_score(state["question"], k=TOP_K)
    chunks = [doc for doc, _ in scored]
    best = min((score for _, score in scored), default=float("inf"))
    return {
        "retrieved_chunks": chunks,
        "scored_chunks": scored,
        "best_score": float(best),
    }


def grade_relevance(state: RAGState) -> RAGState:
    score = state.get("best_score", float("inf"))
    print(f"[DEBUG grade_relevance] best_score={score:.4f}, threshold={RELEVANCE_THRESHOLD}")
    has_ctx = score <= RELEVANCE_THRESHOLD
    print(f"[DEBUG grade_relevance] has_context={has_ctx}")
    return {"has_context": has_ctx}


def generate_with_context(state: RAGState) -> RAGState:
    context_lines = []
    citations = []
    for chunk in state["retrieved_chunks"]:
        source = chunk.metadata.get("source", "unknown")
        chunk_id = chunk.metadata.get("chunk_id", "?")
        marker = f"{source}#chunk_{chunk_id}"
        context_lines.append(f"[{marker}]\n{chunk.page_content}")
        citations.append(
            {
                "source": source,
                "chunk_id": chunk_id,
                "snippet": chunk.page_content[:180].replace("\n", " "),
            }
        )
    context_block = "\n\n---\n\n".join(context_lines)

    prompt = (
        "You are a helpful assistant. Answer the question using ONLY the "
        "passages provided below. After every factual claim, cite the passage "
        "it came from using the marker in square brackets, e.g. "
        "[01_llms_and_transformers.md#chunk_2]. If the passages do not fully "
        "answer the question, say so plainly.\n\n"
        f"Passages:\n{context_block}\n\n"
        f"Question: {state['question']}\n"
        "Answer:"
    )
    answer = _get_llm().invoke(prompt).content.strip()
    return {"answer": answer, "citations": citations}


def refuse_no_context(state: RAGState) -> RAGState:
    return {"answer": REFUSAL_MESSAGE, "citations": []}


def _route(state: RAGState) -> str:
    return "generate_with_context" if state["has_context"] else "refuse_no_context"


def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_relevance", grade_relevance)
    g.add_node("generate_with_context", generate_with_context)
    g.add_node("refuse_no_context", refuse_no_context)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade_relevance")
    g.add_conditional_edges(
        "grade_relevance",
        _route,
        {
            "generate_with_context": "generate_with_context",
            "refuse_no_context": "refuse_no_context",
        },
    )
    g.add_edge("generate_with_context", END)
    g.add_edge("refuse_no_context", END)
    return g.compile()
