"""Voice + RAG agent — the bonus that combines Section 1 and Section 2.

Reuses:
  - Section 1's providers (Deepgram STT, Groq LLM, Cartesia TTS, silero VAD)
    and its config module for the API keys and model constants.
  - Section 2's compiled LangGraph, exposed via `rag_tool.py` as a
    `@function_tool` the LLM can call mid-conversation.

Nothing from Sections 1 or 2 is duplicated here — everything is imported.

Run:
    python bonus_voice_rag/src/agent.py console
"""

import logging

from _paths import s1_config  # sets up sys.path AND gives us Section 1's config

from livekit import agents
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions
from livekit.plugins import cartesia, deepgram, silero
from livekit.plugins.openai import LLM as OpenAILLM

from rag_tool import answer_from_knowledge_base

log = logging.getLogger("bonus.agent")

PERSONA = (
    "You are Nova, a voice assistant that answers questions from a small "
    "knowledge base about large language models, retrieval-augmented "
    "generation, and agentic AI. "
    "When the user asks a substantive or conceptual question about any of "
    "those topics, ALWAYS call the answer_from_knowledge_base tool — do not "
    "answer from your own memory. "
    "For greetings, small talk, or clarifying questions, respond directly "
    "without calling the tool. "
    "Keep replies short and conversational — the user is hearing you, not "
    "reading you."
)

GREETING = (
    "Hi, this is Nova. I can answer questions about LLMs, RAG, or agentic "
    "AI from a small knowledge base. What would you like to know?"
)


class KnowledgeAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=PERSONA,
            tools=[answer_from_knowledge_base],
        )


async def entrypoint(ctx: JobContext) -> None:
    log.info("Starting bonus voice-RAG session (Nova)")

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model=s1_config.STT_MODEL),
        llm=OpenAILLM(
            model=s1_config.LLM_MODEL,
            base_url="https://api.groq.com/openai/v1",
            api_key=s1_config.GROQ_API_KEY,
        ),
        tts=cartesia.TTS(voice=s1_config.TTS_VOICE),
    )

    @session.on("function_tools_executed")
    def _on_tool(event):  # type: ignore[no-untyped-def]
        for call in getattr(event, "function_calls", []) or []:
            log.info("[tool-call] %s(%s)", call.name, call.arguments)

    await ctx.connect()
    await session.start(agent=KnowledgeAssistant(), room=ctx.room)
    await session.generate_reply(instructions=f"Greet the user with: {GREETING!r}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agents.cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
