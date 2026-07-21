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
    "You are Nova, a voice assistant. Your knowledge is strictly limited "
    "to a small knowledge base about large language models, "
    "retrieval-augmented generation (RAG), and agentic AI. "
    "\n\n"
    "TOOL POLICY — read carefully, this is the whole job:\n"
    "1. For any substantive or conceptual question about LLMs, RAG, or "
    "agentic AI, you MUST call the answer_from_knowledge_base tool. Do "
    "not answer from your own training. Do not guess.\n"
    "2. Call the tool AT MOST ONCE per user turn. Never call it a second "
    "time in the same response, even if the first answer seems brief or "
    "imperfect. The retrieval covers the full knowledge base on one call "
    "— repeating with a refined query returns the same chunks and only "
    "burns rate limits. If the user wants more detail, they will ask a "
    "follow-up question.\n"
    "3. When the tool returns an answer, SPEAK THAT ANSWER TO THE USER. "
    "Do not summarize it. Do not shorten it. Do not paraphrase it. Do "
    "not replace any of its content with your own knowledge. The tool's "
    "answer IS your response.\n"
    "4. You may prepend at most one short conversational lead-in like "
    "'Sure — ' or 'Good question. ' before the tool's answer. That is "
    "the ONLY thing you may add. Do not append closing lines like "
    "'let me know if you have more questions' or 'feel free to ask' or "
    "any meta commentary.\n"
    "5. If the tool returns a refusal string (e.g. 'I couldn't find "
    "information about that in my knowledge base'), speak that refusal "
    "faithfully — do not try to answer from your own memory instead, "
    "and do NOT call the tool again with a different query.\n"
    "\n"
    "For greetings, thanks, or pure small-talk that has no factual "
    "content, respond directly without calling the tool. Keep those "
    "replies to one short sentence."
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
        # Console mode without headphones causes the mic to pick up Nova's
        # own TTS output. Deepgram then transcribes it as user speech and
        # LiveKit treats it as barge-in, cutting off the response. Requiring
        # a longer sustained utterance filters out these short echo bursts.
        # With proper headphones or a production AEC, this can be lowered
        # back to the default (0.5s) to make barge-in more responsive.
        min_interruption_duration=2.0,
        min_interruption_words=3,
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
