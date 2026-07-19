"""SwiftEats support voice agent — primary implementation.

Pipeline: Deepgram STT -> Groq LLM (Llama-3.3-70b) -> Cartesia TTS, with silero VAD.

Run:
    python src/agent.py console     # local mic/speaker, no LiveKit server needed
    python src/agent.py dev         # connect to LiveKit Cloud (needs LIVEKIT_URL etc.)
"""

import logging

from livekit import agents
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions
from livekit.plugins import cartesia, deepgram, silero
from livekit.plugins.openai import LLM as OpenAILLM

import config  # side-effect: loads .env, sets env vars
from tools import get_order_status

log = logging.getLogger("swifteats.agent")

PERSONA = (
    "You are Rida, a friendly voice support assistant for a food delivery app "
    "called SwiftEats. Keep answers short and conversational — the user is "
    "hearing you, not reading you, so avoid lists and long sentences. "
    "When a customer asks about their order, use the get_order_status tool to "
    "look it up. Never make up order details. If a tool call fails, apologize "
    "and offer to connect them to a human."
)

GREETING = "Hi, this is Rida from SwiftEats support. How can I help?"


class SupportAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=PERSONA,
            tools=[get_order_status],
        )


async def entrypoint(ctx: JobContext) -> None:
    log.info("Starting SwiftEats support session")

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model=config.STT_MODEL),
        llm=OpenAILLM(
            model=config.LLM_MODEL,
            base_url="https://api.groq.com/openai/v1",
            api_key=config.GROQ_API_KEY,
        ),
        tts=cartesia.TTS(voice=config.TTS_VOICE),
    )

    # log every tool call so the transcript shows the LLM invoking tools
    @session.on("function_tools_executed")
    def _on_tool(event):  # type: ignore[no-untyped-def]
        for call in getattr(event, "function_calls", []) or []:
            log.info("[tool-call] %s(%s)", call.name, call.arguments)

    await ctx.connect()
    await session.start(agent=SupportAgent(), room=ctx.room)
    await session.generate_reply(instructions=f"Greet the user with: {GREETING!r}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agents.cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
