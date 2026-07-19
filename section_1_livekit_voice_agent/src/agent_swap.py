"""Bonus 1.2: same agent, different STT provider.

This file is intentionally almost identical to agent.py. The ONLY functional
difference is the `stt=` line — Deepgram is swapped for Google Cloud STT
(which ships in the `livekit-plugins-google` package we already use for the
LLM). Everything else — persona, tools, TTS, VAD — is unchanged, which is the
point: `AgentSession` decouples the agent from any one vendor.

Run:
    python src/agent_swap.py console

Note: Google Cloud STT needs GCP application-default credentials, not just
the Gemini API key. If you don't have GCP set up, treat this file as
demonstration-only and see the README for the exact diff.
"""

import logging

from livekit import agents
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions
from livekit.plugins import cartesia, silero
from livekit.plugins.openai import LLM as OpenAILLM

import config
from tools import get_order_status

log = logging.getLogger("swifteats.agent_swap")

PERSONA = (
    "You are Rida, a friendly voice support assistant for a food delivery app "
    "called SwiftEats. Keep answers short and conversational. When a customer "
    "asks about their order, use the get_order_status tool to look it up. "
    "Never make up order details."
)


class SupportAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=PERSONA, tools=[get_order_status])


async def entrypoint(ctx: JobContext) -> None:
    log.info("Starting SwiftEats session (Google STT variant)")

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=google.STT(model="latest_long"),        # <-- swapped from deepgram.STT()
        llm=OpenAILLM(
            model=config.LLM_MODEL,
            base_url="https://api.groq.com/openai/v1",
            api_key=config.GROQ_API_KEY,
        ),
        tts=cartesia.TTS(voice=config.TTS_VOICE),
    )

    @session.on("function_tools_executed")
    def _on_tool(event):  # type: ignore[no-untyped-def]
        for call in getattr(event, "function_calls", []) or []:
            log.info("[tool-call] %s(%s)", call.name, call.arguments)

    await ctx.connect()
    await session.start(agent=SupportAgent(), room=ctx.room)
    await session.generate_reply(
        instructions="Greet the user: 'Hi, this is Rida from SwiftEats support.'"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agents.cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
