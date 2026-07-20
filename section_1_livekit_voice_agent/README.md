# Section 1 — LiveKit Voice Agent

A minimal real-time voice agent using the `livekit-agents` Python SDK. Built
as a food-delivery support persona ("Rida" from SwiftEats) that can look up
order status via a tool call the LLM invokes mid-conversation.

## Pipeline

```
Mic -> silero VAD -> Deepgram STT -> Groq LLM (Llama-3.3-70b) <-> tools -> Cartesia TTS -> Speaker
```

The full graph and turn loop are documented in the plan file. In short: VAD
segments speech into utterances, STT transcribes them, the LLM decides whether
to answer directly or call the `get_order_status` tool, then TTS speaks the
reply. While TTS is playing, VAD keeps listening — so if the user talks over
the agent, the current response is interrupted and a new turn begins.

> **LLM choice note.** The initial design used Gemini 2.5 Flash via
> `livekit-plugins-google`, but its free-tier per-minute request cap (5 RPM)
> throttles a voice conversation immediately. Swapping to **Groq's
> Llama-3.3-70b** (via `livekit-plugins-openai` with Groq's OpenAI-compatible
> endpoint) gives 30 RPM and lower latency for free — a much better fit for a
> real-time voice loop. The rest of the pipeline is untouched, which is
> exactly the vendor-decoupling story the bonus (1.2) is testing.

## Setup

**1. Conda env**
```
conda create -n section1-livekit python=3.11 -y
conda activate section1-livekit
```

**2. Install**
```
cd section_1_livekit_voice_agent
pip install -r requirements.txt
pip install livekit-plugins-openai      # for the Groq/OpenAI-compatible LLM path
```

**3. API keys**

Copy `.env.example` to `.env` and fill in three keys:
- `GROQ_API_KEY` — from https://console.groq.com/ (free tier, 30 RPM)
- `DEEPGRAM_API_KEY` — from https://console.deepgram.com/ ($200 free credit on signup)
- `CARTESIA_API_KEY` — from https://play.cartesia.ai/ (free tier)

## Run

```
python src/agent.py console
```

`console` mode uses your local mic and speaker; no LiveKit Cloud account
needed. Speak into your mic — the agent should greet you, then answer
questions like "Can you check order 5473?" using the mocked backend.

For the bonus (STT-swapped variant):
```
python src/agent_swap.py console
```

## Files

- [`src/config.py`](src/config.py) — env loading, model + voice constants.
- [`src/tools.py`](src/tools.py) — `get_order_status` function tool and mock database.
- [`src/agent.py`](src/agent.py) — the main `SupportAgent` and `AgentSession` setup.
- [`src/agent_swap.py`](src/agent_swap.py) — same agent, Google STT instead of Deepgram (bonus 1.2).
- [`transcripts/example_run.md`](transcripts/example_run.md) — recorded transcript from a real session.

## Mock orders you can ask about

- `5473` — out for delivery (Margherita pizza)
- `9911` — preparing (Chicken shawarma, Pepsi)
- `0001` — delivered (Falafel wrap)

Anything else (e.g. `1234`) triggers the "I couldn't find that order" branch,
which demonstrates the tool's negative-case handling.

## Write-up

### Barge-in / interruption handling

`AgentSession` supports barge-in out of the box because VAD is always
listening, even while TTS is playing. When VAD detects the user speaking mid-
response, the SDK cancels the outgoing TTS stream, drops the un-spoken text
from the assistant transcript so the LLM's next turn is aware only of what
the user actually heard, and starts a fresh STT pass on the new utterance.
In noisy environments the `allow_interruptions=True` flag can be turned off,
or the VAD's `min_speech_duration` / `activation_threshold` can be raised so
short background noises don't count as speech. For debugging or analytics,
subscribing to the `agent_state_changed` and `SpeechCreatedEvent` events on
the session gives you exact timestamps of when a turn started, was
interrupted, or ended, which is useful for measuring end-to-end turn-taking
latency in production. In this codebase, the `session.on("function_tools_
executed")` hook already writes the tool-call to the log so we can prove the
LLM actually invoked the tool during a real session.

### Adding a second tool safely

Safe tool addition is mostly about surface area and error handling. First,
the *schema* is load-bearing: LiveKit passes the Python docstring to the LLM
as the tool's description, so a vague or missing docstring means the LLM
either won't call the tool when it should or will call it with wrong
arguments. Every tool should have a one-line summary plus `Args:` block, and
argument names should be self-descriptive. Second, *validate at the tool
boundary*: `get_order_status` already rejects non-4-digit IDs with a
regex before any lookup, and a real backend call would enforce the same
before hitting the DB. Third, *never let raw exceptions bubble up to the
LLM*: wrap external calls in try/except and return a natural-language error
string like "I'm having trouble reaching the order system, can I try again in
a moment?" — the LLM will speak that back to the user gracefully, whereas
an unhandled exception would tear down the session. Fourth, *destructive
tools need confirmation*: a hypothetical `cancel_order(order_id)` should
first return "About to cancel order 5473 — please confirm" and only actually
mutate state on a second turn, so an ASR misfire ("cancer my order" being
mistranscribed as "cancel my order") doesn't wipe out a live order.
Finally, *keep tool scope narrow*: a wide `run_sql(query)` tool gives the
LLM enough rope to leak PII or bring down the database, so prefer many
small, well-typed tools over one wide-open escape hatch.

## Bonus 1.2 — swapping a provider

`src/agent_swap.py` is the same agent with Deepgram STT replaced by Google
Cloud STT. The full diff versus `agent.py` is one line:

```
- stt=deepgram.STT(model=config.STT_MODEL),
+ stt=google.STT(model="latest_long"),
```

Persona, tools, LLM, TTS and VAD are unchanged. The reason this works is
that `AgentSession` accepts any object implementing the `STT` interface, so
the agent code is completely decoupled from the vendor. The same would apply
to swapping Cartesia for ElevenLabs (`tts=elevenlabs.TTS(voice=...)` after
`pip install "livekit-agents[elevenlabs]"`) or swapping the LLM from Groq
to OpenAI (`llm=OpenAILLM(model="gpt-4o-mini")` — same class, just drop the
`base_url`).

In fact the switch from Gemini to Groq that fixed the rate-limiting (see
LLM choice note above) is a working example of the same decoupling: one
constructor swap in `agent.py`, no changes elsewhere.

Note: Google Cloud STT needs GCP application-default credentials. If you
don't have GCP set up, treat `agent_swap.py` as demonstration code — the
one-line swap is the point.

## Known limitations

- **Console mode only.** Not tested against LiveKit Cloud in this submission.
  The same code should work in `dev` mode with `LIVEKIT_URL` /
  `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` set.
- **Single tool.** The write-up covers how a second tool would be added
  safely, but only `get_order_status` is wired up.
- **Mocked database.** The order lookup is an in-memory dict, not a real
  backend. Everything else in the pipeline — STT, LLM, TTS, tool dispatch —
  is real.
- **First-run cost.** silero VAD downloads a small ONNX model on first use
  (cached under the LiveKit plugins cache directory).
