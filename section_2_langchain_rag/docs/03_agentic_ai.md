# Agentic AI

## What people mean by "agentic"

Agentic AI is the term that describes systems in which a language model is not used as a one-shot text generator but as the decision-maker inside a longer loop. In a plain LLM call, the user asks a question, the model answers, and the interaction ends. In an agentic system, the model is given a goal, a set of tools it can use, and the freedom to decide, step by step, which tool to call and when to stop. The model's outputs are no longer just words for a human to read; some of them are actions that change the state of the world, and some of them are observations of what happened after those actions were taken. This shift, from generation to iterated decision-making, is what makes something feel like an agent rather than a chatbot.

## The core loop

Almost every agentic system, no matter how it is dressed up, follows the same underlying loop. The agent perceives its current situation, which usually means reading the user's request together with any results from previous steps. It then plans what to do next, which in practice means the LLM generates a decision, often expressed as a tool call in a structured format. The chosen action is executed by the surrounding runtime — this is the part that is not the LLM: a Python function runs, an API is called, a database is queried, a file is written. The result of that action is observed, added back into the model's context, and the loop starts over. The agent keeps cycling until it decides that the goal has been reached, or until an external limit stops it.

The mechanics of this loop matter less than the fact that it exists. Once you have a loop, the model can attempt something, see whether it worked, and try a different approach if it did not. That capacity to correct itself is where most of the value of agents comes from, and it is also where most of the risks come from.

## Tools and function calling

Tools are the interface between the language model and the outside world. Modern chat-tuned models can be given a list of tool definitions — each one a name, a natural-language description, and a schema for its arguments — and the model will emit structured tool calls when it thinks a tool is needed. This is usually called function calling. Under the hood, the model is still just generating tokens, but the tokens are constrained to match the tool schema, so the surrounding runtime can parse them reliably and dispatch the actual work. Good tool design matters more than most first-time builders realise. Descriptions should be unambiguous, arguments should be typed, and the return value should be short enough for the model to read without blowing the context window. A tool that returns a two-megabyte JSON blob is worse than useless, because the model will drown in it.

## Planning strategies

Different agentic systems use different planning styles. The simplest, and still one of the most common, is ReAct, short for Reasoning and Acting. In ReAct the model interleaves short chains of reasoning with tool calls, thinking briefly about what to do, taking one action, observing the result, and thinking again. This is a tight loop that reacts to feedback quickly but can wander if the task is long. Plan-and-execute is a different pattern: the model first writes an explicit plan, a numbered list of steps, and then executes them one by one, only re-planning if something goes wrong. This tends to be more efficient on well-defined tasks but is brittle when reality does not match the plan. More sophisticated systems use hierarchical planning, where a top-level agent decomposes a goal into sub-goals and delegates each one to a sub-agent.

## Memory

Because the loop can run for many steps, memory becomes a first-class concern. Short-term memory is the current context window, which holds the recent conversation and the running trace of actions and observations. It is fast, but it is finite, and long agent runs will overflow it. Long-term memory is stored outside the model, usually in a vector database or a structured store, and it is retrieved back into the prompt when relevant. This is essentially RAG applied to the agent's own history rather than to external documents. Well-designed agents also have working memory for intermediate results — scratch space that is not shown to the user but is used to track progress across steps.

## Multi-agent systems

Once one agent works, the obvious next question is whether several agents can cooperate. Multi-agent systems assign different roles to different LLM instances — one plans, one writes code, one reviews, one talks to the user — and let them communicate through a shared message bus or a central coordinator. This can produce impressive results on complex tasks, because specialisation lets each agent have a tighter system prompt and a smaller tool set. It also multiplies the risks: agents can talk in circles, hallucinate at each other, and rack up costs at surprising speed. In practice, most production systems are simpler than the multi-agent literature suggests, and a single well-instrumented agent usually beats a poorly-orchestrated committee.

## Safety and evaluation

Agentic systems are harder to make safe than plain chatbots because they take actions. A model that hallucinates a fact is embarrassing; a model that hallucinates a database write is expensive. Prompt injection is especially dangerous in this setting, because a malicious document retrieved by the agent can contain instructions that the model may follow. Sensible defences include restricting the surface area of each tool, requiring human confirmation for irreversible actions, sandboxing code execution, and enforcing hard limits on cost and step count.

Evaluation is also harder. There is no single correct answer to grade against, because the same goal can be reached by different action sequences. Teams end up combining traditional metrics with trajectory-level evaluation, checking whether the agent reached the goal, how many steps it took, whether it used the right tools, and whether it avoided obviously bad actions along the way.

## When to reach for an agent, and when not to

The honest answer is that most tasks do not need an agent. If the workflow is a fixed pipeline of steps, a plain chain is cheaper, faster, and easier to debug. Agents earn their keep when the number of steps is genuinely unknown ahead of time, when the right next action depends on what the previous action returned, or when the tool space is large enough that pre-scripting every branch is impractical. Reserve them for those cases, and the loop starts to feel less like a party trick and more like a durable pattern.
