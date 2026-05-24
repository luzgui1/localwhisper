# agent.py

import time

import tiktoken
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from logger import log
from tools.tools import (
    get_user_location,
    set_user_location_by_text,
    search_and_rank_places,
    get_session_places,
)

# ── System prompt ─────────────────────────────────────────────────────────────
# Contains ONLY: persona, tone, and output behavior.
# Tool routing lives in each tool's own description and Field definitions.
# Adding a new tool never requires touching this file — except for behavioral
# rules that span multiple tools (like location handling below).

SYSTEM_PROMPT = """You are Ivy, a friendly urban leisure assistant.

## Persona
- Warm and conversational — like a knowledgeable local friend, not a search engine.
- Always respond in the same language the user is using.
- For smalltalk, just chat. Never call a tool unless a recommendation is needed.
- Avoid using emojis, be as close to a human as you can.

## Tool sequencing
- Always call get_user_location first and wait for its result before calling
  search_and_rank_places. Never call both at the same time.
- Only use coordinates returned by get_user_location — never invent or assume them.

## Location handling
- When get_user_location returns a "no_location" error, ask naturally where
  the user is — once, casually, like a friend would. Do not repeat the question.
- Accept any format: neighbourhood ("Pinheiros"), city ("SP", "Sao Paulo"),
  abbreviation, or landmark ("perto do Ibirapuera").
- When they reply with a text location, call set_user_location_by_text.
- If set_user_location_by_text returns "ambiguous: true", present the options
  clearly and ask which one — never pick without confirming.
- When you receive "[usuario compartilhou localizacao via GPS]", the user just
  shared real GPS coordinates via Telegram. Acknowledge naturally and proceed
  with whatever they were asking for.
- Never ask for location again once it is stored in the session.

## When to search vs when to use session
- Call search_and_rank_places ONLY for a new type of venue not yet searched.
- For ANY follow-up about already found places — open now, closest,
  neighbourhood, comparisons — use get_session_places instead.

## When presenting venue results
- Show the top 3-5 options with name, address, rating, distance, and hours.
- Add a personal touch based on reviews.
- Never invent or modify data returned by tools.
"""

# ── Agent ─────────────────────────────────────────────────────────────────────

llm   = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [
    get_user_location,
    set_user_location_by_text,
    search_and_rank_places,
    get_session_places,
]
agent = create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)


# ── History trimming ──────────────────────────────────────────────────────────

_enc = tiktoken.encoding_for_model("gpt-4o")
MAX_HISTORY_TOKENS = 6_000


def _count_tokens(history: list[dict]) -> int:
    total = 0
    for m in history:
        content = m.get("content", "")
        if isinstance(content, str):
            total += len(_enc.encode(content)) + 4
    return total


def _trim_history(history: list[dict]) -> list[dict]:
    """Drop oldest turns until the history fits within MAX_HISTORY_TOKENS."""
    while len(history) > 2 and _count_tokens(history) > MAX_HISTORY_TOKENS:
        history.pop(0)
        if history and history[0]["role"] == "assistant":
            history.pop(0)
        log.debug("history: trimmed to %d messages (%d tokens)",
                  len(history), _count_tokens(history))
    return history


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_lc_messages(history: list[dict]) -> list:
    msgs = []
    for m in history:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            msgs.append(AIMessage(content=m["content"]))
    return msgs


def run(user_input: str, history: list[dict]) -> str:
    log.info("run()  input=%r", user_input[:80])
    t0 = time.perf_counter()

    history  = _trim_history(history)
    messages = _to_lc_messages(history) + [HumanMessage(content=user_input)]
    result   = agent.invoke({"messages": messages})
    reply    = result["messages"][-1].content

    log.info("run() done in %.2fs", time.perf_counter() - t0)
    return reply


def run_verbose(user_input: str, history: list[dict]) -> str:
    log.info("run_verbose()  input=%r", user_input[:80])
    t0 = time.perf_counter()

    history  = _trim_history(history)
    messages = _to_lc_messages(history) + [HumanMessage(content=user_input)]

    print("\n── Agent reasoning ──────────────────────────────────────────")
    final_content = ""
    step_n = 0

    for step in agent.stream({"messages": messages}, stream_mode="values"):
        step_n += 1
        last    = step["messages"][-1]
        elapsed = time.perf_counter() - t0

        if hasattr(last, "tool_calls") and last.tool_calls:
            for tc in last.tool_calls:
                args_preview = {
                    k: (str(v)[:60] + "..." if isinstance(v, str) and len(str(v)) > 60 else v)
                    for k, v in tc.get("args", {}).items()
                }
                print(f"\n[{elapsed:5.1f}s] TOOL CALL  {tc['name']}")
                print(f"          args: {args_preview}")
                log.debug("step=%d  tool_call=%s  args=%s", step_n, tc["name"], tc.get("args"))

        elif getattr(last, "type", None) == "tool":
            content_str = str(last.content)
            preview     = content_str[:100] + ("..." if len(content_str) > 100 else "")
            print(f"\n[{elapsed:5.1f}s] RESULT     {getattr(last, 'name', '?')}")
            print(f"          {preview}")
            log.debug("step=%d  tool_result=%s  preview=%s", step_n, getattr(last, "name", "?"), preview)

        elif getattr(last, "type", None) == "ai" and not getattr(last, "tool_calls", None):
            final_content = last.content
            log.debug("step=%d  final_answer (%d chars)", step_n, len(final_content))

    print("─────────────────────────────────────────────────────────────\n")
    log.info("run_verbose() done in %.2fs  steps=%d", time.perf_counter() - t0, step_n)
    return final_content