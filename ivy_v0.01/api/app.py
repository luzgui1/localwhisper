# api/app.py
#
# FastAPI entry point.
# Run with: uvicorn api.app:app --reload --port 8000

import asyncio
import os

import httpx
from fastapi import FastAPI

from agent import run as agent_run
from api.model import ChatRequest, ChatResponse, TelegramUpdate
from tools import session

app = FastAPI(
    title="LocalWhisper API",
    description="Urban leisure recommendations powered by Ivy.",
    version="0.1.0",
)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_KEY")


# ── Telegram helper ───────────────────────────────────────────────────────────

async def send_telegram_message(chat_id: int, text: str) -> None:
    """
    Send a message back to a Telegram user.
    Telegram doesn't read our webhook response body — we have to
    actively call their API to deliver the reply.
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client:
        await client.post(url, json={"chat_id": chat_id, "text": text})


# ── Shared agent runner ───────────────────────────────────────────────────────

async def _run_agent(session_id: str, user_text: str) -> str:
    """
    Set the active session, load history, run the agent, save history.
    Used by both /chat and /webhook so the logic lives in one place.
    """
    session.set_active(session_id)
    history = session.get_history()

    loop  = asyncio.get_event_loop()
    reply = await loop.run_in_executor(None, agent_run, user_text, history)

    session.append_to_history("user",      user_text)
    session.append_to_history("assistant", reply)

    return reply


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── Chat (generic — any client) ───────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    reply = await _run_agent(req.session_id, req.message)
    return ChatResponse(reply=reply, session_id=req.session_id)


# ── Webhook (Telegram) ────────────────────────────────────────────────────────

@app.post("/webhook")
async def webhook(update: TelegramUpdate):
    """
    Telegram calls this endpoint every time a user sends a message.
    Two message types are handled — text and location — both go through
    the agent so Ivy always responds naturally with full context.
    """
    if not update.message:
        return {"ok": True}

    chat_id    = update.message.chat.id
    session_id = str(chat_id)

    # ── GPS location shared via Telegram attachment button ────────────────────
    # Store the real coordinates in session, then pass a synthetic message to
    # the agent. It has the conversation history so it knows what the user
    # was trying to do — it responds naturally without any hardcoded string.
    if update.message.location:
        loc = update.message.location
        session.set_active(session_id)
        session.set_location({
            "lat":     loc.latitude,
            "lng":     loc.longitude,
            "city":    "compartilhada via GPS",
            "country": "",
        })
        reply = await _run_agent(
            session_id,
            "[usuario compartilhou localizacao via GPS]",
        )
        await send_telegram_message(chat_id, reply)
        return {"ok": True}

    # ── Text message ──────────────────────────────────────────────────────────
    if not update.message.text:
        return {"ok": True}   # photo, sticker, etc. — ignore silently

    reply = await _run_agent(session_id, update.message.text)
    await send_telegram_message(chat_id, reply)
    return {"ok": True}