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
    version="0.0.1",
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


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── Chat (generic — any client) ───────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session.set_active(req.session_id)
    history = session.get_history()

    loop = asyncio.get_event_loop()
    reply = await loop.run_in_executor(None, agent_run, req.message, history)

    session.append_to_history("user",      req.message)
    session.append_to_history("assistant", reply)

    return ChatResponse(reply=reply, session_id=req.session_id)


# ── Webhook (Telegram) ────────────────────────────────────────────────────────

@app.post("/webhook")
async def webhook(update: TelegramUpdate):
    """
    Telegram calls this endpoint every time a user sends a message to the bot.
    We must return 200 OK — Telegram ignores the response body.
    The actual reply goes back via send_telegram_message().
    """
    # Ignore updates that have no text message (photos, stickers, etc.)
    if not update.message or not update.message.text:
        return {"ok": True}

    chat_id    = update.message.chat.id
    user_text  = update.message.text
    session_id = str(chat_id)   # chat_id is an int, session_id is a string

    # Same logic as /chat — just a different source of input
    session.set_active(session_id)
    history = session.get_history()

    loop  = asyncio.get_event_loop()
    reply = await loop.run_in_executor(None, agent_run, user_text, history)

    session.append_to_history("user",      user_text)
    session.append_to_history("assistant", reply)

    # Send the reply back to the user on Telegram
    await send_telegram_message(chat_id, reply)

    return {"ok": True}