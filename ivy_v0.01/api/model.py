# api/models.py

from pydantic import BaseModel, Field


# ── LocalWhisper API models ───────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:    str = Field(description="The user's message.")
    session_id: str = Field(description="A UUID that identifies this conversation.")


class ChatResponse(BaseModel):
    reply:      str = Field(description="Ivy's reply.")
    session_id: str = Field(description="Echo of the session_id from the request.")


# ── Telegram webhook models ───────────────────────────────────────────────────
# Telegram sends a large JSON payload called an Update.
# We only model the fields we actually need — Pydantic ignores the rest.

class TelegramChat(BaseModel):
    id: int   # the chat_id — unique per user, becomes our session_id


class TelegramMessage(BaseModel):
    chat: TelegramChat
    text: str | None = None   # None for photos, stickers, etc. — we ignore those


class TelegramUpdate(BaseModel):
    update_id: int
    message:   TelegramMessage | None = None   # None for other update types