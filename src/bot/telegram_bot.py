"""Telegram bot — Phase 5 access point to the shared query pipeline.

This module is a thin ADAPTER. All retrieval, reranking, and generation live in
`src.query.pipeline` — the exact same code the web API uses. If you find
yourself wanting to add retrieval logic here, add it to `pipeline.py` instead.

The bot runs INSIDE the uvicorn Python process (see the FastAPI lifespan in
`src.web.api`). That means:

  * The embedding model, cross-encoder, and DB pool are loaded ONCE and shared
    between the web and the bot — critical for a RAM-limited VPS.
  * The pipeline call is sync/blocking (Postgres → cross-encoder on CPU →
    OpenRouter HTTP). We offload it to a worker thread with `asyncio.to_thread`
    so uvicorn's event loop stays free to serve web requests concurrently.

Access control is a Telegram-user-ID allowlist read from
`TELEGRAM_ALLOWED_USER_IDS` (comma-separated). An empty list means "allow
anyone" — useful for local testing; production always sets it.
"""
from __future__ import annotations

import asyncio
import html
import logging

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.config import settings
from src.core.generation import NoModelSucceeded
from src.query.pipeline import Answer, answer_question

log = logging.getLogger(__name__)


# --- Message sizing ---------------------------------------------------------
# Telegram hard limit is 4096 chars per message. Leave a budget under that so
# the sources block (or a chunk boundary marker) always fits.
_TELEGRAM_MSG_LIMIT = 4096
_ANSWER_BUDGET = 3800


# --- User-facing strings (English, per project decision) --------------------
_WELCOME = (
    "👋 Hi! I answer questions grounded in an indexed arXiv corpus.\n\n"
    "Just send me a question — no command needed.\n"
    "Example: <i>What is Mamba?</i>\n\n"
    "Every answer cites the papers it was drawn from."
)

_UNAUTHORIZED = (
    "This bot is private. Ask the operator to add your Telegram user ID to "
    "TELEGRAM_ALLOWED_USER_IDS."
)

_OPENROUTER_DOWN = (
    "⚠️ OpenRouter's free models are rate-limited or unavailable right now. "
    "Try again in a few minutes."
)

_INTERNAL_ERROR = (
    "⚠️ Something broke on my side. It's been logged — please try again shortly."
)


def _is_authorized(user_id: int | None) -> bool:
    """Empty allowlist = allow anyone (local dev). Non-empty = strict allowlist."""
    allowed = settings.telegram_allowed_user_ids_list
    if not allowed:
        return True
    return user_id is not None and user_id in allowed


def _format_answer(ans: Answer) -> list[str]:
    """Format an Answer as one or more Telegram HTML messages.

    Single-message layout when everything fits under _ANSWER_BUDGET:
        <answer text>

        📚 Sources:
        1. <a href="URL">Title</a>
        2. <a href="URL">Title</a>

    When the combined length overflows, the answer is chunked (respecting
    paragraph boundaries) into multiple messages and the sources block is sent
    as its own final message. Content is never truncated.
    """
    answer_text = html.escape(ans.text.strip())

    if ans.sources:
        source_lines = [
            f"{i}. <a href=\"{html.escape(s.url)}\">{html.escape(s.title)}</a>"
            for i, s in enumerate(ans.sources, start=1)
        ]
        sources_block = "\n\n📚 <b>Sources:</b>\n" + "\n".join(source_lines)
    else:
        sources_block = ""

    combined = answer_text + sources_block
    if len(combined) <= _ANSWER_BUDGET:
        return [combined]

    messages = _chunk(answer_text, _ANSWER_BUDGET)
    if sources_block:
        messages.append(sources_block.lstrip("\n"))
    return messages


def _chunk(text: str, limit: int) -> list[str]:
    """Split text into chunks <= limit, preferring paragraph then line boundaries."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n\n", 0, limit)
        if cut == -1:
            cut = remaining.rfind("\n", 0, limit)
        if cut == -1:
            # No natural boundary in range — hard cut. Rare on LLM prose.
            cut = limit
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


# --- Handlers ---------------------------------------------------------------
async def _cmd_start(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    message = update.effective_message
    if message is None:
        return
    if not _is_authorized(user.id if user else None):
        await message.reply_text(_UNAUTHORIZED)
        return
    await message.reply_text(_WELCOME, parse_mode=ParseMode.HTML)


async def _cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await _cmd_start(update, ctx)


async def _on_question(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    message = update.effective_message
    if message is None or not message.text:
        return

    if not _is_authorized(user.id if user else None):
        await message.reply_text(_UNAUTHORIZED)
        return

    question = message.text.strip()
    if not question:
        return

    # "typing" indicator while we retrieve + rerank + generate. Best-effort UX.
    try:
        await message.chat.send_action(ChatAction.TYPING)
    except Exception:  # noqa: BLE001
        pass

    try:
        # Pipeline is sync/blocking. Offload to a worker thread so uvicorn's
        # event loop stays free (web requests keep being served in parallel).
        ans = await asyncio.to_thread(answer_question, question)
    except NoModelSucceeded:
        log.warning("Telegram: all OpenRouter models failed for question=%r", question)
        await message.reply_text(_OPENROUTER_DOWN)
        return
    except Exception:  # noqa: BLE001 — user must never be left hanging
        log.exception("Telegram: pipeline crashed on question=%r", question)
        await message.reply_text(_INTERNAL_ERROR)
        return

    for chunk in _format_answer(ans):
        await message.reply_text(
            chunk,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )


async def _on_error(update: object, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Catch-all for exceptions PTB surfaces via add_error_handler."""
    log.exception("Telegram handler raised: %s", ctx.error)
    if isinstance(update, Update) and update.effective_message is not None:
        try:
            await update.effective_message.reply_text(_INTERNAL_ERROR)
        except Exception:  # noqa: BLE001
            pass


# --- Lifecycle (called from FastAPI lifespan) -------------------------------
def create_application() -> Application:
    """Build the PTB Application with our handlers wired up. Does not start polling."""
    if not settings.TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not set — cannot create the Telegram Application."
        )
    application = ApplicationBuilder().token(settings.TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", _cmd_start))
    application.add_handler(CommandHandler("help", _cmd_help))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, _on_question)
    )
    application.add_error_handler(_on_error)
    return application


async def start_bot() -> Application:
    """Start long-polling as a background task. Returns the Application so the
    caller can hand it back to `stop_bot()` at shutdown."""
    application = create_application()
    await application.initialize()
    await application.start()
    # drop_pending_updates=True: on redeploy, skip the backlog that piled up
    # while the container was rebuilding — nobody wants a 20-message replay.
    await application.updater.start_polling(drop_pending_updates=True)
    return application


async def stop_bot(application: Application) -> None:
    """Cleanly stop long-polling and dispose of the Application."""
    try:
        if application.updater is not None and application.updater.running:
            await application.updater.stop()
        await application.stop()
        await application.shutdown()
    except Exception:  # noqa: BLE001
        log.exception("Error while stopping Telegram bot")
