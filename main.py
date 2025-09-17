import os
import re
import asyncio
import logging
from logging.handlers import RotatingFileHandler
from collections import deque
from dotenv import load_dotenv

import google.generativeai as genai
from google.api_core import exceptions

from telegram import Update
from telegram.constants import ChatAction, ChatType
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

# -----------------------------
# Load Environment Variables & Configuration
# -----------------------------
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-2.5-flash-lite" # Changed to a generally available model

if not TELEGRAM_TOKEN or not GEMINI_API_KEY:
    print("FATAL ERROR: TELEGRAM_BOT_TOKEN and GEMINI_API_KEY must be set in the .env file.")
    exit()

# -----------------------------
# Central Logging Setup
# -----------------------------
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
master_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

master_file_handler = RotatingFileHandler(
    os.path.join(LOGS_DIR, "console.log"), maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
)
master_file_handler.setFormatter(master_formatter)
root_logger.addHandler(master_file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(master_formatter)
root_logger.addHandler(console_handler)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
user_loggers = {}

def sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/*?:"<>|]', "", name).replace(" ", "_")
    return name

def get_user_logger(chat_id: int, full_name: str) -> logging.Logger:
    if chat_id in user_loggers:
        return user_loggers[chat_id]
    logger = logging.getLogger(str(chat_id))
    logger.setLevel(logging.INFO)
    logger.propagate = True
    user_file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    safe_fullname = sanitize_filename(full_name)
    log_file_path = os.path.join(LOGS_DIR, f"{safe_fullname}_{chat_id}.log")
    file_handler = RotatingFileHandler(log_file_path, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(user_file_formatter)
    logger.addHandler(file_handler)
    user_loggers[chat_id] = logger
    return logger

# -----------------------------
# Gemini Client
# -----------------------------
try:
    genai.configure(api_key=GEMINI_API_KEY)
    
    system_instruction = """You are a telegram bot offering study group help.
- Always format messages using Telegram MarkdownV2.
- Keep replies short, structured, and engaging.
- Use bullet points, examples, and emojis.
- One or two lines unless a longer answer is explicitly needed.
- Do not use LaTeX or unsupported markup, only MarkdownV2.
"""
    
    gemini_model = genai.GenerativeModel(
        MODEL_NAME,
        system_instruction=system_instruction
    )
    history = deque(maxlen=20)
    root_logger.info(f"Gemini AI client configured successfully with model: {MODEL_NAME}")
except Exception as e:
    root_logger.critical(f"Failed to configure Gemini AI: {e}", exc_info=True)
    exit()

# -----------------------------
# Bot State Management
# -----------------------------
# { chat_id: boolean }
# False (default): Reply to all messages.
# True: Reply only on mention or reply to bot's message.
chat_reply_modes = {}


# -----------------------------
# Gemini & Telegram Interaction Helpers
# -----------------------------
def generate_gemini_answer(prompt: str) -> str:
    history.append({'role': 'user', 'parts': [prompt]})
    
    try:
        chat_session = gemini_model.start_chat(history=list(history))
        response = chat_session.send_message(prompt)
        full_response_text = response.text
        
        if full_response_text.strip():
            history.append({'role': 'model', 'parts': [full_response_text]})
        
        if len(history) > 10:
             history.popleft()
             history.popleft()
        return full_response_text

    except exceptions.GoogleAPICallError as e:
        root_logger.error(f"Gemini API Call Error: {e}")
        return "API Error: Could not get a response."
    except Exception as e:
        root_logger.error(f"Unexpected error in Gemini generation: {e}", exc_info=True)
        return "An unexpected error occurred."

async def generate_and_reply(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt: str):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    
    answer = await asyncio.to_thread(generate_gemini_answer, prompt)
    
    await update.message.reply_text(answer)
    
    user_logger = get_user_logger(update.message.chat.id, update.message.from_user.full_name)
    user_logger.info(f"BOT: {' '.join(answer.splitlines())}")

def log_user_message(func):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.text: return
        user = update.message.from_user
        user_logger = get_user_logger(user.id, user.full_name)
        username_str = f"(@{user.username})" if user.username else ""
        user_logger.info(f"USER {username_str}: {update.message.text}")
        return await func(update, context)
    return wrapper

# -----------------------------
# Command and Message Handlers
# -----------------------------
@log_user_message
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Hello! I’m your study group bot. Ask me anything!")

@log_user_message
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    
    # In private chats, always reply.
    if chat.type == ChatType.PRIVATE:
        await generate_and_reply(update, context, update.message.text)
        return

    # In group chats, check the reply mode.
    # Default to False (reply to everything) if not set.
    mention_only_mode = chat_reply_modes.get(chat.id, False)

    if mention_only_mode:
        message = update.message
        bot_username = f"@{context.bot.username}"
        
        is_reply_to_bot = message.reply_to_message and message.reply_to_message.from_user.id == context.bot.id
        bot_is_mentioned = bot_username in (message.text or "")

        if is_reply_to_bot or bot_is_mentioned:
            prompt = message.text.replace(bot_username, "").strip()
            await generate_and_reply(update, context, prompt)
    else:
        # If mode is False, reply to all messages.
        await generate_and_reply(update, context, update.message.text)


@log_user_message
async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = " ".join(context.args)
    if not question:
        await update.message.reply_text("Please provide a question after /ask.")
        return
    await generate_and_reply(update, context, question)

@log_user_message
async def set_reply_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat

    # This command is not applicable in private chats.
    if chat.type == ChatType.PRIVATE:
        await update.message.reply_text("This command is only for group chats.")
        return

    # ADMIN CHECK IS REMOVED - ANY USER CAN RUN THIS COMMAND

    if not context.args:
        current_mode = "mention/reply" if chat_reply_modes.get(chat.id, False) else "all messages"
        await update.message.reply_text(
            f"Current reply mode: *{current_mode}*.\n\n"
            "Usage: `/replymode true` (only reply on mention) or `/replymode false` (reply to all messages).",
            parse_mode='MarkdownV2'
        )
        return

    new_mode_str = context.args[0].lower()
    if new_mode_str in ["true", "on", "yes"]:
        chat_reply_modes[chat.id] = True
        await update.message.reply_text("✅ Bot will now only reply when mentioned or replied to.")
    elif new_mode_str in ["false", "off", "no"]:
        chat_reply_modes[chat.id] = False
        await update.message.reply_text("📢 Bot will now reply to all messages in the group.")
    else:
        await update.message.reply_text("Invalid option. Please use `true` or `false`.")


@log_user_message
async def tip(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await generate_and_reply(update, context, "Give a short, practical study tip.")

@log_user_message
async def example(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await generate_and_reply(update, context, "Provide a short example question with its answer.")

@log_user_message
async def quiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await generate_and_reply(update, context, "Give a short quiz question with a hidden answer.")

@log_user_message
async def funfact(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await generate_and_reply(update, context, "Share a quick, fun fact about learning.")

@log_user_message
async def rules(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await generate_and_reply(update, context, "Write 4 concise, polite group study rules with emojis.")

@log_user_message
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "Here's what I can do:\n\n"
        "💬 Just chat with me directly with any question!\n\n"
        "Or use these commands:\n"
        "/start - Greeting message\n"
        "/ask <question> - Ask a specific question\n"
        "/tip - Get a study tip\n"
        "/example - See an example question\n"
        "/quiz - Get a mini quiz question\n"
        "/rules - Show group study rules\n"
        "/funfact - Get a fun fact\n\n"
        "🛠️ *Group Commands*:\n"
        "/replymode <true/false> - Set when I reply in groups"
    )
    await update.message.reply_text(help_text, parse_mode='Markdown')

@log_user_message
async def about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("I am a study group assistant bot powered by Google's Gemini AI.")

# -----------------------------
# Main Bot Setup
# -----------------------------
def main():
    root_logger.info("Starting bot...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    command_handlers = {
        "start": start, "help": help_command, "about": about, "ask": ask,
        "tip": tip, "example": example, "quiz": quiz, "funfact": funfact,
        "rules": rules, "replymode": set_reply_mode
    }
    for command, handler in command_handlers.items():
        app.add_handler(CommandHandler(command, handler))
    
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    root_logger.info("Bot is running and polling for updates.")
    app.run_polling()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        root_logger.critical(f"Bot failed to start or crashed: {e}", exc_info=True)