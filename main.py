import os
import re
import asyncio
import logging
from logging.handlers import RotatingFileHandler
from collections import deque
from dotenv import load_dotenv
import threading
import time
from flask import Flask, render_template_string, Response, jsonify, send_from_directory, send_file, request, redirect, url_for, session
from functools import wraps
import zipfile
import io
import requests

# -----------------------------
# Load Environment Variables & Configuration
# -----------------------------
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FLASK_PASSWORD = os.getenv("FLASK_PASSWORD")
WEB_REQUEST_URL = os.getenv("WEB_REQUEST_URL")

MODEL_NAME = "gemini-1.5-flash"

if not TELEGRAM_TOKEN or not GEMINI_API_KEY or not FLASK_PASSWORD:
    print("FATAL ERROR: TELEGRAM_BOT_TOKEN, GEMINI_API_KEY, and FLASK_PASSWORD must be set in the .env file.")
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
logging.getLogger("werkzeug").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

web_request_logger = logging.getLogger('WebRequestLogger')
web_request_logger.setLevel(logging.INFO)
web_request_logger.propagate = False
web_request_handler = RotatingFileHandler(
    os.path.join(LOGS_DIR, "webrequests.log"), maxBytes=1*1024*1024, backupCount=3, encoding="utf-8"
)
web_request_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
web_request_logger.addHandler(web_request_handler)

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
    import google.generativeai as genai
    from google.api_core import exceptions
    genai.configure(api_key=GEMINI_API_KEY)

    system_instruction = """You are a Telegram bot offering study group help.
- Always use Telegram MarkdownV2 formatting.
- Keep replies short, structured, and engaging.
- Use bullet points, examples, and emojis.
- One or two lines unless a longer answer is explicitly needed.
- Never use LaTeX or unsupported markup.
- Escape reserved characters when needed to avoid formatting errors.

📖 Telegram MarkdownV2 Formatting Guide:
1. *Bold* → `*bold*`
2. _Italic_ → `_italic_`
3. __Underline__ → `__underline__`
4. ~Strikethrough~ → `~strikethrough~`
5. ||Spoiler|| → `||hidden text||`
6. `Inline code` → `` `code` ``
7. Multiline code block → ```\ncode here\n```
8. [Inline link](https://example.com) → `[text](https://example.com)`
9. Mention user → `[Name](tg://user?id=USER_ID)`
10. Escape reserved characters with `\` before these:
   `_ * [ ] ( ) ~ ` > # + - = | { } . !`

⚡ Example:
"Hello *world*\! Visit [Google](https://google.com) for more info\."
"""

    gemini_model = genai.GenerativeModel(
        MODEL_NAME,
        system_instruction=system_instruction
    )
    root_logger.info(f"Gemini AI client configured successfully with model: {MODEL_NAME}")
except Exception as e:
    root_logger.critical(f"Failed to configure Gemini AI: {e}", exc_info=True)
    exit()

# -----------------------------
# Telegram Bot Imports
# -----------------------------
from telegram import Update
from telegram.constants import ChatAction, ChatType
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

# -----------------------------
# Bot State Management (Reply Modes)
# -----------------------------
chat_reply_modes = {}

# -----------------------------
# Gemini & Telegram Interaction Helpers
# -----------------------------
def generate_gemini_answer(prompt: str, chat_history: deque) -> str:
    """
    Generates an answer using Gemini, maintaining a specific chat's history.
    """
    chat_history.append({'role': 'user', 'parts': [prompt]})
    try:
        # Pass the history from the deque to the model
        chat_session = gemini_model.start_chat(history=list(chat_history))
        response = chat_session.send_message(prompt)
        full_response_text = response.text

        if full_response_text.strip():
            # Add the model's response back to the history deque
            chat_history.append({'role': 'model', 'parts': [full_response_text]})

        return full_response_text

    except exceptions.GoogleAPICallError as e:
        root_logger.error(f"Gemini API Call Error: {e}")
        # Remove the user's prompt from history if the API call failed
        chat_history.pop()
        return "API Error: Could not get a response."
    except Exception as e:
        root_logger.error(f"Unexpected error in Gemini generation: {e}", exc_info=True)
        # Remove the user's prompt from history if an unknown error occurred
        chat_history.pop()
        return "An unexpected error occurred."

async def generate_and_reply(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt: str):
    """
    Handles the full process of getting a chat's history, generating a response,
    cleaning it, and replying.
    """
    # Get or create a conversation history for the specific chat
    if 'history' not in context.chat_data:
        context.chat_data['history'] = deque(maxlen=20)
    chat_history = context.chat_data['history']

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    # Generate the answer using the specific chat's history
    raw_answer = await asyncio.to_thread(generate_gemini_answer, prompt, chat_history)

    # **FIX:** Clean the model's output to remove unwanted backslashes
    answer = raw_answer.replace('\\\n', '\n').strip()

    await update.message.reply_text(answer, parse_mode="MarkdownV2")
    user_logger = get_user_logger(update.message.chat.id, update.message.from_user.full_name)
    user_logger.info(f"BOT: {' '.join(answer.splitlines())}")

def log_user_message(func):
    """Decorator to log incoming user messages."""
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
    if chat.type == ChatType.PRIVATE:
        await generate_and_reply(update, context, update.message.text)
        return

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
    if chat.type == ChatType.PRIVATE:
        await update.message.reply_text("This command is only for group chats.")
        return
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
def run_bot():
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
    
    # NOTE ON `telegram.error.Conflict`:
    # This error means another instance of your bot is running with the same token.
    # On hosting platforms like Render, this can happen if a new deployment starts
    # before the old one has completely shut down.
    # SOLUTION: Ensure your hosting service is configured to run ONLY ONE instance
    # of this script at a time. Check your service's dashboard and logs.
    app.run_polling()

# ==============================================================================
# FLASK WEBSERVER
# ==============================================================================
flask_app = Flask(__name__)
flask_app.secret_key = os.urandom(24)
HOME_DIR = os.getcwd()

# --- Web Request Function ---
def send_keep_alive_request():
    """Sends a GET request to the specified URL and logs the result separately."""
    if not WEB_REQUEST_URL:
        web_request_logger.warning("WEB_REQUEST_URL is not set. Skipping request.")
        return

    try:
        response = requests.get(WEB_REQUEST_URL, timeout=10)
        web_request_logger.info(
            f"Sent request to {WEB_REQUEST_URL}. Status: {response.status_code}. Response: {response.text[:100]}"
        )
    except requests.exceptions.RequestException as e:
        web_request_logger.error(f"Failed to send request to {WEB_REQUEST_URL}. Error: {e}")

# --- Login Template ---
LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Login</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto; background-color: #121212; color: #e0e0e0; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
        .login-container { background-color: #1e1e1e; padding: 40px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.5); text-align: center; }
        h1 { color: #fff; }
        input[type="password"] { width: 80%; padding: 10px; margin-top: 20px; border-radius: 5px; border: 1px solid #333; background-color: #222; color: #fff; }
        button { background-color: #bb86fc; color: #121212; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; transition: background-color 0.2s; margin-top: 20px; font-weight: bold; }
        button:hover { background-color: #a063f0; }
        .error { color: #cf6679; margin-top: 15px; }
    </style>
</head>
<body>
    <div class="login-container">
        <h1>Control Panel Access</h1>
        <form method="post">
            <input type="password" name="password" placeholder="Password" required>
            <br>
            <button type="submit">Login</button>
        </form>
        {% if error %}
            <p class="error">{{ error }}</p>
        {% endif %}
    </div>
</body>
</html>
"""

# --- HTML & JS TEMPLATE (Main Panel) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bot Control Panel</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; background-color: #121212; color: #e0e0e0; display: flex; height: 100vh; }
        .sidebar { width: 300px; background-color: #1e1e1e; padding: 20px; border-right: 1px solid #333; overflow-y: auto; display: flex; flex-direction: column; }
        .main-content { flex-grow: 1; display: flex; flex-direction: column; }
        .log-container { flex-grow: 1; background-color: #181818; padding: 20px; overflow-y: auto; font-family: 'Courier New', Courier, monospace; font-size: 14px; white-space: pre-wrap; }
        .top-bar { padding: 10px 20px; background-color: #1e1e1e; border-bottom: 1px solid #333; display: flex; justify-content: space-between; align-items: center; }
        h1, h2 { color: #ffffff; border-bottom: 1px solid #444; padding-bottom: 10px; }
        h1 { margin-top: 0; }
        button { background-color: #333; color: #fff; border: none; padding: 10px 15px; border-radius: 5px; cursor: pointer; transition: background-color 0.2s; margin-bottom: 15px; }
        button:hover { background-color: #555; }
        .logout-btn { background-color: #cf6679; margin-top: auto; }
        .logout-btn:hover { background-color: #b05260; }
        ul { list-style: none; padding: 0; }
        li { margin: 5px 0; }
        a { color: #bb86fc; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .file { color: #90caf9; }
        .dir { color: #a5d6a7; font-weight: bold; }
        .file-viewer { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); display: none; justify-content: center; align-items: center; }
        .file-viewer-content { background: #1e1e1e; color: #e0e0e0; width: 80%; height: 80%; padding: 20px; overflow: auto; border: 1px solid #333; font-family: 'Courier New', Courier, monospace;}
        .close-btn { position: absolute; top: 20px; right: 30px; font-size: 30px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="sidebar">
        <div>
            <h1>File Explorer</h1>
            <h2>Root: {{ home_dir }}</h2>
            <a href="/download_zip"><button>Download All as ZIP</button></a>
            <div id="file-list"></div>
        </div>
        <a href="/logout"><button class="logout-btn">Logout</button></a>
    </div>
    <div class="main-content">
        <div class="top-bar">
            <h2>Live Console Log</h2>
            <button id="copy-log">Copy Log</button>
        </div>
        <div class="log-container" id="log-content"></div>
    </div>
    <div class="file-viewer" id="file-viewer">
        <span class="close-btn" onclick="closeFileViewer()">&times;</span>
        <pre id="file-viewer-content" class="file-viewer-content"></pre>
    </div>

    <script>
        // --- Live Log Streaming ---
        const logContent = document.getElementById('log-content');
        const eventSource = new EventSource('/log_stream');
        eventSource.onmessage = function(event) {
            logContent.innerHTML += event.data + '<br>';
            logContent.scrollTop = logContent.scrollHeight;
        };

        // --- Copy Log Button ---
        document.getElementById('copy-log').addEventListener('click', () => {
            navigator.clipboard.writeText(logContent.innerText).then(() => {
                alert('Log copied to clipboard!');
            });
        });

        // --- File Browser ---
        function loadFiles(path = '') {
            fetch(`/files?path=${encodeURIComponent(path)}`)
                .then(response => {
                    if (response.status === 401) {
                        window.location.href = '/login';
                        return;
                    }
                    return response.json();
                })
                .then(data => {
                    if (!data) return;
                    const fileList = document.getElementById('file-list');
                    fileList.innerHTML = '';
                    if (path) {
                        const parentPath = path.substring(0, path.lastIndexOf('/'));
                        const upLink = document.createElement('a');
                        upLink.href = '#';
                        upLink.className = 'dir';
                        upLink.textContent = '[..]';
                        upLink.onclick = (e) => { e.preventDefault(); loadFiles(parentPath); };
                        fileList.appendChild(document.createElement('li')).appendChild(upLink);
                    }
                    data.dirs.forEach(dir => {
                        const li = document.createElement('li');
                        const link = document.createElement('a');
                        link.href = '#';
                        link.className = 'dir';
                        link.textContent = dir + '/';
                        link.onclick = (e) => { e.preventDefault(); loadFiles((path ? path + '/' : '') + dir); };
                        li.appendChild(link);
                        fileList.appendChild(li);
                    });
                    data.files.forEach(file => {
                        const li = document.createElement('li');
                        const fullPath = (path ? path + '/' : '') + file;
                        
                        const viewLink = document.createElement('a');
                        viewLink.href = '#';
                        viewLink.className = 'file';
                        viewLink.textContent = file;
                        viewLink.onclick = (e) => { e.preventDefault(); viewFile(fullPath); };
                        
                        const downloadLink = document.createElement('a');
                        downloadLink.href = `/download/${fullPath}`;
                        downloadLink.textContent = ' (download)';
                        downloadLink.style.fontSize = '0.8em';

                        li.appendChild(viewLink);
                        li.appendChild(downloadLink);
                        fileList.appendChild(li);
                    });
                });
        }
        
        let fileEventSource = null;

        function viewFile(filePath) {
            const viewer = document.getElementById('file-viewer');
            const content = document.getElementById('file-viewer-content');
            viewer.style.display = 'flex';
            content.textContent = 'Loading...';

            if (fileEventSource) {
                fileEventSource.close();
            }

            fileEventSource = new EventSource(`/view/${filePath}`);
            let fullContent = '';
            fileEventSource.onmessage = function(event) {
                if (event.data === '___EOF___') {
                    fileEventSource.close();
                    return;
                }
                fullContent += event.data + '\\n';
                content.textContent = fullContent;
            };
            fileEventSource.onerror = function() {
                content.textContent = 'Error loading file. It may be binary or unreadable.';
                fileEventSource.close();
            }
        }

        function closeFileViewer() {
            if (fileEventSource) {
                fileEventSource.close();
            }
            document.getElementById('file-viewer').style.display = 'none';
        }

        document.addEventListener('DOMContentLoaded', () => loadFiles());
    </script>
</body>
</html>
"""

# --- Login required decorator ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            if request.path.startswith(('/files', '/log_stream', '/view')):
                 return jsonify({"error": "Not authenticated"}), 401
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# --- Flask Routes ---
@flask_app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form.get('password') == FLASK_PASSWORD:
            session['logged_in'] = True
            root_logger.info("Successful login to web panel.")
            next_url = request.args.get('next')
            return redirect(next_url or url_for('index'))
        else:
            error = 'Invalid password. Please try again.'
            root_logger.warning("Failed login attempt to web panel.")
    return render_template_string(LOGIN_TEMPLATE, error=error)

@flask_app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@flask_app.route('/')
@login_required
def index():
    threading.Thread(target=send_keep_alive_request).start()
    return render_template_string(HTML_TEMPLATE, home_dir=HOME_DIR)

@flask_app.route('/log_stream')
@login_required
def log_stream():
    log_file_path = os.path.join(LOGS_DIR, "console.log")
    INITIAL_LOG_LINES = 500

    def generate():
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                initial_lines = deque(f, maxlen=INITIAL_LOG_LINES)
                for line in initial_lines:
                    yield f"data: {line.strip()}\\n\\n"
                
                while True:
                    line = f.readline()
                    if not line:
                        time.sleep(0.1)
                        continue
                    yield f"data: {line.strip()}\\n\\n"
        except FileNotFoundError:
             yield f"data: ERROR: Log file not found at {log_file_path}\\n\\n"

    return Response(generate(), mimetype='text/event-stream')
    
@flask_app.route('/files')
@login_required
def list_files():
    req_path = request.args.get('path', '')
    base_path = os.path.join(HOME_DIR, req_path.strip('/'))
    
    if not os.path.abspath(base_path).startswith(os.path.abspath(HOME_DIR)):
        return jsonify({"error": "Access denied"}), 403

    dirs = []
    files = []
    try:
        if os.path.isdir(base_path):
            for item in os.listdir(base_path):
                if os.path.isdir(os.path.join(base_path, item)):
                    dirs.append(item)
                else:
                    files.append(item)
    except FileNotFoundError:
        return jsonify({"error": "Directory not found"}), 404
        
    return jsonify({"path": req_path, "dirs": sorted(dirs), "files": sorted(files)})

@flask_app.route('/view/<path:filepath>')
@login_required
def view_file(filepath):
    abs_path = os.path.join(HOME_DIR, filepath.strip('/'))
    if not os.path.abspath(abs_path).startswith(os.path.abspath(HOME_DIR)):
        return "Access Denied", 403

    def generate():
        try:
            with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
                while True:
                    line = f.readline()
                    if not line:
                        yield 'data: ___EOF___\n\n'
                        break
                    yield f'data: {line.rstrip()}\n\n'
        except Exception as e:
            yield f'data: Error reading file: {str(e)}\n\n'
            yield 'data: ___EOF___\n\n'

    return Response(generate(), mimetype='text/event-stream')

@flask_app.route('/download/<path:filepath>')
@login_required
def download_file(filepath):
    safe_full_path = os.path.abspath(os.path.join(HOME_DIR, filepath))
    
    if not safe_full_path.startswith(os.path.abspath(HOME_DIR)):
        return "Access Denied: You cannot access files outside the home directory.", 403
        
    try:
        directory = os.path.dirname(safe_full_path)
        filename = os.path.basename(safe_full_path)
        return send_from_directory(directory, filename, as_attachment=True)
    except FileNotFoundError:
        return "File not found.", 404
    except Exception as e:
        root_logger.error(f"Error during file download: {e}", exc_info=True)
        return "An error occurred while trying to download the file.", 500
    
@flask_app.route('/download_zip')
@login_required
def download_zip():
    memory_file = io.BytesIO()
    excluded_files = ['.env']
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(HOME_DIR):
            for file in files:
                if file in excluded_files:
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, HOME_DIR)
                zf.write(file_path, arcname)
    memory_file.seek(0)
    return send_file(memory_file, download_name='project_archive.zip', as_attachment=True)

def run_flask():
    root_logger.info("Starting Flask web server...")
    flask_app.run(host='0.0.0.0', port=8080, use_reloader=False)


# ==============================================================================
# SCRIPT EXECUTION
# ==============================================================================
if __name__ == "__main__":
    try:
        flask_thread = threading.Thread(target=run_flask)
        flask_thread.daemon = True
        flask_thread.start()
        run_bot()
    except Exception as e:
        root_logger.critical(f"Application failed to start or crashed: {e}", exc_info=True)