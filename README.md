# 🤖 TelegramAIBot

> Google Gemini-powered Telegram study assistant with contextual chat memory and an embedded Flask admin control panel.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Google Gemini](https://img.shields.io/badge/AI%20Model-Gemini%202.5%20Flash%20Lite-8E75B2?logo=google&logoColor=white)](https://ai.google.dev/)
[![python-telegram-bot](https://img.shields.io/badge/Bot%20Framework-python--telegram--bot-2CA5E0?logo=telegram&logoColor=white)](https://python-telegram-bot.org/)
[![Flask](https://img.shields.io/badge/Admin%20Panel-Flask-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Status](https://img.shields.io/badge/Status-Active%20%26%20Maintained-brightgreen)](https://github.com/AryansDevStudios/TelegramAIBot)

---

## 📖 Overview

**TelegramAIBot** is an intelligent study companion and group assistant bot for Telegram, powered by Google's fast and capable `gemini-2.5-flash-lite` language model. Designed specifically for students and study groups, it delivers structured, engaging, and concise educational explanations complete with bullet points, practical examples, and Markdown formatting.

Beyond intelligent conversational tutoring, TelegramAIBot incorporates an embedded, password-protected **Flask Web Control Panel** running concurrently on port 8080. This web interface enables administrators to monitor live console logs via Server-Sent Events (SSE), inspect individual user conversation records, securely explore project files, and export complete project backups on the fly.

---

## ✨ Key Features

### 🎓 Telegram Study Companion
- **Gemini 2.5 Flash Lite Engine**: Powered by Google's optimized generative model configured with strict educational system instructions for concise, bulleted, and emoji-enhanced study guidance.
- **Robust MarkdownV2 Safety Net**: Implements an intelligent negative-lookbehind regex escaping filter (`re.sub(r'(?<!\\)([.!+(){}-])', r'\\\1', text)`) combined with `BadRequest` exception fallbacks, preventing Telegram parsing crashes while preserving intentional bold, italic, and inline code formatting.
- **Contextual Conversation Memory**: Maintains a rolling multi-turn dialogue history (`deque(maxlen=20)`) per chat, allowing natural follow-up questions and iterative learning sessions.
- **Curated Educational Commands**:
  - `/start` — Welcoming greeting and quick-start instructions.
  - `/ask <question>` — Direct inquiry prompt for immediate problem solving.
  - `/tip` — Generates a concise, actionable study tip or productivity technique.
  - `/example` — Produces an illustrative academic problem with step-by-step solution.
  - `/quiz` — Generates a practice question with a hidden spoiler answer.
  - `/funfact` — Shares an engaging trivia fact about science, history, or learning.
  - `/rules` — Displays polite, emoji-structured group study conduct rules.
  - `/replymode <true/false>` — Configures group chat behavior: toggle between responding to all messages vs. responding only to direct mentions/replies.
  - `/help` & `/about` — Command reference and bot architecture info.

### 🛡️ Administrative Web Control Panel (Port 8080)
- **Session-Authenticated Login**: Protected by an administrative master password configured via `FLASK_PASSWORD`.
- **Real-Time SSE Log Streaming**: Live-tails the central console log (`logs/console.log`) in a browser-based dark terminal via Server-Sent Events (`/log_stream`).
- **Granular Per-User Logging**: Automatically separates and rotates conversation logs into individual files (`logs/<User_Fullname>_<Chat_ID>.log`) using Python's `RotatingFileHandler` (10 MB cap, 5 backups).
- **Remote File Explorer & Viewer**: Secure file browser (`/files`) with path traversal guards (`os.path.commonpath`), enabling live log inspection (`/view/<filepath>`) and file downloads (`/download/<filepath>`).
- **Safe Project Backup Export**: Creates a downloadable in-memory ZIP archive of the entire repository (`/download_zip`), automatically excluding `.env` to prevent credential exposure.
- **Cloud Keep-Alive Daemon**: Runs a background daemon sending periodic HTTP GET requests to `WEB_REQUEST_URL` every 60 seconds to prevent cold starts on ephemeral hosting platforms (such as Render or Replit).

---

## 🛠️ Tech Stack

- **Language**: [Python 3.10+](https://www.python.org/)
- **AI / LLM API**: [Google Generative AI SDK](https://github.com/google-gemini/generative-ai-python) (`google-generativeai`)
- **Telegram Bot Framework**: [python-telegram-bot](https://python-telegram-bot.org/) (AsyncIO-based `Application`)
- **Web Framework**: [Flask](https://flask.palletsprojects.com/) (Embedded administrative dashboard)
- **Configuration & Environment**: [python-dotenv](https://github.com/theskumar/python-dotenv)
- **Networking & Automation**: [requests](https://requests.readthedocs.io/)
- **Concurrency**: Python `threading` & `asyncio`

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- A Telegram Bot Token from [@BotFather](https://t.me/BotFather)
- A Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AryansDevStudios/TelegramAIBot.git
   cd TelegramAIBot
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Linux/macOS:
   source venv/bin/activate
   # On Windows:
   .\venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Create a `.env` file in the root directory:
   ```env
   TELEGRAM_BOT_TOKEN="your_telegram_bot_token_here"
   GEMINI_API_KEY="your_gemini_api_key_here"
   FLASK_PASSWORD="choose_a_strong_admin_password"
   WEB_REQUEST_URL="https://your-deployment-url.onrender.com"  # Optional keep-alive target
   ```

5. **Run the application**:
   ```bash
   python main.py
   ```

Once started:
- The Telegram bot will begin polling for updates immediately.
- The Flask Control Panel will be accessible at `http://localhost:8080`.

---

## 💻 Web Control Panel Access

1. Open `http://localhost:8080` in your web browser.
2. Enter the password defined in `FLASK_PASSWORD`.
3. Use the navigation sidebar to:
   - **Stream Console**: View real-time log messages as users interact with the bot.
   - **File Explorer**: View user-specific log files, project scripts, and download logs.
   - **Download ZIP**: Export a clean snapshot of the project codebase.

---

## 📂 Project Structure

```
TelegramAIBot/
├── main.py               # Combined Telegram bot service & Flask admin server
├── requirements.txt      # Python package dependencies
├── .env                  # Environment secrets (ignored in Git & ZIP exports)
├── .gitignore            # Git exclusion rules
└── logs/                 # Auto-generated rotating log directories
    ├── console.log       # Master application log (10MB rotation, 5 backups)
    ├── webrequests.log   # Keep-alive heartbeat logs
    └── <name>_<id>.log   # Dedicated per-user chat history logs
```

---

## 🤝 Contributing

Contributions, feedback, and feature suggestions are welcome!
1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/improvement`).
3. Commit your changes (`git commit -m 'feat: Add new study command'`).
4. Push to the branch (`git push origin feature/improvement`).
5. Open a Pull Request.

---

## 📄 License

This project is open-source and available under the terms of the MIT License.
