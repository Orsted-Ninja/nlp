# 🤖 AI/ML Job Alert Telegram Bot

Automatically scrapes **20+ free job sources** every 6 hours and sends filtered AI/ML job alerts directly to your Telegram chat — 100 % free, no paid APIs required.

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Sources** | 20+ sources: RSS feeds, HN Hiring, Naukri, Cutshort, Internshala, Google, Microsoft, Amazon, Meta, Apple, OpenAI, Anthropic, Hugging Face, NVIDIA |
| **Filter keywords** | AI, ML, LLM, GenAI, Python, LangChain, NLP, MLOps, Prompt Engineer, RAG, PyTorch, RLHF, … |
| **Deduplication** | `seen_jobs.json` committed back to repo — no repeated alerts across runs |
| **Schedule** | Every 6 hours via GitHub Actions free tier |
| **Dry-run mode** | `python job_bot.py --test` — prints jobs to console, no Telegram send |
| **Graceful errors** | If any single source fails/blocks, the bot logs it and continues |
| **Cost** | $0 — entirely free |

---

## 🚀 Quick Setup (15 minutes)

### Step 1 — Create a Telegram Bot

1. Open Telegram → search **@BotFather**
2. Send `/newbot` and follow the prompts
3. BotFather gives you a **bot token** like:
   ```
   7123456789:AAHdqTcvCH1vGWJxfSeofSs0K7sCCIhkgmA
   ```
4. Copy and save it.

### Step 2 — Get Your Chat ID

**Personal chat:**
1. Start a chat with your new bot → send `/start`
2. Message **@userinfobot** — it replies with your numeric user ID (e.g. `123456789`)

**Group / channel:**
1. Add your bot to the group as admin
2. Send any message, then visit:
   ```
   https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
   ```
3. Find `"chat":{"id": -100xxxxxxxxxx}` — that negative number is your chat ID.

### Step 3a — GitHub Actions (recommended)

1. **Fork** this repository
2. **Settings → Secrets and variables → Actions → New repository secret**

   | Secret name | Value |
   |---|---|
   | `TELEGRAM_BOT_TOKEN` | your token from Step 1 |
   | `TELEGRAM_CHAT_ID` | your chat ID from Step 2 |

3. Enable Actions: **Actions tab → "I understand my workflows…"**
4. First manual trigger: **Actions → AI/ML Job Alert Bot → Run workflow**

That's it — the bot runs automatically every 6 hours. ✅

### Step 3b — Run Locally

```bash
# 1. Go to the project folder
cd F:\LEARN\job_alert_bot       # Windows
# cd ~/job_alert_bot            # macOS/Linux

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set credentials
# Windows PowerShell:
$env:TELEGRAM_BOT_TOKEN = "7123456789:AAH..."
$env:TELEGRAM_CHAT_ID   = "123456789"
# macOS/Linux:
# export TELEGRAM_BOT_TOKEN="..."
# export TELEGRAM_CHAT_ID="..."

# 5. Dry-run first (no Telegram messages sent)
python job_bot.py --test

# 6. Full run
python job_bot.py
```

---

## 🧪 --test / Dry-Run Mode

```bash
python job_bot.py --test
```

- Scrapes **all enabled sources**
- Prints matching jobs to the **console**
- Does **NOT** send any Telegram messages
- Does **NOT** update `seen_jobs.json`
- No credentials required — perfect for first-time testing

---

## ⚙️ Customisation (`config.py`)

```python
# Add/remove keywords — case-insensitive
KEYWORDS = ["AI", "ML", "LLM", "GenAI", "RLHF", "RAG", ...]

# Enable or disable individual sources
SOURCES = [
    {"name": "Naukri.com",  "type": "naukri",  "url": "...", "enabled": True},
    {"name": "Apple Jobs",  "type": "apple_jobs", "url": "...", "enabled": False},
    ...
]

# Cap the number of Telegram messages per run (default 25)
MAX_JOBS_PER_RUN = 25

# Global delay between requests
REQUEST_DELAY_SECONDS = 2.0

# Delay for FAANG / corporate career pages
FAANG_DELAY_SECONDS = 4.0
```

---

## 📁 File Structure

```
job_alert_bot/
├── job_bot.py                  ← main bot (entry point)
├── config.py                   ← all user settings & source list
├── requirements.txt            ← pip dependencies (3 packages)
├── seen_jobs.json              ← deduplication store (auto-updated)
├── job_bot.log                 ← runtime log
└── .github/
    └── workflows/
        └── job_alert.yml       ← GitHub Actions cron workflow
```

---

## 🔍 Job Sources (20+)

### Free Remote RSS Feeds
| Source | URL |
|---|---|
| We Work Remotely | weworkremotely.com/remote-jobs.rss |
| RemoteOK | remoteok.com/remote-jobs.rss |
| Arbeitnow | arbeitnow.com/rss |
| Jobicy | jobicy.com/?feed=job_feed |
| Remote.co | remote.co/feed/ |
| AI Jobs | aijobs.net/feed/ |
| Remotive | remotive.com/remote-jobs/feed/software-dev |

### APIs & Specialised Scrapers
| Source | Method |
|---|---|
| HackerNews Who's Hiring | Algolia API — free |
| GitHub Hiring Repos | GitHub Search API — 60 req/h free |
| Hugging Face Jobs | Workable public API |

### Indian Job Boards
| Source | Method |
|---|---|
| Naukri.com | JSON-LD + HTML card scraping |
| Cutshort.io | JSON-LD + HTML fallback |
| Internshala | HTML card scraping |

### FAANG & Top AI Company Career Pages
| Source | Method |
|---|---|
| Google Careers | Public JSON API |
| Microsoft Careers | Embedded JSON + HTML fallback |
| Amazon Jobs | Public `/search.json` endpoint |
| Meta Careers | JSON-LD + `__initialData__` extraction |
| Apple Jobs | HTML card scraping |
| OpenAI Careers | JSON-LD + Greenhouse link scraping |
| Anthropic Careers | JSON-LD + Greenhouse link scraping |
| NVIDIA Careers | Workday POST API |

---

## 🛡️ How Deduplication Works

1. Every job gets an **MD5 fingerprint** of `url + title`
2. After each run, fingerprints are saved to `seen_jobs.json`
3. GitHub Actions **commits** the updated file back with `[skip ci]` (prevents loop)
4. Next run **skips already-seen jobs**
5. Records older than **30 days** are automatically pruned

---

## 🐛 Troubleshooting

| Problem | Fix |
|---|---|
| *"CONFIG ERROR: TELEGRAM_BOT_TOKEN is not set"* | Add the secret in GitHub or export env var locally |
| Bot sends nothing | Run `python job_bot.py --test` and check console output |
| Source returns 0 jobs | Some sites block scrapers intermittently — check `job_bot.log` |
| Duplicate alerts | The `seen_jobs.json` commit step may have failed — check Actions log |
| GitHub API rate limit | Set a `GITHUB_TOKEN` personal access token secret (5000 req/h) |
| FAANG pages blocked | Those sources degrade gracefully — bot continues with other sources |
| Too many / few alerts | Adjust `KEYWORDS` and `MAX_JOBS_PER_RUN` in `config.py` |

---

## 📄 License

MIT — free to use, modify, and distribute.
