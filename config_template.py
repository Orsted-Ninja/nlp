"""
config.py â€” User settings for the AI/ML Job Alert Telegram Bot
=============================================================
Edit this file OR set environment variables (for GitHub Actions).
Environment variables always take precedence over hardcoded values.
"""

import os

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TELEGRAM CREDENTIALS
# Get your bot token from @BotFather on Telegram.
# Get your chat ID by messaging @userinfobot or @get_id_bot.
# For GitHub Actions: store these as repository Secrets (Settings â†’ Secrets).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
TELEGRAM_BOT_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str   = os.environ.get("TELEGRAM_CHAT_ID",   "")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# KEYWORD FILTERS  (case-insensitive; a job must match at least one keyword)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
KEYWORDS: list[str] = [
    "AI",
    "ML",
    "LLM",
    "Gen AI",
    "GenAI",
    "Python",
    "LangChain",
    "Machine Learning",
    "NLP",
    "Data Science",
    "Data Scientist",
    "AI Engineer",
    "ML Engineer",
    "Prompt Engineer",
    "Deep Learning",
    "Computer Vision",
    "Transformer",
    "RAG",
    "Retrieval Augmented",
    "Hugging Face",
    "TensorFlow",
    "PyTorch",
    "Scikit",
    "MLOps",
    "AI/ML",
    "Generative AI",
    "Foundation Model",
    "Fine-tuning",
    "Reinforcement Learning",
    "RLHF",
]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# FILE PATHS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SEEN_JOBS_FILE: str = os.environ.get("SEEN_JOBS_FILE", "seen_jobs.json")
LOG_FILE: str       = os.environ.get("LOG_FILE",       "job_bot.log")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SCRAPING SETTINGS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Seconds to wait between HTTP requests (be a polite scraper)
REQUEST_DELAY_SECONDS: float = 2.0

# Longer delay for FAANG / high-value corporate career pages
FAANG_DELAY_SECONDS: float = 4.0

# Maximum jobs to send in a single run (prevents Telegram rate-limit floods)
MAX_JOBS_PER_RUN: int = 25

# HTTP request timeout in seconds
REQUEST_TIMEOUT: int = 20

# User-Agent header used in all HTTP requests â€” mimics a real browser
USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# JOB SOURCES
# Each source has:
#   name     â€” display name used in Telegram messages
#   type     â€” scraper type (maps to a function in job_bot.py)
#   url      â€” primary URL / API endpoint
#   enabled  â€” set False to skip without deleting the entry
#   delay    â€” optional per-source delay override in seconds
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SOURCES: list[dict] = [

    # â”€â”€ FREE REMOTE JOB RSS FEEDS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    {
        "name":    "We Work Remotely",
        "type":    "rss",
        "url":     "https://weworkremotely.com/remote-jobs.rss",
        "enabled": True,
    },
    {
        "name":    "RemoteOK",
        "type":    "rss",
        "url":     "https://remoteok.com/remote-jobs.rss",
        "enabled": True,
    },
    {
        "name":    "Arbeitnow Remote",
        "type":    "rss",
        "url":     "https://www.arbeitnow.com/rss",
        "enabled": True,
    },
    {
        "name":    "Jobicy Remote",
        "type":    "rss",
        "url":     "https://jobicy.com/?feed=job_feed",
        "enabled": True,
    },
    {
        "name":    "Remote.co RSS",
        "type":    "rss",
        "url":     "https://remote.co/feed/",
        "enabled": True,
    },
    {
        "name":    "AI Jobs (aijobs.net)",
        "type":    "rss",
        "url":     "https://aijobs.net/feed/",
        "enabled": True,
    },
    {
        "name":    "Remotive Software Dev RSS",
        "type":    "rss",
        "url":     "https://remotive.com/remote-jobs/feed/software-dev",
        "enabled": True,
    },

    # â”€â”€ HACKERNEWS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    {
        "name":    "HackerNews Who's Hiring",
        "type":    "hn_hiring",
        "url":     (
            "https://hn.algolia.com/api/v1/search_by_date"
            "?query=who+is+hiring&tags=story&hitsPerPage=1"
        ),
        "enabled": True,
    },
    {
        "name":    "HN Who Wants to Be Hired",
        "type":    "hn_hiring",
        "url":     (
            "https://hn.algolia.com/api/v1/search_by_date"
            "?query=who+wants+to+be+hired&tags=story&hitsPerPage=1"
        ),
        "enabled": True,
    },

    # â”€â”€ WELLFOUND / ANGELLIST â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    {
        "name":    "Wellfound ML Engineer",
        "type":    "wellfound",
        "url":     "https://wellfound.com/role/r/machine-learning-engineer",
        "enabled": True,
    },
    {
        "name":    "Wellfound AI Engineer",
        "type":    "wellfound",
        "url":     "https://wellfound.com/role/r/artificial-intelligence-engineer",
        "enabled": True,
    },

    # â”€â”€ GITHUB â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    {
        "name":    "GitHub Hiring Repos",
        "type":    "github_hiring",
        "url":     (
            "https://api.github.com/search/repositories"
            "?q=hiring+AI+ML+remote&sort=updated&order=desc&per_page=10"
        ),
        "enabled": True,
    },

    # â”€â”€ INDIAN JOB BOARDS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    {
        "name":    "Naukri.com AI/ML Jobs",
        "type":    "naukri",
        "url":     "https://www.naukri.com/ai-ml-jobs",
        "enabled": True,
        "delay":   4.0,
    },
    {
        "name":    "Cutshort.io AI/ML Jobs",
        "type":    "cutshort",
        "url":     "https://cutshort.io/jobs",
        "enabled": True,
        "delay":   3.0,
    },
    {
        "name":    "Internshala ML Internships",
        "type":    "internshala",
        "url":     "https://internshala.com/jobs/machine-learning-jobs",
        "enabled": True,
        "delay":   3.0,
    },

    # â”€â”€ FAANG + TOP AI COMPANY CAREER PAGES â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    {
        "name":    "Google Careers (ML/India)",
        "type":    "google_careers",
        "url":     (
            "https://careers.google.com/api/v3/search/"
            "?q=machine+learning&location=India&num=20"
        ),
        "enabled": True,
        "delay":   4.0,
    },
    {
        "name":    "Microsoft Careers (AI/ML India)",
        "type":    "microsoft_careers",
        "url":     (
            "https://jobs.microsoft.com/en-us/search"
            "?q=AI+ML&lc=India&exp=Experienced+professionals"
        ),
        "enabled": True,
        "delay":   4.0,
    },
    {
        "name":    "Amazon Jobs (ML India)",
        "type":    "amazon_jobs",
        "url":     (
            "https://www.amazon.jobs/en/search.json"
            "?base_query=machine+learning&loc_query=India"
            "&job_count=20&result_limit=20&sort=recent"
        ),
        "enabled": True,
        "delay":   4.0,
    },
    {
        "name":    "Meta Careers (ML)",
        "type":    "meta_careers",
        "url":     (
            "https://www.metacareers.com/jobs"
            "?q=machine+learning&offices%5B%5D=India"
        ),
        "enabled": True,
        "delay":   5.0,
    },
    {
        "name":    "Apple Jobs (ML India)",
        "type":    "apple_jobs",
        "url":     (
            "https://jobs.apple.com/en-us/search"
            "?search=machine+learning&sort=relevance&location=india-IND"
        ),
        "enabled": True,
        "delay":   4.0,
    },
    {
        "name":    "OpenAI Careers",
        "type":    "openai_careers",
        "url":     "https://openai.com/careers",
        "enabled": True,
        "delay":   4.0,
    },
    {
        "name":    "Anthropic Careers",
        "type":    "anthropic_careers",
        "url":     "https://www.anthropic.com/careers",
        "enabled": True,
        "delay":   4.0,
    },
    {
        "name":    "Hugging Face Jobs (Workable)",
        "type":    "workable",
        "url":     "https://apply.workable.com/api/v1/widget/listing/hugging-face/",
        "enabled": True,
        "delay":   3.0,
    },
    {
        "name":    "NVIDIA Careers (Workday)",
        "type":    "nvidia_careers",
        "url":     (
            "https://nvidia.wd5.myworkdayjobs.com/wday/cxs/nvidia/"
            "NVIDIAExternalCareerSite/jobs"
        ),
        "enabled": True,
        "delay":   5.0,
    },
]
