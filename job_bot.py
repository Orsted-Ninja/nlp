"""
job_bot.py — AI/ML Job Alert Telegram Bot (v2)
===============================================
Scrapes 20+ free job sources (RSS, API, HTML), filters by AI/ML keywords,
deduplicates via seen_jobs.json, and sends formatted Telegram alerts.

Usage:
    python job_bot.py           # normal run — scrape, filter, send
    python job_bot.py --test    # dry run — print jobs to console, NO Telegram send
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlencode

import requests
from bs4 import BeautifulSoup

# ── project config ────────────────────────────────────────────────────────────
import config

# ── Windows UTF-8 fix (cp1252 terminal can't render Unicode arrows/emojis) ───
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("job_bot")



# ══════════════════════════════════════════════════════════════════════════════
# CLI ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI/ML Job Alert Bot")
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Dry-run mode: scrape all sources and print matching jobs to the console "
            "WITHOUT sending any Telegram messages or updating seen_jobs.json."
        ),
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# DATA MODEL
# ══════════════════════════════════════════════════════════════════════════════

class Job:
    """Normalised job listing."""

    __slots__ = ("title", "company", "location", "salary", "url", "source", "uid")

    def __init__(
        self,
        *,
        title: str,
        company: str = "Unknown",
        location: str = "Remote",
        salary: str = "",
        url: str,
        source: str,
    ) -> None:
        self.title    = self._clean(title)
        self.company  = self._clean(company)
        self.location = self._clean(location)
        self.salary   = self._clean(salary)
        self.url      = url.strip()
        self.source   = source
        # Stable fingerprint — URL + title (MD5 is fine; not used for security)
        self.uid = hashlib.md5(f"{self.url}|{self.title}".encode()).hexdigest()

    @staticmethod
    def _clean(text: str) -> str:
        """Strip HTML tags and normalise whitespace."""
        text = BeautifulSoup(str(text), "html.parser").get_text()
        return " ".join(text.split())

    def matches_keywords(self, keywords: list[str]) -> bool:
        haystack = f"{self.title} {self.company} {self.location}".lower()
        return any(kw.lower() in haystack for kw in keywords)

    def to_telegram_message(self) -> str:
        lines: list[str] = [
            f"🤖 *{self._esc(self.title)}*",
            f"🏢 {self._esc(self.company)}",
            f"📍 {self._esc(self.location)}",
        ]
        if self.salary:
            lines.append(f"💰 {self._esc(self.salary)}")
        lines += [
            f"🔗 [Apply Here]({self.url})",
            f"📡 _Source: {self._esc(self.source)}_",
        ]
        return "\n".join(lines)

    def to_console_str(self) -> str:
        parts = [
            f"  Title   : {self.title}",
            f"  Company : {self.company}",
            f"  Location: {self.location}",
        ]
        if self.salary:
            parts.append(f"  Salary  : {self.salary}")
        parts += [
            f"  URL     : {self.url}",
            f"  Source  : {self.source}",
        ]
        return "\n".join(parts)

    @staticmethod
    def _esc(text: str) -> str:
        """Minimal Markdown v1 escape — protect * _ ` ["""
        for ch in ("*", "_", "`", "["):
            text = text.replace(ch, "\\" + ch)
        return text

    def __repr__(self) -> str:
        return f"<Job uid={self.uid[:8]} title={self.title!r}>"


# ══════════════════════════════════════════════════════════════════════════════
# HTTP HELPERS
# ══════════════════════════════════════════════════════════════════════════════

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": config.USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
})

_PERMANENT_ERRORS = {400, 401, 403, 404, 410, 451}


def safe_get(
    url: str,
    *,
    params: dict | None = None,
    extra_headers: dict | None = None,
    json_mode: bool = False,
    delay: float | None = None,
) -> Any:
    """GET with retry + configurable rate-limiting.

    Returns parsed JSON dict/list when json_mode=True, else a Response object.
    Returns None on permanent failure.
    """
    wait = delay if delay is not None else config.REQUEST_DELAY_SECONDS
    for attempt in range(1, 4):
        try:
            time.sleep(wait)
            headers = {**(extra_headers or {})}
            resp = SESSION.get(
                url,
                params=params,
                headers=headers,
                timeout=config.REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json() if json_mode else resp
        except requests.exceptions.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else 0
            log.warning("HTTP %d for %s (attempt %d)", code, url, attempt)
            if code in _PERMANENT_ERRORS:
                return None
        except requests.exceptions.RequestException as exc:
            log.warning("Request error for %s (attempt %d): %s", url, attempt, exc)
        time.sleep(attempt * 5)
    return None


def extract_json_from_script(soup: BeautifulSoup, pattern: str) -> Any:
    """Find a <script> tag containing `pattern` and extract the first JSON object/array."""
    for tag in soup.find_all("script"):
        text = tag.string or ""
        if pattern in text:
            # Find first JSON object or array
            for start_char, end_char in [('{', '}'), ('[', ']')]:
                start = text.find(start_char)
                if start == -1:
                    continue
                # Walk to find matching close bracket
                depth = 0
                for i, ch in enumerate(text[start:], start):
                    if ch == start_char:
                        depth += 1
                    elif ch == end_char:
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(text[start:i + 1])
                            except json.JSONDecodeError:
                                break
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ── SCRAPER 1: Generic RSS / Atom ────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _parse_rss_item(item: ET.Element, source: str) -> Job | None:
    def tag(name: str) -> str:
        el = item.find(name)
        return (el.text or "").strip() if el is not None else ""

    title   = tag("title")
    link    = tag("link") or tag("guid")
    company = tag("author") or ""
    desc    = tag("description")

    if not company:
        m = re.search(r"\bat\s+([^(|\n]+)", title, re.IGNORECASE)
        company = m.group(1).strip() if m else "Unknown"

    salary = ""
    sal_m  = re.search(
        r"(\$[\d,]+(?:\s*[-–]\s*\$[\d,]+)?(?:\s*[kK])?(?:/yr|/year|/mo|/month)?)",
        f"{title} {desc}",
    )
    if sal_m:
        salary = sal_m.group(1)

    if not title or not link:
        return None
    return Job(title=title, company=company, url=link, source=source, salary=salary)


def scrape_rss(url: str, source: str, *, delay: float | None = None) -> list[Job]:
    log.info("RSS ▶ %s", source)
    resp = safe_get(url, delay=delay)
    if resp is None:
        return []
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as exc:
        log.error("XML parse error for %s: %s", url, exc)
        return []

    items = root.findall(".//item")
    if not items:
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        items = root.findall(".//atom:entry", ns)

    jobs: list[Job] = []
    for item in items:
        job = _parse_rss_item(item, source)
        if job:
            jobs.append(job)

    log.info("  → %d jobs from %s", len(jobs), source)
    return jobs


# ══════════════════════════════════════════════════════════════════════════════
# ── SCRAPER 2: HackerNews Who's Hiring ───────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def scrape_hn_hiring(url: str, source: str, *, delay: float | None = None) -> list[Job]:
    log.info("HN ▶ %s", source)
    data = safe_get(url, json_mode=True, delay=delay)
    if not data or not data.get("hits"):
        return []

    story    = data["hits"][0]
    story_id = story.get("objectID") or story.get("story_id")
    if not story_id:
        return []

    log.info("  HN story ID: %s  title: %s", story_id, story.get("title", ""))

    comments_url = (
        f"https://hn.algolia.com/api/v1/search_by_date"
        f"?tags=comment,story_{story_id}&hitsPerPage=200"
    )
    cdata = safe_get(comments_url, json_mode=True, delay=delay)
    if not cdata:
        return []

    jobs: list[Job] = []
    hn_base = "https://news.ycombinator.com/item?id="

    for hit in cdata.get("hits", []):
        comment_text = hit.get("comment_text", "") or ""
        obj_id       = hit.get("objectID", "")
        if not comment_text:
            continue

        clean      = BeautifulSoup(comment_text, "html.parser").get_text(" ")
        first_line = clean.split("\n")[0][:200]
        parts      = [p.strip() for p in re.split(r"\|", first_line)]

        company    = parts[0] if len(parts) > 0 else "Unknown"
        title      = parts[1] if len(parts) > 1 else first_line
        location   = parts[2] if len(parts) > 2 else "Remote"
        salary_raw = next((p for p in parts if "$" in p or "USD" in p), "")

        jobs.append(Job(
            title=title or first_line,
            company=company,
            location=location,
            salary=salary_raw,
            url=f"{hn_base}{obj_id}",
            source=source,
        ))

    log.info("  → %d HN comments from story %s", len(jobs), story_id)
    return jobs


# ══════════════════════════════════════════════════════════════════════════════
# ── SCRAPER 3: Wellfound (AngelList) ─────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def scrape_wellfound(url: str, source: str, *, delay: float | None = None) -> list[Job]:
    log.info("Wellfound ▶ %s", source)
    resp = safe_get(url, delay=delay)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    jobs: list[Job] = []
    seen_urls: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/jobs/" not in href and "/role/" not in href:
            continue
        full_url = urljoin("https://wellfound.com", href)
        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)

        title = a.get_text(strip=True)
        if not (4 < len(title) < 120):
            continue

        parent  = a.find_parent(["li", "div", "article"])
        company = "Unknown"
        if parent:
            c_el = parent.find(class_=re.compile(r"company|startup", re.I))
            if c_el:
                company = c_el.get_text(strip=True)

        jobs.append(Job(title=title, company=company, url=full_url, source=source))

    log.info("  → %d jobs from Wellfound", len(jobs))
    return jobs


# ══════════════════════════════════════════════════════════════════════════════
# ── SCRAPER 4: GitHub Repos ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def scrape_github_hiring(url: str, source: str, *, delay: float | None = None) -> list[Job]:
    log.info("GitHub ▶ %s", source)
    gh_token = os.environ.get("GITHUB_TOKEN", "")
    headers: dict[str, str] = {}
    if gh_token:
        headers["Authorization"] = f"token {gh_token}"

    try:
        resp = SESSION.get(url, headers=headers, timeout=config.REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning("GitHub API error: %s", exc)
        return []

    jobs: list[Job] = []
    for repo in data.get("items", []):
        title   = repo.get("description") or repo.get("name", "")
        company = repo.get("owner", {}).get("login", "Unknown")
        url_    = repo.get("html_url", "")
        if not title or not url_:
            continue
        jobs.append(Job(title=title, company=company, url=url_, source=source))

    log.info("  → %d repos from GitHub", len(jobs))
    return jobs


# ══════════════════════════════════════════════════════════════════════════════
# ── SCRAPER 5: Naukri.com ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def scrape_naukri(url: str, source: str, *, delay: float | None = None) -> list[Job]:
    """
    Naukri serves job listings as JSON inside a <script id="jsonLD"> tag
    (JSON-LD / schema.org JobPosting).  Falls back to HTML card parsing.
    """
    log.info("Naukri ▶ %s", source)
    resp = safe_get(
        url,
        extra_headers={
            "Accept": "text/html,application/xhtml+xml",
            "Referer": "https://www.naukri.com/",
        },
        delay=delay or config.FAANG_DELAY_SECONDS,
    )
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    jobs: list[Job] = []

    # ── Strategy A: JSON-LD embedded in page ────────────────────────────────
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, AttributeError):
            continue
        if isinstance(data, list):
            items = data
        elif data.get("@type") == "ItemList":
            items = data.get("itemListElement", [])
        else:
            items = [data]

        for item in items:
            posting = item.get("item", item)
            if posting.get("@type") != "JobPosting":
                continue
            title   = posting.get("title", "")
            org     = posting.get("hiringOrganization", {})
            company = org.get("name", "Unknown") if isinstance(org, dict) else str(org)
            loc_raw = posting.get("jobLocation", {})
            location = ""
            if isinstance(loc_raw, dict):
                addr = loc_raw.get("address", {})
                location = addr.get("addressLocality", "") or addr.get("addressRegion", "India")
            salary_raw = posting.get("baseSalary", "")
            salary = ""
            if isinstance(salary_raw, dict):
                val = salary_raw.get("value", {})
                if isinstance(val, dict):
                    lo = val.get("minValue", "")
                    hi = val.get("maxValue", "")
                    currency = salary_raw.get("currency", "INR")
                    if lo or hi:
                        salary = f"{currency} {lo}–{hi}"
            apply_url = posting.get("url", posting.get("sameAs", url))
            if title and apply_url:
                jobs.append(Job(
                    title=title, company=company, location=location or "India",
                    salary=salary, url=apply_url, source=source,
                ))

    if jobs:
        log.info("  → %d jobs from Naukri (JSON-LD)", len(jobs))
        return jobs

    # ── Strategy B: HTML card scraping (fallback) ────────────────────────────
    for card in soup.find_all(attrs={"data-job-id": True}):
        title_el   = card.find(class_=re.compile(r"title", re.I))
        company_el = card.find(class_=re.compile(r"comp-name|company", re.I))
        loc_el     = card.find(class_=re.compile(r"location|loc-wrap", re.I))
        link_el    = card.find("a", href=True)

        title   = title_el.get_text(strip=True)   if title_el   else ""
        company = company_el.get_text(strip=True)  if company_el else "Unknown"
        location= loc_el.get_text(strip=True)      if loc_el     else "India"
        link    = link_el["href"]                  if link_el    else url
        if not title:
            continue
        if not link.startswith("http"):
            link = urljoin("https://www.naukri.com", link)
        jobs.append(Job(title=title, company=company, location=location, url=link, source=source))

    log.info("  → %d jobs from Naukri (HTML cards)", len(jobs))
    return jobs


# ══════════════════════════════════════════════════════════════════════════════
# ── SCRAPER 6: Cutshort.io ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def scrape_cutshort(url: str, source: str, *, delay: float | None = None) -> list[Job]:
    """Cutshort public listing page — parse job cards from HTML."""
    log.info("Cutshort ▶ %s", source)
    resp = safe_get(url, delay=delay or config.REQUEST_DELAY_SECONDS)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    jobs: list[Job] = []

    # Try JSON-LD first
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, AttributeError):
            continue
        entries = data if isinstance(data, list) else [data]
        for entry in entries:
            if entry.get("@type") != "JobPosting":
                continue
            title   = entry.get("title", "")
            org     = entry.get("hiringOrganization", {})
            company = org.get("name", "Unknown") if isinstance(org, dict) else "Unknown"
            apply_url = entry.get("url", url)
            if title:
                jobs.append(Job(title=title, company=company, url=apply_url, source=source))

    if jobs:
        log.info("  → %d jobs from Cutshort (JSON-LD)", len(jobs))
        return jobs

    # Fallback HTML
    for card in soup.find_all(class_=re.compile(r"job[-_]?card|position|listing", re.I)):
        title_el   = card.find(["h2", "h3", "h4"])
        company_el = card.find(class_=re.compile(r"company|org", re.I))
        link_el    = card.find("a", href=True)
        title   = title_el.get_text(strip=True)   if title_el   else ""
        company = company_el.get_text(strip=True)  if company_el else "Unknown"
        link    = link_el["href"]                  if link_el    else url
        if not title:
            continue
        if not link.startswith("http"):
            link = urljoin("https://cutshort.io", link)
        jobs.append(Job(title=title, company=company, url=link, source=source))

    log.info("  → %d jobs from Cutshort (HTML)", len(jobs))
    return jobs


# ══════════════════════════════════════════════════════════════════════════════
# ── SCRAPER 7: Internshala ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def scrape_internshala(url: str, source: str, *, delay: float | None = None) -> list[Job]:
    """Parse Internshala's public job/internship listing cards."""
    log.info("Internshala ▶ %s", source)
    resp = safe_get(url, delay=delay or config.REQUEST_DELAY_SECONDS)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    jobs: list[Job] = []

    for card in soup.find_all("div", class_=re.compile(r"internship_meta|job[-_]internship", re.I)):
        title_el   = card.find(class_=re.compile(r"profile|title", re.I))
        company_el = card.find(class_=re.compile(r"company[-_]name", re.I))
        loc_el     = card.find(class_=re.compile(r"location-names|location_name", re.I))
        link_el    = card.find("a", href=True)

        title   = title_el.get_text(strip=True)   if title_el   else ""
        company = company_el.get_text(strip=True)  if company_el else "Unknown"
        location= loc_el.get_text(strip=True)      if loc_el     else "India"
        link    = link_el["href"]                  if link_el    else url
        if not title:
            continue
        if not link.startswith("http"):
            link = urljoin("https://internshala.com", link)
        jobs.append(Job(title=title, company=company, location=location, url=link, source=source))

    log.info("  → %d jobs from Internshala", len(jobs))
    return jobs


# ══════════════════════════════════════════════════════════════════════════════
# ── SCRAPER 8: Google Careers API ────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def scrape_google_careers(url: str, source: str, *, delay: float | None = None) -> list[Job]:
    """
    Google Careers exposes a JSON search API at:
    https://careers.google.com/api/v3/search/?q=...&location=...&num=N
    """
    log.info("Google Careers ▶ %s", source)
    data = safe_get(url, json_mode=True, delay=delay or config.FAANG_DELAY_SECONDS)
    if not data:
        return []

    jobs: list[Job] = []
    for job_item in data.get("jobs", []):
        title    = job_item.get("title", "")
        locs     = job_item.get("locations", [])
        location = ", ".join(locs) if locs else "India"
        job_id   = job_item.get("id", "")
        apply_url = f"https://careers.google.com/jobs/results/{job_id}/" if job_id else url

        jobs.append(Job(
            title=title or "Google Role",
            company="Google",
            location=location,
            url=apply_url,
            source=source,
        ))

    log.info("  → %d jobs from Google Careers", len(jobs))
    return jobs


# ══════════════════════════════════════════════════════════════════════════════
# ── SCRAPER 9: Microsoft Careers ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def scrape_microsoft_careers(url: str, source: str, *, delay: float | None = None) -> list[Job]:
    """
    Microsoft jobs page renders initial state in a <script> tag as JSON.
    We extract that JSON blob and parse job listings.
    """
    log.info("Microsoft Careers ▶ %s", source)
    resp = safe_get(url, delay=delay or config.FAANG_DELAY_SECONDS)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    jobs: list[Job] = []

    # Try to find jobs JSON in script tags
    for script in soup.find_all("script"):
        text = script.string or ""
        if "jobPostings" not in text and "totalJobs" not in text:
            continue
        # Locate the JSON blob
        m = re.search(r'"operationResult"\s*:\s*(\{.*?"hits".*?\})\s*[,}]', text, re.DOTALL)
        if not m:
            continue
        try:
            blob = json.loads(m.group(1))
            hits = blob.get("result", {}).get("hits", [])
            for hit in hits:
                title    = hit.get("title", "")
                location = hit.get("primaryCity", hit.get("country", "India"))
                job_id   = hit.get("jobId", "")
                apply_url = (
                    f"https://jobs.microsoft.com/en-us/job/{job_id}"
                    if job_id else url
                )
                if title:
                    jobs.append(Job(
                        title=title, company="Microsoft",
                        location=location, url=apply_url, source=source,
                    ))
        except (json.JSONDecodeError, AttributeError):
            pass

    if jobs:
        log.info("  → %d jobs from Microsoft Careers", len(jobs))
        return jobs

    # Fallback: parse HTML job cards
    for card in soup.find_all(class_=re.compile(r"job[-_]?card|ms[-_]job", re.I)):
        title_el = card.find(["h3", "h4", "a"])
        title    = title_el.get_text(strip=True) if title_el else ""
        link_el  = card.find("a", href=True)
        link     = link_el["href"] if link_el else url
        if not title:
            continue
        if not link.startswith("http"):
            link = urljoin("https://jobs.microsoft.com", link)
        jobs.append(Job(title=title, company="Microsoft", url=link, source=source))

    log.info("  → %d jobs from Microsoft Careers (HTML fallback)", len(jobs))
    return jobs


# ══════════════════════════════════════════════════════════════════════════════
# ── SCRAPER 10: Amazon Jobs JSON API ─────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def scrape_amazon_jobs(url: str, source: str, *, delay: float | None = None) -> list[Job]:
    """
    Amazon exposes a public JSON search endpoint at /en/search.json
    """
    log.info("Amazon Jobs ▶ %s", source)
    data = safe_get(url, json_mode=True, delay=delay or config.FAANG_DELAY_SECONDS)
    if not data:
        return []

    jobs: list[Job] = []
    for hit in data.get("jobs", []):
        title    = hit.get("title", "")
        location = hit.get("city", hit.get("country_code", "India"))
        job_path = hit.get("job_path", "")
        apply_url = urljoin("https://www.amazon.jobs", job_path) if job_path else url

        team = hit.get("business_name", "Amazon")
        if title:
            jobs.append(Job(
                title=title, company=team,
                location=location, url=apply_url, source=source,
            ))

    log.info("  → %d jobs from Amazon Jobs", len(jobs))
    return jobs


# ══════════════════════════════════════════════════════════════════════════════
# ── SCRAPER 11: Meta Careers ──────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def scrape_meta_careers(url: str, source: str, *, delay: float | None = None) -> list[Job]:
    """
    Meta's careers page is largely JS-rendered.  We attempt to parse any
    server-rendered JSON-LD or embedded __initialData__ from the HTML.
    """
    log.info("Meta Careers ▶ %s", source)
    resp = safe_get(url, delay=delay or config.FAANG_DELAY_SECONDS)
    if resp is None:
        return []

    soup  = BeautifulSoup(resp.text, "html.parser")
    jobs: list[Job] = []

    # JSON-LD
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, AttributeError):
            continue
        entries = data if isinstance(data, list) else [data]
        for entry in entries:
            if entry.get("@type") != "JobPosting":
                continue
            title = entry.get("title", "")
            apply_url = entry.get("url", url)
            if title:
                jobs.append(Job(title=title, company="Meta", url=apply_url, source=source))

    if jobs:
        log.info("  → %d jobs from Meta Careers (JSON-LD)", len(jobs))
        return jobs

    # __initialData__ pattern
    for script in soup.find_all("script"):
        text = script.string or ""
        if "__initialData__" not in text and "job_listings" not in text:
            continue
        m = re.search(r'"job_listings"\s*:\s*(\[.*?\])', text, re.DOTALL)
        if m:
            try:
                listings = json.loads(m.group(1))
                for listing in listings:
                    title = listing.get("title", "")
                    job_id = listing.get("id", "")
                    apply_url = f"https://www.metacareers.com/jobs/{job_id}/" if job_id else url
                    if title:
                        jobs.append(Job(
                            title=title, company="Meta", url=apply_url, source=source,
                        ))
            except json.JSONDecodeError:
                pass

    log.info("  → %d jobs from Meta Careers", len(jobs))
    return jobs


# ══════════════════════════════════════════════════════════════════════════════
# ── SCRAPER 12: Apple Jobs ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def scrape_apple_jobs(url: str, source: str, *, delay: float | None = None) -> list[Job]:
    """Apple's job search page — parse HTML cards."""
    log.info("Apple Jobs ▶ %s", source)
    resp = safe_get(url, delay=delay or config.FAANG_DELAY_SECONDS)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    jobs: list[Job] = []

    for row in soup.find_all(class_=re.compile(r"table-row|search-result", re.I)):
        title_el = row.find(class_=re.compile(r"table-col-2|role-title", re.I))
        loc_el   = row.find(class_=re.compile(r"table-col-3|location", re.I))
        link_el  = row.find("a", href=True)

        title    = title_el.get_text(strip=True) if title_el else ""
        location = loc_el.get_text(strip=True)   if loc_el   else "India"
        link     = link_el["href"]               if link_el  else url

        if not title:
            continue
        if not link.startswith("http"):
            link = urljoin("https://jobs.apple.com", link)
        jobs.append(Job(
            title=title, company="Apple", location=location, url=link, source=source,
        ))

    log.info("  → %d jobs from Apple Jobs", len(jobs))
    return jobs


# ══════════════════════════════════════════════════════════════════════════════
# ── SCRAPER 13: OpenAI Careers ───────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def scrape_openai_careers(url: str, source: str, *, delay: float | None = None) -> list[Job]:
    """OpenAI careers page — extract JSON-LD or parse listing sections."""
    log.info("OpenAI Careers ▶ %s", source)
    resp = safe_get(url, delay=delay or config.FAANG_DELAY_SECONDS)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    jobs: list[Job] = []

    # JSON-LD
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, AttributeError):
            continue
        entries = data if isinstance(data, list) else [data]
        for entry in entries:
            if entry.get("@type") != "JobPosting":
                continue
            title     = entry.get("title", "")
            apply_url = entry.get("url", url)
            if title:
                jobs.append(Job(title=title, company="OpenAI", url=apply_url, source=source))

    if jobs:
        log.info("  → %d jobs from OpenAI (JSON-LD)", len(jobs))
        return jobs

    # Fallback: grab all section headings + links in the jobs section
    # OpenAI uses a Greenhouse-style listing at https://openai.com/careers/
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "greenhouse.io" in href or "/careers/" in href:
            title = a.get_text(strip=True)
            if len(title) > 5:
                full = href if href.startswith("http") else urljoin("https://openai.com", href)
                jobs.append(Job(title=title, company="OpenAI", url=full, source=source))

    # Deduplicate by URL
    seen: set[str] = set()
    unique: list[Job] = []
    for j in jobs:
        if j.url not in seen:
            seen.add(j.url)
            unique.append(j)

    log.info("  → %d jobs from OpenAI (HTML fallback)", len(unique))
    return unique


# ══════════════════════════════════════════════════════════════════════════════
# ── SCRAPER 14: Anthropic Careers ────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def scrape_anthropic_careers(url: str, source: str, *, delay: float | None = None) -> list[Job]:
    """Anthropic careers — similar Greenhouse-based listing."""
    log.info("Anthropic Careers ▶ %s", source)
    resp = safe_get(url, delay=delay or config.FAANG_DELAY_SECONDS)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    jobs: list[Job] = []

    # JSON-LD attempt
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, AttributeError):
            continue
        entries = data if isinstance(data, list) else [data]
        for entry in entries:
            if entry.get("@type") != "JobPosting":
                continue
            title     = entry.get("title", "")
            apply_url = entry.get("url", url)
            if title:
                jobs.append(Job(title=title, company="Anthropic", url=apply_url, source=source))

    if jobs:
        log.info("  → %d jobs from Anthropic (JSON-LD)", len(jobs))
        return jobs

    # HTML fallback
    seen_urls: set[str] = set()
    for a in soup.find_all("a", href=True):
        href  = a["href"]
        title = a.get_text(strip=True)
        if not title or len(title) < 5 or len(title) > 120:
            continue
        if "greenhouse.io" not in href and "/careers/" not in href:
            continue
        full = href if href.startswith("http") else urljoin("https://www.anthropic.com", href)
        if full not in seen_urls:
            seen_urls.add(full)
            jobs.append(Job(title=title, company="Anthropic", url=full, source=source))

    log.info("  → %d jobs from Anthropic (HTML fallback)", len(jobs))
    return jobs


# ══════════════════════════════════════════════════════════════════════════════
# ── SCRAPER 15: Workable API (Hugging Face, etc.) ────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def scrape_workable(url: str, source: str, *, delay: float | None = None) -> list[Job]:
    """
    Workable widget listing API returns JSON:
    https://apply.workable.com/api/v1/widget/listing/<company>/
    """
    log.info("Workable ▶ %s", source)
    data = safe_get(url, json_mode=True, delay=delay or config.REQUEST_DELAY_SECONDS)
    if not data:
        return []

    jobs: list[Job] = []
    company_name = source.split("(")[0].strip()   # e.g. "Hugging Face Jobs"

    for job_item in data.get("jobs", []):
        title    = job_item.get("title", "")
        location = job_item.get("location", {})
        if isinstance(location, dict):
            loc_str = location.get("city", location.get("country", "Remote"))
        else:
            loc_str = str(location)
        slug     = job_item.get("shortcode", "")
        # Reconstruct apply URL from the slug
        base     = re.sub(r"/api/v1/widget/listing/", "/", url).rstrip("/")
        apply_url = f"{base}/{slug}" if slug else url

        if title:
            jobs.append(Job(
                title=title, company=company_name,
                location=loc_str, url=apply_url, source=source,
            ))

    log.info("  → %d jobs from Workable (%s)", len(jobs), source)
    return jobs


# ══════════════════════════════════════════════════════════════════════════════
# ── SCRAPER 16: NVIDIA Workday ────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def scrape_nvidia_careers(url: str, source: str, *, delay: float | None = None) -> list[Job]:
    """
    Workday-based career sites expose a JSON API endpoint.
    We POST to the Workday jobs JSON endpoint (documented in their public API).
    """
    log.info("NVIDIA Careers ▶ %s", source)
    api_url = url  # e.g. .../NVIDIAExternalCareerSite/jobs

    try:
        time.sleep(delay or config.FAANG_DELAY_SECONDS)
        resp = SESSION.post(
            api_url,
            json={
                "appliedFacets": {},
                "limit": 20,
                "offset": 0,
                "searchText": "machine learning AI",
            },
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            timeout=config.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning("NVIDIA Workday API error: %s", exc)
        return []

    jobs: list[Job] = []
    for hit in data.get("jobPostings", []):
        title    = hit.get("title", "")
        location = hit.get("locationsText", "Remote")
        ext_id   = hit.get("externalPath", "")
        apply_url = (
            f"https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite{ext_id}"
            if ext_id else url
        )
        if title:
            jobs.append(Job(
                title=title, company="NVIDIA",
                location=location, url=apply_url, source=source,
            ))

    log.info("  → %d jobs from NVIDIA Careers", len(jobs))
    return jobs


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE DISPATCHER
# ══════════════════════════════════════════════════════════════════════════════

SCRAPER_MAP: dict[str, Any] = {
    "rss":                scrape_rss,
    "hn_hiring":          scrape_hn_hiring,
    "wellfound":          scrape_wellfound,
    "github_hiring":      scrape_github_hiring,
    "naukri":             scrape_naukri,
    "cutshort":           scrape_cutshort,
    "internshala":        scrape_internshala,
    "google_careers":     scrape_google_careers,
    "microsoft_careers":  scrape_microsoft_careers,
    "amazon_jobs":        scrape_amazon_jobs,
    "meta_careers":       scrape_meta_careers,
    "apple_jobs":         scrape_apple_jobs,
    "openai_careers":     scrape_openai_careers,
    "anthropic_careers":  scrape_anthropic_careers,
    "workable":           scrape_workable,
    "nvidia_careers":     scrape_nvidia_careers,
}


def scrape_all_sources() -> list[Job]:
    """Run every enabled source and aggregate results — never crash the whole bot."""
    all_jobs: list[Job] = []
    for source in config.SOURCES:
        if not source.get("enabled", True):
            log.info("SKIP (disabled) ▶ %s", source["name"])
            continue

        scraper = SCRAPER_MAP.get(source["type"])
        if scraper is None:
            log.warning("Unknown source type '%s' for '%s' — skipping", source["type"], source["name"])
            continue

        per_source_delay = source.get("delay")   # optional per-source override
        try:
            jobs = scraper(source["url"], source["name"], delay=per_source_delay)
            all_jobs.extend(jobs)
        except Exception:
            # Graceful degradation: log the traceback but keep running
            log.exception(
                "Unhandled error scraping '%s' — skipping this source", source["name"]
            )

    log.info("Total raw jobs collected: %d", len(all_jobs))
    return all_jobs


# ══════════════════════════════════════════════════════════════════════════════
# DEDUPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def load_seen_jobs(path: str) -> dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with p.open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        log.error("Could not load %s: %s — starting fresh", path, exc)
        return {}


def save_seen_jobs(seen: dict[str, str], path: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(seen, f, indent=2, ensure_ascii=False)
        log.info("Saved %d seen-job records to %s", len(seen), path)
    except OSError as exc:
        log.error("Could not save %s: %s", path, exc)


def filter_new_jobs(jobs: list[Job], seen: dict[str, str]) -> list[Job]:
    new: list[Job] = []
    for job in jobs:
        if job.uid in seen:
            continue
        if not job.matches_keywords(config.KEYWORDS):
            continue
        new.append(job)
    log.info("New keyword-matching jobs: %d", len(new))
    return new


def prune_seen(seen: dict[str, str], max_days: int = 30) -> dict[str, str]:
    """Drop records older than max_days to keep the file small."""
    now = datetime.now(timezone.utc)
    pruned: dict[str, str] = {}
    for uid, ts in seen.items():
        try:
            recorded = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if (now - recorded).days < max_days:
                pruned[uid] = ts
        except (ValueError, AttributeError):
            pruned[uid] = ts   # keep unparseable records
    return pruned


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM SENDER
# ══════════════════════════════════════════════════════════════════════════════

_TG_API = "https://api.telegram.org/bot{token}/{method}"


def send_telegram_message(text: str, *, token: str, chat_id: str) -> bool:
    url     = _TG_API.format(token=token, method="sendMessage")
    payload = {
        "chat_id":    chat_id,
        "text":       text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": False,
    }
    try:
        resp = SESSION.post(url, json=payload, timeout=config.REQUEST_TIMEOUT)
        if not resp.ok:
            log.error(
                "Telegram error %d: %s",
                resp.status_code,
                resp.json().get("description", resp.text[:200]),
            )
            return False
        return True
    except requests.exceptions.RequestException as exc:
        log.error("Telegram send failed: %s", exc)
        return False


def send_jobs(jobs: list[Job], *, token: str, chat_id: str) -> int:
    if not jobs:
        log.info("No new jobs to send.")
        return 0

    header = (
        f"🔔 *Job Alert — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*\n"
        f"Found *{len(jobs)}* new AI/ML jobs for you! 🚀"
    )
    send_telegram_message(header, token=token, chat_id=chat_id)
    time.sleep(1)

    cap  = config.MAX_JOBS_PER_RUN
    sent = 0
    for i, job in enumerate(jobs[:cap], start=1):
        msg = f"*{i}/{min(len(jobs), cap)}*\n\n" + job.to_telegram_message()
        if send_telegram_message(msg, token=token, chat_id=chat_id):
            sent += 1
        time.sleep(1.2)   # ~1 msg/sec — safe for all Telegram chat types

    if len(jobs) > cap:
        send_telegram_message(
            f"ℹ️ _...and {len(jobs) - cap} more jobs found but not sent to avoid flooding._\n"
            f"_Increase `MAX_JOBS_PER_RUN` in config.py to see all._",
            token=token,
            chat_id=chat_id,
        )

    log.info("Sent %d/%d jobs to Telegram.", sent, min(len(jobs), cap))
    return sent


# ══════════════════════════════════════════════════════════════════════════════
# TEST / DRY-RUN OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def print_jobs_to_console(jobs: list[Job]) -> None:
    """Pretty-print matching jobs to stdout (used in --test mode)."""
    if not jobs:
        print("\n✅ No new jobs found matching your keywords.\n")
        return

    print(f"\n{'═'*60}")
    print(f"  🔍 DRY RUN — {len(jobs)} new AI/ML jobs found")
    print(f"{'═'*60}\n")
    for i, job in enumerate(jobs, start=1):
        print(f"[{i:>3}] {'─'*50}")
        print(job.to_console_str())
        print()
    print(f"{'═'*60}")
    print(f"  Total: {len(jobs)} jobs  |  Max to send: {config.MAX_JOBS_PER_RUN}")
    print(f"{'═'*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_config(test_mode: bool = False) -> None:
    """Abort early if required settings are missing (skipped in --test mode)."""
    if test_mode:
        return
    errors: list[str] = []
    if not config.TELEGRAM_BOT_TOKEN or "YOUR_BOT_TOKEN" in config.TELEGRAM_BOT_TOKEN:
        errors.append(
            "TELEGRAM_BOT_TOKEN is not set. "
            "Edit config.py or set the TELEGRAM_BOT_TOKEN environment variable."
        )
    if not config.TELEGRAM_CHAT_ID or "YOUR_CHAT_ID" in config.TELEGRAM_CHAT_ID:
        errors.append(
            "TELEGRAM_CHAT_ID is not set. "
            "Edit config.py or set the TELEGRAM_CHAT_ID environment variable."
        )
    if errors:
        for err in errors:
            log.error("CONFIG ERROR: %s", err)
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args      = parse_args()
    test_mode = args.test or os.environ.get("DRY_RUN", "false").lower() == "true"

    log.info("═" * 60)
    log.info(
        "Job Alert Bot starting %s— %s",
        "[DRY RUN] " if test_mode else "",
        datetime.now(timezone.utc).isoformat(),
    )
    log.info("═" * 60)

    validate_config(test_mode=test_mode)

    # 1. Load seen-jobs store
    seen = load_seen_jobs(config.SEEN_JOBS_FILE)
    log.info("Loaded %d previously seen jobs.", len(seen))

    # 2. Scrape all sources
    raw_jobs = scrape_all_sources()

    # 3. Filter: new + keyword-matching
    new_jobs = filter_new_jobs(raw_jobs, seen)

    # 4. Send or print
    if test_mode:
        print_jobs_to_console(new_jobs)
        log.info("DRY RUN complete — no Telegram messages sent, seen_jobs.json NOT updated.")
        return

    send_jobs(
        new_jobs,
        token=config.TELEGRAM_BOT_TOKEN,
        chat_id=config.TELEGRAM_CHAT_ID,
    )

    # 5. Update seen store with ALL new matching jobs
    now_iso = datetime.now(timezone.utc).isoformat()
    for job in new_jobs:
        seen[job.uid] = now_iso

    # 6. Prune old records
    seen = prune_seen(seen)

    # 7. Persist
    save_seen_jobs(seen, config.SEEN_JOBS_FILE)

    log.info(
        "Run complete. New jobs: %d | Total seen: %d",
        len(new_jobs),
        len(seen),
    )
    log.info("═" * 60)


if __name__ == "__main__":
    main()
