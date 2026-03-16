"""
scripts/oilprice_backfill.py
────────────────────────────
Harvests historical OilPrice.com articles for 2019-present and saves them
as month-partitioned parquets in data/raw/oilprice/ for the build pipeline.

Strategy
────────
1. Fetch the OilPrice sitemap index → enumerate all per-month sitemap files.
2. Parse each sitemap, extract article URLs + lastmod dates.
3. Filter to relevant URL paths and the target date window.
4. Skip URLs already saved (uses a local checkpoint file).
5. Fetch + score + normalise each article using existing oilnewsscraper logic.
6. Write month-partitioned parquets incrementally (saves every BATCH_SIZE articles).

Usage
─────
    python3 scripts/oilprice_backfill.py                  # 2019-01-01 → today
    python3 scripts/oilprice_backfill.py --from 2022-01-01 --to 2023-12-31
    python3 scripts/oilprice_backfill.py --resume          # skip already done
"""

from __future__ import annotations  # X | Y type hints on Python 3.9

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path
import requests
from bs4 import BeautifulSoup

# ── resolve project root ────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.oilnewsscraper import (
    extract_article,
    headline_score,
    is_relevant_headline,
    assign_topics,
    matched_keywords,
    normalize_to_pipeline_schema,
    save_to_parquet,
    HEADERS,
    TIMEOUT,
    REQUEST_DELAY_SEC,
)
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────────────

OUTPUT_DIR   = ROOT / "data" / "raw" / "oilprice"
CHECKPOINT   = OUTPUT_DIR / "backfill_checkpoint.json"

# Sitemap roots to try (in order)
SITEMAP_ROOTS = [
    "https://oilprice.com/sitemap_index.xml",
    "https://oilprice.com/sitemap.xml",
    "https://oilprice.com/news-sitemap.xml",
]

# URL path prefixes that indicate oil-relevant articles
RELEVANT_PATHS = [
    "/Energy/Crude-Oil/",
    "/Energy/Oil-Prices/",
    "/Energy/Energy-General/",
    "/Energy/Gas-Prices/",
    "/Latest-Energy-News/World-News/",
    "/Geopolitics/Middle-East/",
    "/Geopolitics/International/",
    "/Geopolitics/Asia/",          # China demand, Russia
    "/Geopolitics/South-America/", # Venezuela, Brazil
    "/Geopolitics/",
]

# Exclude these — off-topic for WTI modelling
EXCLUDE_PATHS = [
    "/Alternative-Energy/",
    "/Energy/Coal/",
    "/Renewables/",
    "/Technology/",
    "/Investing/",
    "/Finance/",
    "/The-Environment/",
]

MIN_HEADLINE_SCORE = 3    # lower than live scraper to catch more historical signal
MIN_BODY_LEN       = 0    # body text is JS-rendered on OilPrice; save headline-only articles
BATCH_SIZE         = 50   # save parquet every N articles


# ── Sitemap helpers ─────────────────────────────────────────────────────────────

def get_soup_xml(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "xml")


def find_sitemap_root() -> str | None:
    """Try each candidate sitemap root until one responds."""
    for url in SITEMAP_ROOTS:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            if resp.status_code == 200 and ("sitemap" in resp.text.lower() or "<url" in resp.text):
                print(f"  Sitemap found: {url}")
                return url
        except Exception:
            continue
    return None


def parse_sitemap_index(index_url: str) -> list[str]:
    """Return list of child sitemap URLs from a sitemap index."""
    soup = get_soup_xml(index_url)
    return [loc.text.strip() for loc in soup.find_all("loc") if loc.text.strip().endswith(".xml")]


def parse_sitemap_urls(sitemap_url: str, start_date: date, end_date: date) -> list[dict]:
    """
    Parse a leaf sitemap, returning dicts {url, lastmod} filtered to date window.
    """
    try:
        soup = get_soup_xml(sitemap_url)
    except Exception as e:
        print(f"  [WARN] {sitemap_url}: {e}")
        return []

    results = []
    for url_tag in soup.find_all("url"):
        loc = url_tag.find("loc")
        if not loc:
            continue
        raw_url = loc.text.strip()

        # Normalise malformed URLs (some sitemap entries are missing scheme/domain)
        if raw_url.startswith("http"):
            article_url = raw_url
        elif raw_url.startswith("/"):
            article_url = "https://oilprice.com" + raw_url
        elif raw_url.startswith("oilprice.com"):
            article_url = "https://" + raw_url
        else:
            continue  # can't reconstruct

        # Path filter
        path = article_url.replace("https://oilprice.com", "").replace("http://oilprice.com", "")
        if not any(path.startswith(p) for p in RELEVANT_PATHS):
            continue
        if any(path.startswith(p) for p in EXCLUDE_PATHS):
            continue
        if not article_url.endswith(".html"):
            continue

        # Date filter (from lastmod)
        lastmod_tag = url_tag.find("lastmod")
        article_date = None
        if lastmod_tag and lastmod_tag.text.strip():
            try:
                article_date = datetime.fromisoformat(lastmod_tag.text.strip()[:10]).date()
            except ValueError:
                pass

        if article_date:
            if article_date < start_date or article_date > end_date:
                continue

        results.append({"url": article_url, "lastmod": str(article_date) if article_date else None})

    return results


def collect_all_urls(start_date: date, end_date: date) -> list[dict]:
    """
    Walk sitemap index → child sitemaps → filter to date/path → return URL list.
    Falls back to direct leaf parse if the root IS the leaf sitemap.
    """
    root_url = find_sitemap_root()
    if not root_url:
        print("ERROR: Could not find any OilPrice sitemap. Site may have changed.")
        return []

    soup = get_soup_xml(root_url)
    child_sitemaps = [loc.text.strip() for loc in soup.find_all("loc") if loc.text.strip()]

    # Distinguish index (contains .xml children) vs leaf (contains <url> tags)
    is_index = any(u.endswith(".xml") for u in child_sitemaps)

    if is_index:
        # Filter to monthly article sitemaps only — skip static / company-news sitemaps.
        # Pattern: sitemap_articles_YYYY_M.xml
        import re as _re
        article_sms = []
        for sm_url in child_sitemaps:
            m = _re.search(r"sitemap_articles_(\d{4})_(\d{1,2})\.xml", sm_url)
            if m:
                y, mo = int(m.group(1)), int(m.group(2))
                sm_date = date(y, mo, 1)
                if start_date <= sm_date <= end_date.replace(day=1):
                    article_sms.append(sm_url)

        print(f"  Monthly article sitemaps in range: {len(article_sms)}")
        all_urls = []
        for i, sm_url in enumerate(sorted(article_sms)):
            urls = parse_sitemap_urls(sm_url, start_date, end_date)
            print(f"  [{i+1}/{len(article_sms)}] {sm_url.split('/')[-1]}: {len(urls)} matching URLs")
            all_urls.extend(urls)
            time.sleep(0.3)
        return all_urls
    else:
        # Root is itself a leaf sitemap
        print("  Parsing root as leaf sitemap...")
        return parse_sitemap_urls(root_url, start_date, end_date)


# ── Checkpoint helpers ──────────────────────────────────────────────────────────

def load_checkpoint() -> set[str]:
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            return set(json.load(f).get("done_urls", []))
    return set()


def save_checkpoint(done_urls: set[str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT, "w") as f:
        json.dump({"done_urls": sorted(done_urls), "updated_at": datetime.utcnow().isoformat()}, f)


# ── Main backfill loop ──────────────────────────────────────────────────────────

def backfill(start_date: date, end_date: date, resume: bool = True) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    done_urls = load_checkpoint() if resume else set()
    print(f"Checkpoint: {len(done_urls)} URLs already processed")

    # ── 1. Collect candidate URLs from sitemaps ────────────────────────────────
    print(f"\nCollecting sitemap URLs for {start_date} → {end_date} ...")
    candidates = collect_all_urls(start_date, end_date)
    print(f"  Total matching sitemap URLs : {len(candidates)}")

    # Deduplicate and skip already done
    seen = set()
    todo = []
    for c in candidates:
        if c["url"] not in seen and c["url"] not in done_urls:
            seen.add(c["url"])
            todo.append(c)

    print(f"  After dedup + checkpoint skip: {len(todo)} to fetch")
    if not todo:
        print("Nothing to do.")
        return

    # ── 2. Fetch, score, accumulate ────────────────────────────────────────────
    batch_raw  = []
    n_saved    = 0
    n_skipped  = 0
    n_no_body  = 0

    for i, item in enumerate(todo):
        url = item["url"]
        print(f"  [{i+1}/{len(todo)}] {url.split('oilprice.com')[-1][:60]}", end=" ", flush=True)

        try:
            parsed = extract_article(url)
            headline = parsed.get("title") or url.split("/")[-1].replace("-", " ").replace(".html", "")

            score = headline_score(headline)
            if score < MIN_HEADLINE_SCORE:
                print(f"→ skip (score={score})")
                done_urls.add(url)
                n_skipped += 1
                time.sleep(0.3)
                continue

            body = parsed.get("body_text") or ""
            if len(body) < MIN_BODY_LEN:
                print(f"→ no body ({len(body)} chars, JS-gated?)")
                done_urls.add(url)
                n_no_body += 1
                time.sleep(0.3)
                continue

            combined = f"{headline}\n{body}"
            batch_raw.append({
                "source":                    "OilPrice",
                "headline":                  headline,
                "url":                       url,
                "source_page":               url,
                "headline_score":            score,
                "headline_topics":           ",".join(assign_topics(headline)),
                "headline_keyword_matches":  ",".join(matched_keywords(headline)),
                "title":                     parsed.get("title"),
                "author":                    parsed.get("author"),
                "published_at":              parsed.get("published_at") or item.get("lastmod"),
                "section":                   parsed.get("section"),
                "body_text":                 body,
                "fulltext_topics":           ",".join(assign_topics(combined)),
                "fulltext_keyword_matches":  ",".join(matched_keywords(combined)),
                "retrieved_at_utc":          datetime.utcnow().isoformat() + "Z",
            })
            done_urls.add(url)
            print(f"→ ok (score={score})")

        except Exception as e:
            print(f"→ error: {e}")
            done_urls.add(url)  # don't retry broken articles

        time.sleep(REQUEST_DELAY_SEC)

        # ── Save batch ─────────────────────────────────────────────────────────
        if len(batch_raw) >= BATCH_SIZE:
            df_raw  = pd.DataFrame(batch_raw)
            df_norm = normalize_to_pipeline_schema(df_raw)
            save_to_parquet(df_norm, OUTPUT_DIR)
            n_saved += len(df_norm)
            batch_raw = []
            save_checkpoint(done_urls)
            print(f"\n  ── checkpoint: {n_saved} saved, {n_skipped} skipped (score), {n_no_body} no-body ──\n")

    # ── Save final batch ───────────────────────────────────────────────────────
    if batch_raw:
        df_raw  = pd.DataFrame(batch_raw)
        df_norm = normalize_to_pipeline_schema(df_raw)
        save_to_parquet(df_norm, OUTPUT_DIR)
        n_saved += len(df_norm)

    save_checkpoint(done_urls)
    print(f"\nBackfill complete: {n_saved} articles saved, {n_skipped} skipped (low score), {n_no_body} skipped (no body / JS-gated)")
    print(f"Parquets: {OUTPUT_DIR}/")


# ── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical OilPrice articles")
    parser.add_argument("--from", dest="start", default="2019-01-01",
                        help="Start date YYYY-MM-DD (default: 2019-01-01)")
    parser.add_argument("--to",   dest="end",   default=str(date.today()),
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore checkpoint and start fresh")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end   = datetime.strptime(args.end,   "%Y-%m-%d").date()

    print(f"OilPrice backfill: {start} → {end}")
    backfill(start, end, resume=not args.no_resume)
