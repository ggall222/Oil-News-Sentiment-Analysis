import re
import json
import time
import hashlib
from pathlib import Path
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime
from email.utils import parsedate_to_datetime

BASE_URL = "https://oilprice.com"

START_PAGES = [
    "https://oilprice.com/Latest-Energy-News/World-News/",
    "https://oilprice.com/Energy/Crude-Oil/",
    "https://oilprice.com/Energy/Oil-Prices/",
    "https://oilprice.com/Energy/Energy-General/",
    "https://oilprice.com/Geopolitics/Middle-East/",
    "https://oilprice.com/Geopolitics/",
    "https://oilprice.com/Energy/Oil-Prices/",
]

# Number of paginated pages to scrape per category (1 = current page only)
PAGES_PER_CATEGORY = 15

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0; +local academic research)"
}

REQUEST_DELAY_SEC = 1.5
TIMEOUT = 20


# -----------------------------
# Keyword system
# -----------------------------
OIL_TERMS = [
    "oil", "crude", "crude oil", "wti", "brent", "petroleum",
    "oil and gas", "fuel price", "energy products"
]

DRIVER_TERMS = [
    # supply / production
    "opec", "opec+", "production", "output cut", "output cuts",
    "drilling", "rig count", "refinery", "refinery outage",
    "shutdown", "pipeline", "supply disruption", "supply disruptions",
    "inventory", "inventories", "stockpile", "stockpiles", "storage",

    # logistics / trade
    "exports", "imports", "shipping", "us shipping", "tanker",
    "tankers", "trade flows",

    # policy / environment / macro-energy
    "energy policies", "energy policy", "carbon market",
    "carbon emissions", "emissions", "climate change",
    "energy bill", "national defense", "energy",

    # demand / market / consumers
    "pain at the pumps", "commodity markets",
    "gasoline demand", "jet fuel demand", "fuel prices",

    # geopolitics
    "war", "conflict", "sanctions", "iran", "russia", "ukraine",
    "middle east", "strait of hormuz"
]

TOPIC_MAP = {
    "supply": [
        "opec", "opec+", "production", "output cut", "drilling",
        "rig count", "supply disruption", "supply disruptions"
    ],
    "inventory": ["inventory", "inventories", "stockpile", "stockpiles", "storage"],
    "refining": ["refinery", "refinery outage", "shutdown"],
    "flows": ["exports", "imports", "shipping", "us shipping", "tanker", "tankers", "pipeline"],
    "policy": ["energy policies", "energy policy", "energy bill", "national defense"],
    "climate": ["carbon market", "carbon emissions", "emissions", "climate change"],
    "markets": ["commodity markets", "fuel price", "fuel prices", "pain at the pumps"],
    "geopolitics": ["war", "conflict", "sanctions", "iran", "russia", "ukraine", "middle east", "strait of hormuz"],
}

BLOCKLIST = [
    "solar", "wind power", "hydroelectric", "tidal energy", "biofuels"
]


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def headline_score(headline: str) -> int:
    text = normalize_text(headline)
    score = 0

    # Require oil context strongly
    if any(term in text for term in OIL_TERMS):
        score += 3

    # Driver terms add incremental relevance
    for term in DRIVER_TERMS:
        if term in text:
            score += 1

    # Penalize likely off-target alternative-energy-only stories
    if any(term in text for term in BLOCKLIST) and not any(term in text for term in ["oil", "crude", "petroleum", "oil and gas"]):
        score -= 2

    return score


def is_relevant_headline(headline: str, min_score: int = 4) -> bool:
    text = normalize_text(headline)
    has_oil = any(term in text for term in OIL_TERMS)
    has_driver = any(term in text for term in DRIVER_TERMS)
    return has_oil and has_driver and headline_score(headline) >= min_score


def assign_topics(text: str) -> list[str]:
    txt = normalize_text(text)
    tags = []
    for topic, terms in TOPIC_MAP.items():
        if any(term in txt for term in terms):
            tags.append(topic)
    return sorted(set(tags))


def matched_keywords(text: str) -> list[str]:
    txt = normalize_text(text)
    matched = [kw for kw in (OIL_TERMS + DRIVER_TERMS) if kw in txt]
    return sorted(set(matched))


# -----------------------------
# HTTP helpers
# -----------------------------
def get_response(url: str) -> requests.Response:
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp


def get_soup(url: str) -> BeautifulSoup:
    resp = get_response(url)
    return BeautifulSoup(resp.text, "html.parser")


# -----------------------------
# Discovery
# -----------------------------
def looks_like_article_url(url: str) -> bool:
    """
    OilPrice article URLs typically look like:
    /Energy/Crude-Oil/Title-Here.html
    /Latest-Energy-News/World-News/Title-Here.html
    """
    if not url.startswith("http"):
        return False
    if not url.endswith(".html"):
        return False
    # avoid obvious non-article endpoints
    bad_bits = ["/search", "/goto/", "/feed/", "/newsfeeds/", "/auth/", "/account/"]
    return not any(bit in url for bit in bad_bits)


def extract_article_cards_from_category(category_url: str) -> list[dict]:
    soup = get_soup(category_url)
    results = []
    seen = set()

    # Broad strategy:
    # - find anchors that look like article URLs
    # - use nearby text for title/excerpt when possible
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full_url = urljoin(BASE_URL, href)

        if not looks_like_article_url(full_url):
            continue

        title = a.get_text(" ", strip=True)
        if len(title) < 20:
            continue

        if full_url in seen:
            continue
        seen.add(full_url)

        # Try to identify a nearby excerpt/card text
        excerpt = None
        parent = a.parent
        if parent:
            parent_text = parent.get_text(" ", strip=True)
            if parent_text and parent_text != title and len(parent_text) > len(title):
                excerpt = parent_text

        results.append({
            "headline": title,
            "url": full_url,
            "source_page": category_url,
            "excerpt_hint": excerpt
        })

    return results


# -----------------------------
# Article parsing
# -----------------------------
def parse_iso_or_rfc_date(value: str):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        pass
    try:
        return parsedate_to_datetime(value)
    except Exception:
        return None


def extract_article(article_url: str) -> dict:
    resp = get_response(article_url)
    soup = BeautifulSoup(resp.text, "html.parser")

    # Title
    title = None
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(" ", strip=True)

    # Metadata
    author = None
    published_at = None
    section = None

    # Meta tags first
    meta_author = soup.find("meta", attrs={"name": "author"}) or soup.find("meta", attrs={"property": "article:author"})
    if meta_author and meta_author.get("content"):
        author = meta_author["content"].strip()

    meta_pub = (
        soup.find("meta", attrs={"property": "article:published_time"})
        or soup.find("meta", attrs={"name": "pubdate"})
        or soup.find("meta", attrs={"name": "date"})
    )
    if meta_pub and meta_pub.get("content"):
        dt = parse_iso_or_rfc_date(meta_pub["content"].strip())
        published_at = dt.isoformat() if dt else meta_pub["content"].strip()

    # JSON-LD fallback (OilPrice uses this instead of meta tags)
    if not published_at:
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
                raw = data.get("datePublished") or data.get("dateCreated")
                if raw:
                    dt = parse_iso_or_rfc_date(raw)
                    published_at = dt.isoformat() if dt else raw
                    break
            except Exception:
                continue

    # Breadcrumb / category clues
    breadcrumbs = soup.select("nav a, .breadcrumb a, .breadcrumbs a")
    crumb_text = [x.get_text(" ", strip=True) for x in breadcrumbs if x.get_text(" ", strip=True)]
    if crumb_text:
        section = " > ".join(crumb_text[-3:])

    # Body extraction
    body_parts = []

    # Prefer article container, then common content wrappers, then fallback to paragraphs
    container = (
        soup.find("article")
        or soup.find("div", class_=re.compile(r"(article|content|post|entry|story)", re.I))
        or soup.find("main")
    )

    paragraph_candidates = container.find_all("p") if container else soup.find_all("p")

    for p in paragraph_candidates:
        txt = p.get_text(" ", strip=True)
        if not txt:
            continue
        # discard tiny fragments
        if len(txt) < 40:
            continue
        # discard obvious UI junk
        if any(phrase in txt.lower() for phrase in [
            "advertisement", "click here", "related:", "read more"
        ]):
            continue
        body_parts.append(txt)

    body_text = "\n".join(body_parts).strip()
    body_text = re.sub(r"\n{2,}", "\n\n", body_text)

    # Fallback author/time from visible text if meta missing
    if not author or not published_at:
        visible_text = soup.get_text("\n", strip=True)
        m = re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*\|\s*([A-Za-z]{3}\s+\d{1,2},\s+\d{4}.*)", visible_text)
        if m:
            if not author:
                author = m.group(1).strip()
            if not published_at:
                published_at = m.group(2).strip()

    return {
        "title": title,
        "author": author,
        "published_at": published_at,
        "section": section,
        "body_text": body_text
    }


# -----------------------------
# Main pipeline
# -----------------------------
def paginate_url(base_url: str, page_num: int) -> str:
    """Convert a category base URL to its paginated form.
    OilPrice uses the pattern: /Category/Page:2.html
    Page 1 is the bare URL (no suffix).
    """
    if page_num <= 1:
        return base_url.rstrip("/") + "/"
    base = base_url.rstrip("/")
    return f"{base}/Page-{page_num}.html"


def retrieve_relevant_oilprice_articles(start_pages=None, min_score: int = 3, delay_sec: float = REQUEST_DELAY_SEC, pages_per_category: int = PAGES_PER_CATEGORY) -> pd.DataFrame:
    if start_pages is None:
        start_pages = START_PAGES

    discovered = []
    seen_urls = set()

    # Step 1: discover candidate links — scrape multiple pages per category
    unique_start_pages = list(dict.fromkeys(start_pages))  # preserve order, remove duplicates
    for base_page in unique_start_pages:
        for page_num in range(1, pages_per_category + 1):
            page_url = paginate_url(base_page, page_num)
            try:
                cards = extract_article_cards_from_category(page_url)
                new_cards = [c for c in cards if c["url"] not in seen_urls]
                for c in new_cards:
                    seen_urls.add(c["url"])
                    discovered.append(c)
                print(f"  [{base_page.split('/')[-2]} p{page_num}] {len(new_cards)} new articles")
                time.sleep(delay_sec)
                if not new_cards:
                    break  # no new articles on this page, stop paginating
            except Exception as e:
                print(f"[WARN] Discovery failed for {page_url}: {e}")
                break

    # Step 2: headline filter
    filtered = []
    for item in discovered:
        score = headline_score(item["headline"])
        if is_relevant_headline(item["headline"], min_score=min_score):
            filtered.append({
                **item,
                "headline_score": score,
                "headline_topics": ",".join(assign_topics(item["headline"])),
                "headline_keyword_matches": ",".join(matched_keywords(item["headline"]))
            })

    # Step 3: fetch relevant articles only
    rows = []
    for item in filtered:
        try:
            parsed = extract_article(item["url"])
            combined_text = f"{item['headline']}\n{parsed.get('body_text', '')}"

            rows.append({
                "source": "OilPrice",
                "headline": item["headline"],
                "url": item["url"],
                "source_page": item["source_page"],
                "headline_score": item["headline_score"],
                "headline_topics": item["headline_topics"],
                "headline_keyword_matches": item["headline_keyword_matches"],
                "title": parsed.get("title"),
                "author": parsed.get("author"),
                "published_at": parsed.get("published_at"),
                "section": parsed.get("section"),
                "body_text": parsed.get("body_text"),
                "fulltext_topics": ",".join(assign_topics(combined_text)),
                "fulltext_keyword_matches": ",".join(matched_keywords(combined_text)),
                "retrieved_at_utc": datetime.utcnow().isoformat() + "Z"
            })
            time.sleep(delay_sec)
        except Exception as e:
            rows.append({
                "source": "OilPrice",
                "headline": item["headline"],
                "url": item["url"],
                "source_page": item["source_page"],
                "headline_score": item["headline_score"],
                "headline_topics": item["headline_topics"],
                "headline_keyword_matches": item["headline_keyword_matches"],
                "title": None,
                "author": None,
                "published_at": None,
                "section": None,
                "body_text": None,
                "fulltext_topics": None,
                "fulltext_keyword_matches": None,
                "retrieved_at_utc": datetime.utcnow().isoformat() + "Z",
                "error": str(e)
            })

    df = pd.DataFrame(rows)

    # Final dedupe by URL
    if not df.empty and "url" in df.columns:
        df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)

    return df


# -----------------------------
# Pipeline normalisation
# -----------------------------

def normalize_to_pipeline_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert OilPrice scraped output to the Benzinga-compatible schema used by
    build_features.py:  id, title, teaser, body, author, url, date, query_strategy, query_tier

    - id          : stable integer hash of the article URL
    - title       : article title (falls back to headline)
    - teaser      : headline used as a short summary proxy
    - body        : full body_text
    - date        : date string (YYYY-MM-DD) parsed from published_at
    - query_strategy / query_tier : fixed tags identifying the source
    """
    out = pd.DataFrame()

    out["id"] = df["url"].apply(
        lambda u: int(hashlib.md5(u.encode()).hexdigest()[:15], 16)
    )
    out["title"]  = df["title"].fillna(df["headline"])
    out["teaser"] = df["headline"]
    out["body"]   = df["body_text"].fillna("")
    out["author"] = df["author"].fillna("")
    out["url"]    = df["url"]

    # Parse date — try ISO / RFC formats, fall back to NaT
    def _parse_date(val):
        if not val or not isinstance(val, str):
            return None
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d", "%b %d, %Y"):
            try:
                return datetime.strptime(val[:19], fmt[:len(val[:19])]).strftime("%Y-%m-%d")
            except ValueError:
                continue
        try:
            return parsedate_to_datetime(val).strftime("%Y-%m-%d")
        except Exception:
            return None

    out["date"]           = df["published_at"].apply(_parse_date)
    out["query_strategy"] = "oilprice"
    out["query_tier"]     = "broad"

    # Drop rows where date couldn't be parsed
    out = out.dropna(subset=["date"]).reset_index(drop=True)
    return out


def save_to_parquet(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Save normalised OilPrice articles to month-partitioned parquets:
        data/raw/oilprice/YYYY-MM.parquet
    Merges with any existing file for the same month to avoid duplicates.
    """
    if df.empty:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    df["year_month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)

    for period, group in df.groupby("year_month"):
        path = output_dir / f"{period}.parquet"
        group = group.drop(columns="year_month")

        if path.exists():
            existing = pd.read_parquet(path)
            group = pd.concat([existing, group], ignore_index=True).drop_duplicates(subset="id")

        group.to_parquet(path, index=False)
        print(f"  Saved {len(group)} articles → {path}")


# -----------------------------
# Entry point
# -----------------------------

if __name__ == "__main__":
    raw_df  = retrieve_relevant_oilprice_articles()
    norm_df = normalize_to_pipeline_schema(raw_df)

    out_dir = Path(__file__).parent.parent / "data" / "raw" / "oilprice"
    save_to_parquet(norm_df, out_dir)

    show_cols = ["date", "headline_score", "headline", "headline_topics", "url"]
    print(raw_df[show_cols].head(25).to_string(index=False))
    print(f"\nScraped  : {len(raw_df)} articles")
    print(f"Saved    : {len(norm_df)} articles (with parseable dates)")
    print(f"Location : {out_dir}/YYYY-MM.parquet")