# src/features/topic_classifier.py
#
# Rule-based classification of oil-market news into 8 topic categories.
# Each article can belong to multiple categories (multi-label).
#
# Categories (from oil price forecasting literature):
#   1. opec_production       — OPEC supply decisions
#   2. us_production         — U.S. shale / EIA production
#   3. oil_inventories       — stockpiles and storage levels
#   4. oil_demand            — consumption and demand signals
#   5. financial_factors     — macro / monetary / FX drivers
#   6. extreme_weather       — weather-related supply disruptions
#   7. pandemics             — disease outbreaks affecting demand
#   8. geopolitical_conflicts — wars, sanctions, regional tensions

from typing import List
import pandas as pd

TOPIC_KEYWORDS = {
    "opec_production": [
        "opec", "opec+", "opec plus", "production cut", "production cuts",
        "output cut", "output cuts", "quota", "compliance", "saudi arabia",
        "saudi output", "saudi production", "riyadh", "vienna meeting",
        "opec meeting", "opec decision", "barrel target",
        "voluntary cut", "production agreement", "production deal",
        "uae output", "kuwait output", "iraq output",
    ],
    "us_production": [
        "shale", "permian", "bakken", "eagle ford", "marcellus",
        "haynesville", "rig count", "baker hughes", "u.s. production",
        "us production", "american production", "domestic output",
        "u.s. output", "us output", "eia production", "eia output",
        "u.s. shale", "us shale", "fracking", "frac", "wellbore",
        "drilling activity", "completion activity",
    ],
    "oil_inventories": [
        "inventory", "inventories", "stockpile", "stockpiles",
        "crude stocks", "crude stockpile", "eia crude", "api crude",
        "cushing", "storage capacity", "storage levels",
        "oil reserves", "commercial inventory", "commercial stocks",
        "gasoline stocks", "distillate stocks", "draw", "build",
        "inventory build", "inventory draw", "storage draw",
    ],
    "oil_demand": [
        "demand", "consumption", "iea outlook", "iea forecast",
        "china demand", "asian demand", "global demand",
        "oil consumption", "fuel demand", "gasoline demand",
        "jet fuel demand", "diesel demand", "aviation demand",
        "economic growth", "gdp growth", "iea report",
        "demand forecast", "demand outlook", "demand growth",
        "demand slowdown", "demand weakness",
    ],
    "financial_factors": [
        "fed", "federal reserve", "interest rate", "interest rates",
        "rate hike", "rate cut", "dollar", "us dollar", "usd",
        "inflation", "cpi", "deflation", "monetary policy",
        "quantitative easing", "yield", "treasury", "bond",
        "recession", "gdp", "hedge fund", "speculative", "futures",
        "contango", "backwardation", "options", "derivatives",
    ],
    "extreme_weather": [
        "hurricane", "tropical storm", "cyclone", "typhoon",
        "gulf of mexico storm", "winter storm", "freeze", "freezing",
        "cold snap", "polar vortex", "flood", "flooding",
        "wildfire", "wildfire risk", "weather disruption",
        "severe weather", "extreme weather", "force majeure",
        "power outage", "pipeline freeze",
    ],
    "pandemics": [
        "covid", "coronavirus", "pandemic", "lockdown", "quarantine",
        "outbreak", "virus", "epidemic", "WHO", "disease",
        "travel ban", "mobility restriction", "economic shutdown",
        "demand collapse", "demand destruction",
    ],
    "geopolitical_conflicts": [
        "war", "warfare", "conflict", "military", "airstrike", "missile",
        "attack", "attacked", "drone strike", "sanctions", "sanctioned",
        "iran", "russia", "ukraine", "middle east", "strait of hormuz",
        "hormuz", "red sea", "houthi", "houthis", "israel", "gaza",
        "venezuela sanctions", "iran nuclear", "tanker attack",
        "pipeline attack", "refinery attack", "oil facility",
        "geopolitical", "geopolitical risk", "geopolitical tension",
    ],
}


def classify_topics(text: str) -> List[str]:
    """
    Return a list of matching topic categories for a piece of text.
    Case-insensitive. An article may match multiple categories.
    """
    text_lower = text.lower()
    matched = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            matched.append(topic)
    return matched


def assign_topics_column(
    df: pd.DataFrame,
    text_col: str = "text",
    topics_col: str = "lm_topics",
) -> pd.DataFrame:
    """
    Add a column listing matched topic categories for each article.
    Topics are stored as a comma-separated string.
    """
    df = df.copy()
    df[topics_col] = df[text_col].apply(
        lambda x: ",".join(classify_topics(str(x))) if pd.notna(x) else ""
    )
    return df


CATEGORIES = list(TOPIC_KEYWORDS.keys())
