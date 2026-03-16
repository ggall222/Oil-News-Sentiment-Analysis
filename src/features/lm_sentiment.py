# src/features/lm_sentiment.py
#
# LM-S: Loughran-McDonald financial lexicon extended with oil-specific terms.
#
# Pipeline:
#   1. Base lexicon: core LM positive/negative words + oil-specific extensions
#   2. Corpus expansion: compute word frequency ratios across rise/fall-labeled
#      headlines to discover domain-specific polarity words
#   3. Scoring: normalized polarity sum over article tokens → score in [-1, 1]

import re
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd

# ── Stop words ────────────────────────────────────────────────────────────────

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "as", "by", "from", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "this",
    "that", "these", "those", "it", "its", "he", "she", "they", "we",
    "you", "i", "me", "him", "her", "us", "them", "my", "your", "his",
    "their", "our", "not", "no", "nor", "so", "yet", "both", "either",
    "if", "then", "than", "when", "where", "how", "what", "which", "who",
    "all", "each", "every", "any", "some", "such", "said", "also", "about",
    "after", "before", "into", "up", "out", "more", "most", "just", "over",
    "new", "per", "its", "s", "t", "re", "ve", "ll", "d",
}

# ── Base LM lexicon (Loughran-McDonald 2011 Financial Dictionary) ─────────────
# Positive: words associated with favorable financial conditions
LM_BASE_POSITIVE = [
    "able", "abundance", "accurate", "achieve", "adequate", "advance",
    "advantageous", "ambitious", "assurance", "attractive", "best", "better",
    "breakthrough", "bright", "broad", "capability", "certainty", "clarity",
    "commitment", "competitive", "confident", "consistent", "credible",
    "decisive", "dedicated", "deliver", "dependable", "desirable",
    "distinguished", "dominant", "durable", "effective", "efficient",
    "enable", "enhance", "exceptional", "expansive", "experienced",
    "favorable", "feasible", "flourish", "foremost", "gain", "growing",
    "genuine", "highest", "improved", "innovative", "integrity", "leading",
    "merit", "notable", "optimal", "outstanding", "positive", "powerful",
    "profitability", "progress", "proven", "quality", "reliable",
    "resilience", "resolve", "robust", "secure", "significant", "skilled",
    "stable", "steadfast", "strength", "strong", "success", "sufficient",
    "superior", "sustainable", "trusted", "valuable", "viable",
]

# Negative: words associated with adverse financial conditions
LM_BASE_NEGATIVE = [
    "abandon", "abnormal", "accused", "adversely", "alarming", "allege",
    "allegations", "bankrupt", "bankruptcy", "barrier", "breach", "burden",
    "cease", "challenge", "challenged", "closing", "collapse", "complaint",
    "concern", "constrained", "contamination", "corruption", "crisis",
    "cutback", "danger", "declining", "default", "deficiency", "deficit",
    "delay", "difficult", "diminished", "disappointment", "disputed",
    "disruption", "doubt", "downfall", "downgrade", "dysfunction", "erode",
    "excessive", "failed", "failure", "fallout", "faulty", "flawed",
    "fraud", "fraudulent", "guilty", "halt", "harmful", "harm", "idle",
    "impair", "impairment", "improper", "inaccurate", "incident",
    "insufficient", "irregular", "inadequate", "lack", "layoff", "liability",
    "litigation", "loss", "losses", "miss", "negligence", "obstacle",
    "outage", "overestimate", "overvalued", "penalty", "poor", "problem",
    "reduced", "rejected", "restriction", "sanction", "scandal", "shortfall",
    "shortage", "slow", "struggling", "suffering", "suspend", "tension",
    "threat", "troubled", "unable", "uncertain", "unfavorable", "unstable",
    "unusual", "violation", "warning", "weaken", "worsen",
]

# ── Oil-market extensions (price-direction polarity) ─────────────────────────
# Positive → associated with price RISES (supply tightening / demand surge)
OIL_POSITIVE = [
    "tightening", "tight", "disrupted", "disruption", "outage", "conflict",
    "sanctions", "sanctioned", "hurricane", "attack", "attacked", "shortage",
    "shortages", "constrained", "constrain", "curtailed", "curtailment",
    "cuts", "cut", "rally", "rallied", "soar", "soared", "surge", "surged",
    "spike", "spiked", "rebound", "rebounded", "recover", "recovery",
    "war", "warfare", "unrest", "escalation", "escalated", "geopolitical",
    "threat", "threats", "threatened", "restricted", "restriction",
    "blockade", "blockaded", "refinery outage", "pipeline explosion",
    "tanker attack", "stranded", "supply risk", "supply shock",
]

# Negative → associated with price FALLS (oversupply / demand weakness)
OIL_NEGATIVE = [
    "oversupply", "oversupplied", "glut", "surplus", "ample", "excess",
    "excessive supply", "weakened", "slump", "slumped", "plunge", "plunged",
    "bearish", "slowdown", "recession", "recessionary", "abundant",
    "easing", "ease", "builds", "build", "increase production",
    "ramp up", "ramping", "inventory build", "stockpile build",
    "demand weakness", "weak demand", "slowing demand", "softening",
    "overshooting", "gluts",
]

# ── Price-direction labeling ──────────────────────────────────────────────────

OIL_CONTEXT = {
    "oil", "crude", "wti", "brent", "petroleum", "energy", "barrel", "barrels",
    "opec", "refinery", "gasoline", "diesel", "fuel", "lng", "natural gas",
}

PRICE_CONTEXT = {
    "price", "prices", "cost", "per barrel", "market", "trading", "futures",
    "spot", "commodity", "commodities",
}

RISE_WORDS = {
    "rise", "rises", "rising", "rose", "surge", "surges", "surging", "surged",
    "soar", "soars", "soaring", "soared", "jump", "jumps", "jumping", "jumped",
    "climb", "climbs", "climbing", "climbed", "rally", "rallies", "rallied",
    "gain", "gains", "gaining", "gained", "increase", "increases", "increased",
    "upturn", "rebound", "rebounds", "rebounding", "rebounded", "spike", "spikes",
    "spiked", "higher", "high", "record", "boost", "boosted", "strengthen",
    "strengthens", "strengthened", "tighten", "tightens", "tightened",
}

FALL_WORDS = {
    "fall", "falls", "falling", "fell", "drop", "drops", "dropping", "dropped",
    "decline", "declines", "declining", "declined", "plunge", "plunges",
    "plunging", "plunged", "slump", "slumps", "slumped", "tumble", "tumbles",
    "tumbled", "slide", "slides", "sliding", "slid", "decrease", "decreases",
    "decreased", "lower", "low", "collapse", "collapses", "collapsed",
    "weaken", "weakens", "weakened", "ease", "eases", "eased",
}


def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, remove stop words, min length 3."""
    tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return [t for t in tokens if t not in STOP_WORDS]


def label_price_direction(headline: str) -> Optional[str]:
    """
    Label a headline as 'rise', 'fall', or None (unlabeled/ambiguous).

    Requires oil context (explicit oil/energy term) plus a directional keyword.
    """
    text = headline.lower()
    tokens = set(text.split())

    has_oil = any(kw in text for kw in OIL_CONTEXT)
    has_rise = bool(tokens & RISE_WORDS)
    has_fall = bool(tokens & FALL_WORDS)

    if not has_oil:
        return None

    if has_rise and not has_fall:
        return "rise"
    if has_fall and not has_rise:
        return "fall"
    return None  # ambiguous (both or neither directional)


# ── LM-S corpus expansion ────────────────────────────────────────────────────

def build_lm_s_lexicon(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "price_direction",
    min_freq: int = 3,
    min_ratio: float = 1.5,
    smoothing: float = 0.5,
) -> Dict[str, float]:
    """
    Build the LM-S lexicon by expanding the base LM lexicon with corpus-derived
    polarity scores from rise/fall-labeled articles.

    Parameters
    ----------
    df         : DataFrame with a text column and a price_direction column
    text_col   : column containing article text to analyze
    label_col  : column with 'rise' / 'fall' / None labels
    min_freq   : minimum total occurrences for a word to be considered
    min_ratio  : minimum (rise_count/fall_count) or (fall_count/rise_count) ratio
                 to add a word from the corpus expansion
    smoothing  : Laplace smoothing for ratio computation

    Returns
    -------
    dict mapping word → polarity score in [-1, 1]
    """
    # Start with base LM + oil extensions
    lexicon: Dict[str, float] = {}
    for w in LM_BASE_POSITIVE:
        lexicon[w] = 1.0
    for w in LM_BASE_NEGATIVE:
        lexicon[w] = -1.0
    for w in OIL_POSITIVE:
        lexicon[w] = 1.0
    for w in OIL_NEGATIVE:
        lexicon[w] = -1.0

    # Corpus expansion using labeled articles
    rise_df = df[df[label_col] == "rise"]
    fall_df = df[df[label_col] == "fall"]

    if rise_df.empty or fall_df.empty:
        return lexicon

    rise_counts: Dict[str, int] = defaultdict(int)
    fall_counts: Dict[str, int] = defaultdict(int)

    for text in rise_df[text_col].dropna():
        for token in tokenize(str(text)):
            rise_counts[token] += 1

    for text in fall_df[text_col].dropna():
        for token in tokenize(str(text)):
            fall_counts[token] += 1

    all_words = set(rise_counts) | set(fall_counts)
    for word in all_words:
        if word in lexicon:
            continue  # base lexicon takes priority

        rc = rise_counts.get(word, 0)
        fc = fall_counts.get(word, 0)

        if rc + fc < min_freq:
            continue

        rise_ratio = (rc + smoothing) / (fc + smoothing)
        fall_ratio = (fc + smoothing) / (rc + smoothing)

        if rise_ratio >= min_ratio:
            # Polarity score proportional to log-ratio, capped at 1
            score = min(1.0, (rise_ratio - 1) / (rise_ratio + 1))
            lexicon[word] = round(score, 4)
        elif fall_ratio >= min_ratio:
            score = -min(1.0, (fall_ratio - 1) / (fall_ratio + 1))
            lexicon[word] = round(score, 4)

    return lexicon


def score_text(text: str, lexicon: Dict[str, float]) -> float:
    """
    Compute LM-S sentiment score for a piece of text.

    Score = sum of matched polarity values / total token count.
    Returns a float in [-1, 1]; 0.0 if no tokens match.
    """
    tokens = tokenize(str(text))
    if not tokens:
        return 0.0
    total_polarity = sum(lexicon.get(t, 0.0) for t in tokens)
    return max(-1.0, min(1.0, total_polarity / len(tokens)))


def score_articles(
    df: pd.DataFrame,
    lexicon: Dict[str, float],
    text_col: str = "text",
    score_col: str = "lm_sentiment",
) -> pd.DataFrame:
    """Add an LM-S sentiment score column to a DataFrame of articles."""
    df = df.copy()
    df[score_col] = df[text_col].apply(lambda x: score_text(x, lexicon))
    return df
