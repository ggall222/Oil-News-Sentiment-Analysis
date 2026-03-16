# scripts/expand_keywords.py
#
# Trains Word2Vec on the full Benzinga corpus and expands all keyword lists
# using semantic similarity — automatically discovering new oil-market terms.
#
# Outputs:
#   data/expanded_keywords.json   — full ranked suggestions per group
#   (stdout)                      — human-readable summary of top suggestions
#
# Usage (from WTI_News_API/ directory):
#   python3 scripts/expand_keywords.py
#
# Optional flags:
#   --threshold 0.70     cosine similarity cutoff (default 0.65)
#   --top-n 25           candidates per seed (default 20)
#   --save-model         save the Word2Vec model to data/word2vec.model

import os
import re
import sys
import json
import math
import logging
import argparse
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.keyword_expansion import (
    build_corpus,
    train_word2vec,
    expand_all_groups,
    tokenize_sentence,
    seed_to_model_token,
)
from src.features.lm_sentiment import label_price_direction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

BASE_DIR    = Path(__file__).parent.parent
RAW_DIR     = BASE_DIR / "data/raw/benzinga"
OILPRICE_CSV = BASE_DIR / "data/raw/oilprice_relevant_articles.csv"
OUT_JSON    = BASE_DIR / "data/expanded_keywords.json"
QUERIES_PY  = BASE_DIR / "src/data/benzinga_queries.py"

# Maps topic-classifier keyword groups → strategy keys in benzinga_queries.py.
# All 8 classifier groups are mapped here so auto-update stays aligned with
# src/features/topic_classifier.py categories.
QUERY_GROUP_MAP = {
    "topic_opec_production":         "opec",
    "topic_us_production":           "broad",
    "topic_oil_inventories":         "broad",
    "topic_oil_demand":              "macro",
    "topic_financial_factors":       "macro",
    "topic_extreme_weather":         "disruption",
    "topic_pandemics":               "macro",
    "topic_geopolitical_conflicts":  "disruption",
}

# ── Seed keyword groups ────────────────────────────────────────────────────────
# Each group uses its existing keywords as Word2Vec anchors.
# New candidates will NOT overlap with any term already listed here.

KEYWORD_GROUPS = {

    # Benzinga query tier: disruption — currently thinly covered (646 articles)
    "disruption_query": [
        "pipeline outage", "pipeline explosion", "pipeline attack",
        "refinery outage", "refinery fire", "refinery shutdown",
        "supply disruption", "shipping disruption", "tanker attack",
        "force majeure", "Red Sea", "Strait of Hormuz", "Suez Canal",
        "Houthi", "Libya oil", "Nigeria oil", "sanctions",
    ],

    # Benzinga query tier: broad — expand to catch more supply/price news
    "broad_query": [
        "crude oil", "WTI", "Brent", "OPEC", "production cut", "quota",
        "crude inventory", "oil inventory", "EIA", "SPR", "strategic reserve",
        "pipeline", "refinery", "outage", "supply disruption",
        "Red Sea", "Strait of Hormuz", "oil demand", "demand outlook",
        "petroleum", "barrel",
    ],

    # Benzinga query tier: macro — demand-side coverage
    "macro_query": [
        "oil demand", "demand outlook", "demand forecast",
        "China oil", "China demand", "IEA outlook", "OPEC outlook",
        "recession oil", "economic slowdown oil",
        "jet fuel demand", "gasoline demand",
    ],

    # Topic classifier: 8 categories
    "topic_opec_production": [
        "opec", "production cut", "output cut", "quota", "compliance",
        "saudi arabia", "saudi output", "vienna meeting", "opec meeting",
        "opec decision", "voluntary cut",
    ],
    "topic_us_production": [
        "shale", "permian", "bakken", "eagle ford", "marcellus",
        "rig count", "baker hughes", "us production", "eia production",
        "fracking", "drilling activity",
    ],
    "topic_oil_inventories": [
        "inventory", "stockpile", "crude stocks", "eia crude", "api crude",
        "cushing", "storage capacity", "commercial inventory",
        "inventory build", "inventory draw",
    ],
    "topic_oil_demand": [
        "demand", "consumption", "iea outlook", "china demand",
        "global demand", "fuel demand", "gasoline demand",
        "jet fuel demand", "diesel demand",
    ],
    "topic_financial_factors": [
        "fed", "federal reserve", "interest rate", "rate hike", "rate cut",
        "dollar", "inflation", "monetary policy", "yield", "treasury",
        "recession", "hedge fund", "futures", "contango", "backwardation",
    ],
    "topic_extreme_weather": [
        "hurricane", "tropical storm", "cyclone", "typhoon",
        "winter storm", "freeze", "polar vortex", "flood",
        "wildfire", "severe weather", "force majeure",
    ],
    "topic_pandemics": [
        "covid", "coronavirus", "pandemic", "lockdown", "quarantine",
        "outbreak", "virus", "epidemic", "who", "disease",
        "travel ban", "mobility restriction", "economic shutdown",
        "demand collapse", "demand destruction",
    ],
    "topic_geopolitical_conflicts": [
        "war", "conflict", "military", "airstrike", "missile", "attack",
        "drone strike", "sanctions", "iran", "russia", "ukraine",
        "middle east", "strait of hormuz", "red sea", "houthi",
        "tanker attack", "pipeline attack", "geopolitical",
    ],

    # LM-S lexicon: rise and fall seed words
    "lm_rise_direction": [
        "surge", "soar", "jump", "rally", "spike", "climb", "rebound",
        "recover", "tighten", "tightening", "disruption", "shortage",
        "outage", "conflict", "sanctions", "hurricane", "attack",
    ],
    "lm_fall_direction": [
        "plunge", "drop", "decline", "slump", "tumble", "fall", "slide",
        "oversupply", "glut", "surplus", "bearish", "slowdown",
        "recession", "easing", "weakened",
    ],
}


# ── Auto-update benzinga_queries.py ───────────────────────────────────────────

def update_benzinga_queries(
    results: dict,
    queries_path: Path,
    query_group_map: dict,
) -> dict[str, list[str]]:
    """
    Patch benzinga_queries.py in-place, appending new keyword candidates
    to the topics list of each mapped QueryStrategy.

    Only terms not already present in the topics list are added.
    Returns a dict of {strategy_name: [newly_added_terms]}.
    """
    source = queries_path.read_text()
    added: dict[str, list[str]] = {}

    for group_name, strategy_name in query_group_map.items():
        candidates = results.get(group_name, [])
        if not candidates:
            continue

        # Locate this strategy's topics=[...] block inside its QueryStrategy(...)
        # Pattern: "<strategy_name>": QueryStrategy( ... topics=[ ... ] ...
        strat_pattern = re.compile(
            r'("' + re.escape(strategy_name) + r'"\s*:\s*QueryStrategy\(.*?topics=\[)(.*?)(\])',
            re.DOTALL,
        )
        match = strat_pattern.search(source)
        if not match:
            logging.warning(f"Could not locate strategy '{strategy_name}' in {queries_path}")
            continue

        existing_block = match.group(2)
        # Parse out terms already in the list (strip quotes and whitespace)
        existing_terms = {
            t.lower()
            for t in re.findall(r'"([^"]+)"', existing_block)
        }

        new_terms = [
            s["display"]
            for s in candidates
            if s["display"].lower() not in existing_terms
        ]
        if not new_terms:
            logging.info(f"  '{strategy_name}': no new terms to add (all already present)")
            continue

        # Format new entries with consistent indentation (8 spaces, matching the file)
        indent = "            "
        new_entries = "".join(f'{indent}"{t}",\n' for t in new_terms)

        # Insert before the closing ] of the topics list
        updated_block = match.group(1) + match.group(2) + new_entries + match.group(3)
        source = strat_pattern.sub(updated_block, source, count=1)
        added[strategy_name] = new_terms
        logging.info(f"  '{strategy_name}': adding {len(new_terms)} new terms: {new_terms}")

    if added:
        queries_path.write_text(source)
        logging.info(f"Updated {queries_path}")
    else:
        logging.info("No new terms to add — benzinga_queries.py unchanged")

    return added


# ── Data loading ───────────────────────────────────────────────────────────────

def _normalize_article_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply consistent filtering/sorting across all sources.
    """
    out = df.copy()
    for col in ["title", "teaser", "body"]:
        out[col] = out.get(col, "").fillna("").astype(str).str.strip()

    # Keep rows with at least one non-empty text field.
    has_text = (out["title"] != "") | (out["teaser"] != "") | (out["body"] != "")
    out = out[has_text]

    # Parse dates consistently and drop undated rows.
    out["date"] = (
        pd.to_datetime(out.get("date"), errors="coerce", utc=True)
        .dt.tz_convert(None)
    )
    out = out.dropna(subset=["id", "date"])

    # Same ordering rule for every source.
    out = out.sort_values(["date", "id"], kind="mergesort").reset_index(drop=True)
    return out


def load_all_benzinga(include_oilprice: bool = True) -> pd.DataFrame:
    """Load Benzinga tiers (+ optional OilPrice) for vocabulary coverage."""
    all_files = sorted(glob(str(RAW_DIR / "**" / "*.parquet"), recursive=True))
    if not all_files:
        raise FileNotFoundError(f"No Benzinga parquets found in {RAW_DIR}")
    benzinga_frames = [pd.read_parquet(f) for f in all_files]
    df_benzinga = pd.concat(benzinga_frames, ignore_index=True)
    df_benzinga = df_benzinga.drop_duplicates(subset="id")
    logging.info(f"Loaded {len(df_benzinga):,} unique Benzinga articles")

    if include_oilprice and OILPRICE_CSV.exists():
        oil_df = pd.read_csv(OILPRICE_CSV)
        if not oil_df.empty:
            oil_df = oil_df.copy()
            headline_col = oil_df["headline"] if "headline" in oil_df else pd.Series("", index=oil_df.index)
            body_col = oil_df["body_text"] if "body_text" in oil_df else pd.Series("", index=oil_df.index)
            url_col = oil_df["url"] if "url" in oil_df else pd.Series("", index=oil_df.index)
            published_col = oil_df["published_at"] if "published_at" in oil_df else pd.Series("", index=oil_df.index)

            oil_df["title"] = headline_col.fillna("").astype(str)
            oil_df["teaser"] = ""
            oil_df["body"] = body_col.fillna("").astype(str)
            oil_df["id"] = (
                "oilprice::" + url_col.fillna("").astype(str).str.strip()
            )
            empty_id = oil_df["id"] == "oilprice::"
            oil_df.loc[empty_id, "id"] = [
                f"oilprice::row_{i}" for i in oil_df.index[empty_id]
            ]
            oil_df["date"] = pd.to_datetime(published_col, errors="coerce", utc=True)
            keep_cols = ["id", "title", "teaser", "body", "date"]
            oil_df = oil_df[keep_cols].drop_duplicates(subset="id")
            logging.info(f"Loaded {len(oil_df):,} unique OilPrice articles")
            df = pd.concat([df_benzinga, oil_df], ignore_index=True, sort=False)
        else:
            df = df_benzinga
    else:
        if include_oilprice:
            logging.info("OilPrice corpus not found, continuing with Benzinga only")
        df = df_benzinga

    df = df.drop_duplicates(subset="id")
    df = _normalize_article_rows(df)
    logging.info(f"Loaded {len(df):,} filtered/sorted articles across combined corpus")
    return df


def build_token_doc_index(
    df: pd.DataFrame,
    bigram_phraser,
) -> tuple[dict[str, set[int]], int]:
    """
    Build token->doc_id index for PMI using bigram-transformed token sets.
    """
    token_docs: dict[str, set[int]] = defaultdict(set)
    doc_id = 0

    for _, row in df.iterrows():
        combined = " ".join(
            str(row.get(col, "") or "")
            for col in ["title", "teaser", "body"]
        ).strip()
        if not combined:
            continue

        tokens = tokenize_sentence(combined)
        if not tokens:
            continue

        doc_tokens = set(bigram_phraser[tokens])
        if not doc_tokens:
            continue

        for t in doc_tokens:
            token_docs[t].add(doc_id)
        doc_id += 1

    return token_docs, doc_id


def _mean_w2v_similarity(
    model,
    token: str,
    seed_tokens: list[str],
) -> Optional[float]:
    vals = []
    if token not in model.wv:
        return None

    for seed in seed_tokens:
        if seed in model.wv:
            vals.append(float(model.wv.similarity(token, seed)))
    if not vals:
        return None
    return sum(vals) / len(vals)


def _pair_pmi(
    token_docs: dict[str, set[int]],
    total_docs: int,
    a: str,
    b: str,
) -> Optional[float]:
    if total_docs == 0:
        return None
    docs_a = token_docs.get(a)
    docs_b = token_docs.get(b)
    if not docs_a or not docs_b:
        return None

    inter = len(docs_a & docs_b)
    if inter == 0:
        return None

    pa = len(docs_a) / total_docs
    pb = len(docs_b) / total_docs
    pab = inter / total_docs
    return math.log2(pab / (pa * pb))


def _mean_pmi(
    token_docs: dict[str, set[int]],
    total_docs: int,
    token: str,
    seed_tokens: list[str],
) -> Optional[float]:
    vals = []
    for seed in seed_tokens:
        v = _pair_pmi(token_docs, total_docs, token, seed)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return sum(vals) / len(vals)


def build_expanded_lm_seeds(
    df: pd.DataFrame,
    bigram_phraser,
    h_threshold: float,
    min_freq: int,
    max_added_per_side: int,
) -> tuple[list[str], list[str]]:
    """
    Build expanded rise/fall seed lists using frequency-gap threshold H.
    """
    rise_counts: dict[str, int] = defaultdict(int)
    fall_counts: dict[str, int] = defaultdict(int)

    for _, row in df.iterrows():
        headline = str(row.get("title", "") or "")
        label = label_price_direction(headline)
        if label not in {"rise", "fall"}:
            continue

        combined = " ".join(
            str(row.get(col, "") or "")
            for col in ["title", "teaser", "body"]
        ).strip()
        if not combined:
            continue

        tokens = tokenize_sentence(combined)
        if not tokens:
            continue

        for tok in bigram_phraser[tokens]:
            if label == "rise":
                rise_counts[tok] += 1
            else:
                fall_counts[tok] += 1

    base_rise = set(
        t for s in KEYWORD_GROUPS["lm_rise_direction"]
        for t in [seed_to_model_token(s, bigram_phraser)]
        if t
    )
    base_fall = set(
        t for s in KEYWORD_GROUPS["lm_fall_direction"]
        for t in [seed_to_model_token(s, bigram_phraser)]
        if t
    )

    rise_seeds = set(base_rise)
    fall_seeds = set(base_fall)

    rise_total = sum(rise_counts.values())
    fall_total = sum(fall_counts.values())
    if rise_total == 0 or fall_total == 0:
        logging.warning("LM-S seed expansion skipped: no rise/fall labeled token counts")
        return sorted(rise_seeds), sorted(fall_seeds)

    rise_candidates = []
    fall_candidates = []
    vocab = set(rise_counts) | set(fall_counts)
    for tok in vocab:
        if tok in rise_seeds or tok in fall_seeds:
            continue
        rc = rise_counts.get(tok, 0)
        fc = fall_counts.get(tok, 0)
        if rc + fc < min_freq:
            continue

        rise_rate = rc / rise_total
        fall_rate = fc / fall_total
        gap = rise_rate - fall_rate
        if gap > h_threshold:
            rise_candidates.append((tok, gap))
        elif -gap > h_threshold:
            fall_candidates.append((tok, -gap))

    rise_candidates.sort(key=lambda x: x[1], reverse=True)
    fall_candidates.sort(key=lambda x: x[1], reverse=True)
    for tok, _ in rise_candidates[:max_added_per_side]:
        rise_seeds.add(tok)
    for tok, _ in fall_candidates[:max_added_per_side]:
        fall_seeds.add(tok)

    logging.info(
        "LM-S seeds via H-threshold: rise=%d (base=%d), fall=%d (base=%d), "
        "added_rise=%d, added_fall=%d",
        len(rise_seeds), len(base_rise), len(fall_seeds), len(base_fall),
        min(len(rise_candidates), max_added_per_side),
        min(len(fall_candidates), max_added_per_side),
    )
    return sorted(rise_seeds), sorted(fall_seeds)


def filter_results_with_relevance_agreement(
    results: dict,
    model,
    bigram_phraser,
    token_docs: dict[str, set[int]],
    total_docs: int,
    pmi_min: float,
    cosine_min: float,
) -> dict:
    """
    Keep query/topic candidates only when PMI and Word2Vec both indicate relevance.
    """
    filtered = {}

    for group_name, suggestions in results.items():
        # Only gate groups that are mapped to query strategy updates.
        if group_name not in QUERY_GROUP_MAP:
            filtered[group_name] = suggestions
            continue

        seed_tokens = [
            t
            for t in (
                seed_to_model_token(seed, bigram_phraser)
                for seed in KEYWORD_GROUPS.get(group_name, [])
            )
            if t
        ]
        if not seed_tokens:
            filtered[group_name] = []
            continue

        kept = []
        for s in suggestions:
            token = s["word"]
            mean_cos = _mean_w2v_similarity(model, token, seed_tokens)
            mean_pmi = _mean_pmi(token_docs, total_docs, token, seed_tokens)
            if mean_cos is None or mean_pmi is None:
                continue
            if mean_cos >= cosine_min and mean_pmi >= pmi_min:
                s2 = dict(s)
                s2["mean_seed_cosine"] = round(mean_cos, 4)
                s2["mean_seed_pmi"] = round(mean_pmi, 4)
                kept.append(s2)

        filtered[group_name] = sorted(kept, key=lambda x: x["similarity"], reverse=True)
        logging.info(
            "Relevance gate '%s': kept %d / %d (cos>=%.2f, pmi>=%.2f)",
            group_name, len(kept), len(suggestions), cosine_min, pmi_min,
        )

    return filtered


def refine_lm_direction_groups(
    results: dict,
    model,
    token_docs: dict[str, set[int]],
    total_docs: int,
    rise_seeds: list[str],
    fall_seeds: list[str],
    pmi_polarity_min: float,
    cosine_polarity_min: float,
    neutral_band: float,
    min_vote_confidence: float,
) -> dict:
    """
    Re-assign LM rise/fall candidate words using PMI+Word2Vec polarity agreement.
    """
    rise_candidates = results.get("lm_rise_direction", [])
    fall_candidates = results.get("lm_fall_direction", [])

    candidate_map = {}
    for s in rise_candidates + fall_candidates:
        candidate_map[s["word"]] = s

    existing = {
        t.lower()
        for t in KEYWORD_GROUPS["lm_rise_direction"] + KEYWORD_GROUPS["lm_fall_direction"]
    }

    rise_out = []
    fall_out = []

    for token, base in candidate_map.items():
        display = token.replace("_", " ")
        if display.lower() in existing:
            continue

        pmi_rise = _mean_pmi(token_docs, total_docs, token, rise_seeds)
        pmi_fall = _mean_pmi(token_docs, total_docs, token, fall_seeds)
        w2v_rise = _mean_w2v_similarity(model, token, rise_seeds)
        w2v_fall = _mean_w2v_similarity(model, token, fall_seeds)
        if None in (pmi_rise, pmi_fall, w2v_rise, w2v_fall):
            continue

        pmi_score = pmi_rise - pmi_fall
        w2v_score = w2v_rise - w2v_fall

        if abs(pmi_score) < pmi_polarity_min and abs(w2v_score) < cosine_polarity_min:
            continue

        # Integration rules (paper-style fusion):
        # Derive rise/neutral/fall pseudo-probabilities from each base model,
        # then fuse the two model votes.
        def _triple_from_score(score: float) -> tuple[float, float, float]:
            rise = max(score, 0.0)
            fall = max(-score, 0.0)
            neutral = max(0.0, neutral_band - abs(score))
            total = rise + neutral + fall
            if total <= 0:
                return (0.0, 1.0, 0.0)
            return (fall / total, neutral / total, rise / total)

        pmi_fall, pmi_neutral, pmi_rise_p = _triple_from_score(pmi_score)
        w2v_fall, w2v_neutral, w2v_rise_p = _triple_from_score(w2v_score)

        vote_fall = pmi_fall + w2v_fall
        vote_neutral = pmi_neutral + w2v_neutral
        vote_rise = pmi_rise_p + w2v_rise_p
        vote_sum = vote_fall + vote_neutral + vote_rise
        if vote_sum <= 0:
            continue

        conf_fall = vote_fall / vote_sum
        conf_neutral = vote_neutral / vote_sum
        conf_rise = vote_rise / vote_sum

        label = "neutral"
        label_conf = conf_neutral
        if conf_rise >= conf_fall and conf_rise >= conf_neutral:
            label = "rise"
            label_conf = conf_rise
        elif conf_fall >= conf_rise and conf_fall >= conf_neutral:
            label = "fall"
            label_conf = conf_fall

        if label == "neutral" or label_conf < min_vote_confidence:
            continue

        out = dict(base)
        out["display"] = display
        out["pmi_polarity"] = round(pmi_score, 4)
        out["w2v_polarity"] = round(w2v_score, 4)
        out["vote_rise"] = round(conf_rise, 4)
        out["vote_neutral"] = round(conf_neutral, 4)
        out["vote_fall"] = round(conf_fall, 4)
        out["agreement_score"] = round(label_conf, 6)

        if label == "rise":
            rise_out.append(out)
        elif label == "fall":
            fall_out.append(out)

    rise_out.sort(key=lambda x: x["agreement_score"], reverse=True)
    fall_out.sort(key=lambda x: x["agreement_score"], reverse=True)

    logging.info(
        "LM polarity gate: rise %d->%d, fall %d->%d",
        len(rise_candidates), len(rise_out), len(fall_candidates), len(fall_out),
    )

    results = dict(results)
    results["lm_rise_direction"] = rise_out
    results["lm_fall_direction"] = fall_out
    return results


# ── Pretty printing ────────────────────────────────────────────────────────────

def print_expansion_report(results: dict, top_display: int = 10):
    print("\n" + "=" * 72)
    print("KEYWORD EXPANSION REPORT")
    print("=" * 72)

    for group, suggestions in results.items():
        print(f"\n▸ {group.upper().replace('_', ' ')}  ({len(suggestions)} new candidates)")
        if not suggestions:
            print("    (no candidates above threshold)")
            continue
        for i, s in enumerate(suggestions[:top_display], 1):
            print(f"    {i:2d}. {s['display']:<30}  sim={s['similarity']:.3f}  "
                  f"← '{s['similar_to']}'")
        if len(suggestions) > top_display:
            print(f"    ... and {len(suggestions) - top_display} more (see expanded_keywords.json)")

    print("\n" + "=" * 72)
    total = sum(len(v) for v in results.values())
    print(f"Total new candidates across all groups: {total}")
    print(f"Full results saved to: {OUT_JSON}")
    print("=" * 72)


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Expand oil-market keyword lists using Word2Vec")
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="Minimum cosine similarity (default: 0.65)")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Candidates to retrieve per seed (default: 20)")
    parser.add_argument("--save-model", action="store_true",
                        help="Save the trained Word2Vec model to data/word2vec.model")
    parser.add_argument("--auto-update", action="store_true",
                        help="Automatically append new terms to benzinga_queries.py")
    parser.add_argument("--no-oilprice", action="store_true",
                        help="Disable adding OilPrice articles to the training corpus")
    parser.add_argument("--h-threshold", type=float, default=0.00012,
                        help="LM-S normalized frequency-gap threshold H (default: 0.00012)")
    parser.add_argument("--lm-min-freq", type=int, default=3,
                        help="Minimum token frequency for LM-S seed expansion (default: 3)")
    parser.add_argument("--lm-max-seeds-per-side", type=int, default=500,
                        help="Maximum added LM-S seeds per polarity side (default: 500)")
    parser.add_argument("--relevance-pmi-min", type=float, default=0.02,
                        help="PMI threshold for query/topic relevance gating (default: 0.02)")
    parser.add_argument("--relevance-cosine-min", type=float, default=0.50,
                        help="Cosine threshold for query/topic relevance gating (default: 0.50)")
    parser.add_argument("--polarity-pmi-min", type=float, default=0.05,
                        help="Minimum |PMI polarity| for LM rise/fall agreement (default: 0.05)")
    parser.add_argument("--polarity-cosine-min", type=float, default=0.02,
                        help="Minimum |cosine polarity| for LM rise/fall agreement (default: 0.02)")
    parser.add_argument("--polarity-neutral-band", type=float, default=0.03,
                        help="Neutral band width for PMI/W2V vote fusion (default: 0.03)")
    parser.add_argument("--polarity-min-vote-confidence", type=float, default=0.35,
                        help="Minimum fused vote confidence to keep rise/fall candidate (default: 0.35)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 1. Load corpus
    logging.info("Loading Benzinga corpus...")
    df = load_all_benzinga(include_oilprice=not args.no_oilprice)

    # 2. Build tokenized corpus (title + teaser + body for rich vocabulary)
    logging.info("Building tokenized corpus...")
    corpus = build_corpus(df, text_cols=["title", "teaser", "body"])
    logging.info(f"Corpus ready: {len(corpus):,} documents")

    # 3. Train Word2Vec with bigram detection
    model, bigram_phraser = train_word2vec(corpus)

    if args.save_model:
        model_path = BASE_DIR / "data" / "word2vec.model"
        model.save(str(model_path))
        logging.info(f"Model saved → {model_path}")

    # 4. Expand all keyword groups
    logging.info(f"Expanding keywords (threshold={args.threshold}, top_n={args.top_n})...")
    results = expand_all_groups(
        model,
        bigram_phraser,
        keyword_groups=KEYWORD_GROUPS,
        top_n=args.top_n,
        similarity_threshold=args.threshold,
    )

    # 5. Build PMI index for dual-model filtering
    logging.info("Building token-document index for PMI...")
    token_docs, total_docs = build_token_doc_index(df, bigram_phraser)
    logging.info("PMI index ready: %d docs, %d tokens", total_docs, len(token_docs))

    # 6. Gate mapped query/topic groups by PMI+Word2Vec relevance agreement
    results = filter_results_with_relevance_agreement(
        results,
        model,
        bigram_phraser,
        token_docs=token_docs,
        total_docs=total_docs,
        pmi_min=args.relevance_pmi_min,
        cosine_min=args.relevance_cosine_min,
    )

    # 7. Build LM-S seeds using H threshold, then enforce PMI+Word2Vec polarity agreement
    rise_seeds, fall_seeds = build_expanded_lm_seeds(
        df,
        bigram_phraser,
        h_threshold=args.h_threshold,
        min_freq=args.lm_min_freq,
        max_added_per_side=args.lm_max_seeds_per_side,
    )
    results = refine_lm_direction_groups(
        results,
        model,
        token_docs=token_docs,
        total_docs=total_docs,
        rise_seeds=rise_seeds,
        fall_seeds=fall_seeds,
        pmi_polarity_min=args.polarity_pmi_min,
        cosine_polarity_min=args.polarity_cosine_min,
        neutral_band=args.polarity_neutral_band,
        min_vote_confidence=args.polarity_min_vote_confidence,
    )

    # 8. Save full results to JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # 9. Print summary
    print_expansion_report(results, top_display=10)

    # 10. Optionally patch benzinga_queries.py with new terms
    if args.auto_update:
        logging.info(f"Auto-updating {QUERIES_PY}...")
        added = update_benzinga_queries(results, QUERIES_PY, QUERY_GROUP_MAP)
        if added:
            print("\nTerms added to benzinga_queries.py:")
            for strategy, terms in added.items():
                print(f"  [{strategy}] {terms}")
            print("\nRun scripts/run_backfill.py to fetch articles for the new terms.")
    else:
        disruption_suggestions = results.get("disruption_query", [])
        if disruption_suggestions:
            print("\nQuick-copy — new disruption terms (or run with --auto-update):")
            terms = [f'"{s["display"]}"' for s in disruption_suggestions[:15]]
            print("    " + ",\n    ".join(terms))
