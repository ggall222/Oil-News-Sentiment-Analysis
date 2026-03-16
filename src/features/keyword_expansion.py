# src/features/keyword_expansion.py
#
# Word2Vec-based keyword expansion for oil-market terminology.
#
# Workflow:
#   1. Train a Word2Vec model on the full Benzinga article corpus
#      using gensim Phrases to detect common bigrams first
#      (so "production_cut", "supply_disruption" become single tokens)
#   2. Given a list of seed keywords, query the model for nearest neighbours
#   3. Filter candidates by similarity threshold and novelty
#      (reject anything already in any existing keyword list)
#   4. Return ranked suggestions per seed → per keyword group

import re
import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ── Stop words (shared with lm_sentiment.py) ─────────────────────────────────
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
    "after", "before", "into", "more", "most", "just", "over", "new",
    "per", "s", "t", "re", "ve", "ll", "d", "said", "says", "say",
}


def tokenize_sentence(text: str) -> List[str]:
    """Lowercase, strip punctuation, remove stop words, min length 3."""
    tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return [t for t in tokens if t not in STOP_WORDS]


def build_corpus(df: pd.DataFrame, text_cols: List[str]) -> List[List[str]]:
    """
    Build a tokenized sentence corpus from a DataFrame.
    Each row becomes one 'document' (list of tokens).
    """
    corpus = []
    for _, row in df.iterrows():
        parts = []
        for col in text_cols:
            val = row.get(col, "")
            if isinstance(val, str) and val.strip():
                parts.append(val)
        combined = " ".join(parts)
        tokens = tokenize_sentence(combined)
        if tokens:
            corpus.append(tokens)
    return corpus


def train_word2vec(
    corpus: List[List[str]],
    vector_size: int = 150,
    window: int = 6,
    min_count: int = 3,
    workers: int = 4,
    epochs: int = 15,
    bigram_min_count: int = 5,
    bigram_threshold: float = 10.0,
):
    """
    Train a Word2Vec model on the corpus with automatic bigram detection.

    Bigrams are detected first (e.g. 'production' + 'cut' → 'production_cut')
    so multi-word seed terms can be looked up in the model vocabulary.

    Returns (model, bigram_phraser) so callers can transform new text.
    """
    try:
        from gensim.models import Word2Vec, Phrases
        from gensim.models.phrases import Phraser
    except ImportError:
        raise ImportError(
            "gensim is required for keyword expansion.\n"
            "Install with:  pip install gensim"
        )

    logger.info(f"Detecting bigrams (min_count={bigram_min_count}, threshold={bigram_threshold})...")
    phrases = Phrases(corpus, min_count=bigram_min_count, threshold=bigram_threshold)
    bigram_phraser = Phraser(phrases)

    bigrammed_corpus = [bigram_phraser[doc] for doc in corpus]

    unique_tokens = set(t for doc in bigrammed_corpus for t in doc)
    logger.info(f"Corpus: {len(bigrammed_corpus):,} docs, {len(unique_tokens):,} unique tokens")

    logger.info(f"Training Word2Vec (size={vector_size}, window={window}, epochs={epochs})...")
    model = Word2Vec(
        sentences=bigrammed_corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        sg=1,             # skip-gram (better for rare words)
    )
    logger.info(f"Vocabulary size: {len(model.wv):,} terms")
    return model, bigram_phraser


def seed_to_model_token(seed: str, bigram_phraser) -> Optional[str]:
    """
    Convert a human-readable seed phrase to its model token form.

    "production cut"  →  "production_cut"  (if detected as bigram)
    "pipeline"        →  "pipeline"
    Returns None if the token is not in the model vocabulary.
    """
    # Tokenize and run through bigram phraser
    tokens = tokenize_sentence(seed)
    if not tokens:
        return None
    phrased = bigram_phraser[tokens]
    # Take the longest resulting token (the compound one if it exists)
    candidate = "_".join(phrased) if len(phrased) > 1 else phrased[0]
    return candidate


def expand_keywords(
    model,
    bigram_phraser,
    seed_keywords: List[str],
    all_existing_keywords: List[str],
    top_n: int = 10,
    similarity_threshold: float = 0.60,
) -> List[Dict]:
    """
    Find new keyword candidates similar to the seed list.

    Parameters
    ----------
    model                 : trained Word2Vec model
    bigram_phraser        : gensim Phraser for bigram detection
    seed_keywords         : list of existing keywords to use as anchors
    all_existing_keywords : flat list of ALL known keywords (to filter out non-novel suggestions)
    top_n                 : number of neighbours to retrieve per seed
    similarity_threshold  : minimum cosine similarity to include a suggestion

    Returns
    -------
    List of dicts: {word, display, similarity, similar_to}
    sorted by similarity descending, deduplicated.
    """
    # Normalise existing keywords for novelty check
    existing_normalised = set()
    for kw in all_existing_keywords:
        tokens = tokenize_sentence(kw)
        phrased = bigram_phraser[tokens]
        token = "_".join(phrased) if len(phrased) > 1 else (phrased[0] if phrased else "")
        if token:
            existing_normalised.add(token)

    seen: Dict[str, float] = {}   # token → best similarity seen
    details: Dict[str, Dict] = {}

    for seed in seed_keywords:
        token = seed_to_model_token(seed, bigram_phraser)
        if token is None or token not in model.wv:
            continue

        try:
            neighbours = model.wv.most_similar(token, topn=top_n)
        except KeyError:
            continue

        for neighbour, sim in neighbours:
            if sim < similarity_threshold:
                continue
            if neighbour in existing_normalised:
                continue
            if len(neighbour.replace("_", "")) < 4:   # skip very short tokens
                continue

            if neighbour not in seen or sim > seen[neighbour]:
                seen[neighbour] = sim
                details[neighbour] = {
                    "word": neighbour,
                    "display": neighbour.replace("_", " "),
                    "similarity": round(sim, 4),
                    "similar_to": seed,
                }

    return sorted(details.values(), key=lambda x: x["similarity"], reverse=True)


def expand_all_groups(
    model,
    bigram_phraser,
    keyword_groups: Dict[str, List[str]],
    top_n: int = 10,
    similarity_threshold: float = 0.60,
) -> Dict[str, List[Dict]]:
    """
    Expand multiple keyword groups at once.

    Parameters
    ----------
    keyword_groups : {group_name: [seed_keywords]}

    Returns
    -------
    {group_name: [suggestion_dicts]}
    """
    # Build flat list of ALL existing keywords for novelty filtering
    all_existing = [kw for seeds in keyword_groups.values() for kw in seeds]

    results = {}
    for group_name, seeds in keyword_groups.items():
        logger.info(f"Expanding '{group_name}' ({len(seeds)} seeds)...")
        suggestions = expand_keywords(
            model,
            bigram_phraser,
            seed_keywords=seeds,
            all_existing_keywords=all_existing,
            top_n=top_n,
            similarity_threshold=similarity_threshold,
        )
        results[group_name] = suggestions
        logger.info(f"  → {len(suggestions)} new candidates found")

    return results
