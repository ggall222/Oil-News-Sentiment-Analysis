"""
Paragraph-level multi-label topic classifier for 8 oil-news categories.

Workflow:
1) Split article text into paragraphs.
2) Build weak paragraph labels using existing rule-based topic keywords.
3) Train one-vs-rest ML classifiers on TF-IDF vectors.
4) Predict paragraph topic probabilities and aggregate to article multi-label topics.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer

from src.features.topic_classifier import CATEGORIES, classify_topics

try:
    from lightgbm import LGBMClassifier
    _HAS_LIGHTGBM = True
except Exception:
    _HAS_LIGHTGBM = False


_PARA_SPLIT = re.compile(r"\n\s*\n+")


def split_paragraphs(text: str, min_chars: int = 60) -> List[str]:
    """Split text to paragraph-like chunks with a sentence fallback."""
    if not isinstance(text, str):
        return []
    txt = text.strip()
    if not txt:
        return []

    raw = [p.strip() for p in _PARA_SPLIT.split(txt) if p.strip()]
    if len(raw) <= 1:
        # Fallback: break long single blocks into sentence groups.
        sents = re.split(r"(?<=[.!?])\s+", txt)
        groups: List[str] = []
        cur: List[str] = []
        cur_len = 0
        for s in sents:
            s = s.strip()
            if not s:
                continue
            cur.append(s)
            cur_len += len(s) + 1
            if cur_len >= 260:
                groups.append(" ".join(cur).strip())
                cur = []
                cur_len = 0
        if cur:
            groups.append(" ".join(cur).strip())
        raw = groups if groups else [txt]

    return [p for p in raw if len(p) >= min_chars]


def build_paragraph_frame(
    articles: pd.DataFrame,
    text_col: str = "text",
    date_col: str = "date",
    min_chars: int = 60,
) -> pd.DataFrame:
    """Explode article-level rows to paragraph-level rows."""
    rows = []
    base = articles.reset_index(drop=True).copy()

    for article_idx, row in base.iterrows():
        text = str(row.get(text_col, "") or "")
        paras = split_paragraphs(text, min_chars=min_chars)
        if not paras and text.strip():
            paras = [text.strip()]
        for para_idx, para in enumerate(paras):
            rows.append(
                {
                    "article_idx": article_idx,
                    "paragraph_idx": para_idx,
                    "date": row.get(date_col),
                    "paragraph_text": para,
                    "weak_labels": classify_topics(para),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "paragraph_text"]).reset_index(drop=True)
    return out


@dataclass
class ParagraphTopicModel:
    vectorizer: TfidfVectorizer
    binarizer: MultiLabelBinarizer
    estimators: List[object]
    class_priors: np.ndarray
    model_name: str

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        x = self.vectorizer.transform(texts)
        out = np.zeros((x.shape[0], len(CATEGORIES)), dtype=float)
        for i, est in enumerate(self.estimators):
            if est is None:
                out[:, i] = self.class_priors[i]
                continue
            p = est.predict_proba(x)
            out[:, i] = p[:, 1]
        return out


def train_paragraph_classifier(
    paragraph_df: pd.DataFrame,
    min_pos_per_class: int = 30,
) -> ParagraphTopicModel:
    """
    Train multi-label paragraph topic classifier from weak labels.
    """
    train_df = paragraph_df[paragraph_df["weak_labels"].map(bool)].copy()
    if train_df.empty:
        raise ValueError("No weakly-labeled paragraphs available for training.")

    n_train = len(train_df)
    min_df = 3 if n_train >= 200 else (2 if n_train >= 40 else 1)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.95,
        max_features=60000,
    )
    x = vectorizer.fit_transform(train_df["paragraph_text"].astype(str).tolist())

    mlb = MultiLabelBinarizer(classes=CATEGORIES)
    y = mlb.fit_transform(train_df["weak_labels"])

    class_pos = np.array([int(y[:, i].sum()) for i in range(y.shape[1])], dtype=int)

    if _HAS_LIGHTGBM:
        base = LGBMClassifier(
            objective="binary",
            learning_rate=0.05,
            n_estimators=250,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
        )
        model_name = "lightgbm"
    else:
        warnings.warn(
            "LightGBM unavailable in this environment (missing runtime deps). "
            "Falling back to LogisticRegression.",
            RuntimeWarning,
        )
        base = LogisticRegression(
            max_iter=400,
            class_weight="balanced",
            solver="liblinear",
        )
        model_name = "logistic_regression"

    estimators: List[object] = []
    class_priors = np.zeros(len(CATEGORIES), dtype=float)
    for i in range(len(CATEGORIES)):
        yi = y[:, i]
        class_priors[i] = float(yi.mean())
        if yi.sum() < min_pos_per_class or yi.sum() == len(yi):
            estimators.append(None)
            continue
        est = base.__class__(**base.get_params())
        est.fit(x, yi)
        estimators.append(est)

    return ParagraphTopicModel(
        vectorizer=vectorizer,
        binarizer=mlb,
        estimators=estimators,
        class_priors=class_priors,
        model_name=model_name,
    )


def assign_topics_from_paragraph_model(
    articles: pd.DataFrame,
    text_col: str = "text",
    date_col: str = "date",
    topics_col: str = "lm_topics",
    paragraph_proba_threshold: float = 0.40,
    min_chars: int = 60,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    """
    Train paragraph model and assign article-level multi-label topics from
    paragraph predictions.
    """
    para = build_paragraph_frame(articles, text_col=text_col, date_col=date_col, min_chars=min_chars)
    if para.empty:
        out = articles.copy()
        out[topics_col] = ""
        return out, {"model": None, "paragraphs": 0, "labeled_paragraphs": 0}

    model = train_paragraph_classifier(para)
    proba = model.predict_proba(para["paragraph_text"].astype(str).tolist())

    para_pred_labels = []
    for row in proba:
        labels = [c for c, p in zip(CATEGORIES, row) if p >= paragraph_proba_threshold]
        para_pred_labels.append(labels)
    para["pred_labels"] = para_pred_labels

    # Aggregate paragraph predictions to article-level union of labels.
    article_topics: Dict[int, set[str]] = {}
    for article_idx, labels in zip(para["article_idx"], para["pred_labels"]):
        if article_idx not in article_topics:
            article_topics[article_idx] = set()
        article_topics[article_idx].update(labels)

    out = articles.reset_index(drop=True).copy()
    out[topics_col] = [
        ",".join(sorted(article_topics.get(i, set())))
        for i in range(len(out))
    ]

    info = {
        "model": model.model_name,
        "paragraphs": int(len(para)),
        "labeled_paragraphs": int(para["weak_labels"].map(bool).sum()),
        "article_topic_coverage": int((out[topics_col] != "").sum()),
    }
    return out, info
