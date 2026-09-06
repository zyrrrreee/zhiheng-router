"""Query-only features. Dataset task_type and observed outcomes are never inputs."""

from collections.abc import Sequence
import re
from typing import Self

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .schemas import FeatureConfig, nonempty

CODE = re.compile(r"```|\b(?:def|class|import|return|SELECT)\b|代码|编程|Python|SQL", re.I)
MATH = re.compile(r"[=+*/^∑∫√≤≥×÷]|\\(?:frac|sum|sqrt)")
SEGMENTS = re.compile(r"[\u3400-\u9fff]|[A-Za-z_]+|\d+(?:\.\d+)?")


def task_type_hint(query: str) -> str:
    """Deterministic rules for the rule baseline only, not generator task labels."""
    nonempty(query, "query")
    if CODE.search(query):
        return "code"
    if re.search(r"翻译|译成|译为|translate", query, re.I):
        return "translation"
    if re.search(r"摘要|总结|概括|提炼|summari", query, re.I):
        return "summary"
    if MATH.search(query) or re.search(r"计算|求解|证明|概率|方程|几何|数列", query):
        return "math"
    return "qa"


def structural_features(query: str) -> list[float]:
    nonempty(query, "query")
    text = query.strip()
    return [float(np.log1p(len(text))), float(np.log1p(len(SEGMENTS.findall(text)))),
            float(bool(CODE.search(text))), float(any(c.isdigit() for c in text)),
            float(bool(MATH.search(text)))]


class FeatureExtractor:
    """One shared, sparse transformer fitted only on unique training Query text."""

    def __init__(self, config: FeatureConfig = FeatureConfig()):
        self.config = config
        self.vectorizer: TfidfVectorizer | None = None
        self.scaler = StandardScaler(with_mean=False)
        self.fitted = False

    def fit(self, queries: Sequence[str]) -> Self:
        for query in queries:
            nonempty(query, "query")
        unique = sorted({q.strip() for q in queries})
        if not unique:
            raise ValueError("no training queries")
        self.scaler.fit(np.asarray([structural_features(q) for q in unique]))
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=tuple(self.config.ngram_range),
                                     max_features=self.config.max_features, dtype=np.float64)
        # An entirely short-text training corpus can legitimately have no n-grams.
        analyzer = vectorizer.build_analyzer()
        if any(analyzer(q) for q in unique):
            self.vectorizer = vectorizer.fit(unique)
        else:
            self.vectorizer = None
        self.fitted = True
        return self

    def transform(self, queries: Sequence[str]) -> sparse.csr_matrix:
        if not self.fitted:
            raise ValueError("FeatureExtractor is not fitted")
        for query in queries:
            nonempty(query, "query")
        if not queries:
            raise ValueError("queries is empty")
        unique = sorted({q.strip() for q in queries})
        index = {q: i for i, q in enumerate(unique)}
        numeric = sparse.csr_matrix(self.scaler.transform(
            np.asarray([structural_features(q) for q in unique])))
        text = (self.vectorizer.transform(unique) if self.vectorizer is not None
                else sparse.csr_matrix((len(unique), 0)))
        matrix = sparse.hstack([numeric, text], format="csr")
        return matrix[[index[q.strip()] for q in queries]]
