"""Bag-of-words feature extraction for text classification."""

from __future__ import annotations

import re
from string import digits, punctuation
from typing import Dict, Iterable, List, Tuple

import numpy as np

Array = np.ndarray
STOP = set("a an the and or but in on at to for of is was are were be been being".split())


def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", punctuation + digits))
    return [w for w in text.split() if w and w not in STOP]


def bag_of_words(texts: Iterable[str], min_count: int = 1) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for text in texts:
        for token in tokenize(text):
            counts[token] = counts.get(token, 0) + 1
    return {word: idx for idx, (word, _) in enumerate(
        sorted((w, c) for w, c in counts.items() if c >= min_count)
    )}


def extract_bow_feature_vectors(texts: Iterable[str], dictionary: Dict[str, int]) -> Array:
    texts = list(texts)
    dim = len(dictionary)
    matrix = np.zeros((len(texts), dim))
    for row, text in enumerate(texts):
        for token in tokenize(text):
            if token in dictionary:
                matrix[row, dictionary[token]] += 1
    return matrix


def load_toy_data(path: str) -> Tuple[Array, Array]:
    """Load 2D toy dataset from MITx 6.86x (label, x, y per row)."""
    labels, xs, ys = np.loadtxt(path, delimiter="\t", unpack=True)
    return np.vstack((xs, ys)).T, labels
