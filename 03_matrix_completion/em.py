"""Gaussian mixture EM for matrix completion — MITx 6.86x Project 4 concepts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.special import logsumexp

Array = np.ndarray


@dataclass
class GaussianMixture:
    n_components: int
    mu: Array
    var: Array
    alpha: Array


def init_mixture(X: Array, k: int, seed: int = 0) -> GaussianMixture:
    rng = np.random.default_rng(seed)
    n, d = X.shape
    idx = rng.choice(n, size=k, replace=False)
    mu = X[idx].copy()
    var = np.ones((k, d))
    alpha = np.ones(k) / k
    return GaussianMixture(k, mu, var, alpha)


def estep(X: Array, mixture: GaussianMixture) -> Tuple[Array, float]:
    n, d = X.shape
    k = mixture.n_components
    log_prob = np.zeros((n, k))
    for j in range(k):
        diff = X - mixture.mu[j]
        log_norm = -0.5 * np.sum(diff ** 2 / mixture.var[j], axis=1)
        log_norm -= 0.5 * np.sum(np.log(2 * np.pi * mixture.var[j]))
        log_prob[:, j] = np.log(mixture.alpha[j] + 1e-12) + log_norm
    log_total = logsumexp(log_prob, axis=1, keepdims=True)
    post = np.exp(log_prob - log_total)
    ll = float(np.sum(log_total))
    return post, ll


def mstep(X: Array, post: Array, mixture: GaussianMixture, min_variance: float = 0.25) -> GaussianMixture:
    n, d = X.shape
    k = mixture.n_components
    nk = post.sum(axis=0) + 1e-12
    alpha = nk / n
    mu = (post.T @ X) / nk[:, None]
    var = np.zeros((k, d))
    for j in range(k):
        diff = X - mu[j]
        var[j] = np.sum(post[:, j][:, None] * diff ** 2, axis=0) / nk[j]
        var[j] = np.maximum(var[j], min_variance)
    return GaussianMixture(k, mu, var, alpha)


def run_em(X: Array, k: int = 2, max_iter: int = 50, tol: float = 1e-4) -> Tuple[GaussianMixture, Array]:
    mixture = init_mixture(X, k)
    post = np.ones((X.shape[0], k)) / k
    prev_ll = -np.inf
    for _ in range(max_iter):
        post, ll = estep(X, mixture)
        if abs(ll - prev_ll) < tol:
            break
        mixture = mstep(X, post, mixture)
        prev_ll = ll
    return mixture, post


def complete_matrix(X_incomplete: Array, mixture: GaussianMixture, post: Array) -> Array:
    X_full = X_incomplete.copy()
    mask = X_incomplete == 0
    for i in range(X_incomplete.shape[0]):
        for j in range(X_incomplete.shape[1]):
            if mask[i, j]:
                X_full[i, j] = float(post[i] @ mixture.mu[:, j])
    return X_full
