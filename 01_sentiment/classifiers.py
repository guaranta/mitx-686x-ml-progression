"""Linear classifiers for sentiment analysis — MITx 6.86x Project 1 concepts."""

from __future__ import annotations

import random
from typing import Callable, Iterable, Tuple

import numpy as np

Array = np.ndarray


def get_order(n_samples: int, seed: int = 1) -> list[int]:
    rng = random.Random(seed)
    indices = list(range(n_samples))
    rng.shuffle(indices)
    return indices


def hinge_loss_single(feature_vector: Array, label: float, theta: Array, theta_0: float) -> float:
    margin = label * (np.dot(theta, feature_vector) + theta_0)
    return max(0.0, 1.0 - margin)


def hinge_loss_full(feature_matrix: Array, labels: Array, theta: Array, theta_0: float) -> float:
    margins = labels * (feature_matrix @ theta + theta_0)
    return float(np.mean(np.maximum(0.0, 1.0 - margins)))


def perceptron_single_step_update(
    feature_vector: Array,
    label: float,
    current_theta: Array,
    current_theta_0: float,
) -> Tuple[Array, float]:
    if label * (np.dot(current_theta, feature_vector) + current_theta_0) <= 0:
        return current_theta + label * feature_vector, current_theta_0 + label
    return current_theta, current_theta_0


def perceptron(feature_matrix: Array, labels: Array, T: int) -> Tuple[Array, float]:
    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0.0
    for _ in range(T):
        for i in get_order(feature_matrix.shape[0]):
            theta, theta_0 = perceptron_single_step_update(
                feature_matrix[i], labels[i], theta, theta_0
            )
    return theta, theta_0


def average_perceptron(feature_matrix: Array, labels: Array, T: int) -> Tuple[Array, float]:
    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0.0
    theta_sum = np.zeros_like(theta)
    theta_0_sum = 0.0
    updates = 0
    for _ in range(T):
        for i in get_order(feature_matrix.shape[0]):
            theta, theta_0 = perceptron_single_step_update(
                feature_matrix[i], labels[i], theta, theta_0
            )
            theta_sum += theta
            theta_0_sum += theta_0
            updates += 1
    return theta_sum / updates, theta_0_sum / updates


def pegasos_single_step_update(
    feature_vector: Array,
    label: float,
    L: float,
    eta: float,
    current_theta: Array,
    current_theta_0: float,
) -> Tuple[Array, float]:
    theta = (1 - eta * L) * current_theta
    theta_0 = current_theta_0
    if label * (np.dot(current_theta, feature_vector) + current_theta_0) < 1:
        theta = theta + eta * label * feature_vector
        theta_0 = theta_0 + eta * label
    return theta, theta_0


def pegasos(feature_matrix: Array, labels: Array, T: int, L: float) -> Tuple[Array, float]:
    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0.0
    t = 0
    for _ in range(T):
        for i in get_order(feature_matrix.shape[0]):
            t += 1
            eta = 1.0 / np.sqrt(t)
            theta, theta_0 = pegasos_single_step_update(
                feature_matrix[i], labels[i], L, eta, theta, theta_0
            )
    return theta, theta_0


def predict(feature_matrix: Array, theta: Array, theta_0: float) -> Array:
    scores = feature_matrix @ theta + theta_0
    return np.where(scores >= 0, 1.0, -1.0)


def accuracy(feature_matrix: Array, labels: Array, theta: Array, theta_0: float) -> float:
    preds = predict(feature_matrix, theta, theta_0)
    return float(np.mean(preds == labels))


def classifier_accuracy(
    train_fn: Callable,
    train_x: Array,
    val_x: Array,
    train_y: Array,
    val_y: Array,
    **kwargs,
) -> Tuple[float, float]:
    theta, theta_0 = train_fn(train_x, train_y, **kwargs)
    return accuracy(train_x, train_y, theta, theta_0), accuracy(val_x, val_y, theta, theta_0)
