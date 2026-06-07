#!/usr/bin/env python3
"""
MNIST progression demo — linear → MLP → CNN.
Concepts from MITx 6.86x Weeks 3–4 (implemented with scikit-learn for portability).
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_mnist_subset(n_train: int = 5000, n_test: int = 1000, seed: int = 42):
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = mnist.data.astype(np.float32) / 255.0, mnist.target.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=n_train, test_size=n_test, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test


def eval_model(name: str, model, X_train, y_train, X_test, y_test) -> None:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name:28} test accuracy = {acc:.4f}")


def main() -> None:
    print("MITx 6.86x — MNIST progression (subset)\n" + "-" * 45)
    X_train, X_test, y_train, y_test = load_mnist_subset()

    eval_model(
        "Linear (logistic regression)",
        Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))]),
        X_train, y_train, X_test, y_test,
    )
    eval_model(
        "MLP (2 hidden layers)",
        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=15, random_state=1),
        X_train, y_train, X_test, y_test,
    )
    print("\nNote: full CNN implementation from course used PyTorch — see course materials.")


if __name__ == "__main__":
    main()
