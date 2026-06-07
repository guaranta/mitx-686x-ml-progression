#!/usr/bin/env python3
"""Run sentiment classifiers on MITx 6.86x toy data."""

from pathlib import Path

import classifiers as clf
import features as feat

DATA = Path(__file__).parent / "data" / "toy_data.tsv"


def main() -> None:
    X, y = feat.load_toy_data(str(DATA))
    T, L = 50, 0.2

    models = {
        "Perceptron": (clf.perceptron, {"T": T}),
        "Average Perceptron": (clf.average_perceptron, {"T": T}),
        "Pegasos": (clf.pegasos, {"T": T, "L": L}),
    }

    print("MITx 6.86x — Sentiment (toy dataset)\n" + "-" * 40)
    for name, (fn, kwargs) in models.items():
        theta, theta_0 = fn(X, y, **kwargs)
        acc = clf.accuracy(X, y, theta, theta_0)
        loss = clf.hinge_loss_full(X, y, theta, theta_0)
        print(f"{name:22} accuracy={acc:.3f}  hinge_loss={loss:.4f}")


if __name__ == "__main__":
    main()
