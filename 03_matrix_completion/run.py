#!/usr/bin/env python3
"""Matrix completion demo with Gaussian mixture EM."""

from pathlib import Path

import numpy as np

from em import complete_matrix, run_em

DATA = Path(__file__).parent / "data" / "toy_data.txt"


def main() -> None:
    X = np.loadtxt(DATA)
    observed = X != 0
    print("MITx 6.86x — Matrix completion (toy dataset)")
    print(f"Shape: {X.shape}  missing entries: {int((~observed).sum())}")

    mixture, post = run_em(X, k=2, max_iter=30)
    completed = complete_matrix(X, mixture, post)

    mse = np.mean((completed[observed] - X[observed]) ** 2)
    print(f"Reconstruction MSE (observed): {mse:.4f}")
    print("Completed matrix (rounded):")
    print(np.round(completed, 2))


if __name__ == "__main__":
    main()
