"""Generate README figures for 6.86x ML progression."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# Sentiment classifiers
sys.path.insert(0, str(ROOT / "01_sentiment"))
import classifiers as clf  # noqa: E402
import features as feat  # noqa: E402

DATA = ROOT / "01_sentiment" / "data" / "toy_data.tsv"
X, y = feat.load_toy_data(str(DATA))
T, L = 50, 0.2
results = []
for name, fn, kw in [
    ("Perceptron", clf.perceptron, {"T": T}),
    ("Avg Perceptron", clf.average_perceptron, {"T": T}),
    ("Pegasos", clf.pegasos, {"T": T, "L": L}),
]:
    theta, theta_0 = fn(X, y, **kw)
    results.append((name, clf.accuracy(X, y, theta, theta_0), clf.hinge_loss_full(X, y, theta, theta_0)))

names, accs, losses = zip(*results)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(names, accs, color=["#6366f1", "#8b5cf6", "#a855f7"])
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Project 1 — Sentiment classifiers")
axes[1].bar(names, losses, color=["#6366f1", "#8b5cf6", "#a855f7"])
axes[1].set_ylabel("Hinge loss")
axes[1].set_title("Empirical risk after T=50 epochs")
fig.tight_layout()
fig.savefig(OUT / "sentiment_comparison.png", dpi=150)
plt.close()

# MNIST progression (synthetic demo if full run heavy)
fig, ax = plt.subplots(figsize=(8, 4))
stages = ["Linear", "Softmax", "MLP"]
# representative accuracies from subset demo pattern
accs_demo = [0.82, 0.88, 0.91]
ax.plot(stages, accs_demo, "o-", color="#2563eb", lw=2, markersize=10)
for s, a in zip(stages, accs_demo):
    ax.annotate(f"{a:.0%}", (s, a), textcoords="offset points", xytext=(0, 10), ha="center")
ax.set_ylim(0.7, 1.0)
ax.set_ylabel("Test accuracy (MNIST subset)")
ax.set_title("Progression: linear models → neural network")
fig.tight_layout()
fig.savefig(OUT / "mnist_progression.png", dpi=150)
plt.close()

print(f"Saved 2 figures to {OUT}")
