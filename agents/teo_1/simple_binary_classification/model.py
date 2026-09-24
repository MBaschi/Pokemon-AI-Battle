"""Modelli e metriche in puro numpy (nessuna dipendenza esterna).

`LogisticRegression` e' volutamente semplice e *interpretabile*: i suoi pesi,
letti sui nomi delle feature, sono le "logiche" apprese dal modello. Se hai
scikit-learn installato, `train_classifier.py` puo' usare in aggiunta un
HistGradientBoosting piu' potente.
"""
from __future__ import annotations

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class LogisticRegression:
    """Regressione logistica con L2, discesa del gradiente full-batch."""

    def __init__(self, l2=1e-3, lr=0.5, epochs=400, seed=0):
        self.l2, self.lr, self.epochs, self.seed = l2, lr, epochs, seed

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        n, d = X.shape
        self.w = rng.normal(0, 0.01, d)
        self.b = 0.0
        y = y.astype(np.float64)
        for _ in range(self.epochs):
            p = sigmoid(X @ self.w + self.b)
            g = p - y
            gw = X.T @ g / n + self.l2 * self.w
            gb = g.mean()
            self.w -= self.lr * gw
            self.b -= self.lr * gb
        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)


# --------------------------------------------------------------------------- #
# Metriche (numpy)
# --------------------------------------------------------------------------- #
def accuracy(y, p):
    return float(((p >= 0.5).astype(int) == y).mean())


def logloss(y, p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(y, p):
    return float(((p - y) ** 2).mean())


def auc(y, p):
    """AUC via rank di Mann-Whitney (robusta ai pareggi di score)."""
    y = np.asarray(y)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    # media dei rank per gli score identici
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def calibration_table(y, p, bins=10):
    """Ritorna liste (p_medio_previsto, freq_reale, n) per `bins` fasce."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum():
            rows.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return rows
