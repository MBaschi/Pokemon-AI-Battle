"""Encoding dei sample: standardizzazione + compressione del "bag of cards".

Il vettore multihot delle carte visibili e' alto-dimensionale e sparso. Lo
comprimiamo in poche dimensioni latenti con un **autoencoder lineare** (PCA via
SVD): dipende solo da numpy e gira ovunque. Se hai `torch`, `TorchAutoencoder`
offre una versione non lineare piu' espressiva (opzionale).

Tutti gli oggetti si "fittano" SOLO sul train (media/std/componenti) e poi si
applicano a train e test: evita leakage.
"""
from __future__ import annotations

import numpy as np


class Standardizer:
    """(x - media) / std, colonne a varianza nulla lasciate invariate."""

    def fit(self, X):
        self.mean_ = X.mean(0)
        self.std_ = X.std(0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class PCAEncoder:
    """Autoencoder lineare: proietta il multihot su `n_components` (SVD)."""

    def __init__(self, n_components=16):
        self.n_components = n_components

    def fit(self, X):
        X = X.astype(np.float64)
        self.mean_ = X.mean(0)
        Xc = X - self.mean_
        k = min(self.n_components, min(Xc.shape) - 1) if min(Xc.shape) > 1 else 1
        # SVD economica; le componenti sono le direzioni principali (V)
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = Vt[:k]
        var = (S ** 2) / max(len(X) - 1, 1)
        self.explained_variance_ratio_ = (var[:k] / var.sum()) if var.sum() else np.zeros(k)
        return self

    def transform(self, X):
        return ((X.astype(np.float64) - self.mean_) @ self.components_.T).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class TorchAutoencoder:
    """AE non lineare (opzionale, richiede torch). Fallback esplicito se assente."""

    def __init__(self, n_components=16, hidden=64, epochs=40, lr=1e-3, seed=0):
        self.n_components, self.hidden = n_components, hidden
        self.epochs, self.lr, self.seed = epochs, lr, seed

    def fit(self, X):
        import torch
        from torch import nn
        torch.manual_seed(self.seed)
        d = X.shape[1]
        self.enc = nn.Sequential(nn.Linear(d, self.hidden), nn.ReLU(),
                                 nn.Linear(self.hidden, self.n_components))
        dec = nn.Sequential(nn.Linear(self.n_components, self.hidden), nn.ReLU(),
                            nn.Linear(self.hidden, d))
        self.dec = dec
        params = list(self.enc.parameters()) + list(dec.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)
        xt = torch.tensor(X, dtype=torch.float32)
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = ((dec(self.enc(xt)) - xt) ** 2).mean()
            loss.backward()
            opt.step()
        return self

    def transform(self, X):
        import torch
        with torch.no_grad():
            return self.enc(torch.tensor(X, dtype=torch.float32)).numpy().astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def assemble(X_scalar, multihot, scalar_std: Standardizer, encoder,
             use_cards=True, drop_cols=None, scalar_names=None):
    """Compone la matrice finale: [scalari standardizzati | latenti carte].

    `drop_cols` (lista di nomi) permette le ablation (es. togliere i prize).
    Ritorna (X, feature_names).
    """
    names = list(scalar_names) if scalar_names is not None else [f"s{i}" for i in range(X_scalar.shape[1])]
    Xs = X_scalar
    if drop_cols:
        keep = [i for i, n in enumerate(names) if n not in set(drop_cols)]
        Xs = Xs[:, keep]
        names = [names[i] for i in keep]
    Xs = scalar_std.transform(Xs)
    parts, out_names = [Xs], list(names)
    if use_cards and encoder is not None:
        Z = encoder.transform(multihot)
        parts.append(Z)
        out_names += [f"cardlatent_{i}" for i in range(Z.shape[1])]
    return np.concatenate(parts, axis=1).astype(np.float32), out_names
