"""Allena il classificatore binario "chi vincera'?" sugli stati campionati.

Punti chiave (vedi README):
  * split **per partita** (group split): mai lo stesso game in train e test;
  * metriche **per fase di gioco** (early/mid/late): il late-game e' facile,
    l'early-game e' il vero test;
  * **ablation senza prize**: mostra quanto il modello dipende dal conteggio
    prize vs dalle logiche di board;
  * **pesi della logistica** ordinati = le "logiche" apprese (interpretabilita').

Uso:
    python train_classifier.py --data data/state_samples --latent 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import agents.teo_1.simple_binary_classification.encoding as enc
import agents.teo_1.simple_binary_classification.model as M

HERE = Path(__file__).resolve().parent


def group_split(group_code, test_frac=0.25, seed=0):
    """Split per gruppo (partita): tutti i sample di una partita stanno o in
    train o in test. Con le due prospettive per stato, ogni partita porta sia
    vittorie che sconfitte, quindi entrambe le classi sono sempre presenti."""
    rng = np.random.RandomState(seed)
    groups = np.unique(group_code)
    rng.shuffle(groups)
    n_test = max(1, int(len(groups) * test_frac))
    test_g = set(groups[:n_test].tolist())
    test_mask = np.array([g in test_g for g in group_code])
    return ~test_mask, test_mask


def phase_of(turn_frac):
    return np.where(
        turn_frac < 1 / 3, "early", np.where(turn_frac < 2 / 3, "mid", "late")
    )


def evaluate(tag, y, p, turn_frac):
    print(f"\n[{tag}]  n={len(y)}")
    print(
        f"  accuracy={M.accuracy(y, p):.3f}  auc={M.auc(y, p):.3f}  "
        f"logloss={M.logloss(y, p):.3f}  brier={M.brier(y, p):.3f}"
    )
    ph = phase_of(turn_frac)
    for name in ("early", "mid", "late"):
        m = ph == name
        if m.sum():
            print(
                f"    fase {name:5s}: acc={M.accuracy(y[m], p[m]):.3f}  "
                f"auc={M.auc(y[m], p[m]):.3f}  (n={int(m.sum())})"
            )


def maybe_sklearn_gb(Xtr, ytr):
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except Exception:
        return None
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_depth=None, l2_regularization=1.0
    )
    clf.fit(Xtr, ytr)
    return clf


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", type=Path, default=HERE / "data" / "state_samples")
    ap.add_argument(
        "--latent", type=int, default=16, help="dimensioni latenti del blocco carte"
    )
    ap.add_argument(
        "--no-cards", action="store_true", help="usa solo le feature scalari"
    )
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE / "data" / "win_model.npz")
    a = ap.parse_args()

    d = np.load(a.data.with_suffix(".npz"))
    meta = json.loads(a.data.with_suffix(".meta.json").read_text(encoding="utf-8"))
    X_scalar, multihot = d["X_scalar"], d["multihot"]
    y, turn_frac, group_code = d["y"], d["turn_frac"], d["group_code"]
    names = meta["scalar_names"]
    print(
        f"Dataset: {len(y)} sample | {X_scalar.shape[1]} scalari | "
        f"vocab {multihot.shape[1]} | info={meta['info_mode']} | "
        f"partite {len(np.unique(group_code))}"
    )

    tr, te = group_split(group_code, a.test_frac, a.seed)
    print(
        f"Train: {tr.sum()} sample / {len(np.unique(group_code[tr]))} partite | "
        f"Test: {te.sum()} / {len(np.unique(group_code[te]))}"
    )

    # fit di scaler + encoder SOLO sul train
    std = enc.Standardizer().fit(X_scalar[tr])
    encoder = None
    if not a.no_cards:
        encoder = enc.PCAEncoder(a.latent).fit(multihot[tr])
        ev = encoder.explained_variance_ratio_.sum()
        print(
            f"Encoder carte (PCA lineare): {a.latent} latenti, varianza spiegata {ev:.1%}"
        )

    Xtr, feat_names = enc.assemble(
        X_scalar[tr],
        multihot[tr],
        std,
        encoder,
        use_cards=not a.no_cards,
        scalar_names=names,
    )
    Xte, _ = enc.assemble(
        X_scalar[te],
        multihot[te],
        std,
        encoder,
        use_cards=not a.no_cards,
        scalar_names=names,
    )

    # --- modello principale: regressione logistica interpretabile ---
    lr = M.LogisticRegression().fit(Xtr, y[tr])
    evaluate("Logistica — test", y[te], lr.predict_proba(Xte), turn_frac[te])

    # --- interpretabilita': i pesi = le "logiche" apprese ---
    w = lr.w
    order = np.argsort(w)
    print("\nLogiche apprese (peso della logistica, standardizzato):")
    print("  ↑ favoriscono la VITTORIA:")
    for i in order[::-1][:8]:
        print(f"    {w[i]:+.2f}  {feat_names[i]}")
    print("  ↓ favoriscono la SCONFITTA:")
    for i in order[:8]:
        print(f"    {w[i]:+.2f}  {feat_names[i]}")

    # --- ablation: senza il conteggio prize ---
    prize_cols = [n for n in names if "prizes_remaining" in n]
    std2 = enc.Standardizer().fit(_drop(X_scalar[tr], names, prize_cols))
    Xtr2, _ = enc.assemble(
        X_scalar[tr],
        multihot[tr],
        std2,
        encoder,
        use_cards=not a.no_cards,
        drop_cols=prize_cols,
        scalar_names=names,
    )
    Xte2, _ = enc.assemble(
        X_scalar[te],
        multihot[te],
        std2,
        encoder,
        use_cards=not a.no_cards,
        drop_cols=prize_cols,
        scalar_names=names,
    )
    lr2 = M.LogisticRegression().fit(Xtr2, y[tr])
    evaluate(
        "Logistica SENZA prize — test", y[te], lr2.predict_proba(Xte2), turn_frac[te]
    )

    # --- opzionale: gradient boosting (se sklearn e' installato) ---
    gb = maybe_sklearn_gb(Xtr, y[tr])
    if gb is not None:
        evaluate(
            "HistGradientBoosting — test",
            y[te],
            gb.predict_proba(Xte)[:, 1],
            turn_frac[te],
        )
    else:
        print(
            "\n[info] scikit-learn non installato: salto il gradient boosting "
            "(pip install scikit-learn per abilitarlo)."
        )

    # --- calibrazione del modello principale ---
    print("\nCalibrazione (prob prevista -> freq reale):")
    for pm, fr, n in M.calibration_table(y[te], lr.predict_proba(Xte)):
        print(f"    prev {pm:.2f} -> reale {fr:.2f}  (n={n})")

    # --- salvataggio del bundle per il riuso da un bot ---
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        a.out,
        w=lr.w,
        b=lr.b,
        feat_names=np.array(feat_names, dtype=object),
        scal_mean=std.mean_,
        scal_std=std.std_,
        scalar_names=np.array(names, dtype=object),
        pca_mean=(encoder.mean_ if encoder else np.array([])),
        pca_comp=(encoder.components_ if encoder else np.array([])),
        card_vocab=np.array(meta["card_vocab"], dtype=np.int64),
        info_mode=meta["info_mode"],
    )
    print(f"\nModello salvato in {a.out}")


def _drop(X, names, drop_cols):
    keep = [i for i, n in enumerate(names) if n not in set(drop_cols)]
    return X[:, keep]


if __name__ == "__main__":
    main()
