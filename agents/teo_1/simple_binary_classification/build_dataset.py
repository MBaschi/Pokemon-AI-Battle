"""Costruisce un dataset di *stati di gioco etichettati* dai replay.

Per ogni partita (replay JSON):
  1. determina il vincitore dai `rewards`;
  2. prende uno stato per turno (board di fine turno), scartando la fase di setup;
  3. ne campiona `--per-game` stratificati sull'avanzamento della partita
     (early / mid / late), cosi' il modello vede tutte le fasi;
  4. per ogni stato campionato emette DUE sample — la prospettiva di ciascun
     giocatore — con label = "questa prospettiva ha vinto?" (simmetria A/B).

Output (due file affiancati):
  * ``<out>.npz``      : array numerici (X_scalar, multihot, y, turn_frac, group)
  * ``<out>.meta.json``: nomi feature, vocabolario carte, id partite, config

Uso:
    python build_dataset.py --replays-dir ../../meta_analysis/ptcg_data/replays \
        --per-game 6 --vocab 300 --out data/state_samples
    python build_dataset.py --max-games 400   # smoke test veloce
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

import agents.teo_1.simple_binary_classification.state_features as sf

HERE = Path(__file__).resolve().parent
DEFAULT_REPLAYS = HERE.parents[1] / "meta_analysis" / "ptcg_data" / "replays"


def _episode_id(path: Path) -> str:
    """Ricava l'id partita da nomi tipo '12345.json' o 'episode-12345-replay.json'."""
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return digits or stem


def _stratified_sample(states, per_game, rng):
    """Campiona `per_game` stati distribuiti su tutta la durata della partita."""
    n = len(states)
    if n <= per_game:
        return states
    # bucket uniformi sull'indice (proxy dell'avanzamento) → un pescaggio per bucket
    edges = np.linspace(0, n, per_game + 1).astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        b = max(b, a + 1)
        out.append(states[rng.randrange(a, min(b, n))])
    return out


def build(
    replays_dir: Path,
    out: Path,
    per_game: int,
    vocab_size: int,
    info_mode: str,
    min_turn: int,
    max_games: int | None,
    seed: int,
):
    meta_cards = sf.card_meta()
    rng = random.Random(seed)
    files = sorted(replays_dir.glob("*.json"))
    if max_games:
        files = files[:max_games]
    if not files:
        raise SystemExit(f"Nessun replay in {replays_dir}")

    scalar_names = sf.scalar_feature_names()
    rows_scalar: list[list[float]] = []
    rows_bag: list[Counter] = []
    ys: list[int] = []
    turn_fracs: list[float] = []
    groups: list[str] = []
    card_freq: Counter = Counter()

    n_ok = n_skip = 0
    for fp in files:
        try:
            doc = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            n_skip += 1
            continue
        winner = sf.winner_of(doc)
        if winner is None:
            n_skip += 1
            continue
        turns = sf.states_by_turn(doc)
        turns = [(t, c) for t, c in turns if t >= min_turn]
        if not turns:
            n_skip += 1
            continue
        max_turn = turns[-1][0] or 1
        for t, cur in _stratified_sample(turns, per_game, rng):
            for me in (0, 1):
                try:
                    feats, bag = sf.state_to_features(cur, me, meta_cards, info_mode)
                except Exception:
                    continue
                rows_scalar.append([float(feats[k]) for k in scalar_names])
                rows_bag.append(bag)
                ys.append(int(winner == me))
                turn_fracs.append(t / max_turn)
                groups.append(_episode_id(fp))
                card_freq.update(bag.keys())
        n_ok += 1

    if not rows_scalar:
        raise SystemExit("Nessun sample prodotto.")

    # vocabolario = carte piu' frequenti (limita la dimensione del multihot)
    vocab = [cid for cid, _ in card_freq.most_common(vocab_size)]
    vindex = {cid: j for j, cid in enumerate(vocab)}
    multihot = np.zeros((len(rows_bag), len(vocab)), dtype=np.uint8)
    for i, bag in enumerate(rows_bag):
        for cid, cnt in bag.items():
            j = vindex.get(cid)
            if j is not None:
                multihot[i, j] = min(cnt, 255)

    X_scalar = np.asarray(rows_scalar, dtype=np.float32)
    y = np.asarray(ys, dtype=np.uint8)
    turn_frac = np.asarray(turn_fracs, dtype=np.float32)
    uniq = {g: k for k, g in enumerate(dict.fromkeys(groups))}
    group_code = np.asarray([uniq[g] for g in groups], dtype=np.int32)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out.with_suffix(".npz"),
        X_scalar=X_scalar,
        multihot=multihot,
        y=y,
        turn_frac=turn_frac,
        group_code=group_code,
    )
    meta = {
        "scalar_names": scalar_names,
        "card_vocab": vocab,
        "card_vocab_names": [meta_cards.get(c, {}).get("name", str(c)) for c in vocab],
        "groups": list(uniq.keys()),
        "info_mode": info_mode,
        "per_game": per_game,
        "min_turn": min_turn,
        "n_games_used": n_ok,
        "n_games_skipped": n_skip,
        "n_samples": int(y.shape[0]),
        "seed": seed,
    }
    out.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"Partite usate: {n_ok} | saltate: {n_skip}")
    print(
        f"Sample: {y.shape[0]} | scalari: {X_scalar.shape[1]} | vocab carte: {len(vocab)}"
    )
    print(f"Bilanciamento label (vittorie): {y.mean():.3f}")
    print(f"Scritti: {out.with_suffix('.npz')} + {out.with_suffix('.meta.json')}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--replays-dir", type=Path, default=DEFAULT_REPLAYS)
    ap.add_argument("--out", type=Path, default=HERE / "data" / "state_samples")
    ap.add_argument(
        "--per-game", type=int, default=6, help="stati campionati per partita"
    )
    ap.add_argument(
        "--vocab", type=int, default=300, help="dimensione vocabolario carte (multihot)"
    )
    ap.add_argument("--info-mode", choices=["realistic", "oracle"], default="realistic")
    ap.add_argument(
        "--min-turn", type=int, default=2, help="salta i primi turni di setup"
    )
    ap.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="cap sul numero di replay (smoke test)",
    )
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    build(
        a.replays_dir,
        a.out,
        a.per_game,
        a.vocab,
        a.info_mode,
        a.min_turn,
        a.max_games,
        a.seed,
    )


if __name__ == "__main__":
    main()
