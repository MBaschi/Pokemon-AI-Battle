"""Costruisce un dataset di stati *reali* per allenare il ramo value di
``encoding.MyModel`` (lo stesso modello di teo_1) via apprendimento
supervisionato sull'esito effettivo della partita, invece che via self-play
MCTS (agents/teo_1/main.py).

A differenza di agents/state_evaluator/ (feature ingegnerizzate ad alto
livello), qui riusiamo l'encoding a livello di carta di teo_1
(``get_encoder_input``), passando dallo stesso ``to_observation_class`` che il
bot usa durante una partita reale: stessa rappresentazione, dataset più
grande, potenzialmente più espressivo.

Da dove vengono i dati:
  * board state per-step: meta_analysis/ptcg_data/replays/<episode_id>.json,
    ``steps[i][player]["observation"]`` (``current`` + ``select``) — è la
    stessa forma di ``obs_dict`` che il bot riceve dal vivo dall'engine.
  * decklist e vincitore: meta_analysis/ptcg_data/matches.db
    (``match_decks``, ``matches``/``rewards`` del replay) — il replay JSON da
    solo non contiene la decklist completa di ciascun giocatore (solo
    hand/deckCount), quindi serve il join col DB per riempire
    ``your_deck`` come richiesto da ``get_encoder_input``.

Richiede il vero package ``cg/`` alla root del repository (vedi README.md):
senza, ``to_observation_class``/``get_encoder_input`` non sono disponibili.

Uso:
    python build_replay_dataset.py --max-games 20   # smoke test veloce
    python build_replay_dataset.py --per-game 12    # dataset completo
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sqlite3
from pathlib import Path

from encoding import get_encoder_input
from cg.api import to_observation_class

HERE = Path(__file__).resolve().parent
DEFAULT_REPLAYS = HERE.parents[1] / "meta_analysis" / "ptcg_data" / "replays"
DEFAULT_DB = HERE.parents[1] / "meta_analysis" / "ptcg_data" / "matches.db"


def _episode_id(path: Path) -> str:
    """Ricava l'id partita da nomi tipo '12345.json'."""
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return digits or stem


def load_decks(db_path: Path) -> dict[str, dict[int, list[int]]]:
    """episode_id -> {player_idx: [card_id, ...]} (60 carte, esplose per copies)."""
    con = sqlite3.connect(str(db_path))
    decks: dict[str, dict[int, list[int]]] = {}
    for episode_id, player_idx, card_id, copies in con.execute(
        "SELECT episode_id, player_idx, card_id, copies FROM match_decks"
    ):
        deck = decks.setdefault(episode_id, {}).setdefault(player_idx, [])
        deck.extend([card_id] * copies)
    con.close()
    return decks


def game_outcome(doc: dict) -> tuple[int | None, bool]:
    """(indice vincitore o None se pareggio, True se l'esito e' valido)."""
    rewards = doc.get("rewards") or []
    if len(rewards) < 2 or rewards[0] is None or rewards[1] is None:
        return None, False
    if rewards[0] == rewards[1]:
        return None, True  # pareggio
    return (0 if rewards[0] > rewards[1] else 1), True


def decision_points(doc: dict, min_turn: int):
    """(turn, player_idx, observation_dict) per ogni osservazione reale.

    Teniamo l'osservazione di ENTRAMBI i giocatori quando disponibile (non
    solo di chi ha agito in quello step): ciascuna e' una prospettiva
    "realistic" valida e autoconsistente sullo stesso stato di gioco, il che
    ci da' gratis la simmetria A/B che agents/state_evaluator usa.
    """
    for step in doc.get("steps") or []:
        for player_idx, entry in enumerate(step or []):
            obs = (entry or {}).get("observation")
            if not obs or not obs.get("current") or not obs.get("select"):
                continue
            turn = obs["current"].get("turn")
            if not isinstance(turn, int) or turn < min_turn:
                continue
            yield turn, player_idx, obs


def _stratified_sample(points: list, per_game: int, rng: random.Random) -> list:
    """Campiona `per_game` punti distribuiti su tutta la durata della partita."""
    n = len(points)
    if n <= per_game:
        return points
    edges = [n * i // per_game for i in range(per_game + 1)]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        b = max(b, a + 1)
        out.append(points[rng.randrange(a, min(b, n))])
    return out


def build(
    replays_dir: Path,
    db_path: Path,
    out: Path,
    per_game: int,
    min_turn: int,
    max_games: int | None,
    seed: int,
):
    decks_by_episode = load_decks(db_path)
    rng = random.Random(seed)
    files = sorted(replays_dir.glob("*.json"))
    if max_games:
        files = files[:max_games]
    if not files:
        raise SystemExit(f"Nessun replay in {replays_dir}")

    samples: list[dict] = []
    n_ok = n_skip = 0
    for fp in files:
        episode_id = _episode_id(fp)
        decks = decks_by_episode.get(episode_id)
        if not decks or 0 not in decks or 1 not in decks:
            n_skip += 1
            continue
        try:
            doc = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            n_skip += 1
            continue

        winner, valid = game_outcome(doc)
        if not valid:
            n_skip += 1
            continue

        points = list(decision_points(doc, min_turn))
        if not points:
            n_skip += 1
            continue
        max_turn = max(t for t, _, _ in points) or 1

        for turn, player_idx, obs_dict in _stratified_sample(points, per_game, rng):
            try:
                obs = to_observation_class(obs_dict)
                sv = get_encoder_input(obs, decks[player_idx])
            except Exception:
                continue
            label = 0.0 if winner is None else (1.0 if player_idx == winner else -1.0)
            samples.append(
                {
                    "index": sv.index,
                    "value": sv.value,
                    "offset": sv.offset,
                    "label": label,
                    "episode_id": episode_id,
                    "turn_frac": turn / max_turn,
                }
            )
        n_ok += 1

    if not samples:
        raise SystemExit("Nessun sample prodotto.")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(
            {"samples": samples, "per_game": per_game, "min_turn": min_turn, "seed": seed},
            f,
        )

    n_wins = sum(1 for s in samples if s["label"] > 0)
    n_draws = sum(1 for s in samples if s["label"] == 0)
    print(f"Partite usate: {n_ok} | saltate: {n_skip}")
    print(f"Sample: {len(samples)}")
    print(
        f"Bilanciamento label: vittorie={n_wins/len(samples):.3f} "
        f"pareggi={n_draws/len(samples):.3f}"
    )
    print(f"Scritto: {out}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--replays-dir", type=Path, default=DEFAULT_REPLAYS)
    ap.add_argument("--matches-db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--out", type=Path, default=HERE / "data" / "replay_value_samples.pkl")
    ap.add_argument(
        "--per-game", type=int, default=12, help="punti di decisione campionati per partita"
    )
    ap.add_argument(
        "--min-turn", type=int, default=2, help="salta i primi turni di setup"
    )
    ap.add_argument(
        "--max-games", type=int, default=None, help="cap sul numero di replay (smoke test)"
    )
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    build(a.replays_dir, a.matches_db, a.out, a.per_game, a.min_turn, a.max_games, a.seed)


if __name__ == "__main__":
    main()
