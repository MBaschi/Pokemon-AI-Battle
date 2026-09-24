"""Costruisce un dataset supervisionato dai replay reali della ladder.

Ogni step di un replay Kaggle contiene l'osservazione completa (compreso
`select.option`, cioe' le mosse legali) e l'azione che l'agente ha davvero
scelto. Sono quindi coppie (stato, mossa) allineabili al nostro encoding, da
cui si ricavano tre target per volta:

  - **policy**: one-hot sulla mossa dell'esperto. E' un segnale molto piu'
    netto della distribuzione di visite della MCTS, che con poche simulazioni
    resta quasi piatta;
  - **value**: l'esito vero della partita dal punto di vista di quel giocatore;
  - **phi**: calcolato dallo stato, come nel self-play.

Perche' funziona su teo_2 e non funzionerebbe su teo_1: questi replay coprono
decine di mazzi diversi. `teo_1` codifica le carte per card ID, quindi userebbe
solo i replay dei mazzi che gia' conosce. `teo_2` codifica per *attributi*
(cards.py), quindi li digerisce tutti.

Filtri di qualita' (i "maestri" contano): sulla ladder scaricata i migliori
agenti stanno al 62-66% di win rate, i peggiori al 43%. Clonare
indiscriminatamente insegnerebbe anche le mosse dei perdenti.

Esempi:
    python build_replay_dataset.py --min-winrate 0.58 --winner-only
    python build_replay_dataset.py --max-samples 300000 --out data/replays
"""

import argparse
import collections
import json
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np

import cgpath  # noqa: F401  -- deve precedere qualsiasi import di cg.*

from cards import ATTACK_TABLE, CARD_TABLE
from encoding import encode_actions, encode_state, enumerate_actions
from replay_data import ReplaySample
from reward import phi_vector

from cg.api import to_observation_class

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPLAYS = REPO_ROOT / "meta_analysis" / "ptcg_data" / "replays"
DEFAULT_DB = REPO_ROOT / "meta_analysis" / "ptcg_data" / "matches.db"


def agent_winrates(db_path):
    """Win rate per agente, usato come proxy di forza."""
    if not Path(db_path).exists():
        return {}
    con = sqlite3.connect(str(db_path))
    wins = collections.Counter()
    games = collections.Counter()
    for a0, a1, wi in con.execute("select agent0,agent1,winner_idx from matches"):
        if wi is None:
            continue
        games[a0] += 1
        games[a1] += 1
        if wi == 0:
            wins[a0] += 1
        elif wi == 1:
            wins[a1] += 1
    con.close()
    # Sotto una manciata di partite il win rate e' rumore, non forza.
    return {n: wins[n] / games[n] for n in games if games[n] >= 30}


def episode_decks(db_path):
    """{episode_id: {player_idx: [60 card id]}} da match_decks."""
    if not Path(db_path).exists():
        return {}
    con = sqlite3.connect(str(db_path))
    out = {}
    q = "select episode_id, player_idx, card_id, copies from match_decks"
    for ep, pi, cid, copies in con.execute(q):
        out.setdefault(ep, {}).setdefault(pi, []).extend([cid] * max(1, copies))
    con.close()
    return out


def extract(path, decks_by_ep, winrates, args):
    """Estrae i sample da un replay. Ritorna (lista_sample, statistiche)."""
    stats = collections.Counter()
    try:
        ep = json.loads(Path(path).read_text())
    except Exception:
        stats["file_illeggibile"] += 1
        return [], stats

    rewards = ep.get("rewards") or []
    names = (ep.get("info") or {}).get("TeamNames") or []
    # `rewards` puo' contenere None quando un agente e' andato in errore o in
    # timeout: la partita non ha un esito utilizzabile come target di value.
    # E' un caso raro (0 su 400 replay campionati) quindi si scarta, invece di
    # inventare un vincitore.
    if len(rewards) != 2 or any(r is None for r in rewards):
        stats["senza_esito"] += 1
        return [], stats
    if rewards[0] == rewards[1]:
        winner = None            # patta: nessun segnale di value affidabile
    else:
        winner = 0 if rewards[0] > rewards[1] else 1

    # Filtro di qualita': tiene solo i giocatori abbastanza forti.
    keep_slot = [True, True]
    if args.min_winrate > 0 and winrates:
        for i in (0, 1):
            nm = names[i] if i < len(names) else None
            keep_slot[i] = winrates.get(nm, 0.0) >= args.min_winrate
    if args.winner_only and winner is not None:
        for i in (0, 1):
            keep_slot[i] = keep_slot[i] and (i == winner)
    if not any(keep_slot):
        stats["partita_scartata"] += 1
        return [], stats

    ep_id = (ep.get("info") or {}).get("EpisodeId")
    decks = decks_by_ep.get(ep_id, {})

    out = []
    for step in ep.get("steps", []):
        for ag in step:
            if ag.get("status") != "ACTIVE":
                continue
            action = ag.get("action")
            obsd = ag.get("observation")
            if not action or not obsd or not obsd.get("select"):
                continue
            try:
                o = to_observation_class(obsd)
            except Exception:
                stats["obs_non_parsabile"] += 1
                continue
            if o.current is None or o.select is None:
                continue
            slot = o.current.yourIndex
            if slot is None or slot > 1 or not keep_slot[slot]:
                continue

            try:
                acts = enumerate_actions(o.select)
            except Exception:
                stats["enumerazione_fallita"] += 1
                continue
            if len(acts) <= 1:
                stats["decisione_banale"] += 1
                continue

            # Allineamento: la mossa registrata deve corrispondere a una delle
            # azioni che enumeriamo. ~11% non corrisponde (azioni piu' lunghe
            # di maxCount o indici fuori range: agenti che restituiscono liste
            # non validate). Si scartano, non si indovinano.
            want = sorted(action)
            target = None
            for i, a in enumerate(acts):
                if sorted(a) == want:
                    target = i
                    break
            if target is None:
                stats["azione_non_allineata"] += 1
                continue

            if winner is None:
                value = 0.0
            else:
                value = 1.0 if slot == winner else -1.0

            try:
                deck = decks.get(slot, [])
                st = encode_state(o, deck)
                ae = encode_actions(o, acts)
                phi_t = np.array(
                    phi_vector(o.current, slot, CARD_TABLE, ATTACK_TABLE),
                    dtype=np.float32,
                )
            except Exception:
                stats["encoding_fallito"] += 1
                continue

            policy = np.zeros(len(acts), dtype=np.float32)
            policy[target] = 1.0
            out.append(ReplaySample(st, ae, policy, value, phi_t))
            stats["ok"] += 1

    return out, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replays", default=str(DEFAULT_REPLAYS))
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--out", default="data/replays")
    ap.add_argument("--min-winrate", type=float, default=0.55,
                    help="tiene solo le mosse di agenti almeno cosi' forti "
                         "(0 = nessun filtro). Sulla ladder scaricata i migliori "
                         "stanno a 0.62-0.66, i peggiori a 0.43")
    ap.add_argument("--winner-only", action="store_true",
                    help="tiene solo le mosse di chi ha vinto la partita")
    ap.add_argument("--max-samples", type=int, default=400000)
    ap.add_argument("--shard-size", type=int, default=50000,
                    help="sample per file: il pretrain carica uno shard per volta")
    ap.add_argument("--max-files", type=int, default=0, help="0 = tutti (smoke test)")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.replays).glob("*.json"))
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        print(f"nessun replay in {args.replays}", file=sys.stderr)
        return 1

    print(f"replay: {len(files)} | filtro win rate >= {args.min_winrate} "
          f"| solo vincitori: {args.winner_only}")
    winrates = agent_winrates(args.db)
    print(f"agenti con win rate noto: {len(winrates)}")
    decks_by_ep = episode_decks(args.db)
    print(f"mazzi noti: {len(decks_by_ep)} partite")

    totals = collections.Counter()
    shard = []
    shard_i = 0
    written = 0

    for n, f in enumerate(files, 1):
        # Isolamento per file: un replay malformato non deve poter buttare via
        # i 15 minuti di lavoro gia' fatti sugli altri 8000. `extract` gestisce
        # gia' i casi noti, questo copre quelli che ancora non conosciamo.
        try:
            samples, stats = extract(f, decks_by_ep, winrates, args)
        except BaseException as exc:
            totals["replay_scartato_per_errore"] += 1
            if totals["replay_scartato_per_errore"] <= 5:
                print(f"\n  [!] {Path(f).name}: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
            continue
        totals.update(stats)
        shard.extend(samples)
        while len(shard) >= args.shard_size:
            path = outdir / f"shard_{shard_i:04d}.pkl"
            with open(path, "wb") as fh:
                pickle.dump(shard[: args.shard_size], fh, protocol=4)
            written += args.shard_size
            shard = shard[args.shard_size:]
            shard_i += 1
            sys.stderr.write(f"\r  {n}/{len(files)} replay -> {written} sample   ")
            sys.stderr.flush()
        if written >= args.max_samples:
            break
        if n % 50 == 0:
            sys.stderr.write(f"\r  {n}/{len(files)} replay -> {written + len(shard)} sample   ")
            sys.stderr.flush()

    if shard and written < args.max_samples:
        with open(outdir / f"shard_{shard_i:04d}.pkl", "wb") as fh:
            pickle.dump(shard, fh, protocol=4)
        written += len(shard)
    sys.stderr.write("\n")

    print(f"\nscritti {written} sample in {outdir} ({shard_i + 1} shard)")
    print("dettaglio:")
    for k, v in totals.most_common():
        print(f"  {k:<24} {v}")
    kept = totals["ok"]
    seen = kept + totals["azione_non_allineata"]
    if seen:
        print(f"tasso di allineamento: {100 * kept / seen:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
