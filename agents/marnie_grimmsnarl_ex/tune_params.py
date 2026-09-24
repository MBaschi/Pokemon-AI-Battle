"""Ottimizzazione black-box dei pesi di un agente euristico.

Riusabile su qualunque agente del repo senza modifiche: basta copiarlo nella
sua cartella. L'unico requisito e' che l'agente carichi i pesi da `params.json`
e dia la precedenza alla variabile d'ambiente `PTCG_PARAMS` (vedi
`_load_params()` in gharchomp_ex/main.py).

L'agente e' una *struttura* euristica con ~60 pesi liberi (PARAMS). La
struttura codifica la conoscenza di dominio (cosa conta), l'ottimizzatore trova
i numeri. E' lo stesso schema con cui gio_v1 arriva a ~900 Elo, ed e' molto
piu' efficiente in campioni di un RL puro: qui si stimano ~50 parametri, non
7.2 milioni.

Metodo: **evolution strategy (mu+lambda)** con perturbazione gaussiana e passo
adattivo. Scelta rispetto alle alternative:

  - gradiente: non disponibile, l'obiettivo e' una simulazione;
  - grid/coordinate search: troppo lenta oltre una decina di dimensioni;
  - CMA-ES: migliore in teoria, ma richiede una dipendenza in piu' e con un
    obiettivo cosi' rumoroso il guadagno sul mu+lambda e' modesto;
  - bayesian optimization: pensata per valutazioni *costose*, mentre qui una
    partita dura millisecondi e conviene fare tanti campioni grezzi.

Il punto delicato e' il **rumore**: la win rate su poche partite ha una banda
di +-15%, abbastanza da far vincere un candidato peggiore. Due contromisure:

  1. **confronto appaiato** contro un avversario fisso, con i lati alternati,
     invece di una win rate assoluta;
  2. **ri-valutazione dell'elite**: il campione migliore viene rigiocato a ogni
     generazione, cosi' un risultato fortunato non sopravvive a lungo.

Esempi:
    python tune_params.py --generations 30 --games 200 --workers 8
    python tune_params.py --opponent ../gio_v1/main.py --games 400
"""

import argparse
import json
import multiprocessing
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HERE = Path(__file__).resolve().parent

# Pesi che non hanno senso perturbare come numeri continui.
BOOLEAN_KEYS = {"prefer_go_first"}

# Chiavi escluse dall'ottimizzazione (nessuna, per ora: tenuto come punto di
# controllo se un peso si rivelasse instabile).
FROZEN_KEYS = set()


def load_base_params():
    with open(HERE / "params.json") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Valutazione: un candidato gioca N partite contro un avversario fisso
# ---------------------------------------------------------------------------

_W = {}


def _worker_init(agent_path, opponent_path):
    """Ogni worker carica i due agenti una volta sola.

    L'avversario si importa qui; l'agente sotto tuning va invece ricaricato per
    ogni candidato, perche' legge i suoi pesi dall'ambiente al momento
    dell'import (vedi _eval_candidate)."""
    import importlib.util

    from benchmark_agents import load_agent

    _W["load_agent"] = load_agent
    _W["agent_path"] = Path(agent_path)
    _W["opponent"] = load_agent(Path(opponent_path), "opponent")
    _W["importlib"] = importlib.util


def _play_match(a0, a1, deck0, deck1, max_turns=200):
    """Una partita. a0 occupa lo slot 0. Ritorna 0/1/2 (vincitore/patta)."""
    from cg.game import battle_finish, battle_select, battle_start

    obs, start = battle_start(deck0, deck1)
    if start.errorPlayer is not None and start.errorPlayer >= 0:
        battle_finish()
        raise ValueError(f"deck non valido (tipo {start.errorType})")
    try:
        while obs["current"]["result"] < 0:
            if obs["current"]["turn"] > max_turns:
                return 2
            slot = obs["current"]["yourIndex"]
            fn = a0 if slot == 0 else a1
            try:
                sel = fn(obs)
            except Exception:
                return 1 - slot        # un crash perde la partita
            try:
                obs = battle_select(sel)
            except (ValueError, IndexError):
                return 1 - slot        # mossa illegale: idem
        return obs["current"]["result"]
    finally:
        battle_finish()


def _eval_candidate(task):
    """Gioca `games` partite col candidato e ritorna (indice, vittorie, decise)."""
    idx, params, games, seed = task
    try:
        # I pesi passano dall'ambiente: due processi possono valutare candidati
        # diversi in parallelo senza scriversi addosso sullo stesso params.json.
        # Il nome della variabile e' generico apposta, cosi' questo file si
        # copia su un agente nuovo senza toccare una riga.
        os.environ["PTCG_PARAMS"] = json.dumps(params)
        cand = _W["load_agent"](_W["agent_path"], f"cand{idx}")
        opp = _W["opponent"]

        rng = random.Random(seed)
        wins = decided = 0
        for g in range(games):
            # Lati alternati: chi inizia ha un vantaggio sistematico, e senza
            # alternare si misurerebbe quello invece del candidato.
            if g % 2 == 0:
                res = _play_match(cand.fn, opp.fn, cand.deck, opp.deck)
                mine = 0
            else:
                res = _play_match(opp.fn, cand.fn, opp.deck, cand.deck)
                mine = 1
            if res == 2 or res < 0:
                continue
            decided += 1
            if res == mine:
                wins += 1
            rng.random()
        return idx, wins, decided
    except BaseException as exc:
        return idx, 0, 0 if not isinstance(exc, KeyboardInterrupt) else 0


# ---------------------------------------------------------------------------
# Evolution strategy
# ---------------------------------------------------------------------------

def perturb(params, sigma, rng, keys):
    """Un figlio: rumore gaussiano relativo su ogni peso."""
    child = dict(params)
    for k in keys:
        v = params[k]
        if k in BOOLEAN_KEYS:
            if rng.random() < 0.1:            # i flag si girano, non si scalano
                child[k] = 1 - int(v)
            continue
        scale = max(abs(v), 1.0) * sigma
        nv = v + rng.gauss(0.0, scale)
        # I pesi restano non negativi: sono priorita', un segno negativo
        # inverte il significato della regola invece di regolarla.
        child[k] = max(0.0, round(nv, 3))
    return child


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--opponent", default=str(REPO_ROOT / "agents" / "gio_v1" / "main.py"))
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--population", type=int, default=8, help="figli per generazione")
    ap.add_argument("--games", type=int, default=200, help="partite per candidato")
    ap.add_argument("--sigma", type=float, default=0.25, help="ampiezza iniziale del passo")
    ap.add_argument("--sigma-min", type=float, default=0.03)
    ap.add_argument("--workers", type=int, default=0, help="0 = automatico")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "params.json"))
    ap.add_argument("--log", default=str(HERE / "tune_log.txt"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    base = load_base_params()
    keys = [k for k in base if k not in FROZEN_KEYS]
    workers = args.workers if args.workers > 0 else max(1, min(8, (os.cpu_count() or 2) // 2))

    def log(msg):
        print(msg, flush=True)
        if args.log:
            try:
                with open(args.log, "a", encoding="utf-8") as fh:
                    fh.write(msg + "\n")
            except OSError:
                pass

    log(f"tuning di {len(keys)} pesi | avversario: {args.opponent}")
    log(f"{args.generations} generazioni x {args.population} figli x {args.games} partite "
        f"| {workers} worker")

    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(workers, initializer=_worker_init,
                    initargs=(str(HERE / "main.py"), args.opponent))

    best = dict(base)
    best_wr = None
    sigma = args.sigma
    t_start = time.time()

    try:
        for gen in range(args.generations):
            children = [perturb(best, sigma, rng, keys) for _ in range(args.population)]
            # L'elite rientra in gara ogni volta: cosi' un punteggio fortunato
            # deve essere confermato invece di restare campione per sempre.
            candidates = [best] + children

            tasks = [(i, c, args.games, args.seed * 1009 + gen * 31 + i)
                     for i, c in enumerate(candidates)]
            results = {}
            for idx, wins, decided in pool.imap_unordered(_eval_candidate, tasks):
                results[idx] = (wins, decided)

            scored = []
            for i, c in enumerate(candidates):
                w, d = results.get(i, (0, 0))
                scored.append((w / d if d else 0.0, d, i, c))
            scored.sort(key=lambda r: -r[0])
            top_wr, top_d, top_i, top_c = scored[0]

            improved = top_i != 0
            if improved:
                best, best_wr = top_c, top_wr
                sigma = min(args.sigma, sigma * 1.15)   # ha funzionato: passo piu' largo
            else:
                best_wr = top_wr
                sigma = max(args.sigma_min, sigma * 0.85)   # stringi e raffina

            with open(args.out, "w") as fh:
                json.dump(best, fh, indent=2, sort_keys=True)

            mark = "  <- migliorato" if improved else ""
            log(f"gen {gen:3d}: miglior wr={100 * top_wr:.1f}% su {top_d} partite "
                f"| sigma={sigma:.3f}{mark} [{time.time() - t_start:.0f}s]")
    except KeyboardInterrupt:
        log("interrotto: l'ultimo params.json salvato resta valido")
    finally:
        pool.terminate()
        pool.join()

    log(f"fatto. miglior win rate ~{100 * (best_wr or 0):.1f}% -> {args.out}")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
