"""Gioca una partita con teo_2 e la esporta come replay HTML *annotato*.

E' l'equivalente della schermata di analisi di un motore di scacchi:

  - una **curva di valutazione** (win probability del giocatore 0) sotto la
    plancia, cliccabile per saltare alla mossa;
  - per ogni decisione, le **mosse candidate** con il valore che la MCTS ha
    assegnato loro, le visite e il prior della rete;
  - una riga di debug per passo (valore alla radice, Φ euristico, quanto
    e' cambiata la valutazione dopo la mossa).

Il viewer di `view_replays/replay_render.py` sa gia' disegnare tutto questo:
legge tre campi opzionali per ogni step del replay (`eval_p0`, `agent_scores`,
`debug_out`). Qui li produciamo giocando la partita in locale sul motore `cg` e
prendendo gli step dal motore stesso (`cg.game.visualize_data`), invece che da
un replay scaricato da Kaggle.

Attenzione a cosa *significa* la curva: la valutazione e' quella della value
head di teo_2, cioe' del modello che stiamo giudicando. Una discesa dice "la
rete pensa di stare peggio", non "la mossa era oggettivamente cattiva". Come
guardare l'eval di Stockfish per giudicare Stockfish: utile per trovare i punti
interessanti della partita, inutile come verita' assoluta.

Esempi:
    # teo_2 (checkpoint corrente) contro l'euristica gharchomp_ex
    python replay_export.py --opponent ../gharchomp_ex/main.py

    # due generazioni a confronto, tre partite
    python replay_export.py --opponent self \\
        --checkpoint out/teo2_iter000.pth --opponent-checkpoint out/teo2_latest.pth \\
        --games 3 --sims 32
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

import cgpath  # noqa: F401  -- deve precedere qualsiasi import di cg.*

from cg.api import OptionType, SelectContext, to_observation_class
from cg.game import battle_finish, battle_select, battle_start, visualize_data

from cards import ATTACK_TABLE, CARD_TABLE
from encoding import _source_and_target, encode_actions, encode_state, enumerate_actions
from mcts import SearchConfig, run_mcts
from model import build_model, evaluate
from reward import phi as phi_fn

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def _replay_render():
    """Importa il renderer condiviso in `view_replays/` (non e' un package)."""
    path = str(REPO_ROOT / "view_replays")
    if path not in sys.path:
        sys.path.insert(0, path)
    import replay_render

    return replay_render


# ---------------------------------------------------------------------------
# Etichette leggibili per le opzioni
# ---------------------------------------------------------------------------

_CONTEXT_IT = {
    SelectContext.MAIN: "Turno",
    SelectContext.SETUP_ACTIVE_POKEMON: "Setup attivo",
    SelectContext.SETUP_BENCH_POKEMON: "Setup panchina",
    SelectContext.SWITCH: "Cambio",
    SelectContext.TO_ACTIVE: "In posizione attiva",
    SelectContext.TO_BENCH: "In panchina",
    SelectContext.TO_HAND: "In mano",
    SelectContext.DISCARD: "Scarta",
    SelectContext.ATTACK: "Attacco",
    SelectContext.EVOLVE: "Evoluzione",
    SelectContext.IS_FIRST: "Chi inizia",
    SelectContext.MULLIGAN: "Mulligan",
    SelectContext.ACTIVATE: "Attivare l'effetto?",
    SelectContext.COIN_HEAD: "Testa o croce",
}


def context_label(select):
    if select is None:
        return ""
    try:
        return _CONTEXT_IT.get(select.context, SelectContext(select.context).name)
    except ValueError:
        return str(select.context)


def _card_name(card):
    if card is None:
        return "?"
    data = CARD_TABLE.get(card.id)
    return data.name if data is not None else f"carta {card.id}"


def option_label(obs, option):
    """Descrizione in italiano di una singola opzione di `select`."""
    t = option.type
    me = obs.current.yourIndex
    # `_source_and_target` e' la stessa risoluzione carta/bersaglio usata
    # dall'encoding: riusarla evita che etichetta e feature raccontino due
    # storie diverse sulla stessa opzione.
    src, tgt = _source_and_target(obs, option, me)

    if t == OptionType.END:
        return "Fine turno"
    if t == OptionType.RETREAT:
        return f"Ritirata ({_card_name(src)})" if src else "Ritirata"
    if t == OptionType.ATTACK:
        atk = ATTACK_TABLE.get(option.attackId)
        if atk is not None:
            dmg = f" {atk.damage}" if atk.damage else ""
            return f"Attacco: {atk.name}{dmg}"
        return "Attacco"
    if t == OptionType.PLAY:
        return f"Gioca {_card_name(src)}"
    if t == OptionType.EVOLVE:
        return f"Evolvi {_card_name(tgt)} → {_card_name(src)}"
    if t == OptionType.ATTACH:
        return f"Attacca {_card_name(src)} a {_card_name(tgt)}"
    if t == OptionType.ABILITY:
        return f"Abilità di {_card_name(src)}"
    if t == OptionType.DISCARD:
        return f"Scarta {_card_name(src)}"
    if t == OptionType.YES:
        return "Sì"
    if t == OptionType.NO:
        return "No"
    if t == OptionType.NUMBER:
        return f"Numero {option.number}"
    if t in (OptionType.ENERGY, OptionType.ENERGY_CARD):
        return f"Energia su {_card_name(tgt)}" if tgt else "Energia"
    if t == OptionType.TOOL_CARD:
        return f"Strumento {_card_name(src)}"
    if t == OptionType.SPECIAL_CONDITION:
        return f"Condizione {option.specialConditionType}"
    if src is not None:
        return _card_name(src)
    try:
        return OptionType(t).name.capitalize()
    except ValueError:
        return str(t)


def action_label(obs, action, max_parts=3):
    """Descrizione di un'azione = insieme di indici opzione."""
    if not action:
        return "— non fare nulla —"
    options = [obs.select.option[i] for i in action if i < len(obs.select.option)]
    if not options:
        return "— nessuna opzione —"
    parts = [option_label(obs, o) for o in options[:max_parts]]
    if len(options) > max_parts:
        parts.append(f"+{len(options) - max_parts}")
    return " + ".join(parts)


# ---------------------------------------------------------------------------
# Giocatori
# ---------------------------------------------------------------------------

def load_model(checkpoint=None, device=None):
    """Costruisce TeoNet e ci carica sopra un checkpoint, se c'e'."""
    import torch

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    if checkpoint:
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint non trovato: {path}")
        blob = torch.load(path, map_location=device)
        # `teo2_resume.pth` incapsula i pesi insieme all'optimizer.
        model.load_state_dict(blob["model"] if isinstance(blob, dict) and "model" in blob
                              else blob)
    model.eval()
    return model, device


def read_deck(path):
    rows = Path(path).read_text().split("\n")
    deck = [int(r) for r in rows if r.strip()][:60]
    if len(deck) != 60:
        raise ValueError(f"{path}: il mazzo ha {len(deck)} carte, ne servono 60")
    return deck


class Teo2Player:
    """teo_2 pilotato direttamente dalla MCTS, per avere le statistiche interne.

    Non passa da `main.py` di proposito: l'agente di produzione restituisce solo
    la mossa, mentre qui serve tutta la radice della ricerca.
    """

    def __init__(self, deck, model, device, sims=64, c_puct=1.4, seed=0,
                 name="teo_2", max_actions=64):
        self.deck = deck
        self.model = model
        self.device = device
        self.name = name
        self.rng = random.Random(seed)
        self.cfg = SearchConfig(
            n_simulations=sims,
            c_puct=c_puct,
            dirichlet_eps=0.0,     # analisi: niente rumore di esplorazione
            temperature=0.0,       # deterministico, come in partita vera
            phi_weight=0.0,
            max_actions=max_actions,
        )
        self.opponent_seen = []    # carte avversarie viste: migliora la determinizzazione

    def decide(self, obs_dict, obs):
        import torch

        stats = {}
        with torch.inference_mode():
            action, _policy, _root = run_mcts(
                obs, self.deck, self.model, self.device, self.cfg, self.rng,
                opponent_known=self.opponent_seen, stats_out=stats,
            )
        return list(action), stats

    def observe(self, obs, my_slot):
        """Registra le carte avversarie diventate visibili (campo + scarti).

        `my_slot` e' il posto di *questo* giocatore, non di chi muove: la stessa
        osservazione viene mostrata a entrambi, e chi e' l'avversario dipende da
        chi guarda.
        """
        opp = obs.current.players[1 - my_slot]
        for c in opp.discard:
            self.opponent_seen.append(c.id)
        for p in [x for x in opp.active if x] + [x for x in opp.bench if x]:
            self.opponent_seen.append(p.id)
        if len(self.opponent_seen) > 120:
            del self.opponent_seen[:-120]


class AgentPlayer:
    """Un qualsiasi altro agente del repo, caricato dal suo `main.py`."""

    def __init__(self, main_py, name=None, deck=None):
        sys.path.insert(0, str(REPO_ROOT))
        from benchmark_agents import load_agent

        loaded = load_agent(Path(main_py), name)
        self.fn = loaded.fn
        self.name = name or loaded.name
        self.deck = deck or loaded.deck

    def decide(self, obs_dict, obs):
        return list(self.fn(obs_dict)), None

    def observe(self, obs, my_slot):
        pass


class RandomPlayer:
    """Avversario casuale: il pavimento, utile per vedere partite lunghe in fretta."""

    def __init__(self, deck, seed=0, name="random"):
        self.deck = deck
        self.rng = random.Random(seed)
        self.name = name

    def decide(self, obs_dict, obs):
        actions = enumerate_actions(obs.select)
        return list(self.rng.choice(actions)), None

    def observe(self, obs, my_slot):
        pass


# ---------------------------------------------------------------------------
# Valutazione di stato (la "eval" del motore)
# ---------------------------------------------------------------------------

class NetEvaluator:
    """Valore della value head per il giocatore di turno, in [-1, 1].

    Viene chiamato a *ogni* decisione, anche quando muove l'avversario: e' cosi'
    che la curva diventa continua e leggibile come quella di un motore di
    scacchi, invece di avere buchi a turni alterni.
    """

    def __init__(self, model, device, max_actions=16):
        self.model = model
        self.device = device
        self.max_actions = max_actions

    def value(self, obs, deck):
        import torch

        if obs.select is None:
            return None
        actions = enumerate_actions(obs.select, self.max_actions)
        with torch.inference_mode():
            v, _phi, _priors = evaluate(
                self.model, encode_state(obs, deck), encode_actions(obs, actions),
                self.device,
            )
        return float(v)


def _to_p0(value, mover):
    """Da valore in [-1,1] per chi muove a probabilita' di vittoria del giocatore 0."""
    p_mover = (value + 1.0) / 2.0
    return p_mover if mover == 0 else 1.0 - p_mover


# ---------------------------------------------------------------------------
# Partita
# ---------------------------------------------------------------------------

class Match:
    """Una partita giocata in locale, con gli step del viewer e le annotazioni."""

    def __init__(self, steps, decisions, result, players, seed, aborted=False,
                 crashed=None):
        self.steps = steps            # list[dict] nel formato del viewer
        self.decisions = decisions    # list[dict], una per decisione
        self.result = result          # 0/1 vincitore, 2 patta, -1/-2 non conclusa
        self.players = players        # (nome0, nome1)
        self.seed = seed
        self.aborted = aborted
        self.crashed = crashed

    @property
    def winner(self):
        if self.result in (0, 1):
            return self.players[self.result]
        if self.result == 2:
            return "patta"
        return "non conclusa"

    def summary(self):
        turns = self.steps[-1]["current"]["turn"] if self.steps else 0
        return (f"{self.players[0]} vs {self.players[1]} — {self.winner} "
                f"({len(self.decisions)} decisioni, {turns} turni, seed {self.seed})")

    def to_dataframe(self):
        """Una riga per decisione: e' la tabella su cui si cercano gli errori."""
        import pandas as pd

        rows = []
        for d in self.decisions:
            top = d["options"][:3]
            rows.append({
                "step": d["step"],
                "turno": d["turn"],
                "giocatore": d["player"],
                "contesto": d["context"],
                "n_azioni": d["n_actions"],
                "scelta": d["chosen_label"],
                # Due letture della stessa valutazione: sempre dal punto di
                # vista del giocatore 0 (la curva), e dal punto di vista di chi
                # sta muovendo (la decisione).
                "win%_p0": None if d["eval_p0"] is None else round(100 * d["eval_p0"], 1),
                "win%_di_chi_muove": (None if d["p_mover"] is None
                                      else round(100 * d["p_mover"], 1)),
                "equity_persa": (None if d["equity_loss"] is None
                                 else round(100 * d["equity_loss"], 1)),
                "phi": round(d["phi"], 3),
                "sim": d["n_simulations"],
                "top1": top[0]["label"] if len(top) > 0 else None,
                "top1_win%": top[0]["score"] if len(top) > 0 else None,
                "top2": top[1]["label"] if len(top) > 1 else None,
                "top2_win%": top[1]["score"] if len(top) > 1 else None,
                "top3": top[2]["label"] if len(top) > 2 else None,
                "top3_win%": top[2]["score"] if len(top) > 2 else None,
            })
        return pd.DataFrame(rows)

    def blunders(self, player=None, top=10, min_actions=2):
        """Le decisioni dopo cui la valutazione e' scesa di piu' per chi muoveva.

        E' l'analogo della centipawn loss, con lo stesso caveat di un'analisi a
        bassa profondita': dentro c'e' anche la fortuna (pescate, monetine) e
        l'errore di stima della rete, non solo l'errore di gioco.
        """
        df = self.to_dataframe()
        if df.empty:
            return df
        df = df[df["n_azioni"] >= min_actions]
        if player is not None:
            df = df[df["giocatore"] == player]
        return df.sort_values("equity_persa", ascending=False).head(top)


def _option_rows(obs, stats, chosen_action):
    """Righe "mossa candidata → valore" per il pannello del viewer.

    Il punteggio e' la probabilita' di vittoria *di chi muove*, in percentuale:
    e' l'unita' piu' leggibile qui, e il pannello del viewer arrotonda i
    punteggi a interi (un Q in [-1,1] diventerebbe tutto 0).
    """
    # Decisione obbligata (una sola azione legale): non c'e' nessuna scelta da
    # mostrare, e riempire il pannello di righe da 0 punti nasconde le decisioni
    # vere sotto il rumore.
    if stats.get("source") == "forzata":
        return [], 0

    rows = []
    searched = [a for a in stats.get("actions", [])
                if a["visits"] > 0 and a["q"] is not None]
    pool = searched or stats.get("actions", [])
    use_q = bool(searched)

    for a in pool:
        label = action_label(obs, a["action"])
        prior = a["prior"]
        detail = []
        if use_q:
            detail.append(f"N={a['visits']}")
        if prior is not None:
            detail.append(f"P={100 * prior:.0f}%")
        score = (100.0 * (a["q"] + 1.0) / 2.0) if use_q else 100.0 * (prior or 0.0)
        rows.append({
            "label": f"{label}  [{' '.join(detail)}]" if detail else label,
            "score": round(score, 1),
            "selected": a["action"] == list(chosen_action),
            "visits": a["visits"],
            "prior": prior,
            "q": a["q"],
        })
    rows.sort(key=lambda r: (r["visits"], r["score"]), reverse=True)
    return rows, len(stats.get("actions", [])) - len(searched)


def play_match(player0, player1, seed=0, max_turns=300, evaluator=None, echo=True):
    """Gioca una partita e ritorna un `Match` con gli step gia' annotati."""
    players = (player0, player1)
    decks = (player0.deck, player1.deck)

    obs, start = battle_start(decks[0], decks[1])
    if start.errorPlayer is not None and start.errorPlayer >= 0:
        battle_finish()
        raise ValueError(f"mazzo non valido per il giocatore {start.errorPlayer} "
                         f"(tipo {start.errorType})")

    decisions = []
    result = -1
    aborted = False
    crashed = None
    t0 = time.perf_counter()

    try:
        while obs["current"]["result"] < 0:
            if obs["current"]["turn"] > max_turns:
                aborted = True
                break
            slot = obs["current"]["yourIndex"]
            o = to_observation_class(obs)
            step_index = len(decisions)

            value = evaluator.value(o, decks[slot]) if evaluator else None
            phi_val = phi_fn(o.current, slot, CARD_TABLE, ATTACK_TABLE)

            try:
                action, stats = players[slot].decide(obs, o)
            except BaseException as exc:      # un agente che esplode perde
                crashed = players[slot].name
                result = 1 - slot
                if echo:
                    print(f"  {players[slot].name} ha sollevato: {exc}", file=sys.stderr)
                break

            actions_here = enumerate_actions(o.select)
            rows, unexplored = (_option_rows(o, stats, action) if stats else ([], 0))
            decisions.append({
                "step": step_index,
                "turn": o.current.turn,
                "player": slot,
                "player_name": players[slot].name,
                "context": context_label(o.select),
                "n_actions": len(actions_here),
                "chosen": list(action),
                "chosen_label": action_label(o, action),
                "value": value,
                "p_mover": None if value is None else (value + 1.0) / 2.0,
                "eval_p0": None if value is None else _to_p0(value, slot),
                "root_value": None if not stats else stats.get("root_value"),
                "n_simulations": 0 if not stats else stats.get("n_simulations", 0),
                "source": None if not stats else stats.get("source"),
                "phi": phi_val,
                "options": rows,
                "unexplored": unexplored,
                "equity_loss": None,          # riempito dopo, serve il passo dopo
            })

            for s, p in enumerate(players):
                p.observe(o, s)

            try:
                obs = battle_select(list(action))
            except (ValueError, IndexError) as exc:
                crashed = players[slot].name
                result = 1 - slot
                if echo:
                    print(f"  mossa rifiutata dal motore ({players[slot].name}): {exc}",
                          file=sys.stderr)
                break

            if echo and step_index % 25 == 0:
                sys.stderr.write(f"\r  decisione {step_index} (turno {o.current.turn})   ")
                sys.stderr.flush()

        if not aborted and crashed is None:
            result = obs["current"]["result"]
        elif aborted:
            result = -2
        steps = json.loads(visualize_data())
    finally:
        battle_finish()
    if echo:
        sys.stderr.write("\n")

    _finalize(steps, decisions, result)
    match = Match(steps, decisions, result,
                  (player0.name, player1.name), seed, aborted, crashed)
    if echo:
        print(f"  {match.summary()} [{time.perf_counter() - t0:.0f}s]")
    return match


def _finalize(steps, decisions, result):
    """Attacca le annotazioni agli step del viewer.

    Allineamento: la decisione i-esima si prende sullo stato descritto da
    `steps[i]` (verificato sul motore: `visualize_data()` emette una entry per
    ogni `battle_select`, piu' una per lo stato finale).
    """
    # Equity persa: quanto scende la valutazione *di chi ha appena mosso* fra
    # questa decisione e la successiva. Include la risposta avversaria e la
    # fortuna, esattamente come la centipawn loss di un'analisi veloce.
    for i, d in enumerate(decisions):
        if d["eval_p0"] is None:
            continue
        if i + 1 < len(decisions) and decisions[i + 1]["eval_p0"] is not None:
            after_p0 = decisions[i + 1]["eval_p0"]
        elif result in (0, 1, 2):
            after_p0 = 1.0 if result == 0 else 0.0 if result == 1 else 0.5
        else:
            continue
        after_mover = after_p0 if d["player"] == 0 else 1.0 - after_p0
        d["equity_loss"] = d["p_mover"] - after_mover

    for d in decisions:
        d["debug"] = _debug_line(d)
        i = d["step"]
        if i >= len(steps):
            continue
        step = steps[i]
        if d["eval_p0"] is not None:
            step["eval_p0"] = round(d["eval_p0"], 4)
        if d["options"]:
            step["agent_scores"] = {
                "player": d["player"],
                "context": f"{d['context']} · {d['player_name']}",
                "options": [{"label": r["label"], "score": r["score"],
                             "selected": r["selected"]}
                            for r in d["options"][:10]],
            }
        step["debug_out"] = d["debug"]

    # Ultimo step: la partita e' decisa, la valutazione e' l'esito vero.
    if steps and result in (0, 1, 2):
        steps[-1]["eval_p0"] = 1.0 if result == 0 else 0.0 if result == 1 else 0.5


def _debug_line(d):
    bits = []
    if d["p_mover"] is not None:
        bits.append(f"rete: {100 * d['p_mover']:.1f}% per P{d['player'] + 1}")
    if d["root_value"] is not None and d["source"] == "mcts":
        bits.append(f"radice MCTS {d['root_value']:+.3f} su {d['n_simulations']} sim")
    bits.append(f"Φ={d['phi']:+.3f}")
    bits.append(f"{d['n_actions']} azioni legali")
    if d["unexplored"]:
        bits.append(f"{d['unexplored']} mai esplorate")
    if d["equity_loss"] is not None:
        bits.append(f"Δ dopo la mossa {-100 * d['equity_loss']:+.1f} pt")
    bits.append(f"scelta: {d['chosen_label']}")
    return " · ".join(bits)


# ---------------------------------------------------------------------------
# Misure che le curve di loss non danno
# ---------------------------------------------------------------------------

def arena(player0, player1, games=10, max_turns=300, seed=0, echo=True):
    """Win rate diretto fra due configurazioni (checkpoint, sims, avversari).

    E' l'unica misura di *forza*: loss e KL dicono quanto bene la rete imita la
    propria ricerca, non se il risultato vince partite. I lati si alternano a
    ogni partita, perche' iniziare per primi in questo gioco pesa.
    """
    wins = {player0.name: 0, player1.name: 0}
    draws = other = 0
    for g in range(games):
        a, b = (player0, player1) if g % 2 == 0 else (player1, player0)
        match = play_match(a, b, seed=seed + g, max_turns=max_turns,
                           evaluator=None, echo=False)
        if match.result in (0, 1):
            wins[match.players[match.result]] += 1
        elif match.result == 2:
            draws += 1
        else:
            other += 1
        if echo:
            print(f"  partita {g + 1}/{games}: {match.winner}")
    decisive = sum(wins.values())
    return {
        "wins": wins,
        "draws": draws,
        "non_concluse": other,
        "win_rate": {k: (v / decisive if decisive else 0.0) for k, v in wins.items()},
        "games": games,
    }


def calibration(matches, bins=10, player=None):
    """Il value head e' *calibrato*? Predizione vs esito, a bin.

    Per ogni decisione si prende la probabilita' di vittoria che la rete
    assegnava a chi muoveva e la si confronta con l'esito vero della partita. Se
    la rete e' onesta, nel bin "70%" si vince circa il 70% delle volte. E' la
    diagnosi che la loss non da': una value head puo' avere loss bassa ed essere
    sistematicamente troppo sicura di se'.
    """
    import pandas as pd

    rows = []
    for m in matches:
        if m.result not in (0, 1, 2):
            continue        # partite non concluse: nessun esito da confrontare
        for d in m.decisions:
            if d["p_mover"] is None:
                continue
            if player is not None and d["player"] != player:
                continue
            if m.result == 2:
                outcome = 0.5
            else:
                outcome = 1.0 if m.result == d["player"] else 0.0
            rows.append({"pred": d["p_mover"], "outcome": outcome,
                         "turn": d["turn"], "player": d["player"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    edges = np.linspace(0, 1, bins + 1)
    df["bin"] = pd.cut(df["pred"], edges, include_lowest=True)
    out = df.groupby("bin", observed=True).agg(
        n=("pred", "size"), predetto=("pred", "mean"), reale=("outcome", "mean")
    ).reset_index()
    out["errore"] = out["predetto"] - out["reale"]
    return out


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_html(match, out_path):
    """Scrive il replay HTML autonomo (card data incluse, ~3 MB)."""
    rr = _replay_render()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rr.generate_html(match.steps, str(out_path), list(match.players))
    return out_path


def show(match, height=900):
    """Mostra il replay annotato dentro il notebook."""
    rr = _replay_render()
    return rr.show_replay(match.steps, height=height, player_names=list(match.players))


def default_out_name(match, index=0):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe = lambda s: "".join(c if c.isalnum() or c in "-_" else "_" for c in s)  # noqa: E731
    return f"{stamp}_{safe(match.players[0])}_vs_{safe(match.players[1])}_g{index}.html"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_players(args):
    """Costruisce i due giocatori e il valutatore condiviso.

    Il valutatore usa *sempre* il modello del lato 0: la curva deve raccontare
    la partita dal punto di vista di un solo cervello, altrimenti un salto nella
    curva potrebbe essere solo il cambio di chi la sta stimando.
    """
    deck0 = read_deck(args.deck or HERE / "deck.csv")
    model, device = load_model(args.checkpoint, None)
    if args.name:
        name0 = args.name
    elif args.checkpoint:
        name0 = f"teo_2 ({Path(args.checkpoint).stem})"
    else:
        name0 = "teo_2 (non allenato)"
    p0 = Teo2Player(deck0, model, device, sims=args.sims, seed=args.seed, name=name0)

    if args.opponent == "self":
        deck1 = read_deck(args.opponent_deck) if args.opponent_deck else list(deck0)
        if args.opponent_checkpoint:
            model1, device1 = load_model(args.opponent_checkpoint, None)
            name1 = f"teo_2 ({Path(args.opponent_checkpoint).stem})"
        else:
            model1, device1, name1 = model, device, "teo_2 (stesso checkpoint)"
        p1 = Teo2Player(deck1, model1, device1, sims=args.sims, seed=args.seed + 1,
                        name=name1)
    elif args.opponent == "random":
        deck1 = read_deck(args.opponent_deck) if args.opponent_deck else list(deck0)
        p1 = RandomPlayer(deck1, seed=args.seed + 1)
    else:
        deck1 = read_deck(args.opponent_deck) if args.opponent_deck else None
        p1 = AgentPlayer(args.opponent, deck=deck1)

    return p0, p1, NetEvaluator(model, device)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", default=str(HERE / "out" / "teo2_latest.pth"),
                    help="pesi del lato 0 (vuoto = rete non allenata)")
    ap.add_argument("--opponent", default="self",
                    help="'self', 'random', oppure il path al main.py di un altro agente")
    ap.add_argument("--opponent-checkpoint", default=None,
                    help="con --opponent self: pesi diversi per il lato 1 "
                         "(confronto fra generazioni)")
    ap.add_argument("--deck", default=None, help="deck.csv del lato 0")
    ap.add_argument("--opponent-deck", default=None, help="deck.csv del lato 1")
    ap.add_argument("--name", default=None, help="nome da mostrare per il lato 0")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--sims", type=int, default=64, help="simulazioni MCTS per decisione")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed degli *agenti*. Il mescolamento dei mazzi lo decide "
                         "il motore e non e' controllabile da qui: due partite con "
                         "lo stesso seed non sono identiche")
    ap.add_argument("--max-turns", type=int, default=300)
    ap.add_argument("--out", default=str(HERE / "replays"),
                    help="cartella dove scrivere gli HTML")
    ap.add_argument("--no-html", action="store_true",
                    help="solo statistiche a schermo, senza scrivere il replay")
    args = ap.parse_args()

    p0, p1, evaluator = build_players(args)
    print(f"{p0.name}  vs  {p1.name}  ({args.games} partite, {args.sims} sim/decisione)")

    for g in range(args.games):
        match = play_match(p0, p1, seed=args.seed + g, max_turns=args.max_turns,
                           evaluator=evaluator)
        losses = [d["equity_loss"] for d in match.decisions
                  if d["equity_loss"] is not None and d["player"] == 0
                  and d["n_actions"] > 1]
        if losses:
            print(f"  equity persa da {p0.name}: media {100 * np.mean(losses):+.2f} pt, "
                  f"peggiore {100 * max(losses):+.1f} pt")
        if not args.no_html:
            out = Path(args.out) / default_out_name(match, g)
            export_html(match, out)
            print(f"  replay -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
