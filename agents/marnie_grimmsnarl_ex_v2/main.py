"""marnie_grimmsnarl_ex_v2 -- euristica tarata + value net sui soli casi ambigui.

Entry point della submission. Contratto:

  - `obs["select"] is None`  -> selezione mazzo: ritorna i 60 card ID;
  - altrimenti               -> indici nelle opzioni di `select`.

Come si compongono i due strati (dettagli in search.py):

  heuristic.py  decide *sempre*. E' l'agente v1, con i suoi pesi tarati.
  search.py     entra in gioco solo quando l'euristica giudica due o piu'
                mosse equivalenti, e sceglie tra quelle completando il turno e
                valutando lo stato di confine con la rete.

Degradazione graduale, dalla piu' forte alla piu' debole. Ogni livello e' un
agente completo, non un'emergenza:

  1. checkpoint presente        -> ricerca con la value net;
  2. checkpoint assente         -> stessa ricerca, valore di foglia = Phi;
  3. torch assente / ricerca fallita / budget esaurito -> euristica pura (= v1);
  4. qualunque eccezione        -> prima risposta legale.

Il livello 3 e' il punto: **il pavimento di questo agente e' v1**. Un crash
perde la partita, una mossa mediocre costa poca equity, e la rete non ha modo
di fare peggio dell'euristica se non dentro la banda di ambivalenza.
"""

import os
import random
import sys
import time

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cgpath  # noqa: E402,F401  -- deve precedere qualsiasi import di cg.*

from cg.api import to_observation_class  # noqa: E402

import heuristic  # noqa: E402

DECK_LIST = list(heuristic.DECK_LIST)
my_deck = list(DECK_LIST)   # alias per benchmark_agents.py


def _find_file(name):
    """Kaggle esegue main.py via exec(), senza __file__ affidabile e senza
    chdir: si provano la cartella del modulo, la cwd e il path della submission."""
    dirs = [_HERE, os.getcwd()]
    try:
        import cg as _cg
        dirs.append(os.path.dirname(os.path.dirname(os.path.abspath(_cg.__file__))))
    except Exception:
        pass
    dirs.append("/kaggle_simulations/agent")
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------------------
# Modello e ricerca (entrambi opzionali)
# ---------------------------------------------------------------------------

_AGENT = None      # search.HybridAgent, o None se la ricerca non e' disponibile
_HAS_MODEL = False

# Ampiezza della banda di ambivalenza. E' il knob che regola *quanta* liberta'
# ha la rete: 0 = agente v1 esatto, valori grandi = la rete decide quasi tutto
# (e con una rete poco allenata si perde). Modificabile senza ri-deploy per i
# test A/B, che e' l'unico modo onesto di tararlo.
_REL_MARGIN = float(os.environ.get("MARNIE2_REL_MARGIN", "0.15"))
_ABS_MARGIN = float(os.environ.get("MARNIE2_ABS_MARGIN", "25"))
_MAX_CAND = int(os.environ.get("MARNIE2_MAX_CANDIDATES", "4"))

try:
    import torch

    from search import HybridAgent, SearchConfig, TurnSearch

    _DEVICE = torch.device("cpu")   # la rete e' piccola: la GPU non ripaga il transfer
    _model, _vocab = None, None
    _ckpt = os.environ.get("MARNIE2_CHECKPOINT") or _find_file("out/marnie_v2_latest.pth")
    if _ckpt and os.path.exists(_ckpt):
        import model as _model_mod

        _model, _vocab, _ = _model_mod.load(_ckpt, map_location=_DEVICE)
        _HAS_MODEL = True
    else:
        from cards import build_vocab

        _vocab = build_vocab(DECK_LIST)
        print(
            "marnie_v2: nessun checkpoint (MARNIE2_CHECKPOINT o out/marnie_v2_latest.pth). "
            "Ricerca con valutazione Phi pura: legale e sensata, ma non allenata.",
            file=sys.stderr,
        )

    _AGENT = HybridAgent(TurnSearch(
        model=_model,
        vocab=_vocab,
        device=_DEVICE,
        cfg=SearchConfig(
            rel_margin=_REL_MARGIN,
            abs_margin=_ABS_MARGIN,
            max_candidates=_MAX_CAND,
            # Senza rete il valore di foglia viene interamente da Phi.
            phi_weight=0.0 if _HAS_MODEL else 1.0,
            explore_eps=0.0,        # in partita non si esplora mai
        ),
        rng=random.Random(12345),   # seed fisso: l'agente deve essere riproducibile
        my_deck=DECK_LIST,
    ))
except BaseException as exc:  # pragma: no cover - dipende dall'ambiente
    print(f"marnie_v2: ricerca non disponibile ({exc}); gioco con la sola euristica.",
          file=sys.stderr)
    _AGENT = None


# ---------------------------------------------------------------------------
# Budget temporale: 10 minuti per giocatore per l'intera partita.
# ---------------------------------------------------------------------------

SOFT_BUDGET = 420.0        # margine ampio: il timeout e' sconfitta immediata
MIN_DECISION_TIME = 0.02
MAX_DECISION_TIME = 1.5
EXPECTED_DECISIONS = 400   # stima prudente delle decisioni ancora da prendere

_STATE = {"spent": 0.0, "decisions": 0}
_HEURISTIC_ONLY_OPPONENT = heuristic.OpponentModel()   # usato solo al livello 3


def _reset_if_new_match(state):
    """Lo scope di modulo persiste tra le chiamate ma non deve sopravvivere tra
    partite diverse: si azzera all'inizio del match."""
    global _HEURISTIC_ONLY_OPPONENT
    if state is not None and state.turn is not None and state.turn <= 1:
        if _STATE["decisions"] > 20:
            _STATE["spent"] = 0.0
            _STATE["decisions"] = 0
            _HEURISTIC_ONLY_OPPONENT = heuristic.OpponentModel()
            if _AGENT is not None:
                _AGENT.reset()


def _decision_budget():
    remaining = max(0.0, SOFT_BUDGET - _STATE["spent"])
    left = max(1, EXPECTED_DECISIONS - _STATE["decisions"])
    return max(MIN_DECISION_TIME, min(MAX_DECISION_TIME, remaining / left))


def _fallback(obs_dict):
    try:
        sel = obs_dict.get("select")
        if sel is None:
            return list(DECK_LIST)
        n = min(sel.get("maxCount") or 0, len(sel.get("option") or []))
        return list(range(n)) if n > 0 else []
    except BaseException:
        return []


def _agent_impl(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return list(DECK_LIST)

    state, select = obs.current, obs.select
    _reset_if_new_match(state)

    if _AGENT is not None and _STATE["spent"] < SOFT_BUDGET:
        _AGENT.searcher.cfg.time_budget = _decision_budget()
        return _AGENT.select(obs)

    # Livello 3: euristica pura. Il modello d'avversario e' tenuto aggiornato
    # qui perche' in questo ramo non passa da HybridAgent.
    board = heuristic.make_board(obs, _HEURISTIC_ONLY_OPPONENT)
    _HEURISTIC_ONLY_OPPONENT.observe(board.them)
    return heuristic.decide(obs, board, select)


def agent(obs_dict: dict) -> list:
    t0 = time.perf_counter()
    try:
        return _agent_impl(obs_dict)
    except BaseException:
        return _fallback(obs_dict)
    finally:
        try:
            _STATE["spent"] += time.perf_counter() - t0
            _STATE["decisions"] += 1
        except BaseException:
            pass


# ---------------------------------------------------------------------------
# SELF_CHECK
# ---------------------------------------------------------------------------
# - Selezione mazzo (select is None) -> 60 card ID: primo ramo di _agent_impl;
#   DECK_LIST ha un assert di lunghezza all'import di heuristic.py.
# - Cardinalita': ogni ritorno passa da heuristic._sanitize() (taglia a
#   maxCount, riempie fino a minCount, scarta duplicati e indici fuori range).
# - Mai solleva: agent() avvolge tutto in try/except BaseException; anche
#   TurnSearch.choose() e heuristic.decide() hanno il proprio try/except e
#   degradano invece di propagare.
# - Budget: _STATE["spent"] accumula fra le chiamate; _decision_budget()
#   riparte il residuo sulle decisioni attese e limita ogni singola ricerca.
#   Oltre SOFT_BUDGET la ricerca viene saltata del tutto (resta l'euristica).
# - Scope per-partita: _reset_if_new_match() azzera tempo, OpponentModel e le
#   carte avversarie osservate dalla ricerca a turn <= 1.
# - Import: stdlib + cg.api + heuristic; torch e' opzionale e la sua assenza
#   degrada al livello 3 (euristica pura).
# - Determinismo: l'unico random e' la determinizzazione della ricerca, con
#   seed fisso; l'euristica non usa random e risolve i pari con sorted() stabile.
