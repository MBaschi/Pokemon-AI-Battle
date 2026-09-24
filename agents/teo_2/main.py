"""teo_2: agente live -- MCTS PUCT guidata dalla rete allenata in self-play.

Contratto di esecuzione (vedi SELF_CHECK in fondo):
  - `obs["select"] is None`  -> selezione mazzo: ritorna i 60 card ID;
  - altrimenti               -> indici nelle opzioni di `select`.

Degradazione graduale, dalla piu' forte alla piu' debole:
  1. checkpoint presente -> MCTS con value/policy della rete;
  2. checkpoint assente  -> stessa MCTS ma con valore di foglia = Phi
     (reward.py). L'agente resta giocabile anche senza allenamento: Phi da'
     una valutazione di stato sensata da sola;
  3. torch assente o ricerca fallita -> scelta greedy su Phi senza ricerca;
  4. qualunque eccezione -> prima opzione legale.

Nessun livello puo' sollevare: un crash perde la partita, una mossa mediocre
costa poca equity.
"""

import os
import random
import sys
import time

# ---------------------------------------------------------------------------
# Bootstrap dei path (Kaggle esegue main.py via exec(), senza __file__ affidabile)
# ---------------------------------------------------------------------------


def _find_file(name):
    dirs = []
    try:
        dirs.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    dirs.append(os.getcwd())
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


try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cgpath  # noqa: E402,F401  -- deve precedere qualsiasi import di cg.*

from cg.api import to_observation_class  # noqa: E402

from encoding import enumerate_actions  # noqa: E402

# ---------------------------------------------------------------------------
# Mazzo
# ---------------------------------------------------------------------------


def _read_deck():
    # TEO2_DECK permette di provare la stessa rete su liste diverse senza
    # toccare deck.csv (che e' il mazzo della submission). Serve a separare la
    # forza dell'agente da quella del mazzo quando si confrontano due agenti.
    path = os.environ.get("TEO2_DECK") or _find_file("deck.csv")
    if path is None:
        raise FileNotFoundError("deck.csv non trovato accanto a main.py")
    rows = open(path).read().split("\n")
    deck = [int(r) for r in rows if r.strip()][:60]
    if len(deck) != 60:
        raise ValueError(f"deck.csv ha {len(deck)} carte, ne servono 60")
    return deck


my_deck = _read_deck()

# ---------------------------------------------------------------------------
# Modello (opzionale)
# ---------------------------------------------------------------------------

_MODEL = None
_DEVICE = None
_HAS_MODEL = False

try:
    import torch

    from mcts import SearchConfig, run_mcts
    from model import build_model

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _MODEL = build_model().to(_DEVICE)
    _ckpt = os.environ.get("TEO2_CHECKPOINT") or _find_file("out/teo2_latest.pth")
    if _ckpt and os.path.exists(_ckpt):
        _MODEL.load_state_dict(torch.load(_ckpt, map_location=_DEVICE))
        _HAS_MODEL = True
    else:
        print(
            "teo_2: nessun checkpoint (TEO2_CHECKPOINT o out/teo2_latest.pth). "
            "Gioco con valutazione Phi pura: legale e ragionevole, ma non allenato.",
            file=sys.stderr,
        )
    _MODEL.eval()
    _TORCH_OK = True
except BaseException as exc:  # pragma: no cover - dipende dall'ambiente
    print(f"teo_2: torch/MCTS non disponibili ({exc}); uso il fallback Phi.", file=sys.stderr)
    _TORCH_OK = False

# ---------------------------------------------------------------------------
# Budget temporale: 10 minuti per giocatore per l'intera partita.
# ---------------------------------------------------------------------------

MATCH_BUDGET = 600.0
SOFT_BUDGET = 420.0        # margine ampio: il timeout e' sconfitta immediata
MIN_DECISION_TIME = 0.05
MAX_DECISION_TIME = 3.0
EXPECTED_DECISIONS = 400   # stima prudente di quante decisioni resta da prendere

_STATE = {"spent": 0.0, "decisions": 0}
_RNG = random.Random(12345)   # seed fisso: l'agente deve essere riproducibile


def _reset_if_new_match(state):
    """Lo scope di modulo persiste tra le chiamate ma non deve sopravvivere
    tra partite diverse: si azzera all'inizio del match."""
    if state is not None and state.turn is not None and state.turn <= 1:
        if _STATE["decisions"] > 20:
            _STATE["spent"] = 0.0
            _STATE["decisions"] = 0


def _decision_budget():
    remaining = max(0.0, SOFT_BUDGET - _STATE["spent"])
    left = max(1, EXPECTED_DECISIONS - _STATE["decisions"])
    return max(MIN_DECISION_TIME, min(MAX_DECISION_TIME, remaining / left))


# ---------------------------------------------------------------------------
# Fallback: greedy su Phi, senza ricerca
# ---------------------------------------------------------------------------


def _phi_greedy(obs, actions):
    """Sceglie senza simulare: preferisce le azioni che *tipicamente* alzano Phi.

    Non potendo simulare (e' il ramo usato quando la ricerca non e'
    disponibile), si usa un ordinamento statico per tipo di opzione derivato
    dalle stesse priorita' che Phi codifica: evolvere e attaccare alzano
    prize/evolution, attaccare energia alza attack_ready, finire il turno no.
    """
    from cg.api import OptionType

    order = {
        OptionType.ABILITY: 9,
        OptionType.EVOLVE: 8,
        OptionType.ATTACK: 7,
        OptionType.ATTACH: 6,
        OptionType.PLAY: 5,
        OptionType.CARD: 4,
        OptionType.YES: 3,
        OptionType.RETREAT: 2,
        OptionType.NO: 1,
        OptionType.END: 0,
    }
    best_i, best_score = 0, -1e9
    for i, action in enumerate(actions):
        if not action:
            score = 0.5     # declinare e' spesso corretto, ma non prioritario
        else:
            score = 0.0
            for oi in action:
                if oi < len(obs.select.option):
                    score += order.get(obs.select.option[oi].type, 1)
            score /= len(action)
        if score > best_score:
            best_score, best_i = score, i
    return actions[best_i]


def _sanitize(result, select):
    """Garantisce minCount <= len <= maxCount, indici unici e in range."""
    n = len(select.option)
    out = []
    for i in result:
        if isinstance(i, int) and 0 <= i < n and i not in out:
            out.append(i)
    if len(out) > select.maxCount:
        out = out[: select.maxCount]
    for i in range(n):
        if len(out) >= select.minCount:
            break
        if i not in out:
            out.append(i)
    return out


def _fallback(obs_dict):
    try:
        sel = obs_dict.get("select")
        if sel is None:
            return list(my_deck)
        maxc = sel.get("maxCount") or 0
        opts = sel.get("option") or []
        n = min(maxc, len(opts))
        return list(range(n)) if n > 0 else []
    except BaseException:
        return []


# ---------------------------------------------------------------------------
# Agente
# ---------------------------------------------------------------------------


def _agent_impl(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return list(my_deck)

    _reset_if_new_match(obs.current)
    select = obs.select

    actions = enumerate_actions(select)
    if len(actions) == 1:
        return _sanitize(actions[0], select)

    budget_left = SOFT_BUDGET - _STATE["spent"]
    if _TORCH_OK and budget_left > 1.0:
        cfg = SearchConfig(
            n_simulations=64 if _HAS_MODEL else 24,
            dirichlet_eps=0.0,        # nessuna esplorazione in partita vera
            temperature=0.0,          # deterministico
            # Senza checkpoint il valore di foglia viene interamente da Phi.
            phi_weight=0.0 if _HAS_MODEL else 1.0,
            time_budget=_decision_budget(),
        )
        try:
            with torch.inference_mode():
                action, _policy, _value = run_mcts(
                    obs, my_deck, _MODEL, _DEVICE, cfg, _RNG
                )
            return _sanitize(list(action), select)
        except BaseException:
            pass

    return _sanitize(_phi_greedy(obs, actions), select)


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
# - Selezione mazzo (select is None) -> 60 card ID: _agent_impl, primo ramo.
# - Cardinalita': _sanitize() applicata a ogni ritorno di _agent_impl; taglia a
#   maxCount, riempie fino a minCount, scarta duplicati e indici fuori range.
# - Mai solleva: agent() avvolge tutto in try/except BaseException; _fallback()
#   e' a sua volta protetto e degrada a [].
# - Budget: _STATE["spent"] accumula il tempo tra le chiamate; _decision_budget()
#   riparte il residuo sulle decisioni attese e viene passato a SearchConfig,
#   che ferma la MCTS alla scadenza.
# - Scope di modulo per-partita: _reset_if_new_match() azzera a turn <= 1.
# - Import: solo stdlib + cg.api + torch (fornito dall'ambiente); l'assenza di
#   torch e' gestita e degrada al ramo Phi.
# - Determinismo: _RNG ha seed fisso, temperature=0 e dirichlet_eps=0 in gioco.
