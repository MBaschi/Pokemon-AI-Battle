"""MCTS PUCT con determinizzazione per teo_2.

Differenze rispetto alla MCTS di teo_1:

  - PUCT vero (AlphaZero): Q + c_puct * P * sqrt(N_padre) / (1 + N_figlio),
    con i prior P presi dalla policy della rete. teo_1 usa una formula affine
    ma con Q preso dal *padre* per i figli non visitati, che sottostima
    sistematicamente le mosse mai provate.
  - Rumore di Dirichlet alla radice: senza, il self-play collassa sulla mossa
    che la rete gia' preferisce e non esplora mai nient'altro -- e' la
    ragione singola piu' comune per cui un training self-play non decolla.
  - Selezione per temperatura: stocastica all'inizio della partita (esplora),
    deterministica dopo (sfrutta).
  - Valore della foglia miscelato con Phi: a inizio training la rete e'
    rumore puro, e Phi da' un segnale sensato da subito. Il peso decade a
    zero mano a mano che la rete diventa affidabile.
  - Budget temporale esplicito: la ricerca si ferma prima di sforare.

Informazione nascosta: `search_begin` pretende che gli si passi una
*ipotesi* completa su mazzi, prize e mano avversaria. Ogni ricerca lavora
quindi su una singola determinizzazione. E' un'approssimazione nota
(strategy fusion), mitigata usando le carte gia' osservate dell'avversario
invece di riempire tutto con un filler come fa teo_1.
"""

import math
import random
import time

import numpy as np

import cgpath  # noqa: F401  -- deve precedere qualsiasi import di cg.*

from cg.api import CardType, search_begin, search_end, search_step

from cards import ATTACK_TABLE, CARD_TABLE
from encoding import encode_actions, encode_state, enumerate_actions
from model import evaluate
from reward import phi as phi_fn

# Carte "riempitivo" per le zone avversarie ignote. Servono solo a soddisfare
# i vincoli di lunghezza di search_begin: devono essere ID validi, e il mazzo
# avversario deve contenere almeno un Pokemon Base al setup.
_BASIC_POKEMON = sorted(
    cid
    for cid, c in CARD_TABLE.items()
    if c.cardType == CardType.POKEMON and getattr(c, "basic", False)
)
_BASIC_ENERGY = sorted(
    cid for cid, c in CARD_TABLE.items() if c.cardType == CardType.BASIC_ENERGY
)
FILLER_BASIC = _BASIC_POKEMON[0] if _BASIC_POKEMON else 1
FILLER_ENERGY = _BASIC_ENERGY[0] if _BASIC_ENERGY else 1


class SearchConfig:
    """Iperparametri della ricerca. Tutti tarabili, tutti con un default sano."""

    def __init__(
        self,
        n_simulations=48,
        c_puct=1.4,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.25,
        phi_weight=0.0,
        temperature=1.0,
        temp_moves=12,
        time_budget=None,
        max_actions=64,
    ):
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.phi_weight = phi_weight        # peso di Phi nel valore di foglia
        self.temperature = temperature
        self.temp_moves = temp_moves        # dopo N turni si gioca deterministico
        self.time_budget = time_budget      # secondi per questa decisione
        self.max_actions = max_actions


class Node:
    __slots__ = (
        "state", "player", "actions", "priors", "children",
        "visits", "value_sum", "terminal_value", "expanded",
    )

    def __init__(self, state, player):
        self.state = state          # SearchState (None se non materializzato)
        self.player = player        # chi muove in questo nodo
        self.actions = []           # list[list[int]]
        self.priors = None          # np.ndarray
        self.children = {}          # indice azione -> Node
        self.visits = 0
        self.value_sum = 0.0        # sempre nella prospettiva del root player
        self.terminal_value = None
        self.expanded = False

    def q(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


def _terminal_value(state, root_player):
    """Valore di uno stato finale, nella prospettiva del root player."""
    result = state.result
    if result < 0:
        return None
    if result == 2:
        return 0.0
    return 1.0 if result == root_player else -1.0


def build_determinization(obs, your_deck, rng, opponent_known=None):
    """Argomenti per `search_begin`: un'ipotesi completa sull'informazione nascosta.

    Le lunghezze devono combaciare *esattamente* con lo stato, altrimenti
    l'API solleva ValueError.
    """
    state = obs.current
    me_idx = state.yourIndex
    me = state.players[me_idx]
    them = state.players[1 - me_idx]

    # Il nostro mazzo: quello che sappiamo di avere, meno cio' che e' gia'
    # visibile altrove. Campionare da qui e' molto meglio che riempire a caso.
    visible = []
    for c in (me.hand or []):
        visible.append(c.id)
    for c in me.discard:
        visible.append(c.id)
    for p in ([x for x in me.active if x] + [x for x in me.bench if x]):
        visible.append(p.id)
        visible.extend(c.id for c in p.energyCards)
        visible.extend(c.id for c in p.tools)

    remaining = list(your_deck)
    for cid in visible:
        if cid in remaining:
            remaining.remove(cid)

    need = me.deckCount + len(me.prize)
    while len(remaining) < need:
        remaining.append(FILLER_ENERGY)
    rng.shuffle(remaining)
    my_deck_guess = remaining[: me.deckCount]
    my_prize_guess = remaining[me.deckCount: me.deckCount + len(me.prize)]

    # L'avversario: partiamo dalle carte che gli abbiamo gia' visto giocare
    # (scarti + campo), il resto e' filler.
    known = list(opponent_known or [])
    for c in them.discard:
        known.append(c.id)

    opp_total = them.deckCount + len(them.prize) + them.handCount
    opp_pool = [c for c in known][:opp_total]
    while len(opp_pool) < opp_total:
        # Almeno un Base deve esserci nel mazzo avversario al setup.
        opp_pool.append(FILLER_BASIC if len(opp_pool) % 4 == 0 else FILLER_ENERGY)
    rng.shuffle(opp_pool)

    opp_deck = opp_pool[: them.deckCount]
    opp_prize = opp_pool[them.deckCount: them.deckCount + len(them.prize)]
    opp_hand = opp_pool[them.deckCount + len(them.prize):]
    if them.deckCount > 0 and FILLER_BASIC not in opp_deck:
        opp_deck[0] = FILLER_BASIC

    opp_active = []
    if them.active and them.active[0] is None:
        opp_active = [FILLER_BASIC]

    return dict(
        your_deck=my_deck_guess,
        your_prize=my_prize_guess,
        opponent_deck=opp_deck,
        opponent_prize=opp_prize,
        opponent_hand=opp_hand,
        opponent_active=opp_active,
    )


def _expand(node, obs, your_deck, model, device, cfg, root_player):
    """Valuta una foglia con la rete e ne crea i figli (non materializzati)."""
    state = obs.current
    term = _terminal_value(state, root_player)
    if term is not None:
        node.terminal_value = term
        node.expanded = True
        return term

    node.actions = enumerate_actions(obs.select, cfg.max_actions)
    state_enc = encode_state(obs, your_deck)
    action_enc = encode_actions(obs, node.actions)
    value, _phi_pred, priors = evaluate(model, state_enc, action_enc, device)

    node.priors = priors
    node.expanded = True

    # La rete valuta dal punto di vista di chi muove: riportiamo alla
    # prospettiva del root player, che e' l'unica in cui i valori si sommano.
    if node.player != root_player:
        value = -value

    if cfg.phi_weight > 0.0:
        phi_val = phi_fn(state, root_player, CARD_TABLE, ATTACK_TABLE)
        value = (1.0 - cfg.phi_weight) * value + cfg.phi_weight * phi_val

    return value


def _select_child(node, cfg, root_player):
    """PUCT. Q e' letto nella prospettiva di chi muove in `node`."""
    total = max(1, node.visits)
    sqrt_total = math.sqrt(total)
    best_score = -1e18
    best = 0
    sign = 1.0 if node.player == root_player else -1.0

    for i in range(len(node.actions)):
        child = node.children.get(i)
        if child is None or child.visits == 0:
            q = 0.0
            visits = 0
        else:
            q = sign * child.q()
            visits = child.visits
        prior = float(node.priors[i]) if node.priors is not None and i < len(node.priors) else 1.0
        u = cfg.c_puct * prior * sqrt_total / (1 + visits)
        score = q + u
        if score > best_score:
            best_score = score
            best = i
    return best


def _fill_stats(stats_out, actions, policy, priors, visits, qs, root_value, source):
    """Riempie il dict di diagnostica della radice (vedi `run_mcts`).

    Puramente osservativo: nessun campo qui dentro rientra nella ricerca.
    """
    stats_out["source"] = source
    stats_out["root_value"] = float(root_value)
    stats_out["n_simulations"] = int(sum(visits)) if visits is not None else 0
    stats_out["actions"] = [
        {
            "index": i,
            "action": list(a),
            "policy": float(policy[i]) if policy is not None and i < len(policy) else 0.0,
            "prior": float(priors[i]) if priors is not None and i < len(priors) else None,
            "visits": int(visits[i]) if visits is not None and i < len(visits) else 0,
            "q": (float(qs[i]) if qs is not None and i < len(qs) and qs[i] is not None
                  else None),
        }
        for i, a in enumerate(actions)
    ]


def run_mcts(obs, your_deck, model, device, cfg, rng=None, opponent_known=None,
             stats_out=None):
    """Esegue la ricerca e restituisce (azione_scelta, policy_target, root_value).

    `policy_target` e' la distribuzione delle visite alla radice: e' il target
    con cui si allena la policy (come in AlphaZero), molto piu' informativo del
    solo indice della mossa scelta.

    Se `search_begin` fallisce (determinizzazione incoerente, API non
    disponibile) si ripiega sulla sola policy della rete: nessuna ricerca, ma
    una mossa legale e ragionevole.

    `stats_out`, se passato, e' un dict che viene riempito con la fotografia
    della radice: valore, e per ogni azione candidata prior della rete, visite e
    Q della MCTS (sempre nella prospettiva di chi muove). Serve solo alla
    dashboard (`replay_export.py`) per mostrare le mosse candidate come fa un
    motore di scacchi; non influenza in alcun modo la ricerca ne' il risultato,
    quindi puo' essere lasciato a None in training e in partita.
    """
    rng = rng or random.Random()
    root_player = obs.current.yourIndex
    deadline = time.perf_counter() + cfg.time_budget if cfg.time_budget else None

    actions = enumerate_actions(obs.select, cfg.max_actions)
    if len(actions) <= 1:
        # Nessuna scelta reale: inutile spendere una ricerca.
        target = np.ones(max(1, len(actions)), dtype=np.float32)
        if stats_out is not None:
            _fill_stats(stats_out, actions, target, None, None, None, 0.0, "forzata")
            stats_out["chosen"] = 0
        return (actions[0] if actions else []), target, 0.0

    try:
        det = build_determinization(obs, your_deck, rng, opponent_known)
        root_state = search_begin(obs, **det)
    except BaseException:
        state_enc = encode_state(obs, your_deck)
        action_enc = encode_actions(obs, actions)
        value, _phi, priors = evaluate(model, state_enc, action_enc, device)
        best = int(np.argmax(priors))
        if stats_out is not None:
            _fill_stats(stats_out, actions, priors, priors, None, None, value, "policy")
            stats_out["chosen"] = best
        return actions[best], priors.astype(np.float32), value

    try:
        root = Node(root_state, root_player)
        root_value = _expand(
            root, root_state.observation, your_deck, model, device, cfg, root_player
        )
        root.visits = 1
        root.value_sum = root_value
        # Copia *prima* del rumore e prima che le mosse rifiutate dal motore
        # vengano azzerate: e' il prior "pulito" della rete, l'unico leggibile.
        raw_priors = None if root.priors is None else root.priors.copy()

        # Rumore di Dirichlet: garantisce che ogni mossa alla radice abbia una
        # probabilita' non nulla di essere provata almeno una volta.
        if root.priors is not None and cfg.dirichlet_eps > 0 and len(root.actions) > 1:
            noise = rng_dirichlet(rng, cfg.dirichlet_alpha, len(root.priors))
            root.priors = (
                (1 - cfg.dirichlet_eps) * root.priors + cfg.dirichlet_eps * noise
            )

        for _ in range(cfg.n_simulations):
            if deadline is not None and time.perf_counter() > deadline:
                break

            node = root
            path = [root]

            # --- selezione ---------------------------------------------------
            while True:
                if node.terminal_value is not None:
                    break
                idx = _select_child(node, cfg, root_player)
                child = node.children.get(idx)
                if child is None:
                    try:
                        next_state = search_step(node.state.searchId, node.actions[idx])
                    except BaseException:
                        # Mossa rifiutata dal motore su questa determinizzazione:
                        # la si esclude e si passa oltre.
                        if node.priors is not None and idx < len(node.priors):
                            node.priors[idx] = 0.0
                        break
                    child = Node(next_state, next_state.observation.current.yourIndex)
                    node.children[idx] = child
                    path.append(child)
                    node = child
                    break
                path.append(child)
                node = child
                if not node.expanded:
                    break

            # --- valutazione -------------------------------------------------
            if node.terminal_value is not None:
                value = node.terminal_value
            elif not node.expanded:
                value = _expand(
                    node, node.state.observation, your_deck, model, device, cfg, root_player
                )
            else:
                value = node.q()

            # --- backup ------------------------------------------------------
            for n in path:
                n.visits += 1
                n.value_sum += value

        # Target di policy = distribuzione delle visite alla radice.
        counts = np.zeros(len(root.actions), dtype=np.float32)
        for i, child in root.children.items():
            counts[i] = child.visits
        if counts.sum() <= 0:
            counts = (
                root.priors.astype(np.float32)
                if root.priors is not None
                else np.ones(len(root.actions), dtype=np.float32)
            )
        policy_target = counts / max(1e-8, counts.sum())

        chosen = _pick_action(policy_target, obs.current.turn, cfg, rng)
        if stats_out is not None:
            qs = [
                (root.children[i].q() if i in root.children and root.children[i].visits
                 else None)
                for i in range(len(root.actions))
            ]
            _fill_stats(stats_out, root.actions, policy_target, raw_priors,
                        counts, qs, root.q(), "mcts")
            stats_out["chosen"] = chosen
        return root.actions[chosen], policy_target, root.q()
    finally:
        # Libera sempre la memoria della ricerca, anche in caso di eccezione.
        try:
            search_end()
        except BaseException:
            pass


def _pick_action(policy, turn, cfg, rng):
    """Campionamento con temperatura: esplora presto, sfrutta tardi."""
    if cfg.temperature <= 0 or turn > cfg.temp_moves:
        return int(np.argmax(policy))
    p = np.power(policy, 1.0 / cfg.temperature)
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        return int(np.argmax(policy))
    p = p / total
    r = rng.random()
    acc = 0.0
    for i, prob in enumerate(p):
        acc += prob
        if r <= acc:
            return i
    return len(p) - 1


def rng_dirichlet(rng, alpha, n):
    """Dirichlet simmetrico da un `random.Random`, per restare riproducibili
    con lo stesso seed senza tirare dentro lo stato globale di numpy."""
    samples = [rng.gammavariate(alpha, 1.0) for _ in range(n)]
    total = sum(samples)
    if total <= 0:
        return np.full(n, 1.0 / n, dtype=np.float32)
    return np.array([s / total for s in samples], dtype=np.float32)
