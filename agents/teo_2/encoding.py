"""Encoding di stato e azioni per teo_2.

Lo stato diventa una sequenza di 18 *token* (uno per slot Pokemon, uno per
giocatore, uno per mano/mazzo/stadio/globale) su cui gira un transformer.
Ogni token porta con se':

  - un vettore dinamico (HP, energie, condizioni, ...);
  - le feature statiche della carta (cards.CARD_FEATURES) -> generalizza;
  - gli ID delle carte coinvolte, per l'embedding -> memorizza.

Il token globale include anche le 8 componenti di Phi (reward.phi_components):
darle in pasto direttamente alla rete evita che debba re-imparare da zero la
prize-race math, che e' esattamente il tipo di conto che una rete impara
lentamente e un'euristica calcola esatto.

Rispetto a teo_1: niente EmbeddingBag su vocabolario sparso da 22k, niente
dipendenza dal mazzo specifico. Le dimensioni non cambiano al cambiare del
mazzo, quindi lo stesso checkpoint gioca qualsiasi lista.
"""

from itertools import combinations

import numpy as np

import cgpath  # noqa: F401  -- deve precedere qualsiasi import di cg.*

from cg.api import AreaType, OptionType

from cards import (
    ATTACK_FEAT_DIM,
    CARD_FEAT_DIM,
    CARD_TABLE,
    ATTACK_TABLE,
    aggregate_card_features,
    attack_features,
    card_features,
)
from reward import phi_vector, _best_attack_readiness, prize_value

# ---------------------------------------------------------------------------
# Layout dei token
# ---------------------------------------------------------------------------

BENCH_SLOTS = 5
N_TOKENS = 2 + 2 * BENCH_SLOTS + 2 + 1 + 1 + 1 + 1  # = 18

TOK_MY_ACTIVE = 0
TOK_OP_ACTIVE = 1
TOK_MY_BENCH = 2                      # 2..6
TOK_OP_BENCH = TOK_MY_BENCH + BENCH_SLOTS   # 7..11
TOK_MY_PLAYER = TOK_OP_BENCH + BENCH_SLOTS  # 12
TOK_OP_PLAYER = TOK_MY_PLAYER + 1     # 13
TOK_HAND = TOK_MY_PLAYER + 2          # 14
TOK_DECK = TOK_MY_PLAYER + 3          # 15
TOK_STADIUM = TOK_MY_PLAYER + 4       # 16
TOK_GLOBAL = TOK_MY_PLAYER + 5        # 17

# Tipo di token (embedding separato, cosi' la rete sa "cosa" sta guardando)
TOKEN_TYPES = np.zeros(N_TOKENS, dtype=np.int64)
TOKEN_TYPES[TOK_MY_ACTIVE] = 0
TOKEN_TYPES[TOK_OP_ACTIVE] = 1
TOKEN_TYPES[TOK_MY_BENCH:TOK_MY_BENCH + BENCH_SLOTS] = 2
TOKEN_TYPES[TOK_OP_BENCH:TOK_OP_BENCH + BENCH_SLOTS] = 3
TOKEN_TYPES[TOK_MY_PLAYER] = 4
TOKEN_TYPES[TOK_OP_PLAYER] = 5
TOKEN_TYPES[TOK_HAND] = 6
TOKEN_TYPES[TOK_DECK] = 7
TOKEN_TYPES[TOK_STADIUM] = 8
TOKEN_TYPES[TOK_GLOBAL] = 9
N_TOKEN_TYPES = 10

DYN_DIM = 24                     # vettore dinamico per token
CARDS_PER_TOKEN = 8              # quante carte per token entrano nell'embedding
SLOT_FEAT_DIM = DYN_DIM + CARD_FEAT_DIM

N_OPTION_TYPES = 17              # OptionType.NUMBER .. SPECIAL_CONDITION
N_CONTEXTS = 49                  # SelectContext.MAIN .. RECOVER_SPECIAL_CONDITION

ACTION_SCALARS = 6
ACTION_FEAT_DIM = (
    N_OPTION_TYPES + N_CONTEXTS + ACTION_SCALARS + 2 * CARD_FEAT_DIM + ATTACK_FEAT_DIM
)
CARDS_PER_ACTION = 2             # carta sorgente + Pokemon bersaglio

MAX_ACTIONS = 64                 # tetto sulle azioni candidate per decisione


class StateEncoding:
    """Tensori (numpy) di uno stato. Convertiti in torch dal modello."""

    __slots__ = ("slot_feats", "card_ids", "card_mask", "token_types", "phi")

    def __init__(self, slot_feats, card_ids, card_mask, phi):
        self.slot_feats = slot_feats      # (N_TOKENS, SLOT_FEAT_DIM)
        self.card_ids = card_ids          # (N_TOKENS, CARDS_PER_TOKEN)
        self.card_mask = card_mask        # (N_TOKENS, CARDS_PER_TOKEN)
        self.token_types = TOKEN_TYPES
        self.phi = phi                    # (8,) componenti, target ausiliario


class ActionEncoding:
    __slots__ = ("feats", "card_ids", "card_mask", "attack_ids", "actions")

    def __init__(self, feats, card_ids, card_mask, attack_ids, actions):
        self.feats = feats                # (A, ACTION_FEAT_DIM)
        self.card_ids = card_ids          # (A, CARDS_PER_ACTION)
        self.card_mask = card_mask        # (A, CARDS_PER_ACTION)
        self.attack_ids = attack_ids      # (A,)
        self.actions = actions            # list[list[int]] indici opzione


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _active_of(ps):
    return ps.active[0] if ps.active else None


def get_card(obs, area, index, player_index):
    """Risolve area/index/playerIndex di un'Option nella carta corrispondente.

    None-safe ovunque: gli slot vuoti e le carte coperte sono normali e non
    devono sollevare eccezioni.
    """
    if area is None or index is None:
        return None
    try:
        if area == AreaType.DECK:
            deck = obs.select.deck
            return deck[index] if deck and index < len(deck) else None
        pi = player_index if player_index is not None else obs.current.yourIndex
        ps = obs.current.players[pi]
        if area == AreaType.HAND:
            return ps.hand[index] if ps.hand and index < len(ps.hand) else None
        if area == AreaType.DISCARD:
            return ps.discard[index] if index < len(ps.discard) else None
        if area == AreaType.ACTIVE:
            return ps.active[index] if index < len(ps.active) else None
        if area == AreaType.BENCH:
            return ps.bench[index] if index < len(ps.bench) else None
        if area == AreaType.PRIZE:
            return ps.prize[index] if index < len(ps.prize) else None
        if area == AreaType.STADIUM:
            st = obs.current.stadium
            return st[index] if st and index < len(st) else None
        if area == AreaType.LOOKING:
            lk = obs.current.looking
            return lk[index] if lk and index < len(lk) else None
    except (IndexError, TypeError, AttributeError):
        return None
    return None


def _pokemon_token(pokemon, is_active):
    """(dyn, cardfeat, ids) per uno slot Pokemon."""
    dyn = np.zeros(DYN_DIM, dtype=np.float32)
    ids = []
    if pokemon is None:
        return dyn, np.zeros(CARD_FEAT_DIM, dtype=np.float32), ids

    max_hp = max(1, pokemon.maxHp)
    data = CARD_TABLE.get(pokemon.id)
    dyn[0] = 1.0
    dyn[1] = pokemon.hp / max_hp
    dyn[2] = max_hp / 400.0
    dyn[3] = max(0, max_hp - pokemon.hp) / max_hp
    dyn[4] = len(pokemon.energies) / 6.0
    dyn[5] = len(pokemon.tools) / 2.0
    dyn[6] = _best_attack_readiness(pokemon, CARD_TABLE, ATTACK_TABLE)
    dyn[7] = 1.0 if pokemon.appearThisTurn else 0.0
    dyn[8] = prize_value(data) / 3.0
    dyn[9] = 1.0 if is_active else 0.0
    dyn[10] = len(pokemon.preEvolution) / 2.0

    ids.append(pokemon.id)
    for c in pokemon.tools[: CARDS_PER_TOKEN - 1]:
        ids.append(c.id)
    for c in pokemon.energyCards[: max(0, CARDS_PER_TOKEN - len(ids))]:
        ids.append(c.id)

    return dyn, card_features(pokemon.id), ids


def _player_token(ps):
    dyn = np.zeros(DYN_DIM, dtype=np.float32)
    dyn[0] = ps.deckCount / 60.0
    dyn[1] = ps.handCount / 10.0
    dyn[2] = len(ps.discard) / 60.0
    dyn[3] = len(ps.prize) / 6.0
    dyn[4] = len(ps.bench) / 5.0
    dyn[5] = max(1, ps.benchMax) / 5.0
    dyn[6] = 1.0 if ps.poisoned else 0.0
    dyn[7] = 1.0 if ps.burned else 0.0
    dyn[8] = 1.0 if ps.asleep else 0.0
    dyn[9] = 1.0 if ps.paralyzed else 0.0
    dyn[10] = 1.0 if ps.confused else 0.0
    dyn[11] = max(0, 6 - ps.deckCount) / 6.0
    # Composizione degli scarti: dice molto su cosa e' gia' stato bruciato.
    return dyn, aggregate_card_features([c.id for c in ps.discard])


def encode_state(obs, your_deck):
    """Costruisce la StateEncoding a partire da un'Observation."""
    state = obs.current
    me_idx = state.yourIndex
    me = state.players[me_idx]
    them = state.players[1 - me_idx]

    slot_feats = np.zeros((N_TOKENS, SLOT_FEAT_DIM), dtype=np.float32)
    card_ids = np.zeros((N_TOKENS, CARDS_PER_TOKEN), dtype=np.int64)
    card_mask = np.zeros((N_TOKENS, CARDS_PER_TOKEN), dtype=np.float32)

    def put(tok, dyn, cardfeat, ids=()):
        slot_feats[tok, :DYN_DIM] = dyn
        slot_feats[tok, DYN_DIM:] = cardfeat
        for k, cid in enumerate(list(ids)[:CARDS_PER_TOKEN]):
            if cid is not None and cid > 0:
                card_ids[tok, k] = cid
                card_mask[tok, k] = 1.0

    # Attivi
    for tok, ps, mine in ((TOK_MY_ACTIVE, me, True), (TOK_OP_ACTIVE, them, False)):
        dyn, cf, ids = _pokemon_token(_active_of(ps), True)
        put(tok, dyn, cf, ids)

    # Panchine
    for base, ps in ((TOK_MY_BENCH, me), (TOK_OP_BENCH, them)):
        for j in range(BENCH_SLOTS):
            p = ps.bench[j] if j < len(ps.bench) else None
            dyn, cf, ids = _pokemon_token(p, False)
            put(base + j, dyn, cf, ids)

    # Riepiloghi giocatore
    for tok, ps in ((TOK_MY_PLAYER, me), (TOK_OP_PLAYER, them)):
        dyn, cf = _player_token(ps)
        put(tok, dyn, cf)

    # Mano (solo la nostra e' visibile)
    hand_ids = [c.id for c in (me.hand or [])]
    hdyn = np.zeros(DYN_DIM, dtype=np.float32)
    hdyn[0] = len(hand_ids) / 10.0
    put(TOK_HAND, hdyn, aggregate_card_features(hand_ids), hand_ids[:CARDS_PER_TOKEN])

    # Mazzo: solo composizione aggregata (l'ordine e' ignoto per definizione)
    ddyn = np.zeros(DYN_DIM, dtype=np.float32)
    ddyn[0] = me.deckCount / 60.0
    put(TOK_DECK, ddyn, aggregate_card_features(list(your_deck)))

    # Stadio
    sdyn = np.zeros(DYN_DIM, dtype=np.float32)
    stadium_ids = [c.id for c in state.stadium] if state.stadium else []
    sdyn[0] = 1.0 if stadium_ids else 0.0
    put(TOK_STADIUM, sdyn, aggregate_card_features(stadium_ids), stadium_ids)

    # Globale + componenti di Phi
    phi_comp = phi_vector(state, me_idx, CARD_TABLE, ATTACK_TABLE)
    gdyn = np.zeros(DYN_DIM, dtype=np.float32)
    gdyn[0] = min(state.turn, 40) / 20.0
    gdyn[1] = 1.0 if state.firstPlayer == me_idx else 0.0
    gdyn[2] = 1.0 if state.supporterPlayed else 0.0
    gdyn[3] = 1.0 if state.stadiumPlayed else 0.0
    gdyn[4] = 1.0 if state.energyAttached else 0.0
    gdyn[5] = 1.0 if state.retreated else 0.0
    gdyn[6] = min(state.turnActionCount, 20) / 10.0
    gdyn[7] = 1.0
    gdyn[8:8 + len(phi_comp)] = phi_comp
    put(TOK_GLOBAL, gdyn, np.zeros(CARD_FEAT_DIM, dtype=np.float32))

    return StateEncoding(
        slot_feats, card_ids, card_mask, np.array(phi_comp, dtype=np.float32)
    )


# ---------------------------------------------------------------------------
# Azioni
# ---------------------------------------------------------------------------

def enumerate_actions(select, max_actions=MAX_ACTIONS):
    """Insiemi di indici opzione legali da valutare.

    Rispetta minCount/maxCount. Se lo spazio combinatorio esplode (multi-select
    ampi) viene troncato in modo *deterministico* ai primi `max_actions`: la
    MCTS lavora comunque su un sottoinsieme legale, e la determinatezza serve a
    non rendere l'agente irriproducibile.
    """
    n = len(select.option)
    if n == 0:
        return [[]]
    minc = max(0, select.minCount or 0)
    maxc = min(select.maxCount or 0, n)
    if maxc <= 0:
        return [[]]

    actions = []
    # Declinare, quando e' legale, e' spesso la mossa giusta: va sempre
    # considerato esplicitamente, non lasciato al caso.
    if minc == 0:
        actions.append([])

    # `maxc` e' gia' limitato a n, quindi combinations() copre da solo anche il
    # caso "prendi tutto" (size == n): non serve un ramo speciale, che anzi
    # duplicherebbe l'azione completa e la farebbe contare due volte alla MCTS.
    for size in range(max(1, minc), maxc + 1):
        for combo in combinations(range(n), size):
            actions.append(list(combo))
            if len(actions) >= max_actions:
                return actions
    return actions if actions else [[]]


def _action_scalars(select, options):
    v = np.zeros(ACTION_SCALARS, dtype=np.float32)
    if not options:
        return v
    o = options[0]
    v[0] = (o.number or 0) / 10.0
    v[1] = (o.count or 0) / 5.0
    v[2] = (o.energyIndex or 0) / 5.0
    v[3] = (o.toolIndex or 0) / 3.0
    v[4] = len(options) / 5.0
    v[5] = 1.0 if len(options) > 1 else 0.0
    return v


def _source_and_target(obs, o, me_idx):
    """Carta sorgente e Pokemon bersaglio di un'opzione, quando esistono."""
    src = None
    tgt = None
    t = o.type
    if t == OptionType.PLAY:
        src = get_card(obs, AreaType.HAND, o.index, me_idx)
    elif t in (OptionType.ATTACH, OptionType.EVOLVE):
        src = get_card(obs, o.area, o.index, me_idx)
        tgt = get_card(obs, o.inPlayArea, o.inPlayIndex, me_idx)
    elif t in (OptionType.ABILITY, OptionType.DISCARD):
        src = get_card(obs, o.area, o.index, me_idx)
    elif t == OptionType.RETREAT:
        src = _active_of(obs.current.players[me_idx])
    elif t == OptionType.CARD:
        src = get_card(obs, o.area, o.index, o.playerIndex)
    elif t == OptionType.TOOL_CARD:
        holder = get_card(obs, o.area, o.index, o.playerIndex)
        tgt = holder
        if holder is not None and o.toolIndex is not None and o.toolIndex < len(holder.tools):
            src = holder.tools[o.toolIndex]
    elif t in (OptionType.ENERGY_CARD, OptionType.ENERGY):
        holder = get_card(obs, o.area, o.index, o.playerIndex)
        tgt = holder
        if holder is not None and o.energyIndex is not None and o.energyIndex < len(holder.energyCards):
            src = holder.energyCards[o.energyIndex]
    return src, tgt


def encode_actions(obs, actions):
    """Costruisce la ActionEncoding per una lista di azioni candidate."""
    a_count = len(actions)
    feats = np.zeros((a_count, ACTION_FEAT_DIM), dtype=np.float32)
    card_ids = np.zeros((a_count, CARDS_PER_ACTION), dtype=np.int64)
    card_mask = np.zeros((a_count, CARDS_PER_ACTION), dtype=np.float32)
    attack_ids = np.zeros(a_count, dtype=np.int64)

    me_idx = obs.current.yourIndex
    context = int(obs.select.context)
    ctx_off = N_OPTION_TYPES
    sc_off = ctx_off + N_CONTEXTS
    src_off = sc_off + ACTION_SCALARS
    tgt_off = src_off + CARD_FEAT_DIM
    atk_off = tgt_off + CARD_FEAT_DIM

    for ai, action in enumerate(actions):
        if 0 <= context < N_CONTEXTS:
            feats[ai, ctx_off + context] = 1.0
        if not action:
            # Azione "declino": resta il solo one-hot del contesto.
            continue

        options = [obs.select.option[i] for i in action if i < len(obs.select.option)]
        if not options:
            continue

        for o in options:
            ot = int(o.type)
            if 0 <= ot < N_OPTION_TYPES:
                feats[ai, ot] = 1.0

        feats[ai, sc_off:sc_off + ACTION_SCALARS] = _action_scalars(obs.select, options)

        # Le feature di carta/attacco si riferiscono alla prima opzione: nei
        # multi-select le opzioni sono omogenee, quindi e' rappresentativa.
        head = options[0]
        src, tgt = _source_and_target(obs, head, me_idx)
        if src is not None:
            feats[ai, src_off:src_off + CARD_FEAT_DIM] = card_features(src.id)
            card_ids[ai, 0] = src.id
            card_mask[ai, 0] = 1.0
        if tgt is not None:
            feats[ai, tgt_off:tgt_off + CARD_FEAT_DIM] = card_features(tgt.id)
            card_ids[ai, 1] = tgt.id
            card_mask[ai, 1] = 1.0
        if head.type == OptionType.ATTACK and head.attackId is not None:
            feats[ai, atk_off:atk_off + ATTACK_FEAT_DIM] = attack_features(head.attackId)
            attack_ids[ai] = head.attackId

    return ActionEncoding(feats, card_ids, card_mask, attack_ids, actions)
