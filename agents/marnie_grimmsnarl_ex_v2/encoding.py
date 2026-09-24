"""Stato -> 18 token, per la sola **value head**.

Differenza strutturale rispetto a teo_2: qui non esiste nessun encoding delle
azioni, e non esiste una policy head.

Il motivo e' il punto di tutto l'agente. Chi propone le mosse e' l'euristica
(heuristic.py), che sul suo mazzo e' molto piu' forte di qualsiasi policy
allenabile con un budget CPU; alla rete resta un solo compito, "quanto e' buona
questa posizione", che e' anche l'unico dei due che una rete piccola impara con
poche migliaia di partite. Imparare V e' un problema di regressione su uno
scalare; imparare pi e' un problema di classificazione su uno spazio di azioni
che qui arriva a decine di alternative eterogenee.

Conseguenza pratica: sparisce l'intero ramo decoder/cross-attention di teo_2
(~2.4M parametri) e sparisce il target di policy, che era la parte piu' fragile
di quel training (entropia, KL, troncamento del buffer...).

I 18 token restano quelli di teo_2, che sono una descrizione sensata del
tabellone:

    0      il mio attivo
    1      il loro attivo
    2..6   la mia panchina
    7..11  la loro panchina
    12,13  riepilogo per giocatore (mazzo, mano, scarti, prize, condizioni)
    14     mano (composizione)
    15     mazzo (composizione)
    16     stadio
    17     globale (turno, flag di turno, le 8 componenti di Phi)
"""

import numpy as np

import cgpath  # noqa: F401  -- deve precedere qualsiasi import di cg.*

from cards import (
    CARD_FEAT_DIM,
    CARD_TABLE,
    ATTACK_TABLE,
    aggregate_card_features,
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
TOK_MY_BENCH = 2                              # 2..6
TOK_OP_BENCH = TOK_MY_BENCH + BENCH_SLOTS     # 7..11
TOK_MY_PLAYER = TOK_OP_BENCH + BENCH_SLOTS    # 12
TOK_OP_PLAYER = TOK_MY_PLAYER + 1             # 13
TOK_HAND = TOK_MY_PLAYER + 2                  # 14
TOK_DECK = TOK_MY_PLAYER + 3                  # 15
TOK_STADIUM = TOK_MY_PLAYER + 4               # 16
TOK_GLOBAL = TOK_MY_PLAYER + 5                # 17

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
CARDS_PER_TOKEN = 6              # quante carte per token entrano nell'embedding
SLOT_FEAT_DIM = DYN_DIM + CARD_FEAT_DIM

N_PHI = 8


class StateEncoding:
    """Tensori (numpy) di uno stato. Convertiti in torch dal modello."""

    __slots__ = ("slot_feats", "card_idx", "card_mask", "token_types", "phi")

    def __init__(self, slot_feats, card_idx, card_mask, phi):
        self.slot_feats = slot_feats      # (N_TOKENS, SLOT_FEAT_DIM)
        self.card_idx = card_idx          # (N_TOKENS, CARDS_PER_TOKEN) indici di vocab
        self.card_mask = card_mask        # (N_TOKENS, CARDS_PER_TOKEN)
        self.token_types = TOKEN_TYPES
        self.phi = phi                    # (8,) componenti, target ausiliario


def _active_of(ps):
    return ps.active[0] if ps.active else None


def _pokemon_token(pokemon, is_active):
    """(dyn, cardfeat, card_ids) per uno slot Pokemon."""
    dyn = np.zeros(DYN_DIM, dtype=np.float32)
    if pokemon is None:
        return dyn, np.zeros(CARD_FEAT_DIM, dtype=np.float32), []

    max_hp = max(1, pokemon.maxHp)
    dyn[0] = 1.0
    dyn[1] = pokemon.hp / max_hp
    dyn[2] = max_hp / 400.0
    dyn[3] = max(0, max_hp - pokemon.hp) / max_hp
    dyn[4] = len(pokemon.energies) / 6.0
    dyn[5] = len(pokemon.tools) / 2.0
    dyn[6] = _best_attack_readiness(pokemon, CARD_TABLE, ATTACK_TABLE)
    dyn[7] = 1.0 if pokemon.appearThisTurn else 0.0
    dyn[8] = prize_value(CARD_TABLE.get(pokemon.id)) / 3.0
    dyn[9] = 1.0 if is_active else 0.0
    dyn[10] = len(pokemon.preEvolution) / 2.0
    # Quante energie di *questo* Pokemon sono gia' spese sul costo dell'attacco
    # e' in dyn[6]; qui interessa il valore assoluto, che dice se e' un muro
    # carico o un corpo vuoto messo li' per riempire la panchina.
    dyn[11] = min(len(pokemon.energies), 4) / 4.0

    ids = [pokemon.id]
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
    return dyn, aggregate_card_features([c.id for c in ps.discard])


def encode_state(obs, my_deck, vocab, me_idx=None, hand_ids=None):
    """StateEncoding dal punto di vista di `me_idx` (default: chi deve muovere).

    `me_idx` esplicito serve al training: a fine turno lo stesso stato va
    registrato da entrambe le prospettive, e la prospettiva non e' sempre
    quella di chi ha il tratto.

    `hand_ids` sovrascrive la mano letta dall'osservazione, ed e' necessario --
    non un optional. Il motore mostra la mano solo al giocatore che ha il
    tratto, e gli stati che questa rete valuta sono per costruzione stati in
    cui il tratto e' appena passato: senza override la mano risulta vuota
    proprio nel momento in cui il suo contenuto conta di piu' (avere Rare Candy
    + Grimmsnarl ex in mano all'inizio del turno e' meta' del piano del mazzo).
    Chi chiama la passa dall'ultimo stato in cui era visibile -- vedi
    `search.complete_turn` e `train.collect_game`.
    """
    state = obs.current
    if me_idx is None:
        me_idx = state.yourIndex
    me = state.players[me_idx]
    them = state.players[1 - me_idx]

    slot_feats = np.zeros((N_TOKENS, SLOT_FEAT_DIM), dtype=np.float32)
    card_idx = np.zeros((N_TOKENS, CARDS_PER_TOKEN), dtype=np.int64)
    card_mask = np.zeros((N_TOKENS, CARDS_PER_TOKEN), dtype=np.float32)

    def put(tok, dyn, cardfeat, ids=()):
        slot_feats[tok, :DYN_DIM] = dyn
        slot_feats[tok, DYN_DIM:] = cardfeat
        for k, cid in enumerate(list(ids)[:CARDS_PER_TOKEN]):
            if cid is None:
                continue
            # Anche una carta fuori vocabolario occupa lo slot (mask=1) con
            # l'indice 0: "c'e' una carta e non so quale" e' informazione
            # diversa da "non c'e' niente".
            card_idx[tok, k] = vocab.index(cid)
            card_mask[tok, k] = 1.0

    for tok, ps in ((TOK_MY_ACTIVE, me), (TOK_OP_ACTIVE, them)):
        dyn, cf, ids = _pokemon_token(_active_of(ps), True)
        put(tok, dyn, cf, ids)

    for base, ps in ((TOK_MY_BENCH, me), (TOK_OP_BENCH, them)):
        for j in range(BENCH_SLOTS):
            p = ps.bench[j] if j < len(ps.bench) else None
            dyn, cf, ids = _pokemon_token(p, False)
            put(base + j, dyn, cf, ids)

    for tok, ps in ((TOK_MY_PLAYER, me), (TOK_OP_PLAYER, them)):
        dyn, cf = _player_token(ps)
        put(tok, dyn, cf)

    hand = list(hand_ids) if hand_ids is not None else [c.id for c in (me.hand or [])]
    hdyn = np.zeros(DYN_DIM, dtype=np.float32)
    hdyn[0] = (len(hand) or me.handCount) / 10.0
    hdyn[1] = 1.0 if hand else 0.0        # mano nota vs solo conteggio
    put(TOK_HAND, hdyn, aggregate_card_features(hand), hand)

    ddyn = np.zeros(DYN_DIM, dtype=np.float32)
    ddyn[0] = me.deckCount / 60.0
    put(TOK_DECK, ddyn, aggregate_card_features(list(my_deck)))

    sdyn = np.zeros(DYN_DIM, dtype=np.float32)
    stadium_ids = [c.id for c in state.stadium] if state.stadium else []
    sdyn[0] = 1.0 if stadium_ids else 0.0
    put(TOK_STADIUM, sdyn, aggregate_card_features(stadium_ids), stadium_ids)

    phi_comp = phi_vector(state, me_idx, CARD_TABLE, ATTACK_TABLE)
    gdyn = np.zeros(DYN_DIM, dtype=np.float32)
    gdyn[0] = min(state.turn, 40) / 20.0
    gdyn[1] = 1.0 if state.firstPlayer == me_idx else 0.0
    gdyn[2] = 1.0 if state.supporterPlayed else 0.0
    gdyn[3] = 1.0 if state.stadiumPlayed else 0.0
    gdyn[4] = 1.0 if state.energyAttached else 0.0
    gdyn[5] = 1.0 if state.retreated else 0.0
    gdyn[6] = min(state.turnActionCount, 20) / 10.0
    # Chi ha il tratto. Senza questa feature la rete valuterebbe alla stessa
    # maniera "sto per giocare io" e "sta per giocare l'avversario", che in un
    # gioco a turni e' quasi mezza partita di differenza -- e la ricerca chiede
    # esattamente stati di confine tra i due turni.
    gdyn[7] = 1.0 if state.yourIndex == me_idx else 0.0
    gdyn[8:8 + len(phi_comp)] = phi_comp
    put(TOK_GLOBAL, gdyn, np.zeros(CARD_FEAT_DIM, dtype=np.float32))

    return StateEncoding(
        slot_feats, card_idx, card_mask, np.array(phi_comp, dtype=np.float32)
    )
