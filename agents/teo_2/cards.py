"""Tabella statica di feature per carta e per attacco.

Questo modulo e' il motivo per cui teo_2 e' *deck-agnostico* dove teo_1 non lo
e'. teo_1 codifica ogni carta come un indice one-hot su ~1300 card ID: una
carta mai vista in training ha un embedding non allenato, quindi il modello va
ri-allenato per ogni mazzo nuovo.

Qui ogni carta viene descritta anche da un vettore di *attributi* (tipo, HP,
stage, costo di ritirata, weakness, danno e costo del miglior attacco, ...).
Una carta mai vista arriva comunque con attributi sensati, quindi la rete puo'
generalizzare: "Stage 2 da 330 HP con attacco da 260" e' interpretabile anche
se quel preciso card ID non e' mai comparso in una partita di training.

Il modello usa entrambe le viste: embedding per card ID (memorizza le carte
che conosce) + proiezione degli attributi (generalizza a quelle che non
conosce). Vedi model.TeoNet.
"""

import numpy as np

import cgpath  # noqa: F401  -- deve precedere qualsiasi import di cg.*

from cg.api import CardType, all_attack, all_card_data

# ---------------------------------------------------------------------------
# Dati statici, caricati una volta sola all'import (mai per chiamata)
# ---------------------------------------------------------------------------

_ALL_CARDS = all_card_data()
_ALL_ATTACKS = all_attack()

CARD_TABLE = {c.cardId: c for c in _ALL_CARDS}
ATTACK_TABLE = {a.attackId: a for a in _ALL_ATTACKS}

MAX_CARD_ID = max(CARD_TABLE) + 1
MAX_ATTACK_ID = max(ATTACK_TABLE) + 1

N_ENERGY_TYPES = 12   # COLORLESS .. TEAM_ROCKET
N_CARD_TYPES = 7      # POKEMON .. SPECIAL_ENERGY

# Riferimenti di normalizzazione: dividere per una costante plausibile tiene
# le feature intorno a [0,1] senza dover calcolare statistiche sul dataset.
HP_SCALE = 400.0
DAMAGE_SCALE = 300.0
RETREAT_SCALE = 4.0
COST_SCALE = 5.0


def _one_hot(size, index):
    v = np.zeros(size, dtype=np.float32)
    if index is not None and 0 <= int(index) < size:
        v[int(index)] = 1.0
    return v


def _energy_one_hot(energy_type):
    """One-hot su 13: i 12 tipi piu' una casella "assente" in fondo."""
    v = np.zeros(N_ENERGY_TYPES + 1, dtype=np.float32)
    if energy_type is None:
        v[N_ENERGY_TYPES] = 1.0
    elif 0 <= int(energy_type) < N_ENERGY_TYPES:
        v[int(energy_type)] = 1.0
    return v


def _attack_features(attack):
    """Vettore di feature per un attacco (17 dim)."""
    if attack is None:
        return np.zeros(ATTACK_FEAT_DIM, dtype=np.float32)
    cost = np.zeros(N_ENERGY_TYPES, dtype=np.float32)
    for e in attack.energies:
        if 0 <= int(e) < N_ENERGY_TYPES:
            cost[int(e)] += 1.0
    text = attack.text or ""
    flags = np.array(
        [
            (attack.damage or 0) / DAMAGE_SCALE,
            len(attack.energies) / COST_SCALE,
            # Piccolo insieme *enumerato* di pattern testuali. Non e' un parser
            # generico del testo: sono le cinque famiglie di effetto che
            # cambiano davvero la valutazione di uno scambio.
            1.0 if "Weakness" in text else 0.0,
            1.0 if "Resistance" in text else 0.0,
            1.0 if "flip a coin" in text.lower() else 0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([cost, flags])


ATTACK_FEAT_DIM = N_ENERGY_TYPES + 5


def _card_features(card):
    """Vettore di attributi statici per una carta."""
    if card is None:
        return np.zeros(CARD_FEAT_DIM, dtype=np.float32)

    best_damage = 0.0
    best_cost = 0.0
    for aid in card.attacks:
        atk = ATTACK_TABLE.get(aid)
        if atk is None:
            continue
        if (atk.damage or 0) > best_damage:
            best_damage = float(atk.damage or 0)
            best_cost = float(len(atk.energies))

    is_pokemon = card.cardType == CardType.POKEMON
    prize = 3.0 if getattr(card, "megaEx", False) else 2.0 if getattr(card, "ex", False) else 1.0

    scalars = np.array(
        [
            (card.hp or 0) / HP_SCALE,
            (card.retreatCost or 0) / RETREAT_SCALE,
            best_damage / DAMAGE_SCALE,
            best_cost / COST_SCALE,
            len(card.attacks) / 3.0,
            1.0 if card.skills else 0.0,          # ha un'abilita'
            prize / 3.0 if is_pokemon else 0.0,
            1.0 if getattr(card, "basic", False) else 0.0,
            1.0 if getattr(card, "stage1", False) else 0.0,
            1.0 if getattr(card, "stage2", False) else 0.0,
            1.0 if getattr(card, "ex", False) else 0.0,
            1.0 if getattr(card, "megaEx", False) else 0.0,
            1.0 if getattr(card, "tera", False) else 0.0,
            1.0 if getattr(card, "aceSpec", False) else 0.0,
            1.0 if card.evolvesFrom else 0.0,
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        [
            _one_hot(N_CARD_TYPES, card.cardType),
            _energy_one_hot(card.energyType if is_pokemon else None),
            _energy_one_hot(card.weakness),
            _energy_one_hot(card.resistance),
            scalars,
        ]
    )


CARD_FEAT_DIM = N_CARD_TYPES + 3 * (N_ENERGY_TYPES + 1) + 15


# ---------------------------------------------------------------------------
# Matrici precalcolate: lookup O(1) per card ID / attack ID.
# L'indice 0 di ogni matrice e' il vettore nullo = "nessuna carta" (padding).
# ---------------------------------------------------------------------------

CARD_FEATURES = np.zeros((MAX_CARD_ID, CARD_FEAT_DIM), dtype=np.float32)
for _cid, _card in CARD_TABLE.items():
    CARD_FEATURES[_cid] = _card_features(_card)

ATTACK_FEATURES = np.zeros((MAX_ATTACK_ID, ATTACK_FEAT_DIM), dtype=np.float32)
for _aid, _atk in ATTACK_TABLE.items():
    ATTACK_FEATURES[_aid] = _attack_features(_atk)


def card_features(card_id):
    """Feature statiche di una carta, o vettore nullo se l'ID e' ignoto."""
    if card_id is None or not (0 <= card_id < MAX_CARD_ID):
        return np.zeros(CARD_FEAT_DIM, dtype=np.float32)
    return CARD_FEATURES[card_id]


def attack_features(attack_id):
    if attack_id is None or not (0 <= attack_id < MAX_ATTACK_ID):
        return np.zeros(ATTACK_FEAT_DIM, dtype=np.float32)
    return ATTACK_FEATURES[attack_id]


def aggregate_card_features(card_ids):
    """Media delle feature di un insieme di carte (mano, mazzo, scarti).

    Descrive la *composizione* di una zona senza dipendere dall'ordine ne'
    dalla dimensione: e' cosi' che il modello "legge" un mazzo che non ha mai
    visto.
    """
    if not card_ids:
        return np.zeros(CARD_FEAT_DIM, dtype=np.float32)
    valid = [c for c in card_ids if c is not None and 0 <= c < MAX_CARD_ID]
    if not valid:
        return np.zeros(CARD_FEAT_DIM, dtype=np.float32)
    return CARD_FEATURES[valid].mean(axis=0)
