"""Feature statiche per carta/attacco + **vocabolario ridotto** degli embedding.

E' qui che v2 diverge da teo_2 sul punto che conta di piu' per il budget di
training disponibile (CPU, nessuna GPU).

teo_2 tiene un embedding per *ogni* card ID del gioco: 1268 x 256 = 325k
parametri, piu' 1557 x 256 = 399k per gli attacchi. Sono ~724k pesi di cui, in
una partita di questo mazzo, se ne aggiornano forse 40: tutti gli altri restano
inizializzazione casuale e vengono comunque letti dall'attenzione. E' rumore
pagato a peso pieno.

Qui il vocabolario e' **chiuso e piccolo**:

  - le 19 carte distinte del nostro mazzo   -> indici 1..19
  - le carte dei mazzi avversari attesi     -> indici 20..N (dal pool di
    training, tipicamente ~60 carte in tutto)
  - tutto il resto                          -> indice 0 ("carta ignota")

Una carta fuori vocabolario non diventa invisibile: continua ad arrivare alla
rete attraverso il **vettore di attributi** (tipo, HP, stage, ritirata,
weakness, danno/costo del miglior attacco...), che e' la vista che generalizza.
Perde solo l'identita' memorizzata, che per una carta mai vista in training non
avrebbe comunque significato.

Effetto pratico: l'embedding passa da ~724k a ~8k parametri, e ogni riga che
resta viene aggiornata migliaia di volte invece di zero. E' la differenza tra
una rete che si allena in ore di CPU e una che non si allena affatto.

Il vocabolario viaggia **dentro il checkpoint** (vedi model.save/load): un
checkpoint e il suo vocabolario non possono disallinearsi, che sarebbe l'unico
modo silenzioso di rompere tutto.
"""

import numpy as np

import cgpath  # noqa: F401  -- deve precedere qualsiasi import di cg.*

from cg.api import CardType, all_attack, all_card_data

# ---------------------------------------------------------------------------
# Dati statici, caricati una volta sola all'import (mai per chiamata)
# ---------------------------------------------------------------------------

CARD_TABLE = {c.cardId: c for c in all_card_data()}
ATTACK_TABLE = {a.attackId: a for a in all_attack()}

N_ENERGY_TYPES = 12   # COLORLESS .. TEAM_ROCKET
N_CARD_TYPES = 7      # POKEMON .. SPECIAL_ENERGY

# Riferimenti di normalizzazione: dividere per una costante plausibile tiene le
# feature intorno a [0,1] senza dover calcolare statistiche sul dataset.
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


ATTACK_FEAT_DIM = N_ENERGY_TYPES + 5
CARD_FEAT_DIM = N_CARD_TYPES + 3 * (N_ENERGY_TYPES + 1) + 15


def _attack_features(attack):
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
            # Insieme *enumerato* di pattern testuali, non un parser generico:
            # sono le poche famiglie di effetto che cambiano la valutazione.
            1.0 if "Weakness" in text else 0.0,
            1.0 if "Resistance" in text else 0.0,
            1.0 if "flip a coin" in text.lower() else 0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([cost, flags])


def _card_features(card):
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


# Matrici precalcolate: lookup O(1) per ID. L'indice 0 e' il vettore nullo.
_MAX_CARD_ID = max(CARD_TABLE) + 1
_MAX_ATTACK_ID = max(ATTACK_TABLE) + 1

CARD_FEATURES = np.zeros((_MAX_CARD_ID, CARD_FEAT_DIM), dtype=np.float32)
for _cid, _card in CARD_TABLE.items():
    CARD_FEATURES[_cid] = _card_features(_card)

ATTACK_FEATURES = np.zeros((_MAX_ATTACK_ID, ATTACK_FEAT_DIM), dtype=np.float32)
for _aid, _atk in ATTACK_TABLE.items():
    ATTACK_FEATURES[_aid] = _attack_features(_atk)


def card_features(card_id):
    if card_id is None or not (0 <= card_id < _MAX_CARD_ID):
        return np.zeros(CARD_FEAT_DIM, dtype=np.float32)
    return CARD_FEATURES[card_id]


def attack_features(attack_id):
    if attack_id is None or not (0 <= attack_id < _MAX_ATTACK_ID):
        return np.zeros(ATTACK_FEAT_DIM, dtype=np.float32)
    return ATTACK_FEATURES[attack_id]


def aggregate_card_features(card_ids):
    """Media delle feature di un insieme di carte (mano, mazzo, scarti).

    Descrive la *composizione* di una zona senza dipendere dall'ordine ne'
    dalla dimensione. Non passa dal vocabolario: vale anche per carte ignote.
    """
    if not card_ids:
        return np.zeros(CARD_FEAT_DIM, dtype=np.float32)
    valid = [c for c in card_ids if c is not None and 0 <= c < _MAX_CARD_ID]
    if not valid:
        return np.zeros(CARD_FEAT_DIM, dtype=np.float32)
    return CARD_FEATURES[valid].mean(axis=0)


# ---------------------------------------------------------------------------
# Vocabolario ridotto
# ---------------------------------------------------------------------------

UNKNOWN = 0   # indice riservato: "carta fuori vocabolario"


class Vocab:
    """Mappa card ID -> indice compatto. Indice 0 = fuori vocabolario.

    Immutabile dopo la costruzione: aggiungere una carta a run iniziato
    cambierebbe la dimensione dell'embedding e renderebbe il checkpoint
    incompatibile con se stesso.
    """

    __slots__ = ("ids", "_index")

    def __init__(self, ids):
        # Ordinato e deduplicato: due costruzioni con lo stesso insieme di
        # carte devono dare gli stessi indici, altrimenti un checkpoint
        # salvato oggi legge le carte sbagliate domani.
        self.ids = tuple(sorted({int(c) for c in ids if c}))
        self._index = {cid: i + 1 for i, cid in enumerate(self.ids)}

    def __len__(self):
        """Numero di righe dell'embedding: le carte note piu' la riga 0."""
        return len(self.ids) + 1

    def index(self, card_id):
        if card_id is None:
            return UNKNOWN
        return self._index.get(int(card_id), UNKNOWN)

    def indices(self, card_ids):
        return [self.index(c) for c in card_ids]

    def to_list(self):
        return list(self.ids)


def build_vocab(my_deck, opponent_decks=()):
    """Vocabolario = il nostro mazzo + i mazzi avversari noti al momento del
    training.

    Includere gli avversari non serve a "conoscere" la loro lista: serve a dare
    un'identita' stabile alle poche carte contro cui giochiamo davvero (i loro
    attaccanti principali), che e' esattamente l'informazione che il vettore di
    attributi non riesce a distinguere -- due Stage 2 da 330 HP con attacco da
    260 hanno attributi quasi identici e conseguenze molto diverse.
    """
    ids = set(my_deck)
    for deck in opponent_decks:
        ids.update(deck)
    return Vocab(ids)
