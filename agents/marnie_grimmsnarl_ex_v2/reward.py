"""Phi: funzione potenziale per il reward shaping di teo_2.

Ogni componente e' *antisimmetrica* (io - avversario) e a valori in [-1, 1].
Phi(s) e' la media pesata delle componenti, quindi anch'essa in [-1, 1], ed e'
antisimmetrica per costruzione: phi(s, me) == -phi(s, avversario).

Uso in training (train_selfplay.py): potential-based reward shaping alla
Ng-Harada-Russell (1999),

    r_shaped(s, s') = GAMMA * phi(s') - phi(s)

che e' l'unica forma di shaping che *garantisce* di non cambiare la politica
ottima: qualsiasi cosa sbagliata ci sia nei pesi qui sotto, l'agente non puo'
convergere a una politica peggiore di quella che imparerebbe col solo segnale
terminale (vittoria/sconfitta). I pesi cambiano solo *quanto in fretta*
impara, non *cosa* impara. Per questo si possono tarare liberamente.

Le componenti servono anche come target di una testa ausiliaria della rete
(model.TeoNet.phi_head): predire Phi e' un compito di rappresentazione che
costringe l'encoder a estrarre proprio le feature che contano (prize race,
energia, evoluzioni), accelerando molto l'apprendimento del value head.
"""

import cgpath  # noqa: F401  -- deve precedere qualsiasi import di cg.*

from cg.api import EnergyType

# ---------------------------------------------------------------------------
# Pesi delle componenti. Sono i knob principali dello shaping.
# ---------------------------------------------------------------------------

WEIGHTS = {
    "prize": 1.00,          # rilassamento continuo del contatore prize
    "attack_ready": 0.15,   # copertura del costo del miglior attacco dell'attivo
    "energy": 0.10,         # energia utile sui 2 Pokemon migliori del proprio lato
    "evolution": 0.08,      # stage medio dei Pokemon in gioco
    "board_safety": 0.12,   # numero di Pokemon in gioco, non monotono
    "deckout": 0.10,        # rischio di finire il mazzo
    "type_matchup": 0.05,   # weakness dell'attivo avversario
    "conditions": 0.03,     # condizioni speciali
}

COMPONENT_ORDER = (
    "prize",
    "attack_ready",
    "energy",
    "evolution",
    "board_safety",
    "deckout",
    "type_matchup",
    "conditions",
)

_WEIGHT_SUM = sum(WEIGHTS.values())

PRIZE_TOTAL = 6         # prize di partenza nel formato standard
DECKOUT_HORIZON = 6     # sotto queste carte in mazzo il deckout inizia a pesare

# board_safety: NON monotona. Con 1 solo Pokemon in gioco un KO e' sconfitta
# immediata (-1); con 2 si e' fuori pericolo immediato (0); si cresce fino a 4
# e poi e' piatta -- over-benchare non va premiato, ogni Pokemon fragile in
# piu' e' un bersaglio gratis per effetti tipo Boss's Orders.
_SAFETY_CURVE = {0: -1.0, 1: -1.0, 2: 0.0, 3: 0.5, 4: 1.0}


def _clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def _active(ps):
    """Pokemon attivo, o None se lo slot e' vuoto / la carta e' coperta."""
    if not ps.active:
        return None
    return ps.active[0]


def _in_play(ps):
    """Tutti i Pokemon in gioco (attivo + panchina), ignorando gli slot vuoti."""
    out = []
    a = _active(ps)
    if a is not None:
        out.append(a)
    for p in ps.bench:
        if p is not None:
            out.append(p)
    return out


def prize_value(card_data):
    """Quanti prize cede questo Pokemon quando viene messo KO."""
    if card_data is None:
        return 1
    if getattr(card_data, "megaEx", False):
        return 3
    if getattr(card_data, "ex", False):
        return 2
    return 1


# ---------------------------------------------------------------------------
# Copertura energetica: quanto del costo di un attacco e' gia' pagato.
# ---------------------------------------------------------------------------

def energy_coverage(required, available):
    """Frazione in [0,1] del costo `required` coperta dalle energie `available`.

    Matching greedy corretto: prima si soddisfano i simboli *tipati*, che sono
    vincolanti, e solo dopo i {C} (colorless) con quello che avanza -- perche'
    una {F} puo' pagare un {C} ma una {C} non puo' pagare una {F}. Contare solo
    il numero totale di energie (come fanno molte euristiche) sovrastima la
    prontezza dei mazzi a due tipi.
    """
    if not required:
        return 1.0
    pool = list(available)
    paid = 0
    colorless = 0
    for need in required:
        if need == EnergyType.COLORLESS:
            colorless += 1
            continue
        for i, have in enumerate(pool):
            # RAINBOW paga qualsiasi simbolo.
            if have == need or have == EnergyType.RAINBOW:
                pool.pop(i)
                paid += 1
                break
    # I simboli colorless accettano qualunque energia rimasta.
    paid += min(colorless, len(pool))
    return _clamp(paid / len(required), 0.0, 1.0)


def _best_attack_readiness(pokemon, card_table, attack_table):
    """Frazione del costo del *miglior* attacco (piu' danno) gia' coperta.

    Approssimazione dichiarata: "miglior attacco" = quello con danno base piu'
    alto. Per attacchi a danno variabile (`damage == 0` con testo tipo "10x per
    ogni segnalino") il campo statico vale 0, quindi vengono trattati come
    l'attacco piu' debole. E' conservativo e va bene: il ramo value della rete
    impara comunque il valore reale di quei Pokemon dall'esito delle partite.
    """
    if pokemon is None:
        return 0.0
    data = card_table.get(pokemon.id)
    if data is None or not data.attacks:
        return 0.0
    best = None
    best_damage = -1
    for aid in data.attacks:
        atk = attack_table.get(aid)
        if atk is None:
            continue
        dmg = atk.damage or 0
        if dmg > best_damage:
            best_damage = dmg
            best = atk
    if best is None:
        return 0.0
    return energy_coverage(best.energies, pokemon.energies)


# ---------------------------------------------------------------------------
# Le otto componenti. Ognuna restituisce il valore per UN lato; la componente
# antisimmetrica finale e' calcolata in phi_components().
# ---------------------------------------------------------------------------

def _prize_progress(me, them, card_table):
    """Rilassamento continuo del contatore prize, in [0,1].

    prize presi finora + credito parziale per il danno gia' messo sui Pokemon
    avversari. Il cap a 1 su ogni ko_frac e' essenziale: l'overkill non deve
    valere piu' di un KO, altrimenti l'agente impara a caricare danno su un
    bersaglio gia' morto invece di spalmarlo. Il peso per prize_value
    riproduce la prize-trade math vera: danneggiare un ex vale il doppio.
    """
    taken = PRIZE_TOTAL - len(me.prize)
    partial = 0.0
    for p in _in_play(them):
        if p.maxHp <= 0:
            continue
        damage = max(0, p.maxHp - p.hp)
        ko_frac = min(1.0, damage / p.maxHp)
        partial += prize_value(card_table.get(p.id)) * ko_frac
    # Cap a 1: prendere 6 prize e' la vittoria, non esiste "piu' del 100%".
    return _clamp((taken + partial) / PRIZE_TOTAL, 0.0, 1.0)


def _attack_ready(ps, card_table, attack_table):
    return _best_attack_readiness(_active(ps), card_table, attack_table)


def _energy(ps, card_table, attack_table):
    """Prontezza media dei 2 Pokemon migliori del proprio lato.

    Solo i 2 migliori: spalmare energia sulla panchina non deve essere
    premiato, ma avere un attaccante di riserva gia' carico si'.
    """
    scores = [
        _best_attack_readiness(p, card_table, attack_table)
        for p in _in_play(ps)
    ]
    if not scores:
        return 0.0
    scores.sort(reverse=True)
    top = scores[:2]
    return sum(top) / len(top)


def _evolution(ps, card_table):
    """Stage medio dei Pokemon in gioco, normalizzato su 2 (stage 2 = 1.0)."""
    pokemon = _in_play(ps)
    if not pokemon:
        return 0.0
    total = 0
    for p in pokemon:
        data = card_table.get(p.id)
        if data is None:
            continue
        if getattr(data, "stage2", False):
            total += 2
        elif getattr(data, "stage1", False):
            total += 1
    return _clamp(total / (2.0 * len(pokemon)), 0.0, 1.0)


def _board_safety(ps):
    n = len(_in_play(ps))
    if n >= 4:
        return 1.0
    return _SAFETY_CURVE.get(n, -1.0)


def _deckout_danger(ps):
    """Quanto si e' vicini a perdere per deck-out, in [0,1].

    Morde solo a mazzo quasi vuoto: a 20 carte vale 0 esattamente come a 60,
    perche' fino a li' non e' una considerazione reale.
    """
    return _clamp(max(0, DECKOUT_HORIZON - ps.deckCount) / DECKOUT_HORIZON, 0.0, 1.0)


def _type_matchup(me, them, card_table):
    """+1 se il mio attivo colpisce la weakness del loro, -1 al contrario."""
    my_active = _active(me)
    their_active = _active(them)
    if my_active is None or their_active is None:
        return 0.0
    mine = card_table.get(my_active.id)
    theirs = card_table.get(their_active.id)
    if mine is None or theirs is None:
        return 0.0
    score = 0.0
    if theirs.weakness is not None and theirs.weakness == mine.energyType:
        score += 1.0
    if mine.weakness is not None and mine.weakness == theirs.energyType:
        score -= 1.0
    return _clamp(score)


def _condition_count(ps):
    return sum(
        1
        for flag in (ps.poisoned, ps.burned, ps.asleep, ps.paralyzed, ps.confused)
        if flag
    )


# ---------------------------------------------------------------------------
# API pubblica
# ---------------------------------------------------------------------------

def phi_components(state, your_index, card_table, attack_table):
    """Le 8 componenti antisimmetriche, ognuna in [-1, 1].

    Restituisce un dict nome -> valore. L'ordine canonico per i tensori e'
    COMPONENT_ORDER (usato dalla testa ausiliaria della rete).
    """
    me = state.players[your_index]
    them = state.players[1 - your_index]

    # prize: differenza dei due avanzamenti, ognuno in [0,1].
    my_prize = _prize_progress(me, them, card_table)
    their_prize = _prize_progress(them, me, card_table)

    # board_safety: ogni lato e' gia' in [-1,1], quindi la differenza va
    # dimezzata per restare nel range.
    safety = (_board_safety(me) - _board_safety(them)) / 2.0

    # deckout: e' un *rischio*, quindi il segno e' invertito (il loro rischio
    # e' un mio vantaggio).
    deckout = _deckout_danger(them) - _deckout_danger(me)

    conditions = (_condition_count(them) - _condition_count(me)) / 3.0

    return {
        "prize": _clamp(my_prize - their_prize),
        "attack_ready": _clamp(
            _attack_ready(me, card_table, attack_table)
            - _attack_ready(them, card_table, attack_table)
        ),
        "energy": _clamp(
            _energy(me, card_table, attack_table)
            - _energy(them, card_table, attack_table)
        ),
        "evolution": _clamp(
            _evolution(me, card_table) - _evolution(them, card_table)
        ),
        "board_safety": _clamp(safety),
        "deckout": _clamp(deckout),
        "type_matchup": _type_matchup(me, them, card_table),
        "conditions": _clamp(conditions),
    }


def phi(state, your_index, card_table, attack_table):
    """Potenziale scalare in [-1, 1] dal punto di vista di `your_index`."""
    comps = phi_components(state, your_index, card_table, attack_table)
    total = sum(WEIGHTS[k] * comps[k] for k in COMPONENT_ORDER)
    return _clamp(total / _WEIGHT_SUM)


def phi_vector(state, your_index, card_table, attack_table):
    """Le componenti come lista ordinata (target della testa ausiliaria)."""
    comps = phi_components(state, your_index, card_table, attack_table)
    return [comps[k] for k in COMPONENT_ORDER]


def shaped_reward(state_before, state_after, your_index, card_table, attack_table, gamma=1.0):
    """Reward di shaping potential-based per una transizione.

    r = gamma * phi(s') - phi(s). Con gamma=1 la somma telescopica lungo un
    episodio vale phi(s_finale) - phi(s_iniziale), quindi non altera il
    ritorno totale se non per una costante: e' esattamente la proprieta' che
    rende lo shaping sicuro.
    """
    p0 = phi(state_before, your_index, card_table, attack_table)
    p1 = phi(state_after, your_index, card_table, attack_table)
    return gamma * p1 - p0
