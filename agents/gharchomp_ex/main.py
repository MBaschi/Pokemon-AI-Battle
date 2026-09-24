"""Cynthia's Garchomp ex / Cynthia's Roserade -- agente euristico a parametri.

Deck plan: Gible -> Gabite -> Garchomp ex e' la linea d'attacco (Corkscrew Dive
per danno sostenibile + ricarica mano, Draconic Buster come colpo unico).
Roselia -> Roserade non attacca mai (Leaf Step chiede un'energia Erba e questa
lista non ne ha nessuna): esiste solo per stare in gioco e dare +30 danni a
ogni Pokemon di Cynthia con Cheer On to Glory. Champion's Call di Gabite e' il
motore di ricerca; Poffin / Fighting Gong / Poke Pad / Hilda la consistenza.

Struttura in tre strati, con una distinzione che e' il punto del refactor:

  DECK  (DeckProfile)  -- *dati* del mazzo: ID carte, linee evolutive, ordini
                          di priorita'. Cambiano quando cambia il mazzo.
  PARAMS (params.json) -- *pesi numerici*. Sono la superficie di ottimizzazione
                          black-box: tune_params.py cerca qui dentro, e nient'
                          altro nel file viene toccato dal tuner.
  policy engine        -- un handler per SelectContext, legge DECK + PARAMS.

Separare i due e' cio' che rende il mazzo trasferibile e i pesi ottimizzabili:
finche' erano mescolati in un unico dataclass non si poteva fare ne' l'una ne'
l'altra cosa.

Due innesti previsti per il futuro, entrambi a ruolo *separato* (mai due
sistemi che classificano la stessa cosa):
  - OpponentModel: cosa abbiamo visto del mazzo avversario;
  - VALUE_HOOK: valutazione appresa delle posizioni, se e quando una rete
    dimostrera' di batterne la stima euristica in A/B.
"""

import json
import os
import time
from dataclasses import dataclass, field

from cg.api import (
    AreaType,
    CardType,
    OptionType,
    SelectContext,
    all_card_data,
    all_attack,
    to_observation_class,
)

# ---------------------------------------------------------------------------
# Card / attack ID (verificati con api.all_card_data() / api.all_attack())
# ---------------------------------------------------------------------------

BASIC_F_ENERGY = 6
ROCK_FIGHTING_ENERGY = 20
GIBLE, GABITE, GARCHOMP_EX = 379, 380, 381
ROSELIA, ROSERADE = 341, 342
SPIRITOMB = 387
BUDDY_BUDDY_POFFIN = 1086
FIGHTING_GONG = 1142
LILLIES_DETERMINATION = 1227
POKE_PAD = 1152
CYNTHIAS_POWER_WEIGHT = 1173
HILDA = 1225
BOSSS_ORDERS = 1182
FOREST_OF_VITALITY = 1261
NIGHT_STRETCHER = 1097
SURFER = 1203
UNFAIR_STAMP = 1080
XEROSICS_MACHINATIONS = 1197

ROCK_HURL = 529          # Gible
DRAGONSLICE = 530        # Gabite
CORKSCREW_DIVE = 531     # Garchomp ex
DRACONIC_BUSTER = 532    # Garchomp ex
SPIKE_STING = 475        # Roselia
LEAF_STEP = 476          # Roserade (irraggiungibile: nessuna energia Erba)
RAGING_CURSE = 540       # Spiritomb

CYNTHIAS_POKEMON_IDS = frozenset({GIBLE, GABITE, GARCHOMP_EX, ROSELIA, ROSERADE, SPIRITOMB})

DECK_LIST = (
    [BASIC_F_ENERGY] * 5 + [ROCK_FIGHTING_ENERGY] * 4
    + [GABITE] * 4 + [GIBLE] * 4 + [ROSELIA] * 4
    + [GARCHOMP_EX] * 3 + [ROSERADE] * 3
    + [SPIRITOMB] * 2
    + [BUDDY_BUDDY_POFFIN] * 4 + [FIGHTING_GONG] * 4 + [LILLIES_DETERMINATION] * 4
    + [POKE_PAD] * 4
    + [CYNTHIAS_POWER_WEIGHT] * 3 + [HILDA] * 3
    + [BOSSS_ORDERS] * 2 + [FOREST_OF_VITALITY] * 2 + [NIGHT_STRETCHER] * 2
    + [SURFER] + [UNFAIR_STAMP] + [XEROSICS_MACHINATIONS]
)
assert len(DECK_LIST) == 60

my_deck = list(DECK_LIST)   # alias per gli harness di benchmark locali


# ---------------------------------------------------------------------------
# PARAMS: la superficie di ottimizzazione black-box.
# Ordine di precedenza: default nel codice < params.json < env GHARCHOMP_PARAMS.
# Stessa convenzione di gio_v1, cosi' il tuner e' condivisibile.
# ---------------------------------------------------------------------------

PARAMS = {
    "prefer_go_first": 1,        # 1 = gioca per primo (nega un turno di setup all'avversario)

    # --- ricerca / fetch: quale carta prendere quando se ne puo' prendere una
    # Base per carta, meno un decadimento per ogni copia gia' in mano o in campo.
    "fetch_gible": 70, "fetch_gabite": 90, "fetch_garchomp": 100,
    "fetch_roselia": 80, "fetch_roserade": 95, "fetch_spiritomb": 20,
    "fetch_basic_energy": 55, "fetch_rock_energy": 60,
    "decay_gible": 12, "decay_gabite": 15, "decay_garchomp": 15,
    "decay_roselia": 20, "decay_roserade": 25, "decay_spiritomb": 15,
    "decay_energy": 10,
    "prereq_penalty": 5,         # tetto se manca lo stadio precedente
    "min_search_score": 0,       # sotto questo, si declina una ricerca opzionale

    # --- giocate dalla mano (conta l'ordine relativo: uno per turno)
    "play_poffin": 700, "play_gong": 650, "play_pokepad": 630,
    "play_hilda": 900, "play_lillie": 950, "play_night_stretcher": 500,
    "play_xerosic": 300, "play_surfer": 250, "play_unfair_stamp": 850,
    "play_forest": 200,
    "lillie_six_prize_bonus": 300,   # a 6 prize Lillie pesca 8 invece di 6

    # --- Boss's Orders
    "boss_ko_score": 900, "boss_disruption_score": 40, "boss_prize_bonus": 100,

    # --- strumenti ed energia
    "tool_base": 400, "tool_active_bonus": 300, "tool_roserade_bonus": 150,
    "energy_base": 500, "energy_active_bonus": 100, "energy_rock_bonus": 5,
    "energy_stuck_active": 350,

    # --- evoluzione / abilita'
    "evolve_base": 900,
    "evolve_active_bonus": 60,         # l'attivo si evolve prima della panchina
    "evolve_stage2_bonus": 40,         # e lo Stage 2 prima dello Stage 1
    "evolve_blocks_ko_penalty": 850,   # non evolvere se rinuncia a un KO sicuro
    "ability_score": 800,

    # --- ritirata
    "free_retreat_dodge": 600,   # Garchomp ex ha costo di ritirata 0
    "paid_retreat_dodge": 300,

    # --- attacco
    "attack_base": 100, "attack_ko_bonus": 3000, "attack_prize_bonus": 500,
    "buster_no_ko_penalty": 500,   # Draconic Buster scarta tutta la sua energia

    # --- promozione dopo un KO
    "promote_garchomp": 500, "promote_garchomp_per_energy": 50,
    "promote_gabite": 200, "promote_gabite_per_energy": 30,
    "promote_gible": 120, "promote_spiritomb": 60, "promote_roselia": 40,
    "promote_never_penalty": 1000,

    # --- consapevolezza della corsa ai prize
    "lethal_urgency_bonus": 2000,   # se questo KO chiude la partita, nient'altro conta
    "ex_exposure_penalty": 250,     # non mandare avanti un ex quando gli basta per vincere
    "opp_ex_target_bonus": 150,     # a parita' d'altro, colpisci i loro ex
}


def _find_file(name):
    """Kaggle esegue main.py via exec(), senza __file__ affidabile e senza
    chdir: si prova la cartella del modulo, la cwd, la root del package cg e
    infine il path fisso della submission."""
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


def _load_params():
    """params.json accanto all'agente, poi l'env var (che ha la precedenza).

    L'env var e' come il tuner inietta un candidato senza scrivere su disco,
    quindi senza che due processi di ottimizzazione paralleli si pestino i piedi.
    """
    path = _find_file("params.json")
    if path:
        try:
            with open(path) as fh:
                PARAMS.update(json.load(fh))
        except (OSError, ValueError):
            pass
    # Nome volutamente generico e non legato al mazzo: lo stesso tune_params.py
    # si copia su un agente nuovo senza modifiche.
    blob = os.environ.get("PTCG_PARAMS")
    if blob:
        try:
            PARAMS.update(json.loads(blob) if blob.strip().startswith("{")
                          else json.load(open(blob)))
        except (OSError, ValueError):
            pass


_load_params()
P = PARAMS


# ---------------------------------------------------------------------------
# Dati statici, caricati una volta all'import (mai per chiamata)
# ---------------------------------------------------------------------------

try:
    CARD_DATA = {c.cardId: c for c in all_card_data()}
    ATTACK_DATA = {a.attackId: a for a in all_attack()}
except BaseException:
    CARD_DATA = {}
    ATTACK_DATA = {}

# Attacchi il cui testo sovrascrive la regola di debolezza/resistenza. Insieme
# piccolo ed *enumerato*: non e' un parser generico del testo.
_IGNORES_WEAKNESS = {RAGING_CURSE}
_IGNORES_RESISTANCE = {ROCK_HURL}
for _aid, _atk in ATTACK_DATA.items():
    _txt = _atk.text or ""
    if "affected by Weakness" in _txt:
        _IGNORES_WEAKNESS.add(_aid)
    if "affected by Resistance" in _txt:
        _IGNORES_RESISTANCE.add(_aid)

PRIZE_TOTAL = 6
ROSERADE_AURA = 30       # fissato dal testo di Cheer On to Glory, non e' un knob


# ---------------------------------------------------------------------------
# Strato 1: DeckProfile -- solo DATI del mazzo (nessun peso numerico)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeckProfile:
    deck_list: tuple
    main_attacker_line: tuple      # (base, stadio1, stadio2)
    aura_support_line: tuple       # (base, stadio1) -- non attacca mai
    filler_basics: tuple
    never_promote_ids: frozenset   # da tenere in panchina se possibile
    attacker_ids: frozenset        # chi puo' davvero attaccare

    lead_priority: tuple           # ordine di preferenza per l'attivo iniziale
    bench_priority: tuple          # ordine per riempire la panchina pre-partita
    max_useful_energy: dict        # energia oltre la quale e' sprecata

    fetch_base_keys: dict = field(default_factory=dict)
    fetch_decay_keys: dict = field(default_factory=dict)
    promote_keys: dict = field(default_factory=dict)
    play_keys: dict = field(default_factory=dict)


DECK = DeckProfile(
    deck_list=tuple(DECK_LIST),
    main_attacker_line=(GIBLE, GABITE, GARCHOMP_EX),
    aura_support_line=(ROSELIA, ROSERADE),
    filler_basics=(SPIRITOMB,),
    never_promote_ids=frozenset({ROSERADE}),
    attacker_ids=frozenset({GIBLE, GABITE, GARCHOMP_EX}),
    # Roselia guida per ultima delle tre: e' il pezzo d'aura, piu' scarso.
    lead_priority=(GIBLE, SPIRITOMB, ROSELIA),
    # La panchina e' sicura: ci si mette presto il pezzo d'aura.
    bench_priority=(GIBLE, ROSELIA, SPIRITOMB),
    max_useful_energy={GIBLE: 1, GABITE: 1, GARCHOMP_EX: 2},
    fetch_base_keys={
        GIBLE: "fetch_gible", GABITE: "fetch_gabite", GARCHOMP_EX: "fetch_garchomp",
        ROSELIA: "fetch_roselia", ROSERADE: "fetch_roserade",
        SPIRITOMB: "fetch_spiritomb",
        BASIC_F_ENERGY: "fetch_basic_energy", ROCK_FIGHTING_ENERGY: "fetch_rock_energy",
    },
    fetch_decay_keys={
        GIBLE: "decay_gible", GABITE: "decay_gabite", GARCHOMP_EX: "decay_garchomp",
        ROSELIA: "decay_roselia", ROSERADE: "decay_roserade",
        SPIRITOMB: "decay_spiritomb",
        BASIC_F_ENERGY: "decay_energy", ROCK_FIGHTING_ENERGY: "decay_energy",
    },
    promote_keys={
        GARCHOMP_EX: ("promote_garchomp", "promote_garchomp_per_energy"),
        GABITE: ("promote_gabite", "promote_gabite_per_energy"),
        GIBLE: ("promote_gible", None),
        SPIRITOMB: ("promote_spiritomb", None),
        ROSELIA: ("promote_roselia", None),
    },
    play_keys={
        BUDDY_BUDDY_POFFIN: "play_poffin", FIGHTING_GONG: "play_gong",
        POKE_PAD: "play_pokepad", HILDA: "play_hilda",
        LILLIES_DETERMINATION: "play_lillie", NIGHT_STRETCHER: "play_night_stretcher",
        XEROSICS_MACHINATIONS: "play_xerosic", SURFER: "play_surfer",
        UNFAIR_STAMP: "play_unfair_stamp", FOREST_OF_VITALITY: "play_forest",
    },
)

# Innesto per una valutazione appresa delle posizioni. Resta None finche' una
# rete non dimostra in A/B di battere la stima euristica: un ruolo separato
# (valutare) da quello dell'euristica (ordinare le mosse), mai i due mescolati.
VALUE_HOOK = None


# ---------------------------------------------------------------------------
# Modello dell'avversario
# ---------------------------------------------------------------------------

class OpponentModel:
    """Cosa abbiamo visto del mazzo avversario in questa partita.

    Un'euristica pura e' cieca all'archetipo di fronte. Questo e' il minimo
    utile: quali Pokemon ha rivelato, quanti prize valgono, e se la sua linea
    principale ci batte per debolezza. Serve anche da base per una
    classificazione di archetipo piu' seria, se in futuro servira'.
    """

    __slots__ = ("seen_ids", "max_hp_seen", "ex_seen")

    def __init__(self):
        self.seen_ids = set()
        self.max_hp_seen = 0
        self.ex_seen = 0

    def observe(self, them):
        for p in [x for x in them.active if x] + [x for x in them.bench if x]:
            if p.id not in self.seen_ids:
                self.seen_ids.add(p.id)
                data = CARD_DATA.get(p.id)
                if data is not None:
                    self.max_hp_seen = max(self.max_hp_seen, data.hp or 0)
                    if getattr(data, "ex", False) or getattr(data, "megaEx", False):
                        self.ex_seen += 1
        for c in them.discard:
            self.seen_ids.add(c.id)

    def biggest_threat_hp(self):
        """HP del bersaglio piu' grosso visto: dice se Draconic Buster basta."""
        return self.max_hp_seen


# ---------------------------------------------------------------------------
# Utilita' di dominio
# ---------------------------------------------------------------------------

def prize_value(pokemon):
    data = CARD_DATA.get(pokemon.id) if pokemon is not None else None
    if data is None:
        return 1
    if getattr(data, "megaEx", False):
        return 3
    if getattr(data, "ex", False):
        return 2
    return 1


def estimate_damage(attack_id, attacker, defender, attacker_is_mine, board):
    """Stima prudente. Applica debolezza (x2) / resistenza (-30), l'aura di
    Cheer On to Glory e il piccolo insieme enumerato di attacchi che ignorano
    debolezza o resistenza. La scalatura di Raging Curse e' l'unico pattern
    riconosciuto per attack id, non per analisi del testo."""
    atk = ATTACK_DATA.get(attack_id)
    if atk is None:
        return 0

    if attack_id == RAGING_CURSE:
        counters = 0
        bench = board.my_bench if attacker_is_mine else board.their_bench
        for p in bench:
            if p is not None and p.id in CYNTHIAS_POKEMON_IDS:
                counters += max(0, p.maxHp - p.hp)
        base = 10 * (counters // 10)
    else:
        base = atk.damage or 0

    if (attacker_is_mine and attacker is not None
            and attacker.id in CYNTHIAS_POKEMON_IDS and board.has_roserade_aura()
            and defender is board.their_active):
        base += ROSERADE_AURA   # +30 prima di debolezza/resistenza, solo sull'attivo

    a_data = CARD_DATA.get(attacker.id) if attacker is not None else None
    d_data = CARD_DATA.get(defender.id) if defender is not None else None
    dmg = base
    if a_data and d_data:
        if (attack_id not in _IGNORES_WEAKNESS and d_data.weakness is not None
                and d_data.weakness == a_data.energyType):
            dmg *= 2
        if (attack_id not in _IGNORES_RESISTANCE and d_data.resistance is not None
                and d_data.resistance == a_data.energyType):
            dmg -= 30
    return max(0, dmg)


def usable_attacks(pokemon, extra_energy=0):
    """(attack_id, attack) che il Pokemon puo' pagare, con `extra_energy` in piu'."""
    data = CARD_DATA.get(pokemon.id) if pokemon is not None else None
    if data is None:
        return []
    have = len(pokemon.energies) + extra_energy
    out = []
    for aid in data.attacks:
        atk = ATTACK_DATA.get(aid)
        if atk is not None and len(atk.energies) <= have:
            out.append((aid, atk))
    return out


def score_attack(board, attacker, attack_id, target):
    """Punteggio di un attacco. Qui vive la corsa ai prize."""
    if attacker is None or target is None or attack_id not in ATTACK_DATA:
        return -1
    dmg = estimate_damage(attack_id, attacker, target, True, board)
    ko = dmg >= target.hp

    if attack_id == DRACONIC_BUSTER and not ko:
        # Scarta tutta la propria energia: senza KO immobilizza l'attaccante
        # per due o tre turni in cambio di niente.
        return -P["buster_no_ko_penalty"]

    score = P["attack_base"] + dmg
    if ko:
        pv = prize_value(target)
        score += P["attack_ko_bonus"] + pv * P["attack_prize_bonus"]
        if getattr(CARD_DATA.get(target.id), "ex", False):
            score += P["opp_ex_target_bonus"]
        # Se questo KO chiude la partita, non c'e' niente di meglio da fare.
        if pv >= board.my_prizes_left():
            score += P["lethal_urgency_bonus"]
    return score


def best_attack_now(board, allow_one_more_energy=True):
    """(attack_id, danno, ko) migliore per l'attivo, o None."""
    active, target = board.my_active, board.their_active
    if active is None or target is None:
        return None
    extra = 0
    if allow_one_more_energy and not board.state.energyAttached:
        if board.hand_count(BASIC_F_ENERGY) + board.hand_count(ROCK_FIGHTING_ENERGY) > 0:
            extra = 1
    best = None
    for aid, _atk in usable_attacks(active, extra):
        s = score_attack(board, active, aid, target)
        if s < 0:
            continue
        dmg = estimate_damage(aid, active, target, True, board)
        if best is None or s > best[3]:
            best = (aid, dmg, dmg >= target.hp, s)
    return best[:3] if best else None


def opponent_can_ko_me(board):
    """Minaccia letale visibile sul nostro attivo il turno prossimo.

    Usa solo l'attivo avversario e un'energia in piu': la sua mano non e'
    visibile, quindi e' per forza una sottostima (documentato)."""
    mine, theirs = board.my_active, board.their_active
    if mine is None or theirs is None:
        return False
    for aid, _atk in usable_attacks(theirs, extra_energy=1):
        if estimate_damage(aid, theirs, mine, False, board) >= mine.hp:
            return True
    return False


# ---------------------------------------------------------------------------
# Board: accesso None-safe all'osservazione
# ---------------------------------------------------------------------------

def _counts(ids):
    d = {}
    for i in ids:
        d[i] = d.get(i, 0) + 1
    return d


class Board:
    def __init__(self, obs, opponent=None):
        self.obs = obs
        self.state = obs.current
        self.my_index = self.state.yourIndex
        self.me = self.state.players[self.my_index]
        self.them = self.state.players[1 - self.my_index]
        self.my_active = self.me.active[0] if self.me.active else None
        self.their_active = self.them.active[0] if self.them.active else None
        self.my_bench = [p for p in self.me.bench if p is not None]
        self.their_bench = [p for p in self.them.bench if p is not None]
        self._hand = _counts([c.id for c in (self.me.hand or [])])
        self._field = _counts(
            [p.id for p in ([self.my_active] if self.my_active else []) + self.my_bench]
        )
        self.opponent = opponent

    def hand_count(self, cid):
        return self._hand.get(cid, 0)

    def field_count(self, cid):
        return self._field.get(cid, 0)

    def have(self, cid):
        return self.hand_count(cid) + self.field_count(cid)

    def bench_free(self):
        return max(0, self.me.benchMax - len(self.my_bench))

    def has_roserade_aura(self):
        return self.field_count(ROSERADE) > 0

    def my_prizes_left(self):
        return len(self.me.prize)

    def their_prizes_left(self):
        return len(self.them.prize)


def get_card(obs, area, index, player_index):
    """Risolve area/index/playerIndex di un'Option nella carta corrispondente."""
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


# ---------------------------------------------------------------------------
# Strato 2: motore di policy, un handler per SelectContext
# ---------------------------------------------------------------------------

def _forced(select):
    n = len(select.option)
    return list(range(n)) if n <= select.maxCount else None


def _rank(scores, descending=True):
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=descending)


def fetch_score(board, card_id):
    """Quanto vale prendere questa carta adesso."""
    base_key = DECK.fetch_base_keys.get(card_id)
    if base_key is None:
        return 0
    have = board.have(card_id)
    score = P[base_key] - have * P[DECK.fetch_decay_keys.get(card_id, "decay_energy")]
    # Non cercare lo stadio successivo prima che esista il precedente.
    cap = P["prereq_penalty"]
    if card_id == GABITE and board.have(GIBLE) == 0:
        score = min(score, cap)
    elif card_id == GARCHOMP_EX and board.have(GABITE) == 0:
        score = min(score, cap)
    elif card_id == ROSERADE and board.have(ROSELIA) == 0:
        score = min(score, cap)
    return score


def promote_score(board, pokemon):
    """Chi mandare attivo. Ricorre a ogni KO e si accumula: merita cura."""
    if pokemon is None:
        return -999
    score = pokemon.hp
    if pokemon.id in DECK.never_promote_ids:
        score -= P["promote_never_penalty"]   # Roserade: l'aura serve in panchina
    keys = DECK.promote_keys.get(pokemon.id)
    if keys:
        flat, per_energy = keys
        score += P[flat]
        if per_energy:
            score += len(pokemon.energies) * P[per_energy]
    # A pochi prize dalla sconfitta, mandare avanti un ex regala la partita.
    if prize_value(pokemon) >= board.their_prizes_left():
        score -= P["ex_exposure_penalty"]
    return score


def opponent_switch_score(board, pokemon):
    """Chi tirare fuori dalla panchina avversaria (Boss's Orders).

    Problema inverso rispetto a promote_score: si preferisce un bersaglio che
    l'attivo puo' mettere KO subito. Valutare la sola disruption senza vedere
    la loro mano e' un giudizio che le regole non sanno dare -- resta a
    punteggio basso di proposito."""
    if pokemon is None:
        return -999
    active = board.my_active
    if active is not None:
        for aid, _atk in usable_attacks(active):
            if aid == DRACONIC_BUSTER:
                continue
            if estimate_damage(aid, active, pokemon, True, board) >= pokemon.hp:
                return P["boss_ko_score"] + prize_value(pokemon) * P["boss_prize_bonus"]
    return prize_value(pokemon) * 50 + max(0, pokemon.maxHp - pokemon.hp) - pokemon.hp // 4


def _handle_yes(obs, board, select):
    for i, o in enumerate(select.option):
        if o.type == OptionType.YES:
            return [i]
    forced = _forced(select)
    return forced if forced is not None else [0]


def _handle_is_first(obs, board, select):
    want_yes = bool(P["prefer_go_first"])
    for i, o in enumerate(select.option):
        if (o.type == OptionType.YES) == want_yes:
            return [i]
    return [0]


def _setup_score(obs, o, priority):
    card = get_card(obs, o.area, o.index, o.playerIndex)
    if card is None:
        return -1
    try:
        return len(priority) - priority.index(card.id)
    except ValueError:
        return 0


def _handle_setup_active(obs, board, select):
    forced = _forced(select)
    if forced is not None:
        return forced
    scores = [_setup_score(obs, o, DECK.lead_priority) for o in select.option]
    return _rank(scores)[:select.maxCount]


def _handle_setup_bench(obs, board, select):
    forced = _forced(select)
    if forced is not None:
        return forced
    scores = [_setup_score(obs, o, DECK.bench_priority) for o in select.option]
    return _rank(scores)[:select.maxCount]


def _handle_to_hand(obs, board, select):
    forced = _forced(select)
    if forced is not None and select.minCount == select.maxCount:
        return forced
    scores = []
    for o in select.option:
        card = get_card(obs, o.area, o.index, o.playerIndex)
        scores.append(fetch_score(board, card.id) if card is not None else -999)
    chosen = []
    for i in _rank(scores):
        if len(chosen) >= select.maxCount:
            break
        if len(chosen) < select.minCount or scores[i] > P["min_search_score"]:
            chosen.append(i)
    return chosen


def _handle_to_bench(obs, board, select):
    forced = _forced(select)
    if forced is not None:
        return forced
    scores = []
    for o in select.option:
        card = get_card(obs, o.area, o.index, o.playerIndex)
        scores.append(fetch_score(board, card.id) if card is not None else -999)
    return _rank(scores)[:select.maxCount]


def _handle_promotion(obs, board, select):
    forced = _forced(select)
    if forced is not None and len(select.option) == 1:
        return forced
    scores = []
    for o in select.option:
        pokemon = get_card(obs, o.area, o.index, o.playerIndex)
        if o.playerIndex == board.my_index:
            scores.append(promote_score(board, pokemon))
        else:
            scores.append(opponent_switch_score(board, pokemon))
    return _rank(scores)[:max(select.maxCount, 1)]


def _handle_discard(obs, board, select):
    forced = _forced(select)
    if forced is not None:
        return forced
    scores = []
    for o in select.option:
        card = get_card(obs, o.area, o.index, o.playerIndex)
        scores.append(fetch_score(board, card.id) if card is not None else 0)
    return _rank(scores, descending=False)[:select.maxCount]   # scarta il meno utile


def _score_energy_attach(board, energy_id, target, is_active):
    max_useful = DECK.max_useful_energy.get(target.id, 0)
    if max_useful <= 0:
        # Roselia/Roserade/Spiritomb non vogliono energia Fighting a lungo, ma
        # se uno di loro resta bloccato attivo senza energia non puo' ne'
        # attaccare ne' pagare la ritirata: resta li' a farsi picchiare.
        data = CARD_DATA.get(target.id)
        if is_active and not target.energies and data and data.retreatCost > 0:
            return P["energy_stuck_active"]
        return -1
    if len(target.energies) >= max_useful:
        return -1
    score = P["energy_base"]
    if is_active:
        score += P["energy_active_bonus"]
    if energy_id == ROCK_FIGHTING_ENERGY:
        score += P["energy_rock_bonus"]   # stessa {F}, in piu' immune agli effetti
    return score


def _score_attach(obs, board, o):
    src = get_card(obs, o.area, o.index, board.my_index)
    target = get_card(obs, o.inPlayArea, o.inPlayIndex, board.my_index)
    if src is None or target is None:
        return -1
    data = CARD_DATA.get(src.id)
    if data is None:
        return -1
    # Si usa inPlayArea dell'opzione, non il confronto fra id: due copie della
    # stessa specie sarebbero indistinguibili.
    is_active = (o.inPlayArea == AreaType.ACTIVE)
    if data.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
        return _score_energy_attach(board, src.id, target, is_active)
    if data.cardType == CardType.TOOL:
        if target.id not in CYNTHIAS_POKEMON_IDS or target.tools:
            return -1
        score = P["tool_base"]
        if is_active:
            score += P["tool_active_bonus"]
        elif target.id == ROSERADE:
            score += P["tool_roserade_bonus"]
        return score
    return -1


def _score_evolve(obs, board, o):
    src = get_card(obs, o.area, o.index, board.my_index)
    if src is None:
        return -1
    if o.inPlayArea == AreaType.ACTIVE and src.id == ROSERADE:
        # Roserade non puo' attaccare in questa lista: evolvere l'attivo in lei
        # lascerebbe lo slot senza offesa mentre continua a incassare.
        return -1

    score = P["evolve_base"]
    # Senza queste due preferenze tutte le evoluzioni valgono uguale e il
    # pareggio si risolve per indice dell'opzione, cioe' a caso: nelle tracce
    # l'attivo restava Gible per interi turni mentre si evolveva la panchina.
    if o.inPlayArea == AreaType.ACTIVE:
        score += P["evolve_active_bonus"]   # l'attivo attacca e incassa: viene prima
    if src.id == GARCHOMP_EX:
        score += P["evolve_stage2_bonus"]   # arrivare allo Stage 2 e' il piano del mazzo

    if o.inPlayArea == AreaType.ACTIVE:
        plan = best_attack_now(board)
        if plan and plan[2]:
            score -= P["evolve_blocks_ko_penalty"]   # non rinunciare a un KO sicuro
    return score


def _ready_bench_attacker(board):
    """Un attaccante in panchina gia' carico, pronto a colpire se promosso."""
    target = board.their_active
    if target is None:
        return None
    best = None
    for p in board.my_bench:
        if p.id not in DECK.attacker_ids:
            continue
        for aid, _atk in usable_attacks(p):
            if aid == DRACONIC_BUSTER and estimate_damage(aid, p, target, True, board) < target.hp:
                continue
            if best is None or p.id == GARCHOMP_EX:
                best = p
    return best


def _score_retreat(board):
    active = board.my_active
    if active is None or not board.my_bench:
        return -1
    threatened = opponent_can_ko_me(board)
    plan = best_attack_now(board)
    i_can_ko = bool(plan and plan[2])

    # Reattivo: schivare una minaccia letale a cui non si sa rispondere.
    if threatened and not i_can_ko:
        if active.id == GARCHOMP_EX:
            return P["free_retreat_dodge"]   # costo di ritirata 0: schivare e' gratis
        if any(p.id in (GARCHOMP_EX, GABITE) for p in board.my_bench):
            return P["paid_retreat_dodge"]

    # Proattivo: l'attivo non ha nulla da fare ma in panchina c'e' un
    # attaccante gia' carico. Senza questo l'energia si accumula inutilizzata
    # (era il primo motivo di sconfitta nei benchmark locali).
    if (plan is None or plan[1] <= 0) and _ready_bench_attacker(board) is not None:
        return P["free_retreat_dodge"] if active.id == GARCHOMP_EX else P["paid_retreat_dodge"]
    return -1


def _score_boss_orders(board):
    plan = best_attack_now(board)
    if plan and plan[2]:
        return P["boss_disruption_score"]   # il KO sul loro attivo c'e' gia'
    active = board.my_active
    if active is None:
        return -1
    for opp in board.their_bench:
        for aid, _atk in usable_attacks(active):
            if aid == DRACONIC_BUSTER:
                continue
            if estimate_damage(aid, active, opp, True, board) >= opp.hp:
                return P["boss_ko_score"] + prize_value(opp) * P["boss_prize_bonus"]
    return P["boss_disruption_score"]


def _score_play(obs, board, o):
    card = get_card(obs, AreaType.HAND, o.index, board.my_index)
    if card is None:
        return -1
    data = CARD_DATA.get(card.id)
    if data is None:
        return 0
    if data.cardType == CardType.POKEMON:
        return -1 if board.bench_free() <= 0 else fetch_score(board, card.id)
    if card.id == LILLIES_DETERMINATION:
        score = P["play_lillie"]
        if board.my_prizes_left() == PRIZE_TOTAL:
            score += P["lillie_six_prize_bonus"]   # a 6 prize pesca 8 invece di 6
        return score
    if card.id == BOSSS_ORDERS:
        return _score_boss_orders(board)
    if card.id == BUDDY_BUDDY_POFFIN and board.bench_free() <= 0:
        return -1
    key = DECK.play_keys.get(card.id)
    return P[key] if key else 100


def _score_main(obs, board, o):
    t = o.type
    if t == OptionType.PLAY:
        return _score_play(obs, board, o)
    if t == OptionType.ATTACH:
        return _score_attach(obs, board, o)
    if t == OptionType.EVOLVE:
        return _score_evolve(obs, board, o)
    if t == OptionType.ABILITY:
        return P["ability_score"]      # Champion's Call: valore puro, una volta per turno
    if t == OptionType.RETREAT:
        return _score_retreat(board)
    if t == OptionType.ATTACK:
        return score_attack(board, board.my_active, o.attackId, board.their_active)
    if t == OptionType.DISCARD:
        return -1
    if t == OptionType.END:
        return 0
    return 0


def _handle_main(obs, board, select):
    scores = [_score_main(obs, board, o) for o in select.option]
    if select.maxCount <= 0:
        return []
    return _rank(scores)[:1]


def _handle_attack(obs, board, select):
    active, target = board.my_active, board.their_active
    if active is None or target is None:
        forced = _forced(select)
        return forced if forced is not None else list(range(min(select.maxCount, len(select.option))))
    scores = [score_attack(board, active, o.attackId, target) for o in select.option]
    return _rank(scores)[:max(select.maxCount, 1)]


def _handle_generic(obs, board, select):
    forced = _forced(select)
    if forced is not None:
        return forced
    if select.option and select.option[0].type == OptionType.NUMBER:
        ranked = sorted(range(len(select.option)),
                        key=lambda i: select.option[i].number or 0, reverse=True)
        return ranked[:max(select.maxCount, 1)]
    n = min(select.maxCount, len(select.option))
    return list(range(n)) if n > 0 else []


HANDLERS = {
    int(SelectContext.MAIN): _handle_main,
    int(SelectContext.SETUP_ACTIVE_POKEMON): _handle_setup_active,
    int(SelectContext.SETUP_BENCH_POKEMON): _handle_setup_bench,
    int(SelectContext.SWITCH): _handle_promotion,
    int(SelectContext.TO_ACTIVE): _handle_promotion,
    int(SelectContext.TO_HAND): _handle_to_hand,
    int(SelectContext.TO_BENCH): _handle_to_bench,
    int(SelectContext.TO_FIELD): _handle_to_bench,
    int(SelectContext.DISCARD): _handle_discard,
    int(SelectContext.ATTACK): _handle_attack,
    int(SelectContext.IS_FIRST): _handle_is_first,
    int(SelectContext.MULLIGAN): _handle_yes,
    int(SelectContext.COIN_HEAD): _handle_yes,
    int(SelectContext.ACTIVATE): _handle_yes,
}


# ---------------------------------------------------------------------------
# Budget temporale e stato di partita
# ---------------------------------------------------------------------------

_SOFT_DEADLINE = 480.0   # ampio margine sui 600 s/partita: il timeout e' sconfitta
_STATE = {"spent": 0.0, "opponent": OpponentModel()}


def _reset_if_new_match(state):
    """Lo scope di modulo persiste fra le chiamate ma non deve sopravvivere fra
    partite: si azzera all'inizio del match."""
    if state is not None and state.turn is not None and state.turn <= 1:
        if _STATE["spent"] > 0.5:
            _STATE["spent"] = 0.0
            _STATE["opponent"] = OpponentModel()


def _sanitize(result, select):
    """Garantisce minCount <= len <= maxCount, indici unici e in range."""
    n = len(select.option)
    out = []
    for i in result:
        if isinstance(i, int) and 0 <= i < n and i not in out:
            out.append(i)
    if len(out) > select.maxCount:
        out = out[:select.maxCount]
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

    if _STATE["spent"] > _SOFT_DEADLINE:
        n = min(select.maxCount, len(select.option))
        return list(range(n)) if n > 0 else []

    board = Board(obs, _STATE["opponent"])
    _STATE["opponent"].observe(board.them)

    handler = HANDLERS.get(int(select.context), _handle_generic)
    return _sanitize(handler(obs, board, select), select)


def agent(obs_dict: dict) -> list:
    t0 = time.perf_counter()
    try:
        return _agent_impl(obs_dict)
    except BaseException:
        return _fallback(obs_dict)
    finally:
        try:
            _STATE["spent"] += time.perf_counter() - t0
        except BaseException:
            pass


# ---------------------------------------------------------------------------
# SELF_CHECK
# ---------------------------------------------------------------------------
# - Selezione mazzo (select is None) -> 60 card ID: _agent_impl, primo ramo.
# - Cardinalita': _sanitize() su ogni ritorno; taglia a maxCount, riempie fino a
#   minCount, scarta duplicati e indici fuori range.
# - Mai solleva: agent() avvolge tutto in try/except BaseException; _fallback()
#   e' a sua volta protetto e degrada a [].
# - Budget: _STATE["spent"] accumula fra le chiamate; oltre _SOFT_DEADLINE si
#   salta ogni scoring e si torna la risposta legale piu' economica.
# - Scope per-partita: _reset_if_new_match() azzera tempo e OpponentModel a turn<=1.
# - Import: solo stdlib + cg.api.
# - Determinismo: nessuna casualita'; pareggi risolti dall'ordinamento stabile
#   (indice piu' basso).
# - Handler espliciti, mai il fallback generico, per: MULLIGAN, IS_FIRST,
#   COIN_HEAD, SETUP_ACTIVE_POKEMON, SETUP_BENCH_POKEMON, MAIN, ATTACK, DISCARD,
#   TO_HAND, SWITCH/TO_ACTIVE, ACTIVATE.
