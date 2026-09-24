"""Marnie's Grimmsnarl ex / Munkidori -- STRATO EURISTICO di marnie_grimmsnarl_ex_v2.

Copia (volutamente autonoma, perche' una submission Kaggle deve stare in una
sola cartella) del motore di policy di `agents/marnie_grimmsnarl_ex/main.py`,
con in piu' l'API che serve allo strato di ricerca:

    make_board(obs, opponent)      -> Board
    score_options(obs, board, sel) -> list[float] | None   punteggi per opzione
    decide(obs, board, select)     -> list[int]            la risposta euristica
    heuristic_agent(obs_dict)      -> list[int]            l'agente puro (rung 2)

`score_options` restituisce None nei contesti che non sono un ranking a scelta
singola: li' non esiste una nozione utile di "seconda scelta" e la ricerca non
interviene.

Regola invariante di tutto l'agente v2: **questo file decide sempre**. La rete
puo' solo riordinare le opzioni che l'euristica ha gia' giudicato equivalenti.
Il pavimento e' quindi l'euristica, e la rete puo' solo aggiungere.

Deck plan: Impidimp -> (Rare Candy) -> Marnie's Grimmsnarl ex e' tutta l'offesa.
Shadow Bullet fa 180 sull'attivo + 30 su una panchina avversaria, da dietro un
muro di 320 HP; Punk Up (l'abilita' che scatta *evolvendo* in Grimmsnarl ex)
cerca fino a 5 Basic {D} nel mazzo e le distribuisce sui Pokemon di Marnie, ed
e' l'unico motivo per cui il mazzo sta in piedi con sole 10 energie.

Munkidori e Froslass sono il secondo motore, e nessuno dei due attacca mai:

  - Froslass (Freezing Shroud) mette 1 segnalino su OGNI Pokemon con abilita' a
    ogni checkup, i miei compresi. Su di me e' danno voluto: e' la benzina di
    Munkidori.
  - Munkidori (Adrena-Brain, richiede una {D} attaccata) sposta fino a 3
    segnalini da un mio Pokemon a uno loro. Il danno che Froslass mi mette
    addosso torna indietro moltiplicato in pressione.

Verificato sull'engine, non sul testo delle carte:

  - Mind Bend di Munkidori costa [PSYCHIC, COLORLESS] e in lista non c'e'
    nessuna energia Psichica: **Munkidori non puo' mai attaccare**.
  - Chilly di Snorunt e Frost Smash di Froslass costano {W}: nessuna energia
    Acqua in lista, **nemmeno loro possono mai attaccare**.
  - La ritirata *scarta* l'energia (contesto DISCARD_ENERGY con
    remainEnergyCost): ritirare Grimmsnarl ex butta via 2 {D} su 10 totali.
    Ritirarlo e' quasi sempre sbagliato -- e' un muro, deve incassare.

Struttura in tre strati, come gharchomp_ex:

  DECK  (DeckProfile)  -- *dati* del mazzo: ID, linee evolutive, priorita'.
  PARAMS (params.json) -- *pesi numerici*, unica superficie che il tuner tocca.
  policy engine        -- un handler per SelectContext, legge DECK + PARAMS.

In v1 questo file finiva con "VALUE_HOOK resta None: l'euristica ordina le
mosse, una rete valutera' le posizioni". In v2 quella rete esiste (model.py) ma
il principio non cambia: **ordina l'euristica, valuta la rete**. Mai due
sistemi che classificano la stessa cosa.
"""

import json
import os
import time
from dataclasses import dataclass, field

from cg.api import (
    AreaType,
    CardType,
    EnergyType,
    OptionType,
    SelectContext,
    all_card_data,
    all_attack,
    to_observation_class,
)

# ---------------------------------------------------------------------------
# Card / attack ID (verificati con all_card_data() / all_attack())
# ---------------------------------------------------------------------------

D_ENERGY = 7                     # Basic {D} Energy -- l'unica energia in lista

IMPIDIMP, MORGREM, GRIMMSNARL_EX = 646, 647, 648
MUNKIDORI = 112
SNORUNT, FROSLASS = 860, 104

BUDDY_BUDDY_POFFIN = 1086
LILLIES_DETERMINATION = 1227
POKE_PAD = 1152
SPIKEMUTH_GYM = 1259
TEAM_ROCKETS_PETREL = 1219
NIGHT_STRETCHER = 1097
RARE_CANDY = 1079
BOSSS_ORDERS = 1182
DAWN = 1231
POKEGEAR = 1122
TOOL_SCRAPPER = 1137
UNFAIR_STAMP = 1080

FILCH = 934              # Impidimp, costo [COLORLESS], 0 danni, pesca 1
CORKSCREW_PUNCH_1 = 935  # Impidimp, costo [D], 10
CORKSCREW_PUNCH_2 = 936  # Morgrem, costo [D,D], 60
SHADOW_BULLET = 937      # Grimmsnarl ex, costo [D,D], 180 + 30 a una panchina
MIND_BEND = 141          # Munkidori -- costo {P}: irraggiungibile in questa lista
FROST_SMASH = 131        # Froslass -- costo {W}: idem
CHILLY = 1239            # Snorunt  -- costo {W}: idem

# I Pokemon "di Marnie": gli unici bersagli legali di Punk Up.
MARNIE_POKEMON_IDS = frozenset({IMPIDIMP, MORGREM, GRIMMSNARL_EX})
# Chi ha un'abilita': sono i Pokemon che Freezing Shroud colpisce ogni checkup.
ABILITY_POKEMON_IDS = frozenset({MUNKIDORI, FROSLASS, GRIMMSNARL_EX})

DECK_LIST = (
    [D_ENERGY] * 10
    + [IMPIDIMP] * 4 + [MUNKIDORI] * 4
    + [GRIMMSNARL_EX] * 3 + [MORGREM] * 3
    + [FROSLASS] * 2 + [SNORUNT] * 2
    + [BUDDY_BUDDY_POFFIN] * 4 + [LILLIES_DETERMINATION] * 4
    + [POKE_PAD] * 4 + [SPIKEMUTH_GYM] * 4 + [TEAM_ROCKETS_PETREL] * 4
    + [NIGHT_STRETCHER] * 3 + [RARE_CANDY] * 3
    + [BOSSS_ORDERS] * 2
    + [DAWN] + [POKEGEAR] + [TOOL_SCRAPPER] + [UNFAIR_STAMP]
)
assert len(DECK_LIST) == 60

my_deck = list(DECK_LIST)   # alias per gli harness di benchmark locali


# ---------------------------------------------------------------------------
# PARAMS: la superficie di ottimizzazione black-box.
# Ordine di precedenza: default nel codice < params.json < env PTCG_PARAMS.
# ---------------------------------------------------------------------------

PARAMS = {
    "prefer_go_first": 1,        # nega all'avversario un turno di setup

    # --- ricerca / fetch: quanto vale prendere questa carta adesso.
    # Base per carta, meno un decadimento per ogni copia gia' in mano o in campo.
    "fetch_impidimp": 85, "fetch_morgrem": 60, "fetch_grimmsnarl": 100,
    "fetch_munkidori": 75, "fetch_snorunt": 45, "fetch_froslass": 50,
    "fetch_energy": 55,
    "fetch_rare_candy": 95, "fetch_poffin": 80, "fetch_lillie": 70,
    "fetch_pokepad": 55, "fetch_spikemuth": 50, "fetch_petrel": 60,
    "fetch_night_stretcher": 45, "fetch_boss": 65, "fetch_dawn": 60,
    "fetch_pokegear": 30, "fetch_tool_scrapper": 10, "fetch_unfair_stamp": 40,
    "decay_impidimp": 18, "decay_morgrem": 22, "decay_grimmsnarl": 30,
    "decay_munkidori": 30, "decay_snorunt": 25, "decay_froslass": 25,
    "decay_energy": 8, "decay_trainer": 20,
    "prereq_penalty": 5,         # tetto se manca il pezzo che la rende giocabile
    "min_search_score": 0,       # sotto questo si declina una ricerca opzionale
    "fetch_candy_combo_bonus": 60,   # Rare Candy vale molto di piu' con l'ex in mano
    "fetch_grimmsnarl_ready_bonus": 60,  # e l'ex vale di piu' con la Candy in mano

    # --- giocate dalla mano (conta solo l'ordine relativo: una per iterazione)
    "play_poffin": 700, "play_rare_candy": 1100, "play_lillie": 640,
    "play_petrel": 600, "play_pokepad": 580, "play_spikemuth": 560,
    "play_night_stretcher": 520, "play_dawn": 660, "play_pokegear": 480,
    "play_unfair_stamp": 820, "play_tool_scrapper": 300,
    "play_basic_pokemon": 750,
    "lillie_six_prize_bonus": 200,   # a 6 prize pesca 8 invece di 6
    "lillie_full_hand_penalty": 250, # con la mano gia' piena rimescola valore

    # --- Boss's Orders
    "boss_ko_score": 900, "boss_disruption_score": 40, "boss_prize_bonus": 100,

    # --- energia (attacco manuale, uno per turno)
    "energy_base": 500,
    "energy_enables_attack": 400,   # completa il costo dell'attacco dell'attivo
    "energy_munkidori": 350,        # 1 sola {D} accende Adrena-Brain
    "energy_bench_attacker": 60,    # precarico il prossimo Grimmsnarl
    "energy_active_bonus": 80,
    "energy_stuck_active": 300,     # anti-softlock: un attivo senza energia e con
                                    # costo di ritirata > 0 non puo' fare nulla
    "tool_base": 400, "tool_active_bonus": 100,

    # --- Punk Up (evolvendo in Grimmsnarl ex: fino a 5 {D} dal mazzo)
    "punk_up_count": 5,             # quante energie prendere (0..5)
    "punk_active_attacker": 900,    # prima l'attivo che deve colpire adesso
    "punk_bench_grimmsnarl": 600,   # poi un ex di riserva gia' in panchina
    "punk_bench_preload": 300,      # poi Impidimp/Morgrem, il prossimo attaccante
    "punk_overflow": 10,            # oltre il costo dell'attacco: quasi inutile

    # --- evoluzione
    "evolve_base": 900,
    "evolve_active_bonus": 60,      # l'attivo attacca e incassa: si evolve prima
    "evolve_stage2_bonus": 400,     # e Grimmsnarl ex prima di qualunque altra cosa
    "evolve_froslass_bonus": 40,    # Froslass e' motore, non riempitivo
    "evolve_blocks_ko_penalty": 850,   # non rinunciare a un KO gia' disponibile
    "evolve_blocks_candy_penalty": 700,  # Rare Candy vuole un Basic: evolvere in
                                    # Morgrem distrugge il bersaglio della combo
    "evolve_no_attack_penalty": 500,   # evolvere l'attivo in un corpo che questo
                                    # turno non paga il proprio attacco, mentre
                                    # quello attuale attaccava, e' un turno perso

    # --- abilita'
    "ability_munkidori": 1500,      # sopra l'attacco: attaccare chiude il turno
    "ability_spikemuth": 1400,
    "ability_generic": 1200,

    # --- ritirata (in questo engine la ritirata SCARTA l'energia)
    "retreat_useless_active": 600,  # attivo che non sa attaccare + panchina pronta
    "retreat_dodge": 150,           # schivare un KO: raramente vale l'energia
    "retreat_cost_penalty": 220,    # per ogni energia che la ritirata butta via

    # --- attacco
    "attack_main_base": 300,        # in MAIN sta sotto ogni giocata utile:
                                    # attaccare chiude il turno, quindi per ultimo
    "attack_base": 100, "attack_ko_bonus": 3000, "attack_prize_bonus": 500,
    "filch_draw_value": 45,         # pescare vale piu' dei 10 di Corkscrew Punch
    "lethal_urgency_bonus": 4000,   # se il KO chiude la partita, nient'altro conta

    # --- promozione dopo un KO
    "promote_grimmsnarl": 600, "promote_grimmsnarl_per_energy": 60,
    "promote_morgrem": 250, "promote_morgrem_per_energy": 40,
    "promote_impidimp": 180, "promote_impidimp_per_energy": 20,
    "promote_snorunt": 60, "promote_froslass": 40,
    "promote_munkidori": 0,
    "promote_never_penalty": 400,   # Munkidori non attacca: e' un pezzo da panchina

    # --- Adrena-Brain / Shadow Bullet: dove mettere e togliere i segnalini
    "counter_ko_bonus": 2000,       # i segnalini bastano a mettere KO
    "counter_enables_ko": 1200,     # portano il loro attivo in raggio di Shadow Bullet
    "counter_prize_bonus": 120,     # a parita' d'altro, colpisci i loro ex
    "counter_low_hp_weight": 1,     # poi il bersaglio piu' malridotto
    "remove_from_active": 200,      # curare l'attaccante attivo viene prima
    "remove_full_transfer": 300,    # sorgente con >=3 segnalini: sposta il massimo
    "splash_ko_bonus": 2000,        # i 30 di Shadow Bullet chiudono un panchinaro

    # --- consapevolezza della corsa ai prize
    "ex_exposure_penalty": 250,     # non esporre un ex quando gli basta per vincere
    "opp_ex_target_bonus": 150,
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
    cosi' due processi di ottimizzazione paralleli non si pestano i piedi.
    Il nome e' generico e non legato al mazzo: lo stesso tune_params.py si copia
    su un agente nuovo senza modifiche."""
    path = _find_file("params.json")
    if path:
        try:
            with open(path) as fh:
                PARAMS.update(json.load(fh))
        except (OSError, ValueError):
            pass
    blob = os.environ.get("PTCG_PARAMS")
    if blob:
        try:
            PARAMS.update(json.loads(blob) if blob.strip().startswith("{")
                          else json.load(open(blob)))
        except (OSError, ValueError):
            pass


_load_params()
P = PARAMS

# Traccia diagnostica, spenta salvo env var: serve a guardare cosa fa davvero
# l'agente turno per turno. Nessuna scrittura su file, solo stdout.
_TRACE = bool(os.environ.get("GRIMM_TRACE"))


# ---------------------------------------------------------------------------
# Dati statici, caricati una volta all'import (mai per chiamata)
# ---------------------------------------------------------------------------

try:
    CARD_DATA = {c.cardId: c for c in all_card_data()}
    ATTACK_DATA = {a.attackId: a for a in all_attack()}
except BaseException:
    CARD_DATA = {}
    ATTACK_DATA = {}

# Attacchi il cui testo sovrascrive debolezza/resistenza. Insieme piccolo ed
# *enumerato*: non e' un parser generico del testo.
_IGNORES_WEAKNESS = set()
_IGNORES_RESISTANCE = set()
for _aid, _atk in ATTACK_DATA.items():
    _txt = _atk.text or ""
    if "affected by Weakness" in _txt:
        _IGNORES_WEAKNESS.add(_aid)
    if "affected by Resistance" in _txt:
        _IGNORES_RESISTANCE.add(_aid)

PRIZE_TOTAL = 6
SHADOW_BULLET_SPLASH = 30   # fissato dal testo dell'attacco, non e' un knob
ADRENA_BRAIN_MAX = 3        # "up to 3 damage counters" = 30 danni
FROSLASS_TICK = 10          # 1 segnalino per checkup su ogni Pokemon con abilita'


# ---------------------------------------------------------------------------
# Strato 1: DeckProfile -- solo DATI del mazzo (nessun peso numerico)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeckProfile:
    deck_list: tuple
    main_attacker_line: tuple      # (base, stadio1, stadio2)
    engine_ids: frozenset          # pezzi da panchina che non attaccano mai
    attacker_ids: frozenset        # chi puo' davvero attaccare in questa lista
    marnie_ids: frozenset          # bersagli legali di Punk Up
    never_promote_ids: frozenset

    lead_priority: tuple           # ordine di preferenza per l'attivo iniziale
    bench_priority: tuple          # ordine per riempire la panchina pre-partita
    max_useful_energy: dict        # energia oltre la quale e' sprecata
    attack_cost: dict              # energie necessarie all'attacco principale

    fetch_base_keys: dict = field(default_factory=dict)
    fetch_decay_keys: dict = field(default_factory=dict)
    promote_keys: dict = field(default_factory=dict)
    play_keys: dict = field(default_factory=dict)


DECK = DeckProfile(
    deck_list=tuple(DECK_LIST),
    main_attacker_line=(IMPIDIMP, MORGREM, GRIMMSNARL_EX),
    # Nessuno dei tre puo' pagare il proprio attacco con le energie di questa
    # lista: esistono solo per le abilita' (o, Snorunt, per diventare Froslass).
    engine_ids=frozenset({MUNKIDORI, SNORUNT, FROSLASS}),
    attacker_ids=frozenset({IMPIDIMP, MORGREM, GRIMMSNARL_EX}),
    marnie_ids=MARNIE_POKEMON_IDS,
    # Munkidori non attacca: attivo e' inerte, e in panchina e' il motore.
    never_promote_ids=frozenset({MUNKIDORI}),
    # Impidimp guida sempre: e' l'unico Basic che sa attaccare ed e' il corpo su
    # cui Rare Candy fa scendere Grimmsnarl ex. Snorunt prima di Munkidori
    # perche' e' il piu' sacrificabile dei due.
    lead_priority=(IMPIDIMP, SNORUNT, MUNKIDORI),
    bench_priority=(IMPIDIMP, MUNKIDORI, SNORUNT),
    # 2 = il costo di Shadow Bullet / Corkscrew Punch di Morgrem. Impidimp ne
    # tiene 2 perche' e' il corpo del prossimo Grimmsnarl. Munkidori 1: la {D}
    # che accende Adrena-Brain (e che gli paga la ritirata).
    max_useful_energy={GRIMMSNARL_EX: 2, MORGREM: 2, IMPIDIMP: 2, MUNKIDORI: 1},
    attack_cost={GRIMMSNARL_EX: 2, MORGREM: 2, IMPIDIMP: 1},
    fetch_base_keys={
        IMPIDIMP: "fetch_impidimp", MORGREM: "fetch_morgrem",
        GRIMMSNARL_EX: "fetch_grimmsnarl", MUNKIDORI: "fetch_munkidori",
        SNORUNT: "fetch_snorunt", FROSLASS: "fetch_froslass",
        D_ENERGY: "fetch_energy",
        RARE_CANDY: "fetch_rare_candy", BUDDY_BUDDY_POFFIN: "fetch_poffin",
        LILLIES_DETERMINATION: "fetch_lillie", POKE_PAD: "fetch_pokepad",
        SPIKEMUTH_GYM: "fetch_spikemuth", TEAM_ROCKETS_PETREL: "fetch_petrel",
        NIGHT_STRETCHER: "fetch_night_stretcher", BOSSS_ORDERS: "fetch_boss",
        DAWN: "fetch_dawn", POKEGEAR: "fetch_pokegear",
        TOOL_SCRAPPER: "fetch_tool_scrapper", UNFAIR_STAMP: "fetch_unfair_stamp",
    },
    fetch_decay_keys={
        IMPIDIMP: "decay_impidimp", MORGREM: "decay_morgrem",
        GRIMMSNARL_EX: "decay_grimmsnarl", MUNKIDORI: "decay_munkidori",
        SNORUNT: "decay_snorunt", FROSLASS: "decay_froslass",
        D_ENERGY: "decay_energy",
    },
    promote_keys={
        GRIMMSNARL_EX: ("promote_grimmsnarl", "promote_grimmsnarl_per_energy"),
        MORGREM: ("promote_morgrem", "promote_morgrem_per_energy"),
        IMPIDIMP: ("promote_impidimp", "promote_impidimp_per_energy"),
        SNORUNT: ("promote_snorunt", None),
        FROSLASS: ("promote_froslass", None),
        MUNKIDORI: ("promote_munkidori", None),
    },
    play_keys={
        BUDDY_BUDDY_POFFIN: "play_poffin", RARE_CANDY: "play_rare_candy",
        LILLIES_DETERMINATION: "play_lillie", TEAM_ROCKETS_PETREL: "play_petrel",
        POKE_PAD: "play_pokepad", SPIKEMUTH_GYM: "play_spikemuth",
        NIGHT_STRETCHER: "play_night_stretcher", DAWN: "play_dawn",
        POKEGEAR: "play_pokegear", UNFAIR_STAMP: "play_unfair_stamp",
        TOOL_SCRAPPER: "play_tool_scrapper",
    },
)

# Innesto per una valutazione appresa delle posizioni. Resta None finche' una
# rete non dimostra in A/B di battere la stima euristica: ruolo *separato* da
# quello dell'euristica, che ordina le mosse.
VALUE_HOOK = None


# ---------------------------------------------------------------------------
# Modello dell'avversario
# ---------------------------------------------------------------------------

class OpponentModel:
    """Cosa abbiamo visto del mazzo avversario in questa partita.

    Il minimo utile: quali Pokemon ha rivelato, quanto sono grossi, e quanti ex
    ha mostrato. Serve a sapere se 180 bastano o se servono due colpi."""

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


def estimate_damage(attack_id, attacker, defender, board, is_bench_target=False):
    """Stima prudente: debolezza (x2), resistenza (-30), e il piccolo insieme
    enumerato di attacchi che le ignorano.

    `is_bench_target` copre gli 30 di rimbalzo di Shadow Bullet, che il testo
    esclude esplicitamente da debolezza e resistenza."""
    atk = ATTACK_DATA.get(attack_id)
    if atk is None:
        return 0
    if is_bench_target:
        return SHADOW_BULLET_SPLASH if attack_id == SHADOW_BULLET else 0

    dmg = atk.damage or 0
    a_data = CARD_DATA.get(attacker.id) if attacker is not None else None
    d_data = CARD_DATA.get(defender.id) if defender is not None else None
    if a_data and d_data:
        if (attack_id not in _IGNORES_WEAKNESS and d_data.weakness is not None
                and d_data.weakness == a_data.energyType):
            dmg *= 2
        if (attack_id not in _IGNORES_RESISTANCE and d_data.resistance is not None
                and d_data.resistance == a_data.energyType):
            dmg -= 30
    return max(0, dmg)


def _energy_units(pokemon):
    """Quante energie contano per pagare un costo. `energies` e' la lista dei
    tipi gia' espansa dall'engine, quindi basta la lunghezza."""
    return len(pokemon.energies) if pokemon is not None else 0


def usable_attacks(pokemon, extra_energy=0):
    """(attack_id, attack) che il Pokemon puo' pagare, con `extra_energy` in piu'.

    Il controllo e' sul *tipo*, non solo sul numero: e' l'errore che rende
    Munkidori, Snorunt e Froslass inutili offensivamente in questa lista, e
    contarli come attaccanti falserebbe ogni decisione a valle."""
    data = CARD_DATA.get(pokemon.id) if pokemon is not None else None
    if data is None:
        return []
    have = list(pokemon.energies) + [EnergyType.DARKNESS] * extra_energy
    out = []
    for aid in data.attacks:
        atk = ATTACK_DATA.get(aid)
        if atk is None:
            continue
        pool = list(have)
        ok = True
        # Prima i costi colorati (che vincolano), poi i {C} che accettano tutto.
        for need in sorted(atk.energies, key=lambda e: 0 if e != EnergyType.COLORLESS else 1):
            if need == EnergyType.COLORLESS:
                if not pool:
                    ok = False
                    break
                pool.pop()
            else:
                match = next((e for e in pool
                              if e == need or e == EnergyType.RAINBOW
                              or (e == EnergyType.TEAM_ROCKET
                                  and need in (EnergyType.PSYCHIC, EnergyType.DARKNESS))), None)
                if match is None:
                    ok = False
                    break
                pool.remove(match)
        if ok:
            out.append((aid, atk))
    return out


def can_ever_attack(card_id):
    """Un Pokemon di questa lista sa fare qualcosa con le energie che il mazzo
    contiene davvero? (Munkidori/Snorunt/Froslass: no.)"""
    return card_id in DECK.attacker_ids


def score_attack(board, attacker, attack_id, target):
    """Punteggio di un attacco fra due attacchi disponibili. Qui vive la corsa
    ai prize."""
    if attacker is None or target is None or attack_id not in ATTACK_DATA:
        return -1
    dmg = estimate_damage(attack_id, attacker, target, board)
    ko = dmg >= target.hp

    score = P["attack_base"] + dmg
    if attack_id == FILCH:
        # Filch fa 0 danni ma pesca: in un mazzo che deve montare Grimmsnarl
        # vale piu' dei 10 di Corkscrew Punch.
        score += P["filch_draw_value"]
    if attack_id == SHADOW_BULLET:
        # I 30 di rimbalzo sono danno garantito in piu', su un bersaglio che
        # sceglie l'agente.
        score += SHADOW_BULLET_SPLASH
        for opp in board.their_bench:
            if opp.hp <= SHADOW_BULLET_SPLASH:
                score += P["splash_ko_bonus"]
                break
    if ko:
        pv = prize_value(target)
        score += P["attack_ko_bonus"] + pv * P["attack_prize_bonus"]
        if getattr(CARD_DATA.get(target.id), "ex", False):
            score += P["opp_ex_target_bonus"]
        if pv >= board.my_prizes_left():
            score += P["lethal_urgency_bonus"]   # questo KO chiude la partita
    return score


def best_attack_now(board, allow_one_more_energy=True):
    """(attack_id, danno, ko) migliore per l'attivo, o None."""
    active, target = board.my_active, board.their_active
    if active is None or target is None:
        return None
    extra = 0
    if allow_one_more_energy and not board.state.energyAttached and board.hand_count(D_ENERGY) > 0:
        extra = 1
    best = None
    for aid, _atk in usable_attacks(active, extra):
        s = score_attack(board, active, aid, target)
        if s < 0:
            continue
        dmg = estimate_damage(aid, active, target, board)
        if best is None or s > best[3]:
            best = (aid, dmg, dmg >= target.hp, s)
    return best[:3] if best else None


def best_attack_damage(board):
    """Danno del colpo piu' forte disponibile all'attivo (0 se non attacca)."""
    active, target = board.my_active, board.their_active
    if active is None or target is None:
        return 0
    extra = 0
    if not board.state.energyAttached and board.hand_count(D_ENERGY) > 0:
        extra = 1
    return max([estimate_damage(aid, active, target, board)
                for aid, _ in usable_attacks(active, extra)] or [0])


def opponent_can_ko_me(board):
    """Minaccia letale visibile sul nostro attivo il turno prossimo.

    Usa solo l'attivo avversario e un'energia in piu': la sua mano non e'
    visibile, quindi e' per forza una sottostima (documentato)."""
    mine, theirs = board.my_active, board.their_active
    if mine is None or theirs is None:
        return False
    for aid, _atk in usable_attacks(theirs, extra_energy=1):
        if estimate_damage(aid, theirs, mine, board) >= mine.hp:
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
        self._discard = _counts([c.id for c in (self.me.discard or [])])
        self.opponent = opponent

    def hand_count(self, cid):
        return self._hand.get(cid, 0)

    def field_count(self, cid):
        return self._field.get(cid, 0)

    def discard_count(self, cid):
        return self._discard.get(cid, 0)

    def have(self, cid):
        return self.hand_count(cid) + self.field_count(cid)

    def hand_size(self):
        return self.me.handCount or 0

    def bench_free(self):
        return max(0, self.me.benchMax - len(self.my_bench))

    def my_pokemon(self):
        return ([self.my_active] if self.my_active else []) + self.my_bench

    def their_pokemon(self):
        return ([self.their_active] if self.their_active else []) + self.their_bench

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
    decay_key = DECK.fetch_decay_keys.get(card_id, "decay_trainer")
    score = P[base_key] - have * P[decay_key]

    # Non cercare un pezzo prima che esista quello che lo rende giocabile.
    cap = P["prereq_penalty"]
    if card_id == MORGREM and board.have(IMPIDIMP) == 0:
        score = min(score, cap)
    elif card_id == GRIMMSNARL_EX and board.have(IMPIDIMP) == 0 and board.have(MORGREM) == 0:
        score = min(score, cap)
    elif card_id == FROSLASS and board.have(SNORUNT) == 0:
        score = min(score, cap)
    elif card_id == RARE_CANDY:
        # Rare Candy senza un Basic in campo (o senza lo Stage 2 in mano) e' carta morta.
        if board.field_count(IMPIDIMP) == 0:
            score = min(score, cap)
        elif board.hand_count(GRIMMSNARL_EX) > 0:
            score += P["fetch_candy_combo_bonus"]
    if card_id == GRIMMSNARL_EX and board.hand_count(RARE_CANDY) > 0 and board.field_count(IMPIDIMP) > 0:
        score += P["fetch_grimmsnarl_ready_bonus"]
    return score


def promote_score(board, pokemon):
    """Chi mandare attivo. Ricorre a ogni KO e si accumula: merita cura."""
    if pokemon is None:
        return -999
    score = pokemon.hp
    if pokemon.id in DECK.never_promote_ids:
        # Munkidori attivo e' inerte: non attacca e smette di essere il motore.
        score -= P["promote_never_penalty"]
    keys = DECK.promote_keys.get(pokemon.id)
    if keys:
        flat, per_energy = keys
        score += P[flat]
        if per_energy:
            score += min(_energy_units(pokemon),
                         DECK.max_useful_energy.get(pokemon.id, 2)) * P[per_energy]
    # Chi puo' attaccare *subito* vale piu' di chi va caricato per due turni.
    if usable_attacks(pokemon) and can_ever_attack(pokemon.id):
        score += 150
    # A pochi prize dalla sconfitta, mandare avanti un ex regala la partita.
    if prize_value(pokemon) >= board.their_prizes_left():
        score -= P["ex_exposure_penalty"]
    return score


def opponent_switch_score(board, pokemon):
    """Chi tirare fuori dalla panchina avversaria (Boss's Orders).

    Problema *inverso* rispetto a promote_score: per i miei scelgo chi
    sopravvive, per i loro chi muore. Valutare la disruption pura senza vedere
    la loro mano e' un giudizio che le regole non sanno dare: resta di
    proposito a punteggio basso."""
    if pokemon is None:
        return -999
    active = board.my_active
    if active is not None:
        for aid, _atk in usable_attacks(active):
            if estimate_damage(aid, active, pokemon, board) >= pokemon.hp:
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
    """Poffin: fino a 2 Basic con <=70 HP dal mazzo (Impidimp e Snorunt, mai
    Munkidori che ne ha 110). Limitato dagli slot liberi in panchina."""
    room = board.bench_free()
    scores = []
    for o in select.option:
        card = get_card(obs, o.area, o.index, o.playerIndex)
        scores.append(fetch_score(board, card.id) if card is not None else -999)
    chosen = []
    for i in _rank(scores):
        if len(chosen) >= min(select.maxCount, max(room, select.minCount)):
            break
        if len(chosen) < select.minCount or scores[i] > P["min_search_score"]:
            chosen.append(i)
    return chosen


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


def _handle_discard_energy(obs, board, select):
    """Pagamento della ritirata (o un effetto avversario): l'engine chiede
    *quali* energie togliere. Si tolgono da chi ne ha meno bisogno."""
    forced = _forced(select)
    if forced is not None:
        return forced
    scores = []
    for o in select.option:
        pokemon = get_card(obs, o.area, o.index, o.playerIndex)
        if pokemon is None:
            scores.append(0)
            continue
        # Piu' alto = piu' volentieri scartata.
        need = DECK.max_useful_energy.get(pokemon.id, 0)
        value = need * 100 - max(0, _energy_units(pokemon) - need) * 50
        if o.area == AreaType.ACTIVE and can_ever_attack(pokemon.id):
            value += 150
        scores.append(-value)
    return _rank(scores)[:max(select.minCount, 1)]


def _handle_discard_tool(obs, board, select):
    """Tool Scrapper: si scartano solo strumenti avversari."""
    scores = []
    for o in select.option:
        scores.append(1 if o.playerIndex != board.my_index else -1)
    chosen = [i for i in _rank(scores)
              if scores[i] > 0][:select.maxCount]
    return chosen if chosen else list(range(min(select.minCount, len(select.option))))


def _handle_count(obs, board, select):
    """Contesti NUMBER (quanti segnalini spostare, quante carte pescare):
    il massimo e' sempre corretto in questo mazzo -- Adrena-Brain cura i miei
    e colpisce i loro, e pescare non ha costo."""
    ranked = sorted(range(len(select.option)),
                    key=lambda i: select.option[i].number or 0, reverse=True)
    return ranked[:max(select.maxCount, 1)]


def _handle_evolve_ctx(obs, board, select):
    """Contesto EVOLVE (Rare Candy): la coppia (carta in mano, Pokemon in campo)."""
    forced = _forced(select)
    if forced is not None and len(select.option) == 1:
        return forced
    scores = [_score_evolve(obs, board, o) for o in select.option]
    return _rank(scores)[:max(select.maxCount, 1)]


# --- Punk Up ---------------------------------------------------------------

def _handle_attach_to(obs, board, select):
    """Punk Up, primo passo: quante {D} pescare dal mazzo. Sono tutte identiche,
    quindi conta solo il numero. Prenderle assottiglia anche il mazzo."""
    want = int(P["punk_up_count"])
    n = max(select.minCount, min(select.maxCount, want))
    return list(range(min(n, len(select.option))))


def _punk_target_score(board, pokemon, is_active):
    """Punk Up, secondo passo: a chi attaccare questa {D}.

    L'ordine e' quello di un giocatore: prima chi deve colpire *adesso*, poi
    l'ex di riserva in panchina, poi il corpo che diventera' il prossimo
    Grimmsnarl. Oltre il costo dell'attacco l'energia e' quasi sprecata."""
    if pokemon is None:
        return -999
    have = _energy_units(pokemon)
    need = DECK.attack_cost.get(pokemon.id, 2)
    if have >= need:
        return P["punk_overflow"] - have   # stabile e decrescente: riempie a giro
    if is_active and can_ever_attack(pokemon.id):
        return P["punk_active_attacker"] + (need - have)
    if pokemon.id == GRIMMSNARL_EX:
        return P["punk_bench_grimmsnarl"] + (need - have)
    return P["punk_bench_preload"] + (need - have)


def _handle_attach_from(obs, board, select):
    """Sceglie il Pokemon che riceve l'energia (Punk Up, una alla volta)."""
    forced = _forced(select)
    if forced is not None and len(select.option) == 1:
        return forced
    scores = []
    for o in select.option:
        pokemon = get_card(obs, o.area, o.index, o.playerIndex)
        scores.append(_punk_target_score(board, pokemon, o.area == AreaType.ACTIVE))
    return _rank(scores)[:max(select.maxCount, 1)]


# --- segnalini danno (Adrena-Brain, rimbalzo di Shadow Bullet) -------------

def _counter_target_score(board, pokemon, damage):
    """Dove mettere `damage` danni su un Pokemon avversario.

    Tre casi in ordine: uccide adesso; porta il loro attivo dentro il raggio di
    Shadow Bullet (e' il vero uso di Adrena-Brain: 180 non bastano su un ex da
    200+, 180+30 si'); altrimenti il bersaglio piu' malridotto e piu' costoso."""
    if pokemon is None:
        return -999
    score = 0.0
    if pokemon.hp <= damage:
        score += P["counter_ko_bonus"] + prize_value(pokemon) * P["counter_prize_bonus"]
        return score
    if pokemon is board.their_active:
        hit = best_attack_damage(board)
        if hit > 0 and pokemon.hp > hit and pokemon.hp - damage <= hit:
            score += P["counter_enables_ko"]
    score += prize_value(pokemon) * P["counter_prize_bonus"]
    score -= pokemon.hp * P["counter_low_hp_weight"] / 10.0
    return score


def _handle_damage_counter(obs, board, select):
    """DAMAGE_COUNTER: dove *mettere* i segnalini.

    Puo' arrivare anche da un effetto avversario che li mette sui miei: allora
    il problema e' inverso e si sceglie il bersaglio meno importante."""
    forced = _forced(select)
    if forced is not None and len(select.option) == 1:
        return forced
    damage = (select.remainDamageCounter or ADRENA_BRAIN_MAX) * 10
    scores = []
    for o in select.option:
        pokemon = get_card(obs, o.area, o.index, o.playerIndex)
        if o.playerIndex == board.my_index:
            # Su di me: il male minore. Un Pokemon con abilita' che sopravvive
            # e' comunque preferibile a perdere l'attaccante.
            scores.append(-_my_pokemon_value(board, pokemon, damage))
        else:
            scores.append(_counter_target_score(board, pokemon, damage))
    return _rank(scores)[:max(select.maxCount, 1)]


_handle_damage = _handle_damage_counter   # rimbalzo di Shadow Bullet: stessa logica


def _my_pokemon_value(board, pokemon, damage):
    """Quanto mi costa perdere/danneggiare questo mio Pokemon."""
    if pokemon is None:
        return 0
    value = 100.0
    if pokemon.id == GRIMMSNARL_EX:
        value += 500
    elif pokemon.id == MUNKIDORI:
        value += 250   # e' il motore, e vale comunque solo 1 prize
    elif pokemon.id in (IMPIDIMP, MORGREM):
        value += 200
    elif pokemon.id == FROSLASS:
        value += 150
    if pokemon.hp <= damage:
        value += 400 * prize_value(pokemon)   # morirebbe: regala prize
    if pokemon is board.my_active:
        value += 100
    return value


def _handle_remove_damage_counter(obs, board, select):
    """Adrena-Brain, primo passo: da quale mio Pokemon togliere i segnalini.

    Ne sposta *fino a* 3, e il tetto e' quanti ne ha la sorgente: prendere da
    un Pokemon con un solo segnalino sposta 10 invece di 30. A parita', si cura
    l'attaccante attivo, che e' quello che incassa davvero."""
    forced = _forced(select)
    if forced is not None and len(select.option) == 1:
        return forced
    scores = []
    for o in select.option:
        pokemon = get_card(obs, o.area, o.index, o.playerIndex)
        if pokemon is None:
            scores.append(-999)
            continue
        counters = max(0, pokemon.maxHp - pokemon.hp) // 10
        s = min(counters, ADRENA_BRAIN_MAX) * 100
        if counters >= ADRENA_BRAIN_MAX:
            s += P["remove_full_transfer"]
        if o.area == AreaType.ACTIVE:
            s += P["remove_from_active"]
        if pokemon.id == GRIMMSNARL_EX:
            s += 100
        scores.append(s)
    return _rank(scores)[:max(select.maxCount, 1)]


# --- MAIN ------------------------------------------------------------------

def _score_energy_attach(board, target, is_active):
    max_useful = DECK.max_useful_energy.get(target.id, 0)
    have = _energy_units(target)

    if max_useful <= 0:
        # Snorunt/Froslass non useranno mai l'energia per attaccare, ma se uno
        # resta bloccato attivo senza energia non puo' nemmeno ritirarsi: resta
        # li' a farsi picchiare finche' muore. Dagli il minimo per muoversi.
        data = CARD_DATA.get(target.id)
        if is_active and have == 0 and data and data.retreatCost > 0:
            return P["energy_stuck_active"]
        return -1

    if have >= max_useful:
        return -1

    score = P["energy_base"]
    if is_active:
        score += P["energy_active_bonus"]
    if target.id == MUNKIDORI:
        # Una sola {D} accende Adrena-Brain -- e gli paga anche la ritirata.
        score = P["energy_munkidori"] + (P["energy_active_bonus"] if is_active else 0)
    elif is_active and can_ever_attack(target.id):
        need = DECK.attack_cost.get(target.id, 2)
        if have + 1 >= need:
            score += P["energy_enables_attack"]   # attacca gia' questo turno
    elif not is_active:
        score = P["energy_bench_attacker"] + have
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
        return _score_energy_attach(board, target, is_active)
    if data.cardType == CardType.TOOL:
        if target.tools:
            return -1
        return P["tool_base"] + (P["tool_active_bonus"] if is_active else 0)
    return -1


def _can_pay_attack_now(board, card_id, body):
    """Il corpo `body`, una volta diventato `card_id`, riesce a pagare il suo
    attacco gia' questo turno? (Conta anche l'attacco manuale non ancora usato.)"""
    have = _energy_units(body)
    if not board.state.energyAttached and board.hand_count(D_ENERGY) > 0:
        have += 1
    return have >= DECK.attack_cost.get(card_id, 99)


def _score_evolve(obs, board, o):
    """Vale sia per le evoluzioni in MAIN sia per il contesto EVOLVE di Rare
    Candy: in entrambi i casi l'opzione porta area/index (la carta in mano) e
    inPlayArea/inPlayIndex (il corpo in campo)."""
    src = get_card(obs, o.area, o.index, board.my_index)
    if src is None:
        return -1
    body = get_card(obs, o.inPlayArea, o.inPlayIndex, board.my_index)
    is_active = (o.inPlayArea == AreaType.ACTIVE)

    # Bug gia' pagato altrove: evolvere l'attivo in un Pokemon che non sa
    # attaccare lascia lo slot senza offesa mentre continua a incassare.
    if is_active and not can_ever_attack(src.id):
        return -1

    score = P["evolve_base"]
    # Senza queste preferenze tutte le evoluzioni valgono uguale e il pareggio
    # si risolve per indice dell'opzione, cioe' a caso.
    if is_active:
        score += P["evolve_active_bonus"]
    if src.id == GRIMMSNARL_EX:
        score += P["evolve_stage2_bonus"]   # e' tutto il piano del mazzo
    elif src.id == FROSLASS:
        score += P["evolve_froslass_bonus"]

    if src.id == MORGREM:
        # Rare Candy fa scendere Grimmsnarl ex *su un Basic*, saltando lo Stage 1
        # e accendendo Punk Up. Evolvere quell'Impidimp in Morgrem butta via la
        # combo: nella traccia era il motivo per cui l'ex non arrivava mai.
        if board.hand_count(RARE_CANDY) > 0 and board.hand_count(GRIMMSNARL_EX) > 0:
            score -= P["evolve_blocks_candy_penalty"]
        # Idem se e' l'unico Impidimp in campo e l'ex e' gia' in mano: meglio
        # aspettare la Candy che bruciare il corpo.
        elif board.hand_count(GRIMMSNARL_EX) > 0 and board.field_count(IMPIDIMP) <= 1:
            score -= P["evolve_blocks_candy_penalty"] / 2.0

    if is_active:
        plan = best_attack_now(board)
        if plan and plan[2]:
            score -= P["evolve_blocks_ko_penalty"]   # non rinunciare a un KO sicuro
        # Grimmsnarl ex e' l'eccezione: Punk Up cerca 5 {D} nel mazzo e si paga
        # l'attacco da solo nello stesso turno.
        if (src.id != GRIMMSNARL_EX and usable_attacks(board.my_active)
                and not _can_pay_attack_now(board, src.id, body)):
            score -= P["evolve_no_attack_penalty"]
    return score


def _ready_bench_attacker(board):
    """Un attaccante in panchina gia' in grado di colpire se promosso."""
    best = None
    for p in board.my_bench:
        if not can_ever_attack(p.id) or not usable_attacks(p):
            continue
        if best is None or p.id == GRIMMSNARL_EX:
            best = p
    return best


def _score_retreat(board):
    """In questo engine la ritirata SCARTA l'energia: con 10 {D} in tutto, e'
    una spesa vera. Si ritira quasi solo per liberare lo slot da un Pokemon che
    non sa attaccare."""
    active = board.my_active
    if active is None or not board.my_bench:
        return -1
    data = CARD_DATA.get(active.id)
    cost = data.retreatCost if data else 1
    ready = _ready_bench_attacker(board)
    if ready is None:
        return -1

    penalty = cost * P["retreat_cost_penalty"]

    # Proattivo: l'attivo non sa attaccare (Munkidori/Snorunt/Froslass, o un
    # attaccante scarico) e in panchina c'e' chi puo'. Senza questo l'energia si
    # accumula inutilizzata mentre lo slot attivo non fa niente.
    if not usable_attacks(active) or not can_ever_attack(active.id):
        return P["retreat_useless_active"] - penalty

    # Reattivo: schivare un KO a cui non si sa rispondere. Raramente conviene su
    # Grimmsnarl ex, che costa 2 energie e ha 320 HP per incassare.
    plan = best_attack_now(board)
    if opponent_can_ko_me(board) and not (plan and plan[2]):
        return P["retreat_dodge"] - penalty
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
            if estimate_damage(aid, active, opp, board) >= opp.hp:
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
        if board.bench_free() <= 0:
            return -1
        return P["play_basic_pokemon"] + fetch_score(board, card.id) / 10.0

    if card.id == LILLIES_DETERMINATION:
        score = P["play_lillie"]
        if board.my_prizes_left() == PRIZE_TOTAL:
            score += P["lillie_six_prize_bonus"]   # a 6 prize pesca 8 invece di 6
        if board.hand_size() > 6:
            # Rimescola la mano: con una mano gia' grande si perde piu' di
            # quanto si peschi.
            score -= P["lillie_full_hand_penalty"]
        return score

    if card.id == BOSSS_ORDERS:
        return _score_boss_orders(board)

    if card.id == BUDDY_BUDDY_POFFIN and board.bench_free() <= 0:
        return -1

    if card.id == TOOL_SCRAPPER:
        if not any(p.tools for p in board.their_pokemon()):
            return -1
        return P["play_tool_scrapper"]

    if card.id == NIGHT_STRETCHER:
        # Vale qualcosa solo se nella discard c'e' qualcosa che serve davvero.
        useful = (board.discard_count(GRIMMSNARL_EX) + board.discard_count(IMPIDIMP)
                  + board.discard_count(MORGREM) + board.discard_count(MUNKIDORI)
                  + board.discard_count(D_ENERGY))
        if useful == 0:
            return -1
        return P["play_night_stretcher"]

    if card.id == RARE_CANDY:
        # L'engine la offre solo se legale, ma senza lo Stage 2 in mano
        # resterebbe comunque una giocata a vuoto.
        if board.hand_count(GRIMMSNARL_EX) == 0:
            return -1
        return P["play_rare_candy"]

    key = DECK.play_keys.get(card.id)
    return P[key] if key else 100


def _score_ability(obs, board, o):
    """Le abilita' vanno *prima* dell'attacco: attaccare chiude il turno.

    Adrena-Brain ha senso solo se ho davvero segnalini da spostare, altrimenti
    l'opzione e' rumore che sposta l'ordinamento."""
    card = get_card(obs, o.area, o.index, o.playerIndex if o.playerIndex is not None else board.my_index)
    cid = card.id if card is not None else 0
    if cid == MUNKIDORI:
        damaged = any(p.hp < p.maxHp for p in board.my_pokemon())
        return P["ability_munkidori"] if damaged else -1
    if cid == SPIKEMUTH_GYM:
        return P["ability_spikemuth"]
    return P["ability_generic"]


def _score_main(obs, board, o):
    t = o.type
    if t == OptionType.PLAY:
        return _score_play(obs, board, o)
    if t == OptionType.ATTACH:
        return _score_attach(obs, board, o)
    if t == OptionType.EVOLVE:
        return _score_evolve(obs, board, o)
    if t == OptionType.ABILITY:
        return _score_ability(obs, board, o)
    if t == OptionType.RETREAT:
        return _score_retreat(board)
    if t == OptionType.ATTACK:
        # Attaccare chiude il turno: in MAIN sta sotto ogni giocata utile, cosi'
        # si evolve, si attacca energia e si pescano carte *prima* di colpire.
        # L'unica eccezione e' il colpo che vince la partita: li' non c'e'
        # nessuna preparazione che valga il rischio di cambiare il tabellone.
        s = score_attack(board, board.my_active, o.attackId, board.their_active)
        if s < 0:
            return -1
        base = P["attack_main_base"] + min(s, 400) / 10.0
        target = board.their_active
        if target is not None:
            dmg = estimate_damage(o.attackId, board.my_active, target, board)
            if dmg >= target.hp and prize_value(target) >= board.my_prizes_left():
                base += P["lethal_urgency_bonus"]
        return base
    if t == OptionType.DISCARD:
        return -1
    if t == OptionType.END:
        return 0
    return 0


def _handle_main(obs, board, select):
    if select.maxCount <= 0:
        return []
    scores = [_score_main(obs, board, o) for o in select.option]
    ranked = _rank(scores)
    if _TRACE:
        _trace_main(obs, board, select, scores, ranked)
    return ranked[:1]


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
        return _handle_count(obs, board, select)
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
    # Specifici di questo mazzo
    int(SelectContext.EVOLVE): _handle_evolve_ctx,               # Rare Candy
    int(SelectContext.ATTACH_TO): _handle_attach_to,             # Punk Up, quali {D}
    int(SelectContext.ATTACH_FROM): _handle_attach_from,         # Punk Up, a chi
    int(SelectContext.DAMAGE): _handle_damage,                   # rimbalzo Shadow Bullet
    int(SelectContext.DAMAGE_COUNTER): _handle_damage_counter,   # Adrena-Brain, dove
    int(SelectContext.DAMAGE_COUNTER_ANY): _handle_damage_counter,
    int(SelectContext.REMOVE_DAMAGE_COUNTER): _handle_remove_damage_counter,
    int(SelectContext.REMOVE_DAMAGE_COUNTER_COUNT): _handle_count,
    int(SelectContext.DAMAGE_COUNTER_COUNT): _handle_count,
    int(SelectContext.DRAW_COUNT): _handle_count,
    int(SelectContext.DISCARD_ENERGY): _handle_discard_energy,   # costo di ritirata
    int(SelectContext.DISCARD_TOOL_CARD): _handle_discard_tool,
}


# ---------------------------------------------------------------------------
# Traccia diagnostica (spenta salvo GRIMM_TRACE)
# ---------------------------------------------------------------------------

def _name(cid):
    d = CARD_DATA.get(cid)
    return d.name if d else str(cid)


def _describe(obs, board, o):
    t = OptionType(int(o.type))
    if t == OptionType.PLAY:
        c = get_card(obs, AreaType.HAND, o.index, board.my_index)
        return f"PLAY {_name(c.id) if c else '?'}"
    if t == OptionType.ATTACH:
        s = get_card(obs, o.area, o.index, board.my_index)
        d = get_card(obs, o.inPlayArea, o.inPlayIndex, board.my_index)
        where = "attivo" if o.inPlayArea == AreaType.ACTIVE else f"panchina{o.inPlayIndex}"
        return f"ATTACH {_name(s.id) if s else '?'} -> {_name(d.id) if d else '?'}({where})"
    if t == OptionType.EVOLVE:
        s = get_card(obs, o.area, o.index, board.my_index)
        where = "attivo" if o.inPlayArea == AreaType.ACTIVE else f"panchina{o.inPlayIndex}"
        return f"EVOLVE -> {_name(s.id) if s else '?'} ({where})"
    if t == OptionType.ABILITY:
        c = get_card(obs, o.area, o.index, board.my_index)
        return f"ABILITY {_name(c.id) if c else '?'}"
    if t == OptionType.ATTACK:
        a = ATTACK_DATA.get(o.attackId)
        return f"ATTACK {a.name if a else o.attackId}"
    return t.name


def _trace_main(obs, board, select, scores, ranked):
    st = board.state
    act = board.my_active
    opp = board.their_active
    print(f"\n[T{st.turn} p{board.my_index}] attivo={_name(act.id) if act else '-'}"
          f" hp={act.hp if act else '-'} en={_energy_units(act) if act else 0}"
          f" | loro={_name(opp.id) if opp else '-'} hp={opp.hp if opp else '-'}"
          f" | prize {board.my_prizes_left()}-{board.their_prizes_left()}"
          f" | mano={board.hand_size()} panchina={len(board.my_bench)}")
    for i in ranked[:4]:
        mark = "->" if i == ranked[0] else "  "
        print(f"   {mark} {scores[i]:8.1f}  {_describe(obs, board, select.option[i])}")


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


# ---------------------------------------------------------------------------
# API pubblica per lo strato di ricerca (search.py)
# ---------------------------------------------------------------------------

# I contesti in cui la scelta e' un *ranking a scelta singola*: l'handler
# calcola un punteggio per opzione e ne prende una sola. Solo qui esiste una
# nozione utile di "seconda scelta", quindi solo qui la ricerca puo' entrare.
#
# Deliberatamente esclusi: TO_HAND / TO_BENCH / DISCARD (multi-select, lo
# spazio delle azioni esplode e l'euristica di fetch e' gia' il pezzo piu'
# tarato), i contesti NUMBER (il massimo e' sempre giusto in questo mazzo) e
# tutti i contesti forzati.
RANKED_CONTEXTS = frozenset({
    int(SelectContext.MAIN),
    int(SelectContext.ATTACK),
    int(SelectContext.SWITCH),
    int(SelectContext.TO_ACTIVE),
    int(SelectContext.DAMAGE_COUNTER),
    int(SelectContext.DAMAGE_COUNTER_ANY),
    int(SelectContext.DAMAGE),
})


def make_board(obs, opponent=None):
    return Board(obs, opponent)


def score_options(obs, board, select):
    """Punteggio euristico per *ogni* opzione, o None se il contesto non e' un
    ranking a scelta singola.

    E' esattamente lo stesso codice che l'handler usa per decidere: la ricerca
    non ricalcola niente per conto suo, legge la stessa classifica. Se i due
    numeri divergessero, l'agente avrebbe due opinioni sulla stessa mossa --
    che e' il modo classico in cui un ibrido diventa peggiore di entrambe le
    sue meta'.
    """
    ctx = int(select.context)
    if ctx not in RANKED_CONTEXTS or len(select.option) < 2:
        return None
    if select.minCount > 1 or select.maxCount != 1:
        # Multi-select: un "candidato" non e' piu' una singola opzione.
        return None

    if ctx == int(SelectContext.MAIN):
        return [_score_main(obs, board, o) for o in select.option]
    if ctx == int(SelectContext.ATTACK):
        active, target = board.my_active, board.their_active
        if active is None or target is None:
            return None
        return [score_attack(board, active, o.attackId, target) for o in select.option]
    if ctx in (int(SelectContext.SWITCH), int(SelectContext.TO_ACTIVE)):
        out = []
        for o in select.option:
            pokemon = get_card(obs, o.area, o.index, o.playerIndex)
            out.append(promote_score(board, pokemon)
                       if o.playerIndex == board.my_index
                       else opponent_switch_score(board, pokemon))
        return out
    # DAMAGE / DAMAGE_COUNTER(_ANY): dove piazzare i segnalini.
    damage = (select.remainDamageCounter or ADRENA_BRAIN_MAX) * 10
    out = []
    for o in select.option:
        pokemon = get_card(obs, o.area, o.index, o.playerIndex)
        if o.playerIndex == board.my_index:
            out.append(-_my_pokemon_value(board, pokemon, damage))
        else:
            out.append(_counter_target_score(board, pokemon, damage))
    return out


def is_winning_attack(board, option):
    """L'opzione e' l'attacco che chiude la partita adesso?

    Se lo e', nessuna ricerca deve poterla scavalcare: il valore di uno stato
    vinto non e' stimabile meglio di cosi'.
    """
    if option.type not in (OptionType.ATTACK,):
        return False
    target = board.their_active
    if target is None or option.attackId is None:
        return False
    dmg = estimate_damage(option.attackId, board.my_active, target, board)
    return dmg >= target.hp and prize_value(target) >= board.my_prizes_left()


def decide(obs, board=None, select=None):
    """La risposta puramente euristica, gia' sanificata. Non solleva mai."""
    select = select if select is not None else obs.select
    try:
        if board is None:
            board = Board(obs)
        handler = HANDLERS.get(int(select.context), _handle_generic)
        return _sanitize(handler(obs, board, select), select)
    except BaseException:
        n = min(select.maxCount or 0, len(select.option))
        return list(range(n)) if n > 0 else []


# ---------------------------------------------------------------------------
# Agente puramente euristico (identico a v1): serve come gradino della scala di
# training e come termine di paragone in benchmark.
# ---------------------------------------------------------------------------

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


def heuristic_agent(obs_dict: dict) -> list:
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


# `agent` non e' definito qui di proposito: l'entry point della submission e'
# main.py (euristica + ricerca). Chi vuole il solo strato euristico importa
# `heuristic_agent`, cosi' non si puo' spedire per sbaglio la meta' sbagliata.
