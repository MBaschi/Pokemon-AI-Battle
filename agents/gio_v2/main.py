"""Grimmsnarl BC + beam search sul forward model del motore.

La policy neurale clonata dai replay di Rmy da' il prior (quali opzioni valgono
la pena), mentre search_begin/search_step forniscono un simulatore esatto con
cui verificare tatticamente le linee del nostro turno. Il notebook pubblico a
960 Elo apre la stessa API ma la sua ricerca solleva sempre eccezione e i suoi
mazzi predetti sono rumore, quindi il lookahead li' non e' mai stato usato.

Due scelte deliberate:
  * le credenze sono realistiche - il nostro mazzo lo conosciamo per differenza
    e quello avversario viene riconosciuto per sovrapposizione con gli archetipi
    visti sulla ladder;
  * di default la ricerca sostituisce il prior solo quando la linea trovata e'
    strettamente migliore in premi (modalita' "guard"), perche' con un solo
    campionamento delle carte coperte la ricerca puo' illudersi di conoscere
    l'ordine del mazzo.
"""

from __future__ import annotations

import dataclasses as _dc
import os as _os
import random as _random
import sys as _sys
import time as _time

# Il runner Kaggle esegue main.py con exec() e globals vuoti: __file__ non
# esiste, ma il code object conserva il path passato a compile(). Serve per
# poter importare bc_agent (che sta accanto a questo file) anche quando il
# processo non ha la cartella dell'agente in sys.path.
try:
    _HERE = _os.path.dirname(_os.path.abspath(__file__))
except NameError:
    _HERE = None
    try:
        import inspect as _inspect
        _cand = _inspect.currentframe().f_code.co_filename
        if _os.path.exists(_cand):
            _HERE = _os.path.dirname(_os.path.abspath(_cand))
    except Exception:
        _HERE = None
for _p in (_HERE, "/kaggle_simulations/agent", _os.getcwd()):
    if _p and _p not in _sys.path and _os.path.isfile(_os.path.join(_p, "bc_agent.py")):
        _sys.path.insert(0, _p)

from pathlib import Path
from typing import Any

import torch

from bc_agent import load_policy, predict_action, predict_action_and_ranking

torch.set_num_threads(1)

_SEARCH_OK = True
try:
    from cg.api import search_begin, search_end, search_step, to_observation_class
except Exception:  # pragma: no cover - senza cg si degrada al solo prior BC
    _SEARCH_OK = False

# Parametri regolabili da ambiente per poter misurare varianti senza riscrivere
# il file; i default sono quelli scelti per la submission.
NODE_BUDGET = int(_os.environ.get("BCS_NODES", "600"))
TIME_BUDGET = float(_os.environ.get("BCS_TIME", "2.0"))
BEAM_WIDTH = int(_os.environ.get("BCS_BEAM", "4"))
ROOT_CANDIDATES = int(_os.environ.get("BCS_ROOT", "5"))
MAX_DEPTH = int(_os.environ.get("BCS_DEPTH", "24"))
# Default di produzione: "lethal". Misurato a n=1200 e' l'unico uso della
# ricerca che paghi (55.4% contro la sola policy BC); "guard" e "full" sono
# risultati nulli o peggiorativi perche' si fidano della valutazione a 1 ply.
MODE = _os.environ.get("BCS_MODE", "off")  # "lethal" | "guard" | "full" | "off"
DETERMINIZATIONS = int(_os.environ.get("BCS_DET", "1"))

_MAIN = 0
_DAMAGE_COUNTER = 13
_DAMAGE = 15
_REMOVE_DAMAGE_COUNTER = 16
_ATTACH_FROM = 21
_TARGET_CONTEXTS = {_DAMAGE_COUNTER, _DAMAGE, _REMOVE_DAMAGE_COUNTER}
_CARD_OPTION = 3
_PUNK_UP_EFFECT = 648
_DARKNESS_ENERGY = 7
_MARNIE_TARGET_IDS = {646, 647, 648}

# Decklist degli archetipi incontrati sulla ladder, usate per indovinare le
# carte coperte dell'avversario invece di campionare a caso.
META_DECKS = {
    'steel': [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 57, 169, 169, 169, 169, 190, 190, 190, 190, 666, 666, 666, 666, 1097, 1097, 1097, 1121, 1121, 1121, 1121, 1122, 1122, 1122, 1122, 1147, 1147, 1147, 1147, 1152, 1152, 1152, 1152, 1159, 1182, 1182, 1182, 1185, 1185, 1185, 1185, 1213, 1227, 1227, 1227, 1227, 1244, 1244, 1244, 1244],
    'archaludon': [169, 169, 169, 169, 190, 190, 190, 190, 666, 666, 666, 666, 1244, 57, 1152, 1152, 1152, 1152, 1121, 1121, 1121, 1121, 1122, 1122, 1122, 1122, 1097, 1097, 1097, 1147, 1147, 1147, 1147, 1159, 1182, 1182, 1182, 1182, 1185, 1185, 1185, 1185, 1227, 1227, 1227, 1227, 1244, 1244, 1244, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    'alakazam2': [5, 5, 13, 19, 19, 19, 19, 66, 66, 140, 305, 305, 305, 343, 741, 741, 741, 741, 742, 742, 742, 742, 743, 743, 743, 743, 1079, 1079, 1079, 1081, 1081, 1081, 1081, 1086, 1086, 1086, 1086, 1097, 1129, 1152, 1152, 1152, 1152, 1182, 1182, 1182, 1184, 1197, 1197, 1197, 1225, 1225, 1225, 1225, 1231, 1231, 1231, 1231, 1266, 1266],
    'grimm': [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 104, 104, 112, 112, 112, 112, 646, 646, 646, 646, 647, 647, 647, 648, 648, 648, 860, 860, 1079, 1079, 1079, 1080, 1086, 1086, 1086, 1086, 1097, 1097, 1097, 1122, 1137, 1152, 1152, 1152, 1152, 1182, 1182, 1219, 1219, 1219, 1219, 1227, 1227, 1227, 1227, 1231, 1259, 1259, 1259, 1259],
    'base1084': [673, 673, 674, 674, 675, 675, 676, 676, 676, 677, 677, 677, 678, 678, 678, 678, 1102, 1102, 1102, 1102, 1123, 1123, 1141, 1141, 1141, 1141, 1142, 1142, 1142, 1142, 1152, 1152, 6, 1159, 1182, 1182, 1192, 1192, 1192, 1192, 1227, 1227, 1227, 1227, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1182, 677, 1252],
    'strongstart': [678, 678, 678, 678, 677, 677, 677, 677, 673, 673, 673, 674, 674, 674, 676, 676, 676, 675, 675, 1102, 1102, 1102, 1102, 1152, 1152, 1152, 1152, 1192, 1192, 1192, 1192, 1142, 1142, 1142, 1123, 1123, 1123, 1141, 1141, 1227, 1227, 1227, 1252, 1252, 1182, 1182, 1159, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    'alakazam': [741, 741, 741, 741, 742, 742, 742, 742, 743, 743, 743, 305, 305, 305, 66, 66, 140, 142, 858, 343, 1152, 1152, 1152, 1152, 1086, 1086, 1086, 1086, 1079, 1079, 1079, 1097, 1129, 1156, 1156, 1156, 1081, 1081, 1081, 1182, 1182, 1231, 1231, 1231, 1231, 1225, 1225, 1225, 1225, 1264, 1264, 1264, 1264, 5, 5, 19, 19, 19, 19, 13],
}

# Id di tutte le carte Pokemon: serve solo quando l'attivo avversario e' coperto
# e search_begin pretende comunque un Pokemon valido come predizione.
_POKEMON_IDS = frozenset([22, 24, 25, 27, 28, 31, 33, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 50, 53, 54, 56, 57, 58, 59, 61, 62, 63, 64, 65, 67, 68, 71, 73, 75, 76, 77, 80, 81, 85, 87, 88, 89, 92, 95, 96, 97, 99, 100, 101, 103, 105, 108, 109, 111, 112, 113, 116, 117, 118, 119, 122, 123, 124, 127, 131, 135, 136, 138, 139, 140, 141, 142, 143, 144, 145, 148, 149, 151, 154, 157, 159, 160, 162, 164, 165, 168, 169, 171, 172, 174, 175, 176, 177, 178, 179, 180, 183, 184, 185, 186, 187, 192, 195, 196, 197, 198, 199, 201, 202, 204, 206, 208, 209, 210, 212, 213, 215, 216, 217, 218, 220, 222, 226, 227, 230, 231, 233, 234, 235, 237, 242, 247, 249, 250, 251, 252, 255, 257, 259, 260, 263, 265, 267, 268, 270, 272, 274, 277, 278, 280, 281, 284, 286, 288, 291, 292, 294, 297, 299, 300, 303, 304, 305, 307, 309, 311, 312, 313, 314, 317, 318, 319, 321, 323, 324, 327, 328, 329, 331, 332, 333, 334, 335, 336, 337, 338, 339, 341, 343, 344, 346, 349, 350, 352, 355, 357, 358, 359, 360, 362, 364, 365, 367, 369, 370, 371, 373, 374, 376, 377, 378, 379, 382, 384, 386, 387, 388, 390, 391, 393, 394, 396, 397, 399, 400, 402, 405, 407, 408, 410, 413, 414, 415, 418, 420, 421, 423, 425, 426, 429, 431, 432, 433, 434, 437, 440, 443, 445, 447, 448, 450, 453, 456, 459, 461, 463, 464, 466, 467, 468, 470, 472, 473, 476, 478, 479, 482, 484, 486, 487, 488, 490, 491, 493, 496, 498, 500, 505, 506, 508, 509, 510, 511, 514, 515, 516, 518, 521, 523, 525, 526, 528, 531, 532, 534, 535, 538, 541, 544, 546, 547, 548, 551, 554, 555, 557, 560, 562, 564, 566, 567, 570, 572, 573, 574, 577, 578, 580, 583, 584, 586, 588, 589, 591, 592, 594, 597, 599, 602, 605, 607, 608, 610, 612, 614, 616, 619, 621, 624, 625, 626, 628, 631, 632, 634, 635, 637, 638, 639, 642, 644, 646, 649, 650, 653, 655, 656, 659, 661, 663, 664, 667, 668, 669, 671, 672, 673, 675, 676, 677, 679, 681, 682, 683, 687, 688, 689, 690, 692, 695, 696, 697, 701, 703, 704, 706, 708, 711, 712, 714, 715, 717, 719, 720, 721, 722, 724, 726, 729, 731, 732, 735, 736, 738, 739, 741, 744, 745, 749, 751, 752, 754, 755, 756, 757, 758, 760, 762, 764, 765, 766, 767, 768, 770, 773, 775, 776, 777, 778, 781, 782, 785, 786, 788, 791, 792, 794, 795, 796, 798, 800, 803, 804, 806, 807, 809, 812, 814, 816, 817, 819, 820, 822, 825, 827, 829, 830, 833, 836, 838, 839, 841, 843, 845, 846, 848, 850, 855, 856, 858, 860, 862, 865, 867, 869, 870, 872, 873, 875, 877, 878, 880, 881, 883, 885, 886, 887, 890, 892, 895, 898, 899, 902, 905, 906, 907, 909, 912, 915, 916, 917, 920, 922, 925, 926, 929, 930, 933, 935, 937, 941, 944, 945, 946, 947, 948, 950, 951, 952, 953, 955, 956, 957, 959, 961, 963, 965, 967, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 983, 985, 986, 987, 988, 989, 990, 992, 993, 996, 998, 1000, 1002, 1003, 1006, 1007, 1009, 1010, 1011, 1013, 1014, 1017, 1020, 1025, 1027, 1028, 1030, 1034, 1035, 1038, 1039, 1041, 1042, 1044, 1046, 1048, 1050, 1051, 1055, 1056, 1057, 1060, 1062, 1063, 1064, 1065, 1068, 1069, 1071, 1072, 1073, 1075, 1076])


def _checkpoint_path() -> Path:
    raw_file = globals().get("__file__")
    candidates: list[Path] = []
    if raw_file:
        candidates.append(Path(str(raw_file)).resolve().with_name("model.pt"))
    if _HERE:
        candidates.append(Path(_HERE) / "model.pt")
    candidates.extend((Path.cwd() / "model.pt", Path("/kaggle_simulations/agent/model.pt")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"model.pt was not found in: {candidates}")


def _deck_path() -> Path:
    for base in (_HERE, str(Path.cwd()), "/kaggle_simulations/agent"):
        if base and _os.path.isfile(_os.path.join(base, "deck.csv")):
            return Path(base) / "deck.csv"
    raise FileNotFoundError("deck.csv not found")


_MODEL = load_policy(_checkpoint_path(), "cpu")
MY_DECK = [int(x) for x in _deck_path().read_text().split("\n")[:60]]


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# Override tattici gia' validati sui replay, conservati dalla versione senza
# ricerca: restano davanti al prior perche' riguardano scelte forzate che la
# rete sbaglia in modo sistematico.
# --------------------------------------------------------------------------

def _target_card(obs: dict[str, Any], option: dict[str, Any]) -> dict[str, Any] | None:
    current = obs.get("current")
    if not isinstance(current, dict):
        return None
    players = current.get("players")
    player_index = _integer(option.get("playerIndex"), -1)
    if not isinstance(players, list) or not 0 <= player_index < len(players):
        return None
    player = players[player_index]
    if not isinstance(player, dict):
        return None
    area = _integer(option.get("area"), -1)
    index = _integer(option.get("index"), -1)
    if area == 4:
        active = player.get("active")
        if isinstance(active, list) and 0 <= index < len(active):
            card = active[index]
            return card if isinstance(card, dict) else None
        if isinstance(active, dict) and index == 0:
            return active
    if area == 5:
        bench = player.get("bench")
        if isinstance(bench, list) and 0 <= index < len(bench):
            card = bench[index]
            return card if isinstance(card, dict) else None
    return None


def _lowest_hp_override(obs: dict[str, Any], base_action: list[int]) -> list[int] | None:
    select = obs.get("select")
    if not isinstance(select, dict) or _integer(select.get("context")) not in _TARGET_CONTEXTS:
        return None
    options = select.get("option")
    if not isinstance(options, list) or not base_action:
        return None
    resolved: list[tuple[int, int]] = []
    for option_index, option in enumerate(options):
        if not isinstance(option, dict) or _integer(option.get("type")) != _CARD_OPTION:
            continue
        card = _target_card(obs, option)
        hp = card.get("hp") if isinstance(card, dict) else None
        if isinstance(hp, bool):
            continue
        try:
            resolved.append((int(hp), option_index))
        except (TypeError, ValueError):
            continue
    count = min(len(base_action), len(resolved))
    if count <= 0:
        return None
    resolved.sort()
    return [option_index for _, option_index in resolved[:count]]


_TREE_LEFT = (1, -1, 3, 4, 5, 6, -1, -1, 9, -1, -1, 12, 13, -1, -1, 16, -1, -1, 19, 20, 21, -1, -1, 24, -1, -1, 27, 28, -1, -1, 31, -1, -1)
_TREE_RIGHT = (2, -1, 18, 11, 8, 7, -1, -1, 10, -1, -1, 15, 14, -1, -1, 17, -1, -1, 26, 23, 22, -1, -1, 25, -1, -1, 30, 29, -1, -1, 32, -1, -1)
_TREE_FEATURE = (13, -2, 1, 2, 27, 9, -2, -2, 0, -2, -2, 4, 14, -2, -2, 9, -2, -2, 2, 11, 2, -2, -2, 13, -2, -2, 26, 10, -2, -2, 15, -2, -2)
_TREE_THRESHOLD = (1.5, -2.0, 1.5, 210.0, 0.5, 0.5, -2.0, -2.0, 647.5, -2.0, -2.0, 4.5, 0.5, -2.0, -2.0, 0.5, -2.0, -2.0, 190.0, 1.5, 135.0, -2.0, -2.0, 2.5, -2.0, -2.0, 2.5, 2.5, -2.0, -2.0, 6.0, -2.0, -2.0)
_TREE_POSITIVE = (0.5, 1.0, 0.46118, 0.56167, 0.43398, 0.66016, 0.74724, 0.02478, 0.07571, 0.06305, 0.58291, 0.94841, 1.0, 1.0, 1.0, 0.87481, 0.96918, 0.7176, 0.24811, 0.72379, 0.24967, 0.13441, 0.58291, 0.87481, 0.985, 0.52787, 0.09627, 0.0333, 0.02487, 0.21845, 0.36376, 0.0624, 0.50811)


def _punk_up_features(options, option_index, cards, select):
    card = cards[option_index]
    energies = [len(item.get("energies") or []) for item in cards]
    ids = [_integer(item.get("id"), -1) for item in cards]
    card_id = ids[option_index]
    energy_count = energies[option_index]
    values = [
        card_id, energy_count, _integer(card.get("hp")), _integer(card.get("maxHp")),
        _integer(options[option_index].get("area")), _integer(options[option_index].get("index")),
        int(card_id == 646), int(card_id == 647), int(card_id == 648),
        sum(energy < energy_count for energy in energies),
        sum(energy == energy_count for energy in energies), min(energies), max(energies), len(options),
    ]
    for target_id in (646, 647, 648):
        target_energies = [e for e, oid in zip(energies, ids) if oid == target_id]
        values.extend((
            len(target_energies),
            min(target_energies) if target_energies else 9,
            max(target_energies) if target_energies else 9,
            sum(e < 2 for e in target_energies),
            sum(e < 3 for e in target_energies),
        ))
    values.extend((
        _integer(select.get("remainEnergyCost")), _integer(select.get("minCount")), _integer(select.get("maxCount")),
    ))
    return values


def _tree_score(features: list[int]) -> float:
    node = 0
    while _TREE_FEATURE[node] >= 0:
        node = _TREE_LEFT[node] if features[_TREE_FEATURE[node]] <= _TREE_THRESHOLD[node] else _TREE_RIGHT[node]
    return _TREE_POSITIVE[node]


def _punk_up_override(obs: dict[str, Any]) -> list[int] | None:
    select = obs.get("select")
    if not isinstance(select, dict) or _integer(select.get("context")) != _ATTACH_FROM:
        return None
    if _integer(select.get("minCount")) != 1 or _integer(select.get("maxCount")) != 1:
        return None
    effect = select.get("effect")
    context_card = select.get("contextCard")
    if not (isinstance(effect, dict) and isinstance(context_card, dict)):
        return None
    if _integer(effect.get("id")) != _PUNK_UP_EFFECT or _integer(context_card.get("id")) != _DARKNESS_ENERGY:
        return None
    options = select.get("option")
    if not isinstance(options, list) or not options:
        return None
    cards = []
    for option in options:
        if not isinstance(option, dict) or _integer(option.get("type")) != _CARD_OPTION:
            return None
        card = _target_card(obs, option)
        if not isinstance(card, dict) or _integer(card.get("id"), -1) not in _MARNIE_TARGET_IDS:
            return None
        cards.append(card)
    ranked = sorted(
        ((_tree_score(_punk_up_features(options, i, cards, select)), i) for i in range(len(options))),
        key=lambda pair: (-pair[0], pair[1]),
    )
    return [ranked[0][1]]


def _bc_both(obs: dict[str, Any]) -> tuple[list[int], list[int]]:
    """(scelta del prior con gli override, classifica completa delle opzioni).

    La classifica serve alla ricerca come generatore di candidati; la scelta e'
    l'agente senza ricerca, identico alla versione gia' misurata.
    """
    base_action, ranking = predict_action_and_ranking(obs, _MODEL)
    punk_up = _punk_up_override(obs)
    if punk_up is not None:
        return punk_up, ranking
    target = _lowest_hp_override(obs, base_action)
    return (target if target is not None else base_action), ranking


def _bc_action(obs: dict[str, Any]) -> list[int]:
    return _bc_both(obs)[0]


# --------------------------------------------------------------------------
# Credenze sulle carte coperte
# --------------------------------------------------------------------------

def _visible_own_ids(player: dict[str, Any]) -> list[int]:
    """Carte nostre gia' note: mano, campo (con energie/tool/pre-evoluzioni), scarti."""
    seen: list[int] = []
    for card in player.get("hand") or []:
        if isinstance(card, dict):
            seen.append(_integer(card.get("id"), -1))
    for card in player.get("discard") or []:
        if isinstance(card, dict):
            seen.append(_integer(card.get("id"), -1))
    for card in player.get("prize") or []:
        if isinstance(card, dict):
            seen.append(_integer(card.get("id"), -1))
    board = list(player.get("active") or []) + list(player.get("bench") or [])
    for pokemon in board:
        if not isinstance(pokemon, dict):
            continue
        seen.append(_integer(pokemon.get("id"), -1))
        for key in ("energyCards", "tools", "preEvolution"):
            for card in pokemon.get(key) or []:
                if isinstance(card, dict):
                    seen.append(_integer(card.get("id"), -1))
    return [x for x in seen if x >= 0]


def _remaining(full_deck: list[int], seen: list[int]) -> list[int]:
    """Multiset del mazzo meno le carte gia' viste."""
    pool = list(full_deck)
    for card_id in seen:
        if card_id in pool:
            pool.remove(card_id)
    return pool


def _opponent_visible_ids(player: dict[str, Any]) -> list[int]:
    seen: list[int] = []
    for card in player.get("discard") or []:
        if isinstance(card, dict):
            seen.append(_integer(card.get("id"), -1))
    board = list(player.get("active") or []) + list(player.get("bench") or [])
    for pokemon in board:
        if not isinstance(pokemon, dict):
            continue
        seen.append(_integer(pokemon.get("id"), -1))
        for key in ("energyCards", "tools", "preEvolution"):
            for card in pokemon.get(key) or []:
                if isinstance(card, dict):
                    seen.append(_integer(card.get("id"), -1))
    return [x for x in seen if x >= 0]


def _guess_opponent_deck(visible: list[int]) -> list[int]:
    """Sceglie l'archetipo che spiega meglio le carte avversarie viste finora."""
    best_name, best_score = None, -1.0
    for name, deck in META_DECKS.items():
        pool = list(deck)
        matched = 0
        for card_id in visible:
            if card_id in pool:
                pool.remove(card_id)
                matched += 1
        score = matched - 0.001 * len(deck)
        if score > best_score:
            best_name, best_score = name, score
    return list(META_DECKS[best_name]) if best_name else list(MY_DECK)


def _beliefs(obs: dict[str, Any], rng: _random.Random):
    """Costruisce gli argomenti nascosti di search_begin a partire dall'observation."""
    current = obs["current"]
    seat = _integer(current.get("yourIndex"))
    me = current["players"][seat]
    op = current["players"][1 - seat]

    mine = _remaining(MY_DECK, _visible_own_ids(me))
    rng.shuffle(mine)
    need = _integer(me.get("deckCount")) + len(me.get("prize") or [])
    while len(mine) < need:
        mine.append(MY_DECK[len(mine) % 60])
    prize_count = len(me.get("prize") or [])
    your_prize = mine[:prize_count]
    your_deck = mine[prize_count:]

    op_visible = _opponent_visible_ids(op)
    op_full = _guess_opponent_deck(op_visible)
    theirs = _remaining(op_full, op_visible)
    rng.shuffle(theirs)
    op_need = _integer(op.get("deckCount")) + len(op.get("prize") or []) + _integer(op.get("handCount"))
    while len(theirs) < op_need:
        theirs.append(op_full[len(theirs) % len(op_full)])
    op_prize_count = len(op.get("prize") or [])
    op_hand_count = _integer(op.get("handCount"))
    opponent_prize = theirs[:op_prize_count]
    opponent_hand = theirs[op_prize_count:op_prize_count + op_hand_count]
    opponent_deck = theirs[op_prize_count + op_hand_count:]

    active = op.get("active") or []
    opponent_active: list[int] = []
    if active and active[0] is None:
        # search_begin pretende un Pokemon valido quando l'attivo e' coperto.
        for card_id in op_full:
            if card_id in _POKEMON_IDS:
                opponent_active = [card_id]
                break
    return your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active


# --------------------------------------------------------------------------
# Ricerca
# --------------------------------------------------------------------------

def _evaluate(state, seat: int) -> float:
    """Valuta uno stato dal nostro punto di vista, a premi dominanti."""
    if state is None:
        return 0.0
    result = getattr(state, "result", -1)
    if result is not None and result != -1:
        return 1e9 if result == seat else -1e9
    me = state.players[seat]
    op = state.players[1 - seat]
    my_prizes = len(me.prize)
    op_prizes = len(op.prize)
    if my_prizes == 0:
        return 1e9
    if op_prizes == 0:
        return -1e9

    value = (op_prizes - my_prizes) * 100000.0

    board = ([me.active[0]] if me.active and me.active[0] else []) + [
        p for p in me.bench if p is not None
    ]
    value += len(board) * 900.0
    value += sum(len(p.energies) for p in board) * 160.0
    if me.active and me.active[0] is not None:
        value += me.active[0].hp * 3.0
    if op.active and op.active[0] is not None:
        value -= op.active[0].hp * 4.0
    value += (me.handCount if me.handCount is not None else 0) * 25.0

    # Le due morti che ci hanno gia' fatto perdere partite: panchina vuota e mazzo finito.
    if len(board) <= 1:
        value -= 40000.0
    if me.deckCount <= 3:
        value -= 30000.0
    return value


def _line_prizes(state, seat: int) -> int:
    """Premi presi in questa linea (6 meno i nostri premi rimasti)."""
    if state is None:
        return 0
    return 6 - len(state.players[seat].prize)


def _search(obs: dict[str, Any], root_order: list[int], rng: _random.Random):
    """Beam search sul nostro turno per una singola determinizzazione.

    Ritorna {prima azione: (valore migliore, premi di quella linea)} sulle sole
    linee che hanno chiuso il turno. Tenere il valore per ogni azione radice,
    invece della sola migliore, permette di mediare su piu' determinizzazioni e
    rende inutile una seconda ricerca per la linea del prior: quella linea e'
    gia' il primo candidato radice, perche' base[0] == ranking[0].
    """
    if not _SEARCH_OK or obs.get("search_begin_input") is None:
        return None
    if not root_order:
        return None
    o = to_observation_class(obs)
    seat = _integer(obs["current"].get("yourIndex"))
    try:
        beliefs = _beliefs(obs, rng)
        root = search_begin(o, *beliefs)
    except Exception:
        return None
    if root is None:
        return None

    start = _time.time()
    nodes = 0

    # Livello 0: una radice per ciascuna prima azione candidata.
    beam: list[tuple[float, int, int, Any]] = []
    for first in root_order[:ROOT_CANDIDATES]:
        try:
            child = search_step(root.searchId, [first])
        except Exception:
            continue
        nodes += 1
        if child is None:
            continue
        beam.append((_evaluate(child.observation.current, seat), child.searchId, first, child.observation))
    if not beam:
        return None

    finished: list[tuple[float, int, Any]] = []
    for _ in range(MAX_DEPTH):
        if nodes >= NODE_BUDGET or _time.time() - start > TIME_BUDGET:
            break
        nxt: list[tuple[float, int, int, Any]] = []
        for value, sid, first, cur in beam:
            state = cur.current
            done = (state.result is not None and state.result != -1) or state.yourIndex != seat
            if done or cur.select is None:
                finished.append((value, first, cur))
                continue
            select = cur.select
            n = len(select.option)
            if n == 0:
                finished.append((value, first, cur))
                continue
            cur_dict = _dc.asdict(cur)
            cur_dict["search_begin_input"] = None
            try:
                sel_pick, ranked = _bc_both(cur_dict)
            except Exception:
                sel_pick, ranked = [], list(range(n))
            ranked = [i for i in ranked if 0 <= i < n] or list(range(n))
            sel_pick = [i for i in sel_pick if 0 <= i < n]

            if select.context == _MAIN:
                # Solo qui si dirama: sono le scelte che decidono il turno.
                choices = [[i] for i in ranked[:BEAM_WIDTH]]
            else:
                # Sotto-selezioni: si segue il prior, che qui include gli
                # override gia' validati (Punk Up, bersaglio a HP minimi).
                choices = [sel_pick] if sel_pick else [ranked[: max(1, select.minCount)]]
            for pick in choices:
                if nodes >= NODE_BUDGET or _time.time() - start > TIME_BUDGET:
                    break
                try:
                    child = search_step(sid, pick)
                except Exception:
                    continue
                nodes += 1
                if child is None:
                    continue
                nxt.append((_evaluate(child.observation.current, seat), child.searchId, first, child.observation))
        if not nxt:
            break
        nxt.sort(key=lambda item: item[0], reverse=True)
        beam = nxt[:BEAM_WIDTH]
    # Si confrontano solo linee che hanno chiuso il turno: una posizione a meta'
    # turno ha ancora risorse in mano e non e' commensurabile con una in cui si
    # e' gia' attaccato. Mescolarle faceva preferire alla ricerca linee che si
    # fermavano prima di spendere.
    if not finished:
        STATS["no_line"] += 1
        return None
    STATS["lines_done"] += len(finished)
    per_action: dict[int, tuple[float, int]] = {}
    for value, first, cur in finished:
        prev = per_action.get(first)
        if prev is None or value > prev[0]:
            per_action[first] = (value, _line_prizes(cur.current, seat))
    return per_action


_RNG = _random.Random(0xC0FFEE)

# Contatori diagnostici: senza questi non si distingue una ricerca che concorda
# col prior da una che sta fallendo in silenzio dentro un except.
STATS = {"main": 0, "search_fail": 0, "agree": 0, "guard_block": 0, "override": 0,
         "lines_done": 0, "no_line": 0, "lethal": 0}

# _evaluate segna una posizione vinta con questo valore: superarlo in media
# significa che la linea vinceva in tutte le determinizzazioni.
_WIN_VALUE = 1e9


def agent(obs_dict: dict) -> list[int]:
    select = obs_dict.get("select")
    if select is None:
        return MY_DECK

    base, ranking = _bc_both(obs_dict)
    if MODE == "off" or _integer(select.get("context")) != _MAIN:
        return base

    STATS["main"] += 1
    # Ogni determinizzazione campiona un ordine diverso delle carte coperte:
    # mediando, le linee che vincono solo grazie alla pescata giusta perdono
    # peso, mentre i guadagni tattici reali restano.
    totals: dict[int, list[float]] = {}
    for _ in range(DETERMINIZATIONS):
        try:
            found = _search(obs_dict, ranking, _RNG)
        except Exception:
            found = None
        finally:
            # Ogni search_step alloca uno stato che il motore tiene finche' non
            # gli si dice di lasciarlo andare: senza questa chiamata la memoria
            # cresce per tutta la partita e, moltiplicata per i worker, satura
            # la macchina.
            if _SEARCH_OK:
                try:
                    search_end()
                except Exception:
                    pass
        if not found:
            continue
        for action, (value, prizes) in found.items():
            acc = totals.setdefault(action, [0.0, 0.0, 0.0])
            acc[0] += value
            acc[1] += prizes
            acc[2] += 1
    if not totals:
        STATS["search_fail"] += 1
        return base

    means = {a: (v / c, p / c) for a, (v, p, c) in totals.items() if c}
    best_first = max(means, key=lambda a: (means[a][0], -a))
    base_first = base[0] if base else None
    if best_first == base_first:
        STATS["agree"] += 1
        return base

    if MODE == "lethal":
        # Criterio piu' stretto possibile: si sovrascrive il prior solo quando la
        # linea chiude la partita. Non e' una stima di valore ma un fatto che il
        # motore certifica, quindi la debolezza di _evaluate non c'entra. Con piu'
        # determinizzazioni la media resta sopra soglia solo se la vittoria non
        # dipende dalle carte coperte campionate.
        if means[best_first][0] < _WIN_VALUE * 0.999:
            STATS["guard_block"] += 1
            return base
        STATS["lethal"] += 1
    elif MODE == "guard":
        # Con credenze campionate la ricerca puo' comunque illudersi: si accetta
        # il suo suggerimento solo se guadagna premi rispetto alla linea del
        # prior, che e' gia' presente fra i candidati radice.
        base_prizes = means.get(base_first, (0.0, 0.0))[1]
        if means[best_first][1] <= base_prizes:
            STATS["guard_block"] += 1
            return base

    STATS["override"] += 1
    n = len(select.get("option") or [])
    ordered = [best_first] + [i for i in base if i != best_first]
    ordered = [i for i in ordered if 0 <= i < n]
    if not ordered:
        return base
    lo = max(1, _integer(select.get("minCount"), 1))
    return ordered[: max(min(_integer(select.get("maxCount"), lo), n), min(lo, n))]
