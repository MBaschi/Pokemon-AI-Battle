"""Estrazione delle feature di stato da un replay Kaggle (Pokémon TCG AI Battle).

Un replay contiene, per ogni "visualize step", lo stato completo della partita in
`doc["steps"][0][0]["visualize"][k]["current"]`. Da ogni stato costruiamo un
vettore di feature *da una prospettiva* (io / avversario) + un "bag of cards"
delle carte visibili, e la label = "questa prospettiva ha vinto?".

Due modalità informative:
  * ``realistic`` (default): usa solo cio' che un bot vede davvero durante la
    partita — la propria mano, i due board, gli scarti (pubblici) e i *conteggi*
    di mano/mazzo avversari. E' la modalita' giusta se il modello deve poi
    guidare un bot.
  * ``oracle``: usa anche mano e mazzo avversari (visibili solo nel replay).
    Utile come limite superiore / per capire quanto pesa l'informazione nascosta.

Il modulo non dipende da torch/sklearn: solo csv + (opzionale) pandas altrove.
"""
from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path

# --------------------------------------------------------------------------- #
# Metadati carte (is_pokemon / is_ex / is_mega) da EN_Card_Data.csv
# --------------------------------------------------------------------------- #
_STAGE_COL = "Stage (Pokémon)/Type (Energy and Trainer)"
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]

_CARD_META: dict | None = None


def _find_card_csv() -> Path:
    cands = [
        _ROOT / "view_replays" / "card_data" / "EN_Card_Data.csv",
        _ROOT / "doc" / "pokemon-tcg-ai-battle" / "EN_Card_Data.csv",
        _ROOT / "meta_analysis" / "ptcg_data" / "EN_Card_Data.csv",
    ]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(
        "EN_Card_Data.csv non trovato (cercato in view_replays/card_data, doc/, ptcg_data/)."
    )


def card_meta() -> dict:
    """{card_id: {"is_pokemon","is_ex","is_mega","name"}} (cache di modulo)."""
    global _CARD_META
    if _CARD_META is not None:
        return _CARD_META
    meta: dict = {}
    with open(_find_card_csv(), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                cid = int(row["Card ID"])
            except (KeyError, ValueError):
                continue
            if cid in meta:
                continue
            stage = (row.get(_STAGE_COL) or "")
            rule = (row.get("Rule") or "")
            meta[cid] = {
                "name": row.get("Card Name", ""),
                "is_pokemon": "Pokémon" in stage,
                "is_ex": "ex" in rule.lower(),
                "is_mega": "mega" in rule.lower(),
            }
    _CARD_META = meta
    return meta


# --------------------------------------------------------------------------- #
# Navigazione del replay
# --------------------------------------------------------------------------- #
def winner_of(doc: dict):
    """Indice del giocatore vincente (0/1) dai rewards, o None per pareggio."""
    r = doc.get("rewards") or []
    if len(r) < 2 or r[0] is None or r[1] is None or r[0] == r[1]:
        return None
    return 0 if r[0] > r[1] else 1


def _visualize(doc: dict):
    try:
        return doc["steps"][0][0]["visualize"] or []
    except (KeyError, IndexError, TypeError):
        return []


def states_by_turn(doc: dict) -> list[tuple[int, dict]]:
    """Uno stato per turno (l'ultimo visto in quel turno = board di fine turno).

    Collassa i molti micro-step del visualizer in un solo snapshot per turno,
    riducendo la correlazione tra sample della stessa partita.
    """
    last: dict[int, dict] = {}
    for step in _visualize(doc):
        cur = (step or {}).get("current")
        if not cur:
            continue
        t = cur.get("turn")
        if isinstance(t, int):
            last[t] = cur
    return sorted(last.items())


# --------------------------------------------------------------------------- #
# Feature di un lato (i propri Pokémon in campo + zone)
# --------------------------------------------------------------------------- #
def _mon_list(player: dict) -> list:
    act = player.get("active")
    act = act if isinstance(act, list) else ([act] if act else [])
    bench = player.get("bench") or []
    return [m for m in (act + bench) if isinstance(m, dict)]


def _side_features(player: dict, meta: dict) -> dict:
    mons = _mon_list(player)
    hp = [float(m.get("hp") or 0) for m in mons]
    maxhp = [float(m.get("maxHp") or 0) for m in mons]
    energy = [len(m.get("energies") or []) for m in mons]
    tools = [len(m.get("tools") or []) for m in mons]
    ids = [m.get("id") for m in mons]
    is_ex = sum(1 for i in ids if meta.get(i, {}).get("is_ex"))
    is_mega = sum(1 for i in ids if meta.get(i, {}).get("is_mega"))
    n_evolved = sum(1 for m in mons if (m.get("preEvolution") or []))
    active = player.get("active")
    active = active[0] if isinstance(active, list) and active else (active or {})
    discard = player.get("discard") or []
    n_poke_discard = sum(1 for c in discard if meta.get(c.get("id"), {}).get("is_pokemon"))
    return {
        "prizes_remaining": len(player.get("prize") or []),
        "pokemon_in_play": len(mons),
        "total_hp": sum(hp),
        "total_damage": sum(mx - h for mx, h in zip(maxhp, hp)),
        "total_energy": sum(energy),
        "n_tools": sum(tools),
        "n_ex": is_ex,
        "n_mega": is_mega,
        "n_evolved": n_evolved,
        "max_maxhp": max(maxhp) if maxhp else 0.0,
        "active_hp": float(active.get("hp") or 0) if isinstance(active, dict) else 0.0,
        "active_energy": len(active.get("energies") or []) if isinstance(active, dict) else 0,
        "hand_count": int(player.get("handCount") or len(player.get("hand") or [])),
        "deck_count": int(player.get("deckCount") or len(player.get("deck") or [])),
        "discard_count": len(discard),
        "n_poke_discard": n_poke_discard,
    }


# carte visibili per il "bag of cards", per modalita' informativa
def _card_bag(cur: dict, me: int, opp: int, meta: dict, info_mode: str) -> Counter:
    bag: Counter = Counter()

    def add_mons(player):
        for m in _mon_list(player):
            if isinstance(m.get("id"), int):
                bag[m["id"]] += 1

    def add_zone(player, zone):
        for c in player.get(zone) or []:
            if isinstance(c, dict) and isinstance(c.get("id"), int):
                bag[c["id"]] += 1

    pm, po = cur["players"][me], cur["players"][opp]
    # board di entrambi + scarti (pubblici) sempre visibili
    add_mons(pm); add_mons(po)
    add_zone(pm, "discard"); add_zone(po, "discard")
    add_zone(pm, "hand")  # la propria mano la conosco
    if info_mode == "oracle":
        add_zone(po, "hand")
        add_zone(pm, "deck"); add_zone(po, "deck")
    return bag


# --------------------------------------------------------------------------- #
# Feature complete di uno stato da una prospettiva
# --------------------------------------------------------------------------- #
def state_to_features(cur: dict, me: int, meta: dict, info_mode: str = "realistic"):
    """Ritorna (scalar_features: dict, card_bag: Counter) per la prospettiva `me`."""
    opp = 1 - me
    players = cur.get("players") or []
    if len(players) < 2:
        raise ValueError("stato senza due giocatori")

    mine = _side_features(players[me], meta)
    theirs = _side_features(players[opp], meta)

    turn = int(cur.get("turn") or 0)
    first = cur.get("firstPlayer")
    current_player = ((first if isinstance(first, int) else 0) + max(turn - 1, 0)) % 2

    feats = {"turn": turn,
             "is_my_turn": int(current_player == me),
             "i_go_first": int(first == me),
             "has_stadium": int(bool(cur.get("stadium")))}
    for k, v in mine.items():
        feats[f"me_{k}"] = v
    for k, v in theirs.items():
        feats[f"opp_{k}"] = v
    # differenze (segnali forti per un modello lineare)
    for k in mine:
        feats[f"diff_{k}"] = mine[k] - theirs[k]

    bag = _card_bag(cur, me, opp, meta, info_mode)
    return feats, bag


# ordine stabile dei nomi delle feature scalari (indipendente dal dict)
def scalar_feature_names() -> list[str]:
    dummy_side = {
        "prizes_remaining": 0, "pokemon_in_play": 0, "total_hp": 0, "total_damage": 0,
        "total_energy": 0, "n_tools": 0, "n_ex": 0, "n_mega": 0, "n_evolved": 0,
        "max_maxhp": 0, "active_hp": 0, "active_energy": 0, "hand_count": 0,
        "deck_count": 0, "discard_count": 0, "n_poke_discard": 0,
    }
    names = ["turn", "is_my_turn", "i_go_first", "has_stadium"]
    names += [f"me_{k}" for k in dummy_side]
    names += [f"opp_{k}" for k in dummy_side]
    names += [f"diff_{k}" for k in dummy_side]
    return names
