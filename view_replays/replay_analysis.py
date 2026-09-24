"""Parse Kaggle Pokémon-TCG replay JSON into per-match KPIs and agent stats.

Replay JSON schema (downloaded via `kaggle competitions replay <episode_id>`):
    doc["info"]        -> {"EpisodeId":..., "TeamNames":[t0, t1], ...}
    doc["rewards"]     -> [reward_p0, reward_p1]        (winner = higher reward)
    doc["steps"]       -> [[stepP0, stepP1], ...]       standard kaggle-env steps,
                          each step has ["observation"]["current"] game state
    doc["steps"][0][0]["visualize"] -> rich per-UI-step list used by the viewer,
                          each item: {"current": {...}, "logs": [...]}

Archetype hints + the card-walking helpers are adapted from
`ptcg-replay-triage-starter`.
"""
from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

# --- archetype classification (card-ID fingerprints) ------------------------
# Overlap-scored: the bucket sharing the most visible card IDs wins.
ARCHETYPE_HINTS = {
    "archaludon_metal": {169, 190, 840, 666},
    "marnie_grimmsnarl": {646, 647, 648, 860},
    "alakazam_psychic": {741, 742, 743, 66, 305},
    "mega_lucario": {677, 678},
    "ogerpon_toolbox": {116, 117, 134, 712, 713, 748, 1051, 1052, 1256},
    "starmie_froslass": {1030, 1031},
    "great_tusk_crustle": {344, 345, 532},
    "hop_trevenant": {288, 289, 299, 304, 307, 308, 309, 310, 878, 879},
    "chandelure_control": {97, 98, 494},
}

CARD_HINT_KEYS = {
    "hp", "cardType", "energyCards", "tools", "attacks",
    "damage", "preEvolution", "specialConditions",
}


def classify_archetype(card_counts: Counter) -> str:
    ids = set(card_counts)
    scored = [(len(ids & hints), name) for name, hints in ARCHETYPE_HINTS.items()
              if ids & hints]
    return sorted(scored, reverse=True)[0][1] if scored else "unknown"


def walk(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def visible_card_ids(doc: Any, player_index=None) -> Counter:
    """Count every card ID visible in the replay, optionally for one player."""
    counts: Counter = Counter()
    for item in walk(doc):
        if not isinstance(item, dict):
            continue
        if player_index is not None and item.get("playerIndex") != player_index:
            continue
        card_id = item.get("cardId")
        if isinstance(card_id, int):
            counts[card_id] += 1
        generic_id = item.get("id")
        if isinstance(generic_id, int) and CARD_HINT_KEYS.intersection(item.keys()):
            counts[generic_id] += 1
    return counts


# --- card names (for a human-readable archetype signature) ------------------
_CARD_NAMES = None


def load_card_names(data_dir=None) -> dict:
    """{cardId: english name} from EN_Card_Data.csv (cached; {} if not found)."""
    global _CARD_NAMES
    if _CARD_NAMES is not None:
        return _CARD_NAMES
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "card_data")
    path = os.path.join(data_dir, "EN_Card_Data.csv")
    names = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    names[int(row["Card ID"])] = row.get("Card Name", "")
                except (ValueError, KeyError):
                    continue
    _CARD_NAMES = names
    return names


def archetype_signature(card_counts: Counter, top=3) -> str:
    """Human-readable guess: the most-seen Pokémon-ish cards by name."""
    names = load_card_names()
    if not names:
        return ""
    picks = []
    for cid, _ in card_counts.most_common():
        nm = names.get(cid, "")
        # Skip basic energy / generic trainers; keep evolving/ex Pokémon-looking names.
        if not nm or nm.lower().startswith("basic ") or "energy" in nm.lower():
            continue
        picks.append(nm)
        if len(picks) >= top:
            break
    return ", ".join(picks)


# --- per-episode KPI extraction ---------------------------------------------
def _visualize_steps(doc: dict):
    try:
        return doc["steps"][0][0]["visualize"] or []
    except (KeyError, IndexError, TypeError):
        return []


def _final_state(vsteps):
    for step in reversed(vsteps):
        cur = (step or {}).get("current")
        if cur:
            return cur
    return None


def _prizes_taken(player_state) -> int:
    prize = (player_state or {}).get("prize")
    return 6 - len(prize) if isinstance(prize, list) else 0


def winner_from_rewards(rewards):
    """Return winning player index from a rewards pair, or None for a draw/unknown."""
    if not rewards or len(rewards) < 2 or rewards[0] is None or rewards[1] is None:
        return None
    if rewards[0] > rewards[1]:
        return 0
    if rewards[1] > rewards[0]:
        return 1
    return None


def analyze_replay(doc: dict, my_index=None, my_team=None) -> dict:
    """Extract a flat KPI record for one episode from a replay JSON dict.

    `my_index` (0/1) or `my_team` (name matched against info.TeamNames) fixes which
    seat is "you"; defaults to seat 0. KO counts assume a KO log's `playerIndex` is
    the owner of the Pokémon that fainted.
    """
    info = doc.get("info") or {}
    teams = info.get("TeamNames") or []
    rewards = doc.get("rewards") or []

    if my_index is None and my_team and my_team in teams:
        my_index = teams.index(my_team)
    if my_index is None:
        my_index = 0
    opp = 1 - my_index

    winner = winner_from_rewards(rewards)
    won = None if winner is None else (winner == my_index)

    vsteps = _visualize_steps(doc)
    final = _final_state(vsteps)
    turns = final.get("turn") if final else None
    players = (final or {}).get("players") or []

    prizes_me = _prizes_taken(players[my_index]) if len(players) > my_index else None
    prizes_opp = _prizes_taken(players[opp]) if len(players) > opp else None

    # Event tallies from the visualize logs.
    ko = Counter()
    damage = Counter()
    attacks = Counter()
    for step in vsteps:
        for l in (step or {}).get("logs") or []:
            pi = l.get("playerIndex")
            t = l.get("type")
            if t == "KO" and pi is not None:
                ko[pi] += 1
            elif t == "Damage" and pi is not None:
                damage[pi] += int(l.get("damage") or 0)
            elif t == "Attack" and pi is not None:
                attacks[pi] += 1

    my_cards = visible_card_ids(doc, my_index)
    opp_cards = visible_card_ids(doc, opp)

    return {
        "episode_id": info.get("EpisodeId"),
        "my_seat": my_index,
        "my_team": teams[my_index] if len(teams) > my_index else "",
        "opp_team": teams[opp] if len(teams) > opp else "",
        "result": "win" if won else ("loss" if won is False else "draw/unknown"),
        "won": won,
        "reward_me": rewards[my_index] if len(rewards) > my_index else None,
        "reward_opp": rewards[opp] if len(rewards) > opp else None,
        "turns": turns,
        "n_steps": len(vsteps),
        "my_archetype": classify_archetype(my_cards),
        "opp_archetype": classify_archetype(opp_cards),
        "opp_signature": archetype_signature(opp_cards),
        "prizes_taken": prizes_me,
        "prizes_conceded": prizes_opp,
        "opp_pokemon_koed": ko.get(opp),      # opponent Pokémon that fainted = your KOs
        "my_pokemon_koed": ko.get(my_index),  # your Pokémon that fainted
        "damage_dealt": damage.get(my_index),
        "damage_taken": damage.get(opp),
        "my_attacks": attacks.get(my_index),
    }


def analyze_replay_file(path, my_index=None, my_team=None) -> dict:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    rec = analyze_replay(doc, my_index=my_index, my_team=my_team)
    rec["file"] = str(path)
    if rec.get("episode_id") is None:
        rec["episode_id"] = Path(path).stem
    return rec


def analyze_folder(folder, my_index=None, my_team=None) -> pd.DataFrame:
    """Analyze every replay JSON in a folder into an episodes KPI DataFrame."""
    rows = []
    for path in sorted(Path(folder).glob("*.json")):
        try:
            rows.append(analyze_replay_file(path, my_index=my_index, my_team=my_team))
        except Exception as e:  # keep going on a single bad file
            rows.append({"file": str(path), "episode_id": path.stem, "error": str(e)})
    return pd.DataFrame(rows)


# --- agent-level aggregate KPIs ---------------------------------------------
def agent_stats(episodes_df: pd.DataFrame) -> dict:
    """Headline KPIs over an episodes KPI DataFrame (output of analyze_folder)."""
    df = episodes_df[episodes_df.get("won").notna()] if "won" in episodes_df else episodes_df
    n = len(df)
    if n == 0:
        return {"episodes": len(episodes_df), "decided": 0}
    wins = int(df["won"].sum())
    losses = int((~df["won"].astype(bool)).sum())

    def _mean(col):
        return round(float(df[col].dropna().mean()), 2) if col in df and df[col].notna().any() else None

    return {
        "episodes": len(episodes_df),
        "decided": n,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / n, 3),
        "avg_turns": _mean("turns"),
        "avg_prizes_taken": _mean("prizes_taken"),
        "avg_prizes_conceded": _mean("prizes_conceded"),
        "avg_damage_dealt": _mean("damage_dealt"),
        "avg_damage_taken": _mean("damage_taken"),
    }


def merge_episodes_metadata(kpi_df: pd.DataFrame, episodes_df, keywords=("score", "elo", "rating", "seat", "index")) -> pd.DataFrame:
    """Left-join Kaggle's `episodes` listing (ELO/score/seat columns) onto KPIs.

    The exact columns vary by CLI version, so this is best-effort: it finds the
    id column in `episodes_df`, joins on the KPI `episode_id`, and keeps only
    columns whose name matches one of `keywords` (case-insensitive).
    """
    if not isinstance(episodes_df, pd.DataFrame) or episodes_df.empty or kpi_df.empty:
        return kpi_df
    id_col = next((c for c in ("id", "episodeId", "episode_id", "ref")
                   if c in episodes_df.columns), episodes_df.columns[0])
    keep = [id_col] + [c for c in episodes_df.columns
                       if c != id_col and any(k in c.lower() for k in keywords)]
    meta = episodes_df[keep].copy()
    meta = meta.rename(columns={id_col: "episode_id"})
    # Drop rows whose id isn't a real number (e.g. the CLI's trailing help line)
    # and align the id dtype so the merge keys match the int64 KPI ids.
    meta["episode_id"] = pd.to_numeric(meta["episode_id"], errors="coerce")
    meta = meta.dropna(subset=["episode_id"])
    try:
        meta["episode_id"] = meta["episode_id"].astype(kpi_df["episode_id"].dtype)
    except Exception:
        meta["episode_id"] = meta["episode_id"].astype("int64")
    return kpi_df.merge(meta, on="episode_id", how="left")


def winrate_by_opponent(episodes_df: pd.DataFrame) -> pd.DataFrame:
    """Win rate broken down by opponent archetype."""
    if "opp_archetype" not in episodes_df or "won" not in episodes_df:
        return pd.DataFrame(columns=["opp_archetype", "games", "wins", "win_rate"])
    df = episodes_df[episodes_df["won"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["opp_archetype", "games", "wins", "win_rate"])
    df["won"] = df["won"].astype(int)
    g = df.groupby("opp_archetype")["won"]
    out = pd.DataFrame({"games": g.size(), "wins": g.sum()})
    out["win_rate"] = (out["wins"] / out["games"]).round(3)
    return out.reset_index().sort_values("games", ascending=False)
