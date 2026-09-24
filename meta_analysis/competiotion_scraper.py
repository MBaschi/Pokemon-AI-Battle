"""
Scraper per Pokémon TCG AI Battle Challenge (Kaggle) → DB SQLite delle partite
==============================================================================

Pipeline (interamente PUBBLICA, senza credenziali):

    GetCompetition(nome)        → competitionId          [API interna Kaggle]
    GetLeaderboard(competId)    → teamId, submissionId   [API interna Kaggle]
    ListEpisodes(submissionId)  → episodeId, reward      [API interna Kaggle]
    replay.json(episodeId)      → mazzi + log giocate    [endpoint pubblico]
                                → parsing → SQLite (matches / match_decks / cards_played)

Un notebook dedicato (03_competition_matches_analysis.ipynb) legge poi il DB.

>>> NIENTE kaggle.json NECESSARIO <<<
    Tutti gli endpoint usati sono quelli che consuma il frontend del sito e sono
    accessibili senza login (serve solo il token anti-CSRF di sessione, che lo
    script ottiene da solo). Verificato sul competitionId 116727.
    (La vecchia via `kaggle competitions episodes/team-submissions`, che invece
     richiede autenticazione, non è più necessaria.)

REQUISITI
---------
    pip install pandas requests truststore

USO (CLI)
---------
    python competiotion_scraper.py check                 # verifica accesso API
    python competiotion_scraper.py inspect <episode_id>  # stampa lo schema di un replay
    python competiotion_scraper.py scrape --teams 15 --episodes 8
    python competiotion_scraper.py demo                  # DB sintetico (offline)

SCHEMA DEL REPLAY (verificato su un episodio reale)
---------------------------------------------------
    replay["info"]["Agents"]          -> [{"Name": ...}, {"Name": ...}]
    replay["rewards"]                 -> [r0, r1]  (es. [-1, 1])
    replay["steps"][0][0]["visualize"][0]["action"]
                                      -> [[60 card id p0], [60 card id p1]]  (mazzi)
    replay["steps"][k][p]["observation"]["logs"]
                                      -> [{"cardId","playerIndex","serial","type"}, ...]
    I card id sono interi che corrispondono a "Card ID" di EN_Card_Data.csv.
"""
from __future__ import annotations

import argparse
import http.cookiejar
import json
import sqlite3
import time
import urllib.request

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    import truststore  # usa il trust store dell'OS (necessario dietro proxy)
    truststore.inject_into_ssl()
except Exception:  # noqa: BLE001
    pass

# --------------------------------------------------------------------------- #
# Config / path
# --------------------------------------------------------------------------- #
COMPETITION = "pokemon-tcg-ai-battle"
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "ptcg_data"
REPLAY_DIR = OUTPUT_DIR / "replays"
DB_PATH = OUTPUT_DIR / "matches.db"
DEMO_DB_PATH = OUTPUT_DIR / "matches_demo.db"
for _d in (OUTPUT_DIR, REPLAY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

REPLAY_URL = "https://www.kaggle.com/competitions/episodes/{episode_id}/replay.json"
SLEEP_BETWEEN_REQUESTS = 0.8
UA = "Mozilla/5.0 (compatible; PTCG-meta-analysis/1.0)"


# --------------------------------------------------------------------------- #
# Client API interna Kaggle (pubblica, senza login — solo token anti-CSRF)
# --------------------------------------------------------------------------- #
class KaggleWeb:
    BASE = "https://www.kaggle.com/api/i"

    def __init__(self, competition: str = COMPETITION):
        self.cj = http.cookiejar.CookieJar()
        self.op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.cj))
        self.op.addheaders = [("User-Agent", UA)]
        # warm-up: ottiene i cookie di sessione (XSRF-TOKEN)
        try:
            self.op.open(f"https://www.kaggle.com/competitions/{competition}", timeout=25).read()
        except Exception:  # noqa: BLE001
            pass
        self.xsrf = next((c.value for c in self.cj if c.name == "XSRF-TOKEN"), "")

    def _post(self, service_method: str, body: dict) -> dict:
        headers = {
            "User-Agent": UA, "Content-Type": "application/json", "Accept": "application/json",
            "x-xsrf-token": self.xsrf, "x-requested-with": "kaggle-web-client",
        }
        req = urllib.request.Request(f"{self.BASE}/{service_method}",
                                     data=json.dumps(body).encode(), headers=headers, method="POST")
        with self.op.open(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8", "ignore"))

    def competition_id(self, name: str = COMPETITION) -> int:
        return int(self._post("competitions.CompetitionService/GetCompetition",
                              {"competitionName": name})["id"])

    def leaderboard(self, competition_id: int) -> list[dict]:
        data = self._post("competitions.LeaderboardService/GetLeaderboard",
                          {"competitionId": competition_id})
        return data.get("publicLeaderboard", [])

    def episodes(self, submission_id: int) -> list[dict]:
        data = self._post("competitions.EpisodeService/ListEpisodes",
                          {"submissionId": int(submission_id)})
        return data.get("episodes", [])


def check_access() -> bool:
    """Verifica che l'API pubblica risponda (nessuna credenziale richiesta)."""
    try:
        api = KaggleWeb()
        cid = api.competition_id()
        lb = api.leaderboard(cid)
        print(f"OK — competitionId={cid}, leaderboard pubblica con {len(lb)} team (nessun login).")
        if lb:
            top = lb[0]
            print(f"     top: team {top.get('teamId')} sub {top.get('submissionId')} score {top.get('displayScore')}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[!] API non raggiungibile: {type(exc).__name__}: {exc}")
        return False


# --------------------------------------------------------------------------- #
# Replay (endpoint pubblico, con cache su disco)
# --------------------------------------------------------------------------- #
def fetch_replay(episode_id) -> dict | None:
    cache = REPLAY_DIR / f"{episode_id}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache.unlink(missing_ok=True)
    resp = requests.get(REPLAY_URL.format(episode_id=episode_id), timeout=30,
                        headers={"User-Agent": UA})
    if resp.status_code != 200:
        print(f"  [!] HTTP {resp.status_code} per episodio {episode_id}")
        return None
    cache.write_text(resp.text, encoding="utf-8")
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    return resp.json()


# --------------------------------------------------------------------------- #
# Parsing del replay (schema REALE, verificato)
# --------------------------------------------------------------------------- #
def parse_replay(replay: dict, episode_id) -> dict:
    """Estrae agenti, esito, mazzi iniziali (60+60 id) e sequenza di carte giocate."""
    info = replay.get("info", {}) if isinstance(replay, dict) else {}
    agents = [a.get("Name") for a in info.get("Agents", [])]
    rewards = replay.get("rewards") or []
    steps = replay.get("steps", [])

    # --- mazzi: steps[0][0].visualize[0].action = [[60 id p0], [60 id p1]] ---
    decks = {0: [], 1: []}
    try:
        action = steps[0][0]["visualize"][0]["action"]
        if isinstance(action, list) and len(action) >= 2:
            decks[0] = [int(c) for c in action[0]]
            decks[1] = [int(c) for c in action[1]]
    except (IndexError, KeyError, TypeError):
        pass

    # --- giocate: observation.logs, deduplicati per 'serial' ---
    by_serial: dict = {}
    for step in steps:
        players = step if isinstance(step, list) else [step]
        for pl in players:
            if not isinstance(pl, dict):
                continue
            for log in (pl.get("observation", {}) or {}).get("logs", []) or []:
                if isinstance(log, dict) and "serial" in log and "cardId" in log:
                    by_serial[log["serial"]] = log
    play_sequence = [
        {"serial": s, "playerIndex": lg.get("playerIndex"),
         "cardId": lg.get("cardId"), "type": str(lg.get("type"))}
        for s, lg in sorted(by_serial.items(), key=lambda kv: (kv[0] is None, kv[0]))
    ]

    winner = None
    if len(rewards) == 2 and rewards[0] is not None and rewards[1] is not None and rewards[0] != rewards[1]:
        winner = 0 if rewards[0] > rewards[1] else 1

    return {
        "episode_id": str(episode_id),
        "agents": agents,
        "rewards": list(rewards),
        "winner": winner,
        "n_steps": len(steps),
        "n_plays": len(play_sequence),
        "deck_player0": decks[0],
        "deck_player1": decks[1],
        "play_sequence": play_sequence,
    }


def inspect_replay(episode_id) -> None:
    replay = fetch_replay(episode_id)
    if not replay:
        print("Replay non scaricato."); return
    print("Top-level keys:", list(replay.keys()))
    print("info:", json.dumps(replay.get("info", {}))[:300])
    print("rewards:", replay.get("rewards"), "| n_steps:", len(replay.get("steps", [])))
    p = parse_replay(replay, episode_id)
    print("\n--- parse_replay ---")
    print("agents:", p["agents"], "| winner:", p["winner"])
    print("deck0 (60):", p["deck_player0"][:12], "...")
    print("n_plays:", p["n_plays"], "| prime giocate:", p["play_sequence"][:5])


# --------------------------------------------------------------------------- #
# DB SQLite
# --------------------------------------------------------------------------- #
SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    episode_id TEXT PRIMARY KEY,
    agent0 TEXT, agent1 TEXT,
    reward0 REAL, reward1 REAL,
    winner_idx INTEGER, n_steps INTEGER, n_plays INTEGER,
    scraped_at TEXT
);
CREATE TABLE IF NOT EXISTS match_decks (
    episode_id TEXT, player_idx INTEGER,
    card_id INTEGER, card_name TEXT, copies INTEGER,
    PRIMARY KEY (episode_id, player_idx, card_id)
);
CREATE TABLE IF NOT EXISTS cards_played (
    episode_id TEXT, serial INTEGER, player_idx INTEGER,
    card_id INTEGER, action_type TEXT
);
CREATE INDEX IF NOT EXISTS ix_decks_ep ON match_decks(episode_id);
CREATE INDEX IF NOT EXISTS ix_plays_ep ON cards_played(episode_id);
"""

_CARD_NAMES: dict | None = None


def _card_name(card_id: int):
    global _CARD_NAMES
    if _CARD_NAMES is None:
        import pandas as pd
        csv = HERE.parent / "doc" / "pokemon-tcg-ai-battle" / "EN_Card_Data.csv"
        s = pd.read_csv(csv).drop_duplicates("Card ID").set_index("Card ID")["Card Name"]
        _CARD_NAMES = s.to_dict()
    return _CARD_NAMES.get(card_id)


def get_db(path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def save_episode(conn: sqlite3.Connection, parsed: dict) -> None:
    ep = parsed["episode_id"]
    ag = list(parsed["agents"]) + [None, None]
    rw = list(parsed["rewards"]) + [None, None]
    conn.execute("INSERT OR REPLACE INTO matches VALUES (?,?,?,?,?,?,?,?,?)",
                 (ep, ag[0], ag[1], rw[0], rw[1], parsed["winner"],
                  parsed["n_steps"], parsed["n_plays"],
                  datetime.now(timezone.utc).isoformat(timespec="seconds")))
    conn.execute("DELETE FROM match_decks WHERE episode_id=?", (ep,))
    conn.execute("DELETE FROM cards_played WHERE episode_id=?", (ep,))
    for pidx in (0, 1):
        counts = Counter(int(c) for c in parsed[f"deck_player{pidx}"])
        conn.executemany("INSERT OR REPLACE INTO match_decks VALUES (?,?,?,?,?)",
                         [(ep, pidx, cid, _card_name(cid), n) for cid, n in counts.items()])
    conn.executemany("INSERT INTO cards_played VALUES (?,?,?,?,?)",
                     [(ep, p["serial"], p["playerIndex"], p["cardId"], p["type"])
                      for p in parsed["play_sequence"]])
    conn.commit()


# --------------------------------------------------------------------------- #
# Scrape end-to-end (pubblico)
# --------------------------------------------------------------------------- #
def scrape(top_n_teams: int = 15, episodes_per_submission: int = 8,
           db_path: Path = DB_PATH) -> None:
    api = KaggleWeb()
    cid = api.competition_id()
    lb = api.leaderboard(cid)[:top_n_teams]
    print(f"competitionId={cid} | team da processare: {len(lb)}")
    conn = get_db(db_path)
    n_saved = 0
    for row in lb:
        sub_id = row.get("submissionId")
        rank = row.get("rank")
        if not sub_id:
            continue
        print(f"\n== rank {rank} | submission {sub_id} (score {row.get('displayScore')}) ==")
        try:
            eps = api.episodes(sub_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  [!] episodi: {exc}"); continue
        # episodi più recenti prima
        eps = sorted(eps, key=lambda e: e.get("endTime") or "", reverse=True)
        for ep in eps[:episodes_per_submission]:
            ep_id = ep.get("id")
            replay = fetch_replay(ep_id)
            if not replay:
                continue
            save_episode(conn, parse_replay(replay, ep_id))
            n_saved += 1
            print(f"    episodio {ep_id} salvato")
    conn.close()
    print(f"\nFatto: {n_saved} partite nel DB {db_path}")


# --------------------------------------------------------------------------- #
# DB DIMOSTRATIVO (offline) — utile solo se non c'è rete
# --------------------------------------------------------------------------- #
def make_demo_db(path: Path = DEMO_DB_PATH, n_matches: int = 60, seed: int = 7) -> Path:
    """DB sintetico con la stessa struttura del reale (dati NON reali)."""
    import random
    import pandas as pd

    random.seed(seed)
    if path.exists():
        path.unlink()
    conn = get_db(path)
    csv = HERE.parent / "doc" / "pokemon-tcg-ai-battle" / "EN_Card_Data.csv"
    pool = pd.read_csv(csv)[["Card ID"]].drop_duplicates().sample(120, random_state=seed).reset_index(drop=True)
    agents = ["ChampAI", "GreedyBot", "MCTS-v2", "RuleBased", "RandomBot", "DragapultBot"]
    archetypes = {a: pool.sample(24, random_state=seed + i)["Card ID"].tolist() for i, a in enumerate(agents)}
    strength = {a: 0.9 - 0.14 * i for i, a in enumerate(agents)}
    for m in range(n_matches):
        a0, a1 = random.sample(agents, 2)
        p0 = strength[a0] / (strength[a0] + strength[a1])
        w = 0 if random.random() < p0 else 1
        parsed = {"episode_id": f"demo-{m:04d}", "agents": [a0, a1],
                  "rewards": [1.0, -1.0] if w == 0 else [-1.0, 1.0], "winner": w,
                  "n_steps": random.randint(30, 120), "n_plays": 0,
                  "deck_player0": [], "deck_player1": [], "play_sequence": []}
        for pidx, ag in ((0, a0), (1, a1)):
            cards = archetypes[ag]
            parsed[f"deck_player{pidx}"] = random.choices(cards, k=60)
            for t in range(random.randint(5, 20)):
                parsed["play_sequence"].append(
                    {"serial": m * 100 + t, "playerIndex": pidx, "cardId": random.choice(cards), "type": "4"})
        parsed["n_plays"] = len(parsed["play_sequence"])
        save_episode(conn, parsed)
    conn.close()
    print(f"DB demo creato: {path} ({n_matches} partite sintetiche)")
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Scraper partite PTCG AI Battle → SQLite (pubblico)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="verifica accesso all'API pubblica")
    ins = sub.add_parser("inspect", help="stampa lo schema di UN replay")
    ins.add_argument("episode_id")
    scr = sub.add_parser("scrape", help="popola il DB reale")
    scr.add_argument("--teams", type=int, default=15)
    scr.add_argument("--episodes", type=int, default=8)
    sub.add_parser("demo", help="crea un DB dimostrativo offline")
    args = ap.parse_args()
    if args.cmd == "check":
        check_access()
    elif args.cmd == "inspect":
        inspect_replay(args.episode_id)
    elif args.cmd == "scrape":
        scrape(args.teams, args.episodes)
    elif args.cmd == "demo":
        make_demo_db()


if __name__ == "__main__":
    main()
