"""
Helper condivisi per i notebook di meta-analysis (Pokémon TCG AI Battle Challenge).

Contiene:
  * gestione path del progetto
  * abilitazione TLS che usa il trust store del sistema operativo
    (necessaria perché sulla macchina di sviluppo c'è un proxy che rompe la
    verifica dei certificati con il bundle di default di Python)
  * fetch HTTP con cache su disco e throttling "gentile"
  * caricamento + arricchimento del dataset carte della competizione
    (EN_Card_Data.csv)
  * parser per la meta di limitlesstcg.com (tabella archetipi + core cards)
  * un piccolo layer opzionale per la Kaggle API (leaderboard)

Il modulo è pensato per essere importato dai notebook, ma può anche essere
lanciato come script (`python meta_utils.py`) per una self-verification veloce.
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Path del progetto
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DOC = ROOT / "doc" / "pokemon-tcg-ai-battle"
EN_CARD_CSV = DOC / "EN_Card_Data.csv"
JP_CARD_CSV = DOC / "JP_Card_Data.csv"
SAMPLE_SUBMISSION_DECK = DOC / "sample_submission" / "deck.csv"
AGENTS_DIR = ROOT / "agents"

OUTPUT_DIR = HERE / "output"
FIG_DIR = OUTPUT_DIR / "figures"
DATA_DIR = OUTPUT_DIR / "data"
CACHE_DIR = HERE / ".cache"
for _d in (OUTPUT_DIR, FIG_DIR, DATA_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# TLS / rete
# --------------------------------------------------------------------------- #
_TLS_MODE = None


def enable_tls(verbose: bool = True) -> str:
    """Configura il modulo ssl per usare il trust store del sistema.

    Ritorna una stringa che indica la strategia usata: ``truststore`` (ideale,
    include eventuali CA aziendali/proxy), ``certifi`` oppure ``unverified``
    (ultima spiaggia: disabilita la verifica TLS).
    """
    global _TLS_MODE
    if _TLS_MODE:
        return _TLS_MODE
    try:
        import truststore

        truststore.inject_into_ssl()
        _TLS_MODE = "truststore"
    except Exception:
        try:
            import ssl

            import certifi

            ssl._create_default_https_context = lambda *a, **k: ssl.create_default_context(
                cafile=certifi.where()
            )
            _TLS_MODE = "certifi"
        except Exception:
            import ssl

            ssl._create_default_https_context = ssl._create_unverified_context
            _TLS_MODE = "unverified"
    if verbose:
        print(f"[meta_utils] TLS mode: {_TLS_MODE}")
    return _TLS_MODE


_UA = "Mozilla/5.0 (compatible; PTCG-meta-analysis/1.0; research/non-commercial)"


def http_get(
    url: str,
    cache: bool = True,
    ttl_hours: float = 24.0,
    throttle: float = 0.7,
    timeout: int = 30,
) -> str:
    """GET con cache su disco (``.cache/``) e piccola pausa tra le richieste.

    La cache evita di ribombardare limitlesstcg ad ogni run del notebook ed è
    fondamentale per la buona educazione verso il sito. ``ttl_hours`` controlla
    la freschezza; imposta ``cache=False`` per forzare il refetch.
    """
    enable_tls(verbose=False)
    key = urllib.parse.quote(url, safe="")
    fp = CACHE_DIR / f"{key}.html"
    if cache and fp.exists():
        age_h = (time.time() - fp.stat().st_mtime) / 3600
        if age_h < ttl_hours:
            return fp.read_text(encoding="utf-8", errors="ignore")
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        html = resp.read().decode("utf-8", "ignore")
    if cache:
        fp.write_text(html, encoding="utf-8")
    if throttle:
        time.sleep(throttle)
    return html


# --------------------------------------------------------------------------- #
# Dataset carte della competizione
# --------------------------------------------------------------------------- #
_STAGE_COL = "Stage (Pokémon)/Type (Energy and Trainer)"


def _first_int(x) -> float:
    """Estrae il primo numero intero da una stringa tipo '50+', '30×', '120'."""
    if pd.isna(x):
        return float("nan")
    m = re.search(r"\d+", str(x))
    return float(m.group()) if m else float("nan")


def _energy_cost(x) -> float:
    """Conta i simboli di energia in un costo tipo '{R}{R}●' -> 3.

    ``{X}`` = energia di tipo specifico, ``●`` = energia incolore/qualsiasi.
    """
    if pd.isna(x):
        return float("nan")
    s = str(x)
    return float(len(re.findall(r"\{[^}]*\}", s)) + s.count("●"))


def load_cards(path: Path = EN_CARD_CSV) -> pd.DataFrame:
    """Carica EN_Card_Data.csv e aggiunge colonne derivate utili all'analisi.

    Il CSV ha una riga per *mossa*: una carta può comparire su più righe
    (es. Pokémon con più attacchi). Colonne aggiunte:
      hp_num, damage_num, energy_cost, dmg_per_energy, subtype, is_pokemon,
      is_trainer, is_energy, is_ex, is_mega, is_ace_spec, energy_type.
    """
    df = pd.read_csv(path)
    df["hp_num"] = df["HP"].apply(_first_int)
    df["damage_num"] = df["Damage"].apply(_first_int)
    df["energy_cost"] = df["Cost"].apply(_energy_cost)
    df["dmg_per_energy"] = df["damage_num"] / df["energy_cost"].replace(0, pd.NA)

    stage = df[_STAGE_COL].fillna("")
    df["subtype"] = stage
    df["is_pokemon"] = stage.str.contains("Pokémon", na=False)
    df["is_trainer"] = stage.isin(["Item", "Supporter", "Pokémon Tool", "Stadium"])
    df["is_energy"] = stage.str.contains("Energy", na=False)

    rule = df["Rule"].fillna("")
    df["is_ex"] = rule.str.contains("ex", case=False, na=False)
    df["is_mega"] = rule.str.contains("Mega", case=False, na=False)
    df["is_ace_spec"] = rule.str.contains("ACE SPEC", case=False, na=False)
    df["energy_type"] = df["Type"].fillna("")
    return df


def cards_unique(df: pd.DataFrame) -> pd.DataFrame:
    """Collassa il dataset a una riga per carta (Card ID).

    Aggrega le mosse: tiene il danno massimo, il costo/efficienza della mossa
    più efficiente e il numero di mosse.
    """
    g = df.groupby("Card ID")
    base_cols = [
        "Card Name",
        "Expansion",
        "subtype",
        "Rule",
        "hp_num",
        "energy_type",
        "Weakness",
        "Retreat",
        "is_pokemon",
        "is_trainer",
        "is_energy",
        "is_ex",
        "is_mega",
        "is_ace_spec",
    ]
    out = g.agg(
        **{c: (c, "first") for c in base_cols},
        n_moves=("Move Name", lambda s: s.notna().sum()),
        max_damage=("damage_num", "max"),
        best_dmg_per_energy=("dmg_per_energy", "max"),
    ).reset_index()
    return out


# --------------------------------------------------------------------------- #
# limitlesstcg.com  (meta reale dei tornei ufficiali)
# --------------------------------------------------------------------------- #
LIMITLESS = "https://limitlesstcg.com"


def fetch_meta_table(fmt: str = "standard", **kw) -> pd.DataFrame:
    """Scarica la tabella meta degli archetipi (limitlesstcg.com/decks).

    Ritorna un DataFrame con: rank, archetype, deck_id, points, share_pct.
    """
    from bs4 import BeautifulSoup

    html = http_get(f"{LIMITLESS}/decks?format={fmt}", **kw)
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    rows = []
    for tr in table.find_all("tr")[1:]:
        tds = tr.find_all("td")
        if len(tds) < 5:
            continue
        link = tr.find("a", href=True)
        deck_id = None
        if link and "/decks/" in link["href"]:
            deck_id = link["href"].rsplit("/", 1)[-1]
        rows.append(
            {
                "rank": int(tds[0].get_text(strip=True) or 0),
                "archetype": tds[2].get_text(" ", strip=True),
                "deck_id": deck_id,
                "points": int(re.sub(r"\D", "", tds[3].get_text()) or 0),
                "share_pct": float(tds[4].get_text(strip=True).replace("%", "") or 0),
            }
        )
    return pd.DataFrame(rows)


_SHARE_RE = re.compile(r"(\d+)\s+in\s+([\d.]+)%")


def fetch_deck_core(deck_id: str, fmt: str = "standard", **kw) -> pd.DataFrame:
    """Scarica le 'core card' (carte staple) di un archetipo.

    Ritorna un DataFrame con: set, number, copies, inclusion_pct, card_url.
    Il nome carta va risolto a parte con :func:`resolve_card_name` (richiede
    una richiesta extra per carta, quindi è messo in cache).
    """
    from bs4 import BeautifulSoup

    html = http_get(f"{LIMITLESS}/decks/{deck_id}?format={fmt}", **kw)
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for cc in soup.select(".core-card"):
        img = cc.find("img")
        share = cc.find("span", class_="share")
        if img is None or share is None:
            continue
        m = _SHARE_RE.search(share.get_text(" ", strip=True))
        a = cc.find("a", href=True)
        rows.append(
            {
                "set": img.get("data-set"),
                "number": img.get("data-number"),
                "copies": int(m.group(1)) if m else None,
                "inclusion_pct": float(m.group(2)) if m else None,
                "card_url": (LIMITLESS + a["href"]) if a else None,
            }
        )
    return pd.DataFrame(rows)


_name_cache: dict[str, str] = {}


def resolve_card_name(set_code: str, number: str, **kw) -> str:
    """Risolve il nome di una carta da set+numero (con cache su disco+memoria)."""
    key = f"{set_code}/{number}"
    if key in _name_cache:
        return _name_cache[key]
    cache_file = CACHE_DIR / "card_names.json"
    if not _name_cache and cache_file.exists():
        _name_cache.update(json.loads(cache_file.read_text(encoding="utf-8")))
        if key in _name_cache:
            return _name_cache[key]
    try:
        from bs4 import BeautifulSoup

        html = http_get(f"{LIMITLESS}/cards/{set_code}/{number}", **kw)
        title = BeautifulSoup(html, "lxml").title.get_text(strip=True)
        name = title.split(" - ")[0].strip()
    except Exception:
        name = key
    _name_cache[key] = name
    cache_file.write_text(json.dumps(_name_cache, ensure_ascii=False, indent=0), encoding="utf-8")
    return name


# --------------------------------------------------------------------------- #
# Kaggle (opzionale, richiede credenziali dell'utente in ~/.kaggle/kaggle.json)
# --------------------------------------------------------------------------- #
def kaggle_leaderboard(competition: str = "pokemon-tcg-ai-battle") -> pd.DataFrame | None:
    """Scarica la leaderboard pubblica via Kaggle API.

    Richiede il pacchetto ``kaggle`` e un token in ``~/.kaggle/kaggle.json``.
    Ritorna ``None`` (senza sollevare) se le credenziali non sono presenti, in
    modo che il notebook resti eseguibile anche senza account Kaggle.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        entries = api.competition_leaderboard_view(competition)
        return pd.DataFrame(
            [{"rank": i + 1, "team": e.teamName, "score": e.score, "submitted": e.submissionDate}
             for i, e in enumerate(entries)]
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[meta_utils] leaderboard non disponibile ({type(exc).__name__}: {exc}).")
        print("  -> installa 'kaggle' e metti kaggle.json in ~/.kaggle/ per abilitarla.")
        return None


# --------------------------------------------------------------------------- #
# Self-check
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("Self-check meta_utils")
    cards = load_cards()
    print(f"  cards rows={len(cards)}  unique={cards['Card ID'].nunique()}")
    uni = cards_unique(cards)
    print(f"  unique cards frame: {uni.shape}")
    top = uni.sort_values("max_damage", ascending=False).head(3)
    print("  top damage:", list(zip(top["Card Name"], top["max_damage"])))
    try:
        meta = fetch_meta_table()
        print(f"  limitless meta rows: {len(meta)}; top: {meta.iloc[0]['archetype']}")
    except Exception as exc:  # noqa: BLE001
        print(f"  limitless fetch failed: {exc}")
