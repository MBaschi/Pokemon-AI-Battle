"""Thin wrappers around the `kaggle` CLI for the Pokémon TCG competition.

Refactored out of `pokemon_tcg_submission_monitor.ipynb` so the dashboard notebook
stays short. Every call goes through `run()` so you can see the exact CLI command,
and list-style commands are parsed into pandas DataFrames.

Prereqs: `pip install kaggle`, a token at ~/.kaggle/kaggle.json, and having clicked
"I Understand and Accept" on the competition page.
"""
from __future__ import annotations

import io
import os
import shlex
import subprocess
from pathlib import Path

import pandas as pd

COMPETITION = "pokemon-tcg-ai-battle"


def run(args, check=True, echo=True):
    """Run `kaggle <args...>`, optionally echo it, and return (stdout, stderr)."""
    cmd = ["kaggle"] + [str(a) for a in args]
    if echo:
        print("$", " ".join(shlex.quote(a) for a in cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        msg = "kaggle CLI not found on PATH — run `pip install kaggle`."
        if echo:
            print("[error]", msg)
        if check:
            raise RuntimeError(msg)
        return "", msg
    if echo:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(cmd)}")
    return result.stdout, result.stderr


def run_to_df(args, csv_flag="-v", echo=True):
    """Run a `kaggle` command with the CSV flag and parse stdout into a DataFrame.

    Falls back to returning raw text if the subcommand doesn't support -v/--csv.
    """
    cmd_args = list(args)
    if csv_flag and csv_flag not in cmd_args:
        cmd_args = cmd_args + [csv_flag]
    stdout, _ = run(cmd_args, check=False, echo=echo)
    if not stdout.strip():
        return pd.DataFrame()
    try:
        return pd.read_csv(io.StringIO(stdout))
    except Exception:
        if echo:
            print("[info] Could not parse output as CSV — returning raw text.")
        return stdout


# --- high-level helpers -----------------------------------------------------

def list_submissions(competition=COMPETITION, echo=True):
    """All of your submissions to the competition, most recent first."""
    return run_to_df(["competitions", "submissions", competition], echo=echo)


def list_episodes(submission_id, echo=True):
    """Episodes (matches) played by a given submission."""
    return run_to_df(["competitions", "episodes", str(submission_id)], echo=echo)


def leaderboard(competition=COMPETITION, echo=True):
    return run_to_df(["competitions", "leaderboard", competition, "-s"], echo=echo)


def download_replay(episode_id, dest="./replays", echo=True):
    """Download an episode replay JSON, normalising its name to <episode_id>.json.

    The CLI's output filename varies by version (sometimes a generic name that gets
    overwritten each call), so we snapshot the folder before/after and rename the
    freshly written file to a per-episode name. Returns the path, or None on failure.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / f"{episode_id}.json"

    before = {p: p.stat().st_mtime for p in dest.glob("*.json")}
    run(["competitions", "replay", str(episode_id), "-p", str(dest)], echo=echo)

    # Prefer a file the CLI already named after this episode.
    strict = _strict_episode_json(dest, episode_id)
    if strict:
        return strict
    # Otherwise find what the download just created/updated and rename it.
    after = list(dest.glob("*.json"))
    new_or_touched = [p for p in after if p not in before or p.stat().st_mtime > before[p]]
    if not new_or_touched:
        if echo:
            print(f"  [warn] no replay JSON produced for episode {episode_id}")
        return None
    src = max(new_or_touched, key=lambda p: p.stat().st_mtime)
    if src.resolve() != target.resolve():
        src.replace(target)
    return str(target)


def download_logs(episode_id, agent_index=0, dest="./logs", echo=True):
    """Download one agent's stdout/stderr logs for an episode."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    run(["competitions", "logs", str(episode_id), str(agent_index), "-p", str(dest)], echo=echo)
    return dest


def get_replay_path(episode_id, dest="./replays", download=True, echo=True):
    """Return the local replay JSON path for an episode, downloading if needed.

    Caching is strict: a replay counts as already-present only if a file actually
    matches this episode id (never a stray json from another episode).
    """
    existing = _strict_episode_json(Path(dest), episode_id)
    if existing:
        return existing
    if download:
        return download_replay(episode_id, dest=dest, echo=echo)
    return None


def _strict_episode_json(dest: Path, episode_id):
    """Path to the cached replay for exactly this episode, or None.

    Only matches <episode_id>.json or a filename containing the episode id — never
    falls back to 'any json in the folder' (that would make every episode reuse the
    first download).
    """
    if not dest.exists():
        return None
    exact = dest / f"{episode_id}.json"
    if exact.exists():
        return str(exact)
    matches = sorted(dest.glob(f"*{episode_id}*.json"))
    return str(matches[0]) if matches else None


def episode_ids_from_df(df: pd.DataFrame, limit=None):
    """Extract clean integer episode ids from an `episodes` listing DataFrame.

    The CLI appends a help footer line (e.g. 'Use "kaggle competitions replay ..."')
    that pandas parses as a data row, so we coerce the id column to numeric and drop
    anything non-numeric before casting to int.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    col = pick_id_column(df)
    ids = pd.to_numeric(df[col], errors="coerce").dropna().astype("int64").tolist()
    return ids[:limit] if limit else ids


def pick_id_column(df: pd.DataFrame):
    """Best-effort guess of the id column in a submissions/episodes DataFrame."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    for cand in ("id", "ref", "episodeId", "episode_id", "submissionId"):
        if cand in df.columns:
            return cand
    return df.columns[0]
