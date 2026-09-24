"""Benchmark two agents against each other using the local `cg` engine.

Usage:
    python benchmark_agents.py --agent1 agents/gio_v1/main.py --agent2 agents/teo_1/main.py --games 20

    # same, saving 3 randomly picked games as viewable HTML replays
    python benchmark_agents.py --agent1 ... --agent2 ... --games 20 --save-replays 3

Each `--agentN` argument is the path to that agent's `main.py`. The agent's
`deck.csv` (and any other files it loads relative to its own directory, e.g.
`params.json` or a model checkpoint) is picked up automatically, the same way
it would be when the agent runs standalone.

Both agents are loaded into the same process, so games run back-to-back
without the overhead of spawning subprocesses. To avoid the two `main.py`
modules (and any same-named helper files they import, e.g. `encoding.py`)
clobbering each other in `sys.modules`, each agent is imported under a
private module name and its local helper modules are evicted from
`sys.modules` right after import (see `load_agent`).
"""

import argparse
import importlib.util
import json
import random
import statistics
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cg.game import battle_finish, battle_select, battle_start, visualize_data  # noqa: E402

DEFAULT_REPLAY_DIR = REPO_ROOT / "benchmark_replays"


class LoadedAgent:
    def __init__(self, name: str, fn, deck: list[int], path: Path):
        self.name = name
        self.fn = fn
        self.deck = deck
        self.path = path


def load_agent(main_py: Path, name: str | None = None) -> LoadedAgent:
    """Import an agent's main.py in isolation and return its `agent` fn + deck."""
    main_py = main_py.resolve()
    if not main_py.exists():
        raise FileNotFoundError(main_py)
    agent_dir = main_py.parent
    mod_name = f"_bench_agent_{agent_dir.name}_{id(main_py)}"

    before = set(sys.modules)
    sys.path.insert(0, str(agent_dir))
    try:
        spec = importlib.util.spec_from_file_location(mod_name, main_py)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(agent_dir))
        # Evict helper modules this agent pulled in from its own directory
        # (e.g. `encoding`) so a second agent with a same-named helper file
        # doesn't silently reuse the first agent's cached module.
        for mod in set(sys.modules) - before:
            if mod == mod_name:
                continue
            f = getattr(sys.modules[mod], "__file__", None)
            if f and str(Path(f).resolve()).startswith(str(agent_dir)):
                del sys.modules[mod]

    if not hasattr(module, "agent"):
        raise AttributeError(f"{main_py} has no top-level `agent(obs)` function")

    deck = getattr(module, "my_deck", None)
    if deck is None:
        deck_path = agent_dir / "deck.csv"
        rows = deck_path.read_text().split("\n")
        deck = [int(rows[i]) for i in range(60)]
    if len(deck) != 60:
        raise ValueError(f"{main_py}: deck has {len(deck)} cards, expected 60")

    return LoadedAgent(name or agent_dir.name, module.agent, deck, main_py)


DECK_ERRORS = {
    1: "invalid card ID in deck",
    2: "more than 4 copies of a non-basic-energy card",
    3: "no Basic Pokemon in deck",
    4: "more than one ACE SPEC card in deck",
}


def play_game(agents: tuple[LoadedAgent, LoadedAgent], max_turns: int,
              capture_replay: bool = False) -> dict:
    """Play one game. `agents[i]` occupies player slot i. Returns a result dict.

    With `capture_replay`, the result also carries `steps`: the viewer data of
    the game as produced by the engine (`cg.game.visualize_data`), the same
    format `view_replays/replay_render.py` renders. It has to be read *before*
    `battle_finish` frees the battle, hence the grab in the `finally` block.
    """
    obs, start_data = battle_start(agents[0].deck, agents[1].deck)
    if start_data.errorPlayer is not None and start_data.errorPlayer >= 0:
        reason = DECK_ERRORS.get(start_data.errorType, f"error type {start_data.errorType}")
        raise ValueError(f"deck error for {agents[start_data.errorPlayer].name}: {reason}")

    result = {"result": -2, "turn": obs["current"]["turn"], "crashed": None, "steps": None}
    try:
        while obs["current"]["result"] < 0:
            if obs["current"]["turn"] > max_turns:
                result.update(result=-2, turn=obs["current"]["turn"])
                break
            slot = obs["current"]["yourIndex"]
            try:
                selection = agents[slot].fn(obs)
            except Exception:
                traceback.print_exc()
                result.update(result=1 - slot, turn=obs["current"]["turn"], crashed=agents[slot].name)
                break
            try:
                obs = battle_select(selection)
            except (ValueError, IndexError):
                traceback.print_exc()
                result.update(result=1 - slot, turn=obs["current"]["turn"], crashed=agents[slot].name)
                break
        else:
            result.update(result=obs["current"]["result"], turn=obs["current"]["turn"])
    finally:
        if capture_replay:
            try:
                result["steps"] = json.loads(visualize_data())
            except Exception:
                traceback.print_exc()
        battle_finish()
    return result


def _replay_render():
    """Import the shared renderer in `view_replays/` (it isn't a package)."""
    path = str(REPO_ROOT / "view_replays")
    if path not in sys.path:
        sys.path.insert(0, path)
    import replay_render

    return replay_render


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def save_replay(steps, slots: tuple[LoadedAgent, LoadedAgent], out_dir: Path, index: int) -> Path:
    """Write a self-contained HTML replay named `<slot0>_vs_<slot1>_<datetime>`.

    The agents appear in the order they occupied the player slots in *that*
    game, so with alternating sides the name also tells who moved first. The
    game index keeps two replays from the same second apart.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{_safe(slots[0].name)}_vs_{_safe(slots[1].name)}_{stamp}_g{index}.html"
    # The names also go into the viewer, so the replay says "gharchomp_ex's
    # turn" instead of "Player1's turn".
    _replay_render().generate_html(steps, str(path), [slots[0].name, slots[1].name])
    return path


def run_benchmark(agent1: LoadedAgent, agent2: LoadedAgent, games: int, max_turns: int,
                   alternate_sides: bool, verbose: bool, save_replays: int = 0,
                   replay_dir: Path = DEFAULT_REPLAY_DIR) -> None:
    wins = {agent1.name: 0, agent2.name: 0}
    draws = 0
    aborted = 0
    crashes = {agent1.name: 0, agent2.name: 0}
    turns = []
    saved = []
    # Which games get recorded is drawn up front, so the sample is uniform over
    # the whole run instead of biased towards the first (or last) games.
    to_record = set(random.sample(range(games), min(save_replays, games))) if save_replays > 0 else set()
    t0 = time.time()

    for i in range(games):
        swap = alternate_sides and (i % 2 == 1)
        slots = (agent2, agent1) if swap else (agent1, agent2)

        result = play_game(slots, max_turns, capture_replay=i in to_record)
        turns.append(result["turn"])

        if result["steps"]:
            try:
                path = save_replay(result["steps"], slots, replay_dir, i + 1)
                saved.append(path)
            except Exception:      # a broken renderer must not kill the benchmark
                traceback.print_exc()

        if result["crashed"]:
            crashes[result["crashed"]] += 1
            winner = slots[result["result"]].name
            wins[winner] += 1
            outcome = f"{winner} wins (opponent crashed: {result['crashed']})"
        elif result["result"] == -2:
            aborted += 1
            outcome = f"aborted (exceeded {max_turns} turns)"
        elif result["result"] == 2:
            draws += 1
            outcome = "draw"
        else:
            winner = slots[result["result"]].name
            wins[winner] += 1
            outcome = f"{winner} wins"

        if verbose:
            print(f"game {i + 1}/{games}: {outcome} (turn {result['turn']})")
            if saved and saved[-1].name.endswith(f"_g{i + 1}.html"):
                print(f"    replay -> {saved[-1]}")

    elapsed = time.time() - t0
    decisive = wins[agent1.name] + wins[agent2.name]

    print()
    print("=" * 56)
    print(f"{agent1.name}  vs  {agent2.name}   ({games} games, {elapsed:.1f}s)")
    print("=" * 56)
    for name in (agent1.name, agent2.name):
        rate = 100 * wins[name] / decisive if decisive else 0.0
        print(f"  {name:20s}  wins: {wins[name]:3d}   win rate (of decisive): {rate:5.1f}%   crashes: {crashes[name]}")
    print(f"  draws: {draws}   aborted: {aborted}")
    if turns:
        print(f"  avg turns/game: {statistics.mean(turns):.1f}   median: {statistics.median(turns):.0f}"
              f"   min: {min(turns)}   max: {max(turns)}")
    if saved:
        print(f"  replays saved: {len(saved)} in {Path(replay_dir)}")
    print("=" * 56)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent1", required=True, type=Path, help="path to first agent's main.py")
    parser.add_argument("--agent2", required=True, type=Path, help="path to second agent's main.py")
    parser.add_argument("--name1", default=None, help="display name for agent1 (default: its folder name)")
    parser.add_argument("--name2", default=None, help="display name for agent2 (default: its folder name)")
    parser.add_argument("--games", type=int, default=20, help="number of games to play (default: 20)")
    parser.add_argument("--max-turns", type=int, default=300, help="abort a game past this many turns (default: 300)")
    parser.add_argument("--no-alternate", action="store_true",
                         help="don't alternate which agent goes in player slot 0/1 each game")
    parser.add_argument("--seed", type=int, default=None, help="random seed (affects agents that use `random`)")
    parser.add_argument("--save-replays", type=int, default=0, metavar="N",
                         help="save N randomly picked games as HTML replays (default: 0)")
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR,
                         help=f"where to write them (default: {DEFAULT_REPLAY_DIR.name}/)")
    parser.add_argument("-v", "--verbose", action="store_true", help="print the outcome of every game")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print(f"loading agent1 from {args.agent1} ...")
    agent1 = load_agent(args.agent1, args.name1)
    print(f"loading agent2 from {args.agent2} ...")
    agent2 = load_agent(args.agent2, args.name2)
    if agent1.name == agent2.name:
        agent1.name += " (1)"
        agent2.name += " (2)"

    run_benchmark(agent1, agent2, args.games, args.max_turns, not args.no_alternate, args.verbose,
                  args.save_replays, args.replay_dir)


if __name__ == "__main__":
    main()
