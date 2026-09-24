import json
import os

from kaggle_environments import make

from main import agent

# Where to write the replay the viewer serves.
# Default matches setup-viewer.sh (./cabt-viewer next to this script);
# override with CABT_REPLAY_PATH if your viewer lives elsewhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_REPLAY = os.path.join(
    _HERE,
    "cabt-viewer",
    "kaggle_environments", "envs", "cabt", "visualizer", "default",
    "replays", "test-replay.json",
)
REPLAY_PATH = os.environ.get("CABT_REPLAY_PATH", _DEFAULT_REPLAY)

with open("deck.csv") as f:
    deck = [int(line) for line in f if line.strip()]

env = make("cabt", configuration={"decks": [deck, deck]})
env.run([agent, agent])

# Export the replay in the exact shape the visualizer expects.
os.makedirs(os.path.dirname(REPLAY_PATH), exist_ok=True)
with open(REPLAY_PATH, "w", encoding="utf-8") as f:
    json.dump(env.toJSON(), f)

final = env.steps[-1]
print("rewards: ", [p["reward"] for p in final])
print("statuses:", [p["status"] for p in final])
print("replay ->", REPLAY_PATH)
print("Simulation finished. Refresh the viewer tab to watch it.")
