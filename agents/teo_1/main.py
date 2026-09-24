"""teo_1 live agent: orchestration only.

Decisions are handled in three tiers, cheapest first:
  1. Forced / no real choice (a single legal option, or you must select
     everything available) -> just take it.
  2. A genuine single choice among alternatives -- the "critical" case -- is
     resolved with a 1-ply action-value search: simulate the immediate
     result of each candidate option and let the trained state-evaluation
     network (encoding.py) rank them.
  3. Genuine multi-select (rare: pick several of many) falls back to a small
     generic, deck-agnostic heuristic, since scoring every combination with
     the network is combinatorially expensive.

Training lives elsewhere: train_selfplay.py (self-play MCTS) and
build_replay_dataset.py / train_value_function_from_replays.py (supervised
value training from real match replays). The network/card-encoding code
itself is in encoding.py.
"""

import os
import random
import sys

import torch

from encoding import build_model, dummy_decoder_input, eval_nn, get_card, get_encoder_input

from cg.api import (
    Observation,
    OptionType,
    Pokemon,
    SelectContext,
    search_begin,
    search_end,
    search_step,
    to_observation_class,
)


def _find_file(name):
    """Locate a bundled file. Kaggle runs main.py via exec() without __file__
    and without chdir, so fall back to the bundled cg package location and
    the fixed /kaggle_simulations/agent path (same approach as gio_v1)."""
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


def _read_deck() -> list[int]:
    path = _find_file("deck.csv")
    with open(path, "r") as f:
        rows = f.read().split("\n")
    return [int(rows[i]) for i in range(60)]


my_deck = _read_deck()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model().to(device)
_checkpoint = os.environ.get("TEO1_CHECKPOINT") or _find_file("out/value_from_replays.pth")
if _checkpoint and os.path.exists(_checkpoint):
    model.load_state_dict(torch.load(_checkpoint, map_location=device))
else:
    print(
        "teo_1: nessun checkpoint trovato (TEO1_CHECKPOINT o out/value_from_replays.pth), "
        "uso pesi non allenati: la value search sara' rumore finche' non alleni un modello.",
        file=sys.stderr,
    )
model.eval()


@torch.inference_mode()
def _state_value(next_obs: Observation, your_index: int, your_deck: list[int]) -> float:
    """Value of `next_obs`'s board from `your_index`'s point of view."""
    state = next_obs.current
    if state.result >= 0:
        if state.result == 2:
            return 0.0
        return 1.0 if state.result == your_index else -1.0
    sv = get_encoder_input(next_obs, your_deck)
    value, _ = eval_nn(sv, dummy_decoder_input(), model)
    return value if next_obs.current.yourIndex == your_index else -value


def _value_search(obs: Observation, your_deck: list[int]) -> int:
    """1-ply action-value search over every legal option: simulate the
    immediate result of each and return the index the value network likes
    most. Same simulation setup train_selfplay.py's MCTS uses (we don't know
    the opponent's exact hand/deck order, so those are filled with
    placeholders), just without building a search tree."""
    your_index = obs.current.yourIndex
    state = obs.current
    opp = state.players[1 - your_index]
    search_state = search_begin(
        obs,
        your_deck=random.sample(your_deck, state.players[your_index].deckCount),
        your_prize=random.sample(your_deck, len(state.players[your_index].prize)),
        opponent_deck=[1072] * opp.deckCount,  # Fill with Snorlax (no deep meaning).
        opponent_prize=[1] * len(opp.prize),  # Fill with Basic Energy.
        opponent_hand=[1] * opp.handCount,  # Fill with Basic Energy.
        opponent_active=[1072] if len(opp.active) > 0 and opp.active[0] is None else [],
    )

    best_index, best_value = 0, -1e9
    for i in range(len(obs.select.option)):
        next_state = search_step(search_state.searchId, [i])
        value = _state_value(next_state.observation, your_index, your_deck)
        if value > best_value:
            best_index, best_value = i, value
    search_end()
    return best_index


def _naive_multiselect(obs: Observation) -> list[int]:
    """Fallback for genuine multi-select decisions (rare: e.g. discard/attach
    several cards at once). Evaluating every combination with the value
    network is combinatorially expensive, so we rank options with a small
    generic (deck-agnostic) score instead."""
    select = obs.select

    def score(o) -> float:
        if o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            value = 0.0
            if isinstance(card, Pokemon):
                value = card.hp + len(card.energyCards) * 20 + len(card.tools) * 10
            if select.context in (SelectContext.DISCARD, SelectContext.DISCARD_ENERGY):
                return -value  # discard-type contexts: get rid of the least valuable cards
            return value
        if o.type == OptionType.NUMBER:
            return o.number
        return 0.0

    ranked = sorted(range(len(select.option)), key=lambda i: score(select.option[i]), reverse=True)
    return ranked[: select.maxCount]


def agent(obs_dict: dict) -> list[int]:
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        # Initial selection: return the 60-card deck.
        return my_deck

    select = obs.select
    n_options = len(select.option)

    # Forced: no real choice (a single option, or every option must be
    # taken) -- e.g. a hand item that just draws cards, with nothing to
    # weigh. Take it without spending a value-search on it.
    if n_options <= select.maxCount:
        return list(range(n_options))

    # A genuine single choice among alternatives -- the critical case: let
    # the trained value network rank the resulting states.
    if select.minCount == 1 and select.maxCount == 1:
        return [_value_search(obs, my_deck)]

    # Genuine multi-select: naive generic fallback (see docstring above).
    return _naive_multiselect(obs)
