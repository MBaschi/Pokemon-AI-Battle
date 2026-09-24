"""Allena `encoding.MyModel` (value + policy) tramite self-play + MCTS.

Alternativa: train_value_function_from_replays.py allena solo il ramo value
via apprendimento supervisionato su partite reali scaricate, invece che
self-play. Il modello e l'encoding delle carte sono condivisi (encoding.py);
l'agente di gioco vero e proprio (che usa il modello allenato) e' in main.py.
"""

import math
import os
import random
import sys

import torch
import torch.optim

from encoding import (
    LearnInput,
    MyModel,
    build_model,
    eval_nn,
    get_decoder_input,
    get_encoder_input,
)

from cg.api import SearchState, search_begin, search_end, search_step, to_observation_class
from cg.game import battle_start, battle_finish, battle_select

SEARCH_COUNT = 10  # MCTS Search count


# Single Training Sample
class LearnSample:
    def __init__(
        self,
        value: float,
        policy: list[float],
        sv_enc,
        sv_dec,
    ):
        self.value = value  # Encoder output
        self.policy = policy  # Decoder output
        self.sv_enc = sv_enc
        self.sv_dec = sv_dec


# MCTS Node Child
class Child:
    node: "Node | None"
    select: list[int]  # Selected option indices
    prob: float  # Probability

    def __init__(self, select: list[int], prob: float):
        self.node = None
        self.select = select
        self.prob = prob


# MCTS Node
class Node:
    value: float  # Self value
    total: float  # Total value
    visit: int  # Visit count
    parent: "Node | None"  # Parent node
    children: list[Child]
    state: SearchState  # Search State of this node

    def __init__(self, parent: "Node | None", state: SearchState):
        self.value = -2.0
        self.total = 0.0
        self.visit = 0
        self.parent = parent
        self.children = []
        self.state = state

    # Backpropagation value
    def backprop(self, value: float):
        self.total += value
        self.visit += 1
        if self.parent != None:
            self.parent.backprop(value)


def create_node(
    parent: Node | None,
    search_state: SearchState,
    your_index: int,
    your_deck: list[int],
    model: MyModel,
) -> tuple[Node, LearnSample | None]:
    node = Node(parent, search_state)

    obs = search_state.observation
    state = obs.current
    if state.result >= 0:
        # Battle finished
        if state.result == 2:
            node.value = 0
        elif state.result == your_index:
            node.value = 1
        else:
            node.value = -1
        node.backprop(node.value)
        sample = None
    else:
        actions = []
        indices = list(range(obs.select.maxCount))
        for _ in range(64):
            actions.append(indices.copy())
            for i in range(len(indices)):
                index = len(indices) - i - 1
                if indices[index] < len(obs.select.option) - i - 1:
                    indices[index] += 1
                    for j in range(index + 1, len(indices)):
                        indices[j] = indices[j - 1] + 1
                    break
            else:
                break

        sv_enc = get_encoder_input(obs, your_deck)
        sv_dec = get_decoder_input(obs, actions)
        value, policy = eval_nn(sv_enc, sv_dec, model)
        v = value
        if state.yourIndex != your_index:
            v = -v
        node.value = v
        node.backprop(v)

        sum = 0.0
        for i in range(len(policy)):
            p = math.exp(policy[i] * 10.0)
            node.children.append(Child(actions[i], p))
            sum += p
        for c in node.children:
            c.prob /= sum
        sample = LearnSample(value, policy, sv_enc, sv_dec)

    return (node, sample)


# We will perform exploration using MCTS and select actions. At the same time, we will also generate training data.
def mcts_agent(
    obs_dict: dict, your_deck: list[int], model: MyModel
) -> tuple[list[int], LearnSample]:
    obs = to_observation_class(obs_dict)
    your_index = obs.current.yourIndex
    state = obs.current
    active = state.players[1 - your_index].active
    search_state = search_begin(
        obs,
        your_deck=random.sample(
            your_deck, state.players[your_index].deckCount
        ),  # Randomly select from deck.
        your_prize=random.sample(
            your_deck, len(state.players[your_index].prize)
        ),  # Randomly select from deck.
        opponent_deck=[1072]
        * state.players[
            1 - your_index
        ].deckCount,  # Fill with Snorlax (There is no deep meaning).
        opponent_prize=[1]
        * len(
            state.players[1 - your_index].prize
        ),  # Fill with Basic Energy (There is no deep meaning)
        opponent_hand=[1]
        * state.players[1 - your_index].handCount,  # Fill with Basic Energy.
        opponent_active=[1072] if len(active) > 0 and active[0] == None else [],
    )  # Fill with Snorlax.
    root, sample = create_node(
        None, search_state, your_index, your_deck, model
    )  # Create root node.

    # Search
    for _ in range(SEARCH_COUNT):
        current = root
        while True:
            value = -1e9
            c = 0.4 * math.sqrt(current.visit)
            for child in current.children:
                visit = 0
                if child.node == None:
                    v = current.total / current.visit
                else:
                    v = child.node.total / child.node.visit
                    visit = child.node.visit
                if current.state.observation.current.yourIndex != your_index:
                    v = -v
                v += c * child.prob / (1 + visit)
                if value < v:
                    value = v
                    next = child

            if next.node == None:
                search_state = search_step(current.state.searchId, next.select)
                next.node, _ = create_node(
                    current, search_state, your_index, your_deck, model
                )
                break
            else:
                current = next.node
                if current.state.observation.current.result >= 0:
                    current.backprop(current.value)
                    break

    # Select the most visited node.
    max_child = None
    max_visit = -1
    min_value = 10
    for child in root.children:
        if child.node != None:
            if max_visit < child.node.visit:
                max_child = child
                max_visit = child.node.visit
            v = child.node.total / child.node.visit
            if min_value > v:
                min_value = v

    # Generate training data
    sample.value = root.total / root.visit
    for i in range(len(root.children)):
        child = root.children[i]
        v = sample.value
        if child.node == None:
            v = min_value - v - 0.03
        else:
            v = child.node.total / child.node.visit - v
        sample.policy[i] = max(-1.0, min(1.0, v))

    search_end()
    return (max_child.select, sample)


# Opponent for evaluation.
def random_agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    return random.sample(
        list(range(len(obs.select.option))), obs.select.maxCount
    )  # Select at random.


# For displaying progress.
def progress(count: int, text: str):
    current = 0
    while True:
        percent = 100 * current // count
        sys.stderr.write(f"\r{text} {percent}%   ")
        sys.stderr.flush()
        if current >= count:
            sys.stderr.write("\n")
            sys.stderr.flush()
            break
        yield current
        current += 1


# A sample deck for training.
sample_deck = [
    721,
    721,
    722,
    722,
    722,
    722,
    723,
    723,
    723,
    723,
    1092,
    1121,
    1121,
    1145,
    1145,
    1163,
    1163,
    1219,
    1219,
    1219,
    1219,
    1227,
    1227,
    1227,
    1227,
    1262,
    1262,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model()
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn_enc = torch.nn.HuberLoss(delta=0.2)  # Encoder loss function
loss_fn_dec = torch.nn.HuberLoss(reduction="none", delta=0.1)  # Decoder loss function
os.makedirs("out", exist_ok=True)

# The main training loop.
for counter in range(5):
    torch.save(
        model.state_dict(), "out/model" + str(counter) + ".pth"
    )  # Save the current model.
    sample_list: list[LearnSample] = []  # List of training data samples.

    model.eval()
    with torch.inference_mode():
        # Evaluation
        results = [0, 0, 0]

        for i in progress(50, "Evaluating... "):
            obs, start_data = battle_start(sample_deck, sample_deck)
            if start_data.errorPlayer >= 0:
                error = "Deck error."
                if start_data.errorType == 1:
                    error = "The deck contains invalid card ID."
                elif start_data.errorType == 2:
                    error = "You can include up to four cards with the same name in the deck, excluding basic Energy cards."
                elif start_data.errorType == 3:
                    error = "There are no Basic Pokémon in the deck."
                elif start_data.errorType == 4:
                    error = "You can include only one Ace Spec card in the deck."
                raise ValueError(error)
            your_index = i % 2
            while True:
                # Break the loop if the game has ended.
                if obs["current"]["result"] >= 0:
                    break

                if obs["current"]["yourIndex"] == your_index:
                    selected, _ = mcts_agent(obs, sample_deck, model)
                else:
                    selected = random_agent(obs)
                obs = battle_select(selected)

            battle_finish()  # Finalize the game.

            if obs["current"]["result"] == 2:  # Draw
                results[2] += 1
            elif obs["current"]["result"] == your_index:  # Win
                results[0] += 1
            else:  # Lose
                results[1] += 1
        print(
            "Evaluation win rate "
            + str(100 * results[0] // (results[0] + results[1]))
            + "%",
            flush=True,
        )

        # Self Play
        for _ in progress(100, "Training Data Collecting... "):
            obs, _ = battle_start(sample_deck, sample_deck)
            samples: list[list[LearnSample]] = [
                [],
                [],
            ]  # [Player0 samples, Player1 samples]
            while True:
                if obs["current"]["result"] >= 0:
                    break

                # The MCTS agent generates an action and a training sample.
                selected, sample = mcts_agent(obs, sample_deck, model)
                samples[obs["current"]["yourIndex"]].append(sample)
                obs = battle_select(selected)

            battle_finish()  # Finalize the game.

            # Calculate the training labels and add them to the training data list.
            for i in range(2):
                LAMBDA = 0.9
                # The final value is 1.0 for a win and -1.0 for a loss.
                value = 1.0 if i == obs["current"]["result"] else -1.0

                # Iterate backwards from the end of the game to calculate values.
                for sample in reversed(samples[i]):
                    label = (value + sample.value) * 0.5
                    value = value * LAMBDA + sample.value * (1.0 - LAMBDA)
                    sample.value = label
                    sample_list.append(sample)

    # Train on the training data collected through self-play.
    print("Training Start.")
    model.train()
    random.shuffle(sample_list)
    BATCH_SIZE = 128
    batch_count = len(sample_list) // BATCH_SIZE
    for i in range(batch_count):
        # Prepare a batch of data.
        input_enc = LearnInput()
        input_dec = LearnInput()
        mask = []
        label_enc = []
        label_dec = []
        start = BATCH_SIZE * i
        for j in range(start, start + BATCH_SIZE):
            sample = sample_list[j]
            input_enc.add(sample.sv_enc)
            input_dec.add(sample.sv_dec)
            label_enc.append(sample.value)
            label_dec.extend(sample.policy)
            for _ in range(len(sample.policy)):
                mask.append(1.0)
            for _ in range(64 - len(sample.policy)):
                mask.append(0.0)
                label_dec.append(0.0)
                input_dec.offset.append(len(input_dec.index))

        # Convert data to PyTorch tensors.
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)
        mask_tensor = mask_tensor.view(BATCH_SIZE, -1)
        label_tensor_enc = torch.tensor(label_enc, dtype=torch.float32, device=device)
        label_tensor_enc = label_tensor_enc.view(BATCH_SIZE, -1)
        label_tensor_dec = torch.tensor(label_dec, dtype=torch.float32, device=device)
        label_tensor_dec = label_tensor_dec.view(BATCH_SIZE, -1)

        optimizer.zero_grad()

        # Get model predictions for the batch.
        out_enc, out_dec = model(
            torch.tensor(input_enc.index, dtype=torch.int32, device=device),
            torch.tensor(input_enc.value, dtype=torch.float32, device=device),
            torch.tensor(input_enc.offset, dtype=torch.int32, device=device),
            torch.tensor(input_dec.index, dtype=torch.int32, device=device),
            torch.tensor(input_dec.value, dtype=torch.float32, device=device),
            torch.tensor(input_dec.offset, dtype=torch.int32, device=device),
        )

        # Calculate loss.
        loss_enc = loss_fn_enc(out_enc, label_tensor_enc)
        loss_dec = loss_fn_dec(out_dec, label_tensor_dec)
        loss_dec = loss_dec * mask_tensor
        loss_dec = loss_dec.sum() / float(BATCH_SIZE)
        loss = loss_enc + loss_dec

        # Backpropagate the loss and update model parameters.
        loss.backward()
        optimizer.step()
    print("Training Finish.")
