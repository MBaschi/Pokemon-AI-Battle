"""Reusable pieces of teo_1's model: card/state encoding, MyModel, batching helpers.

Extracted from main.py so they can be imported both by main.py (MCTS self-play
training) and by other scripts (e.g. build_replay_dataset.py /
train_value_from_replays.py, which train the value head from real match
replays instead of self-play).
"""

import glob
import sys
from pathlib import Path


def _add_cg_to_path():
    kaggle_matches = glob.glob("/kaggle/input/**/cg-lib", recursive=True)
    if kaggle_matches:
        sys.path.append(kaggle_matches[0])
        return
    # Local development: cg/ is not pip-installable, it's manually placed at
    # the repository root from the Kaggle competition's starter materials
    # (see README.md, section "Recuperare l'engine della competizione").
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "cg").is_dir():
        sys.path.append(str(repo_root))
        return
    raise ImportError(
        "cg engine package not found. On Kaggle it's provided via the "
        "competition's attached dataset; locally, download the starter "
        "materials from the Kaggle competition page and place the `cg/` "
        "package (with its .so) at the repository root. See README.md."
    )


_add_cg_to_path()

import torch
import torch.nn
import torch.nn.functional

from cg.api import (
    AreaType,
    Card,
    Observation,
    OptionType,
    PlayerState,
    Pokemon,
    SelectContext,
    all_attack,
    all_card_data,
    to_observation_class,
)

# Load all card data from the API's helper function
all_card = all_card_data()
# Create a lookup table (dictionary) to quickly access card data by its cardId
card_table = {c.cardId: c for c in all_card}
card_count = max(all_card, key=lambda c: c.cardId).cardId + 1  # Max Card ID + 1

attack_count = (
    max(all_attack(), key=lambda a: a.attackId).attackId + 1
)  # Max Attack ID + 1

num_words_encoder = 24
encoder_size = 22000  # Encoder input size exceeding the vocabulary size

decoder_main_feature = 8  # Feature count of SelectContext.Main
decoder_attack_offset = 14  # First index of Attack feature
decoder_card_offset = (
    decoder_attack_offset + attack_count
)  # First index of Card Feature
decoder_size = (
    decoder_card_offset
    + (1 + decoder_main_feature + SelectContext.RECOVER_SPECIAL_CONDITION) * card_count
)  # Decoder input vocabulary size


# Decoder Layer of MyModel
class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_feedforward: int):
        super(DecoderLayer, self).__init__()

        self.attention = torch.nn.MultiheadAttention(d_model, num_heads)
        self.fc1 = torch.nn.Linear(d_model, d_feedforward)
        self.fc2 = torch.nn.Linear(d_feedforward, d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        y, _ = self.attention(x, encoder_out, encoder_out, need_weights=False)
        res = self.norm1(x + y)
        y = self.fc1(res)
        y = torch.nn.functional.relu(y)
        y = self.fc2(y)
        return self.norm2(res + y)


# My Transformer Model
class MyModel(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_feedforward: int,
        num_layers_encoder: int,
        num_layers_decoder: int,
    ):
        super(MyModel, self).__init__()

        self.d_model = d_model

        self.encoder_bag = torch.nn.EmbeddingBag(encoder_size, d_model, mode="sum")
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model, num_heads, d_feedforward, 0
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers_encoder, enable_nested_tensor=False
        )
        self.encoder_fc = torch.nn.Linear(d_model, 1)
        self.decoder_bag = torch.nn.EmbeddingBag(decoder_size, d_model, mode="sum")
        self.decoder = torch.nn.ModuleList()
        for _ in range(num_layers_decoder):
            self.decoder.append(DecoderLayer(d_model, num_heads, d_feedforward))
        self.decoder_fc = torch.nn.Linear(d_model, 1)

    def forward(
        self,
        index_encoder: torch.Tensor,
        value_encoder: torch.Tensor,
        offset_encoder: torch.Tensor,
        index_decoder: torch.Tensor,
        value_decoder: torch.Tensor,
        offset_decoder: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.encoder_bag(index_encoder, offset_encoder, value_encoder)
        v = v.reshape(-1, num_words_encoder, self.d_model).transpose(0, 1)
        batch_size = v.size(1)
        encoder_out = self.encoder(v)
        v = self.encoder_fc(encoder_out)
        v = torch.tanh(v.mean(0))

        p = self.decoder_bag(index_decoder, offset_decoder, value_decoder)
        p = p.reshape(batch_size, -1, self.d_model).transpose(0, 1)
        for layer in self.decoder:
            p = layer(p, encoder_out)
        p = self.decoder_fc(p)
        p = p.transpose(0, 1).view(batch_size, -1)
        p = torch.tanh(p)
        return (v, p)


# Hyperparameters shared by every script that builds a MyModel, so training
# and inference always agree on the architecture.
MODEL_CONFIG = dict(
    d_model=128, num_heads=2, d_feedforward=256, num_layers_encoder=1, num_layers_decoder=1
)


def build_model() -> MyModel:
    return MyModel(**MODEL_CONFIG)


# torch.nn.EmbeddingBag input
class SparseVector:
    index: list[int]
    value: list[float]
    offset: list[int]
    pos: int

    def __init__(self):
        self.index = []
        self.value = []
        self.offset = []
        self.pos = 0

    def add(self, index: int, value: float | int | bool):
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos + index)
            self.value.append(value)

    def add_pos(self, pos: int):
        self.pos += pos

    def add_single(self, value: float | int | bool):
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos)
            self.value.append(value)
        self.pos += 1

    def word_start(self):
        self.offset.append(len(self.index))


def dummy_decoder_input() -> SparseVector:
    """A single empty decoder "word" (same encoding get_decoder_input uses for
    an empty action). Used when only the value head matters: MyModel.forward
    needs some decoder input to satisfy its signature, but the encoder/value
    branch is computed independently of the decoder branch, so this has no
    effect on the value output."""
    sv = SparseVector()
    sv.word_start()
    sv.add(0, 1)
    return sv


# Add encoder card feature
def add_card(sv: SparseVector, card: Card | Pokemon | None):
    if card != None:
        sv.add(card.id, 1)
    sv.add_pos(card_count)


# Add encoder cards feature
def add_cards(sv: SparseVector, cards: list[Card] | None, value: float):
    if cards != None:
        for card in cards:
            sv.add(card.id, value)
    sv.add_pos(card_count)


# Add encoder Pokémon feature
def add_pokemon(sv: SparseVector, poke: Pokemon | None):
    if poke == None:
        sv.add_single(1)
        sv.add_pos(1 + 3 * card_count)
    else:
        sv.add_single(0)
        sv.add_single(poke.hp / 400)
        add_card(sv, poke)
        add_cards(sv, poke.tools, 1.0)
        add_cards(sv, poke.energyCards, 0.5)


# Add encoder player feature
def add_player(sv: SparseVector, ps: PlayerState):
    sv.add_single(ps.deckCount / 60)
    sv.add_single(len(ps.discard) / 60)
    sv.add_single(ps.handCount / 8)
    sv.add_single(len(ps.bench) / 5)
    sv.add(len(ps.prize), 1)
    sv.add_pos(7)

    sv.add_single(ps.poisoned)
    sv.add_single(ps.burned)
    sv.add_single(ps.asleep)
    sv.add_single(ps.paralyzed)
    sv.add_single(ps.confused)

    add_cards(sv, ps.discard, 0.25)


def get_encoder_input(obs: Observation, your_deck: list[int]) -> SparseVector:
    your_index = obs.current.yourIndex
    state = obs.current

    sv = SparseVector()
    for i in range(2):
        ps = state.players[i ^ your_index]
        for j in range(8):  # For bench
            sv.word_start()
            pos = sv.pos
            if j < len(ps.bench):
                add_pokemon(sv, ps.bench[j])
            else:
                add_pokemon(sv, None)
            if j != 7:  # Not last
                sv.pos = pos  # Return to the previous position

    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        if 0 < len(ps.active):
            add_pokemon(sv, ps.active[0])
        else:
            add_pokemon(sv, None)

    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        add_player(sv, ps)

    sv.word_start()
    add_cards(sv, state.players[your_index].hand, 0.25)

    sv.word_start()
    for id in your_deck:
        sv.add(id, 0.25)
    sv.add_pos(card_count)

    sv.word_start()
    add_cards(sv, state.stadium, 1.0)

    sv.word_start()
    sv.add_single(1)
    sv.add_single(state.turn / 10)
    sv.add_single(state.firstPlayer == your_index)
    return sv


def get_card(
    obs: Observation, area: AreaType, index: int, player_index: int
) -> Pokemon | Card | None:
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return ps.hand[index]
        case AreaType.DISCARD:
            return ps.discard[index]
        case AreaType.ACTIVE:
            return ps.active[index]
        case AreaType.BENCH:
            return ps.bench[index]
        case AreaType.PRIZE:
            return ps.prize[index]
        case AreaType.STADIUM:
            return obs.current.stadium[index]
        case AreaType.LOOKING:
            return obs.current.looking[index]
        case _:
            return None


# Add decoder Main Select feature
def decoder_main(sv: SparseVector, feature_index: int, card: Card | Pokemon | None):
    if card != None:
        sv.add(decoder_card_offset + feature_index * card_count + card.id, 1)


# Add decoder Card ID feature
def decoder_card_id(sv: SparseVector, context: SelectContext, card_id: int):
    sv.add(
        decoder_card_offset + (decoder_main_feature + context) * card_count + card_id, 1
    )


# Add decoder Card feature
def decoder_card(sv: SparseVector, context: SelectContext, card: Card | Pokemon | None):
    if card != None:
        decoder_card_id(sv, context, card.id)


def get_decoder_input(obs: Observation, actions: list[list[int]]) -> SparseVector:
    sv = SparseVector()
    your_index = obs.current.yourIndex
    ps = obs.current.players[your_index]
    context = obs.select.context
    for action in actions:
        sv.word_start()

        if len(action) == 0:
            sv.add(0, 1)
            continue

        for i in action:
            o = obs.select.option[i]
            match o.type:
                case OptionType.END:
                    sv.add(1, 1)
                case OptionType.YES:
                    sv.add(2, 1)
                case OptionType.NO:
                    sv.add(3, 1)
                case OptionType.SPECIAL_CONDITION:
                    sv.add(4 + o.specialConditionType, 1)
                case OptionType.NUMBER:
                    sv.add(9 + min(o.number, 4), 1)
                case OptionType.ATTACK:
                    sv.add(decoder_attack_offset + o.attackId, 1)
                case OptionType.PLAY:
                    decoder_main(sv, 0, ps.hand[o.index])
                case OptionType.ATTACH:
                    decoder_main(sv, 1, get_card(obs, o.area, o.index, your_index))
                    decoder_main(
                        sv, 2, get_card(obs, o.inPlayArea, o.inPlayIndex, your_index)
                    )
                case OptionType.EVOLVE:
                    decoder_main(sv, 3, get_card(obs, o.area, o.index, your_index))
                    decoder_main(
                        sv, 4, get_card(obs, o.inPlayArea, o.inPlayIndex, your_index)
                    )
                case OptionType.ABILITY:
                    decoder_main(sv, 5, get_card(obs, o.area, o.index, your_index))
                case OptionType.DISCARD:
                    decoder_main(sv, 6, get_card(obs, o.area, o.index, your_index))
                case OptionType.RETREAT:
                    decoder_main(sv, 7, ps.active[0])
                case OptionType.CARD:
                    decoder_card(
                        sv, context, get_card(obs, o.area, o.index, o.playerIndex)
                    )
                case OptionType.TOOL_CARD:
                    card = get_card(obs, o.area, o.index, o.playerIndex)
                    decoder_card(sv, context, card.tools[o.toolIndex])
                case OptionType.ENERGY_CARD | OptionType.ENERGY:
                    card = get_card(obs, o.area, o.index, o.playerIndex)
                    decoder_card(sv, context, card.energyCards[o.energyIndex])
                case OptionType.SKILL:
                    decoder_card_id(sv, context, o.cardId)

    return sv


# Evaluate with My Model
def eval_nn(
    sv_enc: SparseVector, sv_dec: SparseVector, model: MyModel
) -> tuple[float, list[float]]:
    device = next(model.parameters()).device
    value, policy = model(
        torch.tensor(sv_enc.index, dtype=torch.int32, device=device),
        torch.tensor(sv_enc.value, dtype=torch.float32, device=device),
        torch.tensor(sv_enc.offset, dtype=torch.int32, device=device),
        torch.tensor(sv_dec.index, dtype=torch.int32, device=device),
        torch.tensor(sv_dec.value, dtype=torch.float32, device=device),
        torch.tensor(sv_dec.offset, dtype=torch.int32, device=device),
    )

    return (value.tolist()[0][0], policy.tolist()[0])


# Helper class to construct batch inputs for the neural network.
class LearnInput:
    index: list[int]
    value: list[float]
    offset: list[int]

    def __init__(self):
        self.index = []
        self.value = []
        self.offset = []

    def add(self, sv: SparseVector):
        count = len(self.index)
        self.index.extend(sv.index)
        self.value.extend(sv.value)
        for o in sv.offset:
            self.offset.append(o + count)
