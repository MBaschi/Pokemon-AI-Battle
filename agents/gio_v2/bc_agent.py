"""
v9 BC reuses v8 per-slot histogram + static features + attention architecture.

v8 extends v7 by adding:
- Board self-attention (2-layer Transformer over 51 card channels with slot type embeddings)
- Option cross-attention (each option queries board slots for relevant information)
- Option static feature lookup (candidate/target/attack card properties via pre-computed tables)
- Type effectiveness features (super-effective / resisted per attack option)
- Deck/prize tracking (remaining card estimates from known deck composition)
- Opponent revealed card statistics (visible zone aggregates for archetype inference)

Model replaces the pure MLP fusion with attention-based slot interaction and
option-aware board querying, while retaining the v6 dual-path card encoding.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


CARD_VOCAB_SIZE = 2048
ATTACK_VOCAB_SIZE = 4096

# ---------------------------------------------------------------------------
# v3 channel layout (unchanged in v6)
# ---------------------------------------------------------------------------
# Own:  hand(1) + discard(1) + active×4(4) + bench×5×4(20) = 26
# Opp:            active×4(4) + bench×5×4(20) = 24
# Global: stadium/looking/context/effect = 1
STATE_CARD_CHANNELS = 51
BENCH_CHANNELS_PER_SLOT = 4  # pokemon, energy, tool, pre-evo

# ---------------------------------------------------------------------------
# Static feature dimensions
# ---------------------------------------------------------------------------
CARD_TYPE_DIM = 7
ENERGY_TYPE_DIM = 12
CARD_TYPE_OFFSET = 0
CARD_ENERGY_TYPE_OFFSET = CARD_TYPE_OFFSET + CARD_TYPE_DIM
CARD_HP_INDEX = CARD_ENERGY_TYPE_OFFSET + ENERGY_TYPE_DIM
CARD_RETREAT_INDEX = CARD_HP_INDEX + 1
CARD_WEAKNESS_OFFSET = CARD_RETREAT_INDEX + 1
CARD_WEAKNESS_DIM = ENERGY_TYPE_DIM + 1  # 12 energy types + None
CARD_RESISTANCE_OFFSET = CARD_WEAKNESS_OFFSET + CARD_WEAKNESS_DIM
CARD_RESISTANCE_DIM = ENERGY_TYPE_DIM + 1
CARD_STAGE_OFFSET = CARD_RESISTANCE_OFFSET + CARD_RESISTANCE_DIM
CARD_SPECIAL_OFFSET = CARD_STAGE_OFFSET + 3
CARD_FEAT_DIM = CARD_SPECIAL_OFFSET + 4

ATTACK_DAMAGE_INDEX = 0
ATTACK_ENERGY_OFFSET = 1
ATTACK_ENERGY_COUNT_INDEX = ATTACK_ENERGY_OFFSET + ENERGY_TYPE_DIM
ATTACK_FEAT_DIM = ATTACK_ENERGY_COUNT_INDEX + 1
STATIC_FEATURE_CONTRACT_VERSION = "bc-static-v3"

# ---------------------------------------------------------------------------
# Numeric dimensions (v8 expanded)
# ---------------------------------------------------------------------------
# Per-player: v3's 24 + bench-per-slot 5hp + 5energy + active retreatCost(1)
#   + maxAttackDamage(1) + attackCount(1) + weakness(1) + resistance(1) = 39
PLAYER_NUMERIC_DIM = 39
GLOBAL_NUMERIC_DIM = 8
SELECT_NUMERIC_DIM = 7
# v8 new: deck/prize tracking aggregates + opponent revealed zone stats
DECK_TRACK_DIM = 15
OPP_REVEALED_DIM = 17
STATE_NUMERIC_DIM = (
    2 * PLAYER_NUMERIC_DIM + GLOBAL_NUMERIC_DIM + SELECT_NUMERIC_DIM
    + DECK_TRACK_DIM + OPP_REVEALED_DIM
)  # 93 + 15 + 17 = 125

# Option: v3's 12 + attack_damage(1) + card_type(1) + super_effective(1) + resisted(1) = 16
OPTION_NUMERIC_DIM = 16
OPTION_CATEGORICAL_DIM = 5


@dataclass(frozen=True)
class ModelConfig:
    card_vocab_size: int = CARD_VOCAB_SIZE
    attack_vocab_size: int = ATTACK_VOCAB_SIZE
    state_card_channels: int = STATE_CARD_CHANNELS
    state_numeric_dim: int = STATE_NUMERIC_DIM
    option_numeric_dim: int = OPTION_NUMERIC_DIM
    card_feat_dim: int = CARD_FEAT_DIM
    attack_feat_dim: int = ATTACK_FEAT_DIM
    # Larghezze della rete. I default riproducono ESATTAMENTE l'architettura v9
    # originale, cosi' i checkpoint vecchi continuano a caricarsi: un checkpoint
    # salvato senza queste chiavi ricostruisce la stessa rete di prima.
    proj_dim: int = 32          # card_channel_proj: 2048 -> proj_dim, per canale
    card_emb_dim: int = 128     # card_encoder
    feat_emb_dim: int = 64      # card_feat_encoder
    numeric_emb_dim: int = 64   # state_numeric
    state_dim: int = 128        # state_encoder (fusione)
    slot_dim: int = 64          # transformer di board
    board_layers: int = 2
    board_heads: int = 4
    xattn_dim: int = 32
    option_dim: int = 64        # option_encoder
    opt_card_emb: int = 32      # embedding carta/bersaglio dell'opzione
    opt_small_emb: int = 16     # embedding tipo/contesto/attacco dell'opzione
    opt_numeric_emb: int = 32
    opt_static_emb: int = 32
    head_dim: int = 128         # score_head / no_action_head
    dropout: float = 0.0
    # Testa di valore: serve solo al fine-tuning RL come baseline. Di default
    # assente, cosi' i checkpoint BC hanno esattamente le stesse chiavi di prima.
    value_head: bool = False



class AreaIndex(NamedTuple):
    """Episode 级卡牌区域预索引，避免逐 option 线性扫描。

    composite key: area * 10 + player_index
    area 7 (stadium) / 12 (looking) 为全局区域，同时存储两份。
    """
    cards_by_area: dict[int, list[dict[str, Any]]]
    card_by_area_index: dict[int, dict[int, dict[str, Any]]]

    def get_card(self, area: int, player_index: int, idx: int) -> dict[str, Any] | None:
        key = area * 10 + player_index
        area_dict = self.card_by_area_index.get(key)
        if area_dict is None:
            return None
        card = area_dict.get(idx)
        return card if isinstance(card, dict) else None


# ===================== static card & attack feature tables =====================


_CG_API: Any | None = None


def _load_cg_api() -> Any:
    """加载提交包内或仓库 sample submission 内的官方 cg.api。"""
    global _CG_API
    if _CG_API is not None:
        return _CG_API

    errors: list[BaseException] = []
    try:
        _CG_API = importlib.import_module("cg.api")
        return _CG_API
    except Exception as error:
        errors.append(error)

    source_path = Path(__file__).resolve()
    for parent in source_path.parents:
        sample_root = (
            parent
            / "pokemon-tcg-ai-battle"
            / "sample_submission"
            / "sample_submission"
        )
        if not (sample_root / "cg" / "api.py").is_file():
            continue
        sample_root_str = str(sample_root)
        if sample_root_str not in sys.path:
            sys.path.insert(0, sample_root_str)
        try:
            _CG_API = importlib.import_module("cg.api")
            return _CG_API
        except Exception as error:
            errors.append(error)

    detail = f"{type(errors[-1]).__name__}: {errors[-1]}" if errors else "unknown"
    raise RuntimeError(
        "无法加载官方 cg.api，不能生成静态卡牌特征；"
        "请保留仓库 sample submission 或提交包内的 cg 目录。"
        f"最后错误：{detail}"
    ) from (errors[-1] if errors else None)


def _build_card_features() -> np.ndarray:
    """Build (CARD_VOCAB_SIZE, CARD_FEAT_DIM) static card feature matrix."""
    feats = np.zeros((CARD_VOCAB_SIZE, CARD_FEAT_DIM), dtype=np.float32)
    cards = _load_cg_api().all_card_data()

    # CardType enum: POKEMON=0, ITEM=1, TOOL=2, SUPPORTER=3, STADIUM=4,
    #                 BASIC_ENERGY=5, SPECIAL_ENERGY=6
    for card in cards:
        idx = int(card.cardId)
        if idx < 0 or idx >= CARD_VOCAB_SIZE:
            continue

        ct = int(card.cardType)
        if 0 <= ct < CARD_TYPE_DIM:
            feats[idx, CARD_TYPE_OFFSET + ct] = 1.0

        et = int(card.energyType)
        if 0 <= et < ENERGY_TYPE_DIM:
            feats[idx, CARD_ENERGY_TYPE_OFFSET + et] = 1.0

        feats[idx, CARD_HP_INDEX] = float(card.hp) / 400.0
        feats[idx, CARD_RETREAT_INDEX] = float(card.retreatCost) / 5.0

        w = card.weakness
        if w is not None and 0 <= int(w) < ENERGY_TYPE_DIM:
            feats[idx, CARD_WEAKNESS_OFFSET + int(w)] = 1.0
        else:
            feats[idx, CARD_WEAKNESS_OFFSET + ENERGY_TYPE_DIM] = 1.0

        r = card.resistance
        if r is not None and 0 <= int(r) < ENERGY_TYPE_DIM:
            feats[idx, CARD_RESISTANCE_OFFSET + int(r)] = 1.0
        else:
            feats[idx, CARD_RESISTANCE_OFFSET + ENERGY_TYPE_DIM] = 1.0

        feats[idx, CARD_STAGE_OFFSET : CARD_STAGE_OFFSET + 3] = [
            float(card.basic),
            float(card.stage1),
            float(card.stage2),
        ]
        feats[idx, CARD_SPECIAL_OFFSET : CARD_SPECIAL_OFFSET + 4] = [
            float(card.ex),
            float(card.megaEx),
            float(card.tera),
            float(card.aceSpec),
        ]

    if not np.any(feats):
        raise RuntimeError("官方 cg.api 未返回任何有效卡牌静态特征")

    return feats


def _build_attack_features() -> np.ndarray:
    """Build (ATTACK_VOCAB_SIZE, ATTACK_FEAT_DIM) static attack feature matrix."""
    feats = np.zeros((ATTACK_VOCAB_SIZE, ATTACK_FEAT_DIM), dtype=np.float32)
    attacks = _load_cg_api().all_attack()

    for atk in attacks:
        idx = int(atk.attackId)
        if idx < 0 or idx >= ATTACK_VOCAB_SIZE:
            continue

        feats[idx, ATTACK_DAMAGE_INDEX] = float(atk.damage) / 300.0
        energy_list = atk.energies if isinstance(atk.energies, list) else []
        for et in energy_list:
            energy_type = int(et)
            if 0 <= energy_type < ENERGY_TYPE_DIM:
                feats[idx, ATTACK_ENERGY_OFFSET + energy_type] += 1.0
        feats[idx, ATTACK_ENERGY_COUNT_INDEX] = len(energy_list) / 5.0

    if not np.any(feats):
        raise RuntimeError("官方 cg.api 未返回任何有效攻击静态特征")

    return feats


# Built at module import time; loaded as nn.Buffer in BCPolicy
_CARD_FEAT_TABLE: np.ndarray | None = None
_ATTACK_FEAT_TABLE: np.ndarray | None = None
_CARD_ATTACK_IDS: tuple[tuple[int, ...], ...] | None = None
_STATIC_FEATURE_FINGERPRINT: str | None = None
# 预计算：卡牌类型索引和攻击伤害，避免 encode_options 中逐 option 的 np.argmax
_CARD_TYPE_IDX: np.ndarray | None = None
_ATTACK_DAMAGE: np.ndarray | None = None


def _get_card_type_idx() -> np.ndarray:
    global _CARD_TYPE_IDX
    result = _CARD_TYPE_IDX
    if result is None:
        card_feats = _get_card_feat_table()
        result = np.argmax(
            card_feats[:, CARD_TYPE_OFFSET:CARD_TYPE_OFFSET + CARD_TYPE_DIM], axis=1
        )
        _CARD_TYPE_IDX = result
    return result

def _get_attack_damage() -> np.ndarray:
    global _ATTACK_DAMAGE
    result = _ATTACK_DAMAGE
    if result is None:
        result = _get_attack_feat_table()[:, 0]
        _ATTACK_DAMAGE = result
    return result



def _get_card_feat_table() -> np.ndarray:
    global _CARD_FEAT_TABLE
    if _CARD_FEAT_TABLE is None:
        _CARD_FEAT_TABLE = _build_card_features()
    return _CARD_FEAT_TABLE


def _get_attack_feat_table() -> np.ndarray:
    global _ATTACK_FEAT_TABLE
    if _ATTACK_FEAT_TABLE is None:
        _ATTACK_FEAT_TABLE = _build_attack_features()
    return _ATTACK_FEAT_TABLE


def _get_card_attack_ids() -> tuple[tuple[int, ...], ...]:
    global _CARD_ATTACK_IDS
    if _CARD_ATTACK_IDS is None:
        attack_ids: list[tuple[int, ...]] = [()] * CARD_VOCAB_SIZE
        for card in _load_cg_api().all_card_data():
            card_id = int(card.cardId)
            if 0 <= card_id < CARD_VOCAB_SIZE:
                attacks = card.attacks if isinstance(card.attacks, list) else []
                attack_ids[card_id] = tuple(int(attack_id) for attack_id in attacks)
        _CARD_ATTACK_IDS = tuple(attack_ids)
    return _CARD_ATTACK_IDS


def _card_attacks(card_id: int) -> tuple[int, ...]:
    """Return cached attack IDs for a card, or an empty tuple."""
    if 0 <= card_id < CARD_VOCAB_SIZE:
        return _get_card_attack_ids()[card_id]
    return ()


def static_feature_fingerprint() -> str:
    """返回会影响转换、训练和推理语义的静态特征指纹。"""
    global _STATIC_FEATURE_FINGERPRINT
    if _STATIC_FEATURE_FINGERPRINT is None:
        digest = hashlib.sha256(STATIC_FEATURE_CONTRACT_VERSION.encode("ascii"))
        for table in (_get_card_feat_table(), _get_attack_feat_table()):
            digest.update(np.asarray(table, dtype="<f4", order="C").tobytes(order="C"))
        for attack_ids in _get_card_attack_ids():
            digest.update(len(attack_ids).to_bytes(2, "little"))
            digest.update(np.asarray(attack_ids, dtype="<i4").tobytes(order="C"))
        _STATIC_FEATURE_FINGERPRINT = digest.hexdigest()
    return _STATIC_FEATURE_FINGERPRINT


def _active_attack_info(player: dict[str, Any]) -> tuple[float, float]:
    """Return (max_attack_damage/300, attack_count/4) for active pokemon, or (0,0)."""
    active = _active_pokemon(player)
    if active is None:
        return 0.0, 0.0
    card_id = _card_id(active)
    attack_ids = _card_attacks(card_id)
    if not attack_ids:
        return 0.0, 0.0

    attack_feats = _get_attack_feat_table()
    max_damage = 0.0
    for aid in attack_ids:
        if 0 <= aid < ATTACK_VOCAB_SIZE:
            max_damage = max(max_damage, float(attack_feats[aid, 0]))
    return max_damage, len(attack_ids) / 4.0
# ===================== helpers (unchanged from v2) =====================


def _int(value: Any, default: int = 0) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _card_id(card: Any, vocab_size: int = CARD_VOCAB_SIZE) -> int:
    if not isinstance(card, dict):
        return 0
    card_id = _int(card.get("id"), 0)
    return max(0, min(card_id, vocab_size - 1))


def _cards(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [card for card in value if isinstance(card, dict)]


def _active_pokemon(player: dict[str, Any]) -> dict[str, Any] | None:
    active = player.get("active")
    if not isinstance(active, list):
        return None
    for pokemon in active:
        if isinstance(pokemon, dict):
            return pokemon
    return None


def _state_and_players(
    obs: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    current = obs.get("current")
    if not isinstance(current, dict):
        return {}, [{}, {}], 0
    players_raw = current.get("players")
    players = players_raw if isinstance(players_raw, list) else []
    players = [player if isinstance(player, dict) else {} for player in players[:2]]
    while len(players) < 2:
        players.append({})
    your_index = max(0, min(_int(current.get("yourIndex"), 0), 1))
    return current, players, your_index

# ===================== numeric encoding (v6 enhanced per-player) =====================


def _player_numeric(player: dict[str, Any]) -> list[float]:
    """39-dim per-player numeric, up from 24 in v3."""
    active = _active_pokemon(player)
    bench = _cards(player.get("bench"))
    bench_hp = sum(_float(pokemon.get("hp")) for pokemon in bench)
    bench_max_hp = sum(_float(pokemon.get("maxHp")) for pokemon in bench)
    bench_energy = sum(len(_cards(pokemon.get("energyCards"))) for pokemon in bench)

    # Active attack & type info (from static tables)
    max_atk_dmg, atk_count = _active_attack_info(player)
    retreat_cost = 0.0
    weakness_norm = 0.0
    resistance_norm = 0.0
    if active is not None:
        card_id = _card_id(active)
        card_feats = _get_card_feat_table()
        if 0 <= card_id < CARD_VOCAB_SIZE:
            retreat_cost = float(card_feats[card_id, CARD_RETREAT_INDEX])
            weakness = card_feats[
                card_id,
                CARD_WEAKNESS_OFFSET : CARD_WEAKNESS_OFFSET + CARD_WEAKNESS_DIM,
            ]
            weakness_norm = float(np.argmax(weakness)) / ENERGY_TYPE_DIM
            resistance = card_feats[
                card_id,
                CARD_RESISTANCE_OFFSET : CARD_RESISTANCE_OFFSET
                + CARD_RESISTANCE_DIM,
            ]
            resistance_norm = float(np.argmax(resistance)) / ENERGY_TYPE_DIM

    # Basic counts (same as v3)
    features = [
        _float(player.get("deckCount")) / 60.0,           # 0
        _float(player.get("handCount")) / 20.0,           # 1
        len(_cards(player.get("discard"))) / 60.0,        # 2
        len(_cards(player.get("prize"))) / 6.0,           # 3
        len(bench) / 5.0,                                  # 4
        float(active is not None),                         # 5
        float(bool(player.get("poisoned"))),               # 6
        float(bool(player.get("burned"))),                 # 7
        float(bool(player.get("asleep"))),                 # 8
        float(bool(player.get("paralyzed"))),              # 9
        float(bool(player.get("confused"))),               # 10
        # Active details
        (_float(active.get("hp")) / 400.0) if active else 0.0,       # 11
        (_float(active.get("maxHp")) / 400.0) if active else 0.0,    # 12
        (len(_cards(active.get("energyCards"))) / 10.0) if active else 0.0,  # 13
        (len(_cards(active.get("tools"))) / 4.0) if active else 0.0,        # 14
        (len(_cards(active.get("preEvolution"))) / 2.0) if active else 0.0, # 15
        # Active combat info (v6 new)
        retreat_cost,          # 16
        max_atk_dmg,           # 17
        atk_count,             # 18
        weakness_norm,         # 19
        resistance_norm,       # 20
    ]

    # Per-bench-slot: exists, hp, energy (5 slots × 3 = 15 dims)
    for i in range(5):
        if i < len(bench):
            pokemon = bench[i]
            features.append(1.0)  # exists
            features.append(_float(pokemon.get("hp")) / 400.0)
            features.append(len(_cards(pokemon.get("energyCards"))) / 5.0)
        else:
            features.extend([0.0, 0.0, 0.0])

    # Bench aggregates (v6 new: kept alongside per-slot for global view)
    features.extend(
        [
            bench_energy / 20.0,
            bench_hp / 2000.0,
            bench_max_hp / 2000.0,
        ]
    )
    return features

# ===================== card histogram encoding (per-slot, unchanged from v3) =====================


def _active_channel_base(own: bool) -> int:
    """Return the first channel index for the active region of own/opp."""
    return 2 if own else 26


def _bench_channel_base(own: bool, slot: int) -> int:
    """Return the first channel index for bench[slot] of own/opp.

    own  bench: channels 6-25  (5 slots × 4 channels)
    opp  bench: channels 30-49 (5 slots × 4 channels)
    """
    region_start = 6 if own else 30
    return region_start + slot * BENCH_CHANNELS_PER_SLOT


def _add_pokemon_slot(
    histogram: np.ndarray,
    pokemon: dict[str, Any] | None,
    base_ch: int,
) -> None:
    """Fill 4 channels: pokemon card, energy, tool, pre-evolution."""
    if not isinstance(pokemon, dict):
        return
    cards = _cards([pokemon])
    for card in cards:
        histogram[base_ch, _card_id(card)] += 1.0
    for card in _cards(pokemon.get("energyCards")):
        histogram[base_ch + 1, _card_id(card)] += 1.0
    for card in _cards(pokemon.get("tools")):
        histogram[base_ch + 2, _card_id(card)] += 1.0
    for card in _cards(pokemon.get("preEvolution")):
        histogram[base_ch + 3, _card_id(card)] += 1.0


def _add_cards_channel(histogram: np.ndarray, ch: int, cards: Any) -> None:
    for card in _cards(cards):
        histogram[ch, _card_id(card)] += 1.0

# ===================== v8 new feature helpers =====================


def _visible_card_counter(players: list[dict[str, Any]], your_index: int) -> Counter[int]:
    """统计己方所有可见区域的卡牌多重集：hand + discard + bench + active。"""
    counter: Counter[int] = Counter()
    own = players[your_index]
    for card in _cards(own.get("hand")):
        counter[_card_id(card)] += 1
    for card in _cards(own.get("discard")):
        counter[_card_id(card)] += 1
    active = _active_pokemon(own)
    if active is not None:
        counter[_card_id(active)] += 1
        for card in _cards(active.get("energyCards")):
            counter[_card_id(card)] += 1
        for card in _cards(active.get("tools")):
            counter[_card_id(card)] += 1
        for card in _cards(active.get("preEvolution")):
            counter[_card_id(card)] += 1
    for pokemon in _cards(own.get("bench")):
        counter[_card_id(pokemon)] += 1
        for card in _cards(pokemon.get("energyCards")):
            counter[_card_id(card)] += 1
        for card in _cards(pokemon.get("tools")):
            counter[_card_id(card)] += 1
        for card in _cards(pokemon.get("preEvolution")):
            counter[_card_id(card)] += 1
    return counter


def _deck_remaining_features(
    deck_counter: Counter[int],
    visible_counter: Counter[int],
    card_feat_table: np.ndarray,
) -> list[float]:
    """从完整牌组计数减去可见区域计数，得到 {deck ∪ prize} 剩余牌聚合统计。

    返回值是上界估计（prize 不可区分），但仍是 deck search 决策的关键信息。
    """
    # 多重集减法：每张卡的 deck+prize 剩余张数（上界）
    remaining: Counter[int] = Counter()
    for card_id, deck_count in deck_counter.items():
        visible = visible_counter.get(card_id, 0)
        remaining[card_id] = max(0, deck_count - visible)

    total_remaining = float(sum(remaining.values()))

    # 按卡牌类型聚合剩余数量（7 dims）
    type_counts = [0.0] * CARD_TYPE_DIM
    for card_id, count in remaining.items():
        if 0 <= card_id < CARD_VOCAB_SIZE and count > 0:
            type_idx = int(np.argmax(card_feat_table[card_id, CARD_TYPE_OFFSET:CARD_TYPE_OFFSET + CARD_TYPE_DIM]))
            type_counts[type_idx] += float(count)
    type_features = [c / 4.0 for c in type_counts]  # 最多4张同名卡，归一化

    # 按阶段聚合剩余数量（3 dims）
    stage_counts = [0.0, 0.0, 0.0]  # basic, stage1, stage2
    for card_id, count in remaining.items():
        if 0 <= card_id < CARD_VOCAB_SIZE and count > 0:
            if card_feat_table[card_id, CARD_STAGE_OFFSET] > 0.5:
                stage_counts[0] += float(count)
            if card_feat_table[card_id, CARD_STAGE_OFFSET + 1] > 0.5:
                stage_counts[1] += float(count)
            if card_feat_table[card_id, CARD_STAGE_OFFSET + 2] > 0.5:
                stage_counts[2] += float(count)
    stage_features = [c / 4.0 for c in stage_counts]

    # 特殊标志（2 dims）
    has_ex_remaining = 0.0
    has_mega_remaining = 0.0
    for card_id, count in remaining.items():
        if 0 <= card_id < CARD_VOCAB_SIZE and count > 0:
            if card_feat_table[card_id, CARD_SPECIAL_OFFSET] > 0.5:
                has_ex_remaining = 1.0
            if card_feat_table[card_id, CARD_SPECIAL_OFFSET + 1] > 0.5:
                has_mega_remaining = 1.0

    # 密度特征（3 dims）
    pokemon_remaining = type_counts[0]  # card type 0 = POKEMON
    energy_remaining = type_counts[5] + type_counts[6]  # BASIC_ENERGY + SPECIAL_ENERGY
    density = [
        total_remaining / 60.0,
        pokemon_remaining / max(total_remaining, 1.0),
        energy_remaining / max(total_remaining, 1.0),
    ]

    return type_features + stage_features + [has_ex_remaining, has_mega_remaining] + density
    # 7 + 3 + 2 + 3 = 15 = DECK_TRACK_DIM


def _opponent_revealed_features(
    opp_player: dict[str, Any],
    card_feat_table: np.ndarray,
) -> list[float]:
    """从对手公开区域（discard + bench + active）提取统计特征。

    纯特征提取，不做推断或概率估计。
    """
    revealed_cards: list[dict[str, Any]] = []
    revealed_cards.extend(_cards(opp_player.get("discard")))
    active = _active_pokemon(opp_player)
    if active is not None:
        revealed_cards.append(active)
        revealed_cards.extend(_cards(active.get("energyCards")))
        revealed_cards.extend(_cards(active.get("tools")))
        revealed_cards.extend(_cards(active.get("preEvolution")))
    for pokemon in _cards(opp_player.get("bench")):
        revealed_cards.append(pokemon)
        revealed_cards.extend(_cards(pokemon.get("energyCards")))
        revealed_cards.extend(_cards(pokemon.get("tools")))
        revealed_cards.extend(_cards(pokemon.get("preEvolution")))

    card_ids = [_card_id(c) for c in revealed_cards]
    card_ids_valid = [cid for cid in card_ids if 0 <= cid < CARD_VOCAB_SIZE]
    n_revealed = len(card_ids_valid)

    # 按卡牌类型统计揭示卡（7 dims）
    type_counts = [0.0] * CARD_TYPE_DIM
    for cid in card_ids_valid:
        type_idx = int(np.argmax(card_feat_table[cid, CARD_TYPE_OFFSET:CARD_TYPE_OFFSET + CARD_TYPE_DIM]))
        type_counts[type_idx] += 1.0
    type_features = [c / 10.0 for c in type_counts]

    # 特殊标志（3 dims）
    has_ex = 0.0
    has_mega = 0.0
    has_tera = 0.0
    avg_hp = 0.0
    avg_stage = 0.0
    pkmn_count = 0
    for cid in card_ids_valid:
        if card_feat_table[cid, CARD_TYPE_OFFSET] > 0.5:  # is pokemon
            pkmn_count += 1
            avg_hp += float(card_feat_table[cid, CARD_HP_INDEX])
            if card_feat_table[cid, CARD_SPECIAL_OFFSET] > 0.5:
                has_ex = 1.0
            if card_feat_table[cid, CARD_SPECIAL_OFFSET + 1] > 0.5:
                has_mega = 1.0
            if card_feat_table[cid, CARD_SPECIAL_OFFSET + 2] > 0.5:
                has_tera = 1.0
            avg_stage += (
                float(card_feat_table[cid, CARD_STAGE_OFFSET + 1])
                + 2.0 * float(card_feat_table[cid, CARD_STAGE_OFFSET + 2])
            )
    if pkmn_count > 0:
        avg_hp /= float(pkmn_count)
        avg_stage /= float(pkmn_count)

    # 揭示宝可梦数、揭示卡总数
    special_features = [
        float(pkmn_count) / 5.0,
        float(n_revealed) / 20.0,
        avg_hp,
        avg_stage / 2.0,
    ]

    # 对手 discard 堆大小（独立于卡牌统计）
    discard_size = float(len(_cards(opp_player.get("discard")))) / 20.0
    bench_size = float(len(_cards(opp_player.get("bench")))) / 5.0
    energy_in_play = 0.0
    if active is not None:
        energy_in_play += float(len(_cards(active.get("energyCards"))))
    for pokemon in _cards(opp_player.get("bench")):
        energy_in_play += float(len(_cards(pokemon.get("energyCards"))))

    return type_features + [has_ex, has_mega, has_tera] + special_features + [discard_size, bench_size, energy_in_play / 10.0]
    # 7 + 3 + 4 + 3 = 17 = OPP_REVEALED_DIM


def encode_state(
    obs: dict[str, Any], select: dict[str, Any],
    own_deck_counter: Counter[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """v8 state encoding: v6 features + deck tracking + opponent revealed stats."""

    current, players, your_index = _state_and_players(obs)
    relative_players = [players[your_index], players[1 - your_index]]

    # ---- per-player numeric ----
    numeric: list[float] = []
    for player in relative_players:
        numeric.extend(_player_numeric(player))

    # ---- global numeric ----
    first_player = _int(current.get("firstPlayer"), -1)
    first_relative = -1.0 if first_player < 0 else float(first_player == your_index)
    numeric.extend(
        [
            _float(current.get("turn")) / 100.0,
            _float(current.get("turnActionCount")) / 100.0,
            first_relative,
            float(bool(current.get("supporterPlayed"))),
            float(bool(current.get("stadiumPlayed"))),
            float(bool(current.get("energyAttached"))),
            float(bool(current.get("retreated"))),
            float(your_index),
        ]
    )

    # ---- select numeric ----
    numeric.extend(
        [
            _int(select.get("type")) / 10.0,
            _int(select.get("context")) / 64.0,
            _int(select.get("minCount")) / 6.0,
            _int(select.get("maxCount")) / 6.0,
            _int(select.get("remainDamageCounter")) / 20.0,
            _int(select.get("remainEnergyCost")) / 10.0,
            len(_cards(select.get("option"))) / 64.0,
        ]
    )

    # ---- v8: deck/prize tracking ----
    card_feat_table = _get_card_feat_table()
    if own_deck_counter is not None:
        visible = _visible_card_counter(players, your_index)
        numeric.extend(_deck_remaining_features(own_deck_counter, visible, card_feat_table))
    else:
        numeric.extend([0.0] * DECK_TRACK_DIM)

    # ---- v8: opponent revealed stats ----
    opp_player = relative_players[1]
    numeric.extend(_opponent_revealed_features(opp_player, card_feat_table))

    # ---- card histogram ----
    histogram = np.zeros((STATE_CARD_CHANNELS, CARD_VOCAB_SIZE), dtype=np.float32)

    for is_own in (True, False):
        player = relative_players[0] if is_own else relative_players[1]

        if is_own:
            _add_cards_channel(histogram, 0, player.get("hand"))
            _add_cards_channel(histogram, 1, player.get("discard"))

        active_pkmn = _active_pokemon(player)
        _add_pokemon_slot(histogram, active_pkmn, _active_channel_base(is_own))

        bench = _cards(player.get("bench"))
        for slot_idx, pokemon in enumerate(bench[:5]):
            _add_pokemon_slot(histogram, pokemon, _bench_channel_base(is_own, slot_idx))

    _add_cards_channel(histogram, 50, current.get("stadium"))
    _add_cards_channel(histogram, 50, current.get("looking"))
    _add_cards_channel(histogram, 50, select.get("deck"))
    _add_cards_channel(histogram, 50, [select.get("contextCard"), select.get("effect")])

    histogram = np.minimum(histogram, 4.0) / 4.0
    return np.asarray(numeric, dtype=np.float32), histogram


# ===================== option encoding (v6 enhanced) =====================


def build_area_index(obs: dict[str, Any], select: dict[str, Any]) -> AreaIndex:
    """构建 episode 级卡牌区域预索引。

    对 hand(2)/discard(3)/active(4)/bench(5)/prize(6)/deck(1) 按 (area, player_index) 复合键存储；
    stadium(7)/looking(12) 作为全局区域，两个 player_index 各存一份。

    返回 AreaIndex，后续所有 _option_cards_fast 调用可在 O(1) 查表。
    """
    current, players, your_index = _state_and_players(obs)
    cards_by_area: dict[int, list[dict[str, Any]]] = {}
    card_by_area_index: dict[int, dict[int, dict[str, Any]]] = {}

    def _make_key(area: int, pi: int) -> int:
        return area * 10 + pi

    for pi, player in enumerate(players):
        per_player_areas = {
            1: select.get("deck") if isinstance(select.get("deck"), list)
               else _cards(player.get("deck")),
            2: _cards(player.get("hand")),
            3: _cards(player.get("discard")),
            4: [player["active"]] if isinstance(player.get("active"), dict) else [],
            5: _cards(player.get("bench")),
            6: _cards(player.get("prize")),
        }
        for area, cards in per_player_areas.items():
            key = _make_key(area, pi)
            cards_by_area[key] = cards
            card_by_area_index[key] = {
                i: card for i, card in enumerate(cards) if isinstance(card, dict)
            }

    # 全局区域：stadium(7) / looking(12)，两个 player_index 各存一份
    global_areas = {
        7: _cards(current.get("stadium")),
        12: _cards(current.get("looking")),
    }
    for area, cards in global_areas.items():
        for pi in range(2):
            key = _make_key(area, pi)
            cards_by_area[key] = cards
            card_by_area_index[key] = {
                i: card for i, card in enumerate(cards) if isinstance(card, dict)
            }

    return AreaIndex(cards_by_area=cards_by_area,
                     card_by_area_index=card_by_area_index)




def _area_cards(
    obs: dict[str, Any], select: dict[str, Any], area: int, player_index: int
) -> list[Any]:
    current, players, _ = _state_and_players(obs)
    if area == 1:
        deck = select.get("deck")
        if isinstance(deck, list):
            return deck
        return _cards(players[player_index].get("deck"))
    if area == 2:
        return _cards(players[player_index].get("hand"))
    if area == 3:
        return _cards(players[player_index].get("discard"))
    if area == 4:
        return players[player_index].get("active") or []
    if area == 5:
        return _cards(players[player_index].get("bench"))
    if area == 6:
        return _cards(players[player_index].get("prize"))
    if area == 7:
        return _cards(current.get("stadium"))
    if area == 12:
        return _cards(current.get("looking"))
    return []


def _area_card(
    obs: dict[str, Any],
    select: dict[str, Any],
    area: int,
    index: int,
    player_index: int,
) -> dict[str, Any] | None:
    cards = _area_cards(obs, select, area, player_index)
    if 0 <= index < len(cards) and isinstance(cards[index], dict):
        return cards[index]
    return None


def _option_cards(
    obs: dict[str, Any], select: dict[str, Any], option: dict[str, Any]
) -> tuple[int, int, int]:
    option_type = _int(option.get("type"))
    _, _, your_index = _state_and_players(obs)
    player_index = _int(option.get("playerIndex"), your_index)
    player_index = max(0, min(player_index, 1))
    card: dict[str, Any] | None = None
    target: dict[str, Any] | None = None

    if option_type == 7:  # PLAY
        card = _area_card(obs, select, 2, _int(option.get("index")), your_index)
    elif option_type in {3, 4, 5, 6, 10, 11}:
        card = _area_card(
            obs,
            select,
            _int(option.get("area")),
            _int(option.get("index")),
            player_index,
        )
        if option_type == 4 and card is not None:
            tools = _cards(card.get("tools"))
            tool_index = _int(option.get("toolIndex"))
            card = tools[tool_index] if 0 <= tool_index < len(tools) else None
        if option_type in {5, 6} and card is not None:
            energies = _cards(card.get("energyCards"))
            energy_index = _int(option.get("energyIndex"))
            card = energies[energy_index] if 0 <= energy_index < len(energies) else None
    elif option_type == 8:  # ATTACH
        card = _area_card(
            obs,
            select,
            _int(option.get("area")),
            _int(option.get("index")),
            player_index,
        )
        target = _area_card(
            obs,
            select,
            _int(option.get("inPlayArea")),
            _int(option.get("inPlayIndex")),
            your_index,
        )
    elif option_type == 9:  # EVOLVE
        card = _area_card(
            obs,
            select,
            _int(option.get("area")),
            _int(option.get("index")),
            player_index,
        )
        target = _area_card(
            obs,
            select,
            _int(option.get("inPlayArea")),
            _int(option.get("inPlayIndex")),
            your_index,
        )
    elif option_type == 12:  # RETREAT
        _, players, _ = _state_and_players(obs)
        active = players[your_index].get("active") or []
        target = active[0] if active and isinstance(active[0], dict) else None

    candidate_id = (
        _card_id(card)
        if card
        else max(0, min(_int(option.get("cardId"), 0), CARD_VOCAB_SIZE - 1))
    )
    target_id = _card_id(target)
    attack_id = max(0, min(_int(option.get("attackId")), ATTACK_VOCAB_SIZE - 1))
    return candidate_id, target_id, attack_id


def encode_options(
    obs: dict[str, Any], select: dict[str, Any],
    area_index: AreaIndex | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    options = _cards(select.get("option"))
    option_cats = np.zeros((len(options), OPTION_CATEGORICAL_DIM), dtype=np.int64)
    option_numeric = np.zeros((len(options), OPTION_NUMERIC_DIM), dtype=np.float32)
    option_count = max(len(options), 1)

    card_feats = _get_card_feat_table()
    card_type_idx = _get_card_type_idx()
    attack_damage_arr = _get_attack_damage()
    current, players, your_index = _state_and_players(obs)
    own_active = _active_pokemon(players[your_index])
    opp_active = _active_pokemon(players[1 - your_index])
    own_type_idx = 0
    super_effective = 0.0
    resisted = 0.0
    if own_active is not None and opp_active is not None:
        own_card_id = _card_id(own_active)
        opp_card_id = _card_id(opp_active)
        if 0 <= own_card_id < CARD_VOCAB_SIZE and 0 <= opp_card_id < CARD_VOCAB_SIZE:
            own_type_idx = int(card_type_idx[own_card_id])
            opp_weakness = card_feats[
                opp_card_id,
                CARD_WEAKNESS_OFFSET:CARD_WEAKNESS_OFFSET + ENERGY_TYPE_DIM,
            ]
            opp_resistance = card_feats[
                opp_card_id,
                CARD_RESISTANCE_OFFSET:CARD_RESISTANCE_OFFSET + ENERGY_TYPE_DIM,
            ]
            if 0 <= own_type_idx < ENERGY_TYPE_DIM:
                super_effective = float(opp_weakness[own_type_idx] > 0.5)
                resisted = float(opp_resistance[own_type_idx] > 0.5)

    for index, option in enumerate(options):
        if area_index is not None:
            candidate_id, target_id, attack_id = _option_cards_fast(area_index, obs, select, option)
        else:
            candidate_id, target_id, attack_id = _option_cards(obs, select, option)
        option_cats[index] = [
            max(0, min(_int(option.get("type")), 16)),
            max(0, min(_int(select.get("context")), 63)),
            candidate_id,
            target_id,
            attack_id,
        ]
        player_index = _int(option.get("playerIndex"), your_index)
        attack_damage = 0.0
        if 0 < attack_id < ATTACK_VOCAB_SIZE:
            attack_damage = float(attack_damage_arr[attack_id])
        option_card_type = 0.0
        if 0 < candidate_id < CARD_VOCAB_SIZE:
            option_card_type = float(card_type_idx[candidate_id]) / (CARD_TYPE_DIM - 1)

        option_numeric[index] = [
            _int(option.get("number")) / 6.0,
            _int(option.get("index")) / 60.0,
            float(player_index == your_index),
            _int(option.get("toolIndex")) / 4.0,
            _int(option.get("energyIndex")) / 10.0,
            _int(option.get("count")) / 10.0,
            _int(option.get("area")) / 12.0,
            _int(option.get("inPlayArea")) / 12.0,
            _int(option.get("inPlayIndex")) / 5.0,
            _int(option.get("specialConditionType")) / 5.0,
            (index + 1) / option_count,
            float(candidate_id > 0 or target_id > 0 or attack_id > 0),
            attack_damage,
            option_card_type,
            super_effective,
            resisted,
        ]
    return option_cats, option_numeric


def _option_cards_fast(
    area_index: AreaIndex,
    obs: dict[str, Any],
    select: dict[str, Any],
    option: dict[str, Any],
) -> tuple[int, int, int]:
    """_option_cards 的快速版本：用 AreaIndex 做 O(1) 查表替代 _area_cards() 线性扫描。"""
    option_type = _int(option.get("type"))
    _, _, your_index = _state_and_players(obs)
    player_index = _int(option.get("playerIndex"), your_index)
    player_index = max(0, min(player_index, 1))
    card: dict[str, Any] | None = None
    target: dict[str, Any] | None = None

    if option_type == 7:  # PLAY
        card = area_index.get_card(2, your_index, _int(option.get("index")))
    elif option_type in {3, 4, 5, 6, 10, 11}:
        card = area_index.get_card(
            _int(option.get("area")),
            player_index,
            _int(option.get("index")),
        )
        if option_type == 4 and card is not None:
            tools = _cards(card.get("tools"))
            tool_index = _int(option.get("toolIndex"))
            card = tools[tool_index] if 0 <= tool_index < len(tools) else None
        if option_type in {5, 6} and card is not None:
            energies = _cards(card.get("energyCards"))
            energy_index = _int(option.get("energyIndex"))
            card = energies[energy_index] if 0 <= energy_index < len(energies) else None
    elif option_type == 8:  # ATTACH
        card = area_index.get_card(
            _int(option.get("area")),
            player_index,
            _int(option.get("index")),
        )
        target = area_index.get_card(
            _int(option.get("inPlayArea")),
            your_index,
            _int(option.get("inPlayIndex")),
        )
    elif option_type == 9:  # EVOLVE
        card = area_index.get_card(
            _int(option.get("area")),
            player_index,
            _int(option.get("index")),
        )
        target = area_index.get_card(
            _int(option.get("inPlayArea")),
            your_index,
            _int(option.get("inPlayIndex")),
        )
    elif option_type == 12:  # RETREAT
        _, players, _ = _state_and_players(obs)
        active = players[your_index].get("active") or []
        target = active[0] if active and isinstance(active[0], dict) else None

    candidate_id = (
        _card_id(card)
        if card
        else max(0, min(_int(option.get("cardId"), 0), CARD_VOCAB_SIZE - 1))
    )
    target_id = _card_id(target)
    attack_id = max(0, min(_int(option.get("attackId")), ATTACK_VOCAB_SIZE - 1))
    return candidate_id, target_id, attack_id



# ===================== v8 encode_decision =====================


def encode_decision(
    obs: dict[str, Any],
    own_deck_counter: Counter[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    select = obs.get("select")
    if not isinstance(select, dict):
        raise ValueError("BC features require an observation with select data")
    area_index = build_area_index(obs, select)
    state_numeric, state_cards = encode_state(obs, select, own_deck_counter)
    option_cats, option_numeric = encode_options(obs, select, area_index=area_index)
    return state_numeric, state_cards, option_cats, option_numeric


def encode_episode_batch(
    records: list,
    own_deck_counter: Counter[int] | None = None,
    max_options: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """批量编码同一 episode 的决策列表，复用 AreaIndex 避免逐 option 线性扫描。

    对共享同一 observation 的连续 decisions（同一 step 的不同 actions）仅构建一次 AreaIndex。
    返回 (state_numeric_batch, state_cards_batch, option_cats_batch, option_numeric_batch)。
    """
    N = len(records)
    state_numeric_batch = np.zeros((N, STATE_NUMERIC_DIM), dtype=np.float32)
    state_cards_batch = np.zeros((N, STATE_CARD_CHANNELS, CARD_VOCAB_SIZE), dtype=np.float32)
    option_cats_batch = np.zeros((N, max_options, OPTION_CATEGORICAL_DIM), dtype=np.int64)
    option_numeric_batch = np.zeros((N, max_options, OPTION_NUMERIC_DIM), dtype=np.float32)

    last_obs_id: int | None = None
    area_index: AreaIndex | None = None

    for i, record in enumerate(records):
        obs = record.observation
        obs_id = id(obs)
        if obs_id != last_obs_id:
            select = obs.get("select") or {}
            area_index = build_area_index(obs, select)
            last_obs_id = obs_id

        sn, sc = encode_state(obs, obs.get("select") or {}, own_deck_counter)
        oc, onum = encode_options(obs, obs.get("select") or {}, area_index=area_index)

        state_numeric_batch[i] = sn
        state_cards_batch[i] = sc
        n_opts = min(oc.shape[0], max_options)
        option_cats_batch[i, :n_opts] = oc[:n_opts]
        option_numeric_batch[i, :n_opts] = onum[:n_opts]

    return state_numeric_batch, state_cards_batch, option_cats_batch, option_numeric_batch


# ===================== v9 model: attention-augmented architecture =====================


class BCPolicy(nn.Module):
    """v9 policy: v8 dual-path card encoding + board self-attention + option cross-attention."""


    # constants
    # ----------------------------------------------------------------
    SLOT_EMBED_DIM: int = 64
    BOARD_NUM_HEADS: int = 4
    BOARD_NUM_LAYERS: int = 2
    XATTN_DIM: int = 32

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        channels = cfg.state_card_channels  # 51

        # Le larghezze vengono dal config e non piu' da costanti di classe: con i
        # default la rete e' identica alla v9, ma cosi' si puo' scalare la
        # capacita' senza toccare forward().
        self.PROJ_DIM = cfg.proj_dim
        self.SLOT_EMBED_DIM = cfg.slot_dim
        self.BOARD_NUM_HEADS = cfg.board_heads
        self.BOARD_NUM_LAYERS = cfg.board_layers
        self.XATTN_DIM = cfg.xattn_dim

        # -- static feature tables (non-trainable buffers) --
        card_feat_np = _get_card_feat_table()
        self.register_buffer("card_feat_table", torch.from_numpy(card_feat_np).float())
        attack_feat_np = _get_attack_feat_table()
        self.register_buffer("attack_feat_table", torch.from_numpy(attack_feat_np).float())

        # -- v3 path: learned per-channel card projection (kept for v6 compat) --
        self.card_channel_proj = nn.Linear(cfg.card_vocab_size, cfg.proj_dim)
        self.card_encoder = nn.Sequential(
            nn.Linear(channels * cfg.proj_dim, cfg.card_emb_dim), nn.GELU(),
            nn.LayerNorm(cfg.card_emb_dim),
        )

        # -- v6 path: static card feature projection (kept for v6 compat) --
        self.card_feat_encoder = nn.Sequential(
            nn.Linear(channels * cfg.card_feat_dim, cfg.feat_emb_dim), nn.GELU(),
            nn.LayerNorm(cfg.feat_emb_dim),
        )

        # -- state numeric --
        self.state_numeric = nn.Sequential(
            nn.Linear(cfg.state_numeric_dim, cfg.numeric_emb_dim), nn.GELU(),
        )

        # -- v8: board self-attention path (parallel on top of the v3 per-channel proj) --
        self.slot_type_embedding = nn.Embedding(channels, cfg.slot_dim)
        self.slot_input_proj = nn.Linear(cfg.proj_dim, cfg.slot_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.slot_dim,
            nhead=cfg.board_heads,
            dim_feedforward=cfg.slot_dim * 2,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.board_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.board_layers
        )

        # state_encoder: fonde le 4 sorgenti (carte + feature statiche + numerico + board)
        self.state_encoder = nn.Sequential(
            nn.Linear(cfg.card_emb_dim + cfg.feat_emb_dim + cfg.numeric_emb_dim
                      + cfg.slot_dim, cfg.state_dim),
            nn.GELU(), nn.LayerNorm(cfg.state_dim),
        )

        # -- option embeddings --
        self.option_type = nn.Embedding(17, cfg.opt_small_emb)
        self.option_context = nn.Embedding(64, cfg.opt_small_emb)
        self.option_card = nn.Embedding(cfg.card_vocab_size, cfg.opt_card_emb)
        self.option_target = nn.Embedding(cfg.card_vocab_size, cfg.opt_card_emb)
        self.option_attack = nn.Embedding(cfg.attack_vocab_size, cfg.opt_small_emb)
        self.option_numeric = nn.Sequential(
            nn.Linear(cfg.option_numeric_dim, cfg.opt_numeric_emb), nn.GELU(),
        )
        # v8: static feature lookup for option cards
        self.option_static_proj = nn.Sequential(
            nn.Linear(cfg.card_feat_dim * 2 + cfg.attack_feat_dim, cfg.opt_static_emb),
            nn.GELU(),
        )

        self.option_encoder = nn.Sequential(
            nn.Linear(3 * cfg.opt_small_emb + 2 * cfg.opt_card_emb
                      + cfg.opt_numeric_emb + cfg.opt_static_emb, cfg.option_dim),
            nn.GELU(), nn.LayerNorm(cfg.option_dim),
        )

        # -- v8: cross-attention: option queries board slots --
        self.xattn_query = nn.Linear(cfg.option_dim, cfg.xattn_dim)
        self.xattn_key = nn.Linear(cfg.slot_dim, cfg.xattn_dim)
        self.xattn_value = nn.Linear(cfg.slot_dim, cfg.option_dim)

        # score_head: concat(attended + option + state)
        self.score_head = nn.Sequential(
            nn.Linear(2 * cfg.option_dim + cfg.state_dim, cfg.head_dim), nn.GELU(),
            nn.Linear(cfg.head_dim, 1),
        )
        self.no_action_head = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.head_dim // 2), nn.GELU(),
            nn.Linear(cfg.head_dim // 2, 1),
        )
        # Stima dell'esito della partita dallo stato: baseline per il vantaggio
        # in RL. In inferenza non viene mai chiamata.
        self.value_head = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.head_dim // 2), nn.GELU(),
            nn.Linear(cfg.head_dim // 2, 1),
        ) if cfg.value_head else None
        self._return_value = False

    # ----------------------------------------------------------------
    # helpers
    # ----------------------------------------------------------------
    @staticmethod
    def _safe_lookup(table: Tensor, indices: Tensor) -> Tensor:
        """Look up rows from table with out-of-bounds indices clamped to 0."""
        vocab = table.size(0)
        clamped = indices.clamp(0, vocab - 1)
        result = table[clamped]
        # zero out rows where indices were out of bounds
        oob_mask = (indices < 0) | (indices >= vocab)
        if oob_mask.any():
            result = result * (~oob_mask).unsqueeze(-1).float()
        return result

    def _board_attention(self, cards_proj: Tensor) -> tuple[Tensor, Tensor]:
        """Run board self-attention over 51 card channels.

        Args:
            cards_proj: (B, 51, 32) — per-channel learned projection output.

        Returns:
            board_slots: (B, 51, 64) — attended slot representations.
            board_pooled: (B, 64) — mean-pooled over slots for global state.
        """
        B, C, _ = cards_proj.shape
        slot_input = self.slot_input_proj(cards_proj)  # (B, 51, 64)
        slot_ids = torch.arange(C, device=cards_proj.device).unsqueeze(0).expand(B, -1)
        slot_input = slot_input + self.slot_type_embedding(slot_ids)  # (B, 51, 64)
        board_slots = self.board_transformer(slot_input)  # (B, 51, 64)
        board_pooled = board_slots.mean(dim=1)  # (B, 64)
        return board_slots, board_pooled

    # ----------------------------------------------------------------
    # forward
    # ----------------------------------------------------------------
    def forward(
        self,
        state_numeric: Tensor,   # (B, STATE_NUMERIC_DIM)
        state_cards: Tensor,     # (B, 51, 2048)
        option_cats: Tensor,     # (B, N, 5)
        option_numeric: Tensor,  # (B, N, OPTION_NUMERIC_DIM)
    ) -> tuple[Tensor, Tensor]:
        B, C, V = state_cards.shape  # C=51, V=2048

        # ---- v3 path: learned per-channel projection (global agg) ----
        cards_flat = state_cards.reshape(B * C, V)
        cards_proj = self.card_channel_proj(cards_flat)  # (B*51, 32)

        cards_agg = cards_proj.reshape(B, C * self.PROJ_DIM)
        state_card_emb = self.card_encoder(cards_agg)  # (B, 128)

        # ---- v6 path: static card feature projection (global agg) ----
        card_feat_table: Tensor = self.card_feat_table  # type: ignore[assignment]
        card_feats_proj = torch.matmul(cards_flat, card_feat_table)  # (B*51, 54)
        card_feats_agg = card_feats_proj.reshape(B, C * self.config.card_feat_dim)
        card_feat_emb = self.card_feat_encoder(card_feats_agg)  # (B, 64)

        # ---- v8: board self-attention ----
        cards_proj_slots = cards_proj.reshape(B, C, self.PROJ_DIM)
        board_slots, board_pooled = self._board_attention(cards_proj_slots)
        # board_slots: (B, 51, 64), board_pooled: (B, 64)

        # ---- state numeric ----
        state_numeric_emb = self.state_numeric(state_numeric)  # (B, 64)

        # ---- state fusion ----
        state_emb = self.state_encoder(
            torch.cat([state_card_emb, card_feat_emb, state_numeric_emb, board_pooled], dim=-1)
        )  # (B, 128)
        # ---- option encoding ----
        # v8: static feature lookup for option cards
        cft: Tensor = self.card_feat_table  # type: ignore[assignment]
        aft: Tensor = self.attack_feat_table  # type: ignore[assignment]
        candidate_feat = self._safe_lookup(cft, option_cats[..., 2].long())
        target_feat = self._safe_lookup(cft, option_cats[..., 3].long())
        attack_feat = self._safe_lookup(aft, option_cats[..., 4].long())
        option_static = torch.cat([candidate_feat, target_feat, attack_feat], dim=-1)
        option_static_emb = self.option_static_proj(option_static)  # (B, N, 32)

        option_emb = torch.cat(
            [
                self.option_type(option_cats[..., 0]),
                self.option_context(option_cats[..., 1]),
                self.option_card(option_cats[..., 2]),
                self.option_target(option_cats[..., 3]),
                self.option_attack(option_cats[..., 4]),
                self.option_numeric(option_numeric),
                option_static_emb,
            ],
            dim=-1,
        )
        option_emb = self.option_encoder(option_emb)  # (B, N, 64)

        # ---- v8: option cross-attention over board slots ----
        N = option_emb.size(1)
        option_q = self.xattn_query(option_emb)                        # (B, N, 32)
        slot_k = self.xattn_key(board_slots)                           # (B, 51, 32)
        slot_v = self.xattn_value(board_slots)                         # (B, 51, 64)
        attn_weights = F.softmax(
            option_q @ slot_k.transpose(-2, -1) / math.sqrt(self.XATTN_DIM), dim=-1
        )  # (B, N, 51)
        attended = attn_weights @ slot_v  # (B, N, 64)

        # ---- score head ----
        state_expanded = state_emb.unsqueeze(1).expand(-1, N, -1)  # (B, N, 128)
        scores = self.score_head(
            torch.cat([attended, option_emb, state_expanded], dim=-1)
        ).squeeze(-1)  # (B, N)

        no_action = self.no_action_head(state_emb).squeeze(-1)  # (B,)

        if self._return_value and self.value_head is not None:
            # `detach`: la testa di valore legge il tronco ma NON lo modifica.
            # Misurato su un batch da 512: il gradiente della regressione di
            # valore (0.5*mse su un bersaglio +-1 difficile da prevedere) ha
            # norma 1.66 contro 0.76 di quello da ricompensa. Condividendo il
            # tronco quel gradiente (a) deforma la rappresentazione BC che vale
            # tutta la forza dell'agente, e (b) col clip globale a 1.0 si mangia
            # due terzi del budget di aggiornamento. Qui il tronco lo muove solo
            # la ricompensa; la testa di valore resta una sonda.
            return scores, no_action, self.value_head(state_emb.detach()).squeeze(-1)
        return scores, no_action


# ===================== I/O =====================


def checkpoint_payload(model: BCPolicy) -> dict[str, Any]:
    expected_table = torch.from_numpy(_get_card_feat_table())
    model_table: Tensor = model.card_feat_table  # type: ignore[assignment]
    if not torch.equal(model_table.detach().cpu(), expected_table):
        raise RuntimeError("模型内的静态卡牌表与当前特征契约不一致")
    return {
        "version": 9,
        "feature_contract": STATIC_FEATURE_CONTRACT_VERSION,
        "feature_hash": static_feature_fingerprint(),
        "config": asdict(model.config),
        "model": model.state_dict(),
    }


def load_policy(path: Path, device: torch.device | str = "cpu") -> BCPolicy:
    payload = torch.load(path, map_location=device, weights_only=False)
    expected_hash = static_feature_fingerprint()
    if (
        not isinstance(payload, dict)
        or payload.get("feature_contract") != STATIC_FEATURE_CONTRACT_VERSION
        or payload.get("feature_hash") != expected_hash
    ):
        raise RuntimeError(
            f"Checkpoint 静态特征契约不匹配：{path}。请使用修复后的数据重新训练。"
        )
    config_data = payload.get("config", {}) if isinstance(payload, dict) else {}
    config = ModelConfig(
        **{key: config_data[key] for key in asdict(ModelConfig()) if key in config_data}
    )
    model = BCPolicy(config).to(device)
    model.load_state_dict(payload["model"])
    expected_table = torch.from_numpy(_get_card_feat_table()).to(device)
    model_table: Tensor = model.card_feat_table  # type: ignore[assignment]
    if not torch.equal(model_table, expected_table):
        raise RuntimeError(f"Checkpoint 内的静态卡牌表不匹配：{path}")
    model.eval()
    return model


_DECK_CSV_CANDIDATES: list[Path] | None = None


def _deck_csv_search_paths() -> list[Path]:
    """返回 deck.csv 候选搜索路径（惰性计算一次）。"""
    global _DECK_CSV_CANDIDATES
    if _DECK_CSV_CANDIDATES is not None:
        return _DECK_CSV_CANDIDATES
    paths: list[Path] = []
    try:
        paths.append(Path(__file__).with_name("deck.csv"))
    except NameError:
        paths.append(Path("/kaggle_simulations/agent/deck.csv"))
    paths.append(Path("deck.csv"))
    paths.append(Path("/kaggle_simulations/agent/deck.csv"))
    _DECK_CSV_CANDIDATES = paths
    return paths


def read_deck_csv(path: Path | None = None) -> list[int]:
    candidates = [path] if path is not None else []
    candidates.extend(_deck_csv_search_paths())
    for candidate in candidates:
        if candidate is None or not candidate.exists():
            continue
        with candidate.open(newline="", encoding="utf-8") as handle:
            values = [int(row[0]) for row in csv.reader(handle) if row]
        if len(values) == 60:
            return values
    raise FileNotFoundError("A valid 60-card deck.csv was not found")




def _fallback_action(select: dict[str, Any]) -> list[int]:
    options = _cards(select.get("option"))
    min_count = max(0, _int(select.get("minCount")))
    if min_count == 0:
        return []
    return list(range(min_count))[: len(options)]
# Module-level deck counter cache (loaded once per game from deck.csv)
_DECK_COUNTER: Counter[int] | None = None


def predict_action_and_ranking(
    obs: dict[str, Any], model: BCPolicy | None
) -> tuple[list[int], list[int]]:
    """Selezione della rete e, insieme, la classifica completa delle opzioni.

    predict_action da solo non basta a guidare una ricerca: per una scelta MAIN
    minCount == maxCount == 1, quindi restituisce un unico indice e un beam
    seminato con quello avrebbe un solo candidato radice. Qui la forward e' la
    stessa (un solo passaggio di rete) ma si espone anche l'ordine di
    preferenza, che serve come generatore di candidati.
    """
    global _DECK_COUNTER
    select = obs.get("select")
    if not isinstance(select, dict):
        _DECK_COUNTER = None  # new game starts → reset
        deck = read_deck_csv()
        return deck, []
    options = _cards(select.get("option"))
    if not options or model is None:
        fallback = _fallback_action(select)
        return fallback, list(fallback)

    # Lazy-load deck counter once per game
    if _DECK_COUNTER is None:
        deck_list = read_deck_csv()
        _DECK_COUNTER = Counter(deck_list)

    state_numeric, state_cards, option_cats, option_numeric = encode_decision(
        obs, _DECK_COUNTER
    )
    device = next(model.parameters()).device
    with torch.inference_mode():
        scores, no_action = model(
            torch.from_numpy(state_numeric).unsqueeze(0).to(device),
            torch.from_numpy(state_cards).unsqueeze(0).to(device),
            torch.from_numpy(option_cats).unsqueeze(0).to(device),
            torch.from_numpy(option_numeric).unsqueeze(0).to(device),
        )
    score_values = scores[0].detach().cpu().tolist()
    no_action_value = float(no_action[0].detach().cpu())
    order = sorted(range(len(options)), key=lambda idx: (-score_values[idx], idx))
    min_count = max(0, min(_int(select.get("minCount")), len(options)))
    max_count = max(min_count, min(_int(select.get("maxCount")), len(options)))

    if min_count == max_count:
        return order[:min_count], order
    selected = [idx for idx in order if score_values[idx] >= no_action_value][
        :max_count
    ]
    if len(selected) < min_count:
        selected = order[:min_count]
    return selected, order


def predict_action(obs: dict[str, Any], model: BCPolicy | None) -> list[int]:
    return predict_action_and_ranking(obs, model)[0]
