import json
import os
import sys
from collections import defaultdict

from cg.api import AreaType, CardType, EnergyType, Observation, SelectContext, OptionType, Card, Pokemon, all_card_data, to_observation_class

"""
Mega Lucario ex deck — improved heuristic agent.

Based on the public rule-based baseline, with:
- Premium Power Pro aware KO planning (play boosts only when they convert a non-KO into a KO)
- Cosmic Beam weakness fix (damage is not affected by weakness/resistance)
- Carmine guard (don't discard valuable hands)
- Opponent-prize aware Mega exposure (a KO'd Mega gives up 3 prizes)
- Explicit setup-bench preferences
All decision weights live in PARAMS and can be overridden via the MYBOT_PARAMS
env var or a params.json next to this file (used by the local tuner).
"""

PARAMS = {
    # feature toggles
    "ppp_plan": 1,
    "cosmic_fix": 1,
    "carmine_guard": 1,
    "mega_guard": 1,
    "bench_setup": 1,
    # attack plan weights
    "plan_active_attacker_bonus": 220,
    "plan_active_target_bonus": 300,
    "aura_jab_discard_bonus": 60,
    "mega_expose_penalty": 500,
    "ppp_cost": 30,
    # play scores
    "ppp_play_score": 5000,
    "boss_score": 3200,
    "carmine_score": 3000,
    "lillie_score": 3100,
    "carmine_value_malus": 400,
    "switch_score": 6000,
    "retreat_score": 2000,
    # promotion (switch / to-active) preferences
    "pr_mega": 20,
    "pr_mega_low": 8,
    "pr_hariyama": 15,
    "pr_makuhita": 10,
    "pr_solrock": 5,
    "pr_riolu": 4,
    "pr_plan_bonus": 100,
    # to-hand (search/draw pick) preferences
    "th_makuhita": 10,
    "th_hariyama": 20,
    "th_lunatone": 60,
    "th_solrock": 50,
    "th_riolu": 40,
    "th_mega": 40,
    "th_energy": 30,
    # energy attachment
    "en_need_bonus": 100,
    "en_done_malus": 50,
    # setup
    "setup_solrock_first": 2,
    "setup_solrock_second": 4,
    "setup_riolu": 3,
    "setup_makuhita": 1,
    "bench_riolu": 40,
    "bench_solrock": 30,
    "bench_lunatone": 25,
    "bench_makuhita": 10,
    "go_first": 1,
    # play Gravity Mountain proactively when the opponent fields a Stage 2
    # (Dragapult ex 320 -> 290 HP: Mega Brave 270 + one PPP knocks it out)
    "gm_proactive": 1,
    "gm_score": 9500,
    # early-game bonus for knocking out evolving non-ex Pokémon (deny their setup)
    "target_evo_bonus": 250,
    "denial_turns": 8,
}

def _find_file(name):
    """Locate a bundled file. Kaggle runs main.py via exec() without __file__
    and without chdir, so fall back to the bundled cg package location and the
    fixed /kaggle_simulations/agent path."""
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


_params_path = os.environ.get("MYBOT_PARAMS") or _find_file("params.json")
if _params_path and os.path.exists(_params_path):
    with open(_params_path) as _f:
        PARAMS.update(json.load(_f))
P = PARAMS

# Load deck.csv in the dataset
with open(_find_file("deck.csv"), "r") as file:
    csv = file.read().split("\n")
my_deck = []
for i in range(60):
    my_deck.append(int(csv[i]))

all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}

# Decklist
Makuhita = 673
Hariyama = 674
Lunatone = 675
Solrock = 676
Riolu = 677
Mega_Lucario_ex = 678
Dusk_Ball = 1102
Switch = 1123
Premium_Power_Pro = 1141
Fighting_Gong = 1142
Poke_Pad = 1152
Hero_Cape = 1159
Boss_Orders = 1182
Carmine = 1192
Lillie_Determination = 1227
Gravity_Mountain = 1252
Basic_Fighting_Energy = 6


class AttackPlan:
    attacker = -1
    target = -1
    attack_index = -1
    remain_hp = -1
    energy = False
    ppp = 0  # total PPP boosts that should be active this turn for the plan


plan = AttackPlan()
pre_turn = 0
ability_used = False
ppp_played_turn = 0  # PPP copies already played this turn


def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
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


def prize_count(pokemon: Pokemon) -> int:
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    for card in pokemon.energyCards:
        if card.id == 12:  # Legacy Energy
            count -= 1
    for card in pokemon.tools:
        if card.id == 1172 and "Lillie" in data.name:  # Lillie's Pearl
            count -= 1
    return max(0, count)


def pokemon_score(pokemon: Pokemon) -> int:
    data = card_table[pokemon.id]
    score = prize_count(pokemon) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130

    id = pokemon.id
    # Noctowl, Fan Rotom, Archaludon ex, Meowth ex
    if id == 173 or id == 174 or id == 190 or id == 1071:
        score -= 200
    if id == 112 and len(pokemon.energies) >= 1:  # Munkidori
        score += 300
    score += pokemon.hp
    return score


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select == None:
        return my_deck

    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]
    my_prize = len(my_state.prize)
    op_prize = len(op_state.prize)

    global plan
    global pre_turn
    global ability_used
    global ppp_played_turn
    if pre_turn != state.turn:
        pre_turn = state.turn
        plan = AttackPlan()
        ability_used = False
        ppp_played_turn = 0

    field_counts = defaultdict(int)
    hand_counts = defaultdict(int)
    discard_counts = defaultdict(int)

    attacker1 = False
    attacker2 = False
    for card in my_state.active + my_state.bench:
        if card == None:
            continue
        field_counts[card.id] += 1
        if card.id == Makuhita or card.id == Hariyama:
            if len(card.energies) >= 3:
                attacker2 = True
        elif card.id == Riolu or card.id == Mega_Lucario_ex:
            if len(card.energies) >= 2:
                attacker1 = True

    for card in my_state.hand:
        hand_counts[card.id] += 1

    for card in my_state.discard:
        discard_counts[card.id] += 1

    stadium_id = 0
    for card in state.stadium:
        stadium_id = card.id

    op_has_stage2 = False
    for card in op_state.active + op_state.bench:
        if card != None and card_table[card.id].stage2:
            op_has_stage2 = True

    can_attack = False
    if context == SelectContext.MAIN:
        can_switch = False
        can_op_switch = False
        can_use_mega_brave = False
        for o in select.option:
            if o.type == OptionType.PLAY:
                card = get_card(obs, AreaType.HAND, o.index, my_index)
                if card.id == Switch:
                    can_switch = True
                elif card.id == Boss_Orders:
                    can_op_switch = True
            elif o.type == OptionType.EVOLVE:
                card = get_card(obs, AreaType.HAND, o.index, my_index)
                if card.id == Hariyama:
                    can_op_switch = True
            elif o.type == OptionType.RETREAT:
                can_switch = True
            elif o.type == OptionType.ATTACK:
                can_attack = True
                if o.attackId == 983:  # Mega Brave
                    can_use_mega_brave = True

        my_cards = [my_state.active[0]]
        for pokemon in my_state.bench:
            my_cards.append(pokemon)
        op_cards = [op_state.active[0]]
        for pokemon in op_state.bench:
            op_cards.append(pokemon)

        ppp_in_hand = hand_counts[Premium_Power_Pro] if P["ppp_plan"] else 0

        if state.turn >= 2:
            best_score = -1
            for i, my_pokemon in enumerate(my_cards):
                if i != 0 and not can_switch:
                    break
                for a in range(2):
                    energy_required = 0
                    base_damage = 0
                    base_score = 0
                    no_wr = False  # attack unaffected by weakness/resistance
                    if my_pokemon.id == Mega_Lucario_ex:
                        if a == 0:
                            energy_required = 1
                            base_damage = 130
                            base_score += P["aura_jab_discard_bonus"] * min(3, discard_counts[Basic_Fighting_Energy])
                        else:
                            energy_required = 2
                            base_damage = 270
                        if P["mega_guard"]:
                            if op_prize <= 3:
                                base_score -= P["mega_expose_penalty"]
                        else:
                            if my_prize == 2 or my_prize == 3:
                                base_score -= 500
                    elif a == 1:
                        break
                    elif my_pokemon.id == Hariyama:
                        energy_required = 3
                        base_damage = 210
                    elif my_pokemon.id == Makuhita:
                        for o in select.option:
                            if o.type == OptionType.EVOLVE:
                                index = o.inPlayIndex
                                if o.inPlayArea == AreaType.BENCH:
                                    index += 1
                                if index == i:
                                    break
                        else:
                            break
                        base_score -= 100
                        energy_required = 3
                        base_damage = 210
                    elif my_pokemon.id == Solrock:
                        if field_counts[Lunatone] >= 1:
                            energy_required = 1
                            base_damage = 70
                            no_wr = bool(P["cosmic_fix"])

                    if base_damage <= 0:
                        continue

                    more_energy = False
                    energy_count = len(my_pokemon.energies)
                    if a == 1 and i == 0 and energy_count >= 2 and not can_use_mega_brave:
                        break
                    if energy_count < energy_required:
                        if hand_counts[Basic_Fighting_Energy] >= 1 and not state.energyAttached:
                            energy_count += 1
                            if energy_count < energy_required:
                                continue
                            else:
                                more_energy = True
                        else:
                            continue

                    for j, op_pokemon in enumerate(op_cards):
                        if j != 0 and not can_op_switch:
                            break
                        data = card_table[op_pokemon.id]
                        weak = (data.weakness == EnergyType.FIGHTING) and not no_wr
                        res = (data.resistance == EnergyType.FIGHTING) and not no_wr

                        def calc_damage(boosts: int) -> int:
                            d = base_damage + 30 * boosts
                            if weak:
                                d *= 2
                            elif res:
                                d -= 30
                            return d

                        # smallest number of PPP boosts that yields a KO (0 if impossible)
                        use_k = 0
                        for k in range(0, ppp_in_hand + 1):
                            if op_pokemon.hp <= calc_damage(ppp_played_turn + k):
                                use_k = k
                                break
                        damage = calc_damage(ppp_played_turn + use_k)

                        prize = 0
                        score = pokemon_score(op_pokemon)
                        if op_pokemon.hp <= damage:
                            prize = prize_count(op_pokemon)
                        else:
                            score *= damage / op_pokemon.hp
                        score += base_score - P["ppp_cost"] * use_k

                        if (prize > 0 and not data.ex
                                and (data.basic or data.stage1)
                                and state.turn <= P["denial_turns"]):
                            score += P["target_evo_bonus"]

                        if op_prize <= prize:
                            score = 50000

                        if i == 0:
                            score += P["plan_active_attacker_bonus"]
                        if j == 0:
                            score += P["plan_active_target_bonus"]
                        score += energy_count
                        if best_score < score:
                            best_score = score
                            plan.attacker = i
                            plan.target = j
                            plan.attack_index = a
                            plan.remain_hp = op_pokemon.hp - damage
                            plan.energy = more_energy
                            plan.ppp = ppp_played_turn + use_k

    # Attach energy score
    def energy_score(pokemon: Pokemon, active: bool) -> int:
        energy_count = len(pokemon.energies)
        score = 8000
        if active:
            score += 10
        if pokemon.id == Makuhita or pokemon.id == Hariyama:
            if pokemon.id == Hariyama:
                score += 1
            if energy_count < 3:
                score += P["en_need_bonus"]
            if attacker2:
                score -= P["en_done_malus"]
        elif pokemon.id == Lunatone:
            score -= 100
        elif pokemon.id == Solrock:
            if energy_count < 1:
                score += 20
            else:
                score -= 100
        elif pokemon.id == Riolu or pokemon.id == Mega_Lucario_ex:
            if pokemon.id == Mega_Lucario_ex:
                score += 1
            if energy_count < 2:
                score += P["en_need_bonus"]
            if attacker1:
                score -= P["en_done_malus"]
        return score

    scores = []
    for o in select.option:
        score = 0
        if o.type == OptionType.NUMBER:
            score = o.number
        elif o.type == OptionType.YES:
            if context == SelectContext.IS_FIRST:
                score = 1 if P["go_first"] else -1
            else:
                score = 1
        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card != None:
                energy_count = 0
                if isinstance(card, Pokemon):
                    energy_count = len(card.energies)
                if context == SelectContext.SWITCH or context == SelectContext.TO_ACTIVE:
                    if o.playerIndex == my_index:
                        score += energy_count * 2
                        if o.index == plan.attacker - 1:
                            score += P["pr_plan_bonus"]
                        if card.id == Mega_Lucario_ex:
                            low = (op_prize <= 3) if P["mega_guard"] else (my_prize == 2 or my_prize == 3)
                            if low:
                                score += P["pr_mega_low"]
                            else:
                                score += P["pr_mega"]
                        elif card.id == Hariyama and energy_count >= 2:
                            score += P["pr_hariyama"]
                        elif card.id == Makuhita and energy_count >= 2:
                            score += P["pr_makuhita"]
                        elif card.id == Solrock:
                            score += P["pr_solrock"]
                        elif card.id == Riolu:
                            score += P["pr_riolu"]
                    else:
                        if o.index == plan.target - 1:
                            score += P["pr_plan_bonus"]
                elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                    if card.id == Solrock:
                        if state.firstPlayer == my_index:
                            score = P["setup_solrock_first"]
                        else:
                            score = P["setup_solrock_second"]
                    elif card.id == Riolu:
                        score = P["setup_riolu"]
                    elif card.id == Makuhita:
                        score = P["setup_makuhita"]
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    if P["bench_setup"]:
                        if card.id == Riolu:
                            score = P["bench_riolu"]
                        elif card.id == Solrock:
                            score = P["bench_solrock"]
                        elif card.id == Lunatone:
                            score = P["bench_lunatone"]
                        elif card.id == Makuhita:
                            score = P["bench_makuhita"]
                elif context == SelectContext.TO_HAND:
                    score = 200 - hand_counts[card.id] * 100
                    if card.id == Makuhita:
                        if field_counts[card.id] >= 1:
                            score -= P["th_makuhita"]
                        else:
                            score += P["th_makuhita"]
                    elif card.id == Hariyama:
                        if field_counts[Makuhita] >= 1:
                            score += P["th_hariyama"]
                        else:
                            score -= P["th_hariyama"]
                    elif card.id == Lunatone:
                        if field_counts[card.id] >= 1:
                            score -= 250
                        else:
                            score += P["th_lunatone"]
                    elif card.id == Solrock:
                        if field_counts[card.id] >= 1:
                            score -= 250
                        else:
                            score += P["th_solrock"]
                    elif card.id == Riolu:
                        if field_counts[card.id] + field_counts[Mega_Lucario_ex] >= 2:
                            score -= 150
                        elif field_counts[card.id] + field_counts[Mega_Lucario_ex] >= 1:
                            score -= 3
                        else:
                            score += P["th_riolu"]
                    elif card.id == Mega_Lucario_ex:
                        if field_counts[Riolu] >= 1:
                            score += P["th_mega"]
                        else:
                            score -= 15
                    elif card.id == Basic_Fighting_Energy:
                        if not ability_used or not state.energyAttached:
                            score += P["th_energy"]
                        else:
                            score -= 1
                elif context == SelectContext.ATTACH_FROM:
                    score = energy_score(card, o.area == AreaType.ACTIVE)
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            data = card_table[card.id]
            if data.cardType == CardType.POKEMON:
                score = 20000
                if card.id == Lunatone or card.id == Solrock:
                    if field_counts[card.id] >= 1:
                        score = -1
                elif card.id == Riolu:
                    if field_counts[card.id] + field_counts[Mega_Lucario_ex] >= 2:
                        score = -1
            else:
                score = 10000
                if card.id == Switch:
                    if plan.attacker <= 0:
                        score = -1
                    else:
                        score = P["switch_score"]
                elif card.id == Premium_Power_Pro:
                    if P["ppp_plan"]:
                        if can_attack and plan.ppp > ppp_played_turn:
                            score = P["ppp_play_score"]
                        elif not can_attack and not state.supporterPlayed and hand_counts[Carmine] > 0 and hand_counts[Lillie_Determination] == 0:
                            score = 3050  # dump before Carmine discards the hand
                        else:
                            score = -1
                    else:
                        if state.supporterPlayed and plan.remain_hp <= 0:
                            score = -1
                        elif not can_attack:
                            if not state.supporterPlayed and hand_counts[Carmine] > 0 and hand_counts[Lillie_Determination] == 0:
                                score = 3050
                            else:
                                score = -1
                        else:
                            score = P["ppp_play_score"]
                elif card.id == Boss_Orders:
                    if plan.target >= 1:
                        score = P["boss_score"]
                    else:
                        score = -1
                elif card.id == Carmine:
                    score = P["carmine_score"]
                    if P["carmine_guard"]:
                        valuable = (hand_counts[Premium_Power_Pro] + hand_counts[Boss_Orders]
                                    + hand_counts[Mega_Lucario_ex] + hand_counts[Hero_Cape]
                                    + hand_counts[Switch])
                        score -= P["carmine_value_malus"] * max(0, valuable - 1)
                elif card.id == Lillie_Determination:
                    score = P["lillie_score"]
                elif card.id == Gravity_Mountain:
                    if stadium_id == Gravity_Mountain:
                        score = -1
                    elif stadium_id == 0:
                        if P["gm_proactive"] and op_has_stage2:
                            score = P["gm_score"]
                        else:
                            score = -1
        elif o.type == OptionType.ATTACH:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if card.id == Hero_Cape:
                score = 7000
                if pokemon.id == Riolu:
                    score += 100
                elif pokemon.id == Mega_Lucario_ex:
                    score += 200
            else:
                score = energy_score(pokemon, o.inPlayArea == AreaType.ACTIVE)
                if o.inPlayArea == AreaType.ACTIVE:
                    if plan.attacker == 0 and plan.energy:
                        score += 200
                else:
                    if plan.attacker == 1 + o.inPlayIndex and plan.energy:
                        score += 200
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = 9000 + len(pokemon.energies)
            if pokemon.id == Makuhita and plan.target == 0:
                score = -1
        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card.id == 1267:  # Lumiose City
                score = 1
            else:
                score = 30000
        elif o.type == OptionType.RETREAT:
            if plan.attacker >= 1:
                score = P["retreat_score"]
            else:
                score = -1
        elif o.type == OptionType.ATTACK:
            score = 1000
            if plan.attack_index == 1:
                if o.attackId == 983:  # Mega Brave
                    score += 100
            else:
                if o.attackId != 983:
                    score += 100

        scores.append(score)

    desc_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    if context == SelectContext.MAIN:
        o = select.option[desc_indices[0]]
        if o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card.id == Lunatone:
                ability_used = True
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            if card.id == Premium_Power_Pro:
                ppp_played_turn += 1
    return desc_indices[:select.maxCount]
