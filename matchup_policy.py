"""Matchup-first action scoring for singles battles.

Type effectiveness, expected damage, KO chance, speed, and switch-in
quality drive the 13 motor channels (moves / tera+moves / switches).
The connectome readout is only a small additive prior on top of this.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from state_engine import (
    HAZARD_REMOVAL_MOVES,
    HAZARD_SETTING_MOVES,
    N_MOTOR_CHANNELS,
    N_MOVES,
    N_SWITCHES,
    PIVOT_MOVES,
    RECOVERY_MOVES,
    SETUP_MOVES,
    VectorizedState,
    _enum_in,
    _enum_value,
    ability_id,
    calculate_damage_range,
    legal_action_mask,
    move_id,
    pokemon_types,
    spikes_fraction,
    status_name,
    stealth_rock_fraction,
    type_effectiveness,
)

PARALYSIS_MOVES = frozenset({"thunderwave", "nuzzle", "glare", "stunspore"})
BURN_MOVES = frozenset({"willowisp"})
SLEEP_MOVES = frozenset({"spore", "sleeppowder", "yawn", "hypnosis", "darkvoid", "sing"})
TOXIC_MOVES = frozenset({"toxic", "poisonpowder"})
PROTECT_MOVES = frozenset(
    {
        "protect",
        "detect",
        "spikyshield",
        "banefulbunker",
        "burningbulwark",
        "silktrap",
        "kingsshield",
        "obstruct",
    }
)
PRIORITY_ATTACKS = frozenset(
    {
        "suckerpunch",
        "extremespeed",
        "aquajet",
        "iceshard",
        "bulletpunch",
        "machpunch",
        "shadowsneak",
        "accelerock",
        "grassyglide",
        "jetpunch",
        "watershuriken",
        "firstimpression",
    }
)


def _hp(mon: Any) -> float:
    if mon is None:
        return 0.0
    try:
        return float(getattr(mon, "current_hp_fraction", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _accuracy(move: Any) -> float:
    acc = getattr(move, "accuracy", True)
    if acc is True or acc is None:
        return 1.0
    try:
        val = float(acc)
    except (TypeError, ValueError):
        return 1.0
    if val > 1.5:
        val /= 100.0
    return float(max(0.0, min(1.0, val)))


def _category(move: Any) -> str:
    return str(getattr(move, "category", "") or "").lower()


def offensive_stab_vs(attacker: Any, defender: Any) -> float:
    types = pokemon_types(attacker)
    defs = pokemon_types(defender)
    if not types or not defs:
        return 1.0
    return float(max(type_effectiveness(t, defs) for t in types))


def type_matchup(us: Any, them: Any) -> float:
    """Positive when we are favored (our STAB hits harder than theirs)."""
    if us is None or them is None:
        return 0.0
    return offensive_stab_vs(us, them) - offensive_stab_vs(them, us)


def switch_hazard_cost(mon: Any, battle: Any) -> float:
    if mon is None or battle is None:
        return 0.0
    side = getattr(battle, "side_conditions", None) or {}
    cost = 0.0
    if _enum_in(side, "STEALTH_ROCK"):
        cost += stealth_rock_fraction(mon)
    layers = _enum_value(side, "SPIKES")
    if layers:
        cost += spikes_fraction(layers, mon, battle)
    return float(cost)


def _alive(team: Any) -> int:
    n = 0
    for mon in (team or {}).values():
        if mon is not None and not getattr(mon, "fainted", False):
            n += 1
    return n


def _move_type_name(move: Any) -> str:
    raw = getattr(move, "type", None)
    return str(getattr(raw, "name", raw) or "").upper()


def score_move(
    battle: Any,
    move: Any,
    dmg: Any,
    *,
    us: Any,
    them: Any,
    outspeed: float,
    tera: bool = False,
) -> float:
    if move is None:
        return -1e6
    mid = move_id(move)
    cat = _category(move)
    acc = _accuracy(move)
    power = float(getattr(move, "base_power", 0) or 0)
    our_hp = _hp(us)
    opp_hp = _hp(them)
    matchup = type_matchup(us, them)
    opp_status = status_name(them)
    our_status = status_name(us)

    expected = float(getattr(dmg, "expected_frac", 0.0) or 0.0) if dmg is not None else 0.0
    ko = float(getattr(dmg, "ko_chance", 0.0) or 0.0) if dmg is not None else 0.0
    eff = float(getattr(dmg, "effectiveness", 1.0) or 1.0) if dmg is not None else 1.0
    if (dmg is None or expected <= 0.0) and power > 0 and them is not None:
        defs = pokemon_types(them)
        mtype = _move_type_name(move)
        if mtype:
            eff = type_effectiveness(mtype, defs)
            expected = max(expected, 0.08 * (power / 80.0) * eff)
            if expected >= opp_hp:
                ko = max(ko, 0.35)

    if cat not in {"status", "3"} and eff <= 0.0:
        return -8.0

    score = 0.0
    if cat not in {"status", "3"}:
        score = expected * 14.0 + ko * 11.0 + (eff - 1.0) * 1.8
        if ko >= 0.875:
            score += 9.0 if outspeed > 0 else 4.0
        elif ko >= 0.5 and outspeed > 0:
            score += 5.5
        if mid in PRIORITY_ATTACKS and outspeed <= 0 and ko > 0:
            score += 3.5 + 6.0 * ko
        if our_hp < 0.35 and ko < 0.5 and outspeed <= 0:
            score -= 1.5
    else:
        score = 0.4
        if mid in HAZARD_SETTING_MOVES:
            if mid == "stealthrock":
                already = _enum_in(getattr(battle, "opponent_side_conditions", None) or {}, "STEALTH_ROCK")
            elif mid == "spikes":
                already = _enum_value(getattr(battle, "opponent_side_conditions", None) or {}, "SPIKES") >= 3
            elif mid == "toxicspikes":
                already = _enum_value(getattr(battle, "opponent_side_conditions", None) or {}, "TOXIC_SPIKES") >= 2
            elif mid == "stickyweb":
                already = _enum_in(getattr(battle, "opponent_side_conditions", None) or {}, "STICKY_WEB")
            if not already and _alive(getattr(battle, "opponent_team", None)) >= 3:
                score = 3.6
            else:
                score = -1.0
        elif mid in HAZARD_REMOVAL_MOVES:
            ours = getattr(battle, "side_conditions", None) or {}
            if _enum_in(ours, "STEALTH_ROCK", "SPIKES", "TOXIC_SPIKES", "STICKY_WEB") and _alive(getattr(battle, "team", None)) >= 2:
                score = 4.2
            else:
                score = -0.8
        elif mid in RECOVERY_MOVES:
            if our_hp <= 0.55:
                score = 3.8 + (0.55 - our_hp) * 6.0
                if matchup >= 0 and outspeed > 0:
                    score += 1.2
            else:
                score = -1.5
        elif mid in SETUP_MOVES:
            if our_hp >= 0.7 and matchup >= 0.0 and ko < 0.6:
                score = 3.2 + matchup
            else:
                score = -2.0
        elif mid in PARALYSIS_MOVES:
            score = 2.8 if not opp_status and outspeed <= 0 else -1.0
        elif mid in BURN_MOVES:
            score = 2.6 if not opp_status else -1.0
        elif mid in SLEEP_MOVES:
            score = 3.4 if not opp_status else -1.5
        elif mid in TOXIC_MOVES:
            score = 2.2 if not opp_status and matchup < 1.0 else -0.8
        elif mid in PROTECT_MOVES:
            score = 1.8 if our_hp < 0.4 or our_status == "TOX" else 0.2
        elif mid in PIVOT_MOVES:
            score = 2.4 if matchup < 0 else 0.6

    if mid in PIVOT_MOVES and cat not in {"status", "3"}:
        if matchup < -0.5:
            score += 2.8
        elif matchup > 1.0:
            score -= 1.2

    score *= 0.35 + 0.65 * acc

    if tera:
        tera_type = getattr(us, "tera_type", None)
        tera_name = str(getattr(tera_type, "name", tera_type) or "").upper()
        if tera_name and them is not None and cat not in {"status", "3"}:
            tera_eff = type_effectiveness(tera_name, pokemon_types(them))
            if tera_eff > eff:
                score += 2.5 * (tera_eff - max(eff, 1.0))
            elif tera_eff < 1.0 and expected < 0.4:
                score -= 2.0
        # Spend tera only when it actually changes the KO / coverage picture.
        if ko < 0.35 and expected < 0.45:
            score -= 1.4
        score += 0.4

    return float(score)


def score_switch(mon: Any, battle: Any, them: Any, *, us: Any) -> float:
    if mon is None or getattr(mon, "fainted", False):
        return -1e6
    matchup = type_matchup(mon, them)
    hp = _hp(mon)
    cost = switch_hazard_cost(mon, battle)
    current = type_matchup(us, them)
    score = matchup * 3.4 + hp * 2.0 - cost * 8.0
    if matchup > 0.9:
        score += 2.0
    if matchup < -0.9:
        score -= 2.5
    # Prefer switching when the active matchup is terrible.
    if current <= -1.0:
        score += 2.2
    if us is not None and ability_id(us) in {"regenerator"} and _hp(us) < 0.66:
        score += 0.8
    return float(score)


def score_actions(
    battle: Any,
    inspected: Optional[VectorizedState] = None,
    *,
    our_estimates: Any = None,
    opp_estimates: Any = None,
) -> np.ndarray:
    """Return a 13-d score vector; illegal actions are zeroed."""
    scores = np.full(N_MOTOR_CHANNELS, -1e5, dtype=np.float32)
    mask = legal_action_mask(battle)
    us = getattr(battle, "active_pokemon", None)
    them = getattr(battle, "opponent_active_pokemon", None)
    moves = list(getattr(battle, "available_moves", None) or [])
    switches = list(getattr(battle, "available_switches", None) or [])
    outspeed = float(getattr(inspected, "outspeed", 0.0) or 0.0) if inspected is not None else 0.0
    damages = list(getattr(inspected, "damage_by_move", None) or []) if inspected is not None else []

    force_switch = bool(getattr(battle, "force_switch", False))
    current = type_matchup(us, them)
    our_hp = _hp(us)
    likely_koed = False
    if them is not None and us is not None:
        # If they have a super-effective STAB and we are slower / low HP, plan a switch.
        their_stab = offensive_stab_vs(them, us)
        likely_koed = their_stab >= 2.0 and (outspeed <= 0 or our_hp < 0.45)

    for i, mv in enumerate(moves[:N_MOVES]):
        dmg = damages[i] if i < len(damages) else None
        if dmg is None:
            try:
                dmg = calculate_damage_range(
                    us, them, mv, battle, user_estimates=our_estimates, target_estimates=opp_estimates
                )
            except Exception:
                dmg = None
        scores[i] = score_move(battle, mv, dmg, us=us, them=them, outspeed=outspeed, tera=False)
        if mask[4 + i] > 0:
            tera_name = None
            if us is not None:
                tera_name = str(
                    getattr(getattr(us, "tera_type", None), "name", getattr(us, "tera_type", None)) or ""
                ) or None
            try:
                tera_dmg = calculate_damage_range(
                    us,
                    them,
                    mv,
                    battle,
                    user_estimates=our_estimates,
                    target_estimates=opp_estimates,
                    tera_override=tera_name,
                )
            except Exception:
                tera_dmg = dmg
            scores[4 + i] = score_move(
                battle, mv, tera_dmg, us=us, them=them, outspeed=outspeed, tera=True
            )

    want_switch = force_switch or (likely_koed and current < 0 and switches)
    for i, mon in enumerate(switches[:N_SWITCHES]):
        scores[8 + i] = score_switch(mon, battle, them, us=us)
        if want_switch:
            scores[8 + i] += 3.0
        if not force_switch and current >= 1.0 and our_hp > 0.5:
            scores[8 + i] -= 2.5

    scores = scores * mask.astype(np.float32)
    # Shift so the best legal action is clearly positive for blending.
    legal = scores[mask > 0]
    if legal.size:
        scores = scores - float(np.max(legal)) + 4.0
        scores = scores * mask.astype(np.float32)
        scores = np.maximum(scores, 0.0)
    return scores.astype(np.float32)


def choose_teampreview(battle: Any) -> str:
    mons = list(getattr(battle, "teampreview_team", None) or (getattr(battle, "team", {}) or {}).values())
    opp = list(
        getattr(battle, "teampreview_opponent_team", None)
        or (getattr(battle, "opponent_team", {}) or {}).values()
    )
    opp_lead = None
    if opp:
        opp_lead = max(
            opp,
            key=lambda m: int((getattr(m, "base_stats", None) or {}).get("spe") or 0),
        )

    def _spe(mon: Any) -> int:
        stats = getattr(mon, "stats", None) or {}
        try:
            spe = int(stats.get("spe") or 0)
        except Exception:
            spe = 0
        if spe <= 0:
            spe = int((getattr(mon, "base_stats", None) or {}).get("spe") or 0)
        return spe

    scored: List[tuple] = []
    for i, mon in enumerate(mons, start=1):
        match = type_matchup(mon, opp_lead) if opp_lead is not None else 0.0
        scored.append((match, _spe(mon), i, mon))
    # Best matchup first; speed as tie-break so we still have a fast lead if even.
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    order = [str(i) for _, _, i, _ in scored]
    while len(order) < 6:
        order.append(str(len(order) + 1))
    chosen = "".join(order[:6])
    for mon in mons:
        try:
            mon._selected_in_teampreview = True
        except Exception:
            pass
    return "/team " + chosen


def self_check_matchups() -> None:
    """Sanity-check type matchups and action scoring without a live server."""
    from types import SimpleNamespace

    assert type_effectiveness("FIRE", ["GRASS"]) == 2.0
    assert type_effectiveness("WATER", ["FIRE"]) == 2.0
    assert type_effectiveness("ELECTRIC", ["GROUND"]) == 0.0

    fire = SimpleNamespace(
        types=["FIRE"],
        base_types=["FIRE"],
        species="Arcanine",
        fainted=False,
        current_hp_fraction=1.0,
        current_hp=100,
        max_hp=100,
        stats={"atk": 150, "spa": 120, "def": 100, "spd": 100, "spe": 110},
        base_stats={"spe": 95},
        boosts={},
        ability=None,
        item=None,
        status=None,
        tera_type="FIRE",
        is_terastallized=False,
        level=100,
    )
    grass = SimpleNamespace(
        types=["GRASS"],
        base_types=["GRASS"],
        species="Rillaboom",
        fainted=False,
        current_hp_fraction=1.0,
        current_hp=100,
        max_hp=100,
        stats={"atk": 140, "spa": 80, "def": 110, "spd": 90, "spe": 85},
        base_stats={"spe": 85},
        boosts={},
        ability=None,
        item=None,
        status=None,
        tera_type="GRASS",
        is_terastallized=False,
        level=100,
        moves={},
    )
    water_switch = SimpleNamespace(
        types=["WATER"],
        base_types=["WATER"],
        species="Dondozo",
        fainted=False,
        current_hp_fraction=1.0,
        current_hp=200,
        max_hp=200,
        stats={"atk": 100, "spa": 80, "def": 160, "spd": 110, "spe": 35},
        base_stats={"spe": 35},
        boosts={},
        ability=None,
        item="heavydutyboots",
        status=None,
        tera_type="WATER",
        is_terastallized=False,
        level=100,
    )
    flamethrower = SimpleNamespace(
        id="flamethrower",
        type="FIRE",
        category="special",
        base_power=90,
        accuracy=100,
        damage=None,
    )
    surf = SimpleNamespace(
        id="surf",
        type="WATER",
        category="special",
        base_power=90,
        accuracy=100,
        damage=None,
    )
    battle = SimpleNamespace(
        active_pokemon=fire,
        opponent_active_pokemon=grass,
        available_moves=[flamethrower, surf],
        available_switches=[water_switch],
        can_tera=False,
        trapped=False,
        force_switch=False,
        side_conditions={},
        opponent_side_conditions={},
        team={"p1: Arcanine": fire, "p1: Dondozo": water_switch},
        opponent_team={"p2: Rillaboom": grass},
        weather=None,
        fields={},
        turn=1,
    )
    scores = score_actions(battle)
    # Fire vs Grass: Flamethrower (idx 0) must beat Surf (idx 1).
    assert scores[0] > scores[1], f"SE Fire move should beat Water vs Grass: {scores}"
    # Offensive type matchup: Fire should prefer staying in vs Grass over switching to Water.
    assert type_matchup(fire, grass) > type_matchup(water_switch, grass)
    assert type_matchup(water_switch, fire) > 0
