"""Dopaminergic (RPE/STDP) and serotonergic (risk/patience) neuromodulation.

Drosophila mushroom-body learning is gated by dopaminergic neurons (DANs):
positive and negative prediction errors depress or potentiate Kenyon-cell →
MBON synapses. Independently, a scalar serotonergic gain S ∈ [0, 1] tilts the
policy between conservative information-gathering and high-risk offense.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from state_engine import (
    hazard_advantage,
    residual_chip_fraction,
    status_name,
)


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _sigmoid(x: float) -> float:
    if x >= 20:
        return 1.0
    if x <= -20:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _hp_frac(mon: Any) -> float:
    if mon is None:
        return 0.0
    try:
        return float(getattr(mon, "current_hp_fraction", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _alive_count(team: Any) -> Tuple[int, float]:
    n = 0
    hp = 0.0
    for mon in (team or {}).values():
        if mon is None or getattr(mon, "fainted", False):
            continue
        n += 1
        hp += _hp_frac(mon)
    return n, hp


@dataclass
class TurnSnapshot:
    """Minimal scalars stored between turns so RPE can be computed after the fact."""

    battle_tag: str = ""
    turn: int = 0
    our_hp: float = 1.0
    opp_hp: float = 1.0
    our_alive: int = 6
    opp_alive: int = 6
    our_status: str = ""
    opp_status: str = ""
    hazard: float = 0.0
    our_species: str = ""
    opp_species: str = ""


@dataclass
class NeuromodState:
    rpe: float = 0.0
    serotonin: float = 0.5
    dopamine: float = 0.0
    predicted_value: float = 0.0
    last_value: float = 0.0
    conservative: bool = False
    aggressive: bool = False


class DopaminergicSystem:
    """Reward-prediction-error DANs + eligibility-traced STDP on KC→MBON weights.

    RPE = ΔDamageDealt − ΔDamageTaken + KO bonus + hazard advantage − status penalty
    """

    def __init__(
        self,
        *,
        lr: float = 0.01,
        tau_eligibility: float = 0.85,
        weight_clip: float = 8.0,
        ko_bonus: float = 1.0,
        status_penalty: float = 0.35,
    ) -> None:
        self.lr = lr
        self.tau = tau_eligibility
        self.weight_clip = weight_clip
        self.ko_bonus = ko_bonus
        self.status_penalty = status_penalty
        self.eligibility: Optional[np.ndarray] = None
        self.last_rpe: float = 0.0
        self.predicted_value: float = 0.0

    def reset(self) -> None:
        self.eligibility = None
        self.last_rpe = 0.0
        self.predicted_value = 0.0

    def snapshot(self, battle: Any) -> TurnSnapshot:
        us = getattr(battle, "active_pokemon", None)
        them = getattr(battle, "opponent_active_pokemon", None)
        our_n, our_hp = _alive_count(getattr(battle, "team", None) or {})
        opp_n, opp_hp = _alive_count(getattr(battle, "opponent_team", None) or {})
        return TurnSnapshot(
            battle_tag=str(getattr(battle, "battle_tag", "") or ""),
            turn=int(getattr(battle, "turn", 0) or 0),
            our_hp=_hp_frac(us),
            opp_hp=_hp_frac(them),
            our_alive=our_n,
            opp_alive=opp_n,
            our_status=status_name(us),
            opp_status=status_name(them),
            hazard=hazard_advantage(battle),
            our_species=str(getattr(us, "species", "") or ""),
            opp_species=str(getattr(them, "species", "") or ""),
        )

    def compute_rpe(self, prev: Optional[TurnSnapshot], battle: Any) -> float:
        """Compute the turn-by-turn reward prediction error."""
        now = self.snapshot(battle)
        if prev is None:
            self.last_rpe = 0.0
            return 0.0

        # Δ HP from our perspective: damage dealt is opponent HP lost.
        switched_us = prev.our_species != now.our_species and bool(prev.our_species)
        switched_them = prev.opp_species != now.opp_species and bool(prev.opp_species)
        dmg_dealt = (prev.opp_hp - now.opp_hp) if not switched_them else max(0.0, prev.opp_hp)
        dmg_taken = (prev.our_hp - now.our_hp) if not switched_us else max(0.0, prev.our_hp)

        ko = 0.0
        if now.opp_alive < prev.opp_alive:
            ko += self.ko_bonus
        if now.our_alive < prev.our_alive:
            ko -= self.ko_bonus

        status_pen = 0.0
        if now.our_status and now.our_status != prev.our_status and now.our_status not in {"", "FNT"}:
            status_pen += self.status_penalty
        if now.opp_status and now.opp_status != prev.opp_status and now.opp_status not in {"", "FNT"}:
            status_pen -= self.status_penalty * 0.8

        us = getattr(battle, "active_pokemon", None)
        them = getattr(battle, "opponent_active_pokemon", None)
        chip = residual_chip_fraction(them, battle, is_player=False) - residual_chip_fraction(
            us, battle, is_player=True
        )

        realized = dmg_dealt - dmg_taken + ko + 0.4 * (now.hazard - prev.hazard) - status_pen + 0.25 * chip
        rpe = float(realized - self.predicted_value)
        # Track a slowly moving value baseline (advantage estimate).
        self.predicted_value = 0.8 * self.predicted_value + 0.2 * realized
        self.last_rpe = rpe
        return rpe

    def update_eligibility(self, kc_spikes: np.ndarray, mbon_spikes: np.ndarray) -> np.ndarray:
        """Outer-product eligibility trace: pre (KC) × post (MBON)."""
        pre = np.asarray(kc_spikes, dtype=np.float64).ravel()
        post = np.asarray(mbon_spikes, dtype=np.float64).ravel()
        trace = np.outer(post, pre)
        if self.eligibility is None or self.eligibility.shape != trace.shape:
            self.eligibility = trace
        else:
            self.eligibility = self.tau * self.eligibility + trace
        return self.eligibility

    def apply_stdp(self, weights: np.ndarray, rpe: float) -> np.ndarray:
        """Modulate KC→MBON weights in-place-safe copy via three-factor STDP.

        Positive RPE potentiates synapses that participated (approach MBONs);
        negative RPE depresses them (avoidance). Weights are clipped.
        """
        if self.eligibility is None or self.eligibility.shape != weights.shape:
            return weights
        dw = self.lr * float(rpe) * self.eligibility
        updated = np.clip(weights + dw, -self.weight_clip, self.weight_clip)
        # Decay the trace after a dopaminergic pulse.
        self.eligibility *= 0.5
        return updated.astype(weights.dtype, copy=False)


class SerotonergicSystem:
    """Scalar risk / patience control S ∈ [0, 1].

    High serotonin (S > 0.6): hazards, recovery, safe switches, scouting.
    Low serotonin (S < 0.3): predictions, double-switches, low-accuracy
    high-power moves, offensive Terastallization.
    """

    HIGH = 0.6
    LOW = 0.3

    def __init__(self, *, temperature: float = 1.0) -> None:
        self.temperature = temperature
        self.last_s: float = 0.5

    def compute(
        self,
        battle: Any,
        *,
        win_probability: Optional[float] = None,
        outspeed: float = 0.0,
    ) -> float:
        our_n, our_hp = _alive_count(getattr(battle, "team", None) or {})
        opp_n, opp_hp = _alive_count(getattr(battle, "opponent_team", None) or {})
        team_ratio = our_n / max(1, our_n + opp_n)
        hp_ratio = our_hp / max(1e-6, our_hp + opp_hp)
        if win_probability is None:
            # Cheap heuristic: remaining HP mass, slightly boosted if we outspeed.
            win_probability = 0.55 * hp_ratio + 0.35 * team_ratio + 0.10 * ((outspeed + 1.0) / 2.0)

        us = getattr(battle, "active_pokemon", None)
        them = getattr(battle, "opponent_active_pokemon", None)
        our_hp_active = _hp_frac(us)
        opp_hp_active = _hp_frac(them)
        threatened = 1.0 if our_hp_active < 0.35 else 0.0
        behind = 1.0 if (our_n < opp_n or hp_ratio < 0.4) else 0.0

        # Positive logits → more patience (higher S).
        logit = (
            1.4 * (win_probability - 0.5)
            + 0.9 * (team_ratio - 0.5)
            + 0.6 * (our_hp_active - opp_hp_active)
            - 0.8 * threatened
            - 0.7 * behind
        )
        s = _sigmoid(logit / max(1e-6, self.temperature))
        self.last_s = s
        return float(s)

    def policy_gains(self, s: Optional[float] = None) -> Dict[str, float]:
        """Multipliers applied to motor channels before argmax.

        Channel groups follow connectome_bridge.MOTOR_LAYOUT.
        """
        s = self.last_s if s is None else float(s)
        conservative = s > self.HIGH
        aggressive = s < self.LOW
        return {
            "serotonin": s,
            "conservative": float(conservative),
            "aggressive": float(aggressive),
            "attack": 1.15 if aggressive else (0.85 if conservative else 1.0),
            "tera": 1.25 if aggressive else (0.55 if conservative else 0.9),
            "switch": 1.20 if conservative else (0.80 if aggressive else 1.0),
            "setup_hazard_recovery": 1.30 if conservative else (0.70 if aggressive else 1.0),
            "high_risk_move": 1.35 if aggressive else (0.65 if conservative else 1.0),
            "explore": (1.0 - s) * 0.25,
        }


class NeuromodulatoryLoop:
    """Combines DAN RPE/STDP and 5-HT risk control for one agent lifetime."""

    def __init__(self) -> None:
        self.da = DopaminergicSystem()
        self.ht = SerotonergicSystem()
        self._prev: Dict[str, TurnSnapshot] = {}
        self.last: NeuromodState = NeuromodState()

    def reset_battle(self, battle_tag: str) -> None:
        self._prev.pop(battle_tag, None)
        self.da.predicted_value = 0.0

    def step(
        self,
        battle: Any,
        *,
        kc_spikes: np.ndarray,
        mbon_spikes: np.ndarray,
        kc_mbon_weights: np.ndarray,
        outspeed: float = 0.0,
        win_probability: Optional[float] = None,
    ) -> Tuple[NeuromodState, np.ndarray]:
        tag = str(getattr(battle, "battle_tag", "") or "")
        prev = self._prev.get(tag)
        rpe = self.da.compute_rpe(prev, battle)
        # Credit the previous turn's KC→MBON eligibility with this RPE, then
        # write a fresh trace for the spikes that will produce the next outcome.
        new_weights = self.da.apply_stdp(kc_mbon_weights, rpe)
        self.da.update_eligibility(kc_spikes, mbon_spikes)
        s = self.ht.compute(battle, win_probability=win_probability, outspeed=outspeed)
        self._prev[tag] = self.da.snapshot(battle)
        state = NeuromodState(
            rpe=rpe,
            serotonin=s,
            dopamine=float(np.tanh(rpe)),
            predicted_value=self.da.predicted_value,
            last_value=self.last.predicted_value,
            conservative=s > SerotonergicSystem.HIGH,
            aggressive=s < SerotonergicSystem.LOW,
        )
        self.last = state
        return state, new_weights

    def gains(self) -> Dict[str, float]:
        return self.ht.policy_gains(self.last.serotonin)
