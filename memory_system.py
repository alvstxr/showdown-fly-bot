"""Episodic battle memory and opponent set inference.

Keeps a Smogon-informed prior over EV spreads / items / abilities / movesets
and Bayesian-style filtering as the opponent reveals information. A rolling
(state, action, opponent action, ΔHP, hazards) buffer supports pattern
detection (switch-on-SE, setup habits).
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Mapping, Optional, Tuple

import numpy as np

from state_engine import (
    _norm,
    _type_name,
    ability_id,
    hazard_advantage,
    item_id,
    move_id,
    pokemon_types,
    type_effectiveness,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_SET_PATH = ROOT / "data" / "smogon_sets.json"


def _species_key(mon: Any) -> str:
    if mon is None:
        return ""
    return _norm(getattr(mon, "species", None) or getattr(mon, "name", None) or "")


def _hp(mon: Any) -> float:
    if mon is None:
        return 0.0
    return float(getattr(mon, "current_hp_fraction", 0.0) or 0.0)


@dataclass
class SmogonSet:
    name: str
    ability: str
    item: str
    tera: str
    nature: str
    evs: Dict[str, int]
    moves: List[str]
    category: str
    weight: float = 1.0

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SmogonSet":
        evs = { _norm(k): int(v) for k, v in (raw.get("evs") or {}).items() }
        return cls(
            name=str(raw.get("name") or "Unknown"),
            ability=_norm(raw.get("ability")),
            item=_norm(raw.get("item")),
            tera=_norm(raw.get("tera")),
            nature=_norm(raw.get("nature")),
            evs=evs,
            moves=[_norm(m) for m in (raw.get("moves") or [])],
            category=_norm(raw.get("category") or "mixed"),
            weight=float(raw.get("weight") or 1.0),
        )

    def ev_list(self) -> List[int]:
        """HP / Atk / Def / SpA / SpD / Spe."""
        return [
            int(self.evs.get("hp", 0)),
            int(self.evs.get("atk", 0)),
            int(self.evs.get("def", 0)),
            int(self.evs.get("spa", 0)),
            int(self.evs.get("spd", 0)),
            int(self.evs.get("spe", 0)),
        ]

    def as_estimate(self) -> Dict[str, Any]:
        return {
            "nature": self.nature,
            "evs": self.ev_list(),
            "ability": self.ability,
            "item": self.item,
            "tera": self.tera,
            "category": self.category,
            "moves": list(self.moves),
        }


@dataclass
class OpponentProfile:
    species: str
    prior_sets: List[SmogonSet] = field(default_factory=list)
    possible_sets: List[SmogonSet] = field(default_factory=list)
    revealed_moves: List[str] = field(default_factory=list)
    revealed_item: str = ""
    revealed_ability: str = ""
    revealed_tera: str = ""
    physical_evidence: float = 0.0
    special_evidence: float = 0.0

    def observe_move(self, mid: str, *, category: str = "") -> None:
        mid = _norm(mid)
        if not mid or mid in self.revealed_moves:
            return
        self.revealed_moves.append(mid)
        cat = _norm(category)
        if cat in {"physical", "1"}:
            self.physical_evidence += 1.0
        elif cat in {"special", "2"}:
            self.special_evidence += 1.0
        self._filter()

    def observe_item(self, item: str) -> None:
        item = _norm(item)
        if not item or item in {"unknownitem", "unknown"}:
            return
        self.revealed_item = item
        self._filter()

    def observe_ability(self, ability: str) -> None:
        ability = _norm(ability)
        if not ability:
            return
        self.revealed_ability = ability
        self._filter()

    def observe_tera(self, tera: str) -> None:
        tera = _norm(tera)
        if not tera:
            return
        self.revealed_tera = tera
        self._filter()

    def _filter(self) -> None:
        scored: List[SmogonSet] = []
        for s in self.prior_sets:
            w = s.weight
            if self.revealed_item and s.item and s.item != self.revealed_item:
                w *= 0.02
            if self.revealed_ability and s.ability and s.ability != self.revealed_ability:
                w *= 0.05
            if self.revealed_tera and s.tera and s.tera != self.revealed_tera:
                w *= 0.4
            for mv in self.revealed_moves:
                if s.moves and mv not in s.moves:
                    # Unknown coverage: keep a sliver of probability.
                    w *= 0.15
            if self.physical_evidence > self.special_evidence + 0.5 and s.category == "special":
                w *= 0.1
            if self.special_evidence > self.physical_evidence + 0.5 and s.category == "physical":
                w *= 0.1
            if w > 1e-4:
                scored.append(
                    SmogonSet(
                        name=s.name,
                        ability=s.ability,
                        item=s.item,
                        tera=s.tera,
                        nature=s.nature,
                        evs=dict(s.evs),
                        moves=list(s.moves),
                        category=s.category,
                        weight=w,
                    )
                )
        if not scored:
            scored = list(self.prior_sets)
        self.possible_sets = scored

    def map_estimate(self) -> Optional[SmogonSet]:
        if not self.possible_sets:
            return None
        return max(self.possible_sets, key=lambda s: s.weight)

    def blended_estimate(self) -> Dict[str, Any]:
        top = self.map_estimate()
        if top is None:
            return {}
        est = top.as_estimate()
        if self.revealed_item:
            est["item"] = self.revealed_item
        if self.revealed_ability:
            est["ability"] = self.revealed_ability
        if self.revealed_tera:
            est["tera"] = self.revealed_tera
        if self.revealed_moves:
            # Keep revealed moves first, then fill from the MAP set.
            merged = list(self.revealed_moves)
            for mv in top.moves:
                if mv not in merged:
                    merged.append(mv)
            est["moves"] = merged[:4]
        return est


@dataclass
class TurnRecord:
    battle_tag: str
    turn: int
    state: np.ndarray
    action: int
    action_name: str
    opponent_action: str
    delta_hp_us: float
    delta_hp_opp: float
    hazard_state: float
    our_species: str
    opp_species: str
    switched: bool
    opp_switched: bool
    super_effective_pending: bool


class PatternDetector:
    """Lightweight counters over the history buffer."""

    def __init__(self) -> None:
        self.switch_on_se = 0
        self.se_opportunities = 0
        self.setup_turns = 0
        self.total_turns = 0
        self.pivot_turns = 0

    def observe(self, rec: TurnRecord) -> None:
        self.total_turns += 1
        if rec.super_effective_pending:
            self.se_opportunities += 1
            if rec.opp_switched:
                self.switch_on_se += 1
        if rec.action_name in {"swordsdance", "nastyplot", "calmmind", "dragondance", "quiverdance"}:
            self.setup_turns += 1
        if rec.action_name in {"uturn", "voltswitch", "flipturn", "partingshot", "chillyreception"}:
            self.pivot_turns += 1

    def switch_on_se_rate(self) -> float:
        if self.se_opportunities <= 0:
            return 0.5
        return self.switch_on_se / self.se_opportunities

    def setup_rate(self) -> float:
        if self.total_turns <= 0:
            return 0.0
        return self.setup_turns / self.total_turns


class MemorySystem:
    def __init__(self, set_path: Optional[Path] = None, history_len: int = 64) -> None:
        self.set_path = Path(set_path) if set_path else DEFAULT_SET_PATH
        self.sets: Dict[str, List[SmogonSet]] = {}
        self.profiles: Dict[str, Dict[str, OpponentProfile]] = defaultdict(dict)
        self.history: Dict[str, Deque[TurnRecord]] = defaultdict(lambda: deque(maxlen=history_len))
        self.patterns: Dict[str, PatternDetector] = defaultdict(PatternDetector)
        self._last_hp: Dict[str, Tuple[float, float]] = {}
        self._pending_se: Dict[str, bool] = {}
        self.load_sets()

    def load_sets(self) -> None:
        if not self.set_path.exists():
            self.sets = {}
            return
        raw = json.loads(self.set_path.read_text(encoding="utf-8"))
        db: Dict[str, List[SmogonSet]] = {}
        for species, entries in raw.items():
            key = _norm(species)
            db[key] = [SmogonSet.from_dict(e) for e in entries]
        self.sets = db

    def _profile(self, battle_tag: str, species: str) -> OpponentProfile:
        bucket = self.profiles[battle_tag]
        if species not in bucket:
            prior = list(self.sets.get(species, []))
            bucket[species] = OpponentProfile(
                species=species, prior_sets=prior, possible_sets=list(prior)
            )
        return bucket[species]

    def update_from_battle(self, battle: Any) -> None:
        tag = str(getattr(battle, "battle_tag", "") or "")
        for mon in (getattr(battle, "opponent_team", None) or {}).values():
            species = _species_key(mon)
            if not species:
                continue
            profile = self._profile(tag, species)
            abil = ability_id(mon)
            if abil:
                profile.observe_ability(abil)
            it = item_id(mon)
            if it:
                profile.observe_item(it)
            tera = _norm(getattr(mon, "tera_type", None))
            if tera and bool(getattr(mon, "is_terastallized", False)):
                profile.observe_tera(tera)
            moves = getattr(mon, "moves", None) or {}
            for mv in moves.values():
                cat = str(getattr(mv, "category", "") or "")
                profile.observe_move(move_id(mv), category=cat)
        active = getattr(battle, "opponent_active_pokemon", None)
        if active is not None and getattr(active, "last_move", None) is not None:
            profile = self._profile(tag, _species_key(active))
            last = active.last_move
            profile.observe_move(move_id(last), category=str(getattr(last, "category", "") or ""))

    def estimate_for(self, battle: Any, mon: Any) -> Dict[str, Any]:
        tag = str(getattr(battle, "battle_tag", "") or "")
        species = _species_key(mon)
        if not species:
            return {}
        return self._profile(tag, species).blended_estimate()

    def all_opponent_estimates(self, battle: Any) -> Dict[str, Dict[str, Any]]:
        tag = str(getattr(battle, "battle_tag", "") or "")
        return {sp: p.blended_estimate() for sp, p in self.profiles[tag].items()}

    def record(
        self,
        battle: Any,
        *,
        state: np.ndarray,
        action: int,
        action_name: str,
    ) -> TurnRecord:
        tag = str(getattr(battle, "battle_tag", "") or "")
        us = getattr(battle, "active_pokemon", None)
        them = getattr(battle, "opponent_active_pokemon", None)
        our_hp, opp_hp = _hp(us), _hp(them)
        prev = self._last_hp.get(tag, (our_hp, opp_hp))
        d_us = prev[0] - our_hp
        d_opp = prev[1] - opp_hp
        hist = self.history[tag]
        opp_action = ""
        opp_switched = False
        if hist:
            last = hist[-1]
            if last.opp_species and last.opp_species != _species_key(them):
                opp_switched = True
                opp_action = f"switch:{_species_key(them)}"
            elif them is not None and getattr(them, "last_move", None) is not None:
                opp_action = move_id(them.last_move)
        rec = TurnRecord(
            battle_tag=tag,
            turn=int(getattr(battle, "turn", 0) or 0),
            state=np.asarray(state, dtype=np.float32),
            action=int(action),
            action_name=_norm(action_name),
            opponent_action=opp_action,
            delta_hp_us=float(d_us),
            delta_hp_opp=float(d_opp),
            hazard_state=float(hazard_advantage(battle)),
            our_species=_species_key(us),
            opp_species=_species_key(them),
            switched=action >= 8,
            opp_switched=opp_switched,
            super_effective_pending=self._pending_se.get(tag, False),
        )
        hist.append(rec)
        self.patterns[tag].observe(rec)
        self._last_hp[tag] = (our_hp, opp_hp)
        # SE threat for next turn: any of our available damaging moves ≥ 2×.
        pending = False
        if us is not None and them is not None:
            for mv in list(getattr(battle, "available_moves", None) or []):
                mtype = getattr(mv, "type", None)
                if mtype is None:
                    continue
                eff = type_effectiveness(
                    _type_name(mtype),
                    pokemon_types(them, ignore_tera=True),
                    tera_type=_type_name(getattr(them, "tera_type", None)),
                    is_terastallized=bool(getattr(them, "is_terastallized", False)),
                )
                if eff >= 2.0 and int(getattr(mv, "base_power", 0) or 0) > 0:
                    pending = True
                    break
        self._pending_se[tag] = pending
        return rec

    def pattern_features(self, battle: Any) -> np.ndarray:
        tag = str(getattr(battle, "battle_tag", "") or "")
        p = self.patterns[tag]
        return np.asarray(
            [p.switch_on_se_rate(), p.setup_rate(), min(1.0, p.total_turns / 30.0)],
            dtype=np.float32,
        )

    def reset_battle(self, battle_tag: str) -> None:
        self.profiles.pop(battle_tag, None)
        self.history.pop(battle_tag, None)
        self.patterns.pop(battle_tag, None)
        self._last_hp.pop(battle_tag, None)
        self._pending_se.pop(battle_tag, None)
