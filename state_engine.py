"""Gen 9 competitive battle state vectorizer and damage engine.

Parses poke-env ``Battle`` / ``Pokemon`` / ``Move`` objects into a fixed-length
normalized float vector for the MaleCNS-inspired network, while exposing the
exact Showdown mechanics used for STAB, stages, status, field, hazards, speed
order, and damage ranges.

The type chart matches games from 2013 onward (X/Y through Gen 9), including
Fairy and the Steel/Ghost/Dark revisions.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Canonical type order (standard chart: Normal … Fairy, plus Stellar)
# ---------------------------------------------------------------------------

TYPES: Tuple[str, ...] = (
    "NORMAL",
    "FIRE",
    "WATER",
    "ELECTRIC",
    "GRASS",
    "ICE",
    "FIGHTING",
    "POISON",
    "GROUND",
    "FLYING",
    "PSYCHIC",
    "BUG",
    "ROCK",
    "GHOST",
    "DRAGON",
    "DARK",
    "STEEL",
    "FAIRY",
    "STELLAR",
)
TYPE_INDEX: Dict[str, int] = {t: i for i, t in enumerate(TYPES)}
N_TYPES = len(TYPES)

# Attacker -> defender -> multiplier. Missing entries are 1.0. Stellar is 1.0
# vs everything; Terastarstorm vs a Terastallized target is handled separately.
_TYPE_CHART: Dict[str, Dict[str, float]] = {
    "NORMAL": {"ROCK": 0.5, "GHOST": 0.0, "STEEL": 0.5},
    "FIRE": {
        "FIRE": 0.5,
        "WATER": 0.5,
        "GRASS": 2.0,
        "ICE": 2.0,
        "BUG": 2.0,
        "ROCK": 0.5,
        "DRAGON": 0.5,
        "STEEL": 2.0,
    },
    "WATER": {
        "FIRE": 2.0,
        "WATER": 0.5,
        "GRASS": 0.5,
        "GROUND": 2.0,
        "ROCK": 2.0,
        "DRAGON": 0.5,
    },
    "ELECTRIC": {
        "WATER": 2.0,
        "ELECTRIC": 0.5,
        "GRASS": 0.5,
        "GROUND": 0.0,
        "FLYING": 2.0,
        "DRAGON": 0.5,
    },
    "GRASS": {
        "FIRE": 0.5,
        "WATER": 2.0,
        "GRASS": 0.5,
        "POISON": 0.5,
        "GROUND": 2.0,
        "FLYING": 0.5,
        "BUG": 0.5,
        "ROCK": 2.0,
        "DRAGON": 0.5,
        "STEEL": 0.5,
    },
    "ICE": {
        "FIRE": 0.5,
        "WATER": 0.5,
        "GRASS": 2.0,
        "ICE": 0.5,
        "GROUND": 2.0,
        "FLYING": 2.0,
        "DRAGON": 2.0,
        "STEEL": 0.5,
    },
    "FIGHTING": {
        "NORMAL": 2.0,
        "ICE": 2.0,
        "POISON": 0.5,
        "FLYING": 0.5,
        "PSYCHIC": 0.5,
        "BUG": 0.5,
        "ROCK": 2.0,
        "GHOST": 0.0,
        "DARK": 2.0,
        "STEEL": 2.0,
        "FAIRY": 0.5,
    },
    "POISON": {
        "GRASS": 2.0,
        "POISON": 0.5,
        "GROUND": 0.5,
        "ROCK": 0.5,
        "GHOST": 0.5,
        "STEEL": 0.0,
        "FAIRY": 2.0,
    },
    "GROUND": {
        "FIRE": 2.0,
        "ELECTRIC": 2.0,
        "GRASS": 0.5,
        "POISON": 2.0,
        "FLYING": 0.0,
        "BUG": 0.5,
        "ROCK": 2.0,
        "STEEL": 2.0,
    },
    "FLYING": {
        "ELECTRIC": 0.5,
        "GRASS": 2.0,
        "FIGHTING": 2.0,
        "BUG": 2.0,
        "ROCK": 0.5,
        "STEEL": 0.5,
    },
    "PSYCHIC": {"FIGHTING": 2.0, "POISON": 2.0, "PSYCHIC": 0.5, "DARK": 0.0, "STEEL": 0.5},
    "BUG": {
        "FIRE": 0.5,
        "GRASS": 2.0,
        "FIGHTING": 0.5,
        "POISON": 0.5,
        "FLYING": 0.5,
        "PSYCHIC": 2.0,
        "GHOST": 0.5,
        "DARK": 2.0,
        "STEEL": 0.5,
        "FAIRY": 0.5,
    },
    "ROCK": {
        "FIRE": 2.0,
        "ICE": 2.0,
        "FIGHTING": 0.5,
        "GROUND": 0.5,
        "FLYING": 2.0,
        "BUG": 2.0,
        "STEEL": 0.5,
    },
    "GHOST": {"NORMAL": 0.0, "PSYCHIC": 2.0, "GHOST": 2.0, "DARK": 0.5},
    "DRAGON": {"DRAGON": 2.0, "STEEL": 0.5, "FAIRY": 0.0},
    "DARK": {"FIGHTING": 0.5, "PSYCHIC": 2.0, "GHOST": 2.0, "DARK": 0.5, "FAIRY": 0.5},
    "STEEL": {
        "FIRE": 0.5,
        "WATER": 0.5,
        "ELECTRIC": 0.5,
        "ICE": 2.0,
        "ROCK": 2.0,
        "STEEL": 0.5,
        "FAIRY": 2.0,
    },
    "FAIRY": {
        "FIRE": 0.5,
        "FIGHTING": 2.0,
        "POISON": 0.5,
        "DRAGON": 2.0,
        "DARK": 2.0,
        "STEEL": 0.5,
    },
    "STELLAR": {},
}

# Attack / Defense / SpA / SpD / Speed stage → exact multiplier.
# +2 = 2.0x, -1 = 2/3 ≈ 0.666..., matching Showdown.
_STAT_STAGE_NUM = {s: (2 + s if s >= 0 else 2) for s in range(-6, 7)}
_STAT_STAGE_DEN = {s: (2 if s >= 0 else 2 - s) for s in range(-6, 7)}
# Accuracy / Evasion use (3+n)/3.
_ACC_STAGE_NUM = {s: (3 + s if s >= 0 else 3) for s in range(-6, 7)}
_ACC_STAGE_DEN = {s: (3 if s >= 0 else 3 - s) for s in range(-6, 7)}

BOOST_STATS: Tuple[str, ...] = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
CORE_STATS: Tuple[str, ...] = ("hp", "atk", "def", "spa", "spd", "spe")

MAJOR_STATUSES: Tuple[str, ...] = ("BRN", "PAR", "TOX", "PSN", "SLP", "FRZ", "FNT")

MINOR_EFFECTS: Tuple[str, ...] = (
    "TAUNT",
    "ENCORE",
    "DISABLE",
    "CONFUSION",
    "LEECH_SEED",
    "SUBSTITUTE",
    "SALT_CURE",
    "ATTRACT",
    "TORMENT",
    "HEAL_BLOCK",
    "YAWN",
    "CURSE",
    "INGRAIN",
    "AQUA_RING",
    "MAGNET_RISE",
    "SMACK_DOWN",
    "PERISH3",
    "LOCKED_MOVE",
    "PROTECT",
    "FLASH_FIRE",
)

WEATHERS: Tuple[str, ...] = (
    "SUNNYDAY",
    "RAINDANCE",
    "SANDSTORM",
    "SNOW",
    "SNOWSCAPE",
    "DESOLATELAND",
    "PRIMORDIALSEA",
    "DELTASTREAM",
)
TERRAINS: Tuple[str, ...] = (
    "ELECTRIC_TERRAIN",
    "GRASSY_TERRAIN",
    "PSYCHIC_TERRAIN",
    "MISTY_TERRAIN",
)

COMPETITIVE_ABILITIES: Tuple[str, ...] = (
    "intimidate",
    "regenerator",
    "magicbounce",
    "supremeoverlord",
    "sturdy",
    "drizzle",
    "drought",
    "sandstream",
    "snowwarning",
    "unaware",
    "adaptability",
    "protean",
    "libero",
    "technician",
    "hugepower",
    "purepower",
    "guts",
    "flashfire",
    "levitate",
    "wellbakedbody",
    "waterabsorb",
    "voltabsorb",
    "lightningrod",
    "stormdrain",
    "dryskin",
    "thickfat",
    "multiscale",
    "shadowshield",
    "disguise",
    "iceface",
    "goodasgold",
    "neutralizinggas",
    "imposter",
    "speedboost",
    "chlorophyll",
    "swiftswim",
    "sandrush",
    "slushrush",
    "quarkdrive",
    "protosynthesis",
    "windrider",
    "swordofruin",
    "beadsofruin",
    "vesselofruin",
    "tabletsofruin",
    "poisonheal",
    "naturalcure",
    "magicguard",
    "roughskin",
    "ironbarbs",
    "flamebody",
    "static",
    "poisonpoint",
    "effectspore",
    "cursedbody",
    "moxie",
    "beastboost",
    "soulheart",
    "defiant",
    "competitive",
    "furcoat",
    "filter",
    "solidrock",
    "prismarmor",
    "wonderguard",
    "harvest",
    "prankster",
    "galewings",
    "triage",
    "queenlymajesty",
    "dazzling",
    "armortail",
    "unburden",
    "moldbreaker",
    "tintedlens",
    "compoundeyes",
    "superluck",
    "sniper",
    "sheerforce",
    "toughclaws",
    "ironfist",
    "strongjaw",
    "sharpness",
    "rockypayload",
    "eartheater",
    "thermalexchange",
    "purifyingsalt",
    "seedsower",
    "electricsurge",
    "psychicsurge",
    "grassysurge",
    "mistysurge",
    "hadronengine",
    "orichalcumpulse",
    "asoneglastrier",
    "asonespectrier",
)

COMPETITIVE_ITEMS: Tuple[str, ...] = (
    "choiceband",
    "choicespecs",
    "choicescarf",
    "lifeorb",
    "leftovers",
    "heavydutyboots",
    "focussash",
    "boosterenergy",
    "assaultvest",
    "blacksludge",
    "rockyhelmet",
    "airballoon",
    "eviolite",
    "lightclay",
    "terrainextender",
    "heatrock",
    "damprock",
    "smoothrock",
    "icyrock",
    "loadeddice",
    "punchingglove",
    "expertbelt",
    "weaknesspolicy",
    "lumberry",
    "sitrusberry",
    "whiteherb",
    "mentalherb",
    "powerherb",
    "throatspray",
    "mirrorherb",
    "clearamulet",
    "covertcloak",
    "safetygoggles",
    "shedshell",
    "redcard",
    "ejectbutton",
    "ejectpack",
    "flameorb",
    "toxicorb",
    "stickybarb",
    "ironball",
    "laggingtail",
    "scopelens",
    "widelens",
    "brightpowder",
    "quickclaw",
    "custapberry",
    "keeberry",
    "marangaberry",
    "blunderpolicy",
    "wellspringmask",
    "hearthflamemask",
    "cornerstonemask",
    "loadeddice",
    "punchingglove",
    "covertcloak",
    "clearamulet",
    "boosterenergy",
    "airballoon",
    "focussash",
    "lifeorb",
    "leftovers",
    "heavydutyboots",
    "choicescarf",
)

# Deduplicate while preserving order so one-hot indices stay stable.
COMPETITIVE_ITEMS = tuple(dict.fromkeys(COMPETITIVE_ITEMS))

ABILITY_INDEX = {a: i for i, a in enumerate(COMPETITIVE_ABILITIES)}
ITEM_INDEX = {it: i for i, it in enumerate(COMPETITIVE_ITEMS)}

HAZARD_REMOVAL_MOVES = frozenset(
    {"rapidspin", "defog", "tidyup", "mortalspin", "courtchange"}
)
HAZARD_SETTING_MOVES = frozenset(
    {
        "stealthrock",
        "spikes",
        "toxicspikes",
        "stickyweb",
        "ceaselessedge",
        "stoneaxe",
        "gmaxsteelsurge",
    }
)
SETUP_MOVES = frozenset(
    {
        "swordsdance",
        "nastyplot",
        "calmmind",
        "quiverdance",
        "dragondance",
        "bulkup",
        "irondefense",
        "coil",
        "tailglow",
        "shellsmash",
        "agility",
        "rockpolish",
        "autotomize",
        "growth",
        "workup",
        "honeclaws",
        "shiftgear",
        "victorydance",
        "clangoroussoul",
        "takeheart",
    }
)
PROTECT_MOVES = frozenset(
    {
        "protect",
        "detect",
        "spikyshield",
        "banefulbunker",
        "kingsshield",
        "obstruct",
        "silktrap",
        "burningbulwark",
        "endure",
    }
)
RECOVERY_MOVES = frozenset(
    {
        "recover",
        "softboiled",
        "roost",
        "slackoff",
        "moonlight",
        "morningsun",
        "synthesis",
        "shoreup",
        "rest",
        "wish",
        "healorder",
        "milkdrink",
        "strengthsap",
        "lunarblessing",
        "junglehealing",
    }
)
PIVOT_MOVES = frozenset(
    {
        "uturn",
        "voltswitch",
        "flipturn",
        "partingshot",
        "teleport",
        "chillyreception",
        "shedtail",
        "batonpass",
    }
)
PRIORITY_TABLE: Dict[str, int] = {
    "helpinghand": 5,
    "protect": 4,
    "detect": 4,
    "endure": 4,
    "spikyshield": 4,
    "banefulbunker": 4,
    "burningbulwark": 4,
    "kingsshield": 4,
    "obstruct": 4,
    "silktrap": 4,
    "maxguard": 4,
    "fakeout": 3,
    "afteryou": 3,
    "allyswitch": 2,
    "feint": 2,
    "firstimpression": 2,
    "extremespeed": 2,
    "followme": 2,
    "ragepowder": 2,
    "spotlight": 2,
    "accelerock": 1,
    "aquajet": 1,
    "bulletpunch": 1,
    "iceshard": 1,
    "machpunch": 1,
    "quickattack": 1,
    "shadowsneak": 1,
    "suckerpunch": 1,
    "vacuumwave": 1,
    "watershuriken": 1,
    "jetpunch": 1,
    "thunderclap": 1,
    "upperhand": 1,
    "aquacutter": 1,
    "vitalthrow": -1,
    "focuspunch": -3,
    "beakblast": -3,
    "shelltrap": -3,
    "avalanche": -4,
    "revenge": -4,
    "counter": -5,
    "mirrorcoat": -5,
    "metalburst": -5,
    "roar": -6,
    "whirlwind": -6,
    "dragontail": -6,
    "circlethrow": -6,
    "trickroom": -7,
}

NATURE_MODS: Dict[str, Tuple[str, str]] = {
    "adamant": ("atk", "spa"),
    "brave": ("atk", "spe"),
    "lonely": ("atk", "def"),
    "naughty": ("atk", "spd"),
    "bold": ("def", "atk"),
    "relaxed": ("def", "spe"),
    "impish": ("def", "spa"),
    "lax": ("def", "spd"),
    "modest": ("spa", "atk"),
    "quiet": ("spa", "spe"),
    "mild": ("spa", "def"),
    "rash": ("spa", "spd"),
    "calm": ("spd", "atk"),
    "sassy": ("spd", "spe"),
    "gentle": ("spd", "def"),
    "careful": ("spd", "spa"),
    "timid": ("spe", "atk"),
    "hasty": ("spe", "def"),
    "jolly": ("spe", "spa"),
    "naive": ("spe", "spd"),
}

# Motor / legal-action layout used by the VNC decoder.
N_MOVES = 4
N_SWITCHES = 5
# Attack 1-4, Tera+Attack 1-4, Switch 1-5  →  13 Showdown orders.
# Compact 10-channel view: Attack 1-4, Switch 1-5, Tera-intent.
N_MOTOR_CHANNELS = 13
N_MOTOR_COMPACT = 10

# Feature block sizes (must match the encoders below).
MOVE_FEATURE_DIM = 40
ACTIVE_FEATURE_DIM = 437
SLOT_FEATURE_DIM = 58
FIELD_FEATURE_DIM = 48
GLOBAL_FEATURE_DIM = 34
STATE_DIM = (
    2 * ACTIVE_FEATURE_DIM
    + 12 * SLOT_FEATURE_DIM
    + FIELD_FEATURE_DIM
    + GLOBAL_FEATURE_DIM
)  # 1652


# ---------------------------------------------------------------------------
# String / enum helpers
# ---------------------------------------------------------------------------

def _norm(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "name"):
        value = value.name
    text = str(value).strip()
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _type_name(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "name"):
        name = str(value.name).upper()
    else:
        name = str(value).upper()
    name = name.replace(" ", "").replace("-", "").replace("_", "")
    aliases = {
        "TYPENULL": "NORMAL",
        "THREEQUESTIONMARKS": "NORMAL",
        "???": "NORMAL",
        "SNOWSCAPE": "SNOW",
    }
    return aliases.get(name, name)


def _one_hot(index: int, size: int) -> np.ndarray:
    vec = np.zeros(size, dtype=np.float32)
    if 0 <= index < size:
        vec[index] = 1.0
    return vec


def _types_one_hot(types: Sequence[str]) -> np.ndarray:
    vec = np.zeros(N_TYPES, dtype=np.float32)
    for t in types:
        idx = TYPE_INDEX.get(_type_name(t))
        if idx is not None:
            vec[idx] = 1.0
    return vec


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _safe_frac(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num / den)


def _enum_in(container: Mapping[Any, Any], *names: str) -> bool:
    if not container:
        return False
    wanted = {_norm(n) for n in names}
    for key in container:
        if _norm(key) in wanted:
            return True
    return False


def _enum_value(container: Mapping[Any, Any], *names: str, default: int = 0) -> int:
    if not container:
        return default
    wanted = {_norm(n) for n in names}
    for key, val in container.items():
        if _norm(key) in wanted:
            try:
                return int(val)
            except (TypeError, ValueError):
                return 1
    return default


def _effect_active(mon: Any, *names: str) -> bool:
    effects = getattr(mon, "effects", None) or {}
    wanted = {_norm(n) for n in names}
    for key in effects:
        if _norm(key) in wanted:
            return True
    return False


# ---------------------------------------------------------------------------
# Type efficacy
# ---------------------------------------------------------------------------

def type_multiplier(attack_type: str, defend_type: str) -> float:
    atk = _type_name(attack_type)
    dfn = _type_name(defend_type)
    if atk not in _TYPE_CHART or dfn not in TYPE_INDEX:
        return 1.0
    return float(_TYPE_CHART[atk].get(dfn, 1.0))


def type_effectiveness(
    attack_type: str,
    defender_types: Sequence[str],
    *,
    tera_type: Optional[str] = None,
    is_terastallized: bool = False,
    move_id: str = "",
) -> float:
    """Dual-type product in {0, 0.25, 0.5, 1, 2, 4}.

    Terastallized Pokémon (except Stellar) are only the Tera type defensively.
    Terastarstorm is 2× against any Terastallized target.
    """
    atk = _type_name(attack_type)
    if _norm(move_id) in {"terastarstorm", "terablast"} and atk == "STELLAR":
        if is_terastallized:
            return 2.0
    if is_terastallized and tera_type and _type_name(tera_type) != "STELLAR":
        types = [_type_name(tera_type)]
    else:
        types = [_type_name(t) for t in defender_types if _type_name(t)]
    if not types:
        types = ["NORMAL"]
    product = 1.0
    for t in types:
        product *= type_multiplier(atk, t)
    return float(product)


def type_chart_matrix() -> np.ndarray:
    mat = np.ones((N_TYPES, N_TYPES), dtype=np.float32)
    for i, atk in enumerate(TYPES):
        for j, dfn in enumerate(TYPES):
            mat[i, j] = type_multiplier(atk, dfn)
    return mat


# ---------------------------------------------------------------------------
# Stat stages
# ---------------------------------------------------------------------------

def stage_multiplier(stage: int, *, accuracy_evasion: bool = False) -> float:
    """Exact Showdown stage multiplier, clamped to [-6, +6]."""
    stage = int(max(-6, min(6, stage)))
    if accuracy_evasion:
        return _ACC_STAGE_NUM[stage] / _ACC_STAGE_DEN[stage]
    return _STAT_STAGE_NUM[stage] / _STAT_STAGE_DEN[stage]


def boosts_of(mon: Any) -> Dict[str, int]:
    raw = getattr(mon, "boosts", None) or {}
    out = {k: 0 for k in BOOST_STATS}
    for key, val in raw.items():
        k = _norm(key)
        if k in ("spatk", "spa", "specialattack"):
            k = "spa"
        elif k in ("spdef", "spd", "specialdefense"):
            k = "spd"
        elif k in ("atk", "attack"):
            k = "atk"
        elif k in ("def", "defense"):
            k = "def"
        elif k in ("spe", "spdstat", "speed"):
            if k == "spdstat":
                k = "spd"
            else:
                k = "spe"
        elif k in ("accuracy", "acc"):
            k = "accuracy"
        elif k in ("evasion", "eva"):
            k = "evasion"
        if k in out:
            try:
                out[k] = int(val)
            except (TypeError, ValueError):
                out[k] = 0
    return out


# ---------------------------------------------------------------------------
# Status, items, abilities
# ---------------------------------------------------------------------------

def status_name(mon: Any) -> str:
    st = getattr(mon, "status", None)
    if st is None:
        return ""
    return _type_name(st) if hasattr(st, "name") else _norm(st).upper()


def ability_id(mon: Any) -> str:
    return _norm(getattr(mon, "ability", None) or getattr(mon, "base_ability", None))


def item_id(mon: Any) -> str:
    return _norm(getattr(mon, "item", None))


def move_id(move: Any) -> str:
    if move is None:
        return ""
    return _norm(getattr(move, "id", None) or getattr(move, "name", None) or move)


def pokemon_types(mon: Any, *, ignore_tera: bool = False) -> List[str]:
    if mon is None:
        return []
    if not ignore_tera and bool(getattr(mon, "is_terastallized", False)):
        tera = _type_name(getattr(mon, "tera_type", None))
        if tera and tera != "STELLAR":
            return [tera]
    raw = getattr(mon, "base_types", None) or getattr(mon, "types", None) or []
    t1 = getattr(mon, "type_1", None)
    t2 = getattr(mon, "type_2", None)
    if t1 is not None and not raw:
        raw = [t1] + ([t2] if t2 is not None else [])
    names = [_type_name(t) for t in raw]
    return [n for n in names if n and n != "STELLAR"]


def is_grounded(mon: Any, battle: Any = None) -> bool:
    if mon is None:
        return True
    fields = getattr(battle, "fields", None) or {}
    if _enum_in(fields, "GRAVITY"):
        return True
    if item_id(mon) == "ironball":
        return True
    if _effect_active(mon, "INGRAIN", "SMACK_DOWN", "THOUSAND_ARROWS"):
        return True
    if item_id(mon) == "airballoon" and not _effect_active(mon, "SMACK_DOWN"):
        return False
    if ability_id(mon) == "levitate" and not _enum_in(fields, "GRAVITY"):
        return False
    if _effect_active(mon, "MAGNET_RISE", "TELEKINESIS"):
        return False
    types = pokemon_types(mon)
    if "FLYING" in types and not _effect_active(mon, "ROOST"):
        return False
    return True


def has_heavy_duty_boots(mon: Any) -> bool:
    return item_id(mon) == "heavydutyboots"


# ---------------------------------------------------------------------------
# Weather / terrain / field
# ---------------------------------------------------------------------------

def active_weather(battle: Any) -> str:
    weather = getattr(battle, "weather", None) or {}
    for key in weather:
        name = _type_name(key)
        if name and name != "UNKNOWN":
            if name == "SNOWSCAPE":
                return "SNOW"
            return name
    return ""


def active_terrain(battle: Any) -> str:
    fields = getattr(battle, "fields", None) or {}
    for key in fields:
        name = _type_name(key)
        if name in TERRAINS:
            return name
    return ""


def weather_power_mod(move_type: str, weather: str) -> float:
    mt = _type_name(move_type)
    w = _type_name(weather)
    if w in {"SUNNYDAY", "DESOLATELAND"}:
        if mt == "FIRE":
            return 1.5
        if mt == "WATER":
            return 0.0 if w == "DESOLATELAND" else 0.5
    if w in {"RAINDANCE", "PRIMORDIALSEA"}:
        if mt == "WATER":
            return 1.5
        if mt == "FIRE":
            return 0.0 if w == "PRIMORDIALSEA" else 0.5
    if w == "DELTASTREAM" and mt in {"ELECTRIC", "ICE", "ROCK"}:
        # Strong Winds: Flying no longer weak to these; handled on defense.
        return 1.0
    return 1.0


def terrain_power_mod(move_type: str, terrain: str, user_grounded: bool) -> float:
    if not user_grounded:
        return 1.0
    mt = _type_name(move_type)
    t = _type_name(terrain)
    if t == "ELECTRIC_TERRAIN" and mt == "ELECTRIC":
        return 1.3
    if t == "GRASSY_TERRAIN" and mt == "GRASS":
        return 1.3
    if t == "PSYCHIC_TERRAIN" and mt == "PSYCHIC":
        return 1.3
    if t == "MISTY_TERRAIN" and mt == "DRAGON":
        return 0.5
    return 1.0


# ---------------------------------------------------------------------------
# Hazards
# ---------------------------------------------------------------------------

def stealth_rock_fraction(mon: Any) -> float:
    """Fraction of max HP lost on switch-in from Stealth Rock."""
    if mon is None or has_heavy_duty_boots(mon):
        return 0.0
    if ability_id(mon) in {"magicguard", "overcoat"}:
        return 0.0
    eff = type_effectiveness("ROCK", pokemon_types(mon))
    return float(0.125 * eff)


def spikes_fraction(layers: int, mon: Any, battle: Any = None) -> float:
    layers = int(max(0, min(3, layers)))
    if layers == 0 or mon is None:
        return 0.0
    if has_heavy_duty_boots(mon) or ability_id(mon) == "magicguard":
        return 0.0
    if not is_grounded(mon, battle):
        return 0.0
    return {1: 1.0 / 8.0, 2: 1.0 / 6.0, 3: 1.0 / 4.0}[layers]


def sticky_web_applies(mon: Any, battle: Any = None) -> bool:
    if mon is None or has_heavy_duty_boots(mon):
        return False
    if ability_id(mon) in {"magicguard", "clearbody", "whitesmoke", "fullmetalbody"}:
        return False
    return is_grounded(mon, battle)


def switch_in_hazard_fraction(mon: Any, side_conditions: Mapping[Any, Any], battle: Any = None) -> float:
    dmg = 0.0
    if _enum_in(side_conditions, "STEALTH_ROCK"):
        dmg += stealth_rock_fraction(mon)
    layers = _enum_value(side_conditions, "SPIKES")
    dmg += spikes_fraction(layers, mon, battle)
    if _enum_in(side_conditions, "G_MAX_STEELSURGE", "GMAXSTEELSURGE"):
        if not has_heavy_duty_boots(mon):
            dmg += 0.125 * type_effectiveness("STEEL", pokemon_types(mon))
    return float(min(1.0, dmg))


# ---------------------------------------------------------------------------
# Stats / speed
# ---------------------------------------------------------------------------

def nature_multiplier(nature: Optional[str], stat: str) -> float:
    n = _norm(nature)
    if n not in NATURE_MODS:
        return 1.0
    plus, minus = NATURE_MODS[n]
    if stat == plus:
        return 1.1
    if stat == minus:
        return 0.9
    return 1.0


def calc_stat(
    base: int,
    *,
    level: int = 50,
    iv: int = 31,
    ev: int = 0,
    nature: float = 1.0,
    hp: bool = False,
) -> int:
    if hp:
        if base == 1:
            return 1
        return int((2 * base + iv + ev // 4) * level / 100) + level + 10
    inner = int((2 * base + iv + ev // 4) * level / 100) + 5
    return int(inner * nature)


def stat_dict(mon: Any, estimates: Optional[Mapping[str, Any]] = None) -> Dict[str, float]:
    """Return {hp, atk, def, spa, spd, spe} as floats.

    Prefer poke-env computed stats; otherwise reconstruct from base stats at
    level 50 with 31 IVs. ``estimates`` may supply EV/nature guesses.
    """
    out = {k: 0.0 for k in CORE_STATS}
    if mon is None:
        return out
    stats = getattr(mon, "stats", None) or {}
    for key in CORE_STATS:
        val = stats.get(key)
        if val is None and key == "spa":
            val = stats.get("spa") or stats.get("spd") if False else stats.get("spa")
        if val is not None:
            try:
                out[key] = float(val)
            except (TypeError, ValueError):
                pass
    if all(out[k] > 0 for k in CORE_STATS if k != "hp") and out["hp"] > 0:
        return out
    bases = getattr(mon, "base_stats", None) or {}
    level = int(getattr(mon, "level", None) or 50)
    evs = list((estimates or {}).get("evs") or getattr(mon, "evs", None) or [0] * 6)
    while len(evs) < 6:
        evs.append(0)
    ivs = list(getattr(mon, "ivs", None) or [31] * 6)
    while len(ivs) < 6:
        ivs.append(31)
    nature = (estimates or {}).get("nature") or getattr(mon, "nature", None)
    mapping = [
        ("hp", 0, True),
        ("atk", 1, False),
        ("def", 2, False),
        ("spa", 3, False),
        ("spd", 4, False),
        ("spe", 5, False),
    ]
    for name, idx, is_hp in mapping:
        if out[name] > 0:
            continue
        base = 0
        try:
            base = int(bases.get(name, 0) or 0)
        except (TypeError, ValueError):
            base = 0
        nat = 1.0 if is_hp else nature_multiplier(nature, name)
        out[name] = float(
            calc_stat(base, level=level, iv=int(ivs[idx]), ev=int(evs[idx]), nature=nat, hp=is_hp)
        )
    max_hp = getattr(mon, "max_hp", None)
    if max_hp and out["hp"] <= 0:
        try:
            out["hp"] = float(max_hp)
        except (TypeError, ValueError):
            pass
    return out


def modified_attack_stat(
    mon: Any,
    *,
    special: bool,
    crit: bool = False,
    unaware_defender: bool = False,
    estimates: Optional[Mapping[str, Any]] = None,
) -> float:
    stats = stat_dict(mon, estimates)
    key = "spa" if special else "atk"
    stage = boosts_of(mon)[key]
    if unaware_defender:
        stage = 0
    if crit and stage < 0:
        stage = 0
    value = stats[key] * stage_multiplier(stage)
    abil = ability_id(mon)
    st = status_name(mon)
    if not special and st == "BRN" and abil != "guts":
        # Burn is applied later as a damage modifier unless Guts; Guts boosts Atk.
        pass
    if abil == "guts" and st and st not in {"", "FNT"}:
        if not special:
            value *= 1.5
    if abil in {"hugepower", "purepower"} and not special:
        value *= 2.0
    if abil == "hugepower":
        pass
    item = item_id(mon)
    if item == "choiceband" and not special:
        value *= 1.5
    if item == "choicespecs" and special:
        value *= 1.5
    if item == "lifeorb":
        pass  # damage-side modifier
    if abil == "orichalcumpulse" and not special:
        value *= 1.3
    if abil == "hadronengine" and special:
        value *= 1.3
    return float(value)


def modified_defense_stat(
    mon: Any,
    *,
    special: bool,
    crit: bool = False,
    unaware_attacker: bool = False,
    snow: bool = False,
    sand: bool = False,
    estimates: Optional[Mapping[str, Any]] = None,
) -> float:
    stats = stat_dict(mon, estimates)
    key = "spd" if special else "def"
    stage = boosts_of(mon)[key]
    if unaware_attacker:
        stage = 0
    if crit and stage > 0:
        stage = 0
    value = stats[key] * stage_multiplier(stage)
    types = pokemon_types(mon)
    if snow and not special and "ICE" in types:
        value *= 1.5
    if sand and special and "ROCK" in types:
        value *= 1.5
    if ability_id(mon) == "furcoat" and not special:
        value *= 2.0
    if item_id(mon) == "eviolite":
        value *= 1.5
    if item_id(mon) == "assaultvest" and special:
        value *= 1.5
    return float(value)


def effective_speed(
    mon: Any,
    battle: Any = None,
    *,
    side_is_player: bool = True,
    estimates: Optional[Mapping[str, Any]] = None,
) -> float:
    """Speed used for order, including items, abilities, paralysis, Tailwind."""
    if mon is None:
        return 0.0
    stats = stat_dict(mon, estimates)
    spe = stats["spe"] * stage_multiplier(boosts_of(mon)["spe"])
    abil = ability_id(mon)
    item = item_id(mon)
    weather = active_weather(battle)
    terrain = active_terrain(battle)
    st = status_name(mon)

    if st == "PAR" and abil != "quickfeet":
        spe *= 0.5
    if abil == "quickfeet" and st and st not in {"", "FNT"}:
        spe *= 1.5
    if item == "choicescarf":
        spe *= 1.5
    if item == "ironball":
        spe *= 0.5
    if item in {"laggingtail", "fullincense"}:
        spe *= 1.0  # trick-room-like priority bracket handled separately
    if abil == "swiftswim" and weather in {"RAINDANCE", "PRIMORDIALSEA"}:
        spe *= 2.0
    if abil == "chlorophyll" and weather in {"SUNNYDAY", "DESOLATELAND"}:
        spe *= 2.0
    if abil == "sandrush" and weather == "SANDSTORM":
        spe *= 2.0
    if abil == "slushrush" and weather in {"SNOW", "SNOWSCAPE", "HAIL"}:
        spe *= 2.0
    if abil == "surgesurfer" and terrain == "ELECTRIC_TERRAIN":
        spe *= 2.0
    if abil == "unburden" and _effect_active(mon, "UNBURDEN"):
        spe *= 2.0
    if _effect_active(
        mon,
        "PROTOSYNTHESISSPE",
        "QUARKDRIVESPE",
        "PROTOSYNTHESIS_SPE",
        "QUARK_DRIVE_SPE",
    ):
        spe *= 1.5
    elif abil in {"protosynthesis", "quarkdrive"} and item == "boosterenergy":
        # If we cannot see which stat was boosted, assume Speed when it is highest.
        if stats["spe"] == max(stats[k] for k in ("atk", "def", "spa", "spd", "spe")):
            spe *= 1.5

    side = getattr(battle, "side_conditions", None) if side_is_player else getattr(
        battle, "opponent_side_conditions", None
    )
    if _enum_in(side or {}, "TAILWIND"):
        spe *= 2.0
    if ability_id(mon) == "windrider" and _enum_in(side or {}, "TAILWIND"):
        # Wind Rider raises Attack, not Speed, when Tailwind goes up.
        pass
    return float(spe)


def trick_room_active(battle: Any) -> bool:
    return _enum_in(getattr(battle, "fields", None) or {}, "TRICK_ROOM")


def speed_order(
    us: Any,
    them: Any,
    battle: Any,
    *,
    our_estimates: Optional[Mapping[str, Any]] = None,
    opp_estimates: Optional[Mapping[str, Any]] = None,
) -> Tuple[float, float, float]:
    """Return (our_speed, their_speed, outspeed_score in [-1, 1])."""
    our = effective_speed(us, battle, side_is_player=True, estimates=our_estimates)
    their = effective_speed(them, battle, side_is_player=False, estimates=opp_estimates)
    if trick_room_active(battle):
        our_cmp, their_cmp = -our, -their
    else:
        our_cmp, their_cmp = our, their
    if our_cmp > their_cmp:
        score = 1.0
    elif our_cmp < their_cmp:
        score = -1.0
    else:
        score = 0.0  # speed tie
    return our, their, score


# ---------------------------------------------------------------------------
# Priority
# ---------------------------------------------------------------------------

def move_priority(move: Any, *, terrain: str = "", user: Any = None) -> int:
    """Priority bracket in [-7, +5]."""
    pid = move_id(move)
    prio = None
    if move is not None and getattr(move, "priority", None) is not None:
        try:
            prio = int(move.priority)
        except (TypeError, ValueError):
            prio = None
    if prio is None:
        prio = PRIORITY_TABLE.get(pid, 0)
    abil = ability_id(user) if user is not None else ""
    cat = _norm(getattr(move, "category", None)) if move is not None else ""
    if pid == "grassyglide" and _type_name(terrain) == "GRASSY_TERRAIN":
        prio = max(prio, 1)
    if abil == "prankster" and cat in {"status", "3"}:
        prio += 1
    if abil == "galewings":
        mtype = _type_name(getattr(move, "type", None)) if move is not None else ""
        hp = float(getattr(user, "current_hp_fraction", 1.0) or 1.0)
        if mtype == "FLYING" and hp >= 1.0:
            prio += 1
    if abil == "triage" and move is not None:
        heal = float(getattr(move, "heal", 0.0) or 0.0)
        drain = float(getattr(move, "drain", 0.0) or 0.0)
        if heal > 0 or drain > 0 or pid in RECOVERY_MOVES:
            prio += 3
    return int(max(-7, min(5, prio)))


def psychic_terrain_blocks_priority(prio: int, target_grounded: bool, terrain: str) -> bool:
    return (
        prio > 0
        and target_grounded
        and _type_name(terrain) == "PSYCHIC_TERRAIN"
    )


# ---------------------------------------------------------------------------
# STAB (including Adaptability + Tera)
# ---------------------------------------------------------------------------

def stab_multiplier(
    move_type: str,
    user: Any,
    *,
    adaptability: Optional[bool] = None,
) -> float:
    """Gen 9 STAB, including Terastallization and Adaptability (2.25 cap)."""
    mt = _type_name(move_type)
    if not mt or user is None:
        return 1.0
    abil = ability_id(user)
    if adaptability is None:
        adaptability = abil == "adaptability"
    original = [_type_name(t) for t in (getattr(user, "base_types", None) or pokemon_types(user, ignore_tera=True))]
    tera = _type_name(getattr(user, "tera_type", None))
    is_tera = bool(getattr(user, "is_terastallized", False))
    matches_original = mt in original
    matches_tera = is_tera and tera == mt and tera != "STELLAR"
    if is_tera and tera == "STELLAR":
        # Stellar: 1.2× once per type on non-Stellar moves; 1.5/2.0 on Stellar moves.
        if mt == "STELLAR":
            base = 2.0 if matches_original else 1.5
        else:
            base = 1.2 if not matches_original else 1.5
    elif matches_tera and matches_original:
        base = 2.0
    elif matches_tera or matches_original:
        base = 1.5
    else:
        base = 1.0
    if adaptability and (matches_original or matches_tera):
        if base >= 2.0:
            base = 2.25
        elif base >= 1.5:
            base = 2.0
    if abil in {"protean", "libero"} and not is_tera:
        # After using a move they gain STAB; treat as 1.5 if we cannot see the change.
        if base < 1.5:
            base = 1.5
    return float(base)


# ---------------------------------------------------------------------------
# Crit
# ---------------------------------------------------------------------------

def crit_stage(move: Any, user: Any) -> int:
    stage = 0
    if move is not None:
        ratio = getattr(move, "crit_ratio", None)
        if ratio is not None:
            try:
                stage += int(ratio)
            except (TypeError, ValueError):
                pass
    if user is not None:
        if _effect_active(user, "FOCUS_ENERGY", "LASER_FOCUS"):
            stage += 2
        if ability_id(user) == "superluck":
            stage += 1
        if item_id(user) in {"scopelens", "razorclaw"}:
            stage += 1
    return int(max(0, min(3, stage)))


def crit_probability(stage: int) -> float:
    return {0: 1.0 / 24.0, 1: 1.0 / 8.0, 2: 0.5, 3: 1.0}.get(int(stage), 1.0)


# ---------------------------------------------------------------------------
# Damage calculation (Gen 9)
# ---------------------------------------------------------------------------

@dataclass
class DamageRange:
    rolls: np.ndarray  # 16 integer rolls (85–100)
    min_frac: float
    max_frac: float
    ko_chance: float
    crit_ko_chance: float
    effectiveness: float
    stab: float
    expected_frac: float

    def as_features(self) -> List[float]:
        return [
            self.min_frac,
            self.max_frac,
            self.ko_chance,
            self.crit_ko_chance,
            self.effectiveness / 4.0,
            self.stab / 2.25,
            self.expected_frac,
        ]


def _chain_mod(value: int, modifier: float) -> int:
    """Showdown-style chained modifier (4096 units), rounded."""
    if modifier == 1.0:
        return value
    return int(math.floor(value * modifier + 0.5))


def gen9_damage(
    *,
    level: int,
    power: int,
    attack: float,
    defense: float,
    modifiers: Sequence[float],
) -> np.ndarray:
    """Return the 16 damage rolls for the Gen 5–9 core formula."""
    if power <= 0 or attack <= 0 or defense <= 0:
        return np.zeros(16, dtype=np.int32)
    base = math.floor(math.floor(math.floor(2 * level / 5 + 2) * power * attack / defense) / 50) + 2
    dmg = int(base)
    for mod in modifiers:
        dmg = _chain_mod(dmg, float(mod))
    rolls = np.array([math.floor(dmg * r / 100) for r in range(85, 101)], dtype=np.int32)
    rolls = np.maximum(rolls, 1)
    return rolls


def screens_mod(
    battle: Any,
    *,
    special: bool,
    crit: bool,
    attacker: Any,
    target_is_opponent: bool = True,
) -> float:
    if crit or ability_id(attacker) == "infiltrator":
        return 1.0
    side = (
        getattr(battle, "opponent_side_conditions", None)
        if target_is_opponent
        else getattr(battle, "side_conditions", None)
    ) or {}
    if _enum_in(side, "AURORA_VEIL"):
        return 0.5
    if special and _enum_in(side, "LIGHT_SCREEN"):
        return 0.5
    if (not special) and _enum_in(side, "REFLECT"):
        return 0.5
    return 1.0


def calculate_damage_range(
    user: Any,
    target: Any,
    move: Any,
    battle: Any,
    *,
    crit: bool = False,
    user_estimates: Optional[Mapping[str, Any]] = None,
    target_estimates: Optional[Mapping[str, Any]] = None,
    tera_override: Optional[str] = None,
) -> DamageRange:
    empty = DamageRange(
        rolls=np.zeros(16, dtype=np.int32),
        min_frac=0.0,
        max_frac=0.0,
        ko_chance=0.0,
        crit_ko_chance=0.0,
        effectiveness=1.0,
        stab=1.0,
        expected_frac=0.0,
    )
    if user is None or target is None or move is None:
        return empty
    cat = _norm(getattr(move, "category", None))
    if cat in {"status", "3"}:
        return empty
    power = int(getattr(move, "base_power", 0) or 0)
    if power <= 0:
        # Fixed-damage / OHKO / variable: encode a conservative placeholder.
        dmg_fixed = getattr(move, "damage", None)
        hp = float(getattr(target, "current_hp", 0) or getattr(target, "max_hp", 1) or 1)
        if dmg_fixed == "level" or _norm(dmg_fixed) == "level":
            lvl = int(getattr(user, "level", 50) or 50)
            frac = _safe_frac(lvl, hp)
            return DamageRange(
                rolls=np.full(16, lvl, dtype=np.int32),
                min_frac=frac,
                max_frac=frac,
                ko_chance=1.0 if lvl >= hp else 0.0,
                crit_ko_chance=1.0 if lvl >= hp else 0.0,
                effectiveness=1.0,
                stab=1.0,
                expected_frac=frac,
            )
        return empty

    mid = move_id(move)
    mtype = _type_name(getattr(move, "type", None))
    if tera_override and bool(getattr(user, "is_terastallized", False) or tera_override):
        # Offensive tera type for STAB / Tera Blast.
        if mid == "terablast":
            mtype = _type_name(tera_override)

    special = cat in {"special", "2"}
    if mid in {"psyshock", "psystrike", "secretsword"}:
        atk_special, def_special = True, False
    elif mid == "bodypress":
        atk_special, def_special = False, False
    elif mid == "foulplay":
        atk_special, def_special = False, False
    else:
        atk_special, def_special = special, special

    unaware_atk = ability_id(user) == "unaware"
    unaware_def = ability_id(target) == "unaware"
    weather = active_weather(battle)
    snow = weather in {"SNOW", "SNOWSCAPE"}
    sand = weather == "SANDSTORM"

    if mid == "bodypress":
        attack = modified_defense_stat(
            user, special=False, crit=crit, unaware_attacker=False, estimates=user_estimates
        )
    elif mid == "foulplay":
        attack = modified_attack_stat(
            target,
            special=False,
            crit=crit,
            unaware_defender=unaware_def,
            estimates=target_estimates,
        )
    else:
        attack = modified_attack_stat(
            user,
            special=atk_special,
            crit=crit,
            unaware_defender=unaware_def,
            estimates=user_estimates,
        )
    defense = modified_defense_stat(
        target,
        special=def_special,
        crit=crit,
        unaware_attacker=unaware_atk,
        snow=snow,
        sand=sand,
        estimates=target_estimates,
    )

    # Ruin abilities on the field.
    def _field_has(abil: str, which: str) -> bool:
        mons = []
        if which == "ours":
            mons.append(getattr(battle, "active_pokemon", None))
        else:
            mons.append(getattr(battle, "opponent_active_pokemon", None))
        return any(ability_id(m) == abil for m in mons if m is not None)

    if _field_has("swordofruin", "ours") or _field_has("swordofruin", "theirs"):
        if ability_id(target) != "swordofruin" and not def_special:
            defense *= 0.75
    if _field_has("beadsofruin", "ours") or _field_has("beadsofruin", "theirs"):
        if ability_id(target) != "beadsofruin" and def_special:
            defense *= 0.75
    if _field_has("tabletsofruin", "ours") or _field_has("tabletsofruin", "theirs"):
        if ability_id(user) != "tabletsofruin" and not atk_special:
            attack *= 0.75
    if _field_has("vesselofruin", "ours") or _field_has("vesselofruin", "theirs"):
        if ability_id(user) != "vesselofruin" and atk_special:
            attack *= 0.75

    # Technician / Supreme Overlord / flash-fire style power tweaks.
    if ability_id(user) == "technician" and power <= 60:
        power = int(power * 1.5)
    fallen = 0
    if ability_id(user) == "supremeoverlord":
        team = getattr(battle, "team", None) or {}
        fallen = sum(1 for p in team.values() if getattr(p, "fainted", False))
        power = int(power * (1.0 + 0.1 * min(5, fallen)))

    tera_type = _type_name(getattr(target, "tera_type", None))
    is_tera = bool(getattr(target, "is_terastallized", False))
    effectiveness = type_effectiveness(
        mtype,
        pokemon_types(target, ignore_tera=True),
        tera_type=tera_type,
        is_terastallized=is_tera,
        move_id=mid,
    )
    if weather == "DELTASTREAM" and "FLYING" in pokemon_types(target) and effectiveness == 2.0:
        if mtype in {"ELECTRIC", "ICE", "ROCK"}:
            effectiveness = 1.0
    if effectiveness == 0.0:
        return DamageRange(
            rolls=np.zeros(16, dtype=np.int32),
            min_frac=0.0,
            max_frac=0.0,
            ko_chance=0.0,
            crit_ko_chance=0.0,
            effectiveness=0.0,
            stab=1.0,
            expected_frac=0.0,
        )

    stab = stab_multiplier(mtype, user)
    weather_mod = weather_power_mod(mtype, weather)
    terrain = active_terrain(battle)
    terrain_mod = terrain_power_mod(mtype, terrain, is_grounded(user, battle))
    burn = 1.0
    if status_name(user) == "BRN" and not special and ability_id(user) != "guts" and mid != "facade":
        burn = 0.5
    screen = screens_mod(battle, special=special, crit=crit, attacker=user, target_is_opponent=True)
    crit_mod = 1.5 if crit else 1.0
    if crit and ability_id(user) == "sniper":
        crit_mod = 2.25

    other = 1.0
    item = item_id(user)
    if item == "lifeorb":
        other *= 1.3
    if item == "expertbelt" and effectiveness > 1.0:
        other *= 1.2
    if item in {"wellspringmask", "hearthflamemask", "cornerstonemask"}:
        other *= 1.2
    if ability_id(user) == "tintedlens" and effectiveness < 1.0:
        other *= 2.0
    if ability_id(target) in {"filter", "solidrock", "prismarmor"} and effectiveness > 1.0:
        other *= 0.75
    if ability_id(target) in {"multiscale", "shadowshield"}:
        if float(getattr(target, "current_hp_fraction", 1.0) or 1.0) >= 1.0:
            other *= 0.5
    if ability_id(target) == "fluffy" and not special:
        other *= 0.5
        if mtype == "FIRE":
            other *= 2.0
    tgt_abil = ability_id(target)
    if tgt_abil == "wellbakedbody" and mtype == "FIRE":
        return empty
    if tgt_abil == "flashfire" and mtype == "FIRE":
        return empty
    if tgt_abil == "dryskin" and mtype == "WATER":
        return empty
    if tgt_abil == "dryskin" and mtype == "FIRE":
        other *= 1.25
    if tgt_abil in {"waterabsorb", "stormdrain"} and mtype == "WATER":
        return empty
    if tgt_abil in {"voltabsorb", "lightningrod", "motordrive"} and mtype == "ELECTRIC":
        return empty
    if ability_id(target) == "eartheater" and mtype == "GROUND":
        return empty
    if ability_id(target) == "levitate" and mtype == "GROUND" and not is_grounded(target, battle):
        return empty
    if ability_id(target) == "thickfat" and mtype in {"FIRE", "ICE"}:
        other *= 0.5
    if ability_id(target) == "purifyingsalt" and mtype == "GHOST":
        other *= 0.5
    if ability_id(user) == "sheerforce" and getattr(move, "secondary", None):
        other *= 1.3
    hp_frac = float(getattr(user, "current_hp_fraction", 1.0) or 1.0)
    if ability_id(user) in {"blaze", "overgrow", "torrent", "swarm"} and hp_frac <= 1.0 / 3.0:
        boosted = {
            "blaze": "FIRE",
            "overgrow": "GRASS",
            "torrent": "WATER",
            "swarm": "BUG",
        }[ability_id(user)]
        if mtype == boosted:
            other *= 1.5

    modifiers = [weather_mod, terrain_mod, crit_mod, stab, effectiveness, burn, screen, other]
    level = int(getattr(user, "level", 50) or 50)
    rolls = gen9_damage(
        level=level,
        power=power,
        attack=max(1.0, attack),
        defense=max(1.0, defense),
        modifiers=modifiers,
    )
    hits = float(getattr(move, "expected_hits", 1.0) or 1.0)
    if hits > 1:
        rolls = (rolls.astype(np.float64) * hits).astype(np.int32)

    hp = float(getattr(target, "current_hp", 0) or 0)
    max_hp = float(getattr(target, "max_hp", 0) or 0)
    # Opponent HP is often 0–100 percent.
    denom = hp if hp > 0 else max_hp
    if denom <= 0:
        denom = 1.0
    # If current_hp looks like a percentage, treat rolls relative to 100.
    looks_percent = max_hp <= 100 and hp <= 100
    if looks_percent and getattr(target, "current_hp_fraction", None) is not None:
        # Approximate real HP from base HP at level 50.
        est = stat_dict(target, target_estimates)["hp"] or 200.0
        real_hp = est * float(target.current_hp_fraction)
        denom = max(1.0, real_hp)
    min_d, max_d = int(rolls.min()), int(rolls.max())
    min_frac = _clip01(min_d / denom)
    max_frac = _clip01(max_d / denom)
    ko = float(np.mean(rolls >= denom))
    expected = float(np.mean(rolls) / denom)

    crit_range = (
        calculate_damage_range(
            user,
            target,
            move,
            battle,
            crit=True,
            user_estimates=user_estimates,
            target_estimates=target_estimates,
            tera_override=tera_override,
        )
        if not crit
        else None
    )
    crit_ko = crit_range.ko_chance if crit_range is not None else ko
    return DamageRange(
        rolls=rolls,
        min_frac=min_frac,
        max_frac=max_frac,
        ko_chance=ko,
        crit_ko_chance=crit_ko,
        effectiveness=effectiveness,
        stab=stab,
        expected_frac=expected,
    )


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

def _pad(vec: np.ndarray, size: int) -> np.ndarray:
    out = np.zeros(size, dtype=np.float32)
    n = min(size, int(vec.size))
    if n:
        out[:n] = vec.astype(np.float32, copy=False).ravel()[:n]
    return out


def encode_move(
    move: Any,
    user: Any,
    target: Any,
    battle: Any,
    *,
    user_estimates: Optional[Mapping[str, Any]] = None,
    target_estimates: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    feats: List[float] = []
    if move is None:
        return np.zeros(MOVE_FEATURE_DIM, dtype=np.float32)
    mtype = _type_name(getattr(move, "type", None))
    feats.extend(_types_one_hot([mtype]).tolist())
    cat = _norm(getattr(move, "category", None))
    feats.extend(
        [
            1.0 if cat in {"physical", "1"} else 0.0,
            1.0 if cat in {"special", "2"} else 0.0,
            1.0 if cat in {"status", "3"} else 0.0,
        ]
    )
    power = float(getattr(move, "base_power", 0) or 0) / 250.0
    acc = float(getattr(move, "accuracy", 1.0) or 0.0)
    if acc > 1.5:
        acc = acc / 100.0
    prio = move_priority(move, terrain=active_terrain(battle), user=user)
    pp = _safe_frac(float(getattr(move, "current_pp", 1) or 0), float(getattr(move, "max_pp", 1) or 1))
    rng = calculate_damage_range(
        user, target, move, battle, user_estimates=user_estimates, target_estimates=target_estimates
    )
    mid = move_id(move)
    feats.extend(
        [
            _clip01(power),
            _clip01(acc),
            (prio + 7) / 12.0,
            _clip01(pp),
            rng.stab / 2.25,
            rng.effectiveness / 4.0,
            rng.min_frac,
            rng.max_frac,
            rng.ko_chance,
            rng.crit_ko_chance,
            crit_probability(crit_stage(move, user)),
            1.0 if mid in HAZARD_SETTING_MOVES else 0.0,
            1.0 if mid in HAZARD_REMOVAL_MOVES else 0.0,
            1.0 if cat in {"status", "3"} else 0.0,
            1.0 if mid in SETUP_MOVES else 0.0,
            1.0 if mid in PROTECT_MOVES else 0.0,
            1.0 if mid in RECOVERY_MOVES else 0.0,
            1.0 if mid in PIVOT_MOVES else 0.0,
        ]
    )
    return _pad(np.asarray(feats, dtype=np.float32), MOVE_FEATURE_DIM)


def encode_active(
    mon: Any,
    foe: Any,
    battle: Any,
    *,
    is_player: bool,
    estimates: Optional[Mapping[str, Any]] = None,
    foe_estimates: Optional[Mapping[str, Any]] = None,
    available_moves: Optional[Sequence[Any]] = None,
) -> np.ndarray:
    feats: List[float] = []
    if mon is None:
        return np.zeros(ACTIVE_FEATURE_DIM, dtype=np.float32)
    hp_frac = float(getattr(mon, "current_hp_fraction", 0.0) or 0.0)
    feats.append(_clip01(hp_frac))
    types = pokemon_types(mon, ignore_tera=True)
    tera = _type_name(getattr(mon, "tera_type", None))
    is_tera = float(bool(getattr(mon, "is_terastallized", False)))
    feats.extend(_types_one_hot(types).tolist())
    feats.extend(_types_one_hot([tera] if tera else []).tolist())
    feats.append(is_tera)
    feats.extend(_types_one_hot(pokemon_types(mon, ignore_tera=False)).tolist())

    stats = stat_dict(mon, estimates)
    for key in ("atk", "def", "spa", "spd", "spe", "hp"):
        feats.append(_clip01(stats[key] / 500.0))

    boosts = boosts_of(mon)
    for name in BOOST_STATS:
        acc = name in {"accuracy", "evasion"}
        feats.append(stage_multiplier(boosts[name], accuracy_evasion=acc) / 4.0)

    st = status_name(mon)
    feats.extend([1.0 if st == s else 0.0 for s in MAJOR_STATUSES])
    counter = float(getattr(mon, "status_counter", 0) or 0)
    feats.append(_clip01(counter / 8.0))

    for name in MINOR_EFFECTS:
        feats.append(1.0 if _effect_active(mon, name) else 0.0)

    abil = ability_id(mon)
    abil_vec = np.zeros(len(COMPETITIVE_ABILITIES), dtype=np.float32)
    if abil in ABILITY_INDEX:
        abil_vec[ABILITY_INDEX[abil]] = 1.0
    feats.extend(abil_vec.tolist())

    it = item_id(mon)
    item_vec = np.zeros(len(COMPETITIVE_ITEMS), dtype=np.float32)
    if it in ITEM_INDEX:
        item_vec[ITEM_INDEX[it]] = 1.0
    feats.extend(item_vec.tolist())

    locked = _effect_active(mon, "LOCKED_MOVE") or it in {"choiceband", "choicespecs", "choicescarf"}
    last = move_id(getattr(mon, "last_move", None))
    feats.append(1.0 if locked else 0.0)
    moves = list(available_moves) if available_moves is not None else list(
        (getattr(mon, "moves", None) or {}).values()
    )
    while len(moves) < N_MOVES:
        moves.append(None)
    moves = moves[:N_MOVES]
    for mv in moves:
        feats.append(1.0 if locked and last and move_id(mv) == last else 0.0)

    feats.extend(
        [
            1.0 if bool(getattr(mon, "first_turn", False)) else 0.0,
            1.0 if bool(getattr(mon, "must_recharge", False)) else 0.0,
            _clip01(float(getattr(mon, "protect_counter", 0) or 0) / 4.0),
            1.0 if bool(getattr(mon, "revealed", True)) else 0.0,
            1.0 if bool(getattr(mon, "fainted", False)) else 0.0,
            1.0 if is_grounded(mon, battle) else 0.0,
            1.0 if has_heavy_duty_boots(mon) else 0.0,
            1.0 if it == "focussash" and hp_frac >= 1.0 else 0.0,
            1.0 if it == "lifeorb" else 0.0,
            1.0 if it == "leftovers" or (it == "blacksludge" and "POISON" in types) else 0.0,
            1.0 if it == "boosterenergy" else 0.0,
            1.0 if abil == "regenerator" else 0.0,
        ]
    )

    for mv in moves:
        feats.extend(
            encode_move(
                mv,
                mon,
                foe,
                battle,
                user_estimates=estimates,
                target_estimates=foe_estimates,
            ).tolist()
        )
    return _pad(np.asarray(feats, dtype=np.float32), ACTIVE_FEATURE_DIM)


def encode_slot(
    mon: Any,
    battle: Any,
    *,
    is_player: bool,
    estimates: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    feats: List[float] = []
    present = mon is not None
    feats.append(1.0 if present else 0.0)
    if not present:
        return _pad(np.zeros(1, dtype=np.float32), SLOT_FEATURE_DIM)
    feats.append(1.0 if bool(getattr(mon, "fainted", False)) else 0.0)
    feats.append(_clip01(float(getattr(mon, "current_hp_fraction", 0.0) or 0.0)))
    feats.extend(_types_one_hot(pokemon_types(mon, ignore_tera=True)).tolist())
    tera = _type_name(getattr(mon, "tera_type", None))
    feats.extend(_types_one_hot([tera] if tera else []).tolist())
    feats.append(1.0 if bool(getattr(mon, "revealed", is_player)) else 0.0)
    feats.append(1.0 if has_heavy_duty_boots(mon) else 0.0)
    feats.append(1.0 if ability_id(mon) == "regenerator" else 0.0)
    side = (
        getattr(battle, "opponent_side_conditions", None)
        if is_player
        else getattr(battle, "side_conditions", None)
    ) or {}
    # Hazard damage this pokemon would take switching *into the opponent's hazards* wait:
    # Player switching in takes player-side hazards.
    entry_side = (
        getattr(battle, "side_conditions", None)
        if is_player
        else getattr(battle, "opponent_side_conditions", None)
    ) or {}
    feats.append(_clip01(switch_in_hazard_fraction(mon, entry_side, battle)))
    stats = stat_dict(mon, estimates)
    feats.append(_clip01(stats["spe"] / 500.0))
    feats.append(_clip01(stats["atk"] / 500.0))
    feats.append(_clip01(stats["spa"] / 500.0))
    st = status_name(mon)
    feats.extend([1.0 if st == s else 0.0 for s in MAJOR_STATUSES])
    feats.append(1.0 if bool(getattr(mon, "active", False)) else 0.0)
    feats.append(1.0 if _enum_in(entry_side, "TOXIC_SPIKES") and is_grounded(mon, battle) else 0.0)
    feats.append(1.0 if sticky_web_applies(mon, battle) and _enum_in(entry_side, "STICKY_WEB") else 0.0)
    return _pad(np.asarray(feats, dtype=np.float32), SLOT_FEATURE_DIM)


def encode_field(battle: Any) -> np.ndarray:
    feats: List[float] = []
    weather = active_weather(battle)
    wmap = {
        "SUNNYDAY": 0,
        "RAINDANCE": 1,
        "SANDSTORM": 2,
        "SNOW": 3,
        "DESOLATELAND": 4,
        "PRIMORDIALSEA": 5,
        "DELTASTREAM": 6,
        "HAIL": 7,
    }
    wvec = np.zeros(8, dtype=np.float32)
    if weather in wmap:
        wvec[wmap[weather]] = 1.0
    elif weather == "SNOWSCAPE":
        wvec[3] = 1.0
    feats.extend(wvec.tolist())
    wdict = getattr(battle, "weather", None) or {}
    start = 0
    if wdict:
        start = list(wdict.values())[0] or 0
    turn = int(getattr(battle, "turn", 0) or 0)
    feats.append(_clip01((turn - int(start)) / 8.0) if weather else 0.0)

    terrain = active_terrain(battle)
    tvec = np.zeros(4, dtype=np.float32)
    if terrain in TERRAINS:
        tvec[list(TERRAINS).index(terrain)] = 1.0
    feats.extend(tvec.tolist())
    fields = getattr(battle, "fields", None) or {}
    t_start = _enum_value(fields, terrain) if terrain else 0
    feats.append(_clip01((turn - t_start) / 8.0) if terrain else 0.0)

    for name in (
        "TRICK_ROOM",
        "GRAVITY",
        "MAGIC_ROOM",
        "WONDER_ROOM",
        "FAIRY_LOCK",
        "NEUTRALIZING_GAS",
    ):
        feats.append(1.0 if _enum_in(fields, name) else 0.0)

    ours = getattr(battle, "side_conditions", None) or {}
    theirs = getattr(battle, "opponent_side_conditions", None) or {}
    for side in (ours, theirs):
        feats.append(1.0 if _enum_in(side, "TAILWIND") else 0.0)
        feats.append(1.0 if _enum_in(side, "REFLECT") else 0.0)
        feats.append(1.0 if _enum_in(side, "LIGHT_SCREEN") else 0.0)
        feats.append(1.0 if _enum_in(side, "AURORA_VEIL") else 0.0)
        feats.append(1.0 if _enum_in(side, "SAFEGUARD") else 0.0)
        feats.append(1.0 if _enum_in(side, "STEALTH_ROCK") else 0.0)
        feats.append(_clip01(_enum_value(side, "SPIKES") / 3.0))
        feats.append(_clip01(_enum_value(side, "TOXIC_SPIKES") / 2.0))
        feats.append(1.0 if _enum_in(side, "STICKY_WEB") else 0.0)
        feats.append(1.0 if _enum_in(side, "G_MAX_STEELSURGE", "GMAXSTEELSURGE") else 0.0)
    return _pad(np.asarray(feats, dtype=np.float32), FIELD_FEATURE_DIM)


def legal_action_mask(battle: Any) -> np.ndarray:
    """13-d mask: moves 0-3, tera+moves 4-7, switches 8-12."""
    mask = np.zeros(N_MOTOR_CHANNELS, dtype=np.float32)
    if battle is None:
        return mask
    moves = list(getattr(battle, "available_moves", None) or [])
    for i in range(min(N_MOVES, len(moves))):
        mask[i] = 1.0
        if bool(getattr(battle, "can_tera", False)):
            mask[4 + i] = 1.0
    if not bool(getattr(battle, "trapped", False)) or bool(getattr(battle, "force_switch", False)):
        switches = list(getattr(battle, "available_switches", None) or [])
        for i in range(min(N_SWITCHES, len(switches))):
            mask[8 + i] = 1.0
    if bool(getattr(battle, "force_switch", False)):
        mask[:8] = 0.0
    return mask


def encode_global(
    battle: Any,
    *,
    our_speed: float,
    opp_speed: float,
    outspeed: float,
) -> np.ndarray:
    feats: List[float] = []
    turn = int(getattr(battle, "turn", 0) or 0)
    feats.append(_clip01(turn / 50.0))
    feats.append(1.0 if bool(getattr(battle, "can_tera", False)) else 0.0)
    feats.append(1.0 if bool(getattr(battle, "used_tera", False)) else 0.0)
    feats.append(1.0 if bool(getattr(battle, "opponent_used_tera", False)) else 0.0)
    feats.append(1.0 if bool(getattr(battle, "trapped", False)) else 0.0)
    feats.append(1.0 if bool(getattr(battle, "maybe_trapped", False)) else 0.0)
    feats.append(1.0 if bool(getattr(battle, "force_switch", False)) else 0.0)

    def _alive_hp(team: Mapping[Any, Any]) -> Tuple[int, float]:
        alive = 0
        hp = 0.0
        for mon in (team or {}).values():
            if mon is None or getattr(mon, "fainted", False):
                continue
            alive += 1
            hp += float(getattr(mon, "current_hp_fraction", 0.0) or 0.0)
        return alive, hp

    our_n, our_hp = _alive_hp(getattr(battle, "team", None) or {})
    opp_n, opp_hp = _alive_hp(getattr(battle, "opponent_team", None) or {})
    feats.append(our_n / 6.0)
    feats.append(opp_n / 6.0)
    feats.append(_clip01(our_hp / 6.0))
    feats.append(_clip01(opp_hp / 6.0))
    feats.append(_clip01(our_speed / 800.0))
    feats.append(_clip01(opp_speed / 800.0))
    feats.append((outspeed + 1.0) / 2.0)
    feats.append(1.0 if outspeed == 0.0 and our_speed > 0 else 0.0)
    feats.append(1.0 if trick_room_active(battle) else 0.0)
    feats.extend(legal_action_mask(battle).tolist())
    return _pad(np.asarray(feats, dtype=np.float32), GLOBAL_FEATURE_DIM)


def _team_list(team: Mapping[Any, Any], active: Any, n: int = 6) -> List[Any]:
    mons = list((team or {}).values())
    ordered: List[Any] = []
    if active is not None:
        ordered.append(active)
    for m in mons:
        if m is active:
            continue
        ordered.append(m)
    while len(ordered) < n:
        ordered.append(None)
    return ordered[:n]


@dataclass
class VectorizedState:
    vector: np.ndarray
    our_speed: float
    opp_speed: float
    outspeed: float
    legal_mask: np.ndarray
    damage_by_move: List[DamageRange] = field(default_factory=list)


class BattleStateEngine:
    """Full Showdown mechanics vectorizer for a singles battle."""

    dim: int = STATE_DIM

    def vectorize(
        self,
        battle: Any,
        *,
        our_estimates: Optional[Mapping[str, Any]] = None,
        opp_estimates: Optional[Mapping[str, Any]] = None,
        opp_set_estimates: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> np.ndarray:
        return self.inspect(
            battle,
            our_estimates=our_estimates,
            opp_estimates=opp_estimates,
            opp_set_estimates=opp_set_estimates,
        ).vector

    def inspect(
        self,
        battle: Any,
        *,
        our_estimates: Optional[Mapping[str, Any]] = None,
        opp_estimates: Optional[Mapping[str, Any]] = None,
        opp_set_estimates: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> VectorizedState:
        us = getattr(battle, "active_pokemon", None)
        them = getattr(battle, "opponent_active_pokemon", None)
        our_speed, opp_speed, outspeed = speed_order(
            us, them, battle, our_estimates=our_estimates, opp_estimates=opp_estimates
        )
        parts = [
            encode_active(
                us,
                them,
                battle,
                is_player=True,
                estimates=our_estimates,
                foe_estimates=opp_estimates,
                available_moves=getattr(battle, "available_moves", None),
            ),
            encode_active(
                them,
                us,
                battle,
                is_player=False,
                estimates=opp_estimates,
                foe_estimates=our_estimates,
                available_moves=list((getattr(them, "moves", None) or {}).values())
                if them is not None
                else None,
            ),
        ]
        our_team = _team_list(getattr(battle, "team", None) or {}, us, 6)
        opp_team = _team_list(getattr(battle, "opponent_team", None) or {}, them, 6)
        for mon in our_team:
            parts.append(encode_slot(mon, battle, is_player=True))
        for mon in opp_team:
            key = _norm(getattr(mon, "species", None) if mon is not None else "")
            est = None
            if opp_set_estimates and key in opp_set_estimates:
                est = opp_set_estimates[key]
            elif mon is them:
                est = opp_estimates
            parts.append(encode_slot(mon, battle, is_player=False, estimates=est))
        parts.append(encode_field(battle))
        parts.append(encode_global(battle, our_speed=our_speed, opp_speed=opp_speed, outspeed=outspeed))
        vector = np.concatenate([p.astype(np.float32, copy=False).ravel() for p in parts])
        vector = _pad(vector, STATE_DIM)
        damages: List[DamageRange] = []
        for mv in list(getattr(battle, "available_moves", None) or [])[:N_MOVES]:
            damages.append(
                calculate_damage_range(
                    us, them, mv, battle, user_estimates=our_estimates, target_estimates=opp_estimates
                )
            )
        return VectorizedState(
            vector=vector,
            our_speed=our_speed,
            opp_speed=opp_speed,
            outspeed=outspeed,
            legal_mask=legal_action_mask(battle),
            damage_by_move=damages,
        )


# ---------------------------------------------------------------------------
# Residual / residual-style helpers used by neuromodulation
# ---------------------------------------------------------------------------

def residual_chip_fraction(mon: Any, battle: Any, *, is_player: bool) -> float:
    """Expected fraction of HP lost to residuals this turn (status + weather)."""
    if mon is None or getattr(mon, "fainted", False):
        return 0.0
    if ability_id(mon) in {"magicguard"}:
        return 0.0
    chip = 0.0
    st = status_name(mon)
    if st == "BRN":
        chip += 1.0 / 16.0
    elif st == "PSN":
        chip += 1.0 / 8.0
    elif st == "TOX":
        n = max(1, int(getattr(mon, "status_counter", 1) or 1))
        chip += n / 16.0
    weather = active_weather(battle)
    types = pokemon_types(mon)
    if weather == "SANDSTORM":
        if not any(t in types for t in ("ROCK", "GROUND", "STEEL")):
            if ability_id(mon) not in {"overcoat", "sandforce", "sandrush", "sandveil"}:
                if item_id(mon) != "safetygoggles":
                    chip += 1.0 / 16.0
    if _effect_active(mon, "SALT_CURE"):
        chip += 0.25 if ("WATER" in types or "STEEL" in types) else 0.125
    if _effect_active(mon, "LEECH_SEED"):
        chip += 1.0 / 8.0
    if item_id(mon) == "lifeorb" and ability_id(mon) != "sheerforce":
        chip += 0.1
    if item_id(mon) in {"leftovers"} or (item_id(mon) == "blacksludge" and "POISON" in types):
        chip -= 1.0 / 16.0
    if active_terrain(battle) == "GRASSY_TERRAIN" and is_grounded(mon, battle):
        chip -= 1.0 / 16.0
    return float(chip)


def hazard_advantage(battle: Any) -> float:
    """Positive when we have more residual pressure from hazards than they do."""
    ours = getattr(battle, "side_conditions", None) or {}
    theirs = getattr(battle, "opponent_side_conditions", None) or {}

    def score(side: Mapping[Any, Any]) -> float:
        s = 0.0
        if _enum_in(side, "STEALTH_ROCK"):
            s += 0.4
        s += 0.15 * _enum_value(side, "SPIKES")
        s += 0.15 * _enum_value(side, "TOXIC_SPIKES")
        if _enum_in(side, "STICKY_WEB"):
            s += 0.2
        return s

    return float(score(theirs) - score(ours))


# ---------------------------------------------------------------------------
# Self-check against the Gen 6+ type chart
# ---------------------------------------------------------------------------

def _self_check_type_chart() -> None:
    cases = [
        ("NORMAL", ["ROCK"], 0.5),
        ("NORMAL", ["GHOST"], 0.0),
        ("NORMAL", ["STEEL"], 0.5),
        ("FIRE", ["GRASS"], 2.0),
        ("FIRE", ["WATER"], 0.5),
        ("FIRE", ["ROCK"], 0.5),
        ("FIRE", ["ICE"], 2.0),
        ("FIRE", ["STEEL"], 2.0),
        ("WATER", ["FIRE"], 2.0),
        ("ELECTRIC", ["GROUND"], 0.0),
        ("ELECTRIC", ["FLYING"], 2.0),
        ("GRASS", ["WATER"], 2.0),
        ("ICE", ["DRAGON"], 2.0),
        ("FIGHTING", ["NORMAL"], 2.0),
        ("FIGHTING", ["GHOST"], 0.0),
        ("FIGHTING", ["FAIRY"], 0.5),
        ("POISON", ["STEEL"], 0.0),
        ("POISON", ["FAIRY"], 2.0),
        ("GROUND", ["FLYING"], 0.0),
        ("GROUND", ["ELECTRIC"], 2.0),
        ("PSYCHIC", ["DARK"], 0.0),
        ("GHOST", ["NORMAL"], 0.0),
        ("GHOST", ["PSYCHIC"], 2.0),
        ("DRAGON", ["FAIRY"], 0.0),
        ("DRAGON", ["DRAGON"], 2.0),
        ("DARK", ["PSYCHIC"], 2.0),
        ("STEEL", ["FAIRY"], 2.0),
        ("FAIRY", ["DRAGON"], 2.0),
        ("FAIRY", ["STEEL"], 0.5),
        ("FIRE", ["GRASS", "ICE"], 4.0),
        ("ROCK", ["FIRE", "FLYING"], 4.0),
        ("GROUND", ["FIRE", "FLYING"], 0.0),
    ]
    for atk, defs, expected in cases:
        got = type_effectiveness(atk, defs)
        assert got == expected, f"{atk} vs {defs}: expected {expected}, got {got}"
    assert abs(stage_multiplier(2) - 2.0) < 1e-9
    assert abs(stage_multiplier(-1) - (2.0 / 3.0)) < 1e-9
    assert abs(stage_multiplier(6) - 4.0) < 1e-9
    assert abs(stage_multiplier(-6) - 0.25) < 1e-9
    rolls = gen9_damage(level=50, power=100, attack=150, defense=100, modifiers=[1.5, 2.0])
    assert rolls.shape == (16,)
    assert rolls[0] < rolls[-1]


if __name__ == "__main__":
    _self_check_type_chart()
    print(f"state_engine OK  STATE_DIM={STATE_DIM}  types={N_TYPES}")
