# MaleCNS Connectome Agent for Pokémon Showdown

Bio-inspired competitive agent for **Gen 9 OU / Standard Singles**. A sparse Kenyon-cell code of the full battle state is read out by mushroom-body output neurons (approach vs avoidance), oriented by a central-complex ring attractor, and decoded by ventral-nerve-cord motor channels into `poke-env` orders.

The wiring is loaded from the open MaleCNS Drosophila connectome (`male-cns:v1.0`, ~166k neurons) when a neuPrint token is present, and otherwise from a Drosophila-statistic synthetic subgraph so the bot still plays.

## Repository layout

```
.
├── state_engine.py          # Gen 9 mechanics vectorizer + damage range engine
├── memory_system.py         # Smogon priors, opponent set inference, turn buffer
├── connectome_bridge.py     # MaleCNS / synthetic SNN (KC, MBON, CX, VNC)
├── neuromodulation.py       # DAN reward-prediction error + STDP, 5-HT risk gain
├── run_agent.py             # poke-env Player + CLI
├── run_bot.sh               # Linux / macOS launcher
├── run_bot.bat              # Windows launcher
├── requirements.txt
├── .env.example
├── data/
│   ├── smogon_sets.json     # Meta set priors for opponent inference
│   ├── connectome/          # Cached MaleCNS subgraph (.npz)
│   └── weights/             # Plastic KC→MBON weights after STDP
└── teams/
    └── gen9ou_sample.txt    # Example Gen 9 OU paste
```

## Turn loop

1. Parse the `poke-env` `Battle` into a fixed **1652-d** normalized vector (`state_engine.py`).
2. Update opponent probabilistic set profiles from revealed moves / items / abilities (`memory_system.py`).
3. Run ≤ **50 ms** of LIF propagation through KC → MBON → CX → VNC (`connectome_bridge.py`).
4. Compute dopaminergic RPE and apply three-factor STDP; set serotonergic risk \(S \in [0,1]\) (`neuromodulation.py`).
5. Mask illegal actions and emit a `BattleOrder` (`run_agent.py`).

VNC motor map (13 channels, covering the singles action space described in the spec):

| Channels | Order |
| --- | --- |
| 0–3 | Attack slots 1–4 |
| 4–7 | Terastallize + Attack 1–4 |
| 8–12 | Switch 1–5 |

A compact **10-channel** readout is also produced (Attack 1–4, Switch 1–5, Tera intent).

Serotonin: \(S > 0.6\) favors hazards, recovery, and safe switches; \(S < 0.3\) favors high-power / low-accuracy clicks, predictions, and offensive Tera.

## Mechanics coverage (`state_engine.py`)

- Dual-type efficacy including 0× / 0.25× / 0.5× / 1× / 2× / 4×, Tera defensive shift, Stellar / Terastarstorm, Adaptability (2.25), Gen 9 Tera STAB.
- Stat stages −6…+6 with exact multipliers (`+2 = 2.0×`, `−1 = 2/3`), Unaware, crit stage overrides, paralysis speed halving.
- Major status (Burn, Paralysis, Toxic counter, Poison, Sleep counter, Freeze) and minor volatiles (Taunt, Encore, Disable, Confusion, Leech Seed, Substitute, Salt Cure, Infatuation, …).
- Weather (Sun / Rain / Sand / Snow + primitives), terrain (Electric / Grassy / Psychic / Misty), Trick Room, Tailwind.
- Stealth Rock (type-scaled), Spikes 1–3, Toxic Spikes 1–2, Sticky Web, Heavy-Duty Boots, Rapid Spin / Defog / Tidy Up / Mortal Spin.
- Choice lock, Focus Sash, Life Orb, Leftovers, Booster Energy, Intimidate / Regenerator / Magic Bounce / Supreme Overlord / Sturdy / weather setters, Tera used/available.
- Priority brackets −7…+5, speed order with Scarf, Swift Swim / Chlorophyll / Sand Rush / Slush Rush, stages, paralysis, Tailwind, Trick Room.
- Gen 9 damage formula with 16 rolls (85–100), STAB, type, crit (1.5×, Sniper 2.25×), screens, burn, items, abilities.

## Setup

```bash
cp .env.example .env
# Edit .env: SHOWDOWN_USERNAME, SHOWDOWN_PASSWORD, NEUPRINT_APPLICATION_TOKEN
```

neuPrint token: sign in at [neuprint.janelia.org](https://neuprint.janelia.org) → account menu → auth token. Dataset: `male-cns:v1.0`. If the token is missing or rejected, the agent logs a warning and uses the synthetic connectome.

### Linux / macOS

```bash
chmod +x run_bot.sh
./run_bot.sh --mode ladder --format gen9ou
./run_bot.sh --mode challenge --challenge-user YOUR_NAME --format gen9ou --n-battles 1
./run_bot.sh --mode accept --format gen9ou
```

The script creates `.venv` if needed, installs `requirements.txt`, sources `.env`, and forwards all arguments to `python run_agent.py`.

### Windows

```bat
run_bot.bat --mode ladder --format gen9ou
```

### Direct Python

```bash
python run_agent.py --mode selfcheck
python run_agent.py --username BOT --password SECRET --mode ladder --format gen9ou --team teams/gen9ou_sample.txt
```

## Local evaluation

`--mode local_eval` (or `--local`) talks to a Pokémon Showdown server at `ws://localhost:8000/showdown/websocket`. Start a local server first, then:

```bash
./run_bot.sh --local --mode local_eval --format gen9ou --n-battles 5
```

The heuristic baseline from `poke-env` is used as the opponent.

## CLI

| Flag | Default | Meaning |
| --- | --- | --- |
| `--username` / `--password` | `.env` | Showdown account |
| `--format` | `gen9ou` | Battle format |
| `--mode` | `ladder` | `ladder`, `challenge`, `accept`, `local_eval`, `selfcheck` |
| `--team` | `teams/gen9ou_sample.txt` | Showdown paste |
| `--local` | off | Localhost PS server |
| `--challenge-user` | — | Target for `--mode challenge` |
| `--n-battles` | `1` | Games to play |

## Connectome notes

The full ~166k-neuron MaleCNS graph is not simulated at 50 ms. The bridge pulls a **task-relevant subgraph** (Kenyon cells, MBONs, ellipsoid / fan-shaped-body types, descending neurons) and embeds it in a fixed-size SNN (2048 KC, 34 MBON, 16 CX, 13 VNC). Plastic KC→MBON weights are stored under `data/weights/kc_mbon.npy`.
