# MaleCNS Connectome Agent for Pokémon Showdown

Bio-inspired competitive agent for **Gen 9 OU / Standard Singles**. A sparse Kenyon-cell code of the battle state is read out by mushroom-body output neurons, oriented by a central-complex ring attractor, and decoded into `poke-env` orders. Move choice is matchup-first (type effectiveness, damage, KO chance, switches).

**Offline by default.** Installing Python packages once is the only setup that needs the internet. After that, running the bot does not contact Pokémon Showdown, neuPrint, or anything else unless you start it with `--mode ladder`, `--mode challenge`, or `--mode accept`. Those three modes log into [play.pokemonshowdown.com](https://play.pokemonshowdown.com).

## 1. Pokémon Showdown username and password

The public server does not accept guests for a bot. You need a **registered** account.

1. Open [https://play.pokemonshowdown.com](https://play.pokemonshowdown.com).
2. Click **Choose name** in the top-right.
3. Type the name the bot should appear as.
4. Click **Register** and set a password (confirm the email if Showdown asks).
5. That registered name is the username. The password you just set is the password.

Use a dedicated bot account if you also play on a personal name. The name you register is what opponents will see.

## 2. Create and edit `.env`

Credentials live in a file named `.env` in this folder (same place as `run_agent.py`). It is not committed to git.

**Windows (Command Prompt, in this folder):**

```bat
copy .env.example .env
notepad .env
```

**Windows (PowerShell):**

```powershell
Copy-Item .env.example .env
notepad .env
```

**Linux / macOS:**

```bash
cp .env.example .env
nano .env
```

Set only these two lines. No quotes, no spaces around `=`:

```
SHOWDOWN_USERNAME=YourBotName
SHOWDOWN_PASSWORD=yourpassword
```

Leave everything else blank. You do **not** need a neuPrint token.

Save the file. You can also skip `.env` and pass `--username YourBotName --password yourpassword` when you start the bot.

## 3. Run it

First-time launch may install packages into `.venv` (needs internet once). After that, a bare start stays **offline** and only self-checks the type chart, matchup scorer, and synthetic connectome.

**Windows:**

```bat
run_bot.bat
```

**Linux / macOS:**

```bash
chmod +x run_bot.sh
./run_bot.sh
```

You should see `selfcheck OK`. No Showdown login happens.

### Go online on the Pokémon Showdown page

These commands log into the public server with the name and password from `.env`.

Ladder (queue on the official server):

```bat
run_bot.bat --mode ladder --format gen9ou
```

Challenge a person who is already on [play.pokemonshowdown.com](https://play.pokemonshowdown.com) (use their Showdown name):

```bat
run_bot.bat --mode challenge --challenge-user THEIR_NAME --format gen9ou --n-battles 1
```

Sit on the server and accept challenges from anyone:

```bat
run_bot.bat --mode accept --format gen9ou
```

On Linux / macOS, use `./run_bot.sh` with the same flags. Open the Showdown site in a browser to watch or accept the match.

## Repository layout

```
.
├── state_engine.py          # Gen 9 mechanics vectorizer + damage range engine
├── matchup_policy.py        # Type / damage / switch scoring
├── memory_system.py         # Smogon priors, opponent set inference, turn buffer
├── connectome_bridge.py     # MaleCNS / synthetic SNN (KC, MBON, CX, VNC)
├── neuromodulation.py       # DAN reward-prediction error + STDP, 5-HT risk gain
├── run_agent.py             # poke-env Player + CLI
├── run_bot.sh               # Linux / macOS launcher
├── run_bot.bat              # Windows launcher
├── requirements.txt
├── .env.example             # Copy to .env and fill SHOWDOWN_USERNAME / PASSWORD
├── data/
│   ├── smogon_sets.json     # Meta set priors for opponent inference
│   ├── connectome/          # Cached subgraph (.npz), created locally
│   └── weights/             # Plastic KC→MBON weights after STDP
└── teams/
    └── gen9ou_sample.txt    # Example Gen 9 OU paste
```

## Turn loop

1. Parse the `poke-env` `Battle` into a fixed **1652-d** normalized vector (`state_engine.py`).
2. Update opponent set profiles from revealed moves / items / abilities (`memory_system.py`).
3. Run ≤ **50 ms** of LIF propagation through KC → MBON → CX → VNC (`connectome_bridge.py`).
4. Score legal moves and switches from type matchups and damage ranges (`matchup_policy.py`).
5. Apply serotonergic gains, mask illegal actions, emit a `BattleOrder` (`run_agent.py`).

VNC motor map (13 channels):

| Channels | Order |
| --- | --- |
| 0–3 | Attack slots 1–4 |
| 4–7 | Terastallize + Attack 1–4 |
| 8–12 | Switch 1–5 |

## Mechanics coverage (`state_engine.py`)

- Dual-type efficacy including 0× / 0.25× / 0.5× / 1× / 2× / 4×, Tera defensive shift, Stellar / Terastarstorm, Adaptability (2.25), Gen 9 Tera STAB.
- Stat stages −6…+6 with exact multipliers (`+2 = 2.0×`, `−1 = 2/3`), Unaware, crit stage overrides, paralysis speed halving.
- Major status (Burn, Paralysis, Toxic counter, Poison, Sleep counter, Freeze) and minor volatiles (Taunt, Encore, Disable, Confusion, Leech Seed, Substitute, Salt Cure, Infatuation, …).
- Weather (Sun / Rain / Sand / Snow + primitives), terrain (Electric / Grassy / Psychic / Misty), Trick Room, Tailwind.
- Stealth Rock (type-scaled), Spikes 1–3, Toxic Spikes 1–2, Sticky Web, Heavy-Duty Boots, Rapid Spin / Defog / Tidy Up / Mortal Spin.
- Choice lock, Focus Sash, Life Orb, Leftovers, Booster Energy, Intimidate / Regenerator / Magic Bounce / Supreme Overlord / Sturdy / weather setters, Tera used/available.
- Priority brackets −7…+5, speed order with Scarf, Swift Swim / Chlorophyll / Sand Rush / Slush Rush, stages, paralysis, Tailwind, Trick Room.
- Gen 9 damage formula with 16 rolls (85–100), STAB, type, crit (1.5×, Sniper 2.25×), screens, burn, items, abilities.

## Optional: local Showdown server

`--mode local_eval` talks to a server at `ws://localhost:8000/showdown/websocket` on your machine. It does not use the public page. Start a local Pokémon Showdown server first, then:

```bat
run_bot.bat --local --mode local_eval --format gen9ou --n-battles 5
```

## CLI

| Flag | Default | Meaning |
| --- | --- | --- |
| `--username` / `--password` | `.env` | Registered Showdown account |
| `--format` | `gen9ou` | Battle format |
| `--mode` | `selfcheck` | Offline test; use `ladder`, `challenge`, or `accept` to go online |
| `--team` | `teams/gen9ou_sample.txt` | Showdown team paste |
| `--local` | off | Localhost PS server (not the public page) |
| `--challenge-user` | — | Target for `--mode challenge` |
| `--n-battles` | `1` | Games to play |
| `--neuprint` | off | Fetch MaleCNS wiring from Janelia (otherwise local synthetic graph) |

## Connectome notes

The full ~166k-neuron MaleCNS graph is not simulated. The agent uses a **2,367-cell** subgraph (256 projection neurons, 2048 Kenyon cells, 34 MBONs, 16 CX, 13 VNC). By default that graph is synthetic and local. `--neuprint` plus a Janelia token is optional and is the only other online path besides Pokémon Showdown.

## License

This project is licensed under the **MIT License**. See `LICENSE` for the full text.

```
MIT License

Copyright (c) 2026 alvstxr

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

