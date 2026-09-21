"""Connectome-driven Pokémon Showdown agent (Gen 9 OU / Standard Singles).

Turn loop:
    Battle → state_engine.vectorize
           → memory_system.update / opponent set inference
           → connectome_bridge 50 ms spike propagation
           → neuromodulation (RPE/STDP + serotonin gains)
           → legal-action filter → poke-env BattleOrder
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from connectome_bridge import (  # noqa: E402
    ConnectomeSNN,
    apply_serotonin_gains,
    load_connectome,
)
from matchup_policy import choose_teampreview, score_actions, self_check_matchups  # noqa: E402
from memory_system import MemorySystem  # noqa: E402
from neuromodulation import NeuromodulatoryLoop  # noqa: E402
from state_engine import (  # noqa: E402
    HAZARD_SETTING_MOVES,
    RECOVERY_MOVES,
    SETUP_MOVES,
    BattleStateEngine,
    _self_check_type_chart,
    move_id,
)

LOGGER = logging.getLogger("flyshowdown")


def _read_team(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Team file not found: {path}")
    return p.read_text(encoding="utf-8")


def _validate_neuprint_token(token: Optional[str]) -> bool:
    if not token:
        LOGGER.warning(
            "NEUPRINT_APPLICATION_TOKEN is not set. "
            "The agent will use a synthetic Drosophila-like connectome."
        )
        return False
    try:
        from neuprint import Client

        server = os.environ.get("NEUPRINT_SERVER", "https://neuprint.janelia.org")
        dataset = os.environ.get("NEUPRINT_DATASET", "male-cns:v1.0")
        client = Client(server, dataset=dataset, token=token)
        client.fetch_custom("MATCH (n:Neuron) RETURN n.bodyId AS id LIMIT 1")
        LOGGER.info("neuPrint token accepted for dataset %s", dataset)
        return True
    except Exception as exc:
        LOGGER.error(
            "Invalid or rejected neuPrint token (%s). "
            "Falling back to a synthetic connectome. "
            "Get a token at https://neuprint.janelia.org",
            exc,
        )
        return False


class ConnectomePlayer:
    """poke-env Player that is mixed in after poke-env is imported."""

    def __init_runtime__(self) -> None:
        self.state_engine = BattleStateEngine()
        self.memory = MemorySystem()
        token_ok = _validate_neuprint_token(
            os.environ.get("NEUPRINT_APPLICATION_TOKEN") or os.environ.get("NEUPRINT_TOKEN")
        )
        conn = load_connectome(force_synthetic=not token_ok)
        self.snn = ConnectomeSNN(conn)
        self.neuromod = NeuromodulatoryLoop()
        self._last_action_name = "pass"
        self._last_action_idx = 0

    def teampreview(self, battle) -> str:  # type: ignore[override]
        return choose_teampreview(battle)

    def choose_move(self, battle):  # type: ignore[override]
        Player = _import_player()
        try:
            self.memory.update_from_battle(battle)
            opp_est = self.memory.estimate_for(battle, battle.opponent_active_pokemon)
            inspected = self.state_engine.inspect(battle, opp_estimates=opp_est or None)
            vec = inspected.vector

            packet = None
            try:
                packet = self.snn.propagate(
                    vec,
                    serotonin=self.neuromod.last.serotonin,
                    rpe=self.neuromod.last.rpe,
                    budget_ms=50.0,
                    legal_mask=inspected.legal_mask,
                )
                state, new_w = self.neuromod.step(
                    battle,
                    kc_spikes=packet.kc,
                    mbon_spikes=packet.mbon,
                    kc_mbon_weights=self.snn.W_mbon,
                    outspeed=inspected.outspeed,
                )
                self.snn.set_mbon_weights(new_w)
                gains = self.neuromod.gains()
                moves = list(getattr(battle, "available_moves", None) or [])
                risk = []
                setup = []
                for mv in moves[:4]:
                    acc = getattr(mv, "accuracy", 1.0)
                    try:
                        acc_f = 1.0 if acc is True or acc is None else float(acc or 1.0)
                    except (TypeError, ValueError):
                        acc_f = 1.0
                    if acc_f > 1.5:
                        acc_f /= 100.0
                    power = float(getattr(mv, "base_power", 0) or 0)
                    risk.append(1.0 if (acc_f < 0.85 and power >= 90) else 0.0)
                    mid = move_id(mv)
                    setup.append(
                        1.0
                        if mid in SETUP_MOVES or mid in HAZARD_SETTING_MOVES or mid in RECOVERY_MOVES
                        else 0.0
                    )
                while len(risk) < 4:
                    risk.append(0.0)
                    setup.append(0.0)
                motor = apply_serotonin_gains(
                    packet.motor, gains, move_risk=risk, move_is_setup=setup
                )
                motor = motor * inspected.legal_mask
            except Exception:
                LOGGER.exception("connectome step failed; continuing with matchup scores")
                state = self.neuromod.last
                motor = np.zeros(inspected.legal_mask.shape, dtype=np.float32)

            matchup = score_actions(
                battle,
                inspected,
                opp_estimates=opp_est or None,
            )
            # Matchup/damage scores are the policy; the SNN is a small prior.
            snn = motor.astype("float64")
            if snn.max() > snn.min():
                snn = (snn - snn.min()) / (snn.max() - snn.min() + 1e-8)
            else:
                snn = snn * 0.0
            combined = 0.84 * matchup.astype("float64") + 0.16 * snn
            combined = combined * inspected.legal_mask.astype("float64")

            if inspected.legal_mask.sum() <= 0:
                order = Player.choose_random_move(battle)
                self._remember(battle, vec, 0, "random")
                return order

            idx = int(combined.argmax())
            order = self._order_from_index(battle, idx, Player)
            self._remember(battle, vec, idx, self._last_action_name)
            LOGGER.info(
                "turn %s chose %s (idx=%s, serotonin=%.2f)",
                getattr(battle, "turn", "?"),
                self._last_action_name,
                idx,
                state.serotonin,
            )
            return order
        except Exception:
            LOGGER.exception("choose_move failed; falling back to a random legal order")
            return Player.choose_random_move(battle)

    def _order_from_index(self, battle, idx: int, Player) -> Any:
        moves = list(getattr(battle, "available_moves", None) or [])
        switches = list(getattr(battle, "available_switches", None) or [])
        tera = bool(getattr(battle, "can_tera", False))
        if 0 <= idx <= 3 and idx < len(moves):
            self._last_action_name = move_id(moves[idx])
            self._last_action_idx = idx
            return Player.create_order(moves[idx])
        if 4 <= idx <= 7 and (idx - 4) < len(moves) and tera:
            mv = moves[idx - 4]
            self._last_action_name = "tera+" + move_id(mv)
            self._last_action_idx = idx
            return Player.create_order(mv, terastallize=True)
        if 8 <= idx <= 12 and (idx - 8) < len(switches):
            mon = switches[idx - 8]
            self._last_action_name = "switch:" + str(getattr(mon, "species", "") or "")
            self._last_action_idx = idx
            return Player.create_order(mon)
        # Repair illegal argmax (numerical edge case).
        if moves:
            self._last_action_name = move_id(moves[0])
            self._last_action_idx = 0
            return Player.create_order(moves[0])
        if switches:
            self._last_action_name = "switch"
            self._last_action_idx = 8
            return Player.create_order(switches[0])
        return Player.choose_random_move(battle)

    def _remember(self, battle, vec, idx: int, name: str) -> None:
        try:
            self.memory.record(battle, state=vec, action=idx, action_name=name)
        except Exception:
            LOGGER.debug("memory.record failed", exc_info=True)

    def save(self) -> None:
        try:
            self.snn.save_plastic_weights()
        except Exception:
            LOGGER.debug("could not persist KC→MBON weights", exc_info=True)


def _import_player():
    try:
        from poke_env.player.player import Player
    except ImportError:  # older poke-env layouts
        from poke_env.player import Player
    return Player


def _make_player_class():
    Player = _import_player()

    class FlyConnectomePlayer(ConnectomePlayer, Player):
        def __init__(self, *args, **kwargs):
            Player.__init__(self, *args, **kwargs)
            ConnectomePlayer.__init_runtime__(self)

        def choose_move(self, battle):
            return ConnectomePlayer.choose_move(self, battle)

        def teampreview(self, battle):
            return ConnectomePlayer.teampreview(self, battle)

    return FlyConnectomePlayer


def _server_and_account(args: argparse.Namespace):
    try:
        from poke_env import (
            AccountConfiguration,
            LocalhostServerConfiguration,
            ShowdownServerConfiguration,
        )
    except ImportError:
        from poke_env.ps_client.account_configuration import AccountConfiguration
        from poke_env.ps_client.server_configuration import (
            LocalhostServerConfiguration,
            ShowdownServerConfiguration,
        )

    username = args.username or os.environ.get("SHOWDOWN_USERNAME")
    password = args.password or os.environ.get("SHOWDOWN_PASSWORD")
    use_local = bool(args.local) or args.mode == "local_eval"
    if use_local:
        cfg = LocalhostServerConfiguration
        username = username or "FlyConnectome"
        account = AccountConfiguration(username, password)
    else:
        cfg = ShowdownServerConfiguration
        if not username or not password:
            raise SystemExit(
                "Public Showdown requires --username/--password or "
                "SHOWDOWN_USERNAME / SHOWDOWN_PASSWORD in .env"
            )
        account = AccountConfiguration(username, password)
    return account, cfg


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MaleCNS connectome agent for Pokémon Showdown (Gen 9 OU)"
    )
    p.add_argument("--username", default=None, help="Showdown username")
    p.add_argument("--password", default=None, help="Showdown password")
    p.add_argument("--format", default="gen9ou", help="Battle format (default: gen9ou)")
    p.add_argument(
        "--mode",
        choices=("ladder", "challenge", "accept", "local_eval", "selfcheck"),
        default="ladder",
        help="ladder | challenge | accept | local_eval | selfcheck",
    )
    p.add_argument("--team", default=str(ROOT / "teams" / "gen9ou_sample.txt"), help="Showdown team paste")
    p.add_argument("--local", action="store_true", help="Connect to a local PS server (ws://localhost:8000)")
    p.add_argument("--challenge-user", default=None, help="Username to challenge (--mode challenge)")
    p.add_argument("--n-battles", type=int, default=1, help="Games to play")
    p.add_argument("--avatar", default=None)
    p.add_argument("--log-level", default="INFO")
    return p


async def _run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.mode == "selfcheck":
        _self_check_type_chart()
        self_check_matchups()
        snn = ConnectomeSNN(load_connectome(force_synthetic=True))
        pkt = snn.propagate(np.random.default_rng(0).random(snn.W_pn.shape[1]).astype("float32"))
        print(
            f"selfcheck OK  STATE via SNN  steps={pkt.steps}  "
            f"elapsed={pkt.elapsed_ms:.1f}ms  source={pkt.source}"
        )
        return

    team = _read_team(args.team) if args.format != "gen9randombattle" else None
    account, server = _server_and_account(args)
    Fly = _make_player_class()
    player = Fly(
        account_configuration=account,
        server_configuration=server,
        battle_format=args.format,
        team=team,
        avatar=args.avatar,
        log_level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        save_replays=False,
        start_timer_on_battle_start=True,
    )
    source = player.snn.conn.get("meta_source", ["unknown"])
    if hasattr(source, "tolist"):
        source = source.tolist()[0]
    LOGGER.info(
        "Connected as %s  format=%s  mode=%s  connectome=%s",
        player.username,
        args.format,
        args.mode,
        source,
    )
    try:
        if args.mode == "ladder":
            await player.ladder(args.n_battles)
        elif args.mode == "challenge":
            if not args.challenge_user:
                raise SystemExit("--challenge-user is required for --mode challenge")
            await player.send_challenges(args.challenge_user, n_challenges=args.n_battles)
        elif args.mode == "accept":
            await player.accept_challenges(None, n_challenges=args.n_battles)
        elif args.mode == "local_eval":
            from poke_env import AccountConfiguration, LocalhostServerConfiguration

            try:
                from poke_env.player import SimpleHeuristicsPlayer
            except ImportError:
                from poke_env.player.baselines import SimpleHeuristicsPlayer

            opp = SimpleHeuristicsPlayer(
                account_configuration=AccountConfiguration("HeuristicOpp", None),
                server_configuration=LocalhostServerConfiguration,
                battle_format=args.format,
                team=team,
            )
            await player.battle_against(opp, n_battles=args.n_battles)
            LOGGER.info(
                "local_eval finished  W/L/T = %s/%s/%s  winrate=%.2f",
                player.n_won_battles,
                player.n_lost_battles,
                player.n_tied_battles,
                player.win_rate,
            )
    finally:
        player.save()


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
