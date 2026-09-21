"""MaleCNS connectome bridge: KC / MBON / CX / VNC spiking network.

Queries ``neuprint-python`` for a MaleCNS subgraph (Kenyon cells → MBONs,
central-complex compass neurons, descending/VNC motor neurons). If the
neuPrint token is missing or rejected, a Drosophila-statistic synthetic
connectome is used so the agent still runs.

Kenyon cells encode high-dimensional sparse combinations of the battle
state. MBONs read out approach vs avoidance valence. The central complex
orients that valence over the discrete action ring (Attack 1-4,
Tera+Attack 1-4, Switch 1-5). VNC motor neurons decode spike rates into
those 13 poke-env channels (a compact 10-channel view is also produced).
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from state_engine import N_MOTOR_CHANNELS, N_MOTOR_COMPACT, STATE_DIM

ROOT = Path(__file__).resolve().parent
CACHE_PATH = ROOT / "data" / "connectome" / "male_cns_subgraph.npz"
WEIGHTS_PATH = ROOT / "data" / "weights" / "kc_mbon.npy"

N_KC = 2048
N_MBON = 34
N_CX = 16
N_VNC = N_MOTOR_CHANNELS  # 13
N_PN = 256  # projection-neuron analogue (compressed state)
KC_CLAWS = 7
KC_SPARSITY = 0.08
DT_MS = 2.0
DEFAULT_BUDGET_MS = 50.0

# Motor layout
# 0-3  attack slots
# 4-7  terastallize + attack
# 8-12 switch slots
MOTOR_ATTACK = slice(0, 4)
MOTOR_TERA = slice(4, 8)
MOTOR_SWITCH = slice(8, 13)

# Approach MBONs (even) vs avoidance (odd), matching MBON-01 / MBON-11 polarity.
APPROACH_MBON = np.arange(0, N_MBON, 2)
AVOID_MBON = np.arange(1, N_MBON, 2)


def _rng(seed: int = 7) -> np.random.Generator:
    return np.random.default_rng(seed)


def _sparse_claws(n_out: int, n_in: int, claws: int, rng: np.random.Generator) -> np.ndarray:
    """Each output neuron samples ``claws`` inputs (Kenyon-cell claw analogue)."""
    w = np.zeros((n_out, n_in), dtype=np.float32)
    for i in range(n_out):
        idx = rng.choice(n_in, size=min(claws, n_in), replace=False)
        w[i, idx] = rng.lognormal(mean=-1.2, sigma=0.4, size=idx.shape).astype(np.float32)
    return w


def _sparse_dense(n_out: int, n_in: int, p: float, rng: np.random.Generator, scale: float = 0.05) -> np.ndarray:
    mask = rng.random((n_out, n_in)) < p
    vals = rng.normal(0.0, scale, size=(n_out, n_in)).astype(np.float32)
    return vals * mask


def synthetic_connectome(seed: int = 7) -> Dict[str, np.ndarray]:
    """Biologically flavoured fallback when MaleCNS cannot be fetched."""
    rng = _rng(seed)
    return {
        "W_pn": _sparse_claws(N_PN, STATE_DIM, claws=24, rng=rng),
        "W_kc": _sparse_claws(N_KC, N_PN, claws=KC_CLAWS, rng=rng),
        "W_mbon": _sparse_dense(N_MBON, N_KC, p=0.35, rng=rng, scale=0.02),
        "W_cx": _sparse_dense(N_CX, N_MBON, p=0.6, rng=rng, scale=0.15),
        "W_vnc": _sparse_dense(N_VNC, N_CX, p=0.8, rng=rng, scale=0.25),
        "W_cx_ring": _cx_ring(rng),
        "meta_source": np.asarray(["synthetic"]),
    }


def _cx_ring(rng: np.random.Generator) -> np.ndarray:
    """Excitatory nearest-neighbour / inhibitory wrap: ellipsoid-body ring attractor."""
    w = np.zeros((N_CX, N_CX), dtype=np.float32)
    for i in range(N_CX):
        w[i, i] = 0.6
        w[i, (i + 1) % N_CX] = 0.35
        w[i, (i - 1) % N_CX] = 0.35
        w[i, (i + N_CX // 2) % N_CX] = -0.45
    w += rng.normal(0.0, 0.02, size=w.shape).astype(np.float32)
    return w


def _try_neuprint(token: str, server: str, dataset: str) -> Optional[Dict[str, np.ndarray]]:
    try:
        from neuprint import Client, NeuronCriteria, fetch_adjacencies, fetch_neurons
    except Exception as exc:  # pragma: no cover - optional dependency
        warnings.warn(f"neuprint-python unavailable ({exc}); using synthetic connectome")
        return None
    try:
        client = Client(server, dataset=dataset, token=token)
        # Lightweight auth/sanity query.
        _ = client.fetch_custom("MATCH (n:Neuron) RETURN n.bodyId AS id LIMIT 1")
    except Exception as exc:
        warnings.warn(f"neuPrint token/dataset rejected ({exc}); using synthetic connectome")
        return None

    def _fetch(type_regex: str, limit: int) -> np.ndarray:
        try:
            neurons, _ = fetch_neurons(NeuronCriteria(type=type_regex, regex=True), client=client)
            ids = np.asarray(neurons["bodyId"].to_numpy()[:limit], dtype=np.int64)
            return ids
        except Exception:
            return np.zeros(0, dtype=np.int64)

    kc_ids = _fetch(".*KC.*", N_KC)
    mbon_ids = _fetch(".*MBON.*", N_MBON)
    cx_ids = _fetch(".*(EPG|PEN|PEG|EL|FB[0-9]|PB[0-9]).*", N_CX)
    dn_ids = _fetch(".*DN[a-z].*", N_VNC)

    conn = synthetic_connectome(seed=11)
    conn["meta_source"] = np.asarray(["neuprint-partial"])

    def _adjacency(src: np.ndarray, tgt: np.ndarray, shape: Tuple[int, int]) -> Optional[np.ndarray]:
        if src.size == 0 or tgt.size == 0:
            return None
        try:
            edges, _ = fetch_adjacencies(src.tolist(), tgt.tolist(), client=client)
        except Exception:
            return None
        mat = np.zeros(shape, dtype=np.float32)
        src_index = {int(b): i for i, b in enumerate(src[: shape[1]])}
        tgt_index = {int(b): i for i, b in enumerate(tgt[: shape[0]])}
        body_cols = [c for c in edges.columns if "body" in c.lower()]
        if "bodyId_pre" in edges.columns and "bodyId_post" in edges.columns:
            pre_col, post_col = "bodyId_pre", "bodyId_post"
        elif len(body_cols) >= 2:
            pre_col, post_col = body_cols[0], body_cols[1]
        else:
            return None
        weight_col = "weight" if "weight" in edges.columns else None
        for _, row in edges.iterrows():
            pre = int(row[pre_col])
            post = int(row[post_col])
            if pre not in src_index or post not in tgt_index:
                continue
            w = float(row[weight_col]) if weight_col else 1.0
            mat[tgt_index[post], src_index[pre]] += w
        maxabs = np.max(np.abs(mat)) or 1.0
        return (mat / maxabs).astype(np.float32)

    w_mbon = _adjacency(kc_ids, mbon_ids, (N_MBON, N_KC))
    if w_mbon is not None and np.any(w_mbon):
        # Pad / crop to the fixed SNN size.
        padded = np.zeros((N_MBON, N_KC), dtype=np.float32)
        padded[: w_mbon.shape[0], : w_mbon.shape[1]] = w_mbon[: N_MBON, : N_KC]
        # Mix a small synthetic prior so unused KCs still learn via STDP.
        padded += 0.15 * conn["W_mbon"]
        conn["W_mbon"] = padded
        conn["meta_source"] = np.asarray(["neuprint-male-cns:v1.0"])
    w_vnc = _adjacency(cx_ids if cx_ids.size else mbon_ids, dn_ids, (N_VNC, N_CX))
    if w_vnc is not None and np.any(w_vnc):
        padded = np.zeros((N_VNC, N_CX), dtype=np.float32)
        h, wd = min(N_VNC, w_vnc.shape[0]), min(N_CX, w_vnc.shape[1])
        padded[:h, :wd] = w_vnc[:h, :wd]
        conn["W_vnc"] = padded + 0.2 * conn["W_vnc"]
    return conn


def load_connectome(
    *,
    token: Optional[str] = None,
    server: Optional[str] = None,
    dataset: Optional[str] = None,
    force_synthetic: bool = False,
) -> Dict[str, np.ndarray]:
    token = token or os.environ.get("NEUPRINT_APPLICATION_TOKEN") or os.environ.get("NEUPRINT_TOKEN")
    server = server or os.environ.get("NEUPRINT_SERVER") or "https://neuprint.janelia.org"
    dataset = dataset or os.environ.get("NEUPRINT_DATASET") or "male-cns:v1.0"
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    use_neuprint = os.environ.get("FLY_USE_NEUPRINT", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if not use_neuprint:
        force_synthetic = True

    if CACHE_PATH.exists() and not force_synthetic:
        try:
            loaded = dict(np.load(CACHE_PATH, allow_pickle=True))
            if int(loaded["W_kc"].shape[0]) == N_KC:
                return loaded
        except Exception:
            pass

    conn: Optional[Dict[str, np.ndarray]] = None
    if token and not force_synthetic:
        conn = _try_neuprint(token, server, dataset)
    if conn is None:
        if token and not force_synthetic:
            warnings.warn("Falling back to a synthetic Drosophila-like connectome.")
        conn = synthetic_connectome()
    try:
        np.savez_compressed(CACHE_PATH, **conn)
    except Exception:
        pass
    return conn


@dataclass
class SpikePacket:
    pn: np.ndarray
    kc: np.ndarray
    mbon: np.ndarray
    cx: np.ndarray
    vnc: np.ndarray
    motor: np.ndarray
    motor_compact: np.ndarray
    elapsed_ms: float
    steps: int
    source: str


class ConnectomeSNN:
    """Rate / LIF hybrid that must finish within ``budget_ms`` (default 50 ms)."""

    def __init__(
        self,
        connectome: Optional[Dict[str, np.ndarray]] = None,
        *,
        device: Optional[str] = None,
    ) -> None:
        self.conn = connectome or load_connectome()
        self.device_name = device or os.environ.get("FLY_TORCH_DEVICE") or "cpu"
        self._torch = None
        self._torch_mod = None
        try:
            import torch

            self._torch_mod = torch
            self._torch = True
            if self.device_name.startswith("cuda") and not torch.cuda.is_available():
                self.device_name = "cpu"
        except Exception:
            self._torch = False
        self.W_pn = self.conn["W_pn"].astype(np.float32)
        self.W_kc = self.conn["W_kc"].astype(np.float32)
        self.W_mbon = self.conn["W_mbon"].astype(np.float32)
        self.W_cx = self.conn["W_cx"].astype(np.float32)
        self.W_vnc = self.conn["W_vnc"].astype(np.float32)
        self.W_cx_ring = self.conn["W_cx_ring"].astype(np.float32)
        self._load_plastic_weights()
        self.last_kc = np.zeros(N_KC, dtype=np.float32)
        self.last_mbon = np.zeros(N_MBON, dtype=np.float32)
        self.last_packet: Optional[SpikePacket] = None

        if self._torch:
            torch = self._torch_mod
            dev = torch.device(self.device_name)
            self._t = {
                "pn": torch.as_tensor(self.W_pn, device=dev),
                "kc": torch.as_tensor(self.W_kc, device=dev),
                "mbon": torch.as_tensor(self.W_mbon, device=dev),
                "cx": torch.as_tensor(self.W_cx, device=dev),
                "vnc": torch.as_tensor(self.W_vnc, device=dev),
                "ring": torch.as_tensor(self.W_cx_ring, device=dev),
            }

    def _load_plastic_weights(self) -> None:
        WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        if WEIGHTS_PATH.exists():
            try:
                loaded = np.load(WEIGHTS_PATH)
                if loaded.shape == self.W_mbon.shape:
                    self.W_mbon = loaded.astype(np.float32)
            except Exception:
                pass

    def save_plastic_weights(self) -> None:
        WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(WEIGHTS_PATH, self.W_mbon)

    def set_mbon_weights(self, weights: np.ndarray) -> None:
        if weights.shape != self.W_mbon.shape:
            return
        self.W_mbon = np.asarray(weights, dtype=np.float32)
        if self._torch:
            torch = self._torch_mod
            self._t["mbon"] = torch.as_tensor(self.W_mbon, device=torch.device(self.device_name))

    def _k_wta(self, drive: np.ndarray, k: int) -> np.ndarray:
        spikes = np.zeros_like(drive, dtype=np.float32)
        if k <= 0:
            return spikes
        idx = np.argpartition(drive, -k)[-k:]
        thresh = np.max(drive[idx]) * 0.0 + np.min(drive[idx])
        active = drive >= thresh
        spikes[active] = 1.0 / (1.0 + np.exp(-drive[active]))
        return spikes

    def propagate(
        self,
        state: np.ndarray,
        *,
        serotonin: float = 0.5,
        rpe: float = 0.0,
        budget_ms: float = DEFAULT_BUDGET_MS,
        legal_mask: Optional[np.ndarray] = None,
    ) -> SpikePacket:
        t0 = time.perf_counter()
        x = np.asarray(state, dtype=np.float32).ravel()
        if x.size < STATE_DIM:
            x = np.pad(x, (0, STATE_DIM - x.size))
        elif x.size > STATE_DIM:
            x = x[:STATE_DIM]

        # Projection neurons: dense compression of the battle vector.
        pn = np.tanh(self.W_pn @ x)
        # Kenyon cells: sparse combinatorial code (~8% active).
        kc_drive = self.W_kc @ pn
        k = max(1, int(N_KC * KC_SPARSITY))
        kc = self._k_wta(kc_drive, k)
        # Dopamine gain on KC (DAN modulation of sparse code).
        kc = kc * (1.0 + 0.25 * np.tanh(rpe))

        steps = max(1, int(budget_ms / DT_MS))
        mbon_v = np.zeros(N_MBON, dtype=np.float32)
        cx_v = np.zeros(N_CX, dtype=np.float32)
        vnc_v = np.zeros(N_VNC, dtype=np.float32)
        mbon_acc = np.zeros(N_MBON, dtype=np.float32)
        cx_acc = np.zeros(N_CX, dtype=np.float32)
        vnc_acc = np.zeros(N_VNC, dtype=np.float32)

        # Serotonin biases approach vs avoidance MBONs (patience vs risk).
        mbon_gain = np.ones(N_MBON, dtype=np.float32)
        mbon_gain[APPROACH_MBON] *= 1.0 + 0.6 * (1.0 - serotonin)
        mbon_gain[AVOID_MBON] *= 1.0 + 0.6 * serotonin

        decay = 0.72
        thresh = 1.0
        actual_steps = 0
        for _ in range(steps):
            if (time.perf_counter() - t0) * 1000.0 >= budget_ms:
                break
            mbon_v = decay * mbon_v + (self.W_mbon @ kc) * mbon_gain
            mbon_spk = (mbon_v > thresh).astype(np.float32)
            mbon_v = mbon_v * (1.0 - mbon_spk)
            mbon_acc += mbon_spk

            cx_v = decay * cx_v + (self.W_cx @ mbon_spk) + (self.W_cx_ring @ np.maximum(cx_v, 0))
            cx_spk = (cx_v > thresh).astype(np.float32)
            cx_v = cx_v * (1.0 - cx_spk)
            cx_acc += cx_spk

            vnc_v = decay * vnc_v + (self.W_vnc @ cx_spk)
            vnc_spk = (vnc_v > thresh).astype(np.float32)
            vnc_v = vnc_v * (1.0 - vnc_spk)
            vnc_acc += vnc_spk
            actual_steps += 1
            if actual_steps >= steps:
                break

        if actual_steps == 0:
            mbon_acc = np.maximum(self.W_mbon @ kc, 0.0)
            cx_acc = np.maximum(self.W_cx @ mbon_acc, 0.0)
            vnc_acc = np.maximum(self.W_vnc @ cx_acc, 0.0)
            actual_steps = 1

        motor = vnc_acc / float(actual_steps)
        if legal_mask is not None:
            motor = motor * np.asarray(legal_mask, dtype=np.float32)
        compact = np.zeros(N_MOTOR_COMPACT, dtype=np.float32)
        compact[:4] = motor[MOTOR_ATTACK]
        compact[4:9] = motor[MOTOR_SWITCH]
        compact[9] = float(np.mean(motor[MOTOR_TERA])) if motor[MOTOR_TERA].size else 0.0

        elapsed = (time.perf_counter() - t0) * 1000.0
        source = str(self.conn.get("meta_source", np.asarray(["unknown"]))[0])
        packet = SpikePacket(
            pn=pn.astype(np.float32),
            kc=kc.astype(np.float32),
            mbon=(mbon_acc / float(actual_steps)).astype(np.float32),
            cx=(cx_acc / float(actual_steps)).astype(np.float32),
            vnc=(vnc_acc / float(actual_steps)).astype(np.float32),
            motor=motor.astype(np.float32),
            motor_compact=compact,
            elapsed_ms=float(elapsed),
            steps=int(actual_steps),
            source=source,
        )
        self.last_kc = packet.kc
        self.last_mbon = packet.mbon
        self.last_packet = packet
        return packet


def apply_serotonin_gains(
    motor: np.ndarray,
    gains: Dict[str, float],
    *,
    move_risk: Optional[np.ndarray] = None,
    move_is_setup: Optional[np.ndarray] = None,
) -> np.ndarray:
    out = motor.astype(np.float64).copy()
    out[MOTOR_ATTACK] *= gains.get("attack", 1.0)
    out[MOTOR_TERA] *= gains.get("tera", 1.0)
    out[MOTOR_SWITCH] *= gains.get("switch", 1.0)
    if move_is_setup is not None:
        setup = np.asarray(move_is_setup, dtype=np.float64)
        out[:4] *= 1.0 + (gains.get("setup_hazard_recovery", 1.0) - 1.0) * setup
        out[4:8] *= 1.0 + (gains.get("setup_hazard_recovery", 1.0) - 1.0) * setup
    if move_risk is not None:
        risk = np.asarray(move_risk, dtype=np.float64)
        out[:4] *= 1.0 + (gains.get("high_risk_move", 1.0) - 1.0) * risk
        out[4:8] *= 1.0 + (gains.get("high_risk_move", 1.0) - 1.0) * risk
    explore = float(gains.get("explore", 0.0))
    if explore > 0:
        out += explore * np.random.default_rng().random(out.shape)
    return out.astype(np.float32)
