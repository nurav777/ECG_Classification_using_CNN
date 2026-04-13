from dataclasses import dataclass
from typing import Literal

import numpy as np

SimulationModeLiteral = Literal["normal", "noisy", "abnormal", "drift"]


@dataclass
class SimulationConfig:
    enabled: bool = False
    mode: SimulationModeLiteral = "normal"
    intensity: float = 0.1


def _minmax_01(x: np.ndarray) -> np.ndarray:
    """Per-window min–max to [0, 1] (shayanfazeli/heartbeat MIT-BIH style)."""
    x = x.astype(np.float32).copy()
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-6:
        return np.full_like(x, 0.5, dtype=np.float32)
    return (x - lo) / (hi - lo)


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x.astype(np.float32), 0.0, 1.0)


def simulate_ecg_signal(signal: list[float], mode: str, *, intensity: float = 0.1) -> list[float]:
    """
    Perturb the ECG window for scenario testing. All modes first map the window to [0, 1] so
    TFLite models (trained on MIT-BIH CSV rows) see in-distribution amplitude; perturbations
    stay in that domain and are clipped so the chart stays stable.
    """
    x = _minmax_01(np.asarray(signal, dtype=np.float32))
    s = float(np.clip(intensity, 0.0, 1.0))
    rng = np.random.default_rng()

    if mode == "normal":
        return np.round(x, 6).tolist()

    if mode == "noisy":
        sigma = float(0.018 + 0.09 * s)
        x = x + rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
        if s >= 0.12 and rng.random() < min(0.4, 0.1 + 0.55 * s):
            burst_len = max(3, int(len(x) * (0.03 + 0.07 * s)))
            burst_len = min(burst_len, len(x))
            i0 = int(rng.integers(0, max(1, len(x) - burst_len + 1)))
            x[i0 : i0 + burst_len] += rng.normal(
                0.0, float(0.04 + 0.12 * s), size=burst_len
            ).astype(np.float32)
        x = _clip01(x)
        return np.round(x, 6).tolist()

    if mode == "abnormal":
        x = x + rng.normal(0.0, 0.012 + 0.03 * s, size=x.shape).astype(np.float32)
        # Flatten / suppress segments (morphology break vs training beats)
        n_crush = max(1, int(1 + 5 * s))
        for _ in range(n_crush):
            w = max(6, int(10 + 28 * s))
            i0 = int(rng.integers(0, max(1, len(x) - w + 1)))
            x[i0 : i0 + w] *= float(0.12 + 0.45 * rng.random())
        # Sharp ectopic-like deflections in [0, 1]
        n_defl = max(2, int(2 + 7 * s))
        for _ in range(n_defl):
            center = int(rng.integers(0, len(x)))
            w = max(3, int(4 + 12 * s))
            i0 = max(0, center - w // 2)
            i1 = min(len(x), i0 + w)
            seg = i1 - i0
            if seg < 1:
                continue
            t = np.arange(seg, dtype=np.float32) - float(seg - 1) / 2.0
            env = np.exp(-(t * t) / max(1.5, float(w * w) / 12.0)).astype(np.float32)
            amp = float((0.2 + 0.42 * s) * rng.choice([-1.0, 1.0]))
            x[i0:i1] += amp * env
        # Brief polarity flip on a segment (strong cue vs smooth normal beats)
        if rng.random() < 0.35 + 0.55 * s:
            w = max(8, min(40, int(len(x) * (0.08 + 0.12 * s))))
            i0 = int(rng.integers(0, max(1, len(x) - w + 1)))
            seg = x[i0 : i0 + w].copy()
            x[i0 : i0 + w] = _clip01(1.0 - seg)
        x = _clip01(x)
        return np.round(x, 6).tolist()

    if mode == "drift":
        t = np.linspace(0.0, 1.0, len(x), dtype=np.float32)
        x = x + s * (
            np.float32(0.14) * t + np.float32(0.06) * np.sin(2.0 * np.pi * 3.0 * t)
        )
        x = _clip01(x)
        return np.round(x, 6).tolist()

    return np.round(x, 6).tolist()


def apply_simulation_config(signal: list[float], cfg: SimulationConfig) -> tuple[list[float], str]:
    """Returns (possibly perturbed signal, simulation_mode label for API)."""
    if not cfg.enabled:
        return list(signal), "none"
    if cfg.mode == "normal":
        return simulate_ecg_signal(signal, "normal", intensity=cfg.intensity), "normal"
    return simulate_ecg_signal(signal, cfg.mode, intensity=cfg.intensity), cfg.mode
