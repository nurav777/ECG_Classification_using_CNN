from __future__ import annotations

from typing import Literal

LayerName = Literal["edge", "fog", "cloud"]

_LAYERS: tuple[LayerName, ...] = ("edge", "fog", "cloud")


class LayerMetricsTracker:
    """Exponential moving averages of per-layer model latency, total service latency, and confidence."""

    def __init__(self, alpha: float = 0.25) -> None:
        self._alpha = float(alpha)
        self._ema_model: dict[LayerName, float | None] = {L: None for L in _LAYERS}
        self._ema_total: dict[LayerName, float | None] = {L: None for L in _LAYERS}
        self._ema_conf: dict[LayerName, float | None] = {L: None for L in _LAYERS}

    def observe(self, layer: LayerName, *, model_latency_ms: float, total_latency_ms: float, confidence: float) -> None:
        a = self._alpha
        lm = self._ema_model[layer]
        lt = self._ema_total[layer]
        lc = self._ema_conf[layer]
        self._ema_model[layer] = model_latency_ms if lm is None else a * model_latency_ms + (1.0 - a) * lm
        self._ema_total[layer] = total_latency_ms if lt is None else a * total_latency_ms + (1.0 - a) * lt
        self._ema_conf[layer] = confidence if lc is None else a * confidence + (1.0 - a) * lc

    def ema_model_latency_ms(self, layer: LayerName) -> float | None:
        return self._ema_model[layer]

    def ema_total_latency_ms(self, layer: LayerName) -> float | None:
        return self._ema_total[layer]

    def ema_confidence(self, layer: LayerName) -> float | None:
        return self._ema_conf[layer]
