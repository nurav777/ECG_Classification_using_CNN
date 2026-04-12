from dataclasses import dataclass
from typing import Literal


LayerName = Literal["edge", "fog", "cloud"]
RouteMode = Literal["auto", "edge", "fog", "cloud"]
RoutingStrategy = Literal["manual", "rule_based", "cascade"]


@dataclass
class RoutingContext:
    mode: RouteMode
    strategy: RoutingStrategy
    preference: Literal["low_latency", "balanced", "high_accuracy"] = "balanced"
    confidence_threshold: float = 0.8


def compute_routing_complexity(signal: list[float]) -> float:
    """Heuristic “how hard is this window?” score from the live ECG window (variance + spread)."""
    n = len(signal)
    if n < 2:
        return 0.0
    mean = sum(signal) / n
    var = sum((x - mean) ** 2 for x in signal) / n
    mad = sum(abs(x - mean) for x in signal) / n
    return var + 0.45 * mad


class DecisionEngine:
    # Balanced rule-based: map complexity to layer (tuned for the synthetic stream).
    _BAL_EDGE_BELOW = 0.035
    _BAL_FOG_BELOW = 0.12

    def choose_initial_layer(self, ctx: RoutingContext, signal: list[float] | None = None) -> LayerName:
        if ctx.strategy == "manual" and ctx.mode in {"edge", "fog", "cloud"}:
            return ctx.mode

        if ctx.strategy == "rule_based":
            if ctx.preference == "low_latency":
                return "edge"
            if ctx.preference == "high_accuracy":
                return "cloud"
            # Balanced: route by window “complexity” so edge / fog / cloud all appear in demos.
            if signal:
                c = compute_routing_complexity(signal)
                if c < self._BAL_EDGE_BELOW:
                    return "edge"
                if c < self._BAL_FOG_BELOW:
                    return "fog"
                return "cloud"
            return "fog"

        # For cascade and default auto behavior, start from edge.
        return "edge"

    @staticmethod
    def cascade_order() -> list[LayerName]:
        return ["edge", "fog", "cloud"]
