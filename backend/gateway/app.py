import asyncio
import time
from collections import deque
from typing import Any, Literal, cast

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from decision_engine import DecisionEngine, LayerName, RoutingContext, RoutingStrategy, compute_routing_complexity
from ecg_simulation import SimulationConfig, apply_simulation_config
from ecg_stream import ECGStreamGenerator
from layer_metrics import LayerMetricsTracker

SERVICE_URLS = {
    "edge": "http://edge-service:8001/predict",
    "fog": "http://fog-service:8002/predict",
    "cloud": "http://cloud-service:8003/predict",
}

SimulationMode = Literal["normal", "noisy", "abnormal", "drift"]


class PredictRequest(BaseModel):
    signal: list[float] = Field(..., min_length=1)
    mode: str = "auto"
    strategy: RoutingStrategy = "rule_based"
    preference: str = "balanced"
    confidence_threshold: float = 0.8
    simulation_enabled: bool = False
    simulation_mode: SimulationMode = "normal"
    simulation_intensity: float = Field(0.1, ge=0.0, le=1.0)
    max_latency_ms: float | None = Field(default=None, ge=0.0)


app = FastAPI(title="ECG Gateway")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

decision_engine = DecisionEngine()
ecg_generator = ECGStreamGenerator()
layer_metrics = LayerMetricsTracker()


async def call_service(layer: str, signal: list[float]) -> dict[str, Any]:
    url = SERVICE_URLS[layer]
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json={"signal": signal})
        response.raise_for_status()
        return response.json()


def _observe_hop(hop: dict[str, Any]) -> None:
    raw = hop.get("layer")
    if raw not in ("edge", "fog", "cloud"):
        return
    layer = cast(LayerName, raw)
    layer_metrics.observe(
        layer,
        model_latency_ms=float(hop["model_latency_ms"]),
        total_latency_ms=float(hop["total_latency_ms"]),
        confidence=float(hop["confidence"]),
    )


def _sum_latencies(hops: list[dict[str, Any]]) -> tuple[float, float, float]:
    model_sum = round(sum(float(h["model_latency_ms"]) for h in hops), 3)
    sim_sum = round(sum(float(h["simulated_latency_ms"]) for h in hops), 3)
    total_sum = round(sum(float(h["total_latency_ms"]) for h in hops), 3)
    return model_sum, sim_sum, total_sum


def _build_response(
    *,
    chosen: dict[str, Any],
    hops: list[dict[str, Any]],
    escalated: bool,
    simulation_mode: str,
    strategy: RoutingStrategy,
    routing_complexity: float | None,
    request_latency_ms: float,
    predicted_input: list[float],
) -> dict[str, Any]:
    model_ms, sim_ms, total_ms = _sum_latencies(hops)
    layer = str(chosen["layer"])
    out: dict[str, Any] = {
        "layer": layer,
        "selected_layer": layer,
        "prediction": int(chosen["prediction"]),
        "confidence": float(chosen["confidence"]),
        "model_latency_ms": model_ms,
        "simulated_latency_ms": sim_ms,
        "total_latency_ms": total_ms,
        "escalated": escalated,
        "simulation_mode": simulation_mode,
        "strategy": strategy,
        "hops": hops,
        "request_latency_ms": round(request_latency_ms, 3),
        "predicted_input": predicted_input,
    }
    if routing_complexity is not None:
        out["routing_complexity"] = routing_complexity
    return out


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "gateway"}


@app.post("/predict")
async def predict(payload: PredictRequest) -> dict[str, Any]:
    t_req0 = time.perf_counter()
    sim_cfg = SimulationConfig(
        enabled=payload.simulation_enabled,
        mode=payload.simulation_mode,
        intensity=payload.simulation_intensity,
    )
    working_signal, sim_label = apply_simulation_config(payload.signal, sim_cfg)

    ctx = RoutingContext(
        mode=payload.mode if payload.mode in {"auto", "edge", "fog", "cloud"} else "auto",
        strategy=payload.strategy,
        preference=payload.preference if payload.preference in {"low_latency", "balanced", "high_accuracy"} else "balanced",
        confidence_threshold=payload.confidence_threshold,
        max_latency_ms=payload.max_latency_ms,
    )

    routing_complexity: float | None = None
    if ctx.strategy == "rule_based" and ctx.preference == "balanced" and ctx.max_latency_ms is None:
        routing_complexity = round(compute_routing_complexity(working_signal), 4)

    try:
        if payload.strategy == "cascade":
            visited: list[dict[str, Any]] = []
            for layer in decision_engine.cascade_order():
                result = await call_service(layer, working_signal)
                visited.append(result)
                _observe_hop(result)
                if float(result["confidence"]) >= ctx.confidence_threshold:
                    break
            chosen = visited[-1]
            escalated = len(visited) > 1
            req_ms = (time.perf_counter() - t_req0) * 1000.0
            return _build_response(
                chosen=chosen,
                hops=visited,
                escalated=escalated,
                simulation_mode=sim_label,
                strategy=payload.strategy,
                routing_complexity=routing_complexity,
                request_latency_ms=req_ms,
                predicted_input=working_signal,
            )

        layer: LayerName = decision_engine.choose_initial_layer(ctx, working_signal)
        result = await call_service(layer, working_signal)
        _observe_hop(result)
        req_ms = (time.perf_counter() - t_req0) * 1000.0
        return _build_response(
            chosen=result,
            hops=[result],
            escalated=False,
            simulation_mode=sim_label,
            strategy=payload.strategy,
            routing_complexity=routing_complexity,
            request_latency_ms=req_ms,
            predicted_input=working_signal,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"Service call failed: {exc}") from exc


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    window = deque(maxlen=256)
    try:
        while True:
            sample = ecg_generator.next_value()
            window.append(sample)
            await websocket.send_json(
                {"timestamp": ecg_generator.now_ms(), "sample": sample, "window": list(window)}
            )
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        return
