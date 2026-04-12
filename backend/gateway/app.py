import asyncio
import time
from collections import deque
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from decision_engine import DecisionEngine, RoutingContext, RoutingStrategy, compute_routing_complexity
from ecg_stream import ECGStreamGenerator

SERVICE_URLS = {
    "edge": "http://edge-service:8001/predict",
    "fog": "http://fog-service:8002/predict",
    "cloud": "http://cloud-service:8003/predict",
}


class PredictRequest(BaseModel):
    signal: list[float] = Field(..., min_length=1)
    mode: str = "auto"
    strategy: RoutingStrategy = "rule_based"
    preference: str = "balanced"
    confidence_threshold: float = 0.8


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


async def call_service(layer: str, signal: list[float]) -> dict[str, Any]:
    url = SERVICE_URLS[layer]
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, json={"signal": signal})
        response.raise_for_status()
        return response.json()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "gateway"}


@app.post("/predict")
async def predict(payload: PredictRequest) -> dict[str, Any]:
    start_ms = time.perf_counter() * 1000
    ctx = RoutingContext(
        mode=payload.mode if payload.mode in {"auto", "edge", "fog", "cloud"} else "auto",
        strategy=payload.strategy,
        preference=payload.preference if payload.preference in {"low_latency", "balanced", "high_accuracy"} else "balanced",
        confidence_threshold=payload.confidence_threshold,
    )

    try:
        if payload.strategy == "cascade":
            visited: list[dict[str, Any]] = []
            for layer in decision_engine.cascade_order():
                result = await call_service(layer, payload.signal)
                visited.append(result)
                if result["confidence"] >= ctx.confidence_threshold:
                    break
            chosen = visited[-1]
            total_latency = round((time.perf_counter() * 1000) - start_ms, 2)
            out: dict[str, Any] = {
                "selected_layer": chosen["layer"],
                "prediction": chosen["prediction"],
                "confidence": chosen["confidence"],
                "latency_ms": total_latency,
                "strategy": payload.strategy,
                "hops": visited,
            }
            if ctx.strategy == "rule_based" and ctx.preference == "balanced":
                out["routing_complexity"] = round(compute_routing_complexity(payload.signal), 4)
            return out

        layer = decision_engine.choose_initial_layer(ctx, payload.signal)
        result = await call_service(layer, payload.signal)
        total_latency = round((time.perf_counter() * 1000) - start_ms, 2)
        out = {
            "selected_layer": result["layer"],
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "latency_ms": total_latency,
            "strategy": payload.strategy,
            "hops": [result],
        }
        if ctx.strategy == "rule_based" and ctx.preference == "balanced":
            out["routing_complexity"] = round(compute_routing_complexity(payload.signal), 4)
        return out
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
