from pathlib import Path

from common.tflite_service import TFLiteECGService
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    signal: list[float] = Field(..., min_length=1)


app = FastAPI(title="Edge ECG Service")
service: TFLiteECGService | None = None


@app.on_event("startup")
async def startup_event() -> None:
    global service
    model_path = Path("model/ecg_3k_params.tflite")
    if not model_path.exists():
        raise RuntimeError(f"Model file not found: {model_path}")
    service = TFLiteECGService(model_path, layer_name="edge", latency_range_ms=(5.0, 10.0))


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "edge"}


@app.post("/predict")
async def predict(payload: PredictRequest) -> dict:
    if service is None:
        raise HTTPException(status_code=503, detail="Model service is not ready.")
    return service.predict(payload.signal)
