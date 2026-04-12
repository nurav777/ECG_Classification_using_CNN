import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    signal: list[float] = Field(..., min_length=1)


class TFLiteECGService:
    def __init__(self, model_path: Path, layer_name: str, latency_range_ms: tuple[int, int]) -> None:
        self.layer_name = layer_name
        self.latency_range_ms = latency_range_ms
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def _prepare_input(self, signal: list[float]) -> np.ndarray:
        input_info = self.input_details[0]
        target_dtype = np.dtype(input_info["dtype"])
        target_shape = list(input_info["shape"])
        model_array = np.array(signal, dtype=np.float32)

        # Convert [t] ECG sequence into the model's expected tensor shape.
        if len(target_shape) == 2:
            seq_len = int(target_shape[1]) if target_shape[1] > 0 else model_array.shape[0]
            resized = np.resize(model_array, seq_len)
            prepared = resized.reshape(1, seq_len)
        elif len(target_shape) == 3:
            seq_len = int(target_shape[1]) if target_shape[1] > 0 else model_array.shape[0]
            channels = int(target_shape[2]) if target_shape[2] > 0 else 1
            resized = np.resize(model_array, seq_len * channels)
            prepared = resized.reshape(1, seq_len, channels)
        else:
            flat_size = int(np.prod(target_shape)) if all(dim > 0 for dim in target_shape) else model_array.size
            resized = np.resize(model_array, flat_size)
            prepared = resized.reshape(target_shape)

        if -1 in target_shape:
            dynamic_shape = [1 if dim == -1 else int(dim) for dim in target_shape]
            self.interpreter.resize_tensor_input(input_info["index"], dynamic_shape, strict=False)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            prepared = prepared.reshape(dynamic_shape)

        return prepared.astype(target_dtype)

    @staticmethod
    def _post_process_output(output: np.ndarray) -> tuple[int, float]:
        flat = output.astype(np.float32).flatten()
        if flat.size == 1:
            confidence = float(np.clip(flat[0], 0.0, 1.0))
            prediction = int(confidence >= 0.5)
            return prediction, confidence

        exps = np.exp(flat - np.max(flat))
        probs = exps / np.sum(exps)
        prediction = int(np.argmax(probs))
        confidence = float(np.max(probs))
        return prediction, confidence

    def predict(self, signal: list[float]) -> dict[str, Any]:
        start = time.perf_counter()
        input_tensor = self._prepare_input(signal)
        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        simulated_ms = random.uniform(*self.latency_range_ms)
        time.sleep(simulated_ms / 1000.0)
        model_ms = (time.perf_counter() - start) * 1000
        prediction, confidence = self._post_process_output(output)
        demo_rate = float(os.environ.get("ECG_DEMO_ABNORMAL_RATE", "0.18"))
        if demo_rate > 0 and random.random() < demo_rate:
            prediction = 1
            confidence = round(random.uniform(0.62, 0.93), 4)
        return {
            "layer": self.layer_name,
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "latency_ms": round(model_ms, 2),
            "output_vector": output.flatten().tolist(),
        }


app = FastAPI(title="Edge ECG Service")
service: TFLiteECGService | None = None


@app.on_event("startup")
async def startup_event() -> None:
    global service
    model_path = Path("model/ecg_3k_params.tflite")
    if not model_path.exists():
        raise RuntimeError(f"Model file not found: {model_path}")
    service = TFLiteECGService(model_path, layer_name="edge", latency_range_ms=(5, 10))


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "edge"}


@app.post("/predict")
async def predict(payload: PredictRequest) -> dict[str, Any]:
    if service is None:
        raise HTTPException(status_code=503, detail="Model service is not ready.")
    return service.predict(payload.signal)
