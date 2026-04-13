import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


class TFLiteECGService:
    """TFLite ECG classifier: truthful model timing (invoke only) vs optional simulated network delay."""

    def __init__(self, model_path: Path, layer_name: str, latency_range_ms: tuple[float, float]) -> None:
        self.layer_name = layer_name
        self.latency_range_ms = latency_range_ms
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    @staticmethod
    def _fit_length(arr: np.ndarray, target_len: int, pad_value: float = 0.0) -> np.ndarray:
        """Truncate or zero-pad to target length; never tile/repeat samples."""
        if arr.size >= target_len:
            return arr[:target_len].copy()
        out = np.full((target_len,), pad_value, dtype=arr.dtype)
        out[: arr.size] = arr
        return out

    def _prepare_input(self, signal: list[float]) -> np.ndarray:
        input_info = self.input_details[0]
        target_dtype = np.dtype(input_info["dtype"])
        target_shape = list(input_info["shape"])
        model_array = np.asarray(signal, dtype=np.float32).ravel()

        if -1 in target_shape:
            dynamic_shape = [1 if dim == -1 else int(dim) for dim in target_shape]
            self.interpreter.resize_tensor_input(input_info["index"], dynamic_shape, strict=False)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            need = int(np.prod(dynamic_shape))
            flat = self._fit_length(model_array, need)
            prepared = flat.reshape(dynamic_shape)
            return prepared.astype(target_dtype)

        if len(target_shape) == 2:
            seq_len = int(target_shape[1]) if target_shape[1] > 0 else model_array.size
            flat = self._fit_length(model_array, seq_len)
            prepared = flat.reshape(1, seq_len)
        elif len(target_shape) == 3:
            seq_len = int(target_shape[1]) if target_shape[1] > 0 else model_array.size
            channels = int(target_shape[2]) if target_shape[2] > 0 else 1
            need = seq_len * channels
            flat = self._fit_length(model_array, need)
            prepared = flat.reshape(1, seq_len, channels)
        else:
            flat_size = int(np.prod(target_shape)) if all(dim > 0 for dim in target_shape) else model_array.size
            flat = self._fit_length(model_array, flat_size)
            prepared = flat.reshape(target_shape)

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
        input_tensor = self._prepare_input(signal)
        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)

        invoke_start = time.perf_counter()
        self.interpreter.invoke()
        model_latency_ms = (time.perf_counter() - invoke_start) * 1000.0

        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        prediction, confidence = self._post_process_output(output)

        simulated_ms = float(random.uniform(*self.latency_range_ms))
        time.sleep(simulated_ms / 1000.0)

        model_rounded = round(model_latency_ms, 3)
        sim_rounded = round(simulated_ms, 3)
        total_rounded = round(model_rounded + sim_rounded, 3)

        return {
            "layer": self.layer_name,
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "model_latency_ms": model_rounded,
            "simulated_latency_ms": sim_rounded,
            "total_latency_ms": total_rounded,
            "output_vector": output.flatten().tolist(),
        }
