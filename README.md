# Edge-Fog-Cloud ECG Classification Simulator

This project simulates an intelligent multi-layer deployment for ECG signal classification:

- **Edge layer** (`ecg_3k_params.tflite`) for fast inference
- **Fog layer** (`ecg_46k_params.tflite`) for balanced latency/accuracy
- **Cloud layer** (`ecg_7m_params.tflite`) for highest accuracy

The frontend streams ECG-like values in real-time and shows routing decisions, prediction confidence, latency, and selected layer.

## Architecture

`Frontend -> Gateway -> Decision Engine -> Edge/Fog/Cloud Services`

- **Frontend**: React + Vite + Recharts + WebSocket
- **Gateway**: FastAPI API gateway + decision logic + ECG stream endpoint
- **Model services**: 3 independent FastAPI services using `tensorflow.lite.Interpreter`
- **Containerization**: Docker Compose

## Project Structure

```text
root/
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── Dockerfile
│   └── package.json
├── backend/
│   ├── gateway/
│   │   ├── app.py
│   │   ├── decision_engine.py
│   │   ├── ecg_stream.py
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── edge_service/
│   │   ├── app.py
│   │   ├── model/ecg_3k_params.tflite
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── fog_service/
│   │   ├── app.py
│   │   ├── model/ecg_46k_params.tflite
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   └── cloud_service/
│       ├── app.py
│       ├── model/ecg_7m_params.tflite
│       ├── requirements.txt
│       └── Dockerfile
├── docker-compose.yml
└── README.md
```

## Routing Modes

Gateway `/predict` supports:

1. **Manual strategy**  
   Route directly to selected layer (`edge`, `fog`, `cloud`).

2. **Rule-based strategy**  
   - `low_latency` -> `edge`
   - `high_accuracy` -> `cloud`
   - otherwise -> `fog`

3. **Cascade strategy**  
   `edge -> fog -> cloud`, escalating only when confidence is below threshold.

## TFLite Inference Flow (Implemented in all model services)

Each service uses TensorFlow Lite exactly as required:

1. `interpreter = tf.lite.Interpreter(model_path=...)`
2. `interpreter.allocate_tensors()`
3. `input_details = interpreter.get_input_details()`
4. `output_details = interpreter.get_output_details()`
5. `interpreter.set_tensor(...)`
6. `interpreter.invoke()`
7. `output = interpreter.get_tensor(...)`

Services automatically reshape/resize incoming ECG windows to expected tensor shapes and return:

- `prediction`
- `confidence`
- `latency_ms`
- `layer`

## Latency Simulation

- **Edge**: 5-10 ms
- **Fog**: 20-50 ms
- **Cloud**: 100-300 ms

These are added inside each service with `time.sleep()` to demonstrate practical routing tradeoffs.

## Setup

### 1) Put model files in service model folders

Copy your `.tflite` models to:

- `backend/edge_service/model/ecg_3k_params.tflite`
- `backend/fog_service/model/ecg_46k_params.tflite`
- `backend/cloud_service/model/ecg_7m_params.tflite`

### 2) Run with Docker Compose

```bash
docker-compose up --build
```

### 3) Open app

- Frontend: `http://localhost:5173`
- Gateway API docs: `http://localhost:8000/docs`
- Edge service docs: `http://localhost:8001/docs`
- Fog service docs: `http://localhost:8002/docs`
- Cloud service docs: `http://localhost:8003/docs`

## API Contract

### `POST /predict` (Gateway)

Request example:

```json
{
  "signal": [0.1, 0.2, -0.1],
  "mode": "auto",
  "strategy": "rule_based",
  "preference": "balanced",
  "confidence_threshold": 0.8
}
```

Response fields:

- `selected_layer`
- `prediction`
- `confidence`
- `latency_ms`
- `strategy`
- `hops` (full service path; useful for cascade visualization)

### `GET /ws/stream` (Gateway WebSocket)

Streams ECG-like values every ~200 ms:

- latest `sample`
- rolling `window` for charting
- `timestamp`

## Notes

- This is a deployment and routing simulator to showcase **latency vs accuracy** behavior.
- For production, add authentication, retries/circuit breaking, structured logging, and observability.
