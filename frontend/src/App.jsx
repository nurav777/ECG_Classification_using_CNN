import { useEffect, useMemo, useRef, useState } from "react";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const DEFAULT_WINDOW = 187;
const API_BASE = import.meta.env.VITE_GATEWAY_URL || "http://localhost:8000";
const WS_BASE = import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws/stream";

const LAYER_THEME = {
  edge: { label: "Edge", pillClass: "layer-pill--edge" },
  fog: { label: "Fog", pillClass: "layer-pill--fog" },
  cloud: { label: "Cloud", pillClass: "layer-pill--cloud" }
};

function App() {
  const [mode, setMode] = useState("auto");
  const [strategy, setStrategy] = useState("rule_based");
  const [preference, setPreference] = useState("balanced");
  const [threshold, setThreshold] = useState(0.8);
  const [samples, setSamples] = useState([]);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("connecting...");
  const lastPredictRef = useRef(0);

  useEffect(() => {
    const socket = new WebSocket(WS_BASE);
    socket.onopen = () => setStatus("streaming");
    socket.onclose = () => setStatus("disconnected");
    socket.onerror = () => setStatus("error");
    socket.onmessage = async (event) => {
      const data = JSON.parse(event.data);
      setSamples(data.window || []);

      const now = Date.now();
      if (now - lastPredictRef.current < 500) {
        return;
      }

      const windowSignal = (data.window || []).slice(-DEFAULT_WINDOW);
      if (windowSignal.length < DEFAULT_WINDOW) {
        return;
      }

      lastPredictRef.current = now;
      try {
        const response = await fetch(`${API_BASE}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            signal: windowSignal,
            mode,
            strategy,
            preference,
            confidence_threshold: Number(threshold)
          })
        });
        if (!response.ok) {
          throw new Error(`Request failed: ${response.status}`);
        }
        const payload = await response.json();
        setResult(payload);
      } catch (error) {
        setStatus(`predict error: ${error.message}`);
      }
    };

    return () => socket.close();
  }, [mode, strategy, preference, threshold]);

  const chartData = useMemo(
    () => samples.map((value, idx) => ({ i: idx, value })),
    [samples]
  );

  const selectedLayer = result?.selected_layer;
  const layerVisual = selectedLayer
    ? LAYER_THEME[selectedLayer] ?? { label: String(selectedLayer), pillClass: "layer-pill--unknown" }
    : null;
  const predictionLabel =
    result?.prediction === 0 || result?.prediction === 1
      ? result.prediction === 1
        ? "Abnormal"
        : "Normal"
      : null;

  return (
    <div className="app">
      <h1>Edge-Fog-Cloud ECG Classification Simulator</h1>
      <p className="status">Stream status: {status}</p>

      <section className="controls">
        <label>
          Strategy
          <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
            <option value="manual">Manual</option>
            <option value="rule_based">Rule-based</option>
            <option value="cascade">Cascade</option>
          </select>
        </label>

        <label>
          Mode
          <select value={mode} onChange={(e) => setMode(e.target.value)} disabled={strategy !== "manual"}>
            <option value="auto">Auto</option>
            <option value="edge">Edge</option>
            <option value="fog">Fog</option>
            <option value="cloud">Cloud</option>
          </select>
        </label>

        <label>
          Preference
          <select
            value={preference}
            onChange={(e) => setPreference(e.target.value)}
            disabled={strategy !== "rule_based"}
          >
            <option value="low_latency">Low latency</option>
            <option value="balanced">Balanced</option>
            <option value="high_accuracy">High accuracy</option>
          </select>
        </label>

        <label>
          Cascade threshold
          <input
            type="number"
            min="0.1"
            max="0.99"
            step="0.01"
            value={threshold}
            onChange={(e) => setThreshold(e.target.value)}
            disabled={strategy !== "cascade"}
          />
        </label>
      </section>

      <section className="chart">
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={chartData}>
            <XAxis dataKey="i" hide />
            <YAxis domain={[-1.2, 1.2]} />
            <Tooltip />
            <Line type="monotone" dataKey="value" stroke="#00c2ff" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </section>

      {result && selectedLayer && layerVisual && (
        <div className={`layer-hero ${layerVisual.pillClass}`}>
          <div className="layer-hero__label">Routed to</div>
          <div className="layer-hero__name">{layerVisual.label}</div>
          {strategy === "rule_based" && preference === "balanced" && result.routing_complexity != null && (
            <div className="layer-hero__hint">
              Rule-based (balanced) uses window complexity ≈ {Number(result.routing_complexity).toFixed(3)} — low → edge,
              mid → fog, high → cloud.
            </div>
          )}
        </div>
      )}

      <section className="layer-legend" aria-label="Layer color key">
        <span className="layer-legend__item">
          <span className="swatch swatch--edge" /> Edge
        </span>
        <span className="layer-legend__item">
          <span className="swatch swatch--fog" /> Fog
        </span>
        <span className="layer-legend__item">
          <span className="swatch swatch--cloud" /> Cloud
        </span>
      </section>

      <section className="metrics">
        <div className="metric-card">
          <div className="metric-card__k">Selected layer</div>
          <div className="metric-card__v">
            {selectedLayer ? (
              <span className={`layer-pill ${layerVisual?.pillClass ?? ""}`}>{layerVisual?.label ?? selectedLayer}</span>
            ) : (
              "—"
            )}
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-card__k">Prediction</div>
          <div className="metric-card__v metric-card__v--prediction">
            {predictionLabel ? (
              <>
                <span className={result.prediction === 1 ? "tag tag--risk" : "tag tag--ok"}>{predictionLabel}</span>
                <span className="metric-card__sub">class {result.prediction}</span>
              </>
            ) : (
              "—"
            )}
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-card__k">Confidence</div>
          <div className="metric-card__v">{result ? Number(result.confidence).toFixed(4) : "—"}</div>
        </div>
        <div className="metric-card">
          <div className="metric-card__k">Latency (end-to-end)</div>
          <div className="metric-card__v">{result ? `${result.latency_ms} ms` : "—"}</div>
        </div>
      </section>

      {result?.hops && result.hops.length > 1 && (
        <section className="cascade-hops">
          <div className="cascade-hops__title">Cascade path</div>
          <div className="cascade-hops__row">
            {result.hops.map((hop, i) => (
              <span key={`${hop.layer}-${i}`} className="cascade-hop">
                <span className={`layer-pill layer-pill--sm ${LAYER_THEME[hop.layer]?.pillClass ?? ""}`}>
                  {LAYER_THEME[hop.layer]?.label ?? hop.layer}
                </span>
                {i < result.hops.length - 1 ? <span className="cascade-arrow">→</span> : null}
              </span>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

export default App;
