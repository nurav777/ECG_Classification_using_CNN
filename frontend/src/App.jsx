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
  const [simulationEnabled, setSimulationEnabled] = useState(false);
  const [simulationMode, setSimulationMode] = useState("noisy");
  const [simulationIntensity, setSimulationIntensity] = useState(0.5);
  const [maxLatencyMs, setMaxLatencyMs] = useState("");
  const [samples, setSamples] = useState([]);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("connecting...");
  const lastPredictRef = useRef(0);
  /** Bumps whenever stream /predict controls change so late responses are ignored. */
  const predictGenRef = useRef(0);
  /** Increments on every /predict so only the latest in-flight response may update the UI (same effect, multiple POSTs). */
  const fetchSeqRef = useRef(0);

  useEffect(() => {
    predictGenRef.current += 1;
    const requestGeneration = predictGenRef.current;
    fetchSeqRef.current = 0;
    setResult(null);
    lastPredictRef.current = 0;

    const abortController = new AbortController();
    const socket = new WebSocket(WS_BASE);
    socket.onopen = () => setStatus("streaming");
    socket.onclose = () => setStatus("disconnected");
    socket.onerror = () => setStatus("error");
    socket.onmessage = async (event) => {
      const data = JSON.parse(event.data);
      if (predictGenRef.current !== requestGeneration) {
        return;
      }
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
      const fetchId = ++fetchSeqRef.current;
      try {
        const maxLat = maxLatencyMs === "" ? null : Number(maxLatencyMs);
        const response = await fetch(`${API_BASE}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: abortController.signal,
          body: JSON.stringify({
            signal: windowSignal,
            mode,
            strategy,
            preference,
            confidence_threshold: Number(threshold),
            simulation_enabled: simulationEnabled,
            simulation_mode: simulationMode,
            simulation_intensity: Number(simulationIntensity),
            max_latency_ms: Number.isFinite(maxLat) ? maxLat : null
          })
        });
        if (!response.ok) {
          throw new Error(`Request failed: ${response.status}`);
        }
        const payload = await response.json();
        if (predictGenRef.current !== requestGeneration) {
          return;
        }
        if (fetchId !== fetchSeqRef.current) {
          return;
        }
        const expectedSim = simulationEnabled ? simulationMode : "none";
        if (payload.simulation_mode !== expectedSim) {
          return;
        }
        setResult(payload);
      } catch (error) {
        if (error.name === "AbortError") {
          return;
        }
        if (predictGenRef.current !== requestGeneration) {
          return;
        }
        setStatus(`predict error: ${error.message}`);
      }
    };

    return () => {
      abortController.abort();
      socket.close();
    };
  }, [mode, strategy, preference, threshold, simulationEnabled, simulationMode, simulationIntensity, maxLatencyMs]);

  // Always use the same length as /predict (187): full `samples` is ~256 points, but
  // `predicted_input` is 187 — mixing them made the chart rescale and look "broken"
  // when a prediction arrived after changing simulation mode.
  const chartSeries = useMemo(() => {
    if (simulationEnabled && result?.predicted_input?.length) {
      return result.predicted_input;
    }
    if (samples.length >= DEFAULT_WINDOW) {
      return samples.slice(-DEFAULT_WINDOW);
    }
    return samples;
  }, [samples, result?.predicted_input, simulationEnabled]);

  const chartData = useMemo(
    () => chartSeries.map((value, idx) => ({ i: idx, value })),
    [chartSeries]
  );

  const chartRemountKey =
    simulationEnabled && result?.predicted_input?.length
      ? `model-${result.simulation_mode}-${result.prediction}-${result.confidence}`
      : "stream-window";

  // Simulated inputs are min–maxed to [0, 1] on the gateway (MIT-BIH scale); keep Y fixed so Recharts stays stable.
  const yDomain = useMemo(() => {
    if (simulationEnabled && result?.predicted_input?.length) {
      return [0, 1];
    }
    return [-1.2, 1.2];
  }, [simulationEnabled, result?.predicted_input]);

  const selectedLayer = result?.layer ?? result?.selected_layer;
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

        <label className="control-span-2">
          <span className="control-inline">
            <input
              type="checkbox"
              checked={simulationEnabled}
              onChange={(e) => setSimulationEnabled(e.target.checked)}
            />
            Input simulation (affects signal only)
          </span>
          <select
            value={simulationMode}
            onChange={(e) => setSimulationMode(e.target.value)}
            disabled={!simulationEnabled}
          >
            <option value="normal">Normal (clean copy)</option>
            <option value="noisy">Noisy</option>
            <option value="abnormal">Abnormal (spikes / bursts)</option>
            <option value="drift">Baseline drift</option>
          </select>
        </label>

        <label>
          Simulation intensity
          <input
            type="number"
            min="0"
            max="1"
            step="0.05"
            value={simulationIntensity}
            onChange={(e) => setSimulationIntensity(e.target.value)}
            disabled={!simulationEnabled}
          />
        </label>

        <label>
          Target latency budget (ms) → layer
          <input
            type="number"
            min="0"
            step="1"
            placeholder="e.g. 400 = fog, 1000 = cloud"
            value={maxLatencyMs}
            onChange={(e) => setMaxLatencyMs(e.target.value)}
            disabled={strategy !== "rule_based"}
          />
          <span className="field-hint">
            Lower = favor speed (≤350 → edge). Mid (~400) → fog. High (~1000+) → cloud. Leave empty to use preference /
            signal complexity instead.
          </span>
        </label>
      </section>

      <section className="chart">
        <ResponsiveContainer width="100%" height={280}>
          <LineChart key={chartRemountKey} data={chartData}>
            <XAxis dataKey="i" hide />
            <YAxis domain={yDomain} allowDataOverflow={false} width={48} />
            <Tooltip />
            <Line
              type="linear"
              dataKey="value"
              stroke="#00c2ff"
              dot={false}
              strokeWidth={2}
              isAnimationActive={false}
            />
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
          {result.escalated ? <div className="layer-hero__escalation">Cascade escalated beyond the first layer.</div> : null}
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

      <section className="metrics metrics--wide">
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
          <div className="metric-card__k">Model latency (Σ hops)</div>
          <div className="metric-card__v">{result?.model_latency_ms != null ? `${result.model_latency_ms} ms` : "—"}</div>
        </div>
        <div className="metric-card">
          <div className="metric-card__k">Simulated delay (Σ hops)</div>
          <div className="metric-card__v">
            {result?.simulated_latency_ms != null ? `${result.simulated_latency_ms} ms` : "—"}
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-card__k">Service total (Σ hops)</div>
          <div className="metric-card__v">{result?.total_latency_ms != null ? `${result.total_latency_ms} ms` : "—"}</div>
        </div>
        <div className="metric-card">
          <div className="metric-card__k">Gateway wall time</div>
          <div className="metric-card__v">
            {result?.request_latency_ms != null ? `${result.request_latency_ms} ms` : "—"}
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-card__k">Input simulation</div>
          <div className="metric-card__v">{result?.simulation_mode ?? "—"}</div>
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
