import { useState, useEffect, useRef } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from "recharts";

// --- Simulate realistic AAPL-like stock data ---
function generateStockData(n = 300) {
  const data = [];
  let price = 150;
  for (let i = 0; i < n; i++) {
    const trend = 0.03;
    const noise = (Math.random() - 0.48) * 3.5;
    const cycle = Math.sin(i / 30) * 2;
    price = Math.max(80, price + trend + noise + cycle * 0.1);
    data.push(parseFloat(price.toFixed(2)));
  }
  return data;
}

// MinMax scale
function minMaxScale(arr) {
  const min = Math.min(...arr);
  const max = Math.max(...arr);
  return { scaled: arr.map(v => (v - min) / (max - min)), min, max };
}
function inverseScale(arr, min, max) {
  return arr.map(v => parseFloat((v * (max - min) + min).toFixed(2)));
}

// Create windows
function createWindows(data, seqLen = 60) {
  const X = [], y = [];
  for (let i = 0; i < data.length - seqLen; i++) {
    X.push(data.slice(i, i + seqLen));
    y.push(data[i + seqLen]);
  }
  return { X, y };
}

// Simulate LSTM predictions (smooth + slightly lagged)
function simulateLSTMPredictions(actual, noise = 0.025) {
  return actual.map((v, i) => {
    if (i === 0) return v;
    const lag = actual[Math.max(0, i - 2)] * 0.3 + actual[Math.max(0, i - 1)] * 0.4 + v * 0.3;
    const n = (Math.random() - 0.5) * noise * (Math.max(...actual) - Math.min(...actual));
    return parseFloat((lag + n).toFixed(2));
  });
}

// Simulate training loss curves
function generateLossCurves(epochs = 50) {
  const train = [], val = [];
  let tl = 0.08, vl = 0.10;
  for (let e = 1; e <= epochs; e++) {
    tl = Math.max(0.002, tl * (0.94 + Math.random() * 0.02));
    vl = Math.max(0.003, vl * (0.95 + Math.random() * 0.025));
    train.push({ epoch: e, loss: parseFloat(tl.toFixed(5)) });
    val.push({ epoch: e, loss: parseFloat(vl.toFixed(5)) });
  }
  return train.map((t, i) => ({ epoch: t.epoch, trainLoss: t.loss, valLoss: val[i].loss }));
}

// Future forecast
function generateFutureForecast(lastPrice, n = 30) {
  const preds = [];
  let p = lastPrice;
  for (let i = 0; i < n; i++) {
    p = p + (Math.random() - 0.47) * 2.5 + 0.05;
    preds.push({ day: `+${i + 1}d`, price: parseFloat(p.toFixed(2)) });
  }
  return preds;
}

const STEPS = [
  { id: 1, label: "Data", sub: "Raw prices" },
  { id: 2, label: "Scale", sub: "Normalize [0,1]" },
  { id: 3, label: "Windows", sub: "Seq length=60" },
  { id: 4, label: "Training", sub: "Loss curves" },
  { id: 5, label: "Predict", sub: "Test set" },
  { id: 6, label: "Forecast", sub: "30 days ahead" },
];

const PALETTE = {
  bg: "#0d1117",
  surface: "#161b22",
  border: "#30363d",
  accent: "#58a6ff",
  green: "#3fb950",
  orange: "#e3b341",
  red: "#f85149",
  text: "#e6edf3",
  muted: "#8b949e",
};

export default function LSTMDemo() {
  const [step, setStep] = useState(0);
  const [animating, setAnimating] = useState(false);
  const [stockData] = useState(() => generateStockData(300));
  const [lossCurves] = useState(() => generateLossCurves(50));
  const [visibleEpochs, setVisibleEpochs] = useState(50);
  const animRef = useRef(null);

  const { scaled, min, max } = minMaxScale(stockData);
  const trainEnd = Math.floor(stockData.length * 0.70);
  const valEnd = Math.floor(stockData.length * 0.85);
  const testActual = stockData.slice(valEnd);
  const testPreds = simulateLSTMPredictions(testActual, 0.03);
  const futureForecast = generateFutureForecast(testActual[testActual.length - 1], 30);

  const rmse = Math.sqrt(
    testActual.reduce((s, v, i) => s + (v - testPreds[i]) ** 2, 0) / testActual.length
  ).toFixed(2);
  const mae = (
    testActual.reduce((s, v, i) => s + Math.abs(v - testPreds[i]), 0) / testActual.length
  ).toFixed(2);

  // Animate training
  useEffect(() => {
    if (step === 3) {
      setVisibleEpochs(1);
      let e = 1;
      animRef.current = setInterval(() => {
        e++;
        setVisibleEpochs(e);
        if (e >= 50) clearInterval(animRef.current);
      }, 40);
    }
    return () => clearInterval(animRef.current);
  }, [step]);

  const rawChartData = stockData.map((v, i) => ({
    day: i,
    price: v,
    segment: i < trainEnd ? "train" : i < valEnd ? "val" : "test",
  }));

  const scaledChartData = scaled.map((v, i) => ({ day: i, value: parseFloat(v.toFixed(4)) }));

  const testChartData = testActual.map((v, i) => ({
    day: i,
    actual: v,
    predicted: testPreds[i],
  }));

  const lossData = lossCurves.slice(0, visibleEpochs);

  const stepContent = [
    // Step 0: Welcome
    <div key="welcome" style={{ textAlign: "center", padding: "40px 0" }}>
      <div style={{ fontSize: 13, color: PALETTE.muted, letterSpacing: 2, marginBottom: 12, textTransform: "uppercase" }}>
        Live Demo
      </div>
      <div style={{ fontSize: 28, fontWeight: 700, color: PALETTE.text, marginBottom: 8 }}>
        LSTM Stock Prediction
      </div>
      <div style={{ fontSize: 15, color: PALETTE.muted, maxWidth: 420, margin: "0 auto 32px" }}>
        Walk through all 6 pipeline stages interactively — from raw price data to a 30-day forecast.
      </div>
      <div style={{ display: "flex", gap: 24, justifyContent: "center", marginBottom: 32 }}>
        {[
          { label: "Ticker", val: "AAPL (simulated)" },
          { label: "Data points", val: "300 days" },
          { label: "Seq length", val: "60 days" },
          { label: "Epochs", val: "50" },
        ].map(s => (
          <div key={s.label} style={{ background: PALETTE.surface, border: `1px solid ${PALETTE.border}`, borderRadius: 10, padding: "12px 20px", minWidth: 110 }}>
            <div style={{ fontSize: 11, color: PALETTE.muted, marginBottom: 4, textTransform: "uppercase", letterSpacing: 1 }}>{s.label}</div>
            <div style={{ fontSize: 15, fontWeight: 600, color: PALETTE.accent }}>{s.val}</div>
          </div>
        ))}
      </div>
      <button onClick={() => setStep(1)} style={btnStyle(PALETTE.accent)}>
        Start walkthrough →
      </button>
    </div>,

    // Step 1: Raw data
    <div key="step1">
      <StepHeader title="Raw stock prices" sub="300 days of simulated AAPL-like closing prices, split into train / val / test" />
      <div style={{ height: 260 }}>
        <ResponsiveContainer>
          <AreaChart data={rawChartData} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
            <defs>
              <linearGradient id="trainGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={PALETTE.accent} stopOpacity={0.25} />
                <stop offset="95%" stopColor={PALETTE.accent} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke={PALETTE.border} />
            <XAxis dataKey="day" stroke={PALETTE.muted} tick={{ fontSize: 11 }} />
            <YAxis stroke={PALETTE.muted} tick={{ fontSize: 11 }} domain={["auto", "auto"]} />
            <Tooltip contentStyle={{ background: PALETTE.surface, border: `1px solid ${PALETTE.border}`, borderRadius: 8, color: PALETTE.text, fontSize: 12 }} />
            <Area type="monotone" dataKey="price" stroke={PALETTE.accent} fill="url(#trainGrad)" strokeWidth={1.5} dot={false} name="Close price" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div style={{ display: "flex", gap: 12, marginTop: 12 }}>
        {[
          { label: `Train (0–${trainEnd})`, color: PALETTE.accent, note: "70%" },
          { label: `Val (${trainEnd}–${valEnd})`, color: PALETTE.orange, note: "15%" },
          { label: `Test (${valEnd}–300)`, color: PALETTE.green, note: "15%" },
        ].map(s => (
          <div key={s.label} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: PALETTE.muted }}>
            <div style={{ width: 10, height: 10, borderRadius: "50%", background: s.color }} />
            {s.label} <span style={{ color: s.color }}>{s.note}</span>
          </div>
        ))}
      </div>
      <InfoBox>The <code>yf.download("AAPL")</code> call gives us a DataFrame. We keep only the <code>Close</code> column and split chronologically — never shuffle time series data!</InfoBox>
    </div>,

    // Step 2: Scaling
    <div key="step2">
      <StepHeader title="Normalize prices → [0, 1]" sub={`MinMaxScaler maps the range $${min.toFixed(0)} – $${max.toFixed(0)} onto 0 to 1`} />
      <div style={{ height: 240 }}>
        <ResponsiveContainer>
          <LineChart data={scaledChartData} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={PALETTE.border} />
            <XAxis dataKey="day" stroke={PALETTE.muted} tick={{ fontSize: 11 }} />
            <YAxis stroke={PALETTE.muted} tick={{ fontSize: 11 }} domain={[0, 1]} />
            <Tooltip contentStyle={{ background: PALETTE.surface, border: `1px solid ${PALETTE.border}`, borderRadius: 8, color: PALETTE.text, fontSize: 12 }} />
            <Line type="monotone" dataKey="value" stroke={PALETTE.orange} strokeWidth={1.5} dot={false} name="Scaled price" />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div style={{ display: "flex", gap: 16, marginTop: 12 }}>
        <StatPill label="Original min" val={`$${min.toFixed(2)}`} color={PALETTE.red} />
        <StatPill label="Original max" val={`$${max.toFixed(2)}`} color={PALETTE.green} />
        <StatPill label="Scaled range" val="0.0 → 1.0" color={PALETTE.orange} />
      </div>
      <InfoBox>
        <code>scaler = MinMaxScaler()</code><br />
        <code>scaled = scaler.fit_transform(df.values)</code><br />
        LSTMs use sigmoid/tanh activations (output ≈ 0–1), so unscaled prices cause slow convergence or NaN gradients.
      </InfoBox>
    </div>,

    // Step 3: Windows
    <div key="step3">
      <StepHeader title="Sliding window sequences" sub="Each sample = 60 consecutive days → predict day 61" />
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 16 }}>
        {Array.from({ length: 8 }).map((_, wi) => (
          <div key={wi} style={{ background: PALETTE.surface, border: `1px solid ${PALETTE.border}`, borderRadius: 8, padding: "10px 14px", fontSize: 11, color: PALETTE.muted }}>
            <div style={{ color: PALETTE.accent, fontWeight: 600, marginBottom: 4 }}>Window {wi + 1}</div>
            <div>Days {wi}–{wi + 59} → predict <span style={{ color: PALETTE.green }}>day {wi + 60}</span></div>
            <div style={{ display: "flex", gap: 2, marginTop: 6 }}>
              {Array.from({ length: 12 }).map((_, j) => (
                <div key={j} style={{ width: 6, height: 16, borderRadius: 2, background: j < 11 ? PALETTE.accent : PALETTE.green, opacity: j < 11 ? 0.6 : 1 }} />
              ))}
              <span style={{ marginLeft: 4, color: PALETTE.green }}>▶</span>
            </div>
          </div>
        ))}
        <div style={{ ...statPillBase, alignSelf: "center", padding: "10px 14px" }}>
          <div style={{ color: PALETTE.muted, fontSize: 11 }}>Total windows</div>
          <div style={{ color: PALETTE.text, fontWeight: 600, fontSize: 16 }}>{300 - 60}</div>
        </div>
      </div>
      <InfoBox>
        <code>X shape: (240, 60, 1)</code> &nbsp;|&nbsp; <code>y shape: (240, 1)</code><br />
        The <code>1</code> in <code>(240, 60, 1)</code> is <code>input_size</code> — we only use close price. Adding RSI, volume etc. would make it <code>(240, 60, 5)</code>.
      </InfoBox>
    </div>,

    // Step 4: Training
    <div key="step4">
      <StepHeader title="Model training — 50 epochs" sub="Training and validation MSE loss per epoch" />
      <div style={{ height: 240 }}>
        <ResponsiveContainer>
          <LineChart data={lossData} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={PALETTE.border} />
            <XAxis dataKey="epoch" stroke={PALETTE.muted} tick={{ fontSize: 11 }} label={{ value: "Epoch", position: "insideBottom", offset: -2, fill: PALETTE.muted, fontSize: 11 }} />
            <YAxis stroke={PALETTE.muted} tick={{ fontSize: 11 }} tickFormatter={v => v.toFixed(4)} />
            <Tooltip contentStyle={{ background: PALETTE.surface, border: `1px solid ${PALETTE.border}`, borderRadius: 8, color: PALETTE.text, fontSize: 12 }} formatter={v => v.toFixed(5)} />
            <Legend wrapperStyle={{ fontSize: 12, color: PALETTE.muted }} />
            <Line type="monotone" dataKey="trainLoss" stroke={PALETTE.accent} strokeWidth={2} dot={false} name="Train loss" isAnimationActive={false} />
            <Line type="monotone" dataKey="valLoss" stroke={PALETTE.orange} strokeWidth={2} dot={false} name="Val loss" isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div style={{ display: "flex", gap: 12, marginTop: 10 }}>
        <StatPill label="Final train loss" val={lossCurves[49].trainLoss.toFixed(5)} color={PALETTE.accent} />
        <StatPill label="Final val loss" val={lossCurves[49].valLoss.toFixed(5)} color={PALETTE.orange} />
        <StatPill label="Epochs" val="50" color={PALETTE.muted} />
      </div>
      <InfoBox>
        Both curves decrease and converge — no overfitting. The model saved the weights at the lowest val loss epoch using <code>torch.save(model.state_dict(), "best_model.pth")</code>.
      </InfoBox>
    </div>,

    // Step 5: Predictions
    <div key="step5">
      <StepHeader title="Test set — actual vs predicted" sub={`${testActual.length} unseen days. Prices inverse-transformed back to USD`} />
      <div style={{ height: 240 }}>
        <ResponsiveContainer>
          <LineChart data={testChartData} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={PALETTE.border} />
            <XAxis dataKey="day" stroke={PALETTE.muted} tick={{ fontSize: 11 }} />
            <YAxis stroke={PALETTE.muted} tick={{ fontSize: 11 }} domain={["auto", "auto"]} />
            <Tooltip contentStyle={{ background: PALETTE.surface, border: `1px solid ${PALETTE.border}`, borderRadius: 8, color: PALETTE.text, fontSize: 12 }} formatter={v => `$${v}`} />
            <Legend wrapperStyle={{ fontSize: 12, color: PALETTE.muted }} />
            <Line type="monotone" dataKey="actual" stroke={PALETTE.accent} strokeWidth={2} dot={false} name="Actual price" />
            <Line type="monotone" dataKey="predicted" stroke={PALETTE.red} strokeWidth={1.5} dot={false} strokeDasharray="5 3" name="Predicted price" />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div style={{ display: "flex", gap: 12, marginTop: 10 }}>
        <StatPill label="RMSE" val={`$${rmse}`} color={PALETTE.orange} />
        <StatPill label="MAE" val={`$${mae}`} color={PALETTE.green} />
        <StatPill label="Test days" val={testActual.length} color={PALETTE.muted} />
      </div>
      <InfoBox>
        RMSE = Root Mean Squared Error (penalizes large errors). MAE = Mean Absolute Error (average dollar-off). Lower = better. For stock prices, an MAE ≈ $2–5 on a $150 stock (~1–3%) is a solid baseline.
      </InfoBox>
    </div>,

    // Step 6: Future forecast
    <div key="step6">
      <StepHeader title="30-day future forecast" sub="The model rolls its own predictions forward as new inputs (autoregressive)" />
      <div style={{ height: 240 }}>
        <ResponsiveContainer>
          <AreaChart data={futureForecast} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
            <defs>
              <linearGradient id="forecastGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={PALETTE.green} stopOpacity={0.3} />
                <stop offset="95%" stopColor={PALETTE.green} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke={PALETTE.border} />
            <XAxis dataKey="day" stroke={PALETTE.muted} tick={{ fontSize: 10 }} interval={4} />
            <YAxis stroke={PALETTE.muted} tick={{ fontSize: 11 }} domain={["auto", "auto"]} />
            <Tooltip contentStyle={{ background: PALETTE.surface, border: `1px solid ${PALETTE.border}`, borderRadius: 8, color: PALETTE.text, fontSize: 12 }} formatter={v => `$${v}`} />
            <Area type="monotone" dataKey="price" stroke={PALETTE.green} fill="url(#forecastGrad)" strokeWidth={2} dot={{ r: 2, fill: PALETTE.green }} name="Forecast" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div style={{ display: "flex", gap: 12, marginTop: 10 }}>
        <StatPill label="Start price" val={`$${futureForecast[0].price}`} color={PALETTE.accent} />
        <StatPill label="Day 30 price" val={`$${futureForecast[29].price}`} color={PALETTE.green} />
        <StatPill label="Δ change" val={`${futureForecast[29].price > futureForecast[0].price ? "+" : ""}${(futureForecast[29].price - futureForecast[0].price).toFixed(2)}`}
          color={futureForecast[29].price > futureForecast[0].price ? PALETTE.green : PALETTE.red} />
      </div>
      <InfoBox>
        Warning: autoregressive forecasting compounds errors. Each predicted price becomes the next input, so uncertainty grows with horizon. Use confidence intervals in production and re-train regularly.
      </InfoBox>
    </div>,
  ];

  return (
    <div style={{ background: PALETTE.bg, color: PALETTE.text, fontFamily: "'JetBrains Mono', 'Fira Code', monospace", minHeight: "100vh", padding: "24px 20px" }}>

      {/* Step pills */}
      <div style={{ display: "flex", gap: 6, marginBottom: 24, flexWrap: "wrap" }}>
        {STEPS.map(s => (
          <button key={s.id} onClick={() => setStep(s.id)}
            style={{
              background: step === s.id ? PALETTE.accent : PALETTE.surface,
              color: step === s.id ? "#000" : PALETTE.muted,
              border: `1px solid ${step === s.id ? PALETTE.accent : PALETTE.border}`,
              borderRadius: 20, padding: "5px 14px", fontSize: 12, cursor: "pointer",
              fontFamily: "inherit", fontWeight: step === s.id ? 700 : 400,
              transition: "all 0.15s",
            }}>
            {s.id}. {s.label}
          </button>
        ))}
      </div>

      {/* Content card */}
      <div style={{ background: PALETTE.surface, border: `1px solid ${PALETTE.border}`, borderRadius: 12, padding: "24px 20px", minHeight: 420 }}>
        {stepContent[step]}
      </div>

      {/* Navigation */}
      {step > 0 && (
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 16 }}>
          <button onClick={() => setStep(s => Math.max(0, s - 1))} style={btnStyle(PALETTE.surface, PALETTE.border, PALETTE.muted)}>
            ← Back
          </button>
          <div style={{ fontSize: 12, color: PALETTE.muted, alignSelf: "center" }}>
            Step {step} of {STEPS.length}
          </div>
          {step < STEPS.length ? (
            <button onClick={() => setStep(s => s + 1)} style={btnStyle(PALETTE.accent)}>
              Next →
            </button>
          ) : (
            <button onClick={() => setStep(0)} style={btnStyle(PALETTE.green)}>
              Restart ↺
            </button>
          )}
        </div>
      )}
    </div>
  );
}

function StepHeader({ title, sub }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 17, fontWeight: 700, color: "#e6edf3", marginBottom: 4 }}>{title}</div>
      <div style={{ fontSize: 12, color: "#8b949e" }}>{sub}</div>
    </div>
  );
}

function InfoBox({ children }) {
  return (
    <div style={{ marginTop: 14, background: "#0d1117", border: "1px solid #30363d", borderLeft: "3px solid #58a6ff", borderRadius: 6, padding: "10px 14px", fontSize: 12, color: "#8b949e", lineHeight: 1.7 }}>
      {children}
    </div>
  );
}

const statPillBase = {
  background: "#0d1117", border: "1px solid #30363d", borderRadius: 8, padding: "8px 12px",
};

function StatPill({ label, val, color }) {
  return (
    <div style={statPillBase}>
      <div style={{ fontSize: 10, color: "#8b949e", marginBottom: 2, textTransform: "uppercase", letterSpacing: 1 }}>{label}</div>
      <div style={{ fontSize: 15, fontWeight: 700, color }}>{val}</div>
    </div>
  );
}

function btnStyle(bg, border = "transparent", color = "#000") {
  return {
    background: bg, color, border: `1px solid ${border}`, borderRadius: 8,
    padding: "8px 20px", fontSize: 13, cursor: "pointer", fontFamily: "inherit", fontWeight: 600,
  };
}
