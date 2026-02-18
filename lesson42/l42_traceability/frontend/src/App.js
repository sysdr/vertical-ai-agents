import React, { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';

const API = 'http://localhost:8042';

// ── Palette ───────────────────────────────────────────────────────────────────
const colors = {
  bg: '#0f1117',
  card: '#1a1d2e',
  border: '#2a2d3e',
  primary: '#4F9CF9',
  success: '#4ade80',
  warn: '#f59e0b',
  danger: '#f87171',
  muted: '#6b7280',
  text: '#e2e8f0',
  subtext: '#94a3b8',
};

const AGENT_COLORS = {
  planner: '#818cf8',
  retriever: '#34d399',
  validator: '#fbbf24',
  synthesizer: '#60a5fa',
};

// ── Utilities ─────────────────────────────────────────────────────────────────
function riskColor(score) {
  if (score >= 0.7) return colors.danger;
  if (score >= 0.4) return colors.warn;
  return colors.success;
}

function statusBadge(status) {
  const map = { complete: colors.success, failed: colors.danger, active: colors.primary };
  return map[status] || colors.muted;
}

// ── Components ────────────────────────────────────────────────────────────────
function Card({ title, children, style }) {
  return (
    <div style={{
      background: colors.card, border: `1px solid ${colors.border}`,
      borderRadius: 12, padding: '20px', marginBottom: 16, ...style
    }}>
      {title && (
        <div style={{ fontSize: 13, fontWeight: 700, color: colors.subtext, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 14 }}>
          {title}
        </div>
      )}
      {children}
    </div>
  );
}

function Badge({ text, color }) {
  return (
    <span style={{
      background: color + '22', color, border: `1px solid ${color}44`,
      borderRadius: 999, padding: '2px 10px', fontSize: 11, fontWeight: 700,
    }}>{text}</span>
  );
}

// ── Query Panel ───────────────────────────────────────────────────────────────
function QueryPanel({ onResult }) {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const EXAMPLES = [
    'Explain how transformer attention works',
    'What is RAG and why is it used?',
    'How do vector databases enable semantic search?',
    'What compliance frameworks apply to enterprise AI?',
  ];

  const submit = async () => {
    if (!query.trim()) return;
    setLoading(true); setError('');
    try {
      const res = await fetch(`${API}/query`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      const data = await res.json();
      onResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card title="🔍 Query the Agentic RAG Pipeline">
      <div style={{ display: 'flex', gap: 10, marginBottom: 10 }}>
        <input
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && submit()}
          placeholder="Enter query..."
          style={{
            flex: 1, background: '#252836', border: `1px solid ${colors.border}`,
            borderRadius: 8, padding: '10px 14px', color: colors.text, fontSize: 14,
            outline: 'none',
          }}
        />
        <button
          onClick={submit}
          disabled={loading}
          style={{
            background: loading ? colors.muted : colors.primary,
            color: '#fff', border: 'none', borderRadius: 8,
            padding: '10px 22px', fontSize: 14, fontWeight: 700, cursor: 'pointer',
          }}
        >
          {loading ? '…' : 'Run'}
        </button>
      </div>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {EXAMPLES.map(ex => (
          <button key={ex} onClick={() => setQuery(ex)} style={{
            background: '#252836', border: `1px solid ${colors.border}`,
            borderRadius: 6, padding: '4px 10px', fontSize: 11, color: colors.subtext,
            cursor: 'pointer',
          }}>{ex}</button>
        ))}
      </div>
      {error && <div style={{ color: colors.danger, fontSize: 12, marginTop: 8 }}>{error}</div>}
    </Card>
  );
}

// ── Result Panel ──────────────────────────────────────────────────────────────
function ResultPanel({ result }) {
  if (!result) return null;
  return (
    <Card title="✅ Pipeline Result">
      <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
        <Badge text={result.status?.toUpperCase()} color={statusBadge(result.status)} />
        {result.intent && <Badge text={`Intent: ${result.intent}`} color={colors.primary} />}
        {result.verdict && <Badge text={`Verdict: ${result.verdict}`} color={riskColor(result.risk_score || 0)} />}
        {result.risk_score != null && (
          <Badge text={`Risk: ${(result.risk_score * 100).toFixed(0)}%`} color={riskColor(result.risk_score)} />
        )}
        {result.confidence_delta != null && (
          <Badge text={`Δconf: ${result.confidence_delta >= 0 ? '+' : ''}${result.confidence_delta.toFixed(3)}`} color={colors.primary} />
        )}
      </div>

      {result.trace_id && (
        <div style={{ fontSize: 11, color: colors.muted, marginBottom: 10 }}>
          trace_id: <span style={{ color: colors.primary, fontFamily: 'monospace' }}>{result.trace_id}</span>
        </div>
      )}

      {result.status === 'failed' && result.error && (
        <div style={{ background: '#2a1a1a', border: `1px solid ${colors.danger}`, borderRadius: 8, padding: 12, marginBottom: 12, fontSize: 13, color: colors.danger }}>
          <strong>Error:</strong> {result.error}
        </div>
      )}

      {result.reasoning_chain && result.reasoning_chain.length > 0 && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 12, color: colors.subtext, marginBottom: 6, fontWeight: 700 }}>Chain-of-Thought</div>
          {result.reasoning_chain.map((step, i) => (
            <div key={i} style={{
              display: 'flex', gap: 10, marginBottom: 4, fontSize: 12, color: colors.text,
              background: '#252836', borderRadius: 6, padding: '6px 10px',
            }}>
              <span style={{ color: colors.primary, fontWeight: 700 }}>Step {i + 1}</span>
              {step}
            </div>
          ))}
        </div>
      )}

      {result.response && (
        <div style={{
          background: '#252836', borderRadius: 8, padding: '12px 14px',
          fontSize: 14, color: colors.text, lineHeight: 1.6,
        }}>{result.response}</div>
      )}

      {result.citations && result.citations.length > 0 && (
        <div style={{ marginTop: 8, fontSize: 11, color: colors.subtext }}>
          Citations: {result.citations.map((c, i) => (
            <span key={i} style={{ color: colors.primary, marginRight: 8 }}>[{i + 1}] {c}</span>
          ))}
        </div>
      )}
    </Card>
  );
}

// ── Waterfall Panel ───────────────────────────────────────────────────────────
function WaterfallPanel({ traceId }) {
  const [trace, setTrace] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!traceId) return;
    setLoading(true);
    fetch(`${API}/trace/${traceId}`)
      .then(r => r.json())
      .then(d => { setTrace(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, [traceId]);

  if (!traceId) return null;
  if (loading) return <Card title="⏱ Trace Waterfall"><div style={{ color: colors.muted }}>Loading…</div></Card>;
  if (!trace?.spans?.length) return <Card title="⏱ Trace Waterfall"><div style={{ color: colors.muted }}>No spans found.</div></Card>;

  const totalMs = trace.spans.reduce((s, sp) => s + (sp.latency_ms || 0), 0);

  return (
    <Card title="⏱ Trace Waterfall">
      <div style={{ marginBottom: 8, fontSize: 11, color: colors.muted }}>
        Total pipeline: <strong style={{ color: colors.text }}>{totalMs.toFixed(1)}ms</strong>
        &nbsp;·&nbsp;
        {trace.spans.length} spans
        &nbsp;·&nbsp;
        Status: <strong style={{ color: statusBadge(trace.status) }}>{trace.status}</strong>
      </div>
      {trace.spans.map((span, i) => {
        const pct = totalMs > 0 ? (span.latency_ms / totalMs) * 100 : 10;
        const agentColor = AGENT_COLORS[span.agent_name] || colors.primary;
        return (
          <div key={span.span_id} style={{ marginBottom: 10 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3, fontSize: 12 }}>
              <span style={{ color: agentColor, fontWeight: 700, textTransform: 'capitalize' }}>
                {span.agent_name}
              </span>
              <span style={{ color: colors.subtext }}>
                {span.latency_ms}ms
                {span.decision && ` · ${span.decision.substring(0, 50)}`}
              </span>
            </div>
            <div style={{ background: '#252836', borderRadius: 4, height: 10, overflow: 'hidden' }}>
              <div style={{
                width: `${Math.max(pct, 2)}%`, height: '100%',
                background: agentColor, borderRadius: 4,
                opacity: span.status === 'error' ? 0.4 : 1,
              }} />
            </div>
            {span.risk_score > 0 && (
              <div style={{ fontSize: 10, color: riskColor(span.risk_score), marginTop: 2 }}>
                Risk: {(span.risk_score * 100).toFixed(0)}%
                {span.compliance_flags?.length > 0 && ` · flags: ${JSON.parse(span.compliance_flags || '[]').join(', ')}`}
              </div>
            )}
          </div>
        );
      })}
    </Card>
  );
}

// ── Risk Heatmap Panel ────────────────────────────────────────────────────────
function RiskHeatmap() {
  const [data, setData] = useState([]);

  useEffect(() => {
    const load = () => fetch(`${API}/stats/risk-timeline`).then(r => r.json()).then(setData).catch(() => {});
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []);

  const allZero = data.length > 0 && data.every(d => (d.max_risk || 0) === 0 && (d.avg_risk || 0) === 0);
  return (
    <Card title="🌡 Risk Heatmap (24h)">
      {data.length === 0
        ? <div style={{ color: colors.muted, fontSize: 12 }}>Run some queries to populate risk data.</div>
        : allZero
        ? <div style={{ color: colors.muted, fontSize: 12 }}>Run a query to see risk scores (validator/synthesizer report risk).</div>
        : data.map(d => (
          <div key={d.agent_name} style={{ marginBottom: 10 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 3 }}>
              <span style={{ color: AGENT_COLORS[d.agent_name] || colors.primary, fontWeight: 700, textTransform: 'capitalize' }}>
                {d.agent_name}
              </span>
              <span style={{ color: riskColor(d.avg_risk) }}>
                avg {(d.avg_risk * 100).toFixed(1)}% · max {(d.max_risk * 100).toFixed(1)}%
              </span>
            </div>
            <div style={{ background: '#252836', borderRadius: 4, height: 10 }}>
              <div style={{
                width: `${Math.min(d.max_risk * 100, 100)}%`, height: '100%',
                background: riskColor(d.max_risk), borderRadius: 4, opacity: 0.8,
              }} />
            </div>
          </div>
        ))
      }
    </Card>
  );
}

// ── Latency Stats ──────────────────────────────────────────────────────────────
function LatencyStats() {
  const [data, setData] = useState([]);
  useEffect(() => {
    const load = () => fetch(`${API}/stats/latency`).then(r => r.json()).then(setData).catch(() => {});
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []);

  return (
    <Card title="⚡ Agent Latency Stats">
      {data.length === 0
        ? <div style={{ color: colors.muted, fontSize: 12 }}>No latency data yet.</div>
        : (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ color: colors.subtext }}>
                {['Agent', 'Avg ms', 'Min ms', 'Max ms', 'Spans'].map(h => (
                  <th key={h} style={{ textAlign: 'left', paddingBottom: 8, fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.map(d => (
                <tr key={d.agent_name} style={{ borderTop: `1px solid ${colors.border}` }}>
                  <td style={{ color: AGENT_COLORS[d.agent_name] || colors.primary, padding: '6px 0', textTransform: 'capitalize', fontWeight: 700 }}>
                    {d.agent_name}
                  </td>
                  <td style={{ color: colors.text }}>{d.avg_ms}</td>
                  <td style={{ color: colors.success }}>{d.min_ms}</td>
                  <td style={{ color: colors.warn }}>{d.max_ms}</td>
                  <td style={{ color: colors.subtext }}>{d.total_spans}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )
      }
    </Card>
  );
}

// ── Live Tail ─────────────────────────────────────────────────────────────────
function LiveTail() {
  const [spans, setSpans] = useState([]);
  const ref = useRef(null);

  useEffect(() => {
    const es = new EventSource(`${API}/live-tail`);
    es.onmessage = e => {
      try {
        const span = JSON.parse(e.data);
        setSpans(prev => [span, ...prev].slice(0, 40));
      } catch {}
    };
    return () => es.close();
  }, []);

  useEffect(() => {
    if (ref.current) ref.current.scrollTop = 0;
  }, [spans]);

  return (
    <Card title="📡 Live Tail" style={{ maxHeight: 320, overflow: 'hidden' }}>
      <div ref={ref} style={{ maxHeight: 260, overflowY: 'auto' }}>
        {spans.length === 0
          ? <div style={{ color: colors.muted, fontSize: 12 }}>Waiting for spans…</div>
          : spans.map((s, i) => (
            <div key={s.span_id + i} style={{
              display: 'flex', gap: 10, padding: '5px 0',
              borderBottom: `1px solid ${colors.border}`, fontSize: 11, color: colors.text,
            }}>
              <span style={{ color: AGENT_COLORS[s.agent_name] || colors.primary, fontWeight: 700, minWidth: 80, textTransform: 'capitalize' }}>
                {s.agent_name}
              </span>
              <span style={{ color: statusBadge(s.status), minWidth: 50 }}>{s.status}</span>
              <span style={{ color: colors.subtext, minWidth: 60 }}>{s.latency_ms}ms</span>
              <span style={{ color: colors.muted, flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {s.decision || s.trace_id?.slice(0, 20)}
              </span>
            </div>
          ))
        }
      </div>
    </Card>
  );
}

// ── Trace History ─────────────────────────────────────────────────────────────
function TraceHistory({ onSelect }) {
  const [traces, setTraces] = useState([]);

  const refresh = useCallback(() => {
    fetch(`${API}/traces?limit=20`).then(r => r.json()).then(setTraces).catch(() => {});
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, [refresh]);

  return (
    <Card title="📋 Trace History">
      <div style={{ maxHeight: 200, overflowY: 'auto' }}>
        {traces.length === 0
          ? <div style={{ color: colors.muted, fontSize: 12 }}>No traces yet.</div>
          : traces.map(t => (
            <div
              key={t.trace_id}
              onClick={() => onSelect(t.trace_id)}
              style={{
                display: 'flex', justifyContent: 'space-between', padding: '8px 0',
                borderBottom: `1px solid ${colors.border}`, cursor: 'pointer',
                fontSize: 12,
              }}
            >
              <span style={{ color: colors.primary, fontFamily: 'monospace', fontSize: 11 }}>
                {t.trace_id?.slice(0, 22)}…
              </span>
              <div style={{ display: 'flex', gap: 8 }}>
                <Badge text={t.status} color={statusBadge(t.status)} />
                {t.confidence_delta != null && t.confidence_delta !== 0 && (
                  <Badge
                    text={`Δ${t.confidence_delta >= 0 ? '+' : ''}${t.confidence_delta.toFixed(2)}`}
                    color={t.confidence_delta >= 0 ? colors.success : colors.warn}
                  />
                )}
              </div>
            </div>
          ))
        }
      </div>
    </Card>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [result, setResult] = useState(null);
  const [selectedTrace, setSelectedTrace] = useState(null);

  const handleResult = useCallback(r => {
    setResult(r);
    setSelectedTrace(r.trace_id);
  }, []);

  return (
    <div className="dashboard" style={{ minHeight: '100vh', color: colors.text, padding: '24px' }}>
      {/* Header */}
      <div style={{
        textAlign: 'center', marginBottom: 28, padding: '18px',
        background: colors.card, borderRadius: 14,
        border: `1px solid ${colors.border}`,
      }}>
        <div style={{ fontSize: 22, fontWeight: 800, color: colors.primary }}>
          L42 · Traceability Layer
        </div>
        <div style={{ fontSize: 13, color: colors.subtext, marginTop: 4 }}>
          Full audit trail: Planner intent → Validator risk → Synthesizer response
        </div>
      </div>

      {/* Query + Result */}
      <QueryPanel onResult={handleResult} />
      <ResultPanel result={result} />

      {/* Waterfall + History */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <WaterfallPanel traceId={selectedTrace} />
        <TraceHistory onSelect={setSelectedTrace} />
      </div>

      {/* Stats Row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
        <RiskHeatmap />
        <LatencyStats />
        <LiveTail />
      </div>
    </div>
  );
}
