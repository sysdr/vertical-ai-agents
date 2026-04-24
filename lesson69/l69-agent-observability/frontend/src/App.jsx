import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, BarChart, Bar
} from 'recharts';

const API  = 'http://localhost:8069';
const WS   = 'ws://localhost:8069/ws/traces';

/* ── Design tokens ─────────────────────────────────────────────────────────── */
const T = {
  bg:      '#08080f',
  surface: '#111120',
  card:    '#16162a',
  border:  '#1e1e3a',
  green:   '#00e87a',
  orange:  '#ff7043',
  blue:    '#4db8ff',
  purple:  '#b57bff',
  muted:   '#4a4a6a',
  text:    '#e2e8f0',
  dim:     '#8888aa',
};

/* ── Span colours ──────────────────────────────────────────────────────────── */
const SPAN_COLORS = {
  'agent.run':          '#4db8ff',
  'agent.build_prompt': '#00e87a',
  'agent.llm_call':     '#ff7043',
  'agent.postprocess':  '#b57bff',
};
const spanColor = name => SPAN_COLORS[name] ?? '#888';

/* ── Utilities ─────────────────────────────────────────────────────────────── */
const fmt = {
  ms:   v => v != null ? `${Number(v).toFixed(1)}ms` : '—',
  cost: v => v != null ? `$${Number(v).toFixed(6)}` : '—',
  id:   s => s ? `${s.slice(0, 8)}…` : '—',
  ts:   s => s ? new Date(Number(s) * 1000).toLocaleTimeString() : '—',
};

/* ── CSS-in-JS helper ──────────────────────────────────────────────────────── */
const css = (...parts) => Object.assign({}, ...parts);

/* ══════════════════════════════════════════════════════════════════════════════
   MetricCard
══════════════════════════════════════════════════════════════════════════════ */
function MetricCard({ label, value, sub, accent }) {
  const color = accent || T.green;
  return (
    <div style={css(S.card, { padding: '16px 20px', minWidth: 140 })}>
      <div style={{ fontSize: 11, color: T.dim, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6 }}>
        {label}
      </div>
      <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 22, fontWeight: 700, color }}>
        {value}
      </div>
      {sub && <div style={{ fontSize: 11, color: T.muted, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════════════════
   SpanWaterfall
══════════════════════════════════════════════════════════════════════════════ */
function SpanWaterfall({ spans }) {
  if (!spans || !spans.length) {
    return (
      <div style={{ textAlign: 'center', color: T.muted, padding: 40, fontSize: 13 }}>
        Select a trace to view span waterfall
      </div>
    );
  }

  const startNs = Math.min(...spans.map(s => s.start_time || 0));
  const endNs   = Math.max(...spans.map(s => s.end_time   || 0));
  const totalNs = Math.max(endNs - startNs, 1);

  return (
    <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 11 }}>
      {spans.map(span => {
        const offsetPct = ((span.start_time - startNs) / totalNs) * 100;
        const widthPct  = Math.max(((span.end_time - span.start_time) / totalNs) * 100, 0.5);
        const color     = spanColor(span.name);
        const attrs     = span.attributes || {};

        return (
          <div key={span.span_id} style={{ marginBottom: 12 }}>
            {/* Label row */}
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4, alignItems: 'center' }}>
              <span style={{ color, fontWeight: 700, fontSize: 11 }}>{span.name}</span>
              <span style={{ color: T.dim, fontSize: 10 }}>{fmt.ms(span.duration_ms)}</span>
            </div>
            {/* Bar */}
            <div style={{ height: 18, background: T.surface, borderRadius: 3, position: 'relative', overflow: 'hidden' }}>
              <div style={{
                position: 'absolute',
                left: `${offsetPct}%`,
                width: `${widthPct}%`,
                height: '100%',
                background: `${color}cc`,
                borderRadius: 3,
                transition: 'width 0.3s ease',
              }}/>
            </div>
            {/* Attributes */}
            {(attrs['llm.cost_usd'] || attrs['llm.input_tokens']) && (
              <div style={{ display: 'flex', gap: 12, marginTop: 4 }}>
                {attrs['llm.input_tokens'] && (
                  <span style={{ color: T.dim, fontSize: 9 }}>
                    in:{attrs['llm.input_tokens']} out:{attrs['llm.output_tokens']}
                  </span>
                )}
                {attrs['llm.cost_usd'] && (
                  <span style={{ color: T.orange, fontSize: 9 }}>
                    {fmt.cost(attrs['llm.cost_usd'])}
                  </span>
                )}
                {attrs['llm.latency_ms'] && (
                  <span style={{ color: T.blue, fontSize: 9 }}>
                    {fmt.ms(attrs['llm.latency_ms'])}
                  </span>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════════════════
   CostChart
══════════════════════════════════════════════════════════════════════════════ */
function CostChart({ data }) {
  const formatted = (data || []).map(d => ({
    t: new Date(Number(d.t) * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    cost: Number((d.cost * 1000).toFixed(4)), // display in milli-dollars
    reqs: d.reqs,
  }));

  return (
    <ResponsiveContainer width="100%" height={120}>
      <AreaChart data={formatted} margin={{ top: 5, right: 5, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="costGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor={T.orange} stopOpacity={0.4}/>
            <stop offset="95%" stopColor={T.orange} stopOpacity={0}/>
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke={T.border}/>
        <XAxis dataKey="t" tick={{ fill: T.dim, fontSize: 9, fontFamily: "'Space Mono'" }} tickLine={false}/>
        <YAxis tick={{ fill: T.dim, fontSize: 9, fontFamily: "'Space Mono'" }} tickLine={false} axisLine={false}/>
        <Tooltip
          contentStyle={{ background: T.card, border: `1px solid ${T.border}`, borderRadius: 6, fontSize: 11, fontFamily: "'Space Mono'" }}
          labelStyle={{ color: T.text }}
          formatter={v => [`${v}m¢`, 'cost']}
        />
        <Area type="monotone" dataKey="cost" stroke={T.orange} strokeWidth={2} fill="url(#costGrad)"/>
      </AreaChart>
    </ResponsiveContainer>
  );
}

/* ══════════════════════════════════════════════════════════════════════════════
   LatencyChart
══════════════════════════════════════════════════════════════════════════════ */
function LatencyBar({ p50, p95, p99 }) {
  const data = [
    { label: 'P50', value: p50, fill: T.green   },
    { label: 'P95', value: p95, fill: T.orange  },
    { label: 'P99', value: p99, fill: '#ff3d5a' },
  ];
  return (
    <ResponsiveContainer width="100%" height={80}>
      <BarChart data={data} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={T.border}/>
        <XAxis dataKey="label" tick={{ fill: T.dim, fontSize: 9, fontFamily: "'Space Mono'" }} tickLine={false}/>
        <YAxis tick={{ fill: T.dim, fontSize: 9, fontFamily: "'Space Mono'" }} tickLine={false} axisLine={false}/>
        <Tooltip
          contentStyle={{ background: T.card, border: `1px solid ${T.border}`, borderRadius: 6, fontSize: 11, fontFamily: "'Space Mono'" }}
          formatter={v => [`${v}ms`]}
        />
        <Bar dataKey="value" radius={[3,3,0,0]}>
          {data.map((d, i) => (
            <rect key={i} fill={d.fill}/>
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/* ══════════════════════════════════════════════════════════════════════════════
   LiveFeed
══════════════════════════════════════════════════════════════════════════════ */
function LiveFeed({ events }) {
  return (
    <div style={{ height: 120, overflowY: 'auto', fontFamily: "'Space Mono',monospace", fontSize: 10 }}>
      {events.length === 0 && (
        <div style={{ color: T.muted, textAlign: 'center', paddingTop: 20 }}>
          Waiting for events…
        </div>
      )}
      {[...events].reverse().map((ev, i) => (
        <div key={i} style={{
          display: 'flex', gap: 8, padding: '3px 0',
          borderBottom: `1px solid ${T.border}`,
          alignItems: 'center',
        }}>
          <span style={{
            width: 6, height: 6, borderRadius: '50%',
            background: ev.status === 'ERROR' ? '#ff3d5a' : T.green,
            flexShrink: 0,
          }}/>
          <span style={{ color: T.dim }}>{new Date(ev.ts * 1000).toLocaleTimeString()}</span>
          <span style={{ color: T.text }}>{fmt.id(ev.trace_id)}</span>
          <span style={{ color: T.blue }}>{fmt.ms(ev.latency_ms)}</span>
          <span style={{ color: T.orange }}>{fmt.cost(ev.cost_usd)}</span>
        </div>
      ))}
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════════════════
   Shared styles
══════════════════════════════════════════════════════════════════════════════ */
const S = {
  card: {
    background:   T.card,
    border:       `1px solid ${T.border}`,
    borderRadius: 10,
  },
  label: {
    fontSize: 10,
    color: T.dim,
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: 8,
    fontFamily: "'Space Mono',monospace",
  },
};

const DEMO_QUERIES = [
  { label: 'Trace & spans',    text: 'What is a trace and what are spans in distributed observability?' },
  { label: 'LLM cost',         text: 'How would you track token cost for an LLM call in a production agent?' },
  { label: 'P95 latency',      text: 'Explain p95 latency and why it matters for agent APIs, in 2–3 sentences.' },
];

/* ══════════════════════════════════════════════════════════════════════════════
   Main App
══════════════════════════════════════════════════════════════════════════════ */
export default function App() {
  const [metrics,       setMetrics]       = useState(null);
  const [traces,        setTraces]        = useState([]);
  const [selectedTrace, setSelectedTrace] = useState(null);
  const [traceDetail,   setTraceDetail]   = useState(null);
  const [liveEvents,    setLiveEvents]    = useState([]);
  const [query,         setQuery]         = useState('');
  const [response,      setResponse]      = useState(null);
  const [loading,       setLoading]       = useState(false);
  const [wsStatus,      setWsStatus]      = useState('connecting');
  const wsRef = useRef(null);

  /* Fetch metrics + traces */
  const refresh = useCallback(async () => {
    try {
      const [mRes, tRes] = await Promise.all([
        fetch(`${API}/api/metrics`),
        fetch(`${API}/api/traces?limit=30`),
      ]);
      if (mRes.ok) setMetrics(await mRes.json());
      if (tRes.ok) setTraces(await tRes.json());
    } catch { /* backend not yet ready */ }
  }, []);

  /* WebSocket */
  useEffect(() => {
    const connect = () => {
      try {
        const ws = new WebSocket(WS);
        wsRef.current = ws;
        ws.onopen  = () => setWsStatus('connected');
        ws.onclose = () => { setWsStatus('disconnected'); setTimeout(connect, 3000); };
        ws.onerror = () => setWsStatus('error');
        ws.onmessage = e => {
          try {
            const ev = JSON.parse(e.data);
            setLiveEvents(prev => [...prev.slice(-49), ev]);
            refresh();
          } catch {}
        };
      } catch { setTimeout(connect, 3000); }
    };
    connect();
    return () => wsRef.current?.close();
  }, [refresh]);

  /* Polling fallback */
  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 8000);
    return () => clearInterval(id);
  }, [refresh]);

  /* Load trace detail */
  useEffect(() => {
    if (!selectedTrace) { setTraceDetail(null); return; }
    fetch(`${API}/api/traces/${selectedTrace}`)
      .then(r => r.json())
      .then(setTraceDetail)
      .catch(() => {});
  }, [selectedTrace]);

  /* Run agent (optional preset string for demo query buttons) */
  const runAgent = async (preset) => {
    const q = typeof preset === 'string' && preset.trim() ? preset.trim() : query.trim();
    if (!q) return;
    setQuery(q);
    setLoading(true); setResponse(null);
    try {
      const r = await fetch(`${API}/api/agent/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q }),
      });
      const data = await r.json();
      if (!r.ok) {
        const msg = data.detail
          ? (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail))
          : (data.message || `HTTP ${r.status}`);
        setResponse({ error: msg });
        return;
      }
      setResponse(data);
      await refresh();
      if (data.trace_id) setSelectedTrace(data.trace_id);
    } catch (e) {
      setResponse({ error: e.message });
    } finally {
      setLoading(false);
    }
  };

  const wsDot = { connected: T.green, connecting: T.orange, disconnected: '#ff3d5a', error: '#ff3d5a' }[wsStatus] ?? T.muted;

  /* ── Render ──────────────────────────────────────────────────────────────── */
  return (
    <div style={{ minHeight: '100vh', background: T.bg, color: T.text, fontFamily: "'Outfit',sans-serif" }}>

      {/* ── Header ── */}
      <header style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 24px', height: 56,
        background: T.surface, borderBottom: `1px solid ${T.border}`,
        position: 'sticky', top: 0, zIndex: 100,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ width: 8, height: 8, borderRadius: '50%', background: T.green, boxShadow: `0 0 8px ${T.green}` }}/>
          <span style={{ fontFamily: "'Space Mono',monospace", fontWeight: 700, fontSize: 14, color: T.green }}>
            VAIA OBSERVABILITY
          </span>
          <span style={{ color: T.muted, fontSize: 12 }}>L69 · Agent Traces</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <span style={{ fontSize: 11, color: T.dim }}>
            <span style={{ display: 'inline-block', width: 6, height: 6, borderRadius: '50%', background: wsDot, marginRight: 5 }}/>
            WS {wsStatus}
          </span>
          {metrics && (
            <span style={{ fontFamily: "'Space Mono',monospace", fontSize: 11, color: T.dim }}>
              {metrics.total_requests} traces · {fmt.cost(metrics.total_cost_usd)} total
            </span>
          )}
        </div>
      </header>

      <div style={{ padding: 20, maxWidth: 1400, margin: '0 auto' }}>

        {/* ── Metrics row ── */}
        <div style={{ display: 'flex', gap: 12, marginBottom: 20, flexWrap: 'wrap' }}>
          <MetricCard label="Requests"   value={metrics?.total_requests ?? '—'}                  sub="last 1h"         accent={T.blue}  />
          <MetricCard label="P50 Latency" value={fmt.ms(metrics?.p50_latency_ms)}                sub="LLM call"        accent={T.green} />
          <MetricCard label="P95 Latency" value={fmt.ms(metrics?.p95_latency_ms)}                sub="LLM call"        accent={T.orange}/>
          <MetricCard label="P99 Latency" value={fmt.ms(metrics?.p99_latency_ms)}                sub="LLM call"        accent="#ff3d5a" />
          <MetricCard label="Total Cost"  value={fmt.cost(metrics?.total_cost_usd)}              sub="last 1h"         accent={T.orange}/>
          <MetricCard label="Errors"      value={`${metrics?.error_count ?? 0}`}                 sub={`${metrics?.error_rate_pct ?? 0}%`} accent={metrics?.error_count ? '#ff3d5a' : T.green}/>
          <MetricCard label="In Tokens"   value={metrics?.total_input_tokens?.toLocaleString() ?? '—'}  sub="last 1h" accent={T.purple}/>
          <MetricCard label="Out Tokens"  value={metrics?.total_output_tokens?.toLocaleString() ?? '—'} sub="last 1h" accent={T.purple}/>
        </div>

        {/* ── Main 3-column layout ── */}
        <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr 300px', gap: 16 }}>

          {/* ── LEFT: Trace list ── */}
          <div>
            <div style={css(S.card, { overflow: 'hidden' })}>
              <div style={{
                padding: '12px 16px', borderBottom: `1px solid ${T.border}`,
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              }}>
                <span style={css(S.label, { margin: 0 })}>Traces</span>
                <span style={{ fontFamily: "'Space Mono',monospace", fontSize: 10, color: T.muted }}>{traces.length}</span>
              </div>
              <div style={{ maxHeight: 520, overflowY: 'auto' }}>
                {traces.length === 0 ? (
                  <div style={{ padding: 20, color: T.muted, fontSize: 12, textAlign: 'center' }}>
                    No traces yet.<br/>Run a query below.
                  </div>
                ) : traces.map(tr => (
                  <div
                    key={tr.trace_id}
                    onClick={() => setSelectedTrace(tr.trace_id)}
                    style={{
                      padding: '10px 16px',
                      borderBottom: `1px solid ${T.border}`,
                      cursor: 'pointer',
                      background: selectedTrace === tr.trace_id ? `${T.green}15` : 'transparent',
                      transition: 'background 0.15s',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                      <span style={{
                        fontFamily: "'Space Mono',monospace", fontSize: 10,
                        color: selectedTrace === tr.trace_id ? T.green : T.text,
                      }}>
                        {fmt.id(tr.trace_id)}
                      </span>
                      <span style={{
                        fontSize: 9, borderRadius: 3, padding: '1px 5px',
                        background: tr.status === 'ERROR' ? '#ff3d5a22' : `${T.green}22`,
                        color: tr.status === 'ERROR' ? '#ff3d5a' : T.green,
                        fontFamily: "'Space Mono',monospace",
                      }}>
                        {tr.status}
                      </span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10 }}>
                      <span style={{ color: T.blue }}>{fmt.ms(tr.duration_ms)}</span>
                      <span style={{ color: T.orange }}>{fmt.cost(tr.total_cost_usd)}</span>
                    </div>
                    <div style={{ fontSize: 9, color: T.muted, marginTop: 2 }}>
                      {tr.span_count} spans · {fmt.ts(tr.created_at)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* ── CENTER: Waterfall + Query ── */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

            {/* Span waterfall */}
            <div style={css(S.card, { padding: 20, flex: 1 })}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
                <div style={S.label}>Span Waterfall</div>
                {traceDetail && (
                  <span style={{ fontFamily: "'Space Mono',monospace", fontSize: 10, color: T.dim }}>
                    {fmt.id(traceDetail.trace_id)} · {traceDetail.span_count} spans
                  </span>
                )}
              </div>
              <SpanWaterfall spans={traceDetail?.spans}/>
            </div>

            {/* Query box */}
            <div style={css(S.card, { padding: 20 })}>
              <div style={S.label}>Run Agent Query</div>
              <div style={{ marginBottom: 10 }}>
                <div style={{ fontSize: 10, color: T.muted, marginBottom: 6, fontFamily: "'Space Mono',monospace" }}>
                  Demo queries
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                  {DEMO_QUERIES.map(d => (
                    <button
                      key={d.label}
                      type="button"
                      onClick={() => runAgent(d.text)}
                      disabled={loading}
                      title={d.text}
                      style={{
                        background: T.surface,
                        border: `1px solid ${T.border}`,
                        borderRadius: 6,
                        padding: '6px 10px',
                        color: T.text,
                        fontSize: 11,
                        fontFamily: "'Outfit',sans-serif",
                        cursor: loading ? 'not-allowed' : 'pointer',
                        opacity: loading ? 0.6 : 1,
                      }}
                    >
                      {d.label}
                    </button>
                  ))}
                </div>
              </div>
              <div style={{ display: 'flex', gap: 10 }}>
                <input
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && runAgent()}
                  placeholder="Enter a query to generate a trace…"
                  style={{
                    flex: 1, background: T.surface, border: `1px solid ${T.border}`,
                    borderRadius: 8, padding: '10px 14px', color: T.text,
                    fontFamily: "'Outfit',sans-serif", fontSize: 13, outline: 'none',
                  }}
                />
                <button
                  onClick={runAgent}
                  disabled={loading || !query.trim()}
                  style={{
                    background: loading ? T.muted : T.green, border: 'none',
                    borderRadius: 8, padding: '10px 20px', color: '#08080f',
                    fontWeight: 700, fontSize: 13, cursor: loading ? 'not-allowed' : 'pointer',
                    fontFamily: "'Outfit',sans-serif", transition: 'opacity 0.2s',
                    minWidth: 80,
                  }}
                >
                  {loading ? '…' : 'Run'}
                </button>
              </div>

              {/* Response */}
              {response && (
                <div style={{
                  marginTop: 14, padding: 14, background: T.surface,
                  borderRadius: 8, border: `1px solid ${T.border}`,
                }}>
                  {response.error ? (
                    <div style={{ color: '#ff3d5a', fontSize: 13 }}>Error: {response.error}</div>
                  ) : (
                    <>
                      <div style={{ fontSize: 13, color: T.text, marginBottom: 10, lineHeight: 1.5 }}>
                        {response.output}
                      </div>
                      <div style={{ display: 'flex', gap: 16, fontSize: 10, fontFamily: "'Space Mono',monospace" }}>
                        <span style={{ color: T.blue }}>{fmt.ms(response.latency_ms)}</span>
                        <span style={{ color: T.orange }}>{fmt.cost(response.cost_usd)}</span>
                        <span style={{ color: T.purple }}>in:{response.input_tokens} out:{response.output_tokens}</span>
                        <span style={{ color: T.dim }}>{fmt.id(response.trace_id)}</span>
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* ── RIGHT: Charts + Live feed ── */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

            {/* Cost trend */}
            <div style={css(S.card, { padding: 16 })}>
              <div style={S.label}>Cost / min (m¢)</div>
              <CostChart data={metrics?.cost_trend}/>
            </div>

            {/* Latency percentiles */}
            <div style={css(S.card, { padding: 16 })}>
              <div style={S.label}>Latency Percentiles (ms)</div>
              <LatencyBar
                p50={metrics?.p50_latency_ms ?? 0}
                p95={metrics?.p95_latency_ms ?? 0}
                p99={metrics?.p99_latency_ms ?? 0}
              />
            </div>

            {/* Live events */}
            <div style={css(S.card, { padding: 16, flex: 1 })}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                <div style={S.label}>Live Events</div>
                <span style={{ fontFamily: "'Space Mono',monospace", fontSize: 9, color: T.muted }}>
                  {liveEvents.length}
                </span>
              </div>
              <LiveFeed events={liveEvents}/>
            </div>

            {/* L70 Breach preview */}
            {metrics?.recent_breaches?.length > 0 && (
              <div style={css(S.card, { padding: 16, border: `1px solid #ff3d5a55` })}>
                <div style={css(S.label, { color: '#ff3d5a' })}>⚠ L70 Breach Queue</div>
                {metrics.recent_breaches.slice(0, 3).map((b, i) => (
                  <div key={i} style={{
                    fontSize: 10, fontFamily: "'Space Mono',monospace",
                    color: '#ff3d5a', marginBottom: 4,
                  }}>
                    {b.metric}: {Number(b.value).toFixed(2)} &gt; {b.threshold}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
