import React, { useState, useEffect, useRef, useCallback } from 'react'
import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Legend
} from 'recharts'

const API = '/api'
const POLL_MS = 3000

const STATE_COLORS = {
  PENDING:      '#B1ADA1',
  DEPLOYING:    '#6B8CBA',
  HEALTHY:      '#5BAB6E',
  DEGRADED:     '#C15F3C',
  ROLLING_BACK: '#E5A623',
  FAILED:       '#C13C3C',
}

const SEVERITY_COLORS = {
  LOW:      '#5BAB6E',
  MEDIUM:   '#E5A623',
  HIGH:     '#C15F3C',
  CRITICAL: '#C13C3C',
}

// ── helpers ──────────────────────────────────────────────────────────────────
async function apiFetch(path) {
  const r = await fetch(`${API}${path}`)
  if (!r.ok) throw new Error(`${r.status}`)
  return r.json()
}

// ── sub-components ────────────────────────────────────────────────────────────
function StateChip({ state }) {
  const color = STATE_COLORS[state] ?? '#B1ADA1'
  return (
    <span style={{
      display: 'inline-block',
      background: color,
      color: '#fff',
      fontWeight: 700,
      fontSize: 13,
      padding: '4px 14px',
      borderRadius: 20,
      letterSpacing: 1,
      boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
    }}>
      {state ?? '…'}
    </span>
  )
}

function MetricCard({ label, value, unit = '', warn, critical, precision = 3 }) {
  const num = typeof value === 'number' ? value.toFixed(precision) : '—'
  let color = '#222'
  if (critical !== undefined && value >= critical) color = '#C15F3C'
  else if (warn !== undefined && value >= warn) color = '#E5A623'
  return (
    <div style={{
      background: '#fff',
      borderRadius: 14,
      padding: '18px 22px',
      boxShadow: '0 4px 18px rgba(0,0,0,0.08)',
      minWidth: 140,
      flex: 1,
    }}>
      <div style={{ fontSize: 11, color: '#B1ADA1', textTransform: 'uppercase', letterSpacing: 1 }}>
        {label}
      </div>
      <div style={{ fontSize: 28, fontWeight: 800, color, marginTop: 4 }}>
        {num}<span style={{ fontSize: 14, fontWeight: 400, marginLeft: 3 }}>{unit}</span>
      </div>
    </div>
  )
}

function AlertRow({ alert }) {
  const color = SEVERITY_COLORS[alert.severity] ?? '#B1ADA1'
  const ts = new Date(alert.ts * 1000).toLocaleTimeString()
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 10,
      padding: '8px 12px', borderRadius: 8,
      background: '#FAFAF8', borderLeft: `4px solid ${color}`,
      marginBottom: 6,
    }}>
      <span style={{ color, fontWeight: 700, minWidth: 70, fontSize: 12 }}>
        {alert.severity}
      </span>
      <span style={{ fontSize: 13, flex: 1, color: '#444' }}>{alert.message}</span>
      <span style={{ fontSize: 11, color: '#B1ADA1' }}>{ts}</span>
    </div>
  )
}

function ChatPanel({ onInjectDrift }) {
  const demoQueries = [
    { text: 'What is blue-green deployment in MLOps?', drift: 0.2 },
    { text: 'How do we monitor model drift?', drift: 0.5 },
    { text: 'When should rollback be triggered?', drift: 0.9 },
    { text: 'Explain p95 latency and error rate thresholds.', drift: 0.5 },
  ]
  const [messages, setMessages] = useState([
    { role: 'assistant', text: "Hello! I'm the L74 MLOps agent. Ask me anything about deployment patterns." }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = useCallback(async (overrideMessage = null, driftToInject = null) => {
    if (loading) return
    const msg = (overrideMessage ?? input).trim()
    if (!msg) return
    setInput('')
    setMessages(m => [...m, { role: 'user', text: msg }])
    setLoading(true)
    try {
      const r = await fetch(`${API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: 'dashboard', message: msg }),
      })
      const data = await r.json()
      setMessages(m => [...m, {
        role: 'assistant',
        text: data.response,
        latency: data.latency_ms
      }])
      if (driftToInject != null && typeof onInjectDrift === 'function') {
        await onInjectDrift(driftToInject)
      }
    } catch (e) {
      setMessages(m => [...m, { role: 'assistant', text: `Error: ${e.message}` }])
    } finally {
      setLoading(false)
    }
  }, [input, loading, onInjectDrift])

  return (
    <div style={{
      background: '#fff', borderRadius: 14, padding: 20,
      boxShadow: '0 4px 18px rgba(0,0,0,0.08)',
      display: 'flex', flexDirection: 'column', height: 420
    }}>
      <div style={{ fontSize: 13, fontWeight: 700, color: '#C15F3C', marginBottom: 12 }}>
        AGENT CHAT
      </div>
      <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 8 }}>
        {messages.map((m, i) => (
          <div key={i} style={{
            alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
            maxWidth: '80%',
            background: m.role === 'user' ? '#C15F3C' : '#F4F3EE',
            color: m.role === 'user' ? '#fff' : '#333',
            padding: '10px 14px',
            borderRadius: m.role === 'user' ? '18px 18px 4px 18px' : '18px 18px 18px 4px',
            fontSize: 13,
          }}>
            {m.text}
            {m.latency && (
              <div style={{ fontSize: 10, color: m.role === 'user' ? 'rgba(255,255,255,0.6)' : '#B1ADA1', marginTop: 3 }}>
                {m.latency}ms
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div style={{ alignSelf: 'flex-start', padding: '10px 14px', background: '#F4F3EE', borderRadius: 18, fontSize: 13, color: '#B1ADA1' }}>
            Thinking…
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 10 }}>
        {demoQueries.map((q) => (
          <button
            key={q.text}
            onClick={() => send(q.text, q.drift)}
            disabled={loading}
            style={{
              border: '1px solid #E0DED8',
              background: '#FAFAF8',
              color: '#666',
              borderRadius: 999,
              padding: '6px 10px',
              fontSize: 11,
              cursor: loading ? 'default' : 'pointer',
              opacity: loading ? 0.7 : 1,
            }}
          >
            {q.text}
          </button>
        ))}
      </div>
      <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && send()}
          placeholder="Ask about MLOps, deployment patterns…"
          style={{
            flex: 1, padding: '10px 14px', borderRadius: 24,
            border: '1.5px solid #E0DED8', fontSize: 13, outline: 'none',
            background: '#FAFAF8',
          }}
        />
        <button
          onClick={() => send()}
          disabled={loading || !input.trim()}
          style={{
            background: loading || !input.trim() ? '#B1ADA1' : '#C15F3C',
            color: '#fff', border: 'none', borderRadius: 24,
            padding: '10px 20px', fontWeight: 700, cursor: loading ? 'default' : 'pointer',
            fontSize: 13,
          }}
        >
          Send
        </button>
      </div>
    </div>
  )
}

// ── main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [state, setState] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [alerts, setAlerts] = useState([])
  const [history, setHistory] = useState([])
  const [driftHistory, setDriftHistory] = useState([])
  const [injectingDrift, setInjectingDrift] = useState(false)
  const tickRef = useRef(0)

  const poll = useCallback(async () => {
    try {
      const [stateData, metricsData, alertsData, histData] = await Promise.all([
        apiFetch('/deployment/state'),
        apiFetch('/deployment/metrics-summary'),
        apiFetch('/deployment/alerts'),
        apiFetch('/deployment/history'),
      ])
      setState(stateData.state)
      setMetrics(metricsData)
      setAlerts((alertsData.alerts ?? []).reverse())
      setHistory(histData.history ?? [])
      tickRef.current += 1
      setDriftHistory(prev => {
        const next = [...prev, {
          t: tickRef.current,
          drift: metricsData.drift_score,
          err: metricsData.error_rate,
          p95: metricsData.p95_latency_s,
        }]
        return next.slice(-40)
      })
    } catch (_) {}
  }, [])

  useEffect(() => {
    poll()
    const id = setInterval(poll, POLL_MS)
    return () => clearInterval(id)
  }, [poll])

  const injectDrift = async (score) => {
    setInjectingDrift(true)
    try {
      await fetch(`${API}/admin/inject-drift?score=${score}`, { method: 'POST' })
      setTimeout(poll, 500)
    } finally {
      setTimeout(() => setInjectingDrift(false), 1000)
    }
  }

  const errPct = metrics?.error_rate != null ? metrics.error_rate * 100 : undefined

  return (
    <div style={{ minHeight: '100vh', background: '#F4F3EE', padding: '28px 32px' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 28 }}>
        <div>
          <div style={{ fontSize: 22, fontWeight: 800, color: '#C15F3C' }}>
            MLOps Pipeline Dashboard
          </div>
          <div style={{ fontSize: 13, color: '#B1ADA1', marginTop: 2 }}>
            L74 · Hands-On MLOps Deployment · Gemini-2.0-flash
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <StateChip state={state} />
        </div>
      </div>

      {/* Metric cards */}
      <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', marginBottom: 24 }}>
        <MetricCard
          label="Drift Score"
          value={metrics?.drift_score}
          warn={0.2} critical={0.35}
        />
        <MetricCard
          label="Error Rate"
          value={errPct ?? NaN}
          unit="%"
          precision={2}
          warn={2} critical={5}
        />
        <MetricCard
          label="P95 Latency"
          value={metrics?.p95_latency_s}
          unit="s"
          warn={1.5} critical={3.0}
        />
        <div style={{
          background: '#fff', borderRadius: 14, padding: '18px 22px',
          boxShadow: '0 4px 18px rgba(0,0,0,0.08)', minWidth: 140, flex: 1,
          display: 'flex', flexDirection: 'column', justifyContent: 'space-between'
        }}>
          <div style={{ fontSize: 11, color: '#B1ADA1', textTransform: 'uppercase', letterSpacing: 1 }}>
            Inject Drift
          </div>
          <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
            {[0.2, 0.5, 0.9].map(s => (
              <button
                key={s}
                onClick={() => injectDrift(s)}
                disabled={injectingDrift}
                style={{
                  flex: 1, padding: '6px 0', borderRadius: 8, border: 'none',
                  background: s < 0.35 ? '#5BAB6E' : s < 0.7 ? '#E5A623' : '#C15F3C',
                  color: '#fff', fontWeight: 700, fontSize: 12,
                  cursor: injectingDrift ? 'default' : 'pointer',
                  opacity: injectingDrift ? 0.6 : 1,
                }}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Charts row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 24 }}>
        {/* Drift chart */}
        <div style={{ background: '#fff', borderRadius: 14, padding: '18px 20px', boxShadow: '0 4px 18px rgba(0,0,0,0.08)' }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: '#B1ADA1', marginBottom: 10, textTransform: 'uppercase', letterSpacing: 1 }}>
            Drift Score
          </div>
          <ResponsiveContainer width="100%" height={160}>
            <AreaChart data={driftHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F0EEE9" />
              <XAxis dataKey="t" hide />
              <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} />
              <Tooltip formatter={(v) => v.toFixed(3)} />
              <ReferenceLine y={0.35} stroke="#C15F3C" strokeDasharray="4 2" label={{ value: 'threshold', fontSize: 10, fill: '#C15F3C' }} />
              <Area type="monotone" dataKey="drift" stroke="#C15F3C" fill="#FDECEA" strokeWidth={2} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Latency + Error chart */}
        <div style={{ background: '#fff', borderRadius: 14, padding: '18px 20px', boxShadow: '0 4px 18px rgba(0,0,0,0.08)' }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: '#B1ADA1', marginBottom: 10, textTransform: 'uppercase', letterSpacing: 1 }}>
            P95 Latency (s) · Error rate (0–1)
          </div>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={driftHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F0EEE9" />
              <XAxis dataKey="t" hide />
              <YAxis yAxisId="left" tick={{ fontSize: 10 }} />
              <YAxis yAxisId="right" orientation="right" domain={[0, 1]} tick={{ fontSize: 10 }} />
              <Tooltip formatter={(v) => v.toFixed(3)} />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              <ReferenceLine yAxisId="left" y={3.0} stroke="#E5A623" strokeDasharray="4 2" />
              <Line yAxisId="left" type="monotone" dataKey="p95" name="P95 s" stroke="#6B8CBA" strokeWidth={2} dot={false} />
              <Line yAxisId="right" type="monotone" dataKey="err" name="Err rate" stroke="#C15F3C" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Bottom row: Alerts + History + Chat */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1.4fr', gap: 16 }}>
        {/* Alerts */}
        <div style={{ background: '#fff', borderRadius: 14, padding: '18px 20px', boxShadow: '0 4px 18px rgba(0,0,0,0.08)', maxHeight: 420, overflowY: 'auto' }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: '#B1ADA1', marginBottom: 12, textTransform: 'uppercase', letterSpacing: 1 }}>
            Alerts ({alerts.length})
          </div>
          {alerts.length === 0
            ? <div style={{ color: '#B1ADA1', fontSize: 13 }}>No alerts yet.</div>
            : alerts.map((a, i) => <AlertRow key={i} alert={a} />)
          }
        </div>

        {/* State history */}
        <div style={{ background: '#fff', borderRadius: 14, padding: '18px 20px', boxShadow: '0 4px 18px rgba(0,0,0,0.08)', maxHeight: 420, overflowY: 'auto' }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: '#B1ADA1', marginBottom: 12, textTransform: 'uppercase', letterSpacing: 1 }}>
            State History
          </div>
          {history.length === 0
            ? <div style={{ color: '#B1ADA1', fontSize: 13 }}>No transitions yet.</div>
            : history.map((h, i) => (
              <div key={i} style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '7px 10px', borderRadius: 8,
                background: i === 0 ? '#FFF7F4' : '#FAFAF8',
                marginBottom: 5, fontSize: 12,
              }}>
                <StateChip state={h.from} />
                <span style={{ color: '#B1ADA1' }}>→</span>
                <StateChip state={h.to} />
                <span style={{ flex: 1, color: '#888', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {h.reason}
                </span>
              </div>
            ))
          }
        </div>

        {/* Chat */}
        <ChatPanel onInjectDrift={injectDrift} />
      </div>
    </div>
  )
}
