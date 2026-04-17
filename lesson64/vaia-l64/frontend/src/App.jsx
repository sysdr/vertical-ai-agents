import React, { useState, useEffect, useCallback } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, AreaChart, Area
} from 'recharts'

const API = ''  // proxied via Vite

const COLORS = {
  green: '#43a047', orange: '#fb8c00', blue: '#1e88e5',
  red: '#e53935', purple: '#8e24aa', bg: '#f8f9fa',
  card: '#ffffff', border: '#e0e0e0', text: '#212121',
  muted: '#757575'
}

const safeNumber = (value, fallback = 0) => (
  Number.isFinite(value) ? value : fallback
)

const DEMO_QUERIES = [
  'Summarize how this VAIA service is currently running.',
  'Explain the difference between liveness and readiness probes.',
  'What does the /metrics endpoint expose for this app?',
  'Give a short demo response for a customer support assistant.'
]

function StatusBadge({ status }) {
  const color = status === 'ok' || status === 'ready' ? COLORS.green :
                status === 'loading' ? COLORS.orange : COLORS.red
  return (
    <span style={{
      background: color + '22', color, border: `1px solid ${color}44`,
      borderRadius: 20, padding: '2px 12px', fontSize: 12, fontWeight: 700
    }}>
      {status.toUpperCase()}
    </span>
  )
}

function Card({ title, children, accent = COLORS.blue }) {
  return (
    <div style={{
      background: COLORS.card, border: `1px solid ${COLORS.border}`,
      borderTop: `3px solid ${accent}`, borderRadius: 10,
      padding: 20, marginBottom: 16,
      boxShadow: '0 2px 8px #0000000d'
    }}>
      <div style={{ fontWeight: 700, color: COLORS.text, marginBottom: 12, fontSize: 13 }}>
        {title}
      </div>
      {children}
    </div>
  )
}

function MetricPill({ label, value, unit = '', color = COLORS.blue }) {
  return (
    <div style={{
      background: color + '11', border: `1px solid ${color}33`,
      borderRadius: 8, padding: '8px 14px', textAlign: 'center', minWidth: 90
    }}>
      <div style={{ fontSize: 20, fontWeight: 800, color }}>{value}{unit}</div>
      <div style={{ fontSize: 10, color: COLORS.muted, marginTop: 2 }}>{label}</div>
    </div>
  )
}

export default function App() {
  const [liveness, setLiveness] = useState('loading')
  const [readiness, setReadiness] = useState('loading')
  const [metrics, setMetrics] = useState({ requests: 0, active: 0, agentReady: 0 })
  const [metricHistory, setMetricHistory] = useState([])
  const [query, setQuery] = useState('')
  const [inferResult, setInferResult] = useState(null)
  const [inferLoading, setInferLoading] = useState(false)
  const [tick, setTick] = useState(0)

  const pollHealth = useCallback(async () => {
    try {
      const r = await fetch(`${API}/health/live`)
      setLiveness((await r.json()).status)
    } catch { setLiveness('error') }
    try {
      const r = await fetch(`${API}/health/ready`)
      if (r.ok) setReadiness((await r.json()).status)
      else setReadiness('not_ready')
    } catch { setReadiness('error') }
  }, [])

  const parsePrometheus = useCallback((text) => {
    const lines = text.split('\n')
    const sumMetric = (name) => lines.reduce((total, line) => {
      if (!line.startsWith(`${name}{`) && !line.startsWith(`${name} `)) return total
      const value = safeNumber(Number(line.trim().split(/\s+/).pop()))
      return total + value
    }, 0)
    const latestMetric = (name) => {
      for (let idx = lines.length - 1; idx >= 0; idx -= 1) {
        const line = lines[idx]
        if (!line.startsWith(`${name}{`) && !line.startsWith(`${name} `)) continue
        return safeNumber(Number(line.trim().split(/\s+/).pop()))
      }
      return 0
    }
    return {
      requests: safeNumber(sumMetric('vaia_requests_total')),
      active: safeNumber(latestMetric('vaia_active_requests')),
      agentReady: safeNumber(latestMetric('vaia_agent_ready')),
    }
  }, [])

  const pollMetrics = useCallback(async () => {
    try {
      const r = await fetch(`${API}/metrics`)
      const text = await r.text()
      const parsed = parsePrometheus(text)
      setMetrics(parsed)
      setMetricHistory(prev => {
        const now = new Date().toLocaleTimeString()
        const next = [...prev.slice(-29), {
          time: now,
          requests: safeNumber(parsed.requests),
          active: safeNumber(parsed.active)
        }]
        return next
      })
    } catch {}
  }, [parsePrometheus])

  useEffect(() => {
    pollHealth(); pollMetrics()
    const id = setInterval(() => { pollHealth(); pollMetrics(); setTick(t => t + 1) }, 5000)
    return () => clearInterval(id)
  }, [pollHealth, pollMetrics])

  const runInfer = async (inputQuery) => {
    if (!inputQuery.trim()) return
    setInferLoading(true); setInferResult(null)
    try {
      const r = await fetch(`${API}/v1/agent/infer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: inputQuery.trim() })
      })
      if (r.ok) setInferResult(await r.json())
      else {
        const errBody = await r.json().catch(() => ({}))
        setInferResult({ error: errBody.detail || `HTTP ${r.status}` })
      }
    } catch (e) { setInferResult({ error: e.message }) }
    await pollMetrics()
    setInferLoading(false)
  }

  const handleInfer = () => runInfer(query)

  const handleDemoInfer = (demoQuery) => {
    setQuery(demoQuery)
    runInfer(demoQuery)
  }

  return (
    <div style={{ minHeight: '100vh', background: COLORS.bg, fontFamily: 'system-ui, sans-serif', padding: 24 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 24 }}>
        <div style={{
          background: COLORS.blue, color: 'white', fontWeight: 800, fontSize: 13,
          padding: '6px 14px', borderRadius: 6
        }}>L64</div>
        <div>
          <div style={{ fontWeight: 700, fontSize: 18, color: COLORS.text }}>
            CD & Containerization Dashboard
          </div>
          <div style={{ fontSize: 12, color: COLORS.muted }}>
            Module 5 · VAIA Curriculum · vaia-agent:local
          </div>
        </div>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
          <StatusBadge status={liveness === 'ok' ? 'live' : liveness} />
          <StatusBadge status={readiness} />
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, marginBottom: 16 }}>
        {/* Health */}
        <Card title="Health Probes" accent={COLORS.green}>
          <div style={{ display: 'flex', gap: 12 }}>
            <MetricPill label="Liveness" value={liveness === 'ok' ? '✓' : '✗'} color={liveness === 'ok' ? COLORS.green : COLORS.red}/>
            <MetricPill label="Readiness" value={readiness === 'ready' ? '✓' : '✗'} color={readiness === 'ready' ? COLORS.green : COLORS.orange}/>
            <MetricPill label="Agent" value={metrics.agentReady ? '✓' : '✗'} color={metrics.agentReady ? COLORS.green : COLORS.orange}/>
          </div>
          <div style={{ fontSize: 11, color: COLORS.muted, marginTop: 10 }}>
            /health/live · /health/ready · vaia_agent_ready gauge
          </div>
        </Card>

        {/* Request Metrics */}
        <Card title="Request Metrics" accent={COLORS.orange}>
          <div style={{ display: 'flex', gap: 12 }}>
            <MetricPill label="Total Req" value={metrics.requests.toFixed(0)} color={COLORS.orange}/>
            <MetricPill label="Active" value={metrics.active.toFixed(0)} color={COLORS.purple}/>
          </div>
          <div style={{ fontSize: 11, color: COLORS.muted, marginTop: 10 }}>
            Prometheus scrape · /metrics endpoint
          </div>
        </Card>

        {/* Container Info */}
        <Card title="Container" accent={COLORS.purple}>
          <div style={{ fontSize: 12, color: COLORS.text, lineHeight: 1.8 }}>
            <div><b>Image:</b> vaia-agent:local</div>
            <div><b>Stage:</b> runtime (multi-stage)</div>
            <div><b>Base:</b> python:3.12-slim</div>
            <div><b>Target size:</b> ~180 MB</div>
          </div>
        </Card>
      </div>

      {/* Request trend chart */}
      <Card title="Total Requests Over Time (Prometheus poll)" accent={COLORS.blue}>
        <ResponsiveContainer width="100%" height={140}>
          <AreaChart data={metricHistory} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0"/>
            <XAxis dataKey="time" tick={{ fontSize: 9 }} interval="preserveStartEnd"/>
            <YAxis tick={{ fontSize: 9 }} allowDecimals={false}/>
            <Tooltip contentStyle={{ fontSize: 11 }}/>
            <Area type="monotone" dataKey="requests" stroke={COLORS.blue} fill={COLORS.blue + '22'} strokeWidth={2}/>
          </AreaChart>
        </ResponsiveContainer>
      </Card>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        {/* Inference Panel */}
        <Card title="Agent Inference Test" accent={COLORS.green}>
          <textarea
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Enter a query to test /v1/agent/infer…"
            rows={3}
            style={{
              width: '100%', boxSizing: 'border-box', padding: '8px 12px',
              border: `1px solid ${COLORS.border}`, borderRadius: 6, resize: 'vertical',
              fontSize: 13, fontFamily: 'inherit', marginBottom: 10
            }}
          />
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 10 }}>
            {DEMO_QUERIES.map((demoQuery) => (
              <button
                key={demoQuery}
                onClick={() => handleDemoInfer(demoQuery)}
                disabled={inferLoading}
                style={{
                  background: '#eef3ff',
                  color: COLORS.blue,
                  border: `1px solid ${COLORS.blue}44`,
                  borderRadius: 16,
                  padding: '6px 10px',
                  cursor: inferLoading ? 'not-allowed' : 'pointer',
                  fontWeight: 600,
                  fontSize: 11
                }}
              >
                {demoQuery}
              </button>
            ))}
          </div>
          <button
            onClick={handleInfer}
            disabled={inferLoading || !query.trim()}
            style={{
              background: inferLoading ? COLORS.muted : COLORS.green,
              color: 'white', border: 'none', borderRadius: 6,
              padding: '8px 20px', cursor: 'pointer', fontWeight: 700, fontSize: 13
            }}
          >
            {inferLoading ? 'Running…' : 'Run Inference'}
          </button>
          {inferResult && (
            <div style={{
              marginTop: 12, background: inferResult.error ? COLORS.red + '11' : COLORS.green + '11',
              border: `1px solid ${inferResult.error ? COLORS.red : COLORS.green}33`,
              borderRadius: 6, padding: '10px 14px'
            }}>
              {inferResult.error
                ? <span style={{ color: COLORS.red, fontSize: 12 }}>Error: {inferResult.error}</span>
                : <>
                    <div style={{ fontSize: 12, color: COLORS.muted, marginBottom: 6 }}>
                      Model: {inferResult.model} · {inferResult.finish_reason}
                    </div>
                    <div style={{ fontSize: 13, color: COLORS.text }}>{inferResult.answer}</div>
                  </>
              }
            </div>
          )}
        </Card>

        {/* K8s Manifests Reference */}
        <Card title="K8s Manifests" accent={COLORS.orange}>
          {[
            { file: 'namespace.yaml', desc: 'vaia-prod namespace' },
            { file: 'configmap.yaml', desc: 'Non-sensitive env vars' },
            { file: 'secret.yaml', desc: 'GEMINI_API_KEY (base64)' },
            { file: 'deployment.yaml', desc: '2 replicas · RollingUpdate · probes' },
            { file: 'service.yaml', desc: 'ClusterIP port 8000' },
            { file: 'hpa.yaml', desc: 'CPU 60% · min 2 / max 10 → L65' },
            { file: 'ingress.yaml', desc: 'nginx · TLS stub' },
          ].map(({ file, desc }) => (
            <div key={file} style={{
              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              padding: '5px 0', borderBottom: `1px solid ${COLORS.border}`, fontSize: 12
            }}>
              <code style={{ color: COLORS.blue, fontSize: 11 }}>k8s/{file}</code>
              <span style={{ color: COLORS.muted, fontSize: 11 }}>{desc}</span>
            </div>
          ))}
          <div style={{ marginTop: 10, fontSize: 11, color: COLORS.muted }}>
            Apply: <code>kubectl apply -f k8s/</code>
          </div>
        </Card>
      </div>

      {/* Pipeline steps */}
      <Card title="CD Pipeline Stages" accent={COLORS.purple}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 0, flexWrap: 'wrap' }}>
          {[
            { label: 'L63 CI Tests', color: COLORS.orange },
            { label: 'docker build', color: COLORS.blue },
            { label: 'Trivy Scan', color: COLORS.red },
            { label: 'docker push', color: COLORS.purple },
            { label: 'kubectl apply', color: COLORS.green },
            { label: '→ L65 Load Test', color: COLORS.muted },
          ].map(({ label, color }, i) => (
            <React.Fragment key={label}>
              <div style={{
                background: color + '18', border: `1px solid ${color}44`,
                borderRadius: 6, padding: '6px 12px', fontSize: 12,
                fontWeight: 600, color
              }}>{label}</div>
              {i < 5 && <div style={{ color: COLORS.border, margin: '0 4px', fontSize: 16 }}>›</div>}
            </React.Fragment>
          ))}
        </div>
      </Card>

      <div style={{ textAlign: 'center', color: COLORS.muted, fontSize: 11, marginTop: 8 }}>
        L64 · Module 5 · VAIA 90-Lesson Curriculum · Poll interval 5s · tick #{tick}
      </div>
    </div>
  )
}
