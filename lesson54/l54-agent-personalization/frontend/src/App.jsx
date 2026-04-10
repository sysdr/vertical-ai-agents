import React, { useState, useEffect, useRef } from 'react'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, BarChart, Bar, Legend, Cell
} from 'recharts'

const API = ''  // proxied via vite

const PERSONA_COLORS = {
  EXECUTIVE: '#f59e0b',
  PRACTITIONER: '#6ee7b7',
  LEARNER: '#818cf8',
  ANALYST: '#fb7185',
  DEFAULT: '#64748b'
}

const PERSONA_ICONS = {
  EXECUTIVE: '⚡',
  PRACTITIONER: '⚙️',
  LEARNER: '📚',
  ANALYST: '📊',
  DEFAULT: '🤖'
}

// ── Utility ──────────────────────────────────────────────────────
async function apiFetch(path, opts = {}) {
  const r = await fetch(`/api${path}`, {
    headers: { 'Content-Type': 'application/json', ...opts.headers },
    ...opts
  })
  if (!r.ok) throw new Error(`API ${r.status}: ${await r.text()}`)
  return r.json()
}

// ── Header ───────────────────────────────────────────────────────
function Header({ activeTab, setActiveTab }) {
  const tabs = ['profiles', 'chat', 'compare', 'analytics']
  return (
    <header style={{
      borderBottom: '1px solid var(--border)',
      padding: '0 24px',
      display: 'flex', alignItems: 'center', gap: 32, height: 56
    }}>
      <div style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 18, color: 'var(--accent)' }}>
        L54<span style={{ color: 'var(--text-muted)', fontWeight: 400 }}> · Personalization</span>
      </div>
      <nav style={{ display: 'flex', gap: 4 }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setActiveTab(t)} style={{
            background: activeTab === t ? 'var(--surface2)' : 'transparent',
            border: activeTab === t ? '1px solid var(--border)' : '1px solid transparent',
            color: activeTab === t ? 'var(--text)' : 'var(--text-muted)',
            padding: '6px 14px', borderRadius: 6, cursor: 'pointer',
            fontFamily: 'var(--font-mono)', fontSize: 13, textTransform: 'capitalize',
            transition: 'all 0.15s'
          }}>
            {t}
          </button>
        ))}
      </nav>
    </header>
  )
}

// ── PersonaBadge ─────────────────────────────────────────────────
function PersonaBadge({ persona, size = 'sm' }) {
  const color = PERSONA_COLORS[persona] || PERSONA_COLORS.DEFAULT
  const icon = PERSONA_ICONS[persona] || '🤖'
  const fs = size === 'lg' ? 15 : 12
  return (
    <span style={{
      background: `${color}18`, border: `1px solid ${color}55`,
      color, borderRadius: 20, padding: size === 'lg' ? '4px 12px' : '2px 8px',
      fontSize: fs, fontWeight: 500, display: 'inline-flex', alignItems: 'center', gap: 4
    }}>
      {icon} {persona}
    </span>
  )
}

// ── PreferenceRadar ───────────────────────────────────────────────
function PreferenceRadar({ vector }) {
  if (!vector) return <div style={{ color: 'var(--text-muted)', fontSize: 13 }}>No preference data yet</div>
  const data = Object.entries(vector).map(([key, val]) => ({
    dim: key.replace(/_/g, ' ').replace('narrative vs list', 'narrative'),
    value: Math.round(val * 100)
  }))
  return (
    <ResponsiveContainer width="100%" height={220}>
      <RadarChart data={data}>
        <PolarGrid stroke="var(--border)" />
        <PolarAngleAxis dataKey="dim" tick={{ fill: 'var(--text-muted)', fontSize: 10 }} />
        <Radar dataKey="value" stroke="var(--accent)" fill="var(--accent)" fillOpacity={0.18} strokeWidth={2} />
      </RadarChart>
    </ResponsiveContainer>
  )
}

// ── ProfileCard ───────────────────────────────────────────────────
function ProfileCard({ profile, onSelect, selected }) {
  const persona = profile.persona || 'DEFAULT'
  const color = PERSONA_COLORS[persona]
  return (
    <div onClick={() => onSelect(profile)} style={{
      background: 'var(--surface)',
      border: `1px solid ${selected ? color : 'var(--border)'}`,
      borderRadius: 10, padding: '14px 16px', cursor: 'pointer',
      transition: 'all 0.15s',
      boxShadow: selected ? `0 0 0 1px ${color}44` : 'none'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
        <div>
          <div style={{ fontFamily: 'var(--font-display)', fontWeight: 600, fontSize: 14, marginBottom: 2 }}>
            {profile.display_name}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{profile.user_id}</div>
        </div>
        <PersonaBadge persona={persona} />
      </div>
      <div style={{ fontSize: 12, color: 'var(--text-muted)', display: 'flex', gap: 12 }}>
        <span>💬 {profile.interaction_count} chats</span>
        <span>{profile.consent_behavioral ? '🔍 tracking on' : '🔒 no tracking'}</span>
      </div>
    </div>
  )
}

// ── ProfileDetail ─────────────────────────────────────────────────
function ProfileDetail({ profile, onRefresh }) {
  const [inferring, setInferring] = useState(false)
  const [inferResult, setInferResult] = useState(null)
  const [personaScores, setPersonaScores] = useState(null)

  useEffect(() => {
    if (!profile) return
    apiFetch(`/profiles/${profile.user_id}/persona`)
      .then(r => setPersonaScores(r.scores))
      .catch(() => {})
  }, [profile])

  const handleInfer = async () => {
    setInferring(true)
    try {
      const r = await apiFetch(`/profiles/${profile.user_id}/infer`, { method: 'POST' })
      setInferResult(r)
      onRefresh()
    } catch (e) {
      setInferResult({ error: e.message })
    }
    setInferring(false)
  }

  if (!profile) return null

  const scoreData = personaScores ? Object.entries(personaScores).map(([p, s]) => ({
    persona: p, score: Math.round(s * 100)
  })) : []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      <div style={{ background: 'var(--surface)', borderRadius: 10, border: '1px solid var(--border)', padding: 16 }}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, textTransform: 'uppercase', letterSpacing: 1 }}>
          Preference Vector
        </div>
        <PreferenceRadar vector={profile.preference_vector} />
        <div style={{ marginTop: 12, display: 'flex', gap: 8 }}>
          <button onClick={handleInfer} disabled={inferring || !profile.consent_behavioral} style={{
            background: inferring ? 'var(--surface2)' : 'var(--accent)', color: 'var(--bg)',
            border: 'none', borderRadius: 6, padding: '7px 14px', cursor: inferring ? 'not-allowed' : 'pointer',
            fontSize: 12, fontWeight: 600, fontFamily: 'var(--font-mono)',
            opacity: !profile.consent_behavioral ? 0.4 : 1
          }}>
            {inferring ? '⟳ Inferring...' : '⚡ Run Inference'}
          </button>
          {!profile.consent_behavioral && (
            <span style={{ fontSize: 11, color: 'var(--text-muted)', alignSelf: 'center' }}>
              Enable behavioral consent first
            </span>
          )}
        </div>
        {inferResult && (
          <div style={{
            marginTop: 10, fontSize: 11, padding: '8px 10px',
            background: inferResult.error ? '#fb718518' : '#6ee7b718',
            borderRadius: 6, color: inferResult.error ? '#fb7185' : 'var(--accent)'
          }}>
            {inferResult.error ? `✗ ${inferResult.error}` : `✓ ${inferResult.persona} · ${inferResult.interactions_analyzed} interactions analyzed`}
            {inferResult.drift && (
              <div style={{ marginTop: 4, color: '#f59e0b' }}>
                ⚠ Drift detected: {inferResult.drift.changed_dimension} Δ{inferResult.drift.delta.toFixed(2)}
              </div>
            )}
          </div>
        )}
      </div>
      <div style={{ background: 'var(--surface)', borderRadius: 10, border: '1px solid var(--border)', padding: 16 }}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, textTransform: 'uppercase', letterSpacing: 1 }}>
          Persona Scores
        </div>
        {scoreData.length > 0 ? (
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={scoreData} layout="vertical" margin={{ left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis type="number" tick={{ fill: 'var(--text-muted)', fontSize: 10 }} domain={[0, 100]} />
              <YAxis type="category" dataKey="persona" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: 'var(--surface2)', border: '1px solid var(--border)', fontSize: 12 }}
                formatter={v => [`${v}%`, 'Similarity']}
              />
              <Bar dataKey="score" radius={[0, 4, 4, 0]}>
                {scoreData.map((entry, i) => (
                  <Cell key={i} fill={PERSONA_COLORS[entry.persona] || PERSONA_COLORS.DEFAULT} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div style={{ color: 'var(--text-muted)', fontSize: 13, marginTop: 20 }}>
            Run inference to see persona scores
          </div>
        )}
        <div style={{ marginTop: 12, fontSize: 12 }}>
          <div style={{ color: 'var(--text-muted)', marginBottom: 6 }}>Explicit preferences:</div>
          {Object.entries(profile.explicit_preferences || {}).map(([k, v]) => (
            <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0', borderBottom: '1px solid var(--border)' }}>
              <span style={{ color: 'var(--text-muted)' }}>{k}</span>
              <span style={{ color: 'var(--accent2)' }}>{typeof v === 'number' ? v.toFixed(2) : v}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ── Profiles Tab ──────────────────────────────────────────────────
function ProfilesTab() {
  const [profiles, setProfiles] = useState([])
  const [selected, setSelected] = useState(null)
  const [loading, setLoading] = useState(true)

  const loadProfiles = async () => {
    try {
      const data = await apiFetch('/profiles')
      setProfiles(data)
      if (data.length > 0 && !selected) setSelected(data[0])
    } catch (e) {
      console.error(e)
    }
    setLoading(false)
  }

  useEffect(() => { loadProfiles() }, [])

  return (
    <div style={{ padding: 24, display: 'grid', gridTemplateColumns: '280px 1fr', gap: 20, height: 'calc(100vh - 56px)', overflow: 'hidden' }}>
      <div style={{ overflow: 'auto' }}>
        <div style={{ fontFamily: 'var(--font-display)', fontWeight: 600, fontSize: 16, marginBottom: 14, color: 'var(--text-muted)' }}>
          User Profiles ({profiles.length})
        </div>
        {loading ? <div style={{ color: 'var(--text-muted)' }}>Loading...</div> : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {profiles.map(p => (
              <ProfileCard key={p.user_id} profile={p} selected={selected?.user_id === p.user_id} onSelect={setSelected} />
            ))}
          </div>
        )}
      </div>
      <div style={{ overflow: 'auto' }}>
        {selected && (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
              <div style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 22 }}>
                {selected.display_name}
              </div>
              <PersonaBadge persona={selected.persona || 'DEFAULT'} size="lg" />
            </div>
            <ProfileDetail
              profile={profiles.find(p => p.user_id === selected.user_id)}
              onRefresh={loadProfiles}
            />
          </>
        )}
      </div>
    </div>
  )
}

// ── Chat Tab ──────────────────────────────────────────────────────
function ChatTab() {
  const [profiles, setProfiles] = useState([])
  const [userId, setUserId] = useState('exec-001')
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [budget, setBudget] = useState(1.0)
  const endRef = useRef(null)
  const demoQueries = [
    "Give me a 5-bullet executive summary of this week's platform risks.",
    'Explain distributed tracing with a concrete OpenTelemetry example.',
    'Teach me Kubernetes in simple terms with one analogy.',
    'Compare Postgres vs MongoDB for analytics workload with tradeoffs.',
    'Create a 30-day learning plan for becoming production-ready in SRE.'
  ]

  useEffect(() => {
    apiFetch('/profiles').then(setProfiles).catch(() => {})
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = async (messageOverride = null) => {
    const msg = (messageOverride ?? input).trim()
    if (!msg || loading) return
    setInput('')
    setMessages(m => [...m, { role: 'user', content: msg }])
    setLoading(true)
    try {
      const r = await apiFetch('/chat', {
        method: 'POST',
        headers: { 'X-User-Id': userId },
        body: JSON.stringify({ message: msg, budget_fraction: budget })
      })
      setMessages(m => [...m, {
        role: 'agent', content: r.response, persona: r.persona,
        tier: r.context_tier, interactionId: r.interaction_id
      }])
    } catch (e) {
      setMessages(m => [...m, { role: 'error', content: e.message }])
    }
    setLoading(false)
  }

  const handleFeedback = async (id, score) => {
    await apiFetch('/chat/feedback', {
      method: 'POST', body: JSON.stringify({ interaction_id: id, score })
    })
  }

  const profile = profiles.find(p => p.user_id === userId)
  const persona = profile?.persona || 'DEFAULT'

  return (
    <div style={{ height: 'calc(100vh - 56px)', display: 'flex', flexDirection: 'column', padding: 24, gap: 16 }}>
      {/* Controls */}
      <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <label style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>User</label>
          <select value={userId} onChange={e => { setUserId(e.target.value); setMessages([]) }} style={{
            background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--text)',
            borderRadius: 6, padding: '6px 10px', fontSize: 13, fontFamily: 'var(--font-mono)'
          }}>
            {profiles.map(p => <option key={p.user_id} value={p.user_id}>{p.display_name}</option>)}
          </select>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <label style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>
            Budget fraction (L53): {budget.toFixed(1)}
          </label>
          <input type="range" min="0.1" max="1.0" step="0.1" value={budget}
            onChange={e => setBudget(parseFloat(e.target.value))}
            style={{ width: 160 }}
          />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <label style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>Active Persona</label>
          <PersonaBadge persona={persona} size="lg" />
        </div>
      </div>

      {/* Demo Queries */}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {demoQueries.map((q, i) => (
          <button key={i} onClick={() => send(q)} disabled={loading} style={{
            background: 'var(--surface2)', border: '1px solid var(--border)',
            color: 'var(--text-muted)', borderRadius: 999, padding: '6px 10px',
            cursor: loading ? 'not-allowed' : 'pointer', fontSize: 12, fontFamily: 'var(--font-mono)',
            opacity: loading ? 0.5 : 1
          }}>
            {q.length > 58 ? `${q.slice(0, 58)}...` : q}
          </button>
        ))}
      </div>

      {/* Messages */}
      <div style={{ flex: 1, overflow: 'auto', display: 'flex', flexDirection: 'column', gap: 12 }}>
        {messages.length === 0 && (
          <div style={{ textAlign: 'center', color: 'var(--text-muted)', marginTop: 60, fontSize: 13 }}>
            Start chatting — responses adapt to {profile?.display_name || 'user'}'s profile
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} style={{
            alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
            maxWidth: '72%'
          }}>
            {m.role === 'user' ? (
              <div style={{
                background: 'var(--surface2)', border: '1px solid var(--border)',
                borderRadius: '12px 12px 2px 12px', padding: '10px 14px', fontSize: 14
              }}>
                {m.content}
              </div>
            ) : m.role === 'agent' ? (
              <div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4, display: 'flex', gap: 8 }}>
                  <PersonaBadge persona={m.persona} />
                  <span style={{
                    background: 'var(--surface2)', border: '1px solid var(--border)',
                    borderRadius: 10, padding: '1px 7px', fontSize: 11
                  }}>
                    {m.tier}
                  </span>
                </div>
                <div style={{
                  background: 'var(--surface)', border: '1px solid var(--border)',
                  borderRadius: '2px 12px 12px 12px', padding: '10px 14px', fontSize: 14,
                  lineHeight: 1.6, whiteSpace: 'pre-wrap'
                }}>
                  {m.content}
                </div>
                <div style={{ marginTop: 6, display: 'flex', gap: 6 }}>
                  <button onClick={() => handleFeedback(m.interactionId, 1.0)} style={{
                    background: 'transparent', border: '1px solid var(--border)',
                    borderRadius: 6, padding: '3px 8px', cursor: 'pointer', fontSize: 12,
                    color: 'var(--text-muted)'
                  }}>👍</button>
                  <button onClick={() => handleFeedback(m.interactionId, -1.0)} style={{
                    background: 'transparent', border: '1px solid var(--border)',
                    borderRadius: 6, padding: '3px 8px', cursor: 'pointer', fontSize: 12,
                    color: 'var(--text-muted)'
                  }}>👎</button>
                </div>
              </div>
            ) : (
              <div style={{ color: '#fb7185', fontSize: 13 }}>Error: {m.content}</div>
            )}
          </div>
        ))}
        {loading && (
          <div style={{ alignSelf: 'flex-start', color: 'var(--text-muted)', fontSize: 13 }}>
            <span style={{ animation: 'pulse 1s infinite' }}>● generating...</span>
          </div>
        )}
        <div ref={endRef} />
      </div>

      {/* Input */}
      <div style={{ display: 'flex', gap: 8 }}>
        <input
          value={input} onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send()}
          placeholder="Ask anything — response adapts to persona..."
          style={{
            flex: 1, background: 'var(--surface2)', border: '1px solid var(--border)',
            borderRadius: 8, padding: '10px 14px', color: 'var(--text)', fontSize: 14,
            fontFamily: 'var(--font-mono)', outline: 'none'
          }}
        />
        <button onClick={send} disabled={loading} style={{
          background: 'var(--accent)', color: 'var(--bg)', border: 'none',
          borderRadius: 8, padding: '10px 20px', cursor: 'pointer',
          fontFamily: 'var(--font-mono)', fontWeight: 600, fontSize: 14,
          opacity: loading ? 0.5 : 1
        }}>Send</button>
      </div>
    </div>
  )
}

// ── Compare Tab ───────────────────────────────────────────────────
function CompareTab() {
  const [profiles, setProfiles] = useState([])
  const [selectedIds, setSelectedIds] = useState([])
  const [message, setMessage] = useState('Explain distributed tracing in a microservices architecture.')
  const [results, setResults] = useState([])
  const [running, setRunning] = useState(false)

  useEffect(() => {
    apiFetch('/profiles').then(d => {
      setProfiles(d)
      if (d.length >= 2) setSelectedIds(d.slice(0, 2).map(p => p.user_id))
    }).catch(() => {})
  }, [])

  const toggle = (id) => {
    setSelectedIds(s => s.includes(id) ? s.filter(x => x !== id) : [...s, id].slice(-4))
  }

  const run = async () => {
    if (selectedIds.length < 1) return
    setRunning(true); setResults([])
    try {
      const r = await apiFetch('/chat/compare', {
        method: 'POST',
        body: JSON.stringify({ message, user_ids: selectedIds })
      })
      setResults(r.comparisons || [])
    } catch (e) { console.error(e) }
    setRunning(false)
  }

  return (
    <div style={{ padding: 24 }}>
      <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 18, marginBottom: 16 }}>
        Persona A/B Comparison
      </div>
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>
          Select users (max 4)
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {profiles.map(p => (
            <button key={p.user_id} onClick={() => toggle(p.user_id)} style={{
              background: selectedIds.includes(p.user_id) ? `${PERSONA_COLORS[p.persona || 'DEFAULT']}22` : 'var(--surface)',
              border: `1px solid ${selectedIds.includes(p.user_id) ? PERSONA_COLORS[p.persona || 'DEFAULT'] : 'var(--border)'}`,
              color: 'var(--text)', borderRadius: 8, padding: '6px 14px',
              cursor: 'pointer', fontSize: 13, fontFamily: 'var(--font-mono)'
            }}>
              {p.display_name}
            </button>
          ))}
        </div>
      </div>
      <div style={{ marginBottom: 14 }}>
        <textarea value={message} onChange={e => setMessage(e.target.value)} style={{
          width: '100%', maxWidth: 700, background: 'var(--surface2)',
          border: '1px solid var(--border)', borderRadius: 8, padding: '10px 14px',
          color: 'var(--text)', fontFamily: 'var(--font-mono)', fontSize: 14,
          resize: 'vertical', rows: 3, minHeight: 80
        }} />
      </div>
      <button onClick={run} disabled={running || selectedIds.length < 1} style={{
        background: 'var(--accent2)', color: 'var(--bg)', border: 'none',
        borderRadius: 8, padding: '8px 20px', cursor: 'pointer',
        fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: 14,
        marginBottom: 20, opacity: running ? 0.5 : 1
      }}>
        {running ? '⟳ Running...' : '▶ Compare Responses'}
      </button>
      {results.length > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: 16 }}>
          {results.map((r, i) => (
            <div key={i} style={{ background: 'var(--surface)', border: `1px solid ${PERSONA_COLORS[r.persona]}55`, borderRadius: 10, padding: 16 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
                <span style={{ fontFamily: 'var(--font-display)', fontWeight: 600, fontSize: 14 }}>{r.user_id}</span>
                <PersonaBadge persona={r.persona} />
              </div>
              <div style={{ fontSize: 13, lineHeight: 1.6, whiteSpace: 'pre-wrap', maxHeight: 300, overflow: 'auto' }}>
                {r.response}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Analytics Tab ─────────────────────────────────────────────────
function AnalyticsTab() {
  const [stats, setStats] = useState(null)

  useEffect(() => {
    const load = () => apiFetch('/analytics/stats').then(setStats).catch(() => {})
    load()
    const id = setInterval(load, 4000)
    return () => clearInterval(id)
  }, [])

  if (!stats) return <div style={{ padding: 24, color: 'var(--text-muted)' }}>Loading...</div>

  const personaData = Object.entries(stats.persona_distribution || {}).map(([k, v]) => ({ persona: k, count: v }))
  const tierData = Object.entries(stats.context_tier_distribution || {}).map(([k, v]) => ({ tier: k, count: v }))

  return (
    <div style={{ padding: 24 }}>
      <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 18, marginBottom: 20 }}>Analytics</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 24 }}>
        {[
          { label: 'Total Profiles', val: stats.total_profiles, color: 'var(--accent)' },
          { label: 'Total Interactions', val: stats.total_interactions, color: 'var(--accent2)' },
          { label: 'Personas Active', val: Object.keys(stats.persona_distribution || {}).length, color: 'var(--accent3)' },
        ].map(m => (
          <div key={m.label} style={{ background: 'var(--surface)', borderRadius: 10, border: '1px solid var(--border)', padding: '16px 20px' }}>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>{m.label}</div>
            <div style={{ fontSize: 36, fontFamily: 'var(--font-display)', fontWeight: 800, color: m.color, marginTop: 4 }}>{m.val}</div>
          </div>
        ))}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div style={{ background: 'var(--surface)', borderRadius: 10, border: '1px solid var(--border)', padding: 16 }}>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>Persona Distribution</div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={personaData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="persona" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
              <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
              <Tooltip contentStyle={{ background: 'var(--surface2)', border: '1px solid var(--border)', fontSize: 12 }} />
              <Bar dataKey="count" fill="var(--accent3)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div style={{ background: 'var(--surface)', borderRadius: 10, border: '1px solid var(--border)', padding: 16 }}>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>Context Tier Usage (L53 Budget Integration)</div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={tierData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="tier" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
              <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
              <Tooltip contentStyle={{ background: 'var(--surface2)', border: '1px solid var(--border)', fontSize: 12 }} />
              <Bar dataKey="count" fill="var(--accent2)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}

// ── App ───────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState('profiles')
  const tabs = { profiles: <ProfilesTab />, chat: <ChatTab />, compare: <CompareTab />, analytics: <AnalyticsTab /> }
  return (
    <div style={{ minHeight: '100vh' }}>
      <Header activeTab={tab} setActiveTab={setTab} />
      {tabs[tab]}
    </div>
  )
}
