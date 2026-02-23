import React, { useState, useCallback, useRef } from 'react';

const API = process.env.REACT_APP_API_URL || 'http://localhost:8043';

const STATE_CONFIG = {
  idle:        { label: 'Idle',        color: '#4a5568', bg: '#1a202c', icon: '●' },
  planning:    { label: 'Planning',    color: '#68d391', bg: '#1c3c2e', icon: '◈' },
  retrieving:  { label: 'Retrieving', color: '#63b3ed', bg: '#1a2c3d', icon: '⟳' },
  validating:  { label: 'Validating', color: '#f6ad55', bg: '#3d2c1a', icon: '⊛' },
  synthesizing:{ label: 'Synthesizing',color: '#d6bcfa', bg: '#2d1a3d', icon: '✦' },
  tracing:     { label: 'Tracing',    color: '#fc8181', bg: '#3d1a1a', icon: '◎' },
  done:        { label: 'Done',        color: '#68d391', bg: '#1c3c2e', icon: '✓' },
  error:       { label: 'Error',       color: '#fc8181', bg: '#3d1a1a', icon: '✕' },
};

const DEMO_QUERIES = [
  "What was our Q3 2025 revenue growth?",
  "Describe the AI deployment guidelines for production models.",
  "What is the incident response protocol for P1 issues?",
  "How many GPU clusters does our ML infrastructure have?",
  "What are the key milestones in the 2026 product roadmap?",
];

function PipelineStages({ currentState }) {
  const stages = ['planning', 'retrieving', 'validating', 'synthesizing', 'tracing', 'done'];
  const stageIndex = stages.indexOf(currentState);

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 4, flexWrap: 'wrap', margin: '12px 0' }}>
      {stages.map((s, i) => {
        const cfg = STATE_CONFIG[s];
        const active = s === currentState;
        const done = stageIndex > i;
        return (
          <React.Fragment key={s}>
            <div style={{
              padding: '4px 10px', borderRadius: 6,
              fontSize: 11, fontFamily: "'IBM Plex Mono', monospace",
              background: active ? cfg.bg : done ? '#1a2a1a' : '#161b22',
              color: active ? cfg.color : done ? '#4caf50' : '#4a5568',
              border: `1px solid ${active ? cfg.color + '66' : done ? '#2d4a2d' : '#2a2f3a'}`,
              transition: 'all 0.3s ease',
              fontWeight: active ? 600 : 400,
            }}>
              {cfg.icon} {cfg.label}
            </div>
            {i < stages.length - 1 && (
              <div style={{ color: done || active ? '#4a5568' : '#2a2f3a', fontSize: 10 }}>›</div>
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}

function CitationCard({ citation, index, query }) {
  const params = new URLSearchParams();
  if (query) params.set('query', query);
  if (citation.text_snippet) params.set('snippet', citation.text_snippet);
  const docUrl = `${API}/documents/source/${encodeURIComponent(citation.source)}${params.toString() ? '?' + params.toString() : ''}`;
  return (
    <div style={{
      background: '#161b22', borderRadius: 8, padding: '10px 12px',
      border: '1px solid #2a2f3a', marginBottom: 8,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
        <span style={{
          background: '#d6bcfa22', color: '#d6bcfa', border: '1px solid #d6bcfa44',
          borderRadius: 4, padding: '1px 6px', fontSize: 11, fontFamily: "'IBM Plex Mono', monospace",
        }}>{citation.marker}</span>
        <a
          href={docUrl}
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: '#63b3ed', fontSize: 11, textDecoration: 'none' }}
          title="Open source content (knowledge base text)"
        >
          {citation.source}
        </a>
      </div>
      <p style={{ color: '#8892a4', fontSize: 12, margin: 0, lineHeight: 1.5 }}>
        {citation.text_snippet}...
      </p>
    </div>
  );
}

function QualityBar({ score }) {
  if (score == null || score <= 0) {
    return (
      <div style={{ marginTop: 8 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
          <span style={{ color: '#8892a4', fontSize: 11 }}>Quality Score</span>
          <span style={{ color: '#4a5568', fontSize: 12, fontFamily: "'IBM Plex Mono', monospace" }}>—</span>
        </div>
        <div style={{ background: '#2a2f3a', borderRadius: 3, height: 4, overflow: 'hidden' }}>
          <div style={{ width: '0%', height: '100%', background: '#4a5568', borderRadius: 3 }}/>
        </div>
      </div>
    );
  }
  const pct = Math.round(score * 100);
  const color = pct >= 70 ? '#68d391' : pct >= 50 ? '#f6ad55' : '#fc8181';
  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ color: '#8892a4', fontSize: 11 }}>Quality Score</span>
        <span style={{ color, fontSize: 12, fontFamily: "'IBM Plex Mono', monospace", fontWeight: 600 }}>{pct}%</span>
      </div>
      <div style={{ background: '#2a2f3a', borderRadius: 3, height: 4, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 3, transition: 'width 0.5s ease' }}/>
      </div>
    </div>
  );
}

function HistoryItem({ item, onClick }) {
  const state = item.final_state || 'done';
  const cfg = STATE_CONFIG[state] || STATE_CONFIG.done;
  const ts = new Date(item.start_time * 1000).toLocaleTimeString();
  return (
    <div onClick={onClick} style={{
      background: '#161b22', borderRadius: 8, padding: '10px 12px',
      border: `1px solid #2a2f3a`, marginBottom: 6, cursor: 'pointer',
      transition: 'border-color 0.2s',
    }}
    onMouseEnter={e => e.currentTarget.style.borderColor = cfg.color + '66'}
    onMouseLeave={e => e.currentTarget.style.borderColor = '#2a2f3a'}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ color: cfg.color, fontSize: 10, fontFamily: "'IBM Plex Mono', monospace" }}>
          {cfg.icon} {state.toUpperCase()}
        </span>
        <span style={{ color: '#4a5568', fontSize: 10 }}>{ts}</span>
      </div>
      <p style={{ color: '#8892a4', fontSize: 12, margin: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {item.result?.query || item.events?.[0]?.input?.query || 'Query'}
      </p>
      <p style={{ color: '#4a5568', fontSize: 10, margin: '4px 0 0', fontFamily: "'IBM Plex Mono', monospace" }}>
        {item.trace_id?.slice(0, 8)}… · {Math.round(item.duration_ms)}ms
      </p>
    </div>
  );
}

export default function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [currentState, setCurrentState] = useState('idle');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [activeTab, setActiveTab] = useState('response');
  const [selectedTrace, setSelectedTrace] = useState(null);
  const abortRef = useRef(null);

  const fetchHistory = useCallback(async () => {
    try {
      const r = await fetch(`${API}/history?n=15`);
      if (r.ok) setHistory(await r.json());
    } catch (_) {}
  }, []);

  const fetchTrace = useCallback(async (trace_id) => {
    try {
      const r = await fetch(`${API}/audit/${trace_id}`);
      if (r.ok) setSelectedTrace(await r.json());
    } catch (_) {}
  }, []);

  const submit = useCallback(async () => {
    if (!query.trim() || loading) return;
    setLoading(true); setError(null); setResult(null); setCurrentState('planning');

    // Animate only up to "synthesizing" so we don't show "tracing" while waiting for API
    const states = ['planning', 'retrieving', 'validating', 'synthesizing'];
    let i = 0;
    const interval = setInterval(() => {
      i = (i + 1) % states.length;
      setCurrentState(states[i]);
    }, 1200);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 600000); // 10 min — pipeline can be slow with retries

    try {
      const resp = await fetch(`${API}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim(), collection: 'enterprise_kb', max_results: 3 }),
        signal: controller.signal,
      });
      clearInterval(interval);
      clearTimeout(timeoutId);
      let data;
      try {
        data = await resp.json();
      } catch (_) {
        data = {};
      }
      if (!resp.ok) throw new Error(data.detail || data.error || `HTTP ${resp.status}`);
      if (!data.trace_id && data.response === undefined) {
        setError('Empty response from server. Try again.');
        setCurrentState('error');
        return;
      }
      const state = (data.pipeline_state != null && String(data.pipeline_state).toLowerCase()) || 'done';
      setResult(data);
      setCurrentState(state === 'error' ? 'error' : 'done');
      await fetchHistory();
    } catch (e) {
      clearInterval(interval);
      clearTimeout(timeoutId);
      if (e.name === 'AbortError') {
        setError('Request timed out. The pipeline is taking longer than 10 minutes. Try again or use a shorter query.');
      } else {
        setError(e.message);
      }
      setCurrentState('error');
    } finally { setLoading(false); }
  }, [query, loading, fetchHistory]);

  React.useEffect(() => { fetchHistory(); }, [fetchHistory]);

  const styles = {
    app: { minHeight: '100vh', background: '#0f1117', fontFamily: "'IBM Plex Sans', sans-serif", color: '#e2e8f0', display: 'flex', flexDirection: 'column' },
    header: { background: '#161b22', borderBottom: '1px solid #2a2f3a', padding: '14px 24px', display: 'flex', alignItems: 'center', gap: 16 },
    pill: { background: '#68d39122', color: '#68d391', border: '1px solid #68d39144', borderRadius: 20, padding: '2px 10px', fontSize: 11, fontFamily: "'IBM Plex Mono', monospace" },
    body: { display: 'flex', flex: 1, overflow: 'hidden' },
    sidebar: { width: 260, background: '#161b22', borderRight: '1px solid #2a2f3a', padding: 16, overflowY: 'auto' },
    main: { flex: 1, padding: 24, overflowY: 'auto' },
    inputRow: { display: 'flex', gap: 10, marginBottom: 12 },
    input: { flex: 1, background: '#161b22', border: '1px solid #2a2f3a', borderRadius: 8, padding: '10px 14px', color: '#e2e8f0', fontSize: 14, outline: 'none', fontFamily: "'IBM Plex Sans', sans-serif" },
    btn: { background: '#68d391', color: '#0f1117', border: 'none', borderRadius: 8, padding: '10px 20px', cursor: 'pointer', fontSize: 13, fontWeight: 700 },
    tabs: { display: 'flex', gap: 8, marginBottom: 16 },
    tab: (active) => ({ background: active ? '#2a2f3a' : 'transparent', color: active ? '#e2e8f0' : '#4a5568', border: '1px solid', borderColor: active ? '#3a3f4a' : 'transparent', borderRadius: 6, padding: '5px 14px', cursor: 'pointer', fontSize: 12 }),
    card: { background: '#161b22', borderRadius: 10, border: '1px solid #2a2f3a', padding: 18, marginBottom: 16 },
    meta: { display: 'flex', gap: 16, flexWrap: 'wrap' },
    metaItem: (color) => ({ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color }),
    responseText: { color: '#d1d5db', fontSize: 14, lineHeight: 1.8, whiteSpace: 'pre-wrap' },
    demoBtn: (i) => ({ background: '#161b22', border: '1px solid #2a2f3a', borderRadius: 6, padding: '6px 10px', cursor: 'pointer', color: '#8892a4', fontSize: 12, textAlign: 'left', marginBottom: 6, width: '100%' }),
  };

  return (
    <div style={styles.app}>
      {/* Header */}
      <div style={styles.header}>
        <div>
          <div style={{ fontSize: 16, fontWeight: 700 }}>Agentic RAG Pipeline</div>
          <div style={{ fontSize: 11, color: '#4a5568' }}>L43 · End-to-End · VAIA Curriculum</div>
        </div>
        <div style={styles.pill}>PLANNER → RETRIEVER → VALIDATOR → SYNTHESIZER</div>
        {result && <div style={{ ...styles.pill, marginLeft: 'auto', background: '#63b3ed22', color: '#63b3ed', borderColor: '#63b3ed44' }}>
          {(result.latency_ms > 0 ? result.latency_ms + 'ms' : '—')} · {(result.tokens_used > 0 ? result.tokens_used + ' tokens' : '—')}
        </div>}
      </div>

      <div style={styles.body}>
        {/* Sidebar */}
        <div style={styles.sidebar}>
          <div style={{ fontSize: 11, color: '#4a5568', marginBottom: 10, textTransform: 'uppercase', letterSpacing: 1 }}>Demo Queries</div>
          {DEMO_QUERIES.map((q, i) => (
            <button key={i} style={styles.demoBtn(i)} onClick={() => setQuery(q)}>
              {q}
            </button>
          ))}

          <div style={{ fontSize: 11, color: '#4a5568', margin: '20px 0 10px', textTransform: 'uppercase', letterSpacing: 1 }}>
            Recent History
            <button onClick={fetchHistory} style={{ float: 'right', background: 'none', border: 'none', color: '#4a5568', cursor: 'pointer', fontSize: 10 }}>↻</button>
          </div>
          {history.length === 0 && <div style={{ color: '#2a2f3a', fontSize: 12 }}>No history yet</div>}
          {history.map((h, i) => (
            <HistoryItem key={i} item={h} onClick={() => { setSelectedTrace(h); setActiveTab('trace'); }} />
          ))}
        </div>

        {/* Main */}
        <div style={styles.main}>
          {/* Input */}
          <div style={styles.inputRow}>
            <input
              style={styles.input}
              placeholder="Ask a question about enterprise knowledge..."
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && submit()}
            />
            <button style={{ ...styles.btn, opacity: loading ? 0.6 : 1 }} onClick={submit} disabled={loading}>
              {loading ? '⟳' : 'Query'}
            </button>
          </div>

          {/* Pipeline stages */}
          <PipelineStages currentState={currentState} />

          {error && (
            <div style={{ background: '#3d1a1a', border: '1px solid #fc818166', borderRadius: 8, padding: '12px 16px', color: '#fc8181', fontSize: 13, marginBottom: 16 }}>
              ✕ {error}
            </div>
          )}

          {result && (
            <>
              {/* Tabs */}
              <div style={styles.tabs}>
                {['response', 'citations', 'trace'].map(t => (
                  <button key={t} style={styles.tab(activeTab === t)} onClick={() => { setActiveTab(t); if (t === 'trace') fetchTrace(result.trace_id); }}>
                    {t.charAt(0).toUpperCase() + t.slice(1)}
                    {t === 'citations' && ` (${result.citations?.length || 0})`}
                  </button>
                ))}
              </div>

              {/* Response tab */}
              {activeTab === 'response' && (
                <div style={styles.card}>
                  <div style={styles.meta}>
                    <span style={styles.metaItem('#68d391')}>✓ {(STATE_CONFIG[result.pipeline_state] || {}).label || result.pipeline_state}</span>
                    <span style={styles.metaItem('#63b3ed')}>{result.latency_ms > 0 ? result.latency_ms + 'ms' : '—'}</span>
                    <span style={styles.metaItem('#d6bcfa')}>{result.tokens_used > 0 ? result.tokens_used + ' tokens' : '—'}</span>
                    <span style={styles.metaItem('#4a5568')}>trace: {result.trace_id?.slice(0, 12)}…</span>
                  </div>
                  <QualityBar score={result.quality_score != null && result.quality_score > 0 ? result.quality_score : null} />
                  <hr style={{ border: 'none', borderTop: '1px solid #2a2f3a', margin: '14px 0' }}/>
                  <div style={styles.responseText}>{result.response}</div>
                </div>
              )}

              {/* Citations tab */}
              {activeTab === 'citations' && (
                <div>
                  {result.citations?.length === 0
                    ? <div style={{ color: '#4a5568', fontSize: 13 }}>No citations extracted.</div>
                    : result.citations?.map((c, i) => <CitationCard key={i} citation={c} index={i} query={result.query}/>)
                  }
                </div>
              )}

              {/* Trace tab */}
              {activeTab === 'trace' && (
                <div style={styles.card}>
                  {!selectedTrace
                    ? <div style={{ color: '#4a5568', fontSize: 13 }}>Loading trace…</div>
                    : (
                      <>
                        <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: '#8892a4', marginBottom: 12 }}>
                          trace_id: {selectedTrace.trace_id} · {Math.round(selectedTrace.duration_ms)}ms
                        </div>
                        {selectedTrace.events?.map((ev, i) => {
                          const cfg = STATE_CONFIG[ev.state] || {};
                          return (
                            <div key={i} style={{ borderLeft: `2px solid ${cfg.color || '#2a2f3a'}`, paddingLeft: 12, marginBottom: 12 }}>
                              <div style={{ display: 'flex', gap: 10, marginBottom: 4 }}>
                                <span style={{ color: cfg.color, fontSize: 11, fontWeight: 600 }}>{ev.agent?.toUpperCase()}</span>
                                <span style={{ color: '#4a5568', fontSize: 11, fontFamily: "'IBM Plex Mono', monospace" }}>{ev.latency_ms}ms</span>
                                {ev.attempt > 1 && <span style={{ color: '#f6ad55', fontSize: 10 }}>attempt {ev.attempt}</span>}
                              </div>
                              <pre style={{ background: '#0f1117', borderRadius: 6, padding: 10, fontSize: 10, color: '#8892a4', overflow: 'auto', margin: 0, fontFamily: "'IBM Plex Mono', monospace" }}>
                                {JSON.stringify(ev.output, null, 2)}
                              </pre>
                            </div>
                          );
                        })}
                      </>
                    )
                  }
                </div>
              )}
            </>
          )}

          {!result && !loading && !error && (
            <div style={{ textAlign: 'center', padding: '60px 20px', color: '#2a2f3a' }}>
              <div style={{ fontSize: 48, marginBottom: 16 }}>◈</div>
              <div style={{ fontSize: 14 }}>Enter a query to run the Agentic RAG pipeline</div>
              <div style={{ fontSize: 12, marginTop: 8 }}>Planner → Retriever → Validator → Synthesizer</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
