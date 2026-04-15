import React, { useState, useEffect, useRef, useCallback } from 'react';

const API = process.env.REACT_APP_API_URL || 'http://localhost:8059';

function apiToWs(url) {
  if (url.startsWith('https://')) return `wss://${url.slice('https://'.length)}`;
  if (url.startsWith('http://')) return `ws://${url.slice('http://'.length)}`;
  return `ws://${url}`;
}

const COLORS = {
  orchestrator: '#4ade80',
  researcher:   '#60a5fa',
  analyst:      '#f59e0b',
  session:      '#a78bfa',
  default:      '#94a3b8',
};

const EVENT_BADGE = {
  LLM_CALL:          { bg: '#1e3a5f', fg: '#60a5fa', label: 'LLM' },
  TOOL_CALL:         { bg: '#3b1f00', fg: '#f59e0b', label: 'TOOL' },
  SEND:              { bg: '#0d2d1a', fg: '#4ade80', label: 'SEND' },
  RECEIVE:           { bg: '#1a2d0d', fg: '#86efac', label: 'RECV' },
  ROUTING_DECISION:  { bg: '#2d1a3b', fg: '#c084fc', label: 'ROUTE' },
  ERROR:             { bg: '#3b0d0d', fg: '#f87171', label: 'ERR' },
  SESSION_START:     { bg: '#1a1a2e', fg: '#a78bfa', label: 'START' },
  SESSION_END:       { bg: '#1a1a2e', fg: '#c084fc', label: 'END' },
};

function ns(n) {
  if (!n) return '—';
  if (n < 1_000) return `${n}ns`;
  if (n < 1_000_000) return `${(n/1000).toFixed(1)}µs`;
  if (n < 1_000_000_000) return `${(n/1_000_000).toFixed(1)}ms`;
  return `${(n/1_000_000_000).toFixed(2)}s`;
}

function Badge({ type }) {
  const b = EVENT_BADGE[type] || { bg: '#1e293b', fg: '#94a3b8', label: type?.slice(0,4) };
  return (
    <span style={{
      background: b.bg, color: b.fg, border: `1px solid ${b.fg}40`,
      fontFamily: 'IBM Plex Mono', fontSize: 9, fontWeight: 700,
      padding: '2px 5px', borderRadius: 3, letterSpacing: 0.5,
    }}>{b.label}</span>
  );
}

function HotspotBar({ agent, duration, max }) {
  const pct = max > 0 ? (duration / max) * 100 : 0;
  const color = COLORS[agent] || COLORS.default;
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
        <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 11, color }}>{agent}</span>
        <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 10, color: '#64748b' }}>{ns(duration)}</span>
      </div>
      <div style={{ background: '#1e293b', borderRadius: 2, height: 6, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, background: color, height: '100%',
          borderRadius: 2, transition: 'width 0.4s ease' }}/>
      </div>
    </div>
  );
}

function TraceRow({ record, onSelect, selected }) {
  const color = COLORS[record.agent_name] || COLORS.default;
  const isSelected = selected?.span_id === record.span_id;
  return (
    <div onClick={() => onSelect(record)} style={{
      display: 'grid', gridTemplateColumns: '90px 110px 70px 90px 1fr',
      gap: 8, padding: '6px 12px', cursor: 'pointer',
      background: isSelected ? '#1e293b' : 'transparent',
      borderLeft: `2px solid ${isSelected ? color : 'transparent'}`,
      borderBottom: '1px solid #1e293b08',
      transition: 'background 0.15s',
    }}>
      <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#475569' }}>
        {record.span_id?.slice(0,8)}
      </span>
      <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 10, color }}>{record.agent_name}</span>
      <Badge type={record.event_type}/>
      <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 10, color: '#64748b' }}>
        {ns(record.duration_ns)}
      </span>
      <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#334155',
        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {JSON.stringify(record.payload || {}).slice(0, 80)}
      </span>
    </div>
  );
}

function Panel({ title, children, flex }) {
  return (
    <div style={{
      background: '#111827', border: '1px solid #1f2937', borderRadius: 8,
      overflow: 'hidden', display: 'flex', flexDirection: 'column', flex,
    }}>
      <div style={{
        padding: '8px 14px', background: '#0d1117', borderBottom: '1px solid #1f2937',
        fontFamily: 'IBM Plex Mono', fontSize: 10, fontWeight: 700, color: '#475569',
        letterSpacing: 1, textTransform: 'uppercase',
      }}>{title}</div>
      <div style={{ flex: 1, overflow: 'auto', padding: 14 }}>{children}</div>
    </div>
  );
}

export default function App() {
  const [query, setQuery] = useState('What are the main challenges in enterprise AI deployment?');
  const [running, setRunning] = useState(false);
  const [liveRecords, setLiveRecords] = useState([]);
  const [result, setResult] = useState(null);
  const [selected, setSelected] = useState(null);
  const [traceList, setTraceList] = useState([]);
  const [activeTrace, setActiveTrace] = useState(null);
  const [activeRecords, setActiveRecords] = useState([]);
  const wsRef = useRef(null);
  const listRef = useRef(null);

  // WebSocket for live trace stream
  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket(`${apiToWs(API)}/ws/traces`);
      wsRef.current = ws;
      ws.onmessage = (e) => {
        try {
          const rec = JSON.parse(e.data);
          setLiveRecords(prev => [...prev.slice(-199), rec]);
          if (listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight;
        } catch {}
      };
      ws.onclose = () => setTimeout(connect, 2000);
    };
    connect();
    return () => wsRef.current?.close();
  }, []);

  // Load trace list on mount
  useEffect(() => {
    fetch(`${API}/api/traces`)
      .then(r => r.json())
      .then(setTraceList)
      .catch(() => {});
  }, [result]);

  const loadTrace = async (traceId) => {
    setActiveTrace(traceId);
    const recs = await fetch(`${API}/api/traces/${traceId}`).then(r => r.json());
    setActiveRecords(recs);
  };

  const runPipeline = async () => {
    if (!query.trim() || running) return;
    setRunning(true);
    setLiveRecords([]);
    setResult(null);
    setSelected(null);
    try {
      const res = await fetch(`${API}/api/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, test_patterns: ['.+'] }),
      });
      const data = await res.json();
      setResult(data);
      // Keep stream panel populated even if WS is unavailable.
      if (data?.trace_id) {
        setActiveTrace(data.trace_id);
        const recs = await fetch(`${API}/api/traces/${data.trace_id}`).then(r => r.json());
        setActiveRecords(recs);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setRunning(false);
    }
  };

  const hotspots = result?.eval_report?.agent_hotspots || {};
  const maxHot = Math.max(...Object.values(hotspots), 1);
  const critPath = result?.eval_report?.critical_path || [];
  const evalResults = result?.eval_report?.test_results || [];
  const displayRecords = activeRecords.length ? activeRecords : liveRecords;

  return (
    <div style={{
      background: '#0d0f14', minHeight: '100vh', color: '#e2e8f0',
      fontFamily: 'IBM Plex Sans, sans-serif', padding: 20,
    }}>
      {/* Header */}
      <div style={{ marginBottom: 20, display: 'flex', alignItems: 'center', gap: 16 }}>
        <div>
          <div style={{ fontFamily: 'IBM Plex Mono', fontSize: 10, color: '#4ade80',
            fontWeight: 700, letterSpacing: 2, marginBottom: 2 }}>VAIA · L59</div>
          <h1 style={{ margin: 0, fontSize: 22, fontWeight: 600, color: '#f8fafc',
            fontFamily: 'IBM Plex Sans' }}>MAS Evaluation & Debugging</h1>
        </div>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8, alignItems: 'center' }}>
          <div style={{
            background: running ? '#1e3a1e' : '#0d2d1a',
            border: `1px solid ${running ? '#4ade80' : '#166534'}`,
            borderRadius: 20, padding: '4px 12px',
            fontFamily: 'IBM Plex Mono', fontSize: 10, color: running ? '#4ade80' : '#166534',
          }}>
            {running ? '● RECORDING' : '○ IDLE'}
          </div>
          <div style={{
            background: '#111827', border: '1px solid #1f2937', borderRadius: 20,
            padding: '4px 12px', fontFamily: 'IBM Plex Mono', fontSize: 10, color: '#475569',
          }}>{liveRecords.length} events</div>
        </div>
      </div>

      {/* Query bar */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 20 }}>
        <input
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && runPipeline()}
          placeholder="Research query…"
          style={{
            flex: 1, background: '#111827', border: '1px solid #1f2937',
            borderRadius: 6, padding: '10px 14px', color: '#e2e8f0',
            fontFamily: 'IBM Plex Mono', fontSize: 12, outline: 'none',
          }}
        />
        <button onClick={runPipeline} disabled={running} style={{
          background: running ? '#1e3a1e' : '#166534',
          border: 'none', borderRadius: 6, padding: '10px 24px',
          color: running ? '#4ade80' : '#86efac', fontFamily: 'IBM Plex Mono',
          fontSize: 12, fontWeight: 700, cursor: running ? 'not-allowed' : 'pointer',
          transition: 'background 0.2s',
        }}>
          {running ? 'RUNNING…' : '▶ RUN'}
        </button>
      </div>

      {/* Main layout: left panels + right detail */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: 14 }}>
        {/* Left column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>

          {/* Live trace stream */}
          <Panel title={`Trace Stream ${activeTrace ? `· ${activeTrace.slice(0,8)}` : '· LIVE'}`} flex="0 0 280px">
            {/* Column headers */}
            <div style={{
              display: 'grid', gridTemplateColumns: '90px 110px 70px 90px 1fr',
              gap: 8, padding: '4px 12px', marginBottom: 4,
              borderBottom: '1px solid #1f2937',
            }}>
              {['SPAN', 'AGENT', 'TYPE', 'DUR', 'PAYLOAD'].map(h => (
                <span key={h} style={{
                  fontFamily: 'IBM Plex Mono', fontSize: 8, fontWeight: 700,
                  color: '#334155', letterSpacing: 0.5,
                }}>{h}</span>
              ))}
            </div>
            <div ref={listRef} style={{ height: 200, overflow: 'auto' }}>
              {displayRecords.length === 0 ? (
                <div style={{ padding: 20, textAlign: 'center',
                  fontFamily: 'IBM Plex Mono', fontSize: 11, color: '#334155' }}>
                  Waiting for traces…
                </div>
              ) : displayRecords.map((r, i) => (
                <TraceRow key={i} record={r} onSelect={setSelected} selected={selected}/>
              ))}
            </div>
          </Panel>

          {/* Output */}
          {result && (
            <Panel title="Final Output" flex="0 0 auto">
              <div style={{
                fontFamily: 'IBM Plex Mono', fontSize: 11, color: '#cbd5e1',
                lineHeight: 1.7, whiteSpace: 'pre-wrap',
              }}>
                {result.output}
              </div>
            </Panel>
          )}

          {/* Trace history */}
          <Panel title="Trace History" flex="1 1 auto">
            {traceList.length === 0 ? (
              <div style={{ fontFamily: 'IBM Plex Mono', fontSize: 11, color: '#334155' }}>
                No traces yet
              </div>
            ) : traceList.map((t, i) => (
              <div key={i} onClick={() => loadTrace(t.trace_id)} style={{
                padding: '6px 0', borderBottom: '1px solid #1f2937',
                cursor: 'pointer', display: 'flex', justifyContent: 'space-between',
                background: activeTrace === t.trace_id ? '#1e293b' : 'transparent',
                padding: '6px 8px', borderRadius: 4,
              }}>
                <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 10, color: '#60a5fa' }}>
                  {t.trace_id?.slice(0,12)}
                </span>
                <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 10, color: '#475569' }}>
                  {t.spans} spans · {ns(t.total_ns)}
                </span>
              </div>
            ))}
          </Panel>
        </div>

        {/* Right column: analysis panels */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>

          {/* Eval results */}
          <Panel title="Eval Results" flex="0 0 auto">
            {!result ? (
              <div style={{ fontFamily: 'IBM Plex Mono', fontSize: 11, color: '#334155' }}>
                Run a query to see results
              </div>
            ) : (
              <>
                <div style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
                  <div style={{
                    flex: 1, background: '#0d2d1a', border: '1px solid #166534',
                    borderRadius: 6, padding: '8px 12px', textAlign: 'center',
                  }}>
                    <div style={{ fontFamily: 'IBM Plex Mono', fontSize: 22,
                      fontWeight: 700, color: '#4ade80' }}>{result.eval_report?.passed}</div>
                    <div style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#166534' }}>PASSED</div>
                  </div>
                  <div style={{
                    flex: 1, background: result.eval_report?.failed > 0 ? '#3b0d0d' : '#111827',
                    border: `1px solid ${result.eval_report?.failed > 0 ? '#991b1b' : '#1f2937'}`,
                    borderRadius: 6, padding: '8px 12px', textAlign: 'center',
                  }}>
                    <div style={{ fontFamily: 'IBM Plex Mono', fontSize: 22,
                      fontWeight: 700, color: result.eval_report?.failed > 0 ? '#f87171' : '#475569' }}>
                      {result.eval_report?.failed}
                    </div>
                    <div style={{ fontFamily: 'IBM Plex Mono', fontSize: 9,
                      color: result.eval_report?.failed > 0 ? '#991b1b' : '#334155' }}>FAILED</div>
                  </div>
                </div>
                {evalResults.map((r, i) => (
                  <div key={i} style={{
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    padding: '5px 0', borderBottom: '1px solid #1f2937',
                  }}>
                    <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 10, color: '#94a3b8' }}>
                      {r.test_name}
                    </span>
                    <div style={{ display: 'flex', gap: 4 }}>
                      <span style={{
                        fontFamily: 'IBM Plex Mono', fontSize: 9,
                        color: r.correctness ? '#4ade80' : '#f87171',
                      }}>{r.correctness ? '✓' : '✗'} correct</span>
                      <span style={{
                        fontFamily: 'IBM Plex Mono', fontSize: 9,
                        color: r.efficiency ? '#4ade80' : '#f87171',
                      }}>{r.efficiency ? '✓' : '✗'} efficient</span>
                    </div>
                  </div>
                ))}
              </>
            )}
          </Panel>

          {/* Agent hotspots */}
          <Panel title="Agent Hotspots" flex="0 0 auto">
            {Object.keys(hotspots).length === 0 ? (
              <div style={{ fontFamily: 'IBM Plex Mono', fontSize: 11, color: '#334155' }}>
                Run analysis to see hotspots
              </div>
            ) : Object.entries(hotspots).map(([agent, dur]) => (
              <HotspotBar key={agent} agent={agent} duration={dur} max={maxHot}/>
            ))}
            {result && (
              <div style={{ marginTop: 10, paddingTop: 8, borderTop: '1px solid #1f2937' }}>
                <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#475569' }}>
                  total tokens: {result.eval_report?.total_tokens?.toLocaleString() || 0}
                </span>
              </div>
            )}
          </Panel>

          {/* Critical path */}
          <Panel title="Critical Path" flex="0 0 auto">
            {critPath.length === 0 ? (
              <div style={{ fontFamily: 'IBM Plex Mono', fontSize: 11, color: '#334155' }}>
                No data
              </div>
            ) : critPath.map((spanId, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', marginBottom: 4 }}>
                {i > 0 && <div style={{ width: 2, height: 12, background: '#f59e0b',
                  marginRight: 8, marginLeft: 4 }}/>}
                <div style={{
                  background: '#3b2000', border: '1px solid #92400e',
                  borderRadius: 4, padding: '3px 8px',
                  fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#fbbf24',
                }}>{spanId.slice(0,8)}</div>
              </div>
            ))}
          </Panel>

          {/* Selected span detail */}
          {selected && (
            <Panel title="Span Detail" flex="0 0 auto">
              {[
                ['span_id', selected.span_id?.slice(0,16)],
                ['agent', selected.agent_name],
                ['event', selected.event_type],
                ['duration', ns(selected.duration_ns)],
                ['modality', selected.modality],
                ['parent', selected.parent_id?.slice(0,12) || '—'],
              ].map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between',
                  padding: '3px 0', borderBottom: '1px solid #0f172a' }}>
                  <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#475569' }}>{k}</span>
                  <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#cbd5e1' }}>{v}</span>
                </div>
              ))}
            </Panel>
          )}
        </div>
      </div>
    </div>
  );
}
