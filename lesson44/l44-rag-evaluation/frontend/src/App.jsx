import React, { useState, useEffect, useCallback } from 'react';
import {
  RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Cell, ReferenceLine
} from 'recharts';
import MetricCard from './components/MetricCard';
import QuestionTable from './components/QuestionTable';

const API = 'http://localhost:8044';

const METRIC_LABELS = {
  faithfulness: 'Faithfulness',
  answer_relevancy: 'Answer Relev.',
  context_recall: 'Ctx Recall',
  context_precision: 'Ctx Precision',
};
const THRESHOLDS = { faithfulness: 0.85, answer_relevancy: 0.80, context_recall: 0.75, context_precision: 0.70 };
const COLORS = { faithfulness: '#3b82f6', answer_relevancy: '#f97316', context_recall: '#22c55e', context_precision: '#7c3aed' };

function StateIndicator({ state }) {
  const map = {
    idle: ['#94a3b8', '⏸', 'Idle'],
    dataset_loading: ['#3b82f6', '📂', 'Loading Dataset'],
    rag_executing: ['#22c55e', '⚙️', 'Running RAG'],
    traces_collected: ['#f59e0b', '📦', 'Traces Collected'],
    evaluating: ['#f97316', '🧮', 'Evaluating…'],
    gate_check: ['#7c3aed', '🚦', 'Gate Check'],
    report_generated: ['#06b6d4', '📊', 'Report Ready'],
    deployed: ['#22c55e', '✅', 'DEPLOYED (PASS)'],
    blocked: ['#ef4444', '🚫', 'BLOCKED (FAIL)'],
    error: ['#ef4444', '❌', 'Error'],
  };
  const [color, icon, label] = map[state] || ['#94a3b8', '❓', state];
  const running = ['dataset_loading', 'rag_executing', 'traces_collected', 'evaluating', 'gate_check'].includes(state);
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      {running && (
        <div style={{
          width: 10, height: 10, borderRadius: '50%', background: color,
          animation: 'pulse 1.2s ease-in-out infinite',
        }}/>
      )}
      <span style={{ fontSize: 16 }}>{icon}</span>
      <span style={{ fontSize: 13, fontWeight: 700, color }}>{label}</span>
    </div>
  );
}

export default function App() {
  const [report, setReport] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiUnreachable, setApiUnreachable] = useState(false);
  const [numQ, setNumQ] = useState(8);
  const [runName, setRunName] = useState('eval-run-1');
  const [activeTab, setTab] = useState('overview');
  const [polling, setPolling] = useState(false);

  const fetchLatest = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/evaluations/latest`);
      setApiUnreachable(false);
      if (res.ok) {
        const data = await res.json();
        setReport(data);
        // Stop polling when terminal state reached
        if (['deployed', 'blocked', 'error', 'report_generated'].includes(data.state)) {
          setPolling(false);
          setLoading(false);
        }
      }
    } catch (e) {
      setApiUnreachable(true);
    }
  }, []);

  const fetchHistory = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/evaluations`);
      setApiUnreachable(false);
      if (res.ok) setHistory(await res.json());
    } catch (e) {
      setApiUnreachable(true);
    }
  }, []);

  // Poll when evaluation is running
  useEffect(() => {
    if (!polling) return;
    const id = setInterval(fetchLatest, 2500);
    return () => clearInterval(id);
  }, [polling, fetchLatest]);

  useEffect(() => {
    fetchLatest();
    fetchHistory();
  }, [fetchLatest, fetchHistory]);

  const startEval = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/api/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          run_name: runName,
          num_test_questions: numQ,
          use_synthetic_dataset: true,
        }),
      });
      setApiUnreachable(false);
      if (!res.ok) throw new Error(await res.text());
      setPolling(true);
      setTimeout(fetchHistory, 3000);
    } catch (e) {
      const msg = e.message || String(e);
      if (msg.includes('Failed to fetch') || msg.includes('NetworkError') || msg.includes('Load failed')) {
        setApiUnreachable(true);
        setError('Cannot reach API at ' + API + '. Is the backend running? Run: ./start.sh from the project root.');
      } else {
        setError(msg);
      }
      setLoading(false);
    }
  };

  // Build radar data
  const radarData = report?.aggregate ? Object.entries(METRIC_LABELS).map(([k, label]) => ({
    subject: label,
    score: Math.round((report.aggregate[k] || 0) * 100),
    threshold: Math.round((THRESHOLDS[k] || 0.75) * 100),
  })) : [];

  // Build bar data for per-question view
  const barData = (report?.items || []).map((item, i) => ({
    name: `Q${i + 1}`,
    faithfulness: Math.round((item.scores?.faithfulness || 0) * 100),
    answer_relevancy: Math.round((item.scores?.answer_relevancy || 0) * 100),
    context_recall: Math.round((item.scores?.context_recall || 0) * 100),
    context_precision: Math.round((item.scores?.context_precision || 0) * 100),
  }));

  const gateColor = report?.gate_passed === true ? '#22c55e' : report?.gate_passed === false ? '#ef4444' : '#94a3b8';
  const gateBg = report?.gate_passed === true ? '#f0fdf4' : report?.gate_passed === false ? '#fff1f2' : '#f8fafc';

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #f0f4f8 0%, #e8edf3 100%)', fontFamily: "'Sora', sans-serif" }}>
      <style>{`
        @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(1.4)} }
        @keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
        .card { animation: fadeIn 0.3s ease; }
        .tab-btn { border: none; cursor: pointer; transition: all 0.15s; }
        .tab-btn:hover { opacity: 0.85; }
      `}</style>

      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 50%, #2563eb 100%)', padding: '20px 32px', color: 'white' }}>
        <div style={{ maxWidth: 1200, margin: '0 auto' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <div style={{ fontSize: 11, opacity: 0.7, letterSpacing: 2, fontWeight: 600, marginBottom: 4 }}>
                VAIA CURRICULUM · MODULE 4 · L44
              </div>
              <h1 style={{ fontSize: 22, fontWeight: 700, margin: 0 }}>Agentic RAG Evaluation Dashboard</h1>
              <div style={{ fontSize: 12, opacity: 0.75, marginTop: 4 }}>
                Faithfulness · Answer Relevancy · Context Recall · Context Precision
              </div>
            </div>
            {report && (
              <div style={{ background: gateBg, border: `2px solid ${gateColor}`, borderRadius: 12, padding: '10px 20px', textAlign: 'center' }}>
                <div style={{ fontSize: 11, color: '#64748b', fontWeight: 600, marginBottom: 2 }}>GATE STATUS</div>
                <div style={{ fontSize: 18, fontWeight: 800, color: gateColor }}>
                  {report.gate_passed === true ? '✅ PASS' : report.gate_passed === false ? '🚫 FAIL' : '⏳ —'}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      <div style={{ maxWidth: 1200, margin: '0 auto', padding: '24px 20px' }}>

        {apiUnreachable && (
          <div style={{ background: '#fef2f2', border: '2px solid #dc2626', borderRadius: 12, padding: '16px 20px', marginBottom: 20, color: '#991b1b' }}>
            <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 6 }}>⚠️ Cannot connect to API (connection refused)</div>
            <div style={{ fontSize: 13, marginBottom: 8 }}>The backend at <code style={{ background: '#fee2e2', padding: '2px 6px', borderRadius: 4 }}>{API}</code> is not running. Start both backend and frontend from the project root:</div>
            <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 12, background: '#fff', padding: '10px 14px', borderRadius: 8, border: '1px solid #fecaca' }}>
              cd l44-rag-evaluation && ./start.sh
            </div>
            <div style={{ fontSize: 12, marginTop: 8, opacity: 0.9 }}>Then refresh this page. If you only ran the frontend, run <strong>./start.sh</strong> in the project directory to start the API server.</div>
          </div>
        )}

        {/* Control Panel */}
        <div className="card" style={{ background: 'white', borderRadius: 16, padding: 24, marginBottom: 20, boxShadow: '0 2px 12px rgba(0,0,0,0.06)', border: '1px solid #e2e8f0' }}>
          <div style={{ display: 'flex', gap: 16, alignItems: 'flex-end', flexWrap: 'wrap' }}>
            <div>
              <label style={{ fontSize: 11, fontWeight: 700, color: '#475569', letterSpacing: 1, display: 'block', marginBottom: 6 }}>RUN NAME</label>
              <input
                value={runName}
                onChange={e => setRunName(e.target.value)}
                style={{ border: '1.5px solid #e2e8f0', borderRadius: 8, padding: '8px 12px', fontSize: 13, width: 180, fontFamily: "'IBM Plex Mono', monospace", outline: 'none' }}
              />
            </div>
            <div>
              <label style={{ fontSize: 11, fontWeight: 700, color: '#475569', letterSpacing: 1, display: 'block', marginBottom: 6 }}>QUESTIONS</label>
              <input
                type="number" min={1} max={10} value={numQ}
                onChange={e => setNumQ(Number(e.target.value))}
                style={{ border: '1.5px solid #e2e8f0', borderRadius: 8, padding: '8px 12px', fontSize: 13, width: 80, outline: 'none' }}
              />
            </div>
            <button
              className="tab-btn"
              onClick={startEval}
              disabled={loading}
              style={{
                background: loading ? '#94a3b8' : 'linear-gradient(135deg, #1d4ed8, #2563eb)',
                color: 'white', padding: '10px 28px', borderRadius: 10, fontWeight: 700,
                fontSize: 13, letterSpacing: 0.5, boxShadow: '0 2px 8px rgba(37,99,235,0.3)',
              }}
            >
              {loading ? '⏳ Evaluating…' : '▶ Run Evaluation'}
            </button>
            {report && (
              <div style={{ marginLeft: 'auto' }}>
                <StateIndicator state={report.state} />
              </div>
            )}
          </div>
          {error && <div style={{ marginTop: 12, color: '#dc2626', fontSize: 12, background: '#fff1f2', padding: '8px 12px', borderRadius: 8 }}>⚠️ {error}</div>}
        </div>

        {/* Metric Cards */}
        {report?.aggregate && (
          <div className="card" style={{ marginBottom: 20 }}>
            {report.state === 'blocked' && Object.values(report.aggregate).every(v => v === 0) && (
              <div style={{ background: '#fef3c7', border: '1px solid #f59e0b', borderRadius: 12, padding: '12px 16px', marginBottom: 16, fontSize: 13, color: '#92400e' }}>
                ⚠️ All metrics are 0. If you expected scores: set a valid GEMINI_API_KEY in the backend and run evaluation again.
              </div>
            )}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16 }}>
              {Object.keys(METRIC_LABELS).map(metric => (
                <MetricCard key={metric} metric={metric} score={report.aggregate[metric]} />
              ))}
            </div>
          </div>
        )}

        {/* Tabs */}
        {report && (
          <>
            <div style={{ display: 'flex', gap: 4, marginBottom: 16, background: 'white', padding: 4, borderRadius: 12, boxShadow: '0 1px 4px rgba(0,0,0,0.06)', width: 'fit-content', border: '1px solid #e2e8f0' }}>
              {['overview', 'per-question', 'history'].map(tab => (
                <button
                  key={tab}
                  className="tab-btn"
                  onClick={() => setTab(tab)}
                  style={{
                    padding: '8px 18px', borderRadius: 9, fontWeight: 600, fontSize: 12,
                    background: activeTab === tab ? '#1d4ed8' : 'transparent',
                    color: activeTab === tab ? 'white' : '#64748b',
                    letterSpacing: 0.3,
                  }}
                >
                  {tab.replace('-', ' ').replace(/\b\w/g, c => c.toUpperCase())}
                </button>
              ))}
            </div>

            <div className="card">
              {/* Overview Tab */}
              {activeTab === 'overview' && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
                  {/* Radar Chart */}
                  <div style={{ background: 'white', borderRadius: 16, padding: 24, boxShadow: '0 2px 10px rgba(0,0,0,0.06)', border: '1px solid #e2e8f0' }}>
                    <h3 style={{ fontSize: 13, fontWeight: 700, color: '#1e293b', marginBottom: 16 }}>📊 Aggregate Metrics Radar</h3>
                    <ResponsiveContainer width="100%" height={260}>
                      <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="70%">
                        <PolarGrid stroke="#f1f5f9" strokeWidth={1.5}/>
                        <PolarAngleAxis dataKey="subject" tick={{ fontSize: 11, fill: '#64748b', fontWeight: 600 }}/>
                        <Radar name="Score" dataKey="score" stroke="#1d4ed8" fill="#3b82f6" fillOpacity={0.25} strokeWidth={2.5}/>
                        <Radar name="Threshold" dataKey="threshold" stroke="#f97316" fill="transparent" strokeWidth={1.5} strokeDasharray="5 3"/>
                      </RadarChart>
                    </ResponsiveContainer>
                    <div style={{ display: 'flex', gap: 20, justifyContent: 'center', fontSize: 11 }}>
                      <span><span style={{ color: '#1d4ed8', fontWeight: 700 }}>━</span> Score</span>
                      <span><span style={{ color: '#f97316', fontWeight: 700 }}>╌</span> Threshold</span>
                    </div>
                  </div>

                  {/* Run Summary */}
                  <div style={{ background: 'white', borderRadius: 16, padding: 24, boxShadow: '0 2px 10px rgba(0,0,0,0.06)', border: '1px solid #e2e8f0' }}>
                    <h3 style={{ fontSize: 13, fontWeight: 700, color: '#1e293b', marginBottom: 16 }}>📋 Run Summary</h3>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                      {[
                        ['Run Name', report.run_name, '#1d4ed8'],
                        ['State', report.state?.toUpperCase(), report.gate_passed === true ? '#16a34a' : report.gate_passed === false ? '#dc2626' : '#64748b'],
                        ['Questions', `${report.evaluated_questions} / ${report.total_questions}`, '#374151'],
                        ['Gate', report.gate_passed === true ? '✅ PASS' : report.gate_passed === false ? '🚫 FAIL' : '—', report.gate_passed === true ? '#16a34a' : '#dc2626'],
                      ].map(([label, val, color]) => (
                        <div key={label} style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid #f1f5f9' }}>
                          <span style={{ fontSize: 12, color: '#64748b' }}>{label}</span>
                          <span style={{ fontSize: 12, fontWeight: 700, color, fontFamily: "'IBM Plex Mono', monospace" }}>{val}</span>
                        </div>
                      ))}
                      {report.blocking_metrics && Object.keys(report.blocking_metrics).length > 0 && (
                        <div style={{ background: '#fff1f2', borderRadius: 8, padding: 10, marginTop: 6 }}>
                          <div style={{ fontSize: 11, fontWeight: 700, color: '#dc2626', marginBottom: 6 }}>🚨 Blocking Metrics</div>
                          {Object.entries(report.blocking_metrics).map(([m, v]) => (
                            <div key={m} style={{ fontSize: 11, color: '#b91c1c', fontFamily: "'IBM Plex Mono', monospace" }}>
                              {m}: {(v.score * 100).toFixed(1)}% &lt; {(v.threshold * 100).toFixed(0)}%
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Per-question Tab */}
              {activeTab === 'per-question' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
                  {/* Bar chart */}
                  <div style={{ background: 'white', borderRadius: 16, padding: 24, boxShadow: '0 2px 10px rgba(0,0,0,0.06)', border: '1px solid #e2e8f0' }}>
                    <h3 style={{ fontSize: 13, fontWeight: 700, color: '#1e293b', marginBottom: 16 }}>📈 Per-Question Score Distribution</h3>
                    <ResponsiveContainer width="100%" height={240}>
                      <BarChart data={barData} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
                        <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#64748b' }}/>
                        <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: '#64748b' }} unit="%"/>
                        <Tooltip formatter={(v) => [`${v}%`]} contentStyle={{ fontSize: 11, borderRadius: 8 }}/>
                        <ReferenceLine y={85} stroke="#3b82f6" strokeDasharray="4 2" strokeWidth={1.5}/>
                        <ReferenceLine y={80} stroke="#f97316" strokeDasharray="4 2" strokeWidth={1.5}/>
                        <Bar dataKey="faithfulness" fill="#3b82f6" radius={[3,3,0,0]} opacity={0.8}/>
                        <Bar dataKey="answer_relevancy" fill="#f97316" radius={[3,3,0,0]} opacity={0.8}/>
                        <Bar dataKey="context_recall" fill="#22c55e" radius={[3,3,0,0]} opacity={0.8}/>
                        <Bar dataKey="context_precision" fill="#7c3aed" radius={[3,3,0,0]} opacity={0.8}/>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  {/* Table */}
                  <div style={{ background: 'white', borderRadius: 16, padding: 20, boxShadow: '0 2px 10px rgba(0,0,0,0.06)', border: '1px solid #e2e8f0' }}>
                    <h3 style={{ fontSize: 13, fontWeight: 700, color: '#1e293b', marginBottom: 14 }}>🔍 Question-Level Scores (click to expand)</h3>
                    <QuestionTable items={report.items || []} />
                  </div>
                </div>
              )}

              {/* History Tab */}
              {activeTab === 'history' && (
                <div style={{ background: 'white', borderRadius: 16, padding: 24, boxShadow: '0 2px 10px rgba(0,0,0,0.06)', border: '1px solid #e2e8f0' }}>
                  <h3 style={{ fontSize: 13, fontWeight: 700, color: '#1e293b', marginBottom: 16 }}>📜 Evaluation History</h3>
                  {history.length === 0 ? (
                    <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40, fontSize: 13 }}>No history yet</div>
                  ) : (
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                      <thead>
                        <tr style={{ background: '#f8fafc' }}>
                          {['Run Name', 'Gate', 'Questions', 'Faith.', 'Relev.', 'Recall', 'Prec.'].map(h => (
                            <th key={h} style={{ padding: '8px 12px', textAlign: h === 'Run Name' ? 'left' : 'center', color: '#475569', fontWeight: 700, borderBottom: '2px solid #e2e8f0' }}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {history.map((r, i) => (
                          <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? 'white' : '#fafafa' }}>
                            <td style={{ padding: '8px 12px', fontFamily: "'IBM Plex Mono', monospace", fontSize: 11 }}>{r.run_name}</td>
                            <td style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 700, color: r.gate_passed === true ? '#16a34a' : r.gate_passed === false ? '#dc2626' : '#94a3b8' }}>
                              {r.gate_passed === true ? '✅' : r.gate_passed === false ? '🚫' : '—'}
                            </td>
                            <td style={{ padding: '8px 12px', textAlign: 'center', color: '#64748b' }}>{r.total_questions}</td>
                            {['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision'].map(m => (
                              <td key={m} style={{ padding: '8px 12px', textAlign: 'center', fontFamily: "'IBM Plex Mono', monospace", fontWeight: 600, color: r.aggregate?.[m] >= THRESHOLDS[m] ? '#16a34a' : '#dc2626' }}>
                                {r.aggregate?.[m] != null ? `${(r.aggregate[m] * 100).toFixed(1)}%` : '—'}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              )}
            </div>
          </>
        )}

        {/* Empty state */}
        {!report && !loading && (
          <div style={{ textAlign: 'center', padding: '60px 20px', color: '#94a3b8' }}>
            <div style={{ fontSize: 48, marginBottom: 16 }}>🔍</div>
            <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>No evaluations yet</div>
            <div style={{ fontSize: 13 }}>Configure a run above and click ▶ Run Evaluation</div>
          </div>
        )}

        {/* Footer */}
        <div style={{ textAlign: 'center', marginTop: 32, fontSize: 11, color: '#94a3b8' }}>
          VAIA Curriculum · L44 Agentic RAG Evaluation · Powered by Gemini 1.5 + Ragas
        </div>
      </div>
    </div>
  );
}
