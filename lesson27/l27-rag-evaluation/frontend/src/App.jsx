import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { PlayCircle, TrendingUp, CheckCircle, AlertCircle, Activity } from 'lucide-react';

function App() {
  const [runs, setRuns] = useState([]);
  const [currentRun, setCurrentRun] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [trends, setTrends] = useState([]);
  const [ws, setWs] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const wsUrl = (window.location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + window.location.host + '/ws';
    const websocket = new WebSocket(wsUrl);
    
    websocket.onopen = () => console.log('WebSocket connected');
    websocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'evaluation_complete' || data.type === 'error') {
          refreshAll().then(() => setLoading(false));
        }
      } catch (_) {}
    };
    websocket.onerror = () => {};
    
    setWs(websocket);
    fetchRuns();
    fetchMetrics();
    fetchTrends();
    
    return () => websocket.close();
  }, []);

  // Periodic refresh while evaluation is running so metrics update in real time
  useEffect(() => {
    if (!loading) return;
    const interval = setInterval(() => refreshAll(), 2000);
    return () => clearInterval(interval);
  }, [loading]);

  const fetchRuns = async () => {
    try {
      const res = await fetch('/api/evaluation/runs', { cache: 'no-store' });
      const data = await res.json();
      setRuns(data);
    } catch (error) {
      console.error('Error fetching runs:', error);
    }
  };

  const fetchMetrics = async () => {
    try {
      const res = await fetch('/api/metrics/summary', { cache: 'no-store' });
      const data = await res.json();
      setMetrics(data);
    } catch (error) {
      console.error('Error fetching metrics:', error);
    }
  };

  const fetchTrends = async () => {
    try {
      const res = await fetch('/api/metrics/trends', { cache: 'no-store' });
      const data = await res.json();
      setTrends(data.trends || []);
    } catch (error) {
      console.error('Error fetching trends:', error);
    }
  };

  const refreshAll = async () => {
    await Promise.all([fetchRuns(), fetchMetrics(), fetchTrends()]);
  };

  const startEvaluation = async () => {
    setLoading(true);
    
    try {
      const queriesRes = await fetch('/api/benchmarks/sample-queries');
      const { queries } = await queriesRes.json();
      
      const res = await fetch('/api/evaluation/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          run_name: `Eval-${Date.now()}`,
          queries: queries.slice(0, 2),
          model_config: {}
        })
      });
      
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || res.statusText);
      }
      
      const data = await res.json();
      setCurrentRun(data);
      const runId = data.run_id;
      
      // Refresh runs immediately so new run appears in list
      fetchRuns();
      
      // Poll run status for real-time updates
      const timeoutMs = 120000;
      const pollIntervalMs = 1500;
      const started = Date.now();
      const poll = async () => {
        if (Date.now() - started > timeoutMs) {
          setLoading(false);
          return;
        }
        try {
          const r = await fetch(`/api/evaluation/runs/${runId}`, { cache: 'no-store' });
          if (!r.ok) return;
          const run = await r.json();
          fetchRuns();
          if (run.status === 'completed' || run.status === 'error') {
            await refreshAll();
            setLoading(false);
            return;
          }
        } catch (_) {}
        setTimeout(poll, pollIntervalMs);
      };
      setTimeout(poll, pollIntervalMs);
    } catch (error) {
      console.error('Error starting evaluation:', error);
      setLoading(false);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="text-green-500" size={20} />;
      case 'running':
        return <Activity className="text-blue-500 animate-pulse" size={20} />;
      case 'error':
        return <AlertCircle className="text-red-500" size={20} />;
      default:
        return <Activity className="text-gray-400" size={20} />;
    }
  };

  return (
    <div style={{ minHeight: '100vh', padding: '40px 20px' }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        {/* Header */}
        <div style={{
          background: 'white',
          borderRadius: '16px',
          padding: '32px',
          marginBottom: '24px',
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h1 style={{ fontSize: '32px', fontWeight: 'bold', marginBottom: '8px', color: '#1a202c' }}>
                RAG Evaluation Dashboard
              </h1>
              <p style={{ color: '#718096', fontSize: '16px' }}>
                L27: Monitor RAG performance metrics and benchmarks
              </p>
            </div>
            <button
              onClick={startEvaluation}
              disabled={loading}
              style={{
                background: loading ? '#cbd5e0' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                color: 'white',
                padding: '12px 24px',
                borderRadius: '8px',
                border: 'none',
                cursor: loading ? 'not-allowed' : 'pointer',
                fontSize: '16px',
                fontWeight: '600',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
              }}
            >
              <PlayCircle size={20} />
              {loading ? 'Running...' : 'Run Evaluation'}
            </button>
          </div>
        </div>

        {/* Metrics Summary */}
        {metrics && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
            gap: '20px',
            marginBottom: '24px'
          }}>
            {[
              { label: 'Faithfulness', value: metrics.avg_faithfulness, color: '#10b981' },
              { label: 'Relevance', value: metrics.avg_relevance, color: '#3b82f6' },
              { label: 'Precision', value: metrics.avg_precision, color: '#f59e0b' },
              { label: 'Recall', value: metrics.avg_recall, color: '#8b5cf6' }
            ].map((metric, idx) => (
              <div key={idx} style={{
                background: 'white',
                borderRadius: '12px',
                padding: '24px',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <p style={{ color: '#718096', fontSize: '14px', marginBottom: '8px' }}>
                      {metric.label}
                    </p>
                    <p style={{ fontSize: '32px', fontWeight: 'bold', color: metric.color }}>
                      {metric.value.toFixed(2)}
                    </p>
                  </div>
                  <TrendingUp size={32} style={{ color: metric.color, opacity: 0.3 }} />
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Trends Chart */}
        <div style={{
          background: 'white',
          borderRadius: '16px',
          padding: '32px',
          marginBottom: '24px',
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
        }}>
          <h2 style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '24px', color: '#1a202c' }}>
            Metric Trends (24h)
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={trends}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="timestamp" stroke="#718096" />
              <YAxis domain={[0, 1]} stroke="#718096" />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="faithfulness" stroke="#10b981" strokeWidth={2} />
              <Line type="monotone" dataKey="relevance" stroke="#3b82f6" strokeWidth={2} />
              <Line type="monotone" dataKey="precision" stroke="#f59e0b" strokeWidth={2} />
              <Line type="monotone" dataKey="recall" stroke="#8b5cf6" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Evaluation Runs */}
        <div style={{
          background: 'white',
          borderRadius: '16px',
          padding: '32px',
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
        }}>
          <h2 style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '24px', color: '#1a202c' }}>
            Recent Evaluation Runs
          </h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {runs.length === 0 ? (
              <p style={{ color: '#718096', textAlign: 'center', padding: '40px' }}>
                No evaluation runs yet. Click "Run Evaluation" to start.
              </p>
            ) : (
              runs.slice(0, 10).map((run) => (
                <div key={run.run_id} style={{
                  border: '1px solid #e2e8f0',
                  borderRadius: '8px',
                  padding: '16px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    {getStatusIcon(run.status)}
                    <div>
                      <p style={{ fontWeight: '600', color: '#1a202c' }}>
                        {run.run_name}
                        {run.demo_fallback && (
                          <span style={{ fontSize: '11px', color: '#f59e0b', marginLeft: '6px' }}>(demo)</span>
                        )}
                      </p>
                      <p style={{ fontSize: '14px', color: '#718096' }}>
                        {run.completed_queries ?? 0}/{run.total_queries ?? 0} queries
                      </p>
                    </div>
                  </div>
                  {run.results && (
                    <div style={{ display: 'flex', gap: '16px' }}>
                      <div style={{ textAlign: 'center' }}>
                        <p style={{ fontSize: '12px', color: '#718096' }}>Faith</p>
                        <p style={{ fontWeight: '600', color: '#10b981' }}>
                          {run.results.faithfulness.toFixed(2)}
                        </p>
                      </div>
                      <div style={{ textAlign: 'center' }}>
                        <p style={{ fontSize: '12px', color: '#718096' }}>Rel</p>
                        <p style={{ fontWeight: '600', color: '#3b82f6' }}>
                          {run.results.answer_relevance.toFixed(2)}
                        </p>
                      </div>
                      <div style={{ textAlign: 'center' }}>
                        <p style={{ fontSize: '12px', color: '#718096' }}>Prec</p>
                        <p style={{ fontWeight: '600', color: '#f59e0b' }}>
                          {run.results.context_precision.toFixed(2)}
                        </p>
                      </div>
                      <div style={{ textAlign: 'center' }}>
                        <p style={{ fontSize: '12px', color: '#718096' }}>Rec</p>
                        <p style={{ fontWeight: '600', color: '#8b5cf6' }}>
                          {run.results.context_recall.toFixed(2)}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
