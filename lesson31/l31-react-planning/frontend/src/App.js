import React, { useState, useEffect } from 'react';
import PlanningForm from './components/PlanningForm';
import TraceViewer from './components/TraceViewer';
import MetricsDashboard from './components/MetricsDashboard';
import { executePlanning } from './services/api';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';
const WS_URL = (process.env.REACT_APP_WS_URL || 'ws://localhost:8000') + '/ws/metrics';
const STATS_URL = API_BASE + '/stats';

function App() {
  const [trace, setTrace] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState(null);

  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    ws.onmessage = (e) => {
      try {
        setStats(JSON.parse(e.data));
      } catch (_) {}
    };
    const poll = setInterval(async () => {
      try {
        const r = await fetch(STATS_URL);
        if (r.ok) setStats(await r.json());
      } catch (_) {}
    }, 3000);
    return () => { ws.close(); clearInterval(poll); };
  }, []);

  const handlePlan = async (task, maxIterations) => {
    setLoading(true);
    setError(null);
    try {
      const result = await executePlanning(task, maxIterations);
      setTrace(result.trace);
      try {
        const r = await fetch(STATS_URL);
        if (r.ok) setStats(await r.json());
      } catch (_) {}
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🧠 ReAct Planning System</h1>
        <p>L31: Deep Dive into ReAct Planning</p>
      </header>
      
      <main className="app-main">
        <MetricsDashboard stats={stats} />
        <PlanningForm onSubmit={handlePlan} loading={loading} />
        {error && <div className="error-message">{error}</div>}
        {trace && <TraceViewer trace={trace} />}
      </main>
    </div>
  );
}

export default App;
