import React, { useState, useEffect, useRef } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [stats, setStats] = useState(null);
  const [tools, setTools] = useState([]);
  const [executionLog, setExecutionLog] = useState([]);
  const wsRef = useRef(null);

  useEffect(() => {
    fetchTools();
    fetchStats();
    // Delay WebSocket connection slightly to allow backend to start
    const wsTimeout = setTimeout(connectWebSocket, 1000);
    
    return () => {
      clearTimeout(wsTimeout);
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounting');
        wsRef.current = null;
      }
    };
  }, []);

  const connectWebSocket = () => {
    // Don't reconnect if component is unmounting
    if (!wsRef.current || wsRef.current.readyState === WebSocket.CLOSED) {
      try {
        const ws = new WebSocket('ws://localhost:8000/ws');
        
        ws.onopen = () => {
          console.log('✅ WebSocket connected');
        };
        
        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            setStats(data);
          } catch (error) {
            console.error('Error parsing WebSocket message:', error, event.data);
          }
        };
        
        ws.onerror = (error) => {
          // Only log if not already closing/closed to avoid spam
          if (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN) {
            console.warn('WebSocket connection error (backend may be starting up)');
          }
        };
        
        ws.onclose = (event) => {
          // Don't reconnect if it was a clean close or if we're intentionally closing
          if (event.code !== 1000 && wsRef.current === ws) {
            console.log('WebSocket disconnected, will retry in 5 seconds...');
            setTimeout(() => {
              // Only reconnect if the ref still points to this connection
              if (wsRef.current === ws || !wsRef.current) {
                connectWebSocket();
              }
            }, 5000);
          }
        };
        
        wsRef.current = ws;
      } catch (error) {
        console.error('Failed to create WebSocket:', error);
        // Retry after delay
        setTimeout(connectWebSocket, 5000);
      }
    }
  };

  const fetchTools = async () => {
    try {
      const response = await fetch('/tools');
      if (!response.ok) {
        if (response.status === 500 || response.status === 0) {
          console.warn('Backend not available. Please start the backend server.');
          return;
        }
        const errorText = await response.text();
        console.error('Error fetching tools:', response.status, errorText);
        return;
      }
      const data = await response.json();
      setTools(data.tools || []);
    } catch (error) {
      if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
        console.warn('Backend not available. Please start the backend server.');
      } else {
        console.error('Error fetching tools:', error);
      }
    }
  };

  const fetchStats = async () => {
    try {
      const response = await fetch('/stats');
      if (!response.ok) {
        if (response.status === 500 || response.status === 0) {
          console.warn('Backend not available. Please start the backend server.');
          return;
        }
        const errorText = await response.text();
        console.error('Error fetching stats:', response.status, errorText);
        return;
      }
      const data = await response.json();
      setStats(data);
    } catch (error) {
      if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
        console.warn('Backend not available. Please start the backend server.');
      } else {
        console.error('Error fetching stats:', error);
      }
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setResult(null);
    setExecutionLog([]);

    try {
      const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, max_turns: 10 })
      });

      if (!response.ok) {
        // Read response body once as text, then try to parse as JSON
        let errorMessage = `Server error: ${response.status} ${response.statusText}`;
        try {
          const errorText = await response.text();
          // Try to parse as JSON
          try {
            const errorData = JSON.parse(errorText);
            errorMessage = errorData.detail || errorData.error || errorMessage;
          } catch {
            // Not JSON, use text as-is
            errorMessage = errorText || errorMessage;
          }
        } catch (err) {
          // Fallback if reading fails
          errorMessage = `Server error: ${response.status} ${response.statusText}`;
        }
        setResult({
          success: false,
          error: errorMessage
        });
        return;
      }

      const data = await response.json();
      setResult(data);
      if (data.execution_log) {
        setExecutionLog(data.execution_log);
      }
      
      // Refresh stats
      fetchStats();
    } catch (error) {
      setResult({
        success: false,
        error: error.message || 'An unexpected error occurred'
      });
    } finally {
      setLoading(false);
    }
  };

  const exampleQueries = [
    "What's the weather in Tokyo and London?",
    "Calculate revenue for Q3 2024 and Q4 2024",
    "Search for laptop products and find weather in Paris",
    "Search documents about VAIA architecture"
  ];

  const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff8042'];

  const getToolUsageData = () => {
    if (!stats || !stats.by_function) return [];
    return Object.entries(stats.by_function).map(([name, data]) => ({
      name,
      calls: data.calls,
      success_rate: (data.success_rate * 100).toFixed(1)
    }));
  };

  const getSuccessFailureData = () => {
    if (!stats) return [];
    return [
      { name: 'Success', value: stats.successful_calls },
      { name: 'Failed', value: stats.failed_calls }
    ];
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🔧 L18: Tool Execution Loop</h1>
        <p>Dynamic function calling with LLM orchestration</p>
      </header>

      <div className="main-content">
        <div className="left-panel">
          <div className="card">
            <h2>Query Interface</h2>
            <form onSubmit={handleSubmit} className="query-form">
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask something that requires tool execution..."
                rows={4}
                disabled={loading}
              />
              <button type="submit" disabled={loading}>
                {loading ? '⏳ Executing...' : '▶ Execute Loop'}
              </button>
            </form>

            <div className="examples">
              <h3>Example Queries</h3>
              {exampleQueries.map((example, i) => (
                <button
                  key={i}
                  className="example-btn"
                  onClick={() => setQuery(example)}
                  disabled={loading}
                >
                  {example}
                </button>
              ))}
            </div>
          </div>

          {result && (
            <div className={`card result-card ${result.success ? 'success' : 'error'}`}>
              <h2>{result.success ? '✅ Response' : '❌ Error'}</h2>
              {result.success ? (
                <>
                  <div className="answer">{result.answer}</div>
                  <div className="metrics">
                    <span>🔄 Turns: {result.turn_count}</span>
                    <span>⚡ Time: {result.execution_time_ms}ms</span>
                    <span>📊 State: {result.state}</span>
                  </div>
                </>
              ) : (
                <div className="error-message">{result.error}</div>
              )}
            </div>
          )}

          {executionLog.length > 0 && (
            <div className="card">
              <h2>📋 Execution Log</h2>
              <div className="execution-log">
                {executionLog.map((log, i) => (
                  <div key={i} className="log-entry">
                    <div className="log-header">
                      <span className="turn">Turn {log.turn}</span>
                      <span className="function">{log.function}</span>
                      <span className="timestamp">{new Date(log.timestamp).toLocaleTimeString()}</span>
                    </div>
                    <div className="log-details">
                      <div className="log-section">
                        <strong>Inputs:</strong>
                        <pre>{JSON.stringify(log.inputs, null, 2)}</pre>
                      </div>
                      <div className="log-section">
                        <strong>Result:</strong>
                        <pre>{JSON.stringify(log.result, null, 2)}</pre>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="right-panel">
          <div className="card">
            <h2>🛠️ Registered Tools</h2>
            <div className="tools-list">
              {tools.map((tool, i) => (
                <div key={i} className="tool-item">
                  <div className="tool-name">{tool.name}</div>
                  <div className="tool-description">{tool.description}</div>
                  <div className="tool-params">
                    {tool.parameters.required.map((param, j) => (
                      <span key={j} className="param">{param}</span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {stats && (
            <>
              <div className="card">
                <h2>📊 Execution Statistics</h2>
                <div className="stats-grid">
                  <div className="stat">
                    <div className="stat-value">{stats.total_calls}</div>
                    <div className="stat-label">Total Calls</div>
                  </div>
                  <div className="stat">
                    <div className="stat-value">{stats.successful_calls}</div>
                    <div className="stat-label">Successful</div>
                  </div>
                  <div className="stat">
                    <div className="stat-value">{stats.failed_calls}</div>
                    <div className="stat-label">Failed</div>
                  </div>
                  <div className="stat">
                    <div className="stat-value">
                      {((stats.successful_calls / stats.total_calls) * 100 || 0).toFixed(1)}%
                    </div>
                    <div className="stat-label">Success Rate</div>
                  </div>
                </div>
              </div>

              <div className="card">
                <h2>📈 Tool Usage</h2>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={getToolUsageData()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" angle={-15} textAnchor="end" height={80} />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="calls" fill="#8884d8" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="card">
                <h2>🎯 Success Distribution</h2>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={getSuccessFailureData()}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {getSuccessFailureData().map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              <div className="card">
                <h2>⚡ Performance Metrics</h2>
                <div className="perf-metrics">
                  {stats.by_function && Object.entries(stats.by_function).map(([name, data]) => (
                    <div key={name} className="perf-item">
                      <div className="perf-name">{name}</div>
                      <div className="perf-stats">
                        <span>Calls: {data.calls}</span>
                        <span>Success: {(data.success_rate * 100).toFixed(1)}%</span>
                        <span>Avg: {data.avg_latency_ms}ms</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
