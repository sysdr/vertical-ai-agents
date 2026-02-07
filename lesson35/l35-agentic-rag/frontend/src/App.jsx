import React, { useState, useEffect, useCallback } from 'react';

// Dashboard metrics - persist across queries
const DEMO_QUERY = "What is agentic RAG and how does it differ from traditional RAG?";

function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [wsState, setWsState] = useState(null);
  const [useWebSocket, setUseWebSocket] = useState(true);
  const [maxTokens, setMaxTokens] = useState(3000);
  const [metrics, setMetrics] = useState({
    queriesProcessed: 0,
    totalTokensUsed: 0,
    documentsRetrieved: 0,
    successCount: 0,
    lastStage: null
  });

  useEffect(() => {
    if (!useWebSocket) return;

    const ws = new WebSocket('ws://localhost:8000/ws');
    
    ws.onopen = () => console.log('WebSocket connected');
    
    ws.onmessage = (event) => {
      const state = JSON.parse(event.data);
      setWsState(state);
      setMetrics(m => ({
        ...m,
        lastStage: state.stage,
        ...(state.stage === 'COMPLETED' || state.stage === 'FAILED' ? {
          queriesProcessed: m.queriesProcessed + 1,
          totalTokensUsed: m.totalTokensUsed + (state.total_tokens || 0),
          documentsRetrieved: m.documentsRetrieved + (state.documents?.length || 0),
          successCount: m.successCount + (state.stage === 'COMPLETED' ? 1 : 0)
        } : {})
      }));
      if (state.stage === 'COMPLETED' || state.stage === 'FAILED') {
        setLoading(false);
        setResult(state);
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setUseWebSocket(false);
    };
    
    window.ws = ws;
    
    return () => ws.close();
  }, [useWebSocket]);

  const handleSubmit = async (e, queryOverride) => {
    e.preventDefault();
    const q = queryOverride ?? query;
    if (!q?.trim()) return;
    setQuery(q);
    setLoading(true);
    setResult(null);
    setWsState(null);

    if (useWebSocket && window.ws && window.ws.readyState === WebSocket.OPEN) {
      window.ws.send(JSON.stringify({ query: q, max_tokens: maxTokens }));
    } else {
      try {
        const response = await fetch('/api/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: q, max_tokens: maxTokens })
        });
        const data = await response.json();
        setResult(data);
        setLoading(false);
        setMetrics(m => ({
          ...m,
          queriesProcessed: m.queriesProcessed + 1,
          totalTokensUsed: m.totalTokensUsed + (data.total_tokens || 0),
          documentsRetrieved: m.documentsRetrieved + (data.documents?.length || 0),
          successCount: m.successCount + (data.stage === 'COMPLETED' ? 1 : 0),
          lastStage: data.stage
        }));
      } catch (error) {
        console.error('Error:', error);
        setResult({ stage: 'FAILED', errors: [error.message] });
        setLoading(false);
        setMetrics(m => ({ ...m, queriesProcessed: m.queriesProcessed + 1, lastStage: 'FAILED' }));
      }
    }
  };

  const runDemo = useCallback((e) => {
    e?.preventDefault();
    const fakeEvent = { preventDefault: () => {} };
    handleSubmit(fakeEvent, DEMO_QUERY);
  }, [maxTokens, useWebSocket]);

  const getStageColor = (stage) => {
    const colors = {
      'IDLE': '#94a3b8',
      'PLANNING': '#fbbf24',
      'RETRIEVING': '#60a5fa',
      'VALIDATING': '#a78bfa',
      'SYNTHESIZING': '#34d399',
      'COMPLETED': '#10b981',
      'FAILED': '#ef4444'
    };
    return colors[stage] || '#94a3b8';
  };

  const currentState = wsState || result;

  return (
    <div style={{ 
      fontFamily: 'system-ui, -apple-system, sans-serif',
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '2rem',
      backgroundColor: '#f8fafc'
    }}>
      <div style={{
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        padding: '2rem',
        borderRadius: '16px',
        marginBottom: '2rem',
        color: 'white'
      }}>
        <h1 style={{ margin: '0 0 0.5rem 0', fontSize: '2.5rem' }}>
          🤖 Agentic RAG System
        </h1>
        <p style={{ margin: 0, opacity: 0.9, fontSize: '1.1rem' }}>
          L35: Multi-Agent Orchestration with Planner, Retriever, Validator & Synthesizer
        </p>
      </div>

      {/* Dashboard Metrics */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
        gap: '1rem',
        marginBottom: '2rem'
      }}>
        <div style={{
          backgroundColor: 'white',
          padding: '1.25rem',
          borderRadius: '12px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '0.8rem', color: '#64748b', fontWeight: '600', marginBottom: '0.25rem' }}>Queries Run</div>
          <div style={{ fontSize: '1.75rem', fontWeight: 'bold', color: '#334155' }}>{metrics.queriesProcessed}</div>
        </div>
        <div style={{
          backgroundColor: 'white',
          padding: '1.25rem',
          borderRadius: '12px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '0.8rem', color: '#64748b', fontWeight: '600', marginBottom: '0.25rem' }}>Total Tokens</div>
          <div style={{ fontSize: '1.75rem', fontWeight: 'bold', color: '#334155' }}>{metrics.totalTokensUsed}</div>
        </div>
        <div style={{
          backgroundColor: 'white',
          padding: '1.25rem',
          borderRadius: '12px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '0.8rem', color: '#64748b', fontWeight: '600', marginBottom: '0.25rem' }}>Docs Retrieved</div>
          <div style={{ fontSize: '1.75rem', fontWeight: 'bold', color: '#334155' }}>{metrics.documentsRetrieved}</div>
        </div>
        <div style={{
          backgroundColor: 'white',
          padding: '1.25rem',
          borderRadius: '12px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '0.8rem', color: '#64748b', fontWeight: '600', marginBottom: '0.25rem' }}>Success Rate</div>
          <div style={{ fontSize: '1.75rem', fontWeight: 'bold', color: '#334155' }}>
            {metrics.queriesProcessed > 0
              ? `${Math.round((metrics.successCount / metrics.queriesProcessed) * 100)}%`
              : '—'}
          </div>
        </div>
        <div style={{
          backgroundColor: 'white',
          padding: '1.25rem',
          borderRadius: '12px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          textAlign: 'center',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center'
        }}>
          <button
            onClick={runDemo}
            disabled={loading}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: loading ? '#94a3b8' : '#10b981',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              fontWeight: '600',
              cursor: loading ? 'not-allowed' : 'pointer',
              fontSize: '0.9rem'
            }}
          >
            {loading ? 'Running...' : '▶ Run Demo'}
          </button>
        </div>
      </div>

      <form onSubmit={handleSubmit} style={{ marginBottom: '2rem' }}>
        <div style={{ 
          backgroundColor: 'white',
          padding: '1.5rem',
          borderRadius: '12px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
        }}>
          <label style={{ 
            display: 'block',
            marginBottom: '0.5rem',
            fontWeight: '600',
            color: '#334155'
          }}>
            Enter Your Query
          </label>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g., What is agentic RAG and how does it differ from traditional RAG?"
            style={{
              width: '100%',
              minHeight: '100px',
              padding: '0.75rem',
              borderRadius: '8px',
              border: '2px solid #e2e8f0',
              fontSize: '1rem',
              fontFamily: 'inherit',
              marginBottom: '1rem',
              resize: 'vertical'
            }}
          />
          
          <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem' }}>
            <div style={{ flex: 1 }}>
              <label style={{ 
                display: 'block',
                marginBottom: '0.5rem',
                fontSize: '0.9rem',
                color: '#64748b'
              }}>
                Max Tokens: {maxTokens}
              </label>
              <input
                type="range"
                min="500"
                max="5000"
                step="100"
                value={maxTokens}
                onChange={(e) => setMaxTokens(Number(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>
            
            <label style={{ 
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              cursor: 'pointer'
            }}>
              <input
                type="checkbox"
                checked={useWebSocket}
                onChange={(e) => setUseWebSocket(e.target.checked)}
              />
              <span style={{ fontSize: '0.9rem', color: '#64748b' }}>
                Real-time updates (WebSocket)
              </span>
            </label>
          </div>

          <button
            type="submit"
            disabled={loading || !query.trim()}
            style={{
              width: '100%',
              padding: '1rem',
              backgroundColor: loading ? '#94a3b8' : '#667eea',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              fontSize: '1.1rem',
              fontWeight: '600',
              cursor: loading ? 'not-allowed' : 'pointer',
              transition: 'background 0.2s'
            }}
          >
            {loading ? '🔄 Processing...' : '🚀 Process Query'}
          </button>
        </div>
      </form>

      {currentState && (
        <div style={{
          backgroundColor: 'white',
          padding: '1.5rem',
          borderRadius: '12px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          marginBottom: '2rem'
        }}>
          <h3 style={{ 
            margin: '0 0 1rem 0',
            color: '#334155',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            Pipeline Status:
            <span style={{
              padding: '0.25rem 0.75rem',
              borderRadius: '999px',
              backgroundColor: getStageColor(currentState.stage) + '20',
              color: getStageColor(currentState.stage),
              fontSize: '0.9rem',
              fontWeight: 'bold'
            }}>
              {currentState.stage}
            </span>
          </h3>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
            {['PLANNING', 'RETRIEVING', 'VALIDATING', 'SYNTHESIZING'].map((stage, idx) => {
              const isActive = currentState.stage === stage;
              const isPast = ['PLANNING', 'RETRIEVING', 'VALIDATING', 'SYNTHESIZING'].indexOf(currentState.stage) > idx;
              const isComplete = currentState.stage === 'COMPLETED';
              
              return (
                <div key={stage} style={{
                  padding: '1rem',
                  borderRadius: '8px',
                  backgroundColor: isActive || isPast || isComplete ? getStageColor(stage) + '20' : '#f1f5f9',
                  border: isActive ? `2px solid ${getStageColor(stage)}` : '2px solid transparent',
                  textAlign: 'center',
                  transition: 'all 0.3s'
                }}>
                  <div style={{ 
                    fontSize: '1.5rem',
                    marginBottom: '0.5rem',
                    filter: isActive || isPast || isComplete ? 'none' : 'grayscale(100%)'
                  }}>
                    {['🎯', '🔍', '✅', '✨'][idx]}
                  </div>
                  <div style={{ 
                    fontSize: '0.85rem',
                    fontWeight: '600',
                    color: isActive || isPast || isComplete ? '#334155' : '#94a3b8'
                  }}>
                    {stage}
                  </div>
                </div>
              );
            })}
          </div>

          {currentState.total_tokens > 0 && (
            <div style={{ 
              padding: '1rem',
              backgroundColor: '#fef3c7',
              borderRadius: '8px',
              marginBottom: '1rem'
            }}>
              <strong>Token Usage:</strong> {currentState.total_tokens} / {maxTokens}
              <div style={{
                height: '8px',
                backgroundColor: '#fde68a',
                borderRadius: '4px',
                marginTop: '0.5rem',
                overflow: 'hidden'
              }}>
                <div style={{
                  height: '100%',
                  backgroundColor: currentState.total_tokens > maxTokens ? '#ef4444' : '#10b981',
                  width: `${Math.min((currentState.total_tokens / maxTokens) * 100, 100)}%`,
                  transition: 'width 0.3s'
                }} />
              </div>
            </div>
          )}

          {currentState.plan && (
            <div style={{ 
              padding: '1rem',
              backgroundColor: '#f0f9ff',
              borderRadius: '8px',
              marginBottom: '1rem'
            }}>
              <strong style={{ color: '#0284c7' }}>🎯 Plan:</strong>
              <pre style={{ 
                margin: '0.5rem 0 0 0',
                fontSize: '0.85rem',
                whiteSpace: 'pre-wrap',
                color: '#334155'
              }}>
                {JSON.stringify(currentState.plan, null, 2)}
              </pre>
            </div>
          )}

          {currentState.documents && currentState.documents.length > 0 && (
            <div style={{ 
              padding: '1rem',
              backgroundColor: '#f0fdf4',
              borderRadius: '8px',
              marginBottom: '1rem'
            }}>
              <strong style={{ color: '#16a34a' }}>🔍 Retrieved Documents:</strong>
              {currentState.documents.map((doc, idx) => (
                <div key={idx} style={{
                  marginTop: '0.5rem',
                  padding: '0.75rem',
                  backgroundColor: 'white',
                  borderRadius: '6px',
                  border: '1px solid #dcfce7'
                }}>
                  <div style={{ fontSize: '0.85rem', color: '#16a34a', fontWeight: '600' }}>
                    {doc.source} (Relevance: {doc.relevance?.toFixed(2)})
                  </div>
                  <div style={{ fontSize: '0.85rem', color: '#334155', marginTop: '0.25rem' }}>
                    {doc.content}
                  </div>
                </div>
              ))}
            </div>
          )}

          {currentState.validation_scores && (
            <div style={{ 
              padding: '1rem',
              backgroundColor: '#faf5ff',
              borderRadius: '8px',
              marginBottom: '1rem'
            }}>
              <strong style={{ color: '#9333ea' }}>✅ Validation:</strong>
              <div style={{ marginTop: '0.5rem', fontSize: '0.9rem', color: '#334155' }}>
                Confidence: {(currentState.validation_scores.overall_confidence * 100).toFixed(1)}%
              </div>
              <div style={{
                height: '8px',
                backgroundColor: '#e9d5ff',
                borderRadius: '4px',
                marginTop: '0.5rem',
                overflow: 'hidden'
              }}>
                <div style={{
                  height: '100%',
                  backgroundColor: '#9333ea',
                  width: `${currentState.validation_scores.overall_confidence * 100}%`,
                  transition: 'width 0.3s'
                }} />
              </div>
            </div>
          )}

          {currentState.response && (
            <div style={{ 
              padding: '1.5rem',
              backgroundColor: '#ecfdf5',
              borderRadius: '8px',
              border: '2px solid #10b981'
            }}>
              <strong style={{ color: '#059669', fontSize: '1.1rem' }}>✨ Response:</strong>
              <div style={{ 
                marginTop: '1rem',
                lineHeight: '1.6',
                color: '#334155',
                whiteSpace: 'pre-wrap'
              }}>
                {currentState.response}
              </div>
            </div>
          )}

          {currentState.errors && currentState.errors.length > 0 && (
            <div style={{ 
              padding: '1rem',
              backgroundColor: '#fef2f2',
              borderRadius: '8px',
              border: '2px solid #ef4444'
            }}>
              <strong style={{ color: '#dc2626' }}>⚠️ Errors:</strong>
              {currentState.errors.map((err, idx) => (
                <div key={idx} style={{ marginTop: '0.5rem', color: '#991b1b', fontSize: '0.9rem' }}>
                  • {err}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
