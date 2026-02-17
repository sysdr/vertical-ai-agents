import React, { useState } from 'react';
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

const DEMO_QUERIES = [
  'How does the Synthesizer Agent implement explainability in production VAIAs?',
  'What is Chain-of-Thought reasoning and how is it used in synthesis?',
  'How does citation granularity work in enterprise RAG systems?',
  'Summarize the key points about XAI and transparency in VAIAs.',
  'What are the main steps in the synthesis pipeline from context to response?',
];

function App() {
  const [query, setQuery] = useState('');
  const [chunks, setChunks] = useState([
    { id: 'chunk_001', content: 'Sample context about VAIAs...', source: 'doc1.pdf', quality_score: 0.9 }
  ]);
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showReasoning, setShowReasoning] = useState(false);
  const [metrics, setMetrics] = useState({
    synthesisCount: 0,
    lastConfidence: 0,
    lastSynthesisTimeMs: 0,
    lastCitationsCount: 0,
    lastReasoningStepsCount: 0
  });

  const updateMetricsFromResponse = (data) => {
    if (!data) return;
    setMetrics(prev => ({
      synthesisCount: prev.synthesisCount + 1,
      lastConfidence: data.overall_confidence ?? 0,
      lastSynthesisTimeMs: data.synthesis_time_ms ?? 0,
      lastCitationsCount: Array.isArray(data.citations) ? data.citations.length : 0,
      lastReasoningStepsCount: Array.isArray(data.reasoning_chain) ? data.reasoning_chain.length : 0
    }));
  };

  const runDemo = async () => {
    setLoading(true);
    try {
      const result = await axios.post(`${API_BASE}/synthesize/demo`);
      setResponse(result.data);
      updateMetricsFromResponse(result.data);
    } catch (error) {
      const msg = error.response?.data?.detail || error.message;
      alert(msg);
    }
    setLoading(false);
  };

  const synthesize = async (queryOverride) => {
    const q = (queryOverride !== undefined ? queryOverride : query).trim();
    if (!q) {
      alert('Please enter a query');
      return;
    }
    
    setLoading(true);
    try {
      const result = await axios.post(`${API_BASE}/synthesize`, {
        query: q,
        validated_chunks: chunks,
        quality_scores: { overall: 0.85 },
        metadata: {},
        user_context: { requires_full_xai: showReasoning }
      });
      setResponse(result.data);
      updateMetricsFromResponse(result.data);
      if (queryOverride !== undefined) setQuery(queryOverride);
    } catch (error) {
      const msg = error.response?.data?.detail || error.message;
      alert(msg);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                Synthesizer Agent & XAI
              </h1>
              <p className="text-gray-600 mt-1">Lesson 41: Advanced Agentic RAG</p>
            </div>
            <div className="flex gap-3">
              <button
                onClick={runDemo}
                disabled={loading}
                className="px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition"
              >
                {loading ? 'Processing...' : 'Run Demo'}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Dashboard metrics - updated by Run Demo / Synthesize */}
      <div className="max-w-7xl mx-auto px-4 py-4">
        <div className="bg-white rounded-xl shadow-sm p-4 border border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Dashboard Metrics</h2>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
              <p className="text-xs font-medium text-gray-500 uppercase">Total Syntheses</p>
              <p className="text-2xl font-bold text-indigo-600">{metrics.synthesisCount}</p>
            </div>
            <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
              <p className="text-xs font-medium text-gray-500 uppercase">Last Confidence</p>
              <p className="text-2xl font-bold text-green-600">{(metrics.lastConfidence * 100).toFixed(1)}%</p>
            </div>
            <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
              <p className="text-xs font-medium text-gray-500 uppercase">Last Time (ms)</p>
              <p className="text-2xl font-bold text-blue-600">{metrics.lastSynthesisTimeMs.toFixed(0)}</p>
            </div>
            <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
              <p className="text-xs font-medium text-gray-500 uppercase">Last Citations</p>
              <p className="text-2xl font-bold text-amber-600">{metrics.lastCitationsCount}</p>
            </div>
            <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
              <p className="text-xs font-medium text-gray-500 uppercase">Reasoning Steps</p>
              <p className="text-2xl font-bold text-purple-600">{metrics.lastReasoningStepsCount}</p>
            </div>
          </div>
          <p className="text-xs text-gray-500 mt-2">Run Demo or Synthesize to update metrics.</p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Input Panel */}
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-sm p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">Query Input</h2>
              <p className="text-sm text-gray-500 mb-2">Try a demo query (click to fill, or Run to fill and synthesize):</p>
              <div className="flex flex-wrap gap-2 mb-3">
                {DEMO_QUERIES.map((q, idx) => (
                  <span key={idx} className="inline-flex gap-1">
                    <button
                      type="button"
                      onClick={() => setQuery(q)}
                      className="px-3 py-1.5 text-sm bg-indigo-50 text-indigo-700 rounded-l-lg border border-indigo-200 hover:bg-indigo-100 transition"
                      title={q}
                    >
                      Query {idx + 1}
                    </button>
                    <button
                      type="button"
                      onClick={() => synthesize(q)}
                      disabled={loading}
                      className="px-2 py-1.5 text-sm bg-green-600 text-white rounded-r-lg hover:bg-green-700 disabled:opacity-50 transition"
                      title="Run this query"
                    >
                      Run
                    </button>
                  </span>
                ))}
              </div>
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Or enter your own query..."
                className="w-full h-32 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              />
              
              <div className="mt-4 flex items-center justify-between">
                <label className="flex items-center gap-2 text-sm text-gray-700">
                  <input
                    type="checkbox"
                    checked={showReasoning}
                    onChange={(e) => setShowReasoning(e.target.checked)}
                    className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                  />
                  Show Full Reasoning Chain (XAI)
                </label>
                <button
                  onClick={synthesize}
                  disabled={loading}
                  className="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 transition"
                >
                  Synthesize
                </button>
              </div>
            </div>

            {/* Context Chunks */}
            <div className="bg-white rounded-xl shadow-sm p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                Validated Context ({chunks.length} chunks)
              </h2>
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {chunks.map((chunk, idx) => (
                  <div key={idx} className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-indigo-600">{chunk.source}</span>
                      <span className="text-xs text-gray-500">Score: {chunk.quality_score}</span>
                    </div>
                    <p className="text-sm text-gray-700">{chunk.content.slice(0, 150)}...</p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Response Panel */}
          <div className="space-y-6">
            {response && (
              <>
                {/* Synthesized Response */}
                <div className="bg-white rounded-xl shadow-sm p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-xl font-semibold text-gray-900">Synthesized Response</h2>
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-gray-600">Confidence:</span>
                      <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                        response.overall_confidence > 0.8 ? 'bg-green-100 text-green-800' :
                        response.overall_confidence > 0.6 ? 'bg-yellow-100 text-yellow-800' :
                        'bg-red-100 text-red-800'
                      }`}>
                        {(response.overall_confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  <div className="prose prose-sm max-w-none">
                    <div className="whitespace-pre-wrap text-gray-700">
                      {response.response_text}
                    </div>
                  </div>
                  <div className="mt-4 flex items-center gap-4 text-sm text-gray-600">
                    <span>⏱️ {response.synthesis_time_ms.toFixed(0)}ms</span>
                    <span>📚 {response.citations.length} citations</span>
                    <span>🧠 {response.reasoning_chain.length} steps</span>
                  </div>
                </div>

                {/* Citations */}
                <div className="bg-white rounded-xl shadow-sm p-6">
                  <h2 className="text-xl font-semibold text-gray-900 mb-4">
                    Citations ({response.citations.length})
                  </h2>
                  <div className="space-y-3 max-h-64 overflow-y-auto">
                    {response.citations.map((cit, idx) => (
                      <div key={idx} className="p-3 bg-blue-50 rounded-lg border border-blue-200">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm font-medium text-blue-900">{cit.source}</span>
                          {cit.section && (
                            <span className="text-xs text-blue-600">{cit.section}</span>
                          )}
                        </div>
                        <p className="text-xs text-gray-700 italic">"{cit.text_excerpt}"</p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Reasoning Chain (if XAI enabled) */}
                {showReasoning && response.reasoning_chain && (
                  <div className="bg-white rounded-xl shadow-sm p-6">
                    <h2 className="text-xl font-semibold text-gray-900 mb-4">
                      Reasoning Chain (XAI)
                    </h2>
                    <div className="space-y-3">
                      {response.reasoning_chain.map((step, idx) => (
                        <div key={idx} className="p-4 bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg border border-purple-200">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-semibold text-purple-900">
                              Step {idx + 1}: {step.step_name}
                            </span>
                            <span className="text-xs text-gray-600">
                              {step.latency_ms.toFixed(0)}ms • {(step.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                          <p className="text-sm text-gray-700">{step.description}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}

            {!response && (
              <div className="bg-white rounded-xl shadow-sm p-12 text-center">
                <div className="text-gray-400">
                  <svg className="mx-auto h-12 w-12 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <p className="text-lg">Enter a query or run demo to see synthesis results</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
