import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function PerformanceTester({ models }) {
  const [selectedModel, setSelectedModel] = useState('gemini-2.0-flash-exp');
  const [testPrompt, setTestPrompt] = useState('Explain quantum computing in simple terms');
  const [numIterations, setNumIterations] = useState(5);
  const [performanceData, setPerformanceData] = useState(null);
  const [loading, setLoading] = useState(false);

  const runPerformanceTest = async () => {
    setLoading(true);
    setPerformanceData(null); // Clear previous data
    try {
      const response = await fetch('http://localhost:8000/analyze/performance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: selectedModel,
          test_prompt: testPrompt,
          num_iterations: numIterations
        })
      });
      
      if (!response.ok) {
        let errorMessage = `HTTP error! status: ${response.status}`;
        try {
          const errorData = await response.json();
          
          // Handle quota/rate limit errors
          if (response.status === 429) {
            errorMessage = 'API Quota Exceeded: The free tier quota for Gemini API has been reached.\n\n';
            if (errorData.detail) {
              errorMessage += `Details: ${errorData.detail}\n\n`;
            }
            errorMessage += 'Options:\n';
            errorMessage += '1. Wait for quota reset (usually daily)\n';
            errorMessage += '2. Upgrade your API plan at https://ai.google.dev/pricing\n';
            errorMessage += '3. Use mock data for demonstration purposes';
          } else {
            errorMessage = errorData.detail || errorData.message || errorMessage;
          }
        } catch (e) {
          // If JSON parsing fails, use default message
          if (response.status === 429) {
            errorMessage = 'API Quota Exceeded: The free tier quota for Gemini API has been reached.';
          }
        }
        throw new Error(errorMessage);
      }
      
      const data = await response.json();
      
      // Validate that the response has the expected structure
      if (data && data.latency_ms && data.latency_ms.mean !== undefined) {
        setPerformanceData(data);
      } else {
        throw new Error('Invalid response format from server');
      }
    } catch (error) {
      console.error('Error running performance test:', error);
      
      // Show user-friendly error message
      const errorMsg = error.message || 'Unknown error occurred';
      alert(`Performance test failed:\n\n${errorMsg}`);
      
      setPerformanceData(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="performance-tester">
      <div className="section-header">
        <h2>Performance Tester</h2>
        <p>Measure real-world latency and compare against specifications</p>
        <div style={{ 
          background: '#fff3cd', 
          border: '1px solid #ffc107', 
          borderRadius: '6px', 
          padding: '0.75rem', 
          marginTop: '1rem',
          fontSize: '0.9rem'
        }}>
          <strong>ℹ️ Note:</strong> This feature requires a valid Gemini API key with available quota. 
          If you see a quota error, the free tier limit has been reached. 
          You can still use other features like Cost Calculator and Model Comparison.
        </div>
      </div>

      <div className="tester-form">
        <div className="form-group">
          <label>Select Model</label>
          <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
            {Object.keys(models).map(modelName => (
              <option key={modelName} value={modelName}>{modelName}</option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label>Test Prompt</label>
          <textarea 
            value={testPrompt}
            onChange={(e) => setTestPrompt(e.target.value)}
            rows={3}
            placeholder="Enter a test prompt to measure inference latency..."
          />
        </div>

        <div className="form-group">
          <label>Number of Iterations</label>
          <input 
            type="number" 
            value={numIterations}
            onChange={(e) => setNumIterations(parseInt(e.target.value) || 1)}
            min="1"
            max="20"
          />
        </div>

        <button 
          className="test-btn" 
          onClick={runPerformanceTest} 
          disabled={loading}
        >
          {loading ? 'Testing...' : '⚡ Run Performance Test'}
        </button>
      </div>

      {performanceData && performanceData.latency_ms && performanceData.latency_ms.mean !== undefined && (
        <div className="performance-results">
          <div className="perf-cards">
            <div className="perf-card">
              <h3>Mean Latency</h3>
              <div className="perf-value">{performanceData.latency_ms.mean.toFixed(2)} ms</div>
            </div>
            <div className="perf-card">
              <h3>P95 Latency</h3>
              <div className="perf-value">{performanceData.latency_ms.p95?.toFixed(2) || 'N/A'} ms</div>
            </div>
            <div className="perf-card">
              <h3>Min / Max</h3>
              <div className="perf-value">
                {performanceData.latency_ms.min?.toFixed(2) || 'N/A'} / {performanceData.latency_ms.max?.toFixed(2) || 'N/A'} ms
              </div>
            </div>
            <div className="perf-card">
              <h3>Specification</h3>
              <div className="perf-value">{performanceData.estimated_spec || 'N/A'} ms</div>
            </div>
          </div>

          <div className="variance-analysis">
            <h3>Variance from Specification</h3>
            {performanceData.variance_from_spec !== undefined && (
              <div className={`variance-indicator ${Math.abs(performanceData.variance_from_spec) < 50 ? 'good' : 'warning'}`}>
                {performanceData.variance_from_spec > 0 ? '+' : ''}{performanceData.variance_from_spec.toFixed(2)} ms
                <span className="variance-label">
                  {Math.abs(performanceData.variance_from_spec) < 50 ? '✓ Within expected range' : '⚠ Significant variance'}
                </span>
              </div>
            )}
          </div>

          <div className="token-analysis">
            <h3>Token Analysis</h3>
            {performanceData.input_tokens && (
              <>
                <p><strong>Average Input Tokens:</strong> {performanceData.input_tokens.mean || 'N/A'}</p>
                <p><strong>Max Input Tokens:</strong> {performanceData.input_tokens.max || 'N/A'}</p>
              </>
            )}
            <p><strong>Test Iterations:</strong> {performanceData.iterations || 'N/A'}</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default PerformanceTester;
