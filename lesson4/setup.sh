#!/bin/bash

# L4: Understanding LLM Parameters - Automated Setup
# Complete implementation with Parameter Analysis Platform

set -e

PROJECT_NAME="l4-llm-parameters"
GEMINI_API_KEY="AIzaSyDGswqDT4wQw_bd4WZtIgYAawRDZ0Gisn8"

echo "🚀 L4: Understanding LLM Parameters - Setup Starting..."

# Create project structure
mkdir -p $PROJECT_NAME/{backend,frontend/src/{components,services,hooks},frontend/public,scripts,docs}
cd $PROJECT_NAME

# Backend: Parameter Analysis Engine
cat > backend/main.py << 'EOF'
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from typing import List, Dict, Optional
import asyncio
from datetime import datetime
import statistics

app = FastAPI(title="LLM Parameter Analysis Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini
genai.configure(api_key="AIzaSyDGswqDT4wQw_bd4WZtIgYAawRDZ0Gisn8")

# Model specifications database (May 2025)
MODEL_SPECS = {
    "gemini-2.0-flash-exp": {
        "parameters": "7B",
        "context_window": 1000000,
        "input_price_per_1m": 0.075,
        "output_price_per_1m": 0.30,
        "cached_input_price_per_1m": 0.01875,
        "latency_estimate_ms": 85,
        "max_rpm": 1500,
        "description": "Ultra-fast inference for high-volume tasks"
    },
    "gemini-1.5-pro": {
        "parameters": "70B",
        "context_window": 2000000,
        "input_price_per_1m": 1.25,
        "output_price_per_1m": 5.00,
        "cached_input_price_per_1m": 0.3125,
        "latency_estimate_ms": 420,
        "max_rpm": 360,
        "description": "Balanced performance for production workloads"
    },
    "gemini-1.5-flash": {
        "parameters": "7B",
        "context_window": 1000000,
        "input_price_per_1m": 0.075,
        "output_price_per_1m": 0.30,
        "cached_input_price_per_1m": 0.01875,
        "latency_estimate_ms": 95,
        "max_rpm": 1500,
        "description": "Cost-effective for classification and extraction"
    }
}

class CostCalculationRequest(BaseModel):
    model_name: str
    requests_per_day: int
    avg_input_tokens: int
    avg_output_tokens: int
    cache_hit_rate: float = 0.5

class PerformanceTestRequest(BaseModel):
    model_name: str
    test_prompt: str
    num_iterations: int = 5

class ContextAnalysisRequest(BaseModel):
    model_name: str
    sample_texts: List[str]

@app.get("/")
async def root():
    return {
        "service": "LLM Parameter Analysis Platform",
        "version": "1.0.0",
        "lesson": "L4",
        "status": "operational"
    }

@app.get("/models")
async def get_models():
    """Retrieve all available model specifications"""
    return {
        "models": MODEL_SPECS,
        "count": len(MODEL_SPECS),
        "last_updated": "2025-05-01"
    }

@app.get("/models/{model_name}")
async def get_model_specs(model_name: str):
    """Get detailed specifications for a specific model"""
    if model_name not in MODEL_SPECS:
        raise HTTPException(status_code=404, detail="Model not found")
    
    specs = MODEL_SPECS[model_name]
    
    # Calculate additional metrics
    params_numeric = float(specs["parameters"].replace("B", ""))
    compute_intensity = params_numeric * specs["context_window"] / 1_000_000
    
    return {
        **specs,
        "compute_intensity_score": round(compute_intensity, 2),
        "cost_efficiency_score": round(
            1000 / (specs["input_price_per_1m"] + specs["output_price_per_1m"]), 2
        ),
        "speed_score": round(1000 / specs["latency_estimate_ms"], 2)
    }

@app.post("/analyze/cost")
async def calculate_cost(request: CostCalculationRequest):
    """Calculate projected monthly costs with caching benefits"""
    
    if request.model_name not in MODEL_SPECS:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = MODEL_SPECS[request.model_name]
    
    # Calculate effective input tokens with caching
    cache_savings_factor = 1 - (request.cache_hit_rate * 0.75)
    effective_input_tokens = request.avg_input_tokens * cache_savings_factor
    
    # Daily costs
    cached_input_cost = (request.avg_input_tokens * request.cache_hit_rate * 
                        model["cached_input_price_per_1m"] / 1_000_000)
    uncached_input_cost = (request.avg_input_tokens * (1 - request.cache_hit_rate) * 
                          model["input_price_per_1m"] / 1_000_000)
    output_cost = request.avg_output_tokens * model["output_price_per_1m"] / 1_000_000
    
    cost_per_request = cached_input_cost + uncached_input_cost + output_cost
    daily_cost = cost_per_request * request.requests_per_day
    
    # Without caching
    cost_without_cache = (
        (request.avg_input_tokens * model["input_price_per_1m"] / 1_000_000) +
        (request.avg_output_tokens * model["output_price_per_1m"] / 1_000_000)
    ) * request.requests_per_day
    
    cache_savings = cost_without_cache - daily_cost
    
    return {
        "model": request.model_name,
        "cost_per_request": round(cost_per_request, 6),
        "daily_cost": round(daily_cost, 2),
        "monthly_cost": round(daily_cost * 30, 2),
        "yearly_cost": round(daily_cost * 365, 2),
        "cache_savings_daily": round(cache_savings, 2),
        "cache_savings_monthly": round(cache_savings * 30, 2),
        "total_requests_monthly": request.requests_per_day * 30,
        "breakdown": {
            "cached_input_cost_per_request": round(cached_input_cost, 8),
            "uncached_input_cost_per_request": round(uncached_input_cost, 8),
            "output_cost_per_request": round(output_cost, 8)
        }
    }

@app.post("/analyze/performance")
async def test_performance(request: PerformanceTestRequest):
    """Run actual inference tests to measure performance"""
    
    if request.model_name not in MODEL_SPECS:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        model = genai.GenerativeModel(request.model_name)
        latencies = []
        token_counts = []
        
        for i in range(request.num_iterations):
            start_time = datetime.now()
            
            response = model.generate_content(
                request.test_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=100,
                    temperature=0.7,
                )
            )
            
            end_time = datetime.now()
            latency = (end_time - start_time).total_seconds() * 1000
            
            latencies.append(latency)
            
            # Count tokens
            token_count = model.count_tokens(request.test_prompt)
            token_counts.append(token_count.total_tokens)
            
            await asyncio.sleep(0.5)  # Rate limit compliance
        
        return {
            "model": request.model_name,
            "iterations": request.num_iterations,
            "latency_ms": {
                "min": round(min(latencies), 2),
                "max": round(max(latencies), 2),
                "mean": round(statistics.mean(latencies), 2),
                "median": round(statistics.median(latencies), 2),
                "p95": round(statistics.quantiles(latencies, n=20)[18], 2) if len(latencies) > 2 else round(max(latencies), 2)
            },
            "input_tokens": {
                "mean": round(statistics.mean(token_counts), 2),
                "max": max(token_counts)
            },
            "estimated_spec": MODEL_SPECS[request.model_name]["latency_estimate_ms"],
            "variance_from_spec": round(
                statistics.mean(latencies) - MODEL_SPECS[request.model_name]["latency_estimate_ms"], 2
            )
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance test failed: {str(e)}")

@app.post("/analyze/context")
async def analyze_context_usage(request: ContextAnalysisRequest):
    """Analyze context window efficiency from sample texts"""
    
    if request.model_name not in MODEL_SPECS:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        model = genai.GenerativeModel(request.model_name)
        token_counts = []
        
        for text in request.sample_texts:
            count = model.count_tokens(text)
            token_counts.append(count.total_tokens)
        
        if not token_counts:
            raise HTTPException(status_code=400, detail="No valid texts provided")
        
        sorted_counts = sorted(token_counts)
        p50_idx = len(sorted_counts) // 2
        p95_idx = int(len(sorted_counts) * 0.95)
        p99_idx = int(len(sorted_counts) * 0.99)
        
        p50 = sorted_counts[p50_idx]
        p95 = sorted_counts[p95_idx] if p95_idx < len(sorted_counts) else sorted_counts[-1]
        p99 = sorted_counts[p99_idx] if p99_idx < len(sorted_counts) else sorted_counts[-1]
        
        context_limit = MODEL_SPECS[request.model_name]["context_window"]
        recommended_limit = p99
        
        # Calculate potential savings
        avg_tokens = statistics.mean(token_counts)
        potential_savings_pct = ((context_limit - recommended_limit) / context_limit) * 100
        
        return {
            "model": request.model_name,
            "samples_analyzed": len(token_counts),
            "token_distribution": {
                "min": min(token_counts),
                "max": max(token_counts),
                "mean": round(avg_tokens, 2),
                "median": p50,
                "p95": p95,
                "p99": p99
            },
            "context_window_max": context_limit,
            "recommended_context_limit": recommended_limit,
            "efficiency_metrics": {
                "utilization_rate": round((recommended_limit / context_limit) * 100, 2),
                "potential_savings_pct": round(potential_savings_pct, 2),
                "tokens_saved_per_request": context_limit - recommended_limit
            },
            "recommendations": [
                f"Set context limit to {recommended_limit} tokens (covers 99% of requests)",
                f"Potential cost reduction: {round(potential_savings_pct, 1)}%",
                "Implement context compression for outliers above p99"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context analysis failed: {str(e)}")

@app.post("/compare")
async def compare_models(
    requests_per_day: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    cache_hit_rate: float = 0.5
):
    """Compare all models for given usage pattern"""
    
    comparisons = []
    
    for model_name, specs in MODEL_SPECS.items():
        # Calculate costs
        cache_savings_factor = 1 - (cache_hit_rate * 0.75)
        
        cached_input_cost = (avg_input_tokens * cache_hit_rate * 
                            specs["cached_input_price_per_1m"] / 1_000_000)
        uncached_input_cost = (avg_input_tokens * (1 - cache_hit_rate) * 
                              specs["input_price_per_1m"] / 1_000_000)
        output_cost = avg_output_tokens * specs["output_price_per_1m"] / 1_000_000
        
        cost_per_request = cached_input_cost + uncached_input_cost + output_cost
        monthly_cost = cost_per_request * requests_per_day * 30
        
        comparisons.append({
            "model": model_name,
            "parameters": specs["parameters"],
            "monthly_cost": round(monthly_cost, 2),
            "cost_per_request": round(cost_per_request, 6),
            "latency_ms": specs["latency_estimate_ms"],
            "context_window": specs["context_window"],
            "max_rpm": specs["max_rpm"],
            "cost_efficiency_score": round(1000 / monthly_cost, 4),
            "speed_score": round(1000 / specs["latency_estimate_ms"], 2)
        })
    
    # Sort by cost efficiency
    comparisons.sort(key=lambda x: x["cost_efficiency_score"], reverse=True)
    
    return {
        "usage_pattern": {
            "requests_per_day": requests_per_day,
            "avg_input_tokens": avg_input_tokens,
            "avg_output_tokens": avg_output_tokens,
            "cache_hit_rate": cache_hit_rate
        },
        "comparisons": comparisons,
        "recommendation": comparisons[0]["model"],
        "cost_range": {
            "min": min(c["monthly_cost"] for c in comparisons),
            "max": max(c["monthly_cost"] for c in comparisons)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

# Backend requirements
cat > backend/requirements.txt << 'EOF'
fastapi==0.115.0
uvicorn[standard]==0.32.0
google-generativeai==0.8.3
pydantic==2.9.2
python-multipart==0.0.12
httpx==0.27.2
EOF

# Frontend: React Dashboard
cat > frontend/package.json << 'EOF'
{
  "name": "l4-parameter-analysis-dashboard",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-scripts": "5.0.1",
    "recharts": "^2.12.7",
    "axios": "^1.7.7"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": ["react-app"]
  },
  "browserslist": {
    "production": [">0.2%", "not dead", "not op_mini all"],
    "development": ["last 1 chrome version", "last 1 firefox version", "last 1 safari version"]
  }
}
EOF

# Frontend: Main App
cat > frontend/src/App.js << 'EOF'
import React, { useState, useEffect } from 'react';
import './App.css';
import ModelComparison from './components/ModelComparison';
import CostCalculator from './components/CostCalculator';
import PerformanceTester from './components/PerformanceTester';
import ContextAnalyzer from './components/ContextAnalyzer';

function App() {
  const [models, setModels] = useState({});
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('comparison');

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await fetch('http://localhost:8000/models');
      const data = await response.json();
      setModels(data.models);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching models:', error);
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="App loading">
        <div className="loader"></div>
        <p>Loading LLM Parameter Analysis Platform...</p>
      </div>
    );
  }

  return (
    <div className="App">
      <header className="app-header">
        <div className="header-content">
          <h1>🧠 LLM Parameter Analysis Platform</h1>
          <p className="subtitle">L4: Understanding Model Parameters, Context Windows & Cost Trade-offs</p>
        </div>
        <div className="header-stats">
          <div className="stat-card">
            <span className="stat-label">Models Analyzed</span>
            <span className="stat-value">{Object.keys(models).length}</span>
          </div>
          <div className="stat-card">
            <span className="stat-label">Max Context</span>
            <span className="stat-value">2M tokens</span>
          </div>
          <div className="stat-card">
            <span className="stat-label">Parameter Range</span>
            <span className="stat-value">7B - 405B+</span>
          </div>
        </div>
      </header>

      <nav className="tab-nav">
        <button 
          className={`tab-button ${activeTab === 'comparison' ? 'active' : ''}`}
          onClick={() => setActiveTab('comparison')}
        >
          📊 Model Comparison
        </button>
        <button 
          className={`tab-button ${activeTab === 'cost' ? 'active' : ''}`}
          onClick={() => setActiveTab('cost')}
        >
          💰 Cost Calculator
        </button>
        <button 
          className={`tab-button ${activeTab === 'performance' ? 'active' : ''}`}
          onClick={() => setActiveTab('performance')}
        >
          ⚡ Performance Tester
        </button>
        <button 
          className={`tab-button ${activeTab === 'context' ? 'active' : ''}`}
          onClick={() => setActiveTab('context')}
        >
          📝 Context Analyzer
        </button>
      </nav>

      <main className="app-main">
        {activeTab === 'comparison' && <ModelComparison models={models} />}
        {activeTab === 'cost' && <CostCalculator models={models} />}
        {activeTab === 'performance' && <PerformanceTester models={models} />}
        {activeTab === 'context' && <ContextAnalyzer models={models} />}
      </main>

      <footer className="app-footer">
        <p>VAIA Curriculum - Lesson 4 | Building on L3: Transformer Architecture | Preparing for L5: Model Landscape</p>
      </footer>
    </div>
  );
}

export default App;
EOF

# Frontend: Model Comparison Component
cat > frontend/src/components/ModelComparison.js << 'EOF'
import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts';

function ModelComparison({ models }) {
  const [comparisonData, setComparisonData] = useState(null);
  const [usagePattern, setUsagePattern] = useState({
    requestsPerDay: 10000,
    avgInputTokens: 500,
    avgOutputTokens: 200,
    cacheHitRate: 0.5
  });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (models && Object.keys(models).length > 0) {
      fetchComparison();
    }
  }, [models, usagePattern]);

  const fetchComparison = async () => {
    setLoading(true);
    try {
      const response = await fetch(
        `http://localhost:8000/compare?` +
        `requests_per_day=${usagePattern.requestsPerDay}&` +
        `avg_input_tokens=${usagePattern.avgInputTokens}&` +
        `avg_output_tokens=${usagePattern.avgOutputTokens}&` +
        `cache_hit_rate=${usagePattern.cacheHitRate}`
      , { method: 'POST' });
      
      const data = await response.json();
      setComparisonData(data);
    } catch (error) {
      console.error('Error fetching comparison:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (field, value) => {
    setUsagePattern(prev => ({ ...prev, [field]: parseFloat(value) || 0 }));
  };

  if (!comparisonData) {
    return <div className="loading">Loading comparison data...</div>;
  }

  return (
    <div className="model-comparison">
      <div className="section-header">
        <h2>Model Comparison Matrix</h2>
        <p>Compare LLM parameters, costs, and performance across your usage pattern</p>
      </div>

      <div className="usage-config">
        <h3>Configure Usage Pattern</h3>
        <div className="config-grid">
          <div className="config-item">
            <label>Requests per Day</label>
            <input 
              type="number" 
              value={usagePattern.requestsPerDay}
              onChange={(e) => handleInputChange('requestsPerDay', e.target.value)}
              min="1"
            />
          </div>
          <div className="config-item">
            <label>Avg Input Tokens</label>
            <input 
              type="number" 
              value={usagePattern.avgInputTokens}
              onChange={(e) => handleInputChange('avgInputTokens', e.target.value)}
              min="1"
            />
          </div>
          <div className="config-item">
            <label>Avg Output Tokens</label>
            <input 
              type="number" 
              value={usagePattern.avgOutputTokens}
              onChange={(e) => handleInputChange('avgOutputTokens', e.target.value)}
              min="1"
            />
          </div>
          <div className="config-item">
            <label>Cache Hit Rate</label>
            <input 
              type="number" 
              step="0.1"
              value={usagePattern.cacheHitRate}
              onChange={(e) => handleInputChange('cacheHitRate', e.target.value)}
              min="0"
              max="1"
            />
          </div>
        </div>
      </div>

      <div className="recommendation-banner">
        <h3>💡 Recommended Model: {comparisonData.recommendation}</h3>
        <p>
          Best cost-efficiency for your usage pattern. 
          Monthly cost range: ${comparisonData.cost_range.min} - ${comparisonData.cost_range.max}
        </p>
      </div>

      <div className="charts-container">
        <div className="chart-box">
          <h3>Monthly Cost Comparison</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={comparisonData.comparisons}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis dataKey="model" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} label={{ value: 'Cost ($)', angle: -90, position: 'insideLeft' }} />
              <Tooltip contentStyle={{ backgroundColor: '#fff', border: '1px solid #ddd' }} />
              <Bar dataKey="monthly_cost" fill="#4CAF50" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-box">
          <h3>Latency vs Cost Trade-off</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={comparisonData.comparisons}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis dataKey="model" tick={{ fontSize: 12 }} />
              <YAxis yAxisId="left" tick={{ fontSize: 12 }} label={{ value: 'Latency (ms)', angle: -90, position: 'insideLeft' }} />
              <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 12 }} label={{ value: 'Cost/Req ($)', angle: 90, position: 'insideRight' }} />
              <Tooltip contentStyle={{ backgroundColor: '#fff', border: '1px solid #ddd' }} />
              <Legend />
              <Line yAxisId="left" type="monotone" dataKey="latency_ms" stroke="#2196F3" strokeWidth={2} />
              <Line yAxisId="right" type="monotone" dataKey="cost_per_request" stroke="#FF9800" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="comparison-table">
        <h3>Detailed Specifications</h3>
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Parameters</th>
              <th>Context Window</th>
              <th>Monthly Cost</th>
              <th>Cost/Request</th>
              <th>Latency (ms)</th>
              <th>Max RPM</th>
              <th>Efficiency Score</th>
            </tr>
          </thead>
          <tbody>
            {comparisonData.comparisons.map((model, idx) => (
              <tr key={idx} className={model.model === comparisonData.recommendation ? 'recommended' : ''}>
                <td><strong>{model.model}</strong></td>
                <td>{model.parameters}</td>
                <td>{(model.context_window / 1000).toFixed(0)}K</td>
                <td>${model.monthly_cost}</td>
                <td>${model.cost_per_request.toFixed(6)}</td>
                <td>{model.latency_ms}</td>
                <td>{model.max_rpm}</td>
                <td>{model.cost_efficiency_score.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default ModelComparison;
EOF

# Frontend: Cost Calculator Component
cat > frontend/src/components/CostCalculator.js << 'EOF'
import React, { useState } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

function CostCalculator({ models }) {
  const [selectedModel, setSelectedModel] = useState('gemini-2.0-flash-exp');
  const [requestsPerDay, setRequestsPerDay] = useState(10000);
  const [avgInputTokens, setAvgInputTokens] = useState(500);
  const [avgOutputTokens, setAvgOutputTokens] = useState(200);
  const [cacheHitRate, setCacheHitRate] = useState(0.5);
  const [costData, setCostData] = useState(null);
  const [loading, setLoading] = useState(false);

  const calculateCost = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/analyze/cost', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: selectedModel,
          requests_per_day: requestsPerDay,
          avg_input_tokens: avgInputTokens,
          avg_output_tokens: avgOutputTokens,
          cache_hit_rate: cacheHitRate
        })
      });
      const data = await response.json();
      setCostData(data);
    } catch (error) {
      console.error('Error calculating cost:', error);
    } finally {
      setLoading(false);
    }
  };

  const COLORS = ['#4CAF50', '#FF9800', '#2196F3'];

  const pieData = costData ? [
    { name: 'Cached Input', value: costData.breakdown.cached_input_cost_per_request * requestsPerDay * 30 },
    { name: 'Uncached Input', value: costData.breakdown.uncached_input_cost_per_request * requestsPerDay * 30 },
    { name: 'Output', value: costData.breakdown.output_cost_per_request * requestsPerDay * 30 }
  ] : [];

  return (
    <div className="cost-calculator">
      <div className="section-header">
        <h2>Cost Calculator</h2>
        <p>Project monthly costs with caching optimization</p>
      </div>

      <div className="calculator-form">
        <div className="form-row">
          <div className="form-group">
            <label>Select Model</label>
            <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
              {Object.keys(models).map(modelName => (
                <option key={modelName} value={modelName}>{modelName}</option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label>Requests per Day</label>
            <input 
              type="number" 
              value={requestsPerDay}
              onChange={(e) => setRequestsPerDay(parseInt(e.target.value) || 0)}
              min="1"
            />
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label>Average Input Tokens</label>
            <input 
              type="number" 
              value={avgInputTokens}
              onChange={(e) => setAvgInputTokens(parseInt(e.target.value) || 0)}
              min="1"
            />
          </div>
          <div className="form-group">
            <label>Average Output Tokens</label>
            <input 
              type="number" 
              value={avgOutputTokens}
              onChange={(e) => setAvgOutputTokens(parseInt(e.target.value) || 0)}
              min="1"
            />
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label>Cache Hit Rate (0.0 - 1.0)</label>
            <input 
              type="number" 
              step="0.1"
              value={cacheHitRate}
              onChange={(e) => setCacheHitRate(parseFloat(e.target.value) || 0)}
              min="0"
              max="1"
            />
          </div>
          <div className="form-group">
            <button className="calculate-btn" onClick={calculateCost} disabled={loading}>
              {loading ? 'Calculating...' : '💰 Calculate Cost'}
            </button>
          </div>
        </div>
      </div>

      {costData && (
        <div className="cost-results">
          <div className="cost-cards">
            <div className="cost-card primary">
              <h3>Monthly Cost</h3>
              <div className="cost-value">${costData.monthly_cost.toFixed(2)}</div>
              <p>{costData.total_requests_monthly.toLocaleString()} requests</p>
            </div>
            <div className="cost-card">
              <h3>Per Request</h3>
              <div className="cost-value">${costData.cost_per_request.toFixed(6)}</div>
              <p>Average cost</p>
            </div>
            <div className="cost-card">
              <h3>Yearly Projection</h3>
              <div className="cost-value">${costData.yearly_cost.toFixed(2)}</div>
              <p>Annual spend</p>
            </div>
            <div className="cost-card success">
              <h3>Cache Savings</h3>
              <div className="cost-value">${costData.cache_savings_monthly.toFixed(2)}</div>
              <p>Monthly savings</p>
            </div>
          </div>

          <div className="cost-visualization">
            <div className="chart-section">
              <h3>Cost Breakdown by Token Type</h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={entry => `${entry.name}: $${entry.value.toFixed(2)}`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="breakdown-details">
              <h3>Detailed Breakdown</h3>
              <table>
                <thead>
                  <tr>
                    <th>Component</th>
                    <th>Cost per Request</th>
                    <th>Monthly Cost</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Cached Input Tokens</td>
                    <td>${costData.breakdown.cached_input_cost_per_request.toFixed(8)}</td>
                    <td>${(costData.breakdown.cached_input_cost_per_request * requestsPerDay * 30).toFixed(2)}</td>
                  </tr>
                  <tr>
                    <td>Uncached Input Tokens</td>
                    <td>${costData.breakdown.uncached_input_cost_per_request.toFixed(8)}</td>
                    <td>${(costData.breakdown.uncached_input_cost_per_request * requestsPerDay * 30).toFixed(2)}</td>
                  </tr>
                  <tr>
                    <td>Output Tokens</td>
                    <td>${costData.breakdown.output_cost_per_request.toFixed(8)}</td>
                    <td>${(costData.breakdown.output_cost_per_request * requestsPerDay * 30).toFixed(2)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default CostCalculator;
EOF

# Frontend: Performance Tester Component
cat > frontend/src/components/PerformanceTester.js << 'EOF'
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
      const data = await response.json();
      setPerformanceData(data);
    } catch (error) {
      console.error('Error running performance test:', error);
      alert('Performance test failed. Check console for details.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="performance-tester">
      <div className="section-header">
        <h2>Performance Tester</h2>
        <p>Measure real-world latency and compare against specifications</p>
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

      {performanceData && (
        <div className="performance-results">
          <div className="perf-cards">
            <div className="perf-card">
              <h3>Mean Latency</h3>
              <div className="perf-value">{performanceData.latency_ms.mean.toFixed(2)} ms</div>
            </div>
            <div className="perf-card">
              <h3>P95 Latency</h3>
              <div className="perf-value">{performanceData.latency_ms.p95.toFixed(2)} ms</div>
            </div>
            <div className="perf-card">
              <h3>Min / Max</h3>
              <div className="perf-value">
                {performanceData.latency_ms.min.toFixed(2)} / {performanceData.latency_ms.max.toFixed(2)} ms
              </div>
            </div>
            <div className="perf-card">
              <h3>Specification</h3>
              <div className="perf-value">{performanceData.estimated_spec} ms</div>
            </div>
          </div>

          <div className="variance-analysis">
            <h3>Variance from Specification</h3>
            <div className={`variance-indicator ${Math.abs(performanceData.variance_from_spec) < 50 ? 'good' : 'warning'}`}>
              {performanceData.variance_from_spec > 0 ? '+' : ''}{performanceData.variance_from_spec.toFixed(2)} ms
              <span className="variance-label">
                {Math.abs(performanceData.variance_from_spec) < 50 ? '✓ Within expected range' : '⚠ Significant variance'}
              </span>
            </div>
          </div>

          <div className="token-analysis">
            <h3>Token Analysis</h3>
            <p><strong>Average Input Tokens:</strong> {performanceData.input_tokens.mean}</p>
            <p><strong>Max Input Tokens:</strong> {performanceData.input_tokens.max}</p>
            <p><strong>Test Iterations:</strong> {performanceData.iterations}</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default PerformanceTester;
EOF

# Frontend: Context Analyzer Component
cat > frontend/src/components/ContextAnalyzer.js << 'EOF'
import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function ContextAnalyzer({ models }) {
  const [selectedModel, setSelectedModel] = useState('gemini-2.0-flash-exp');
  const [sampleTexts, setSampleTexts] = useState('');
  const [contextData, setContextData] = useState(null);
  const [loading, setLoading] = useState(false);

  const analyzeContext = async () => {
    const texts = sampleTexts.split('\n').filter(t => t.trim().length > 0);
    
    if (texts.length === 0) {
      alert('Please enter at least one sample text (one per line)');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/analyze/context', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: selectedModel,
          sample_texts: texts
        })
      });
      const data = await response.json();
      setContextData(data);
    } catch (error) {
      console.error('Error analyzing context:', error);
      alert('Context analysis failed. Check console for details.');
    } finally {
      setLoading(false);
    }
  };

  const distributionData = contextData ? [
    { name: 'Min', value: contextData.token_distribution.min },
    { name: 'Mean', value: contextData.token_distribution.mean },
    { name: 'Median', value: contextData.token_distribution.median },
    { name: 'P95', value: contextData.token_distribution.p95 },
    { name: 'P99', value: contextData.token_distribution.p99 },
    { name: 'Max', value: contextData.token_distribution.max }
  ] : [];

  return (
    <div className="context-analyzer">
      <div className="section-header">
        <h2>Context Window Analyzer</h2>
        <p>Optimize context usage and identify cost savings opportunities</p>
      </div>

      <div className="analyzer-form">
        <div className="form-group">
          <label>Select Model</label>
          <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
            {Object.keys(models).map(modelName => (
              <option key={modelName} value={modelName}>{modelName}</option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label>Sample Texts (one per line)</label>
          <textarea 
            value={sampleTexts}
            onChange={(e) => setSampleTexts(e.target.value)}
            rows={10}
            placeholder="Enter sample texts to analyze token distribution...
Example:
What is machine learning?
Explain the benefits of cloud computing
How do I optimize my database queries?"
          />
        </div>

        <button 
          className="analyze-btn" 
          onClick={analyzeContext} 
          disabled={loading}
        >
          {loading ? 'Analyzing...' : '📝 Analyze Context Usage'}
        </button>
      </div>

      {contextData && (
        <div className="context-results">
          <div className="context-summary">
            <h3>Analysis Summary</h3>
            <div className="summary-grid">
              <div className="summary-item">
                <span className="label">Samples Analyzed:</span>
                <span className="value">{contextData.samples_analyzed}</span>
              </div>
              <div className="summary-item">
                <span className="label">Model Context Limit:</span>
                <span className="value">{(contextData.context_window_max / 1000).toFixed(0)}K tokens</span>
              </div>
              <div className="summary-item">
                <span className="label">Recommended Limit:</span>
                <span className="value">{contextData.recommended_context_limit} tokens</span>
              </div>
              <div className="summary-item highlight">
                <span className="label">Potential Savings:</span>
                <span className="value">{contextData.efficiency_metrics.potential_savings_pct.toFixed(1)}%</span>
              </div>
            </div>
          </div>

          <div className="distribution-chart">
            <h3>Token Distribution</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={distributionData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} label={{ value: 'Tokens', angle: -90, position: 'insideLeft' }} />
                <Tooltip contentStyle={{ backgroundColor: '#fff', border: '1px solid #ddd' }} />
                <Bar dataKey="value" fill="#4CAF50" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="efficiency-metrics">
            <h3>Efficiency Metrics</h3>
            <table>
              <tbody>
                <tr>
                  <td>Context Window Utilization</td>
                  <td><strong>{contextData.efficiency_metrics.utilization_rate.toFixed(2)}%</strong></td>
                </tr>
                <tr>
                  <td>Potential Savings Percentage</td>
                  <td><strong>{contextData.efficiency_metrics.potential_savings_pct.toFixed(2)}%</strong></td>
                </tr>
                <tr>
                  <td>Tokens Saved per Request</td>
                  <td><strong>{contextData.efficiency_metrics.tokens_saved_per_request.toLocaleString()}</strong></td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="recommendations">
            <h3>💡 Recommendations</h3>
            <ul>
              {contextData.recommendations.map((rec, idx) => (
                <li key={idx}>{rec}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

export default ContextAnalyzer;
EOF

# Frontend: CSS Styles
cat > frontend/src/App.css << 'EOF'
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  min-height: 100vh;
}

.App {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.app-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 2rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.header-content h1 {
  font-size: 2.5rem;
  margin-bottom: 0.5rem;
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
}

.subtitle {
  font-size: 1.1rem;
  opacity: 0.95;
  margin-top: 0.5rem;
}

.header-stats {
  display: flex;
  gap: 2rem;
  margin-top: 1.5rem;
}

.stat-card {
  background: rgba(255, 255, 255, 0.15);
  padding: 1rem 1.5rem;
  border-radius: 8px;
  backdrop-filter: blur(10px);
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.stat-label {
  font-size: 0.85rem;
  opacity: 0.9;
}

.stat-value {
  font-size: 1.8rem;
  font-weight: bold;
}

.tab-nav {
  background: white;
  display: flex;
  gap: 0.5rem;
  padding: 1rem 2rem;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.tab-button {
  padding: 0.75rem 1.5rem;
  border: none;
  background: #f5f5f5;
  color: #666;
  font-size: 1rem;
  cursor: pointer;
  border-radius: 8px;
  transition: all 0.3s ease;
  font-weight: 500;
}

.tab-button:hover {
  background: #e0e0e0;
  color: #333;
}

.tab-button.active {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
}

.app-main {
  flex: 1;
  padding: 2rem;
  max-width: 1400px;
  margin: 0 auto;
  width: 100%;
}

.section-header {
  margin-bottom: 2rem;
}

.section-header h2 {
  font-size: 2rem;
  color: #333;
  margin-bottom: 0.5rem;
}

.section-header p {
  color: #666;
  font-size: 1.1rem;
}

/* Model Comparison Styles */
.model-comparison {
  background: white;
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
}

.usage-config {
  background: #f8f9fa;
  padding: 1.5rem;
  border-radius: 8px;
  margin-bottom: 2rem;
}

.usage-config h3 {
  margin-bottom: 1rem;
  color: #333;
}

.config-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1.5rem;
}

.config-item label {
  display: block;
  margin-bottom: 0.5rem;
  color: #666;
  font-weight: 500;
}

.config-item input {
  width: 100%;
  padding: 0.75rem;
  border: 2px solid #e0e0e0;
  border-radius: 6px;
  font-size: 1rem;
  transition: border-color 0.3s ease;
}

.config-item input:focus {
  outline: none;
  border-color: #667eea;
}

.recommendation-banner {
  background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
  padding: 1.5rem;
  border-radius: 8px;
  margin-bottom: 2rem;
  text-align: center;
}

.recommendation-banner h3 {
  font-size: 1.5rem;
  margin-bottom: 0.5rem;
  color: #2c3e50;
}

.charts-container {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
  gap: 2rem;
  margin-bottom: 2rem;
}

.chart-box {
  background: #f8f9fa;
  padding: 1.5rem;
  border-radius: 8px;
}

.chart-box h3 {
  margin-bottom: 1rem;
  color: #333;
}

.comparison-table {
  margin-top: 2rem;
}

.comparison-table table {
  width: 100%;
  border-collapse: collapse;
}

.comparison-table th,
.comparison-table td {
  padding: 1rem;
  text-align: left;
  border-bottom: 1px solid #e0e0e0;
}

.comparison-table th {
  background: #f5f5f5;
  font-weight: 600;
  color: #333;
}

.comparison-table tr.recommended {
  background: #e8f5e9;
  font-weight: 600;
}

/* Cost Calculator Styles */
.cost-calculator {
  background: white;
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
}

.calculator-form {
  background: #f8f9fa;
  padding: 2rem;
  border-radius: 8px;
  margin-bottom: 2rem;
}

.form-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin-bottom: 1.5rem;
}

.form-group {
  display: flex;
  flex-direction: column;
}

.form-group label {
  margin-bottom: 0.5rem;
  color: #666;
  font-weight: 500;
}

.form-group select,
.form-group input {
  padding: 0.75rem;
  border: 2px solid #e0e0e0;
  border-radius: 6px;
  font-size: 1rem;
  transition: border-color 0.3s ease;
}

.form-group select:focus,
.form-group input:focus {
  outline: none;
  border-color: #667eea;
}

.calculate-btn,
.test-btn,
.analyze-btn {
  padding: 1rem 2rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
  box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
}

.calculate-btn:hover,
.test-btn:hover,
.analyze-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
}

.calculate-btn:disabled,
.test-btn:disabled,
.analyze-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

.cost-results {
  margin-top: 2rem;
}

.cost-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
}

.cost-card {
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  padding: 1.5rem;
  border-radius: 12px;
  text-align: center;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
}

.cost-card.primary {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.cost-card.success {
  background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
}

.cost-card h3 {
  font-size: 0.9rem;
  margin-bottom: 0.75rem;
  opacity: 0.9;
}

.cost-value {
  font-size: 2rem;
  font-weight: bold;
  margin-bottom: 0.5rem;
}

.cost-card p {
  font-size: 0.85rem;
  opacity: 0.8;
}

.cost-visualization {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
  margin-top: 2rem;
}

.chart-section,
.breakdown-details {
  background: #f8f9fa;
  padding: 1.5rem;
  border-radius: 8px;
}

.breakdown-details table {
  width: 100%;
  margin-top: 1rem;
}

.breakdown-details td {
  padding: 0.75rem;
  border-bottom: 1px solid #e0e0e0;
}

.breakdown-details tr:last-child td {
  border-bottom: none;
}

/* Performance Tester Styles */
.performance-tester {
  background: white;
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
}

.tester-form {
  background: #f8f9fa;
  padding: 2rem;
  border-radius: 8px;
  margin-bottom: 2rem;
}

.tester-form textarea {
  width: 100%;
  padding: 1rem;
  border: 2px solid #e0e0e0;
  border-radius: 6px;
  font-size: 1rem;
  font-family: inherit;
  resize: vertical;
  transition: border-color 0.3s ease;
}

.tester-form textarea:focus {
  outline: none;
  border-color: #667eea;
}

.performance-results {
  margin-top: 2rem;
}

.perf-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
}

.perf-card {
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
  color: white;
  padding: 1.5rem;
  border-radius: 12px;
  text-align: center;
  box-shadow: 0 4px 6px rgba(240, 147, 251, 0.3);
}

.perf-card h3 {
  font-size: 0.9rem;
  margin-bottom: 0.75rem;
  opacity: 0.95;
}

.perf-value {
  font-size: 2rem;
  font-weight: bold;
}

.variance-analysis {
  background: #f8f9fa;
  padding: 1.5rem;
  border-radius: 8px;
  margin-bottom: 2rem;
}

.variance-indicator {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  padding: 1rem;
  border-radius: 8px;
  font-size: 1.5rem;
  font-weight: bold;
  margin-top: 1rem;
}

.variance-indicator.good {
  background: #e8f5e9;
  color: #2e7d32;
}

.variance-indicator.warning {
  background: #fff3e0;
  color: #f57c00;
}

.variance-label {
  font-size: 1rem;
  font-weight: normal;
}

.token-analysis {
  background: #f8f9fa;
  padding: 1.5rem;
  border-radius: 8px;
}

.token-analysis p {
  margin-bottom: 0.75rem;
  color: #666;
}

/* Context Analyzer Styles */
.context-analyzer {
  background: white;
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
}

.analyzer-form {
  background: #f8f9fa;
  padding: 2rem;
  border-radius: 8px;
  margin-bottom: 2rem;
}

.context-results {
  margin-top: 2rem;
}

.context-summary {
  background: #f8f9fa;
  padding: 1.5rem;
  border-radius: 8px;
  margin-bottom: 2rem;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1.5rem;
  margin-top: 1rem;
}

.summary-item {
  display: flex;
  justify-content: space-between;
  padding: 1rem;
  background: white;
  border-radius: 6px;
}

.summary-item.highlight {
  background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
  font-weight: 600;
}

.summary-item .label {
  color: #666;
}

.summary-item .value {
  font-weight: bold;
  color: #333;
}

.distribution-chart,
.efficiency-metrics {
  background: #f8f9fa;
  padding: 1.5rem;
  border-radius: 8px;
  margin-bottom: 2rem;
}

.efficiency-metrics table {
  width: 100%;
  margin-top: 1rem;
}

.efficiency-metrics td {
  padding: 1rem;
  border-bottom: 1px solid #e0e0e0;
}

.efficiency-metrics tr:last-child td {
  border-bottom: none;
}

.recommendations {
  background: #e8f5e9;
  padding: 1.5rem;
  border-radius: 8px;
}

.recommendations ul {
  list-style: none;
  margin-top: 1rem;
}

.recommendations li {
  padding: 0.75rem;
  margin-bottom: 0.5rem;
  background: white;
  border-radius: 6px;
  border-left: 4px solid #4CAF50;
}

/* Footer */
.app-footer {
  background: #2c3e50;
  color: white;
  text-align: center;
  padding: 1.5rem;
  margin-top: auto;
}

/* Loading States */
.loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 400px;
  color: #666;
}

.loader {
  border: 4px solid #f3f3f3;
  border-top: 4px solid #667eea;
  border-radius: 50%;
  width: 50px;
  height: 50px;
  animation: spin 1s linear infinite;
  margin-bottom: 1rem;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

/* Responsive Design */
@media (max-width: 768px) {
  .header-stats {
    flex-direction: column;
    gap: 1rem;
  }

  .tab-nav {
    flex-direction: column;
  }

  .charts-container,
  .cost-visualization {
    grid-template-columns: 1fr;
  }

  .config-grid,
  .form-row {
    grid-template-columns: 1fr;
  }
}
EOF

# Frontend: index.html
cat > frontend/public/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="LLM Parameter Analysis Platform - L4: Understanding Model Parameters" />
    <title>L4: LLM Parameter Analysis Platform</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
EOF

# Frontend: index.js
cat > frontend/src/index.js << 'EOF'
import React from 'react';
import ReactDOM from 'react-dom/client';
import './App.css';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
EOF

# Build script
cat > build.sh << 'EOF'
#!/bin/bash
set -e

echo "📦 Building L4 Parameter Analysis Platform..."

# Backend setup
cd backend
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate
cd ..

# Frontend setup
cd frontend
npm install
cd ..

echo "✅ Build complete!"
echo "Run ./start.sh to launch the application"
EOF

chmod +x build.sh

# Start script
cat > start.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 Starting L4 Parameter Analysis Platform..."

# Start backend
cd backend
source venv/bin/activate
python main.py &
BACKEND_PID=$!
cd ..

# Wait for backend
echo "⏳ Waiting for backend..."
sleep 5

# Start frontend
cd frontend
BROWSER=none npm start &
FRONTEND_PID=$!
cd ..

echo "✅ Application started!"
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop all services"

# Save PIDs
echo $BACKEND_PID > .backend.pid
echo $FRONTEND_PID > .frontend.pid

# Wait for user interrupt
wait
EOF

chmod +x start.sh

# Stop script
cat > stop.sh << 'EOF'
#!/bin/bash

echo "🛑 Stopping L4 Parameter Analysis Platform..."

if [ -f .backend.pid ]; then
    kill $(cat .backend.pid) 2>/dev/null || true
    rm .backend.pid
fi

if [ -f .frontend.pid ]; then
    kill $(cat .frontend.pid) 2>/dev/null || true
    rm .frontend.pid
fi

# Kill any remaining processes
pkill -f "uvicorn" || true
pkill -f "react-scripts" || true

echo "✅ All services stopped"
EOF

chmod +x stop.sh

# Test script
cat > test.sh << 'EOF'
#!/bin/bash
set -e

echo "🧪 Testing L4 Parameter Analysis Platform..."

# Test backend
echo "Testing backend endpoints..."
curl -s http://localhost:8000/ | grep -q "L4" && echo "✅ Backend health check passed"

# Test model specs
curl -s http://localhost:8000/models | grep -q "gemini" && echo "✅ Model specs endpoint passed"

# Test cost calculation
curl -s -X POST http://localhost:8000/analyze/cost \
  -H "Content-Type: application/json" \
  -d '{"model_name":"gemini-2.0-flash-exp","requests_per_day":1000,"avg_input_tokens":500,"avg_output_tokens":200,"cache_hit_rate":0.5}' \
  | grep -q "monthly_cost" && echo "✅ Cost calculation passed"

# Test model comparison
curl -s -X POST "http://localhost:8000/compare?requests_per_day=1000&avg_input_tokens=500&avg_output_tokens=200&cache_hit_rate=0.5" \
  | grep -q "recommendation" && echo "✅ Model comparison passed"

echo "✅ All tests passed!"
EOF

chmod +x test.sh

# Documentation
cat > README.md << 'EOF'
# L4: Understanding LLM Parameters - Implementation

## Overview
Complete parameter analysis platform demonstrating the relationship between model parameters, context windows, training data scale, and operational costs.

## Quick Start
```bash
# Build
./build.sh

# Start
./start.sh

# Test
./test.sh

# Stop
./stop.sh
```

## Features

### 1. Model Comparison Matrix
- Side-by-side parameter analysis
- Cost projections across usage patterns
- Performance vs cost trade-offs
- Dynamic recommendations

### 2. Cost Calculator
- Real-time cost projections
- Cache optimization analysis
- Monthly/yearly forecasts
- Detailed breakdown by token type

### 3. Performance Tester
- Actual inference benchmarks
- Latency measurement (p50/p95/p99)
- Variance from specifications
- Token count analysis

### 4. Context Window Analyzer
- Token distribution analysis
- Efficiency recommendations
- Cost savings identification
- Optimal context limit suggestions

## Architecture
```
Backend (FastAPI)
├── Model Specifications Database
├── Cost Calculation Engine
├── Performance Testing Framework
└── Context Analysis Tools

Frontend (React)
├── Interactive Dashboards
├── Real-time Charts (Recharts)
├── Cost Projections
└── Performance Metrics
```

## API Endpoints

- `GET /models` - Retrieve all model specs
- `POST /analyze/cost` - Calculate costs
- `POST /analyze/performance` - Run benchmarks
- `POST /analyze/context` - Analyze context usage
- `POST /compare` - Compare all models

## Integration with Curriculum

**Builds on L3**: Uses Transformer architecture knowledge to explain parameter impact
**Prepares for L5**: Creates analytical foundation for model selection

## Access
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
EOF

echo "✅ L4: Understanding LLM Parameters - Setup Complete!"
echo ""
echo "📁 Project created: $PROJECT_NAME/"
echo ""
echo "Next steps:"
echo "1. cd $PROJECT_NAME"
echo "2. ./build.sh"
echo "3. ./start.sh"
echo "4. Open http://localhost:3000"
echo ""
echo "🎓 L4 demonstrates parameter analysis for production VAIA deployments"