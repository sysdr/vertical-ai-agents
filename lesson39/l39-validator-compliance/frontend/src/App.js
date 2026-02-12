import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import ComplianceStats from './components/ComplianceStats';
import ValidationForm from './components/ValidationForm';
import AuditTrail from './components/AuditTrail';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [stats, setStats] = useState(null);
  const [auditTrail, setAuditTrail] = useState([]);
  const [validationResult, setValidationResult] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let mounted = true;
    fetchStats();
    fetchAuditTrail();

    // WebSocket URL from API URL (e.g. http://localhost:8000 -> ws://localhost:8000/ws)
    const wsUrl = (API_URL.replace(/^http/, 'ws').replace(/\/$/, '') || 'ws://localhost:8000') + '/ws';
    let ws;
    try {
      ws = new WebSocket(wsUrl);
      ws.onmessage = (event) => {
        if (!mounted) return;
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'stats_update') setStats(data.data);
        } catch (_) {}
      };
      ws.onerror = () => { ws = null; };
      ws.onclose = () => { ws = null; };
    } catch (_) {
      ws = null;
    }

    return () => {
      mounted = false;
      // Only close if already open; avoids "closed before connection established" in React Strict Mode
      if (ws && ws.readyState === WebSocket.OPEN) ws.close();
    };
  }, []);

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API_URL}/stats`);
      setStats(response.data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const fetchAuditTrail = async () => {
    try {
      const response = await axios.get(`${API_URL}/audit-trail?limit=20`);
      setAuditTrail(response.data.audit_trail);
    } catch (error) {
      console.error('Error fetching audit trail:', error);
    }
  };

  const handleValidation = async (documents, query, domain) => {
    setLoading(true);
    setValidationResult(null);
    
    try {
      const response = await axios.post(`${API_URL}/validate`, {
        documents,
        query,
        domain,
        context: { source: 'dashboard' }
      });
      
      setValidationResult(response.data);
      
      // Refresh stats and audit trail
      setTimeout(() => {
        fetchStats();
        fetchAuditTrail();
      }, 500);
      
    } catch (error) {
      console.error('Validation error:', error);
      setValidationResult({
        approved: false,
        error: error.response?.data?.detail || error.message
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>🛡️ L39: Compliance Validation Dashboard</h1>
        <p>Risk & Compliance Enforcement for Enterprise AI</p>
      </header>

      <div className="dashboard-grid">
        <div className="panel stats-panel">
          <ComplianceStats stats={stats} />
        </div>

        <div className="panel validation-panel">
          <ValidationForm 
            onValidate={handleValidation}
            loading={loading}
            result={validationResult}
          />
        </div>

        <div className="panel audit-panel">
          <AuditTrail trail={auditTrail} />
        </div>
      </div>
    </div>
  );
}

export default App;
