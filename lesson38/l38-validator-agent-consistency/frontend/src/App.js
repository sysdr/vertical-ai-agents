import React, { useState } from 'react';
import ValidatorDashboard from './components/ValidatorDashboard';
import './App.css';

function App() {
  const [apiUrl] = useState(process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1');

  return (
    <div className="App">
      <header className="app-header">
        <h1>🔍 L38: ValidatorAgent - Factual Consistency</h1>
        <p>Cross-reference documents for contradictions using Gemini AI</p>
      </header>
      
      <main className="app-main">
        <ValidatorDashboard apiUrl={apiUrl} />
      </main>
      
      <footer className="app-footer">
        <p>Module 5: Advanced RAG Orchestration | Lesson 38 of 90</p>
        <p>Validates consistency before generation</p>
      </footer>
    </div>
  );
}

export default App;
