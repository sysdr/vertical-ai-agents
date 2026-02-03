import React, { useState } from 'react';

function PlanningForm({ onSubmit, loading }) {
  const [task, setTask] = useState('');
  const [maxIterations, setMaxIterations] = useState(10);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (task.trim()) {
      onSubmit(task, maxIterations);
    }
  };

  return (
    <div className="planning-form">
      <h2>Submit Planning Task</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>Task Description:</label>
          <textarea
            value={task}
            onChange={(e) => setTask(e.target.value)}
            placeholder="E.g., Analyze why the login service is failing and propose solutions"
            rows="4"
            disabled={loading}
          />
        </div>
        
        <div className="form-group">
          <label>Max Iterations:</label>
          <input
            type="number"
            value={maxIterations}
            onChange={(e) => setMaxIterations(parseInt(e.target.value))}
            min="1"
            max="20"
            disabled={loading}
          />
        </div>
        
        <button type="submit" disabled={loading || !task.trim()}>
          {loading ? 'Planning...' : 'Execute Planning'}
        </button>
      </form>
    </div>
  );
}

export default PlanningForm;
