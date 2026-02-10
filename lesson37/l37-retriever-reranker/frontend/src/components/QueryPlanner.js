import React from 'react';

function QueryPlanner({ subQueries }) {
  return (
    <div style={styles.container}>
      <h3 style={styles.title}>📋 Query Decomposition (from PlannerAgent L36)</h3>
      <div style={styles.queries}>
        {subQueries.map((sq, i) => (
          <div key={i} style={styles.query}>
            <span style={styles.badge}>{i + 1}</span>
            {sq}
          </div>
        ))}
      </div>
    </div>
  );
}

const styles = {
  container: {
    marginTop: 20,
    padding: 15,
    background: '#f8f9fa',
    borderRadius: 8,
    border: '1px solid #e0e0e0'
  },
  title: {
    fontSize: '1rem',
    marginBottom: 12,
    color: '#333'
  },
  queries: {
    display: 'flex',
    flexDirection: 'column',
    gap: 10
  },
  query: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    padding: 10,
    background: 'white',
    borderRadius: 6,
    fontSize: '0.95rem'
  },
  badge: {
    display: 'inline-block',
    width: 24,
    height: 24,
    background: '#667eea',
    color: 'white',
    borderRadius: '50%',
    textAlign: 'center',
    lineHeight: '24px',
    fontSize: '0.85rem',
    fontWeight: 'bold'
  }
};

export default QueryPlanner;
