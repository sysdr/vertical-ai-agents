import React from 'react';

function MetricsDashboard({ metrics, events }) {
  return (
    <div>
      <h3 style={styles.title}>📈 System Metrics</h3>
      
      <div style={styles.grid}>
        <div style={styles.card}>
          <div style={styles.cardTitle}>Total Retrievals</div>
          <div style={styles.cardValue}>{metrics.total_retrievals}</div>
        </div>
        
        <div style={styles.card}>
          <div style={styles.cardTitle}>Avg Execution Time</div>
          <div style={styles.cardValue}>{metrics.avg_execution_time_ms}ms</div>
        </div>
        
        <div style={styles.card}>
          <div style={styles.cardTitle}>Reranking Usage</div>
          <div style={styles.cardValue}>{metrics.reranking_usage_pct}%</div>
        </div>
        
        <div style={styles.card}>
          <div style={styles.cardTitle}>Hybrid Search Usage</div>
          <div style={styles.cardValue}>{metrics.hybrid_usage_pct}%</div>
        </div>
      </div>

      <div style={styles.eventsSection}>
        <h4 style={styles.eventsTitle}>🔔 Real-time Events</h4>
        <div style={styles.events}>
          {events.slice(0, 10).map((event, i) => (
            <div key={i} style={styles.event}>
              <span style={styles.eventType}>{event.type}</span>
              <span style={styles.eventTime}>
                {new Date(event.timestamp).toLocaleTimeString()}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

const styles = {
  title: {
    fontSize: '1.3rem',
    marginBottom: 20
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: 15,
    marginBottom: 25
  },
  card: {
    padding: 20,
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    borderRadius: 10,
    color: 'white',
    textAlign: 'center'
  },
  cardTitle: {
    fontSize: '0.9rem',
    opacity: 0.9,
    marginBottom: 10
  },
  cardValue: {
    fontSize: '2rem',
    fontWeight: 'bold'
  },
  eventsSection: {
    marginTop: 20
  },
  eventsTitle: {
    fontSize: '1.1rem',
    marginBottom: 12
  },
  events: {
    maxHeight: 200,
    overflowY: 'auto',
    background: '#f8f9fa',
    borderRadius: 8,
    padding: 10
  },
  event: {
    display: 'flex',
    justifyContent: 'space-between',
    padding: 8,
    marginBottom: 5,
    background: 'white',
    borderRadius: 5,
    fontSize: '0.85rem'
  },
  eventType: {
    fontWeight: 600,
    color: '#667eea'
  },
  eventTime: {
    color: '#888'
  }
};

export default MetricsDashboard;
