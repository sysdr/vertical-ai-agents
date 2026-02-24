import React from 'react';

const thresholds = {
  faithfulness: 0.85,
  answer_relevancy: 0.80,
  context_recall: 0.75,
  context_precision: 0.70,
};

const labels = {
  faithfulness: 'Faithfulness',
  answer_relevancy: 'Answer Relevancy',
  context_recall: 'Context Recall',
  context_precision: 'Context Precision',
};

const icons = {
  faithfulness: '🎯',
  answer_relevancy: '📌',
  context_recall: '🔁',
  context_precision: '🎚️',
};

export default function MetricCard({ metric, score }) {
  const threshold = thresholds[metric] || 0.75;
  const pct = score != null ? Math.round(score * 100) : null;
  const pass = pct != null && score >= threshold;
  const color = pass ? '#16a34a' : '#dc2626';
  const bg = pass ? '#f0fdf4' : '#fff1f2';
  const border = pass ? '#86efac' : '#fca5a5';
  const barColor = pass ? '#22c55e' : '#ef4444';

  return (
    <div style={{
      background: bg, border: `1.5px solid ${border}`,
      borderRadius: 14, padding: '18px 20px',
      boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
      minWidth: 180,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <span style={{ fontSize: 22 }}>{icons[metric]}</span>
        <span style={{
          fontSize: 11, fontWeight: 700, color: pass ? '#15803d' : '#b91c1c',
          background: pass ? '#dcfce7' : '#fee2e2', padding: '2px 8px', borderRadius: 20,
        }}>
          {pass ? '✓ PASS' : '✗ FAIL'}
        </span>
      </div>
      <div style={{ fontSize: 32, fontWeight: 700, color, fontFamily: "'IBM Plex Mono', monospace", lineHeight: 1.1 }}>
        {pct != null ? `${pct}%` : '—'}
      </div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 4, fontWeight: 600 }}>
        {labels[metric]}
      </div>
      <div style={{ marginTop: 10, background: '#e2e8f0', borderRadius: 4, height: 5, overflow: 'hidden' }}>
        <div style={{ width: `${pct || 0}%`, height: '100%', background: barColor, transition: 'width 0.8s ease', borderRadius: 4 }}/>
      </div>
      <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 4 }}>
        threshold: {Math.round(threshold * 100)}%
      </div>
    </div>
  );
}
