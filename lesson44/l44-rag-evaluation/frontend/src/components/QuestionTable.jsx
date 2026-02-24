import React, { useState } from 'react';

const fmt = (v) => v != null ? (v * 100).toFixed(1) + '%' : '—';
const color = (v, t) => {
  if (v == null) return '#64748b';
  return v >= t ? '#16a34a' : '#dc2626';
};

export default function QuestionTable({ items }) {
  const [expanded, setExpanded] = useState(null);
  if (!items || items.length === 0) return (
    <div style={{ color: '#94a3b8', padding: 20, textAlign: 'center', fontSize: 13 }}>
      No evaluation items yet
    </div>
  );

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr style={{ background: '#f8fafc' }}>
            <th style={{ padding: '10px 12px', textAlign: 'left', color: '#475569', fontWeight: 700, borderBottom: '2px solid #e2e8f0' }}>#</th>
            <th style={{ padding: '10px 12px', textAlign: 'left', color: '#475569', fontWeight: 700, borderBottom: '2px solid #e2e8f0', minWidth: 220 }}>Question</th>
            <th style={{ padding: '10px 12px', textAlign: 'center', color: '#1d4ed8', fontWeight: 700, borderBottom: '2px solid #e2e8f0' }}>Faith.</th>
            <th style={{ padding: '10px 12px', textAlign: 'center', color: '#9a3412', fontWeight: 700, borderBottom: '2px solid #e2e8f0' }}>Relev.</th>
            <th style={{ padding: '10px 12px', textAlign: 'center', color: '#15803d', fontWeight: 700, borderBottom: '2px solid #e2e8f0' }}>Recall</th>
            <th style={{ padding: '10px 12px', textAlign: 'center', color: '#5b21b6', fontWeight: 700, borderBottom: '2px solid #e2e8f0' }}>Prec.</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item, i) => (
            <React.Fragment key={i}>
              <tr
                onClick={() => setExpanded(expanded === i ? null : i)}
                style={{
                  background: expanded === i ? '#f0f9ff' : i % 2 === 0 ? 'white' : '#fafafa',
                  cursor: 'pointer', transition: 'background 0.15s',
                  borderBottom: '1px solid #f1f5f9',
                }}
              >
                <td style={{ padding: '10px 12px', color: '#94a3b8' }}>{i + 1}</td>
                <td style={{ padding: '10px 12px', color: '#1e293b', maxWidth: 260 }}>
                  {item.trace?.question?.slice(0, 70)}{item.trace?.question?.length > 70 ? '…' : ''}
                </td>
                <td style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 700, fontFamily: "'IBM Plex Mono', monospace", color: color(item.scores?.faithfulness, 0.85) }}>
                  {fmt(item.scores?.faithfulness)}
                </td>
                <td style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 700, fontFamily: "'IBM Plex Mono', monospace", color: color(item.scores?.answer_relevancy, 0.80) }}>
                  {fmt(item.scores?.answer_relevancy)}
                </td>
                <td style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 700, fontFamily: "'IBM Plex Mono', monospace", color: color(item.scores?.context_recall, 0.75) }}>
                  {fmt(item.scores?.context_recall)}
                </td>
                <td style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 700, fontFamily: "'IBM Plex Mono', monospace", color: color(item.scores?.context_precision, 0.70) }}>
                  {fmt(item.scores?.context_precision)}
                </td>
              </tr>
              {expanded === i && (
                <tr style={{ background: '#f0f9ff' }}>
                  <td colSpan={6} style={{ padding: '12px 16px' }}>
                    <div style={{ fontSize: 11, color: '#475569', lineHeight: 1.6 }}>
                      <strong>Answer:</strong> {item.trace?.answer?.slice(0, 300)}
                    </div>
                  </td>
                </tr>
              )}
            </React.Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
}
