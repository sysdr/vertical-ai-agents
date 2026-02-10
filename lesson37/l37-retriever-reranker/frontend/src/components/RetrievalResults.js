import React from 'react';

function RetrievalResults({ results }) {
  const { chunks, metrics } = results;

  return (
    <div>
      <div style={styles.header}>
        <h3>📊 Retrieved Chunks ({chunks.length})</h3>
        <div style={styles.stats}>
          <span>⏱️ {metrics.execution_time_ms.toFixed(0)}ms</span>
          <span>🔄 {metrics.sub_queries_count} sub-queries</span>
        </div>
      </div>

      {chunks.map((chunk, i) => (
        <div key={i} style={styles.chunk}>
          <div style={styles.chunkHeader}>
            <span style={styles.rank}>#{i + 1}</span>
            <span style={styles.chunkId}>{chunk.chunk_id}</span>
            <div style={styles.scores}>
              {chunk.vector_score !== undefined && (
                <span style={styles.score}>
                  Vector: {(chunk.vector_score * 100).toFixed(1)}
                </span>
              )}
              {chunk.rerank_score !== undefined && (
                <span style={{...styles.score, background: '#4caf50'}}>
                  Rerank: {chunk.rerank_score.toFixed(1)}
                </span>
              )}
              {chunk.rrf_score !== undefined && (
                <span style={{...styles.score, background: '#ff9800'}}>
                  RRF: {chunk.rrf_score.toFixed(3)}
                </span>
              )}
            </div>
          </div>
          <p style={styles.content}>{chunk.content}</p>
          {chunk.metadata && (
            <div style={styles.metadata}>
              Source: {chunk.metadata.source} | Topic: {chunk.metadata.topic}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

const styles = {
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
    paddingBottom: 15,
    borderBottom: '2px solid #e0e0e0'
  },
  stats: {
    display: 'flex',
    gap: 15,
    fontSize: '0.9rem',
    color: '#666'
  },
  chunk: {
    marginBottom: 15,
    padding: 15,
    background: '#f8f9fa',
    borderRadius: 8,
    border: '1px solid #e0e0e0'
  },
  chunkHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    marginBottom: 10,
    flexWrap: 'wrap'
  },
  rank: {
    display: 'inline-block',
    width: 32,
    height: 32,
    background: '#667eea',
    color: 'white',
    borderRadius: '50%',
    textAlign: 'center',
    lineHeight: '32px',
    fontWeight: 'bold',
    fontSize: '0.9rem'
  },
  chunkId: {
    fontSize: '0.85rem',
    color: '#666',
    fontFamily: 'monospace'
  },
  scores: {
    display: 'flex',
    gap: 8,
    marginLeft: 'auto'
  },
  score: {
    padding: '4px 10px',
    background: '#2196f3',
    color: 'white',
    borderRadius: 4,
    fontSize: '0.8rem',
    fontWeight: 600
  },
  content: {
    fontSize: '0.95rem',
    lineHeight: 1.6,
    color: '#333',
    marginBottom: 10
  },
  metadata: {
    fontSize: '0.8rem',
    color: '#888',
    fontStyle: 'italic'
  }
};

export default RetrievalResults;
