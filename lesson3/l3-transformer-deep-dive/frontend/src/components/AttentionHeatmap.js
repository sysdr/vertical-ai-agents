import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import './AttentionHeatmap.css';

const AttentionHeatmap = ({ attentionWeights, tokens }) => {
  const heatmapData = useMemo(() => {
    if (!attentionWeights || !attentionWeights.weights) {
      return null;
    }

    const weights = attentionWeights.weights;
    const numHeads = weights.length;

    return Array.from({ length: numHeads }, (_, headIdx) => ({
      z: weights[headIdx],
      type: 'heatmap',
      colorscale: 'Viridis',
      showscale: headIdx === 0,
      hovertemplate: 
        'From: %{x}<br>' +
        'To: %{y}<br>' +
        'Weight: %{z:.4f}<br>' +
        '<extra></extra>',
      x: tokens,
      y: tokens
    }));
  }, [attentionWeights, tokens]);

  if (!heatmapData) {
    return <div className="no-data">No attention data available</div>;
  }

  const numHeads = heatmapData.length;
  const cols = Math.ceil(Math.sqrt(numHeads));
  const rows = Math.ceil(numHeads / cols);

  return (
    <div className="heatmap-container">
      <div className="heatmap-grid">
        {heatmapData.map((data, headIdx) => (
          <div key={headIdx} className="heatmap-cell">
            <h4>Head {headIdx + 1}</h4>
            <Plot
              data={[data]}
              layout={{
                width: 300,
                height: 300,
                margin: { l: 80, r: 20, t: 20, b: 80 },
                xaxis: {
                  tickangle: -45,
                  tickfont: { size: 9 }
                },
                yaxis: {
                  tickfont: { size: 9 }
                }
              }}
              config={{ displayModeBar: false }}
            />
          </div>
        ))}
      </div>
    </div>
  );
};

export default AttentionHeatmap;
