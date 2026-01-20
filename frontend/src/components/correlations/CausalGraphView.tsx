/**
 * Causal relationship graph visualization component.
 */

import { Card } from '../Card';
import { ErrorState } from './ErrorState';
import { getDomainColor } from './correlationUtils';
import type { CausalGraphResponse, CausalNode } from '../../types/correlations';

interface CausalGraphViewProps {
  data?: CausalGraphResponse;
  isLoading: boolean;
  error?: unknown;
  onRetry: () => void;
}

export function CausalGraphView({ data, isLoading, error, onRetry }: CausalGraphViewProps) {
  if (isLoading) {
    return (
      <Card title="Causal Relationship Graph">
        <div className="h-96 flex items-center justify-center text-slate-400">
          Loading causal graph...
        </div>
      </Card>
    );
  }

  if (error) {
    return <ErrorState title="Causal Relationship Graph" error={error} onRetry={onRetry} />;
  }

  if (!data?.nodes || !data?.edges) {
    return (
      <Card title="Causal Relationship Graph">
        <div className="h-96 flex items-center justify-center text-slate-400">
          No causal data available
        </div>
      </Card>
    );
  }

  const { nodes, edges } = data;

  // Position nodes: causes on left, effects on right, both in center
  const getNodePosition = (node: CausalNode): { x: number; y: number } => {
    const padding = 60;
    const width = 800;
    const height = 500;

    if (node.is_cause && !node.is_effect) {
      // Pure causes on the left
      const causesOnly = nodes.filter(n => n.is_cause && !n.is_effect);
      const idx = causesOnly.indexOf(node);
      return {
        x: padding,
        y: padding + (idx / Math.max(causesOnly.length - 1, 1)) * (height - 2 * padding),
      };
    } else if (!node.is_cause && node.is_effect) {
      // Pure effects on the right
      const effectsOnly = nodes.filter(n => !n.is_cause && n.is_effect);
      const idx = effectsOnly.indexOf(node);
      return {
        x: width - padding,
        y: padding + (idx / Math.max(effectsOnly.length - 1, 1)) * (height - 2 * padding),
      };
    } else {
      // Both cause and effect in center
      const both = nodes.filter(n => n.is_cause && n.is_effect);
      const idx = both.indexOf(node);
      return {
        x: width / 2,
        y: padding + (idx / Math.max(both.length - 1, 1)) * (height - 2 * padding),
      };
    }
  };

  // Create positioned nodes
  const positionedNodes = nodes.map((node) => ({
    ...node,
    position: getNodePosition(node),
  }));

  return (
    <div className="space-y-6">
      <Card title={<>Causal Relationship Graph <span className="text-slate-500 text-sm font-normal ml-2">{nodes.length} metrics, {edges.length} relationships</span></>}>
        <div className="overflow-x-auto">
          <svg width={800} height={500} className="mx-auto">
            {/* Draw edges */}
            {edges.map((edge, idx) => {
              const sourceNode = positionedNodes.find(n => n.metric === edge.source);
              const targetNode = positionedNodes.find(n => n.metric === edge.target);
              if (!sourceNode || !targetNode) return null;

              const dx = targetNode.position.x - sourceNode.position.x;
              const dy = targetNode.position.y - sourceNode.position.y;
              const dist = Math.sqrt(dx * dx + dy * dy);
              const offsetX = (dx / dist) * 35;
              const offsetY = (dy / dist) * 35;

              return (
                <g key={`edge-${idx}`}>
                  <defs>
                    <marker
                      id={`arrow-${idx}`}
                      markerWidth="8"
                      markerHeight="8"
                      refX="8"
                      refY="4"
                      orient="auto"
                    >
                      <path
                        d="M0,0 L8,4 L0,8 Z"
                        fill={edge.relationship === 'causes' ? '#f59e0b' : '#64748b'}
                      />
                    </marker>
                  </defs>
                  <line
                    x1={sourceNode.position.x + offsetX}
                    y1={sourceNode.position.y + offsetY}
                    x2={targetNode.position.x - offsetX}
                    y2={targetNode.position.y - offsetY}
                    stroke={edge.relationship === 'causes' ? '#f59e0b' : '#64748b'}
                    strokeWidth={Math.max(1, edge.strength * 3)}
                    strokeOpacity={0.6}
                    markerEnd={`url(#arrow-${idx})`}
                  />
                </g>
              );
            })}

            {/* Draw nodes */}
            {positionedNodes.map((node) => (
              <g key={node.metric} transform={`translate(${node.position.x}, ${node.position.y})`}>
                <circle
                  r={30}
                  fill={getDomainColor(node.domain)}
                  fillOpacity={0.2}
                  stroke={getDomainColor(node.domain)}
                  strokeWidth={2}
                />
                <text
                  textAnchor="middle"
                  dy="4"
                  className="text-xs fill-white font-medium"
                  style={{ fontSize: '9px' }}
                >
                  {node.metric.length > 12 ? node.metric.substring(0, 10) + '..' : node.metric}
                </text>
              </g>
            ))}
          </svg>
        </div>

        {/* Domain Legend */}
        <div className="mt-4 flex flex-wrap items-center justify-center gap-4 text-xs text-slate-400">
          {['Battery', 'RF', 'Throughput', 'App', 'System', 'Network'].map((domain) => (
            <span key={domain} className="flex items-center gap-1">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: getDomainColor(domain) }} />
              {domain}
            </span>
          ))}
        </div>
      </Card>

      {/* Causal Edges List */}
      <Card title="Causal Relationships">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-h-96 overflow-y-auto">
          {edges.filter(e => e.relationship === 'causes').map((edge, idx) => (
            <div
              key={idx}
              className="flex items-center gap-2 p-2 rounded-lg bg-slate-800/30 border border-slate-700/30"
            >
              <span className="text-xs font-medium text-amber-400">{edge.source}</span>
              <svg className="w-4 h-4 text-amber-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
              <span className="text-xs font-medium text-white">{edge.target}</span>
              <span className="ml-auto text-xs text-slate-500">{(edge.strength * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
