/**
 * Feature Contribution Bar Chart component.
 */

import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  Cell, CartesianGrid,
} from 'recharts';
import type { FeatureContribution } from '../../types/anomaly';

interface FeatureContributionChartProps {
  contributions: FeatureContribution[];
}

export function FeatureContributionChart({ contributions }: FeatureContributionChartProps) {
  if (!contributions || contributions.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-slate-500">
        No feature contribution data available
      </div>
    );
  }

  const data = contributions.slice(0, 8).map((c) => ({
    name: c.feature_display_name,
    contribution: c.contribution_percentage,
    sigma: Math.abs(c.deviation_sigma),
    current: c.current_value_display,
    baseline: c.baseline_value_display,
  }));

  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 100, right: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
          <XAxis type="number" domain={[0, 'dataMax']} stroke="#64748b" fontSize={10} />
          <YAxis type="category" dataKey="name" stroke="#64748b" fontSize={11} width={95} />
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(14, 17, 23, 0.95)',
              border: '1px solid rgba(245, 158, 11, 0.3)',
              borderRadius: '8px',
              fontSize: '11px',
            }}
            formatter={(value: number) => [`${value.toFixed(1)}% contribution`]}
          />
          <Bar dataKey="contribution" radius={[0, 4, 4, 0]}>
            {data.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.sigma >= 3 ? '#ef4444' : entry.sigma >= 2 ? '#f97316' : '#f59e0b'}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
