/**
 * Baseline Comparison Visualization component.
 */

import type { BaselineMetric } from '../../types/anomaly';

interface BaselineComparisonVizProps {
  metrics: BaselineMetric[];
}

export function BaselineComparisonViz({ metrics }: BaselineComparisonVizProps) {
  if (!metrics || metrics.length === 0) {
    return (
      <div className="h-48 flex items-center justify-center text-slate-500">
        No baseline comparison data available
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {metrics.filter(m => m.is_anomalous).map((metric) => (
        <div key={metric.metric_name} className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/50">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-white">{metric.metric_display_name}</span>
            <span className={`text-xs px-2 py-0.5 rounded ${
              Math.abs(metric.deviation_sigma) >= 3 ? 'bg-red-500/20 text-red-400' :
              Math.abs(metric.deviation_sigma) >= 2 ? 'bg-orange-500/20 text-orange-400' :
              'bg-amber-500/20 text-amber-400'
            }`}>
              {metric.deviation_sigma > 0 ? '+' : ''}{metric.deviation_sigma.toFixed(1)}σ
            </span>
          </div>

          {/* Visual bar showing position relative to baseline */}
          <div className="relative h-8 bg-slate-700/50 rounded-lg overflow-hidden mb-2">
            {/* Baseline range (mean ± 2σ) */}
            <div
              className="absolute h-full bg-emerald-500/20"
              style={{
                left: `${Math.max(0, 50 - 25)}%`,
                width: '50%',
              }}
            />
            {/* Mean marker */}
            <div className="absolute h-full w-0.5 bg-emerald-500/50" style={{ left: '50%' }} />
            {/* Current value marker */}
            <div
              className="absolute h-full w-1.5 bg-amber-500 rounded"
              style={{
                left: `${Math.min(95, Math.max(5, metric.percentile_rank))}%`,
                transform: 'translateX(-50%)',
              }}
            />
          </div>

          <div className="flex justify-between text-xs text-slate-500">
            <span>Current: <span className="text-amber-400 font-mono">{metric.current_value_display}</span></span>
            <span>Baseline: <span className="text-emerald-400 font-mono">{metric.baseline_mean.toFixed(1)} {metric.metric_unit}</span></span>
          </div>
        </div>
      ))}
    </div>
  );
}
