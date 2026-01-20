/**
 * Cohort correlation patterns view component.
 */

import clsx from 'clsx';
import { Card } from '../Card';
import { ErrorState } from './ErrorState';
import { getCorrelationColor, formatCorrelation } from './correlationUtils';
import type { CohortCorrelationPatternsResponse } from '../../types/correlations';

interface CohortPatternsViewProps {
  data?: CohortCorrelationPatternsResponse;
  isLoading: boolean;
  error?: unknown;
  onRetry: () => void;
}

export function CohortPatternsView({ data, isLoading, error, onRetry }: CohortPatternsViewProps) {
  if (isLoading) {
    return (
      <Card title="Cohort Correlation Patterns">
        <div className="h-96 flex items-center justify-center text-slate-400">
          Loading cohort patterns...
        </div>
      </Card>
    );
  }

  if (error) {
    return <ErrorState title="Cohort Correlation Patterns" error={error} onRetry={onRetry} />;
  }

  if (!data?.patterns || data.patterns.length === 0) {
    return (
      <Card title="Cohort Correlation Patterns">
        <div className="h-96 flex items-center justify-center text-slate-400">
          No cohort patterns available
        </div>
      </Card>
    );
  }

  return (
    <Card title={<>Cohort Correlation Patterns <span className="text-slate-500 text-sm font-normal ml-2">{data.anomalous_cohorts} anomalous cohorts found</span></>}>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-slate-500 border-b border-slate-700/50">
              <th className="pb-3 font-medium">Cohort</th>
              <th className="pb-3 font-medium">Metric Pair</th>
              <th className="pb-3 font-medium text-right">Cohort r</th>
              <th className="pb-3 font-medium text-right">Fleet r</th>
              <th className="pb-3 font-medium text-right">Deviation</th>
              <th className="pb-3 font-medium text-right">Devices</th>
              <th className="pb-3 font-medium">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-700/30">
            {data.patterns.map((pattern) => (
              <tr key={pattern.cohort_id} className="hover:bg-slate-800/30 transition-colors">
                <td className="py-3">
                  <span className="font-medium text-white">{pattern.cohort_name}</span>
                </td>
                <td className="py-3">
                  <span className="text-slate-400">
                    {pattern.metric_pair[0]} <span className="text-slate-600">↔</span> {pattern.metric_pair[1]}
                  </span>
                </td>
                <td className="py-3 text-right">
                  <span className="font-mono" style={{ color: getCorrelationColor(pattern.cohort_correlation) }}>
                    {formatCorrelation(pattern.cohort_correlation)}
                  </span>
                </td>
                <td className="py-3 text-right">
                  <span className="font-mono text-slate-400">
                    {formatCorrelation(pattern.fleet_correlation)}
                  </span>
                </td>
                <td className="py-3 text-right">
                  <span
                    className={clsx(
                      'font-mono font-medium',
                      Math.abs(pattern.deviation) > 0.3 ? 'text-red-400' : 'text-slate-400'
                    )}
                  >
                    {pattern.deviation > 0 ? '+' : ''}{formatCorrelation(pattern.deviation)}
                  </span>
                </td>
                <td className="py-3 text-right text-slate-400">
                  {pattern.device_count.toLocaleString()}
                </td>
                <td className="py-3">
                  {pattern.is_anomalous ? (
                    <span className="px-2 py-0.5 text-xs font-medium rounded-full bg-red-500/20 text-red-400 border border-red-500/30">
                      Anomalous
                    </span>
                  ) : (
                    <span className="px-2 py-0.5 text-xs font-medium rounded-full bg-slate-500/20 text-slate-400 border border-slate-500/30">
                      Normal
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Insight cards for anomalous patterns */}
      {data.patterns.filter(p => p.is_anomalous && p.insight).length > 0 && (
        <div className="mt-6 space-y-3">
          <h4 className="text-sm font-medium text-slate-400">Anomalous Pattern Insights</h4>
          {data.patterns.filter(p => p.is_anomalous && p.insight).map((pattern) => (
            <div
              key={pattern.cohort_id + '-insight'}
              className="p-3 rounded-lg bg-red-500/10 border border-red-500/20"
            >
              <div className="flex items-start gap-2">
                <svg className="w-4 h-4 text-red-400 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                <div>
                  <span className="text-sm font-medium text-red-400">{pattern.cohort_name}</span>
                  <p className="text-sm text-slate-300 mt-1">{pattern.insight}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}
