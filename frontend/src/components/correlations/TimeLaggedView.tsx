/**
 * Time-lagged correlations view component.
 */

import { Card } from '../Card';
import { ErrorState } from './ErrorState';
import { getCorrelationColor, formatCorrelation } from './correlationUtils';
import type { TimeLagCorrelationsResponse } from '../../types/correlations';

interface TimeLaggedViewProps {
  data?: TimeLagCorrelationsResponse;
  isLoading: boolean;
  error?: unknown;
  onRetry: () => void;
}

export function TimeLaggedView({ data, isLoading, error, onRetry }: TimeLaggedViewProps) {
  if (isLoading) {
    return (
      <Card title="Time-Lagged Correlations">
        <div className="h-96 flex items-center justify-center text-slate-400">
          Loading time-lagged correlations...
        </div>
      </Card>
    );
  }

  if (error) {
    return <ErrorState title="Time-Lagged Correlations" error={error} onRetry={onRetry} />;
  }

  if (!data?.correlations || data.correlations.length === 0) {
    return (
      <Card title="Time-Lagged Correlations">
        <div className="h-96 flex items-center justify-center text-slate-400">
          No time-lagged correlations found
        </div>
      </Card>
    );
  }

  return (
    <Card title={<>Predictive Correlations <span className="text-slate-500 text-sm font-normal ml-2">Cross-time relationships (up to {data.max_lag_analyzed} day lag)</span></>}>
      <div className="space-y-4">
        {data.correlations.map((corr, idx) => (
          <div
            key={idx}
            className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/30"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-lg bg-purple-500/20 flex items-center justify-center">
                  <span className="text-lg font-bold text-purple-400">D{corr.lag_days}</span>
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-white">{corr.metric_a}</span>
                    <svg className="w-5 h-5 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                    </svg>
                    <span className="font-medium text-white">{corr.metric_b}</span>
                  </div>
                  <p className="text-sm text-slate-400 mt-1">{corr.insight}</p>
                </div>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold font-mono" style={{ color: getCorrelationColor(corr.correlation) }}>
                  r={formatCorrelation(corr.correlation)}
                </div>
                <div className="text-xs text-slate-500 mt-1">
                  {corr.lag_days} day{corr.lag_days > 1 ? 's' : ''} lag
                </div>
                {corr.p_value !== null && (
                  <div className="text-xs text-slate-600">
                    p={corr.p_value.toFixed(4)}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}

        {/* Explanation */}
        <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
          <div className="flex items-start gap-2">
            <svg className="w-5 h-5 text-blue-400 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div>
              <span className="text-sm font-medium text-blue-400">What are time-lagged correlations?</span>
              <p className="text-sm text-slate-300 mt-1">
                These correlations show predictive relationships where changes in one metric today predict
                changes in another metric in the future. For example, a drop in signal strength today may
                predict increased battery drain tomorrow.
              </p>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
}
