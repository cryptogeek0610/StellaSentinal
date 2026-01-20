/**
 * Auto-discovered correlation insights view component.
 */

import clsx from 'clsx';
import { Card } from '../Card';
import { ErrorState } from './ErrorState';
import { getCorrelationColor, formatCorrelation, getStrengthColor } from './correlationUtils';
import type { CorrelationInsightsResponse } from '../../types/correlations';

interface InsightsViewProps {
  data?: CorrelationInsightsResponse;
  isLoading: boolean;
  error?: unknown;
  onRetry: () => void;
}

export function InsightsView({ data, isLoading, error, onRetry }: InsightsViewProps) {
  if (isLoading) {
    return (
      <Card title="Correlation Insights">
        <div className="h-96 flex items-center justify-center text-slate-400">
          Loading insights...
        </div>
      </Card>
    );
  }

  if (error) {
    return <ErrorState title="Correlation Insights" error={error} onRetry={onRetry} />;
  }

  if (!data?.insights || data.insights.length === 0) {
    return (
      <Card title="Correlation Insights">
        <div className="h-96 flex items-center justify-center text-slate-400">
          No insights available
        </div>
      </Card>
    );
  }

  const getDirectionIcon = (direction: string) => {
    if (direction === 'positive') {
      return (
        <svg className="w-5 h-5 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
        </svg>
      );
    }
    return (
      <svg className="w-5 h-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
      </svg>
    );
  };

  return (
    <Card title={<>Auto-Discovered Insights <span className="text-slate-500 text-sm font-normal ml-2">{data.total_correlations_analyzed} correlations analyzed</span></>}>
      <div className="space-y-4">
        {data.insights.map((insight) => (
          <div
            key={insight.insight_id}
            className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/30 hover:border-slate-600 transition-all"
          >
            <div className="flex items-start justify-between gap-4">
              <div className="flex items-start gap-3">
                {getDirectionIcon(insight.direction)}
                <div>
                  <h4 className="text-white font-medium">{insight.headline}</h4>
                  <p className="text-sm text-slate-400 mt-1">{insight.description}</p>
                  {insight.recommendation && (
                    <p className="text-sm text-blue-400 mt-2">
                      <span className="font-medium">Recommendation:</span> {insight.recommendation}
                    </p>
                  )}
                  <div className="flex flex-wrap gap-2 mt-3">
                    {insight.metrics_involved.map((metric) => (
                      <span
                        key={metric}
                        className="px-2 py-0.5 text-xs rounded-full bg-slate-700/50 text-slate-300"
                      >
                        {metric}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
              <div className="flex flex-col items-end gap-2 shrink-0">
                <span
                  className={clsx(
                    'px-2 py-1 text-xs font-medium rounded-full border',
                    getStrengthColor(insight.strength)
                  )}
                >
                  {insight.strength}
                </span>
                <span className="text-lg font-bold font-mono" style={{ color: getCorrelationColor(insight.correlation_value) }}>
                  r={formatCorrelation(insight.correlation_value)}
                </span>
                <span className="text-xs text-slate-500">
                  {(insight.confidence * 100).toFixed(0)}% confidence
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}
