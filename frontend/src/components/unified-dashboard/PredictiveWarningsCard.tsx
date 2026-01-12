/**
 * Predictive Warnings Card
 *
 * Uses time-lagged correlations to show predictions.
 * Shows insights like "Based on current patterns, 12 devices likely to experience battery issues in 2-3 days"
 */

import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { api } from '../../api/client';
import clsx from 'clsx';

// Get color based on correlation strength
const getCorrelationColor = (value: number): string => {
  const absValue = Math.abs(value);
  if (absValue >= 0.7) return 'text-purple-400';
  if (absValue >= 0.5) return 'text-blue-400';
  return 'text-slate-400';
};

// Format the lag as human readable
const formatLag = (days: number): string => {
  if (days === 1) return 'tomorrow';
  if (days === 2) return 'in 2 days';
  if (days <= 7) return `in ${days} days`;
  return `in ${days} days`;
};

interface PredictiveWarningsCardProps {
  maxPredictions?: number;
}

export function PredictiveWarningsCard({ maxPredictions = 4 }: PredictiveWarningsCardProps) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['correlations', 'time-lagged'],
    queryFn: () => api.getTimeLaggedCorrelations(),
    refetchInterval: 300000, // Refresh every 5 minutes
  });

  if (isLoading) {
    return (
      <div className="stellar-glass rounded-xl p-4 border border-slate-700/50">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium text-slate-300">Predictive Insights</h3>
        </div>
        <div className="space-y-2">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-12 rounded-lg bg-slate-800/50 animate-pulse" />
          ))}
        </div>
      </div>
    );
  }

  if (error || !data?.correlations || data.correlations.length === 0) {
    return (
      <div className="stellar-glass rounded-xl p-4 border border-slate-700/50">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium text-slate-300">Predictive Insights</h3>
        </div>
        <div className="text-center py-6">
          <div className="w-10 h-10 rounded-full bg-purple-500/10 flex items-center justify-center mx-auto mb-2">
            <svg className="w-5 h-5 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <p className="text-sm text-slate-400">No predictions available</p>
          <p className="text-xs text-slate-500 mt-1">Analyzing time-lagged patterns...</p>
        </div>
      </div>
    );
  }

  const predictions = data.correlations.slice(0, maxPredictions);

  return (
    <div className="stellar-glass rounded-xl p-4 border border-slate-700/50">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-medium text-slate-300">Predictive Insights</h3>
          <span className="text-xs px-1.5 py-0.5 rounded-full bg-purple-500/20 text-purple-400 border border-purple-500/30">
            {data.max_lag_analyzed}d horizon
          </span>
        </div>
        <Link
          to="/insights?tab=correlations"
          className="text-xs text-slate-500 hover:text-amber-400 transition-colors flex items-center gap-1"
        >
          <span>More</span>
          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </Link>
      </div>

      <div className="space-y-2">
        {predictions.map((pred, idx) => (
          <div
            key={idx}
            className="p-3 rounded-lg bg-slate-800/30 border border-slate-700/30 hover:border-purple-500/30 transition-all"
          >
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center shrink-0">
                <span className="text-xs font-bold text-purple-400">D{pred.lag_days}</span>
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5 text-xs">
                  <span className="text-slate-400">{pred.metric_a}</span>
                  <svg className="w-3 h-3 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                  <span className="text-white">{pred.metric_b}</span>
                  <span className="text-slate-500">({formatLag(pred.lag_days)})</span>
                </div>
                <p className="text-xs text-slate-500 mt-1 line-clamp-1">{pred.insight}</p>
              </div>
              <div className="text-right shrink-0">
                <span className={clsx('text-sm font-mono font-bold', getCorrelationColor(pred.correlation))}>
                  r={pred.correlation.toFixed(2)}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-3 p-3 rounded-lg bg-purple-500/10 border border-purple-500/20">
        <div className="flex items-start gap-2">
          <svg className="w-4 h-4 text-purple-400 mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <p className="text-xs text-slate-300">
            <span className="text-purple-400 font-medium">Time-lagged correlations</span> predict how metrics today affect outcomes in future days.
          </p>
        </div>
      </div>
    </div>
  );
}

export default PredictiveWarningsCard;
