/**
 * Correlation Insights Card
 *
 * Displays auto-discovered correlation insights on the dashboard.
 * Shows insights like "Screen time strongly correlates with battery drain (r=0.78)"
 */

import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { api } from '../../api/client';
import clsx from 'clsx';

// Color scale for correlation values
const getCorrelationColor = (value: number): string => {
  if (value >= 0.7) return 'text-green-400';
  if (value >= 0.4) return 'text-lime-400';
  if (value >= 0.1) return 'text-yellow-400';
  if (value >= -0.1) return 'text-slate-400';
  if (value >= -0.4) return 'text-orange-400';
  if (value >= -0.7) return 'text-red-400';
  return 'text-red-500';
};

const getStrengthBadge = (strength: string) => {
  switch (strength) {
    case 'strong':
      return 'bg-green-500/20 text-green-400 border-green-500/30';
    case 'moderate':
      return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
    default:
      return 'bg-slate-500/20 text-slate-400 border-slate-500/30';
  }
};

const DirectionIcon = ({ direction }: { direction: string }) => {
  if (direction === 'positive') {
    return (
      <svg className="w-4 h-4 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
      </svg>
    );
  }
  return (
    <svg className="w-4 h-4 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
    </svg>
  );
};

interface CorrelationInsightsCardProps {
  maxInsights?: number;
}

export function CorrelationInsightsCard({ maxInsights = 3 }: CorrelationInsightsCardProps) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['correlations', 'insights'],
    queryFn: () => api.getCorrelationInsights(),
    refetchInterval: 300000, // Refresh every 5 minutes
  });

  if (isLoading) {
    return (
      <div className="stellar-glass rounded-xl p-4 border border-slate-700/50">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium text-slate-300">Correlation Insights</h3>
        </div>
        <div className="space-y-2">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-14 rounded-lg bg-slate-800/50 animate-pulse" />
          ))}
        </div>
      </div>
    );
  }

  if (error || !data?.insights || data.insights.length === 0) {
    return (
      <div className="stellar-glass rounded-xl p-4 border border-slate-700/50">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium text-slate-300">Correlation Insights</h3>
        </div>
        <div className="text-center py-6">
          <div className="w-10 h-10 rounded-full bg-blue-500/10 flex items-center justify-center mx-auto mb-2">
            <svg className="w-5 h-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
            </svg>
          </div>
          <p className="text-sm text-slate-400">No correlation insights yet</p>
          <p className="text-xs text-slate-500 mt-1">Analyzing metric relationships...</p>
        </div>
      </div>
    );
  }

  const insights = data.insights.slice(0, maxInsights);

  return (
    <div className="stellar-glass rounded-xl p-4 border border-slate-700/50">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-medium text-slate-300">Correlation Insights</h3>
          <span className="text-xs px-1.5 py-0.5 rounded-full bg-blue-500/20 text-blue-400 border border-blue-500/30">
            {data.total_correlations_analyzed} analyzed
          </span>
        </div>
        <Link
          to="/insights?tab=correlations"
          className="text-xs text-slate-500 hover:text-amber-400 transition-colors flex items-center gap-1"
        >
          <span>View Matrix</span>
          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </Link>
      </div>

      <div className="space-y-2">
        {insights.map((insight) => (
          <div
            key={insight.insight_id}
            className="p-3 rounded-lg bg-slate-800/30 border border-slate-700/30 hover:border-slate-600/50 transition-all"
          >
            <div className="flex items-start gap-2">
              <DirectionIcon direction={insight.direction} />
              <div className="flex-1 min-w-0">
                <p className="text-sm text-white font-medium line-clamp-1">{insight.headline}</p>
                <div className="flex items-center gap-2 mt-1.5">
                  <span className={clsx('text-sm font-mono font-bold', getCorrelationColor(insight.correlation_value))}>
                    r={insight.correlation_value.toFixed(2)}
                  </span>
                  <span className={clsx('text-xs px-1.5 py-0.5 rounded border', getStrengthBadge(insight.strength))}>
                    {insight.strength}
                  </span>
                  <span className="text-xs text-slate-500">
                    {(insight.confidence * 100).toFixed(0)}% conf.
                  </span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-3 pt-3 border-t border-slate-700/30">
        <Link
          to="/insights?tab=correlations"
          className="flex items-center justify-center gap-2 w-full py-2 rounded-lg bg-slate-800/50 text-slate-400 hover:text-white hover:bg-slate-700/50 transition-all text-xs"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
          </svg>
          <span>Explore Full Correlation Matrix</span>
        </Link>
      </div>
    </div>
  );
}

export default CorrelationInsightsCard;
