/**
 * Systemic Issues Card
 *
 * Displays fleet-wide patterns detected by cross-device analysis.
 * Shows issues like "Samsung S21 has 2.3x higher crash rate than fleet average"
 */

import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { api } from '../../api/client';
import type { SystemicIssue } from '../../api/client';
import clsx from 'clsx';

const severityConfig = {
  critical: {
    bg: 'bg-red-500/10',
    border: 'border-red-500/30',
    text: 'text-red-400',
    badge: 'bg-red-500/20 text-red-400 border-red-500/30',
  },
  high: {
    bg: 'bg-orange-500/10',
    border: 'border-orange-500/30',
    text: 'text-orange-400',
    badge: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
  },
  medium: {
    bg: 'bg-amber-500/10',
    border: 'border-amber-500/30',
    text: 'text-amber-400',
    badge: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  },
  low: {
    bg: 'bg-slate-500/10',
    border: 'border-slate-500/30',
    text: 'text-slate-400',
    badge: 'bg-slate-500/20 text-slate-400 border-slate-500/30',
  },
};

const issueTypeLabels: Record<string, string> = {
  model_issue: 'Model',
  os_issue: 'OS',
  firmware_issue: 'Firmware',
  carrier_issue: 'Carrier',
  location_issue: 'Location',
};

const issueTypeIcons: Record<string, React.ReactNode> = {
  model_issue: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 18h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
    </svg>
  ),
  os_issue: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
    </svg>
  ),
  firmware_issue: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
    </svg>
  ),
  carrier_issue: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.111 16.404a5.5 5.5 0 017.778 0M12 20h.01m-7.08-7.071c3.904-3.905 10.236-3.905 14.141 0M1.394 9.393c5.857-5.857 15.355-5.857 21.213 0" />
    </svg>
  ),
  location_issue: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
    </svg>
  ),
};

function IssueItem({ issue }: { issue: SystemicIssue }) {
  const config = severityConfig[issue.severity] || severityConfig.low;

  return (
    <div className={clsx('p-3 rounded-lg border', config.bg, config.border)}>
      <div className="flex items-start gap-3">
        <div className={clsx('p-1.5 rounded-lg', config.bg)}>
          <span className={config.text}>
            {issueTypeIcons[issue.issue_type] || issueTypeIcons.model_issue}
          </span>
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className={clsx('text-xs font-medium px-1.5 py-0.5 rounded border', config.badge)}>
              {issueTypeLabels[issue.issue_type] || issue.issue_type}
            </span>
            <span className={clsx('text-xs font-medium px-1.5 py-0.5 rounded border', config.badge)}>
              {issue.severity}
            </span>
          </div>
          <p className="text-sm text-white font-medium mb-1">{issue.cohort_description}</p>
          <p className="text-xs text-slate-400">
            {(issue.deviation_multiplier ?? 0).toFixed(1)}x {issue.metric} vs fleet avg
            <span className="mx-1">·</span>
            {issue.affected_device_count ?? 0} devices
          </p>
        </div>
      </div>
    </div>
  );
}

interface SystemicIssuesCardProps {
  maxIssues?: number;
}

export function SystemicIssuesCard({ maxIssues = 5 }: SystemicIssuesCardProps) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['cross-device', 'systemic-issues'],
    queryFn: () => api.getSystemicIssues(),
    refetchInterval: 60000, // Refresh every minute
  });

  if (isLoading) {
    return (
      <div className="stellar-glass rounded-xl p-4 border border-slate-700/50">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium text-slate-300">Fleet Patterns</h3>
        </div>
        <div className="space-y-2">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-16 rounded-lg bg-slate-800/50 animate-pulse" />
          ))}
        </div>
      </div>
    );
  }

  if (error || !data?.issues || data.issues.length === 0) {
    return (
      <div className="stellar-glass rounded-xl p-4 border border-slate-700/50">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium text-slate-300">Fleet Patterns</h3>
        </div>
        <div className="text-center py-6">
          <div className="w-10 h-10 rounded-full bg-emerald-500/10 flex items-center justify-center mx-auto mb-2">
            <svg className="w-5 h-5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          </div>
          <p className="text-sm text-slate-400">No systemic issues detected</p>
          <p className="text-xs text-slate-500 mt-1">Fleet operating normally</p>
        </div>
      </div>
    );
  }

  const issues = data.issues.slice(0, maxIssues);
  const totalIssues = data.total_count;
  const hasMore = totalIssues > maxIssues;

  return (
    <div className="stellar-glass rounded-xl p-4 border border-slate-700/50">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-medium text-slate-300">Fleet Patterns</h3>
          {totalIssues > 0 && (
            <span className="text-xs px-1.5 py-0.5 rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30">
              {totalIssues} issues
            </span>
          )}
        </div>
        <Link
          to="/insights?tab=correlations"
          className="text-xs text-slate-500 hover:text-amber-400 transition-colors flex items-center gap-1"
        >
          <span>Details</span>
          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </Link>
      </div>

      <div className="space-y-2">
        {issues.map((issue) => (
          <IssueItem key={issue.issue_id} issue={issue} />
        ))}
      </div>

      {hasMore && (
        <div className="mt-3 pt-3 border-t border-slate-700/30 text-center">
          <Link
            to="/insights?tab=correlations"
            className="text-xs text-amber-400 hover:text-amber-300 transition-colors"
          >
            View {totalIssues - maxIssues} more issues
          </Link>
        </div>
      )}

      {data.highest_impact_cohort && (
        <div className="mt-3 pt-3 border-t border-slate-700/30">
          <p className="text-xs text-slate-500">
            Highest impact: <span className="text-slate-300">{data.highest_impact_cohort}</span>
          </p>
        </div>
      )}
    </div>
  );
}

export default SystemicIssuesCard;
