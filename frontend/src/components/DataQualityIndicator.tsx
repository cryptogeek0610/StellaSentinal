/**
 * Data Quality Indicator Component
 *
 * Small badge/indicator showing data quality status.
 * Used in dashboard header and System page.
 */

import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import clsx from 'clsx';
import { useState } from 'react';

const gradeConfig = {
  A: {
    color: 'text-emerald-400',
    bg: 'bg-emerald-500/20',
    border: 'border-emerald-500/30',
    label: 'Excellent',
  },
  B: {
    color: 'text-green-400',
    bg: 'bg-green-500/20',
    border: 'border-green-500/30',
    label: 'Good',
  },
  C: {
    color: 'text-amber-400',
    bg: 'bg-amber-500/20',
    border: 'border-amber-500/30',
    label: 'Fair',
  },
  D: {
    color: 'text-orange-400',
    bg: 'bg-orange-500/20',
    border: 'border-orange-500/30',
    label: 'Poor',
  },
  F: {
    color: 'text-red-400',
    bg: 'bg-red-500/20',
    border: 'border-red-500/30',
    label: 'Critical',
  },
};

const statusConfig = {
  healthy: {
    color: 'text-emerald-400',
    icon: (
      <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
      </svg>
    ),
  },
  warning: {
    color: 'text-amber-400',
    icon: (
      <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
    ),
  },
  critical: {
    color: 'text-red-400',
    icon: (
      <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
  },
};

interface DataQualityIndicatorProps {
  showTooltip?: boolean;
  size?: 'sm' | 'md';
}

export function DataQualityIndicator({ showTooltip = true, size = 'sm' }: DataQualityIndicatorProps) {
  const [tooltipVisible, setTooltipVisible] = useState(false);

  const { data, isLoading, error } = useQuery({
    queryKey: ['data-quality', 'summary'],
    queryFn: () => api.getDataQualitySummary(),
    refetchInterval: 60000, // Refresh every minute
    retry: 1,
  });

  if (isLoading) {
    return (
      <div className={clsx(
        'rounded-full animate-pulse bg-slate-700/50',
        size === 'sm' ? 'w-6 h-6' : 'w-8 h-8'
      )} />
    );
  }

  if (error || !data) {
    return (
      <div
        className={clsx(
          'rounded-full flex items-center justify-center bg-slate-700/50 border border-slate-600/30',
          size === 'sm' ? 'w-6 h-6' : 'w-8 h-8'
        )}
        title="Data quality unavailable"
      >
        <span className="text-slate-500 text-xs">?</span>
      </div>
    );
  }

  const grade = gradeConfig[data.quality_grade] || gradeConfig.C;
  const status = statusConfig[data.status] || statusConfig.healthy;

  return (
    <div className="relative">
      <button
        onMouseEnter={() => showTooltip && setTooltipVisible(true)}
        onMouseLeave={() => setTooltipVisible(false)}
        onClick={() => showTooltip && setTooltipVisible(!tooltipVisible)}
        className={clsx(
          'rounded-full flex items-center justify-center border transition-all',
          grade.bg,
          grade.border,
          size === 'sm' ? 'w-6 h-6' : 'w-8 h-8'
        )}
        title={`Data Quality: ${data.quality_grade} (${grade.label})`}
      >
        <span className={clsx('font-bold', grade.color, size === 'sm' ? 'text-xs' : 'text-sm')}>
          {data.quality_grade}
        </span>
      </button>

      {/* Tooltip */}
      {showTooltip && tooltipVisible && (
        <div className="absolute top-full mt-2 right-0 z-50 w-64 p-3 rounded-lg bg-slate-900 border border-slate-700 shadow-xl">
          <div className="flex items-center gap-2 mb-2">
            <span className={clsx(grade.color)}>{status.icon}</span>
            <span className="text-sm font-medium text-white">Data Quality: {grade.label}</span>
          </div>

          <div className="space-y-1.5 text-xs">
            <div className="flex justify-between">
              <span className="text-slate-400">Match Rate</span>
              <span className="text-white">{(data.match_rate * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Quality Score</span>
              <span className="text-white">{data.quality_score.toFixed(1)}/100</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">XSight Staleness</span>
              <span className={clsx(
                data.xsight_staleness_hours > 24 ? 'text-amber-400' : 'text-white'
              )}>
                {data.xsight_staleness_hours.toFixed(1)}h
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">MobiControl Staleness</span>
              <span className={clsx(
                data.mobicontrol_staleness_hours > 24 ? 'text-amber-400' : 'text-white'
              )}>
                {data.mobicontrol_staleness_hours.toFixed(1)}h
              </span>
            </div>
            {data.issues_count > 0 && (
              <div className="flex justify-between">
                <span className="text-slate-400">Issues</span>
                <span className="text-amber-400">{data.issues_count}</span>
              </div>
            )}
          </div>

          <div className="mt-2 pt-2 border-t border-slate-700/50">
            <a
              href="/system"
              className="text-xs text-slate-400 hover:text-amber-400 transition-colors flex items-center gap-1"
            >
              <span>View details in System</span>
              <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </a>
          </div>
        </div>
      )}
    </div>
  );
}

export default DataQualityIndicator;
