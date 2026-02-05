/**
 * Priority Issues List - Unified Command Center
 *
 * Displays the top priority issues from the AI insights system.
 * Each issue card is clickable and deep-links to the full insights page
 * with the specific insight highlighted.
 */

import { useState, memo } from 'react';
import { motion } from 'framer-motion';
import { Link, useNavigate } from 'react-router-dom';
import clsx from 'clsx';
import { Card } from '../Card';
import { CustomerInsightResponse } from '../../api/client';
import { buildInsightDeepLink } from '../../types/unified-dashboard';
import { ImpactedDevicesPanel } from './ImpactedDevicesPanel';

interface PriorityIssuesListProps {
  issues: CustomerInsightResponse[];
  maxItems?: number;
  isLoading?: boolean;
}

const SEVERITY_CONFIG: Record<string, { bg: string; text: string; border: string; dot: string }> = {
  critical: {
    bg: 'bg-red-500/10',
    text: 'text-red-400',
    border: 'border-red-500/20 hover:border-red-500/40',
    dot: 'bg-red-500',
  },
  high: {
    bg: 'bg-orange-500/10',
    text: 'text-orange-400',
    border: 'border-orange-500/20 hover:border-orange-500/40',
    dot: 'bg-orange-500',
  },
  medium: {
    bg: 'bg-amber-500/10',
    text: 'text-amber-400',
    border: 'border-amber-500/20 hover:border-amber-500/40',
    dot: 'bg-amber-500',
  },
  low: {
    bg: 'bg-slate-500/10',
    text: 'text-slate-400',
    border: 'border-slate-500/20 hover:border-slate-500/40',
    dot: 'bg-slate-500',
  },
  info: {
    bg: 'bg-blue-500/10',
    text: 'text-blue-400',
    border: 'border-blue-500/20 hover:border-blue-500/40',
    dot: 'bg-blue-500',
  },
};

const TREND_ICONS: Record<string, { icon: string; color: string }> = {
  degrading: { icon: '↗', color: 'text-red-400' },
  stable: { icon: '→', color: 'text-slate-400' },
  improving: { icon: '↘', color: 'text-emerald-400' },
};

const CATEGORY_LABELS: Record<string, string> = {
  battery_shift_failure: 'Battery',
  battery_rapid_drain: 'Battery',
  excessive_drops: 'Drops',
  excessive_reboots: 'Reboots',
  wifi_ap_hopping: 'WiFi',
  wifi_dead_zone: 'Network',
  app_crash_pattern: 'Apps',
  app_power_drain: 'Apps',
  network_disconnect_pattern: 'Network',
  device_hidden_pattern: 'Device',
};

const PriorityIssueCard = memo(function PriorityIssueCard({
  issue,
  rank,
  onDevicesClick,
}: {
  issue: CustomerInsightResponse;
  rank: number;
  onDevicesClick: (issue: CustomerInsightResponse) => void;
}) {
  const navigate = useNavigate();
  const config = SEVERITY_CONFIG[issue.severity] || SEVERITY_CONFIG.info;
  const trend = TREND_ICONS[issue.trend_direction] || TREND_ICONS.stable;
  const categoryLabel = CATEGORY_LABELS[issue.category] || issue.category;

  const handleClick = () => {
    navigate(buildInsightDeepLink(issue.insight_id, 'digest'));
  };

  const handleDevicesClick = (e: React.MouseEvent) => {
    e.stopPropagation(); // Don't trigger card navigation
    onDevicesClick(issue);
  };

  return (
    <motion.button
      onClick={handleClick}
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: rank * 0.1 }}
      className={clsx(
        'w-full text-left p-4 rounded-lg border transition-all',
        'bg-slate-800/30 hover:bg-slate-800/50',
        config.border,
        'group cursor-pointer'
      )}
      whileHover={{ x: 4 }}
    >
      <div className="flex items-start gap-3">
        {/* Rank indicator */}
        <div
          className={clsx(
            'flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center',
            'text-xs font-bold font-mono',
            config.bg,
            config.text
          )}
        >
          {rank}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2">
            <div className="flex-1 min-w-0">
              {/* Headline */}
              <h4 className="text-white font-medium text-sm line-clamp-1 group-hover:text-amber-100 transition-colors">
                {issue.headline}
              </h4>

              {/* Impact statement */}
              <p className="text-slate-500 text-xs mt-1 line-clamp-1">
                {issue.impact_statement}
              </p>
            </div>

            {/* Trend indicator */}
            <div className={clsx('flex-shrink-0', trend.color)}>
              <span className="text-lg" title={`Trend: ${issue.trend_direction}`}>
                {trend.icon}
              </span>
            </div>
          </div>

          {/* Meta row */}
          <div className="flex items-center gap-3 mt-2">
            {/* Severity badge */}
            <span
              className={clsx(
                'inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium',
                config.bg,
                config.text
              )}
            >
              <span className={clsx('w-1.5 h-1.5 rounded-full', config.dot)} />
              {issue.severity}
            </span>

            {/* Category */}
            <span className="text-xs text-slate-600">
              {categoryLabel}
            </span>

            {/* Affected devices - clickable */}
            {issue.affected_device_count > 0 && (
              <>
                <span className="text-slate-700">|</span>
                <button
                  onClick={handleDevicesClick}
                  className="text-xs text-slate-500 hover:text-amber-400 transition-colors group/devices"
                >
                  <span className="font-mono text-slate-400 group-hover/devices:text-amber-400">
                    {issue.affected_device_count}
                  </span>{' '}
                  devices
                  <span className="opacity-0 group-hover/devices:opacity-100 ml-1 transition-opacity">
                    →
                  </span>
                </button>
              </>
            )}
          </div>
        </div>

        {/* Arrow */}
        <div className="flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity self-center">
          <svg className="w-4 h-4 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </div>
      </div>
    </motion.button>
  );
});

function LoadingSkeleton() {
  return (
    <div className="space-y-3">
      {[1, 2, 3].map((i) => (
        <div key={i} className="p-4 rounded-lg border border-slate-700/30 bg-slate-800/30 animate-pulse">
          <div className="flex items-start gap-3">
            <div className="w-6 h-6 rounded-full bg-slate-700/50" />
            <div className="flex-1 space-y-2">
              <div className="h-4 w-3/4 bg-slate-700/50 rounded" />
              <div className="h-3 w-1/2 bg-slate-700/50 rounded" />
              <div className="flex gap-2">
                <div className="h-5 w-16 bg-slate-700/50 rounded-full" />
                <div className="h-5 w-12 bg-slate-700/50 rounded" />
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-8 text-center">
      <div className="w-12 h-12 rounded-full bg-emerald-500/10 flex items-center justify-center mb-3">
        <svg className="w-6 h-6 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
        </svg>
      </div>
      <p className="text-slate-400 text-sm">No priority issues detected</p>
      <p className="text-slate-600 text-xs mt-1">Your fleet is operating normally</p>
    </div>
  );
}

export function PriorityIssuesList({
  issues,
  maxItems = 3,
  isLoading = false,
}: PriorityIssuesListProps) {
  const [devicePanelOpen, setDevicePanelOpen] = useState(false);
  const [selectedIssue, setSelectedIssue] = useState<CustomerInsightResponse | null>(null);

  const displayIssues = issues.slice(0, maxItems);
  const hasMore = issues.length > maxItems;

  const handleDevicesClick = (issue: CustomerInsightResponse) => {
    setSelectedIssue(issue);
    setDevicePanelOpen(true);
  };

  return (
    <>
      <Card title="Priority Issues" accent="danger">
        <div className="space-y-3">
          {isLoading ? (
            <LoadingSkeleton />
          ) : displayIssues.length === 0 ? (
            <EmptyState />
          ) : (
            <>
              {displayIssues.map((issue, index) => (
                <PriorityIssueCard
                  key={issue.insight_id}
                  issue={issue}
                  rank={index + 1}
                  onDevicesClick={handleDevicesClick}
                />
              ))}

              {/* View all link */}
              {hasMore && (
                <div className="pt-2 border-t border-slate-700/50">
                  <Link
                    to={buildInsightDeepLink(undefined, 'digest')}
                    className="inline-flex items-center gap-2 text-sm text-slate-400 hover:text-white font-medium transition-colors group"
                  >
                    <span>View all {issues.length} insights</span>
                    <svg
                      className="w-4 h-4 transform group-hover:translate-x-0.5 transition-transform"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                    </svg>
                  </Link>
                </div>
              )}
            </>
          )}
        </div>
      </Card>

      {/* Impacted Devices Panel */}
      {selectedIssue && (
        <ImpactedDevicesPanel
          isOpen={devicePanelOpen}
          onClose={() => setDevicePanelOpen(false)}
          insight={selectedIssue}
        />
      )}
    </>
  );
}

export default PriorityIssuesList;
