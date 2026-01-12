/**
 * ImpactSummaryCard - Hero card for investigations
 *
 * Steve Jobs-inspired: "The hero card should BE the interface."
 * - Always visible (not conditional)
 * - Shows impact summary when groups exist
 * - Shows "All Clear" state when no anomalies
 */

import { motion } from 'framer-motion';
import { useMemo } from 'react';

interface ImpactSummaryCardProps {
  totalGroups: number;
  coveragePercent: number;
  totalAnomalies: number;
  groupedCount: number;
  topGroupName?: string | null;
  topGroupId?: string | null;
  onStartAction?: (groupId: string) => void;
}

// Estimate time saved: bulk action ~30s per group vs ~2min per individual anomaly
const MINUTES_PER_INDIVIDUAL_ANOMALY = 2;
const MINUTES_PER_GROUP_ACTION = 0.5;

export function ImpactSummaryCard({
  totalGroups,
  coveragePercent,
  totalAnomalies,
  groupedCount,
  topGroupName,
  topGroupId,
  onStartAction,
}: ImpactSummaryCardProps) {
  // Calculate time savings
  const timeSaved = useMemo(() => {
    if (groupedCount === 0 || totalGroups === 0) return '0m';

    const individualTime = groupedCount * MINUTES_PER_INDIVIDUAL_ANOMALY;
    const groupedTime = totalGroups * MINUTES_PER_GROUP_ACTION;
    const savedMinutes = individualTime - groupedTime;

    if (savedMinutes >= 60) {
      const hours = Math.round(savedMinutes / 60 * 10) / 10;
      return `${hours}h`;
    }
    return `${Math.round(savedMinutes)}m`;
  }, [groupedCount, totalGroups]);

  // All Clear state - no anomalies at all
  if (totalAnomalies === 0) {
    return (
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative overflow-hidden rounded-xl bg-gradient-to-br from-emerald-900/30 via-slate-800/60 to-slate-900/80 border border-emerald-500/20 p-6"
      >
        <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/5 via-transparent to-emerald-500/5 pointer-events-none" />

        <div className="relative flex items-center gap-6">
          {/* Success icon */}
          <div className="flex-shrink-0 w-16 h-16 rounded-full bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
            <motion.svg
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
              className="w-8 h-8 text-emerald-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </motion.svg>
          </div>

          {/* Message */}
          <div className="flex-1">
            <h3 className="text-2xl font-semibold text-white mb-1">All Clear</h3>
            <p className="text-slate-400">
              No anomalies requiring attention. All systems operating normally.
            </p>
          </div>

          {/* Status badge */}
          <div className="flex-shrink-0 px-4 py-2 bg-emerald-500/10 rounded-lg border border-emerald-500/20">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-emerald-400 rounded-full shadow-[0_0_8px_rgba(16,185,129,0.6)]" />
              <span className="text-sm font-medium text-emerald-400">Healthy</span>
            </div>
          </div>
        </div>
      </motion.div>
    );
  }

  // No groups but some anomalies exist - show minimal state
  if (totalGroups === 0) {
    return (
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative overflow-hidden rounded-xl bg-gradient-to-br from-slate-800/80 via-slate-800/60 to-slate-900/80 border border-slate-700/30 p-6"
      >
        <div className="relative flex items-center gap-6">
          <div className="flex-shrink-0 w-12 h-12 rounded-full bg-slate-700/50 flex items-center justify-center">
            <svg className="w-6 h-6 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
            </svg>
          </div>
          <div className="flex-1">
            <h3 className="text-lg font-medium text-white">
              {totalAnomalies} unique case{totalAnomalies !== 1 ? 's' : ''} to review
            </h3>
            <p className="text-sm text-slate-500">
              No common patterns detected. Each anomaly requires individual review.
            </p>
          </div>
        </div>
      </motion.div>
    );
  }

  // Parse group name to get the action name
  const actionName = topGroupName
    ? topGroupName.replace(/\s*\(\d+\s*devices?\)\s*/gi, '').trim()
    : 'highest impact issue';

  // Normal state - groups exist
  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className="relative overflow-hidden rounded-xl bg-gradient-to-br from-slate-800/80 via-slate-800/60 to-slate-900/80 border border-amber-500/20 p-6"
    >
      <div className="absolute inset-0 bg-gradient-to-r from-amber-500/5 via-transparent to-amber-500/5 pointer-events-none" />

      <div className="relative">
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <svg className="w-5 h-5 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              <span className="text-sm font-medium text-amber-400 uppercase tracking-wider">
                Focus
              </span>
            </div>
            <h3 className="text-xl font-semibold text-white">
              {totalGroups} root cause{totalGroups !== 1 ? 's' : ''} explain{' '}
              <span className="text-amber-400">{Math.round(coveragePercent)}%</span> of anomalies
            </h3>
          </div>

          {/* Quick stats */}
          <div className="flex items-center gap-6">
            {/* Time saved badge */}
            <div className="text-center px-3 py-1.5 bg-emerald-500/10 rounded-lg border border-emerald-500/20">
              <div className="text-lg font-bold text-emerald-400">{timeSaved}</div>
              <div className="text-[10px] text-emerald-400/70 uppercase tracking-wider">saved</div>
            </div>
            {/* Count */}
            <div className="text-right">
              <div className="text-2xl font-bold text-white">{groupedCount}</div>
              <div className="text-xs text-slate-500">of {totalAnomalies}</div>
            </div>
          </div>
        </div>

        {/* Progress bar */}
        <div className="mb-4">
          <div className="h-2 bg-slate-700/50 rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${coveragePercent}%` }}
              transition={{ duration: 0.8, ease: 'easeOut' }}
              className="h-full bg-gradient-to-r from-amber-500 to-amber-400 rounded-full"
            />
          </div>
          <div className="flex justify-between mt-1 text-xs text-slate-500">
            <span>Grouped anomalies</span>
            <span>{Math.round(coveragePercent)}% coverage</span>
          </div>
        </div>

        {/* CTA */}
        {topGroupId && onStartAction && (
          <div className="flex items-center justify-between">
            <p className="text-sm text-slate-400">
              Resolve {totalGroups} group{totalGroups !== 1 ? 's' : ''} to clear{' '}
              <span className="text-white font-medium">{Math.round(coveragePercent)}%</span> of your queue
            </p>
            <button
              onClick={() => onStartAction(topGroupId)}
              className="flex items-center gap-2 px-4 py-2 bg-amber-500/20 hover:bg-amber-500/30 text-amber-400 rounded-lg font-medium text-sm transition-colors whitespace-nowrap ml-4 group"
            >
              Start with {actionName}
              <svg
                className="w-4 h-4 group-hover:translate-x-0.5 transition-transform"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </button>
          </div>
        )}
      </div>
    </motion.div>
  );
}

export default ImpactSummaryCard;
