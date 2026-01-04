/**
 * Executive Summary - Unified Command Center
 *
 * Displays the AI-generated executive summary from the daily digest.
 * This provides operators with a quick narrative overview of the current
 * state of the fleet without needing to navigate to the full insights page.
 */

import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { format, parseISO } from 'date-fns';
import { Card } from '../Card';
import { buildInsightDeepLink } from '../../types/unified-dashboard';

interface ExecutiveSummaryProps {
  summary: string;
  digestDate: string;
  totalInsights: number;
  criticalCount: number;
  highCount: number;
  isLoading?: boolean;
}

export function ExecutiveSummary({
  summary,
  digestDate,
  totalInsights,
  criticalCount,
  highCount,
  isLoading = false,
}: ExecutiveSummaryProps) {
  const deepLink = buildInsightDeepLink(undefined, 'digest');

  // Format the digest date
  const formattedDate = digestDate
    ? format(parseISO(digestDate), 'EEEE, MMMM d')
    : 'Today';

  if (isLoading) {
    return (
      <Card title="AI Intelligence Briefing" accent="plasma">
        <div className="space-y-4 animate-pulse">
          <div className="flex items-center gap-3">
            <div className="h-4 w-32 bg-slate-700/50 rounded" />
            <div className="h-4 w-24 bg-slate-700/50 rounded" />
          </div>
          <div className="space-y-2">
            <div className="h-4 w-full bg-slate-700/50 rounded" />
            <div className="h-4 w-5/6 bg-slate-700/50 rounded" />
            <div className="h-4 w-4/6 bg-slate-700/50 rounded" />
          </div>
        </div>
      </Card>
    );
  }

  return (
    <Card title="AI Intelligence Briefing" accent="plasma">
      <div className="space-y-4">
        {/* Header with date and stats */}
        <div className="flex flex-wrap items-center gap-3 text-sm">
          <div className="flex items-center gap-2 text-slate-400">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
              />
            </svg>
            <span>{formattedDate}</span>
          </div>
          <span className="text-slate-700">|</span>
          <div className="flex items-center gap-4">
            <span className="text-slate-400">
              <span className="font-mono text-indigo-400">{totalInsights}</span> insights
            </span>
            {criticalCount > 0 && (
              <span className="text-red-400">
                <span className="font-mono font-bold">{criticalCount}</span> critical
              </span>
            )}
            {highCount > 0 && (
              <span className="text-orange-400">
                <span className="font-mono font-bold">{highCount}</span> high
              </span>
            )}
          </div>
        </div>

        {/* Summary content */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="relative"
        >
          {/* AI indicator */}
          <div className="absolute -left-3 top-0 bottom-0 w-1 bg-gradient-to-b from-indigo-500 via-purple-500 to-indigo-500 rounded-full opacity-50" />

          <p className="text-slate-300 leading-relaxed pl-2">
            {summary || 'No executive summary available for today.'}
          </p>
        </motion.div>

        {/* Action link */}
        <div className="pt-2 border-t border-slate-700/50">
          <Link
            to={deepLink}
            className="inline-flex items-center gap-2 text-sm text-indigo-400 hover:text-indigo-300 font-medium transition-colors group"
          >
            <span>View Full AI Digest</span>
            <svg
              className="w-4 h-4 transform group-hover:translate-x-0.5 transition-transform"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 7l5 5m0 0l-5 5m5-5H6"
              />
            </svg>
          </Link>
        </div>
      </div>
    </Card>
  );
}

export default ExecutiveSummary;
