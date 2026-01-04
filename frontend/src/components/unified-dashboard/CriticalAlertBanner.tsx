/**
 * Critical Alert Banner - Unified Command Center
 *
 * A prominent, attention-grabbing banner that appears when critical issues
 * require immediate attention. Designed to be the first thing operators see.
 */

import { motion, AnimatePresence } from 'framer-motion';
import { Link } from 'react-router-dom';
import { buildInvestigationsDeepLink } from '../../types/unified-dashboard';

interface CriticalAlertBannerProps {
  criticalCount: number;
  onDismiss?: () => void;
}

export function CriticalAlertBanner({ criticalCount, onDismiss }: CriticalAlertBannerProps) {
  if (criticalCount === 0) return null;

  const deepLink = buildInvestigationsDeepLink({ severity: 'critical', status: 'open' });

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        className="relative overflow-hidden rounded-xl border border-red-500/30 bg-gradient-to-r from-red-950/80 via-red-900/60 to-red-950/80"
      >
        {/* Animated pulse background */}
        <div className="absolute inset-0 bg-red-500/5 animate-pulse" />

        {/* Scan line effect */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute inset-0 bg-gradient-to-b from-transparent via-red-500/10 to-transparent animate-scan" />
        </div>

        <div className="relative flex items-center justify-between gap-4 px-5 py-4">
          <div className="flex items-center gap-4">
            {/* Pulsing indicator */}
            <div className="relative flex items-center justify-center">
              <span className="absolute w-4 h-4 rounded-full bg-red-500 animate-ping opacity-75" />
              <span className="relative w-3 h-3 rounded-full bg-red-500" />
            </div>

            {/* Alert content */}
            <div className="flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-3">
              <span className="text-red-400 font-semibold tracking-wide uppercase text-sm">
                Critical Alert
              </span>
              <span className="hidden sm:block text-red-500/50">|</span>
              <span className="text-white">
                <span className="font-mono font-bold text-red-400">{criticalCount}</span>
                {' '}critical {criticalCount === 1 ? 'issue requires' : 'issues require'} immediate attention
              </span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Action button */}
            <Link
              to={deepLink}
              className="flex items-center gap-2 px-4 py-2 bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 rounded-lg text-red-400 hover:text-red-300 font-medium text-sm transition-all group"
            >
              <span>Investigate</span>
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
                  d="M9 5l7 7-7 7"
                />
              </svg>
            </Link>

            {/* Dismiss button */}
            {onDismiss && (
              <button
                onClick={onDismiss}
                className="p-1.5 text-red-500/50 hover:text-red-400 transition-colors"
                aria-label="Dismiss alert"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}

export default CriticalAlertBanner;
