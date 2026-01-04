/**
 * Compact Activity Feed - Unified Command Center
 *
 * A condensed view of recent system activity. Shows the most recent
 * events without the full detail of the main activity feed.
 * Links to the detailed dashboard for full history.
 */

import { motion } from 'framer-motion';
import { Link, useNavigate } from 'react-router-dom';
import { formatDistanceToNow, parseISO } from 'date-fns';
import clsx from 'clsx';
import { Card } from '../Card';
import type { CompactActivityItem } from '../../types/unified-dashboard';

interface CompactActivityFeedProps {
  items: CompactActivityItem[];
  maxItems?: number;
  isLoading?: boolean;
}

const EVENT_CONFIG: Record<
  CompactActivityItem['type'],
  { icon: React.ReactNode; color: string; bg: string }
> = {
  anomaly_detected: {
    icon: (
      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
    ),
    color: 'text-orange-400',
    bg: 'bg-orange-500/10',
  },
  anomaly_resolved: {
    icon: (
      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
      </svg>
    ),
    color: 'text-emerald-400',
    bg: 'bg-emerald-500/10',
  },
  training_complete: {
    icon: (
      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
    ),
    color: 'text-indigo-400',
    bg: 'bg-indigo-500/10',
  },
  system_event: {
    icon: (
      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
    color: 'text-slate-400',
    bg: 'bg-slate-500/10',
  },
  insight_generated: {
    icon: (
      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
    color: 'text-purple-400',
    bg: 'bg-purple-500/10',
  },
};

const SEVERITY_DOTS: Record<string, string> = {
  critical: 'bg-red-500',
  high: 'bg-orange-500',
  medium: 'bg-amber-500',
  low: 'bg-slate-500',
  info: 'bg-blue-500',
};

function ActivityItem({ item, index }: { item: CompactActivityItem; index: number }) {
  const navigate = useNavigate();
  const config = EVENT_CONFIG[item.type];
  const isClickable = !!item.linkTo;

  const handleClick = () => {
    if (item.linkTo) {
      const params = new URLSearchParams(item.linkParams);
      const path = params.toString() ? `${item.linkTo}?${params.toString()}` : item.linkTo;
      navigate(path);
    }
  };

  const timeAgo = formatDistanceToNow(parseISO(item.timestamp), { addSuffix: true });

  const content = (
    <div className="flex items-start gap-3">
      {/* Icon */}
      <div className={clsx('flex-shrink-0 w-7 h-7 rounded-lg flex items-center justify-center', config.bg, config.color)}>
        {config.icon}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm text-white truncate">{item.title}</span>
          {item.severity && (
            <span className={clsx('w-1.5 h-1.5 rounded-full flex-shrink-0', SEVERITY_DOTS[item.severity])} />
          )}
        </div>
        <span className="text-xs text-slate-600">{timeAgo}</span>
      </div>

      {/* Arrow for clickable items */}
      {isClickable && (
        <div className="flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity self-center">
          <svg className="w-3.5 h-3.5 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </div>
      )}
    </div>
  );

  return (
    <motion.div
      initial={{ opacity: 0, y: 5 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
    >
      {isClickable ? (
        <button
          onClick={handleClick}
          className="w-full text-left py-2.5 px-3 -mx-3 rounded-lg hover:bg-slate-800/50 transition-colors group cursor-pointer"
        >
          {content}
        </button>
      ) : (
        <div className="py-2.5">{content}</div>
      )}
    </motion.div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="space-y-3">
      {[1, 2, 3].map((i) => (
        <div key={i} className="flex items-start gap-3 animate-pulse">
          <div className="w-7 h-7 rounded-lg bg-slate-700/50" />
          <div className="flex-1 space-y-1.5">
            <div className="h-4 w-3/4 bg-slate-700/50 rounded" />
            <div className="h-3 w-20 bg-slate-700/50 rounded" />
          </div>
        </div>
      ))}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-6 text-center">
      <div className="w-10 h-10 rounded-full bg-slate-700/50 flex items-center justify-center mb-2">
        <svg className="w-5 h-5 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      </div>
      <p className="text-slate-500 text-sm">No recent activity</p>
    </div>
  );
}

export function CompactActivityFeed({
  items,
  maxItems = 5,
  isLoading = false,
}: CompactActivityFeedProps) {
  const displayItems = items.slice(0, maxItems);

  return (
    <Card title="Recent Activity" accent="stellar">
      <div className="space-y-1">
        {isLoading ? (
          <LoadingSkeleton />
        ) : displayItems.length === 0 ? (
          <EmptyState />
        ) : (
          <>
            {displayItems.map((item, index) => (
              <ActivityItem key={item.id} item={item} index={index} />
            ))}

            {/* View all link */}
            <div className="pt-3 mt-2 border-t border-slate-700/50">
              <Link
                to="/dashboard/detailed"
                className="inline-flex items-center gap-2 text-sm text-slate-400 hover:text-white font-medium transition-colors group"
              >
                <span>View full activity log</span>
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
          </>
        )}
      </div>
    </Card>
  );
}

export default CompactActivityFeed;
