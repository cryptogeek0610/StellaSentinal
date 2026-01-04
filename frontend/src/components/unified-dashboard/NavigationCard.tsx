/**
 * Navigation Card - Unified Command Center
 *
 * Clickable cards that serve as entry points to different sections
 * of the application. Each card displays a summary metric and deep-links
 * to the relevant dashboard with appropriate filters.
 */

import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import clsx from 'clsx';

interface NavigationCardProps {
  title: string;
  description: string;
  icon: React.ReactNode;
  route: string;
  queryParams?: Record<string, string>;
  metric?: string | number;
  metricLabel?: string;
  accent?: 'stellar' | 'aurora' | 'plasma' | 'danger' | 'warning';
  badge?: {
    value: number;
    variant: 'default' | 'warning' | 'danger';
  };
  isLoading?: boolean;
}

const accentConfig = {
  stellar: {
    border: 'border-amber-500/20 hover:border-amber-500/40',
    bg: 'from-amber-500/10 to-amber-400/5',
    icon: 'bg-amber-500/20 text-amber-400',
    metric: 'text-amber-400',
    glow: 'hover:shadow-[0_0_30px_rgba(245,166,35,0.15)]',
  },
  aurora: {
    border: 'border-emerald-500/20 hover:border-emerald-500/40',
    bg: 'from-emerald-500/10 to-teal-500/5',
    icon: 'bg-emerald-500/20 text-emerald-400',
    metric: 'text-emerald-400',
    glow: 'hover:shadow-[0_0_30px_rgba(16,185,129,0.15)]',
  },
  plasma: {
    border: 'border-indigo-500/20 hover:border-indigo-500/40',
    bg: 'from-indigo-500/10 to-purple-500/5',
    icon: 'bg-indigo-500/20 text-indigo-400',
    metric: 'text-indigo-400',
    glow: 'hover:shadow-[0_0_30px_rgba(99,102,241,0.15)]',
  },
  danger: {
    border: 'border-red-500/20 hover:border-red-500/40',
    bg: 'from-red-500/10 to-orange-500/5',
    icon: 'bg-red-500/20 text-red-400',
    metric: 'text-red-400',
    glow: 'hover:shadow-[0_0_30px_rgba(239,68,68,0.15)]',
  },
  warning: {
    border: 'border-orange-500/20 hover:border-orange-500/40',
    bg: 'from-orange-500/10 to-amber-500/5',
    icon: 'bg-orange-500/20 text-orange-400',
    metric: 'text-orange-400',
    glow: 'hover:shadow-[0_0_30px_rgba(249,115,22,0.15)]',
  },
};

const badgeVariants = {
  default: 'bg-slate-700 text-slate-300',
  warning: 'bg-orange-500/20 text-orange-400 border border-orange-500/30',
  danger: 'bg-red-500/20 text-red-400 border border-red-500/30 animate-pulse',
};

export function NavigationCard({
  title,
  description,
  icon,
  route,
  queryParams,
  metric,
  metricLabel,
  accent = 'stellar',
  badge,
  isLoading = false,
}: NavigationCardProps) {
  const navigate = useNavigate();
  const config = accentConfig[accent];

  const handleClick = () => {
    const searchParams = new URLSearchParams(queryParams);
    const path = searchParams.toString() ? `${route}?${searchParams.toString()}` : route;
    navigate(path);
  };

  return (
    <motion.button
      onClick={handleClick}
      className={clsx(
        'relative w-full text-left rounded-xl border p-5',
        'bg-gradient-to-br',
        'stellar-card',
        config.border,
        config.bg,
        config.glow,
        'transition-all duration-300 group'
      )}
      whileHover={{ scale: 1.02, y: -2 }}
      whileTap={{ scale: 0.98 }}
    >
      {/* Badge */}
      {badge && badge.value > 0 && (
        <div
          className={clsx(
            'absolute -top-2 -right-2 px-2 py-0.5 rounded-full text-xs font-bold font-mono',
            badgeVariants[badge.variant]
          )}
        >
          {badge.value}
        </div>
      )}

      <div className="flex items-start gap-4">
        {/* Icon */}
        <div
          className={clsx(
            'flex-shrink-0 w-12 h-12 rounded-xl flex items-center justify-center',
            'transition-transform duration-300 group-hover:scale-110',
            config.icon
          )}
        >
          {icon}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2">
            <div>
              <h3 className="text-white font-semibold text-base group-hover:text-amber-100 transition-colors">
                {title}
              </h3>
              <p className="text-slate-500 text-sm mt-0.5 line-clamp-1">{description}</p>
            </div>
          </div>

          {/* Metric */}
          {metric !== undefined && (
            <div className="mt-3 flex items-baseline gap-2">
              {isLoading ? (
                <div className="h-7 w-16 bg-slate-700/50 rounded animate-pulse" />
              ) : (
                <>
                  <span className={clsx('text-2xl font-bold font-mono', config.metric)}>
                    {metric}
                  </span>
                  {metricLabel && (
                    <span className="text-xs text-slate-500 uppercase tracking-wide">
                      {metricLabel}
                    </span>
                  )}
                </>
              )}
            </div>
          )}
        </div>

        {/* Arrow indicator */}
        <div className="flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
          <svg
            className="w-5 h-5 text-slate-500 transform group-hover:translate-x-0.5 transition-transform"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </div>
      </div>
    </motion.button>
  );
}

export default NavigationCard;
