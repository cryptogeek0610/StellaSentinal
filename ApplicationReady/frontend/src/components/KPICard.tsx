/**
 * KPI Card - Stellar Operations Telemetry Display
 * 
 * Mission control style metric cards with status indicators,
 * progress bars, and subtle glow effects for visual hierarchy
 */

import React from 'react';
import { motion } from 'framer-motion';
import { InfoTooltip } from './ui/InfoTooltip';

interface KPICardProps {
  title: string;
  value: string | number;
  change?: string;
  trend?: 'up' | 'down' | 'neutral';
  icon?: React.ReactNode;
  color?: 'stellar' | 'aurora' | 'plasma' | 'danger' | 'warning';
  onClick?: () => void;
  isActive?: boolean;
  subtitle?: string;
  className?: string;
  showProgress?: boolean;
  progressValue?: number;
  // UI/UX enhancements
  explainer?: string | React.ReactNode;
  context?: string; // e.g., "vs. last week"
}

const colorConfig = {
  stellar: {
    bg: 'from-amber-500/10 to-amber-400/5',
    border: 'border-amber-500/20',
    borderActive: 'border-amber-500/50',
    icon: 'bg-amber-500/20 text-amber-400',
    glow: 'shadow-stellar',
    progress: 'bg-gradient-to-r from-amber-500 to-amber-400',
    trend: 'text-amber-400',
    dot: 'bg-amber-400',
  },
  aurora: {
    bg: 'from-emerald-500/10 to-teal-500/5',
    border: 'border-emerald-500/20',
    borderActive: 'border-emerald-500/50',
    icon: 'bg-emerald-500/20 text-emerald-400',
    glow: 'shadow-aurora',
    progress: 'bg-gradient-to-r from-emerald-500 to-teal-400',
    trend: 'text-emerald-400',
    dot: 'bg-emerald-400',
  },
  plasma: {
    bg: 'from-indigo-500/10 to-purple-500/5',
    border: 'border-indigo-500/20',
    borderActive: 'border-indigo-500/50',
    icon: 'bg-indigo-500/20 text-indigo-400',
    glow: 'shadow-[0_0_20px_rgba(99,102,241,0.2)]',
    progress: 'bg-gradient-to-r from-indigo-500 to-purple-400',
    trend: 'text-indigo-400',
    dot: 'bg-indigo-400',
  },
  danger: {
    bg: 'from-red-500/10 to-orange-500/5',
    border: 'border-red-500/20',
    borderActive: 'border-red-500/50',
    icon: 'bg-red-500/20 text-red-400',
    glow: 'shadow-danger',
    progress: 'bg-gradient-to-r from-red-500 to-orange-400',
    trend: 'text-red-400',
    dot: 'bg-red-400',
  },
  warning: {
    bg: 'from-orange-500/10 to-amber-500/5',
    border: 'border-orange-500/20',
    borderActive: 'border-orange-500/50',
    icon: 'bg-orange-500/20 text-orange-400',
    glow: 'shadow-[0_0_20px_rgba(255,107,53,0.2)]',
    progress: 'bg-gradient-to-r from-orange-500 to-amber-400',
    trend: 'text-orange-400',
    dot: 'bg-orange-400',
  },
};

export const KPICard: React.FC<KPICardProps> = ({
  title,
  value,
  change,
  trend,
  icon,
  color = 'stellar',
  onClick,
  isActive = false,
  subtitle,
  className = '',
  showProgress = true,
  progressValue,
  explainer,
  context = 'vs. yesterday',
}) => {
  const config = colorConfig[color];

  const TrendIcon = () => {
    if (trend === 'up') {
      return (
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 10l7-7m0 0l7 7m-7-7v18" />
        </svg>
      );
    }
    if (trend === 'down') {
      return (
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
        </svg>
      );
    }
    return (
      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 12h14" />
      </svg>
    );
  };

  // Calculate progress percentage
  const getProgressWidth = () => {
    if (progressValue !== undefined) {
      return Math.min(progressValue, 100);
    }
    if (typeof value === 'number') {
      return Math.min(value, 100);
    }
    if (typeof value === 'string' && value.includes('%')) {
      return parseInt(value.replace('%', ''), 10);
    }
    return 70; // Default
  };

  const CardWrapper = onClick ? motion.button : motion.div;

  // Determine trend background class
  const getTrendBg = () => {
    if (!change) return '';
    if (trend === 'up') return 'trend-positive-bg';
    if (trend === 'down') return 'trend-negative-bg';
    return 'trend-neutral-bg';
  };

  return (
    <CardWrapper
      onClick={onClick}
      className={`
        kpi-card relative overflow-hidden
        stellar-card rounded-2xl p-5
        bg-gradient-to-br ${config.bg}
        border ${isActive ? config.borderActive : config.border}
        ${isActive ? config.glow : ''}
        ${onClick ? 'cursor-pointer' : ''}
        ${getTrendBg()}
        transition-all duration-300
        text-left w-full
        ${className}
      `}
      whileHover={onClick ? { scale: 1.02, y: -2 } : undefined}
      whileTap={onClick ? { scale: 0.98 } : undefined}
    >
      {/* Scan line effect for active state */}
      {isActive && <div className="absolute inset-0 scan-line pointer-events-none" />}

      {/* Content */}
      <div className="relative">
        {/* Header Row */}
        <div className="flex items-start justify-between mb-3">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <p className="telemetry-label">{title}</p>
              {explainer && <InfoTooltip content={explainer} />}
            </div>
            <div className="flex items-baseline gap-2.5">
              <h3 className="text-3xl font-bold text-white tracking-tight font-mono">
                {value}
              </h3>
              {change && (
                <div className="flex items-center gap-2">
                  <span
                    className={`flex items-center gap-1 text-xs font-semibold ${trend === 'up'
                      ? 'text-emerald-400'
                      : trend === 'down'
                        ? 'text-red-400'
                        : 'text-slate-400'
                      }`}
                  >
                    <TrendIcon />
                    <span>{change}</span>
                  </span>
                  {context && (
                    <span className="text-xs text-slate-600">{context}</span>
                  )}
                </div>
              )}
            </div>
          </div>

          {icon && (
            <div
              className={`
                kpi-icon w-12 h-12 rounded-xl flex items-center justify-center
                ${config.icon}
                transition-transform duration-300
              `}
            >
              {icon}
            </div>
          )}
        </div>

        {/* Progress Bar */}
        {showProgress && (
          <div className="progress-bar mt-4">
            <motion.div
              className={`progress-bar-fill ${config.progress} ${isActive ? 'animate-pulse' : ''}`}
              initial={{ width: 0 }}
              animate={{ width: `${getProgressWidth()}%` }}
              transition={{ duration: 0.8, ease: 'easeOut' }}
            />
          </div>
        )}

        {/* Subtitle */}
        {subtitle && <p className="mt-3 text-xs text-slate-500">{subtitle}</p>}
      </div>

      {/* Active indicator dot */}
      {isActive && (
        <div className="absolute top-3 right-3">
          <span className={`w-2 h-2 rounded-full ${config.dot} animate-pulse inline-block`} />
        </div>
      )}

      {/* Click indicator */}
      {onClick && !isActive && (
        <div className="absolute bottom-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity">
          <svg className="w-4 h-4 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </div>
      )}
    </CardWrapper>
  );
};

// Metric display component for inline stats
export const MetricDisplay: React.FC<{
  label: string;
  value: string | number;
  unit?: string;
  color?: 'stellar' | 'aurora' | 'danger' | 'warning' | 'text';
  explainer?: string | React.ReactNode;
}> = ({ label, value, unit, color = 'stellar', explainer }) => {
  const colorMap = {
    stellar: 'text-amber-400',
    aurora: 'text-emerald-400',
    danger: 'text-red-400',
    warning: 'text-orange-400',
    text: 'text-white',
  };

  return (
    <div>
      <div className="flex items-center gap-2">
        <p className="telemetry-label">{label}</p>
        {explainer && <InfoTooltip content={explainer} />}
      </div>
      <p className={`text-lg font-bold font-mono ${colorMap[color]}`}>
        {value}
        {unit && <span className="text-xs text-slate-500 ml-1">{unit}</span>}
      </p>
    </div>
  );
};

// Status badge component
export const StatusBadge: React.FC<{
  status: 'online' | 'warning' | 'critical' | 'offline';
  label?: string;
}> = ({ status, label }) => {
  const config = {
    online: { dot: 'status-online', text: 'text-emerald-400', bg: 'bg-emerald-500/10' },
    warning: { dot: 'status-warning', text: 'text-orange-400', bg: 'bg-orange-500/10' },
    critical: { dot: 'status-critical', text: 'text-red-400', bg: 'bg-red-500/10' },
    offline: { dot: 'status-offline', text: 'text-slate-400', bg: 'bg-slate-700' },
  };

  const c = config[status];

  return (
    <div className={`inline-flex items-center gap-2 px-2.5 py-1 rounded-full ${c.bg}`}>
      <span className={`status-dot ${c.dot}`} />
      {label && <span className={`text-xs font-medium ${c.text}`}>{label}</span>}
    </div>
  );
};

export default KPICard;

