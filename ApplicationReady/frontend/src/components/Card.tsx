/**
 * Card Components - Stellar Operations UI
 * 
 * Reusable card components with glass morphism effects,
 * accent colors, and smooth animations
 */

import React from 'react';
import { motion } from 'framer-motion';
import { clsx } from 'clsx';
import { InfoTooltip } from './ui/InfoTooltip';
import { CardMenu, MenuAction } from './ui/CardMenu';

interface CardProps {
  title?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  noPadding?: boolean;
  accent?: 'stellar' | 'aurora' | 'danger' | 'warning' | 'plasma' | 'cyan' | 'green' | 'red' | 'orange' | 'purple';
  animate?: boolean;
  glow?: boolean;
  // UI/UX enhancements
  showInfoIcon?: boolean;
  infoContent?: string | React.ReactNode;
  showMenu?: boolean;
  onFullscreen?: () => void;
  onExport?: () => void;
  onSettings?: () => void;
  onRemove?: () => void;
  customMenuActions?: MenuAction[];
}

export const Card: React.FC<CardProps> = ({
  title,
  children,
  className,
  noPadding = false,
  accent,
  animate = true,
  glow = false,
  showInfoIcon = false,
  infoContent,
  showMenu = false,
  onFullscreen,
  onExport,
  onSettings,
  onRemove,
  customMenuActions,
}) => {
  const accentColors: Record<string, string> = {
    stellar: 'border-t-amber-500',
    aurora: 'border-t-emerald-500',
    danger: 'border-t-red-500',
    warning: 'border-t-orange-500',
    plasma: 'border-t-indigo-500',
    // Legacy colors for compatibility
    cyan: 'border-t-cyan-500',
    green: 'border-t-emerald-500',
    red: 'border-t-red-500',
    orange: 'border-t-orange-500',
    purple: 'border-t-purple-500',
  };

  const glowStyles: Record<string, string> = {
    stellar: 'shadow-stellar',
    aurora: 'shadow-aurora',
    danger: 'shadow-danger',
    warning: 'shadow-[0_0_20px_rgba(255,107,53,0.2)]',
    plasma: 'shadow-[0_0_20px_rgba(99,102,241,0.2)]',
  };

  const CardWrapper = animate ? motion.div : 'div';
  const animationProps = animate
    ? {
      initial: { opacity: 0, y: 10 },
      animate: { opacity: 1, y: 0 },
      transition: { duration: 0.3 },
    }
    : {};

  return (
    <CardWrapper
      className={clsx(
        'stellar-card rounded-xl',
        accent && `border-t-2 ${accentColors[accent]}`,
        glow && accent && glowStyles[accent],
        className
      )}
      {...animationProps}
    >
      {title && (
        <div className="px-5 py-4 border-b border-slate-700/30">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2 flex-1 min-w-0">
              {typeof title === 'string' ? (
                <h3 className="telemetry-label">{title}</h3>
              ) : (
                title
              )}
              {showInfoIcon && infoContent && (
                <InfoTooltip content={infoContent} position="right" />
              )}
            </div>
            {showMenu && (
              <CardMenu
                onFullscreen={onFullscreen}
                onExport={onExport}
                onSettings={onSettings}
                onRemove={onRemove}
                customActions={customMenuActions}
              />
            )}
          </div>
        </div>
      )}
      <div className={clsx(!noPadding && 'p-5')}>{children}</div>
    </CardWrapper>
  );
};

interface StatCardProps {
  label: string;
  value: string | number;
  change?: {
    value: number;
    type: 'increase' | 'decrease';
  };
  icon?: React.ReactNode;
  color?: 'stellar' | 'aurora' | 'danger' | 'warning' | 'cyan' | 'green' | 'red' | 'orange';
  onClick?: () => void;
}

export const StatCard: React.FC<StatCardProps> = ({
  label,
  value,
  change,
  icon,
  color = 'stellar',
  onClick,
}) => {
  const colorClasses: Record<string, { text: string; glow: string; bg: string }> = {
    stellar: {
      text: 'text-amber-400',
      glow: 'shadow-glow-stellar',
      bg: 'from-amber-500/20',
    },
    aurora: {
      text: 'text-emerald-400',
      glow: 'shadow-glow-green',
      bg: 'from-emerald-500/20',
    },
    danger: {
      text: 'text-red-400',
      glow: 'shadow-glow-red',
      bg: 'from-red-500/20',
    },
    warning: {
      text: 'text-orange-400',
      glow: 'shadow-glow-orange',
      bg: 'from-orange-500/20',
    },
    // Legacy colors
    cyan: {
      text: 'text-cyan-400',
      glow: 'shadow-glow-cyan',
      bg: 'from-cyan-500/20',
    },
    green: {
      text: 'text-emerald-400',
      glow: 'shadow-glow-green',
      bg: 'from-emerald-500/20',
    },
    red: {
      text: 'text-red-400',
      glow: 'shadow-glow-red',
      bg: 'from-red-500/20',
    },
    orange: {
      text: 'text-orange-400',
      glow: 'shadow-glow-orange',
      bg: 'from-orange-500/20',
    },
  };

  const config = colorClasses[color] || colorClasses.stellar;

  return (
    <motion.div
      className={clsx(
        'metric-card cursor-pointer group relative overflow-hidden',
        onClick && 'hover:border-amber-500/30'
      )}
      onClick={onClick}
      whileHover={{ scale: 1.02, y: -2 }}
      whileTap={{ scale: 0.98 }}
    >
      <div className="flex items-start justify-between relative z-10">
        <div className="flex-1">
          <p className="telemetry-label mb-2">{label}</p>
          <p className={clsx('text-3xl font-bold font-mono text-white')}>
            {value}
          </p>
          {change && (
            <div className="flex items-center gap-1.5 mt-2">
              <span
                className={clsx(
                  'flex items-center gap-1 text-xs font-semibold font-mono',
                  change.type === 'increase' ? 'text-emerald-400' : 'text-red-400'
                )}
              >
                {change.type === 'increase' ? (
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 10l7-7m0 0l7 7m-7-7v18" />
                  </svg>
                ) : (
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                  </svg>
                )}
                {Math.abs(change.value)}%
              </span>
              <span className="text-xs text-slate-600">vs. yesterday</span>
            </div>
          )}
        </div>
        {icon && (
          <div
            className={clsx(
              'p-3 rounded-xl bg-gradient-to-br to-transparent transition-transform group-hover:scale-110',
              config.bg,
              config.text
            )}
          >
            {icon}
          </div>
        )}
      </div>

      {/* Animated gradient overlay */}
      <div
        className={clsx(
          'absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-xl pointer-events-none',
          `bg-gradient-to-tr ${config.bg} to-transparent`
        )}
      />
    </motion.div>
  );
};

// Mini stat display for compact layouts
interface MiniStatProps {
  label: string;
  value: string | number;
  color?: 'stellar' | 'aurora' | 'danger' | 'text';
}

export const MiniStat: React.FC<MiniStatProps> = ({
  label,
  value,
  color = 'text',
}) => {
  const colorMap: Record<string, string> = {
    stellar: 'text-amber-400',
    aurora: 'text-emerald-400',
    danger: 'text-red-400',
    text: 'text-white',
  };

  return (
    <div className="text-center">
      <p className={`text-2xl font-bold font-mono ${colorMap[color]}`}>{value}</p>
      <p className="text-[10px] text-slate-500 uppercase tracking-wider mt-1">{label}</p>
    </div>
  );
};
