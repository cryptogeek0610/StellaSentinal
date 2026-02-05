/**
 * CollapsibleSection - Expandable domain section for the unified dashboard
 *
 * Provides progressive disclosure: shows summary when collapsed,
 * reveals full detail when expanded. Maintains spatial context.
 */

import { useState, ReactNode } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import clsx from 'clsx';

interface CollapsibleSectionProps {
  title: string;
  icon: ReactNode;
  /** Summary metrics shown in collapsed state */
  summaryMetrics?: Array<{
    label: string;
    value: string | number;
    variant?: 'default' | 'warning' | 'danger' | 'success';
  }>;
  /** Badge shown next to title */
  badge?: {
    value: string | number;
    variant?: 'default' | 'warning' | 'danger' | 'success';
  };
  /** Content shown when collapsed (summary view) */
  collapsedContent?: ReactNode;
  /** Content shown when expanded (full detail) */
  expandedContent: ReactNode;
  /** Default expanded state */
  defaultExpanded?: boolean;
  /** Accent color for the section */
  accent?: 'amber' | 'emerald' | 'red' | 'purple' | 'blue' | 'slate';
  /** Optional className */
  className?: string;
  /** Callback when expansion state changes */
  onToggle?: (expanded: boolean) => void;
}

const accentStyles = {
  amber: {
    border: 'border-amber-500/20 hover:border-amber-500/40',
    icon: 'text-amber-400',
    glow: 'shadow-amber-500/5',
  },
  emerald: {
    border: 'border-emerald-500/20 hover:border-emerald-500/40',
    icon: 'text-emerald-400',
    glow: 'shadow-emerald-500/5',
  },
  red: {
    border: 'border-red-500/20 hover:border-red-500/40',
    icon: 'text-red-400',
    glow: 'shadow-red-500/5',
  },
  purple: {
    border: 'border-purple-500/20 hover:border-purple-500/40',
    icon: 'text-purple-400',
    glow: 'shadow-purple-500/5',
  },
  blue: {
    border: 'border-blue-500/20 hover:border-blue-500/40',
    icon: 'text-blue-400',
    glow: 'shadow-blue-500/5',
  },
  slate: {
    border: 'border-slate-600/30 hover:border-slate-500/50',
    icon: 'text-slate-400',
    glow: 'shadow-slate-500/5',
  },
};

const variantStyles = {
  default: 'text-slate-300',
  warning: 'text-amber-400',
  danger: 'text-red-400',
  success: 'text-emerald-400',
};

const badgeVariantStyles = {
  default: 'bg-slate-700/50 text-slate-300',
  warning: 'bg-amber-500/20 text-amber-400',
  danger: 'bg-red-500/20 text-red-400',
  success: 'bg-emerald-500/20 text-emerald-400',
};

export function CollapsibleSection({
  title,
  icon,
  summaryMetrics,
  badge,
  collapsedContent,
  expandedContent,
  defaultExpanded = false,
  accent = 'slate',
  className,
  onToggle,
}: CollapsibleSectionProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);
  const styles = accentStyles[accent];

  const handleToggle = () => {
    const newState = !isExpanded;
    setIsExpanded(newState);
    onToggle?.(newState);
  };

  return (
    <motion.div
      layout
      className={clsx(
        'rounded-xl border bg-slate-800/30 backdrop-blur-sm transition-all duration-200',
        styles.border,
        isExpanded && styles.glow,
        isExpanded && 'shadow-lg',
        className
      )}
    >
      {/* Header - Always visible */}
      <button
        onClick={handleToggle}
        aria-expanded={isExpanded}
        aria-label={`${isExpanded ? 'Collapse' : 'Expand'} ${title}`}
        className="w-full flex items-center justify-between p-4 text-left group"
      >
        <div className="flex items-center gap-3">
          {/* Expand/Collapse indicator */}
          <motion.div
            animate={{ rotate: isExpanded ? 90 : 0 }}
            transition={{ duration: 0.2 }}
            className="text-slate-500 group-hover:text-slate-400"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </motion.div>

          {/* Icon */}
          <div className={clsx('flex-shrink-0', styles.icon)}>
            {icon}
          </div>

          {/* Title */}
          <h3 className="text-white font-semibold text-sm uppercase tracking-wide">
            {title}
          </h3>

          {/* Badge */}
          {badge && (
            <span className={clsx(
              'px-2 py-0.5 rounded-full text-xs font-medium',
              badgeVariantStyles[badge.variant || 'default']
            )}>
              {badge.value}
            </span>
          )}
        </div>

        {/* Summary metrics - shown when collapsed */}
        {!isExpanded && summaryMetrics && summaryMetrics.length > 0 && (
          <div className="flex items-center gap-4">
            {summaryMetrics.map((metric, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="text-xs text-slate-500">{metric.label}</span>
                <span className={clsx(
                  'text-sm font-mono font-medium',
                  variantStyles[metric.variant || 'default']
                )}>
                  {metric.value}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Expand indicator text */}
        <span className="text-xs text-slate-500 group-hover:text-slate-400 transition-colors ml-4">
          {isExpanded ? 'Collapse' : 'Expand'}
        </span>
      </button>

      {/* Collapsed content preview */}
      <AnimatePresence>
        {!isExpanded && collapsedContent && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="px-4 pb-4"
          >
            {collapsedContent}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Expanded content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3, ease: 'easeInOut' }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 pt-2 border-t border-slate-700/30">
              {expandedContent}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

export default CollapsibleSection;
