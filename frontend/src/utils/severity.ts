/**
 * Centralized Severity Configuration
 *
 * Single source of truth for anomaly severity levels,
 * colors, labels, and thresholds across the application.
 */

export type SeverityLevel = 'critical' | 'high' | 'medium' | 'low';

export interface SeverityConfig {
  level: SeverityLevel;
  label: string;
  threshold: number;
  color: {
    name: string;
    bg: string;
    bgHover: string;
    text: string;
    border: string;
    dot: string;
    glow: string;
  };
  pulse: boolean;
  icon: 'alert' | 'warning' | 'info' | 'check';
}

export const SEVERITY_CONFIGS: Record<SeverityLevel, SeverityConfig> = {
  critical: {
    level: 'critical',
    label: 'CRITICAL',
    threshold: -0.7,
    color: {
      name: 'red',
      bg: 'bg-red-500/20',
      bgHover: 'hover:bg-red-500/30',
      text: 'text-red-400',
      border: 'border-red-500/30',
      dot: 'bg-red-500',
      glow: 'shadow-[0_0_8px_rgba(239,68,68,0.6)]',
    },
    pulse: true,
    icon: 'alert',
  },
  high: {
    level: 'high',
    label: 'HIGH',
    threshold: -0.5,
    color: {
      name: 'orange',
      bg: 'bg-orange-500/20',
      bgHover: 'hover:bg-orange-500/30',
      text: 'text-orange-400',
      border: 'border-orange-500/30',
      dot: 'bg-orange-500',
      glow: 'shadow-[0_0_8px_rgba(251,146,60,0.6)]',
    },
    pulse: false,
    icon: 'warning',
  },
  medium: {
    level: 'medium',
    label: 'MEDIUM',
    threshold: -0.3,
    color: {
      name: 'amber',
      bg: 'bg-amber-500/20',
      bgHover: 'hover:bg-amber-500/30',
      text: 'text-amber-400',
      border: 'border-amber-500/30',
      dot: 'bg-amber-500',
      glow: 'shadow-[0_0_8px_rgba(251,191,36,0.6)]',
    },
    pulse: false,
    icon: 'warning',
  },
  low: {
    level: 'low',
    label: 'LOW',
    threshold: 0,
    color: {
      name: 'cyan',
      bg: 'bg-cyan-500/20',
      bgHover: 'hover:bg-cyan-500/30',
      text: 'text-cyan-400',
      border: 'border-cyan-500/30',
      dot: 'bg-cyan-500',
      glow: 'shadow-[0_0_8px_rgba(34,211,238,0.6)]',
    },
    pulse: false,
    icon: 'info',
  },
};

/**
 * Get severity configuration based on anomaly score
 */
export function getSeverityFromScore(score: number): SeverityConfig {
  if (score <= SEVERITY_CONFIGS.critical.threshold) return SEVERITY_CONFIGS.critical;
  if (score <= SEVERITY_CONFIGS.high.threshold) return SEVERITY_CONFIGS.high;
  if (score <= SEVERITY_CONFIGS.medium.threshold) return SEVERITY_CONFIGS.medium;
  return SEVERITY_CONFIGS.low;
}

/**
 * Get severity configuration by level name
 */
export function getSeverityByLevel(level: SeverityLevel): SeverityConfig {
  return SEVERITY_CONFIGS[level];
}

/**
 * Get all severity levels in order of priority (critical first)
 */
export function getSeverityLevels(): SeverityLevel[] {
  return ['critical', 'high', 'medium', 'low'];
}

/**
 * Check if a severity level is at least as severe as another
 */
export function isAtLeastAsSevere(level: SeverityLevel, threshold: SeverityLevel): boolean {
  const levels = getSeverityLevels();
  return levels.indexOf(level) <= levels.indexOf(threshold);
}

// ============================================
// Status Configuration (for anomaly status)
// ============================================

export type StatusLevel = 'open' | 'investigating' | 'resolved' | 'false_positive';

export interface StatusConfig {
  level: StatusLevel;
  label: string;
  color: string;
  bg: string;
  border: string;
  icon: string;
}

export const STATUS_CONFIGS: Record<StatusLevel, StatusConfig> = {
  open: {
    level: 'open',
    label: 'Open',
    color: 'text-red-400',
    bg: 'bg-red-500/10',
    border: 'border-red-500/30',
    icon: '●',
  },
  investigating: {
    level: 'investigating',
    label: 'Investigating',
    color: 'text-orange-400',
    bg: 'bg-orange-500/10',
    border: 'border-orange-500/30',
    icon: '◉',
  },
  resolved: {
    level: 'resolved',
    label: 'Resolved',
    color: 'text-emerald-400',
    bg: 'bg-emerald-500/10',
    border: 'border-emerald-500/30',
    icon: '✓',
  },
  false_positive: {
    level: 'false_positive',
    label: 'False Positive',
    color: 'text-slate-400',
    bg: 'bg-slate-700/30',
    border: 'border-slate-600/30',
    icon: '✕',
  },
};

export function getStatusByLevel(level: StatusLevel): StatusConfig {
  return STATUS_CONFIGS[level] || STATUS_CONFIGS.open;
}

// ============================================
// Convenience exports for charts and badges
// ============================================

/** Hex colors for charts (Recharts, etc.) */
export const SEVERITY_COLORS: Record<string, string> = {
  critical: '#ef4444',
  high: '#f97316',
  medium: '#f59e0b',
  low: '#06b6d4',
  Critical: '#ef4444',
  High: '#f97316',
  Medium: '#f59e0b',
  Low: '#06b6d4',
  Error: '#ef4444',
  Warning: '#f59e0b',
  Info: '#3b82f6',
};

/** Badge class combinations for quick styling */
export const SEVERITY_BADGE_CLASSES: Record<SeverityLevel, string> = {
  critical: 'bg-red-500/20 border-red-500/30 text-red-400',
  high: 'bg-orange-500/20 border-orange-500/30 text-orange-400',
  medium: 'bg-amber-500/20 border-amber-500/30 text-amber-400',
  low: 'bg-cyan-500/20 border-cyan-500/30 text-cyan-400',
};
