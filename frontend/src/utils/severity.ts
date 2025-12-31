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
