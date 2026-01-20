/**
 * Utility functions for Investigation Detail page.
 */

import {
  SEVERITY_CONFIGS,
  STATUS_CONFIGS,
  type SeverityLevel,
  type StatusLevel,
} from '../../utils/severity';

// Create local configs that match expected shapes (for backwards compatibility)
export const severityConfig = Object.fromEntries(
  Object.entries(SEVERITY_CONFIGS).map(([key, cfg]) => [
    key,
    { label: cfg.label, color: cfg.color.text, bg: cfg.color.bg, border: cfg.color.border },
  ])
) as Record<SeverityLevel, { label: string; color: string; bg: string; border: string }>;

export const statusConfig = Object.fromEntries(
  Object.entries(STATUS_CONFIGS).map(([key, cfg]) => [
    key,
    { label: cfg.label, color: cfg.color, bg: cfg.bg, icon: cfg.icon },
  ])
) as Record<StatusLevel, { label: string; color: string; bg: string; icon: string }>;

/**
 * Formats a data size in MB to the most appropriate unit (MB, GB, or TB)
 * @param megabytes - Value in megabytes
 * @returns Object with formatted value and unit string
 */
export function formatDataSize(megabytes: number | null | undefined): { value: string; unit: string } {
  if (megabytes == null) {
    return { value: '0', unit: 'MB' };
  }

  const MB_THRESHOLD = 1000;    // Show as GB if >= 1000 MB
  const GB_THRESHOLD = 1000;    // Show as TB if >= 1000 GB

  if (megabytes >= MB_THRESHOLD * GB_THRESHOLD) {
    // Convert to TB (MB / 1,000,000)
    return { value: (megabytes / 1_000_000).toFixed(2), unit: 'TB' };
  } else if (megabytes >= MB_THRESHOLD) {
    // Convert to GB (MB / 1,000)
    return { value: (megabytes / 1_000).toFixed(2), unit: 'GB' };
  } else {
    // Keep as MB
    return { value: megabytes.toFixed(1), unit: 'MB' };
  }
}

export const getSeverityKey = (score: number): keyof typeof severityConfig => {
  if (score <= -0.7) return 'critical';
  if (score <= -0.5) return 'high';
  if (score <= -0.3) return 'medium';
  return 'low';
};

export const formatCurrency = (amount: number): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(amount);
};
