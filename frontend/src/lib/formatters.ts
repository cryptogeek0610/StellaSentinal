/**
 * Formatting utilities for ML metrics display.
 */

/** Format a single AUC value for display (e.g. 0.9234 → "0.923"). */
export function formatAUC(value: number | null | undefined): string {
  if (value == null || isNaN(value)) return '—';
  return value.toFixed(3);
}

/** Compute and format the average AUC from an array of values, ignoring nulls. */
export function formatAvgAUC(values: (number | null | undefined)[]): string {
  const valid = values.filter((v): v is number => v != null && !isNaN(v));
  if (valid.length === 0) return '—';
  const avg = valid.reduce((sum, v) => sum + v, 0) / valid.length;
  return avg.toFixed(3);
}
