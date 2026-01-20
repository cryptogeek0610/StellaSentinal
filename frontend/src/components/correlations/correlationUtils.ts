/**
 * Utility functions for correlation visualization and formatting.
 */

// Domain filter options - expanded for comprehensive correlation analysis
export const DOMAIN_OPTIONS = [
  { id: undefined, label: 'All Domains' },
  { id: 'battery', label: 'Battery' },
  { id: 'power', label: 'Power' },
  { id: 'rf', label: 'RF/Signal' },
  { id: 'network_type', label: 'Network Type' },
  { id: 'throughput', label: 'Throughput' },
  { id: 'usage', label: 'App Usage' },
  { id: 'storage', label: 'Storage' },
  { id: 'system', label: 'System' },
] as const;

// Sub-view types
export type SubView = 'matrix' | 'scatter' | 'causal' | 'insights' | 'cohort' | 'lagged';

// Color scale for correlation values (-1 to +1)
export const getCorrelationColor = (value: number): string => {
  if (value >= 0.7) return '#22c55e';  // Strong positive - green
  if (value >= 0.4) return '#84cc16';  // Moderate positive - lime
  if (value >= 0.1) return '#eab308';  // Weak positive - yellow
  if (value >= -0.1) return '#64748b'; // Near zero - slate
  if (value >= -0.4) return '#f97316'; // Weak negative - orange
  if (value >= -0.7) return '#ef4444'; // Moderate negative - red
  return '#dc2626';                     // Strong negative - dark red
};

// Get text color for contrast on correlation cell
export const getCorrelationTextColor = (value: number): string => {
  const absValue = Math.abs(value);
  return absValue >= 0.5 ? '#ffffff' : '#1e293b';
};

// Format correlation value for display
export const formatCorrelation = (value: number): string => {
  if (value === 1) return '1.00';
  if (value === -1) return '-1.0';
  return value.toFixed(2);
};

// Truncate long metric names for display
export const truncateMetric = (name: string, maxLen: number = 12): string => {
  if (name.length <= maxLen) return name;
  return name.substring(0, maxLen - 2) + '..';
};

// Format p-value for display
export const formatPValue = (p: number): string => {
  if (p < 0.001) return '<0.001';
  if (p < 0.01) return p.toFixed(3);
  return p.toFixed(2);
};

// Get node color by domain for causal graph
export const getDomainColor = (domain: string): string => {
  const colors: Record<string, string> = {
    battery: '#22c55e',
    rf: '#3b82f6',
    throughput: '#8b5cf6',
    app: '#f59e0b',
    system: '#ef4444',
    network: '#06b6d4',
    storage: '#ec4899',
  };
  return colors[domain.toLowerCase()] || '#64748b';
};

// Get strength color class for insights
export const getStrengthColor = (strength: string): string => {
  switch (strength) {
    case 'strong': return 'bg-green-500/20 text-green-400 border-green-500/30';
    case 'moderate': return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
    default: return 'bg-slate-500/20 text-slate-400 border-slate-500/30';
  }
};

// View tab options
export const VIEW_OPTIONS = [
  { id: 'matrix', label: 'Matrix' },
  { id: 'scatter', label: 'Scatter' },
  { id: 'causal', label: 'Causal' },
  { id: 'insights', label: 'Insights' },
  { id: 'cohort', label: 'Cohort' },
  { id: 'lagged', label: 'Lagged' },
] as const;
