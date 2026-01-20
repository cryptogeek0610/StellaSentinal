/**
 * Correlation components exports.
 */

// Utility functions and constants
export {
  DOMAIN_OPTIONS,
  VIEW_OPTIONS,
  getCorrelationColor,
  getCorrelationTextColor,
  formatCorrelation,
  truncateMetric,
  formatPValue,
  getDomainColor,
  getStrengthColor,
} from './correlationUtils';
export type { SubView } from './correlationUtils';

// View components
export { ErrorState } from './ErrorState';
export { CorrelationMatrixView } from './CorrelationMatrixView';
export { AnomalyDetailPanel } from './AnomalyDetailPanel';
export { ScatterExplorerView } from './ScatterExplorerView';
export { CausalGraphView } from './CausalGraphView';
export { InsightsView } from './InsightsView';
export { CohortPatternsView } from './CohortPatternsView';
export { TimeLaggedView } from './TimeLaggedView';
