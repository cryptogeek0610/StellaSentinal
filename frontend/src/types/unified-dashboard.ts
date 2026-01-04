/**
 * Types for the Unified Command Center Dashboard
 *
 * These types define the data structures used exclusively by the unified dashboard
 * for aggregating and displaying high-level summaries from multiple data sources.
 */

import type { CustomerInsight } from './insights';
import type { ConnectionStatus } from './anomaly';

/**
 * Aggregated operational summary for the unified dashboard
 */
export interface OperationalSummary {
  openCases: number;
  criticalCount: number;
  detectedToday: number;
  resolvedToday: number;
  systemHealth: SystemHealthStatus;
  fleetSize: number;
  aiInsightsCount: number;
}

/**
 * System health aggregation from all connected services
 */
export interface SystemHealthStatus {
  connectedCount: number;
  totalCount: number;
  overallStatus: 'healthy' | 'degraded' | 'critical';
  services: ServiceHealthItem[];
}

export interface ServiceHealthItem {
  name: string;
  status: ConnectionStatus['status'];
  displayName: string;
}

/**
 * Intelligence briefing data for the unified dashboard
 */
export interface IntelligenceBriefing {
  executiveSummary: string;
  digestDate: string;
  totalInsights: number;
  criticalInsights: number;
  highInsights: number;
  priorityIssues: CustomerInsight[];
  trendingIssues: CustomerInsight[];
}

/**
 * Quick access navigation card configuration
 */
export interface NavigationCardConfig {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  route: string;
  queryParams?: Record<string, string>;
  metric?: string | number;
  metricLabel?: string;
  accent: 'stellar' | 'aurora' | 'plasma' | 'danger' | 'warning';
  badge?: {
    value: number;
    variant: 'default' | 'warning' | 'danger';
  };
}

/**
 * Compact activity feed item for recent events
 */
export interface CompactActivityItem {
  id: string;
  type: 'anomaly_detected' | 'anomaly_resolved' | 'training_complete' | 'system_event' | 'insight_generated';
  title: string;
  description?: string;
  timestamp: string;
  severity?: 'critical' | 'high' | 'medium' | 'low' | 'info';
  linkTo?: string;
  linkParams?: Record<string, string>;
}

/**
 * Pipeline status summary for ML operations
 */
export interface PipelineStatusSummary {
  lastScoringRun: string | null;
  nextScoringRun: string | null;
  automationEnabled: boolean;
  isTraining: boolean;
  trainingProgress?: number;
  modelHealth: 'healthy' | 'needs_attention' | 'stale';
}

/**
 * Props for deep-linkable insight cards
 */
export interface InsightCardDeepLinkProps {
  insight: CustomerInsight;
  onClick: () => void;
  compact?: boolean;
}

/**
 * Tab configuration for insights deep-linking
 */
export type InsightsTab = 'digest' | 'battery' | 'network' | 'devices' | 'apps';

/**
 * Utility function to build insight deep-link URL
 */
export function buildInsightDeepLink(
  insightId?: string,
  tab?: InsightsTab
): string {
  const params = new URLSearchParams();
  if (tab) params.set('tab', tab);
  if (insightId) params.set('highlight', insightId);
  const queryString = params.toString();
  return `/insights${queryString ? `?${queryString}` : ''}`;
}

/**
 * Utility function to build investigations deep-link URL
 */
export function buildInvestigationsDeepLink(
  filters?: {
    status?: string;
    severity?: string;
    deviceId?: string;
  }
): string {
  const params = new URLSearchParams();
  if (filters?.status) params.set('status', filters.status);
  if (filters?.severity) params.set('severity', filters.severity);
  if (filters?.deviceId) params.set('device', filters.deviceId);
  const queryString = params.toString();
  return `/investigations${queryString ? `?${queryString}` : ''}`;
}
