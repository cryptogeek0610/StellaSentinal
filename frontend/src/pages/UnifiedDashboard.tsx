/**
 * Unified Dashboard - The Single Dashboard Experience
 *
 * This is the ONE AND ONLY dashboard of the application.
 * It merges Command Center, Operational Details, and AI Insights
 * into a single, cohesive, intentional experience.
 *
 * Design Philosophy:
 * - Everything lives on one scrollable page
 * - Progressive disclosure via collapsible sections
 * - Drill-down via slide-over panels (not page navigation)
 * - AI insights embedded contextually within operational data
 * - No tabs that simulate multiple dashboards
 *
 * Information Hierarchy:
 * 1. System Pulse - KPIs and critical alerts (always visible)
 * 2. Intelligence Center - AI summary and priority issues
 * 3. Operational Domains - Expandable sections for each area
 * 4. System & Automation - Pipeline status and activity
 */

import { useState, useMemo, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { Link, useNavigate } from 'react-router-dom';
import { format, formatDistanceToNowStrict } from 'date-fns';
import clsx from 'clsx';

import { api } from '../api/client';
import { KPICard } from '../components/KPICard';
import { Card } from '../components/Card';
import { QueryState } from '../components/ui';
import {
  InvestigationsSection,
  ShiftReadinessSection,
  NetworkSection,
  DeviceHealthSection,
  SlideOverPanel,
} from '../components/unified';
import { LiveAlerts } from '../components/streaming/LiveAlerts';
import { SystemicIssuesCard } from '../components/unified-dashboard/SystemicIssuesCard';
import { CorrelationInsightsCard } from '../components/unified-dashboard/CorrelationInsightsCard';
import { PredictiveWarningsCard } from '../components/unified-dashboard/PredictiveWarningsCard';
import { DataQualityIndicator } from '../components/DataQualityIndicator';
import type { Anomaly } from '../types/anomaly';
import type { CustomerInsightResponse } from '../api/client';

// ============================================================================
// Data Fetching Hook
// ============================================================================

function useUnifiedDashboardData() {
  const [selectedLocation, setSelectedLocation] = useState('store-001');

  // Core operational stats
  const {
    data: stats,
    isLoading: statsLoading,
    isError: statsError,
    error: statsErrorDetail,
    refetch: refetchStats,
  } = useQuery({
    queryKey: ['dashboard', 'stats'],
    queryFn: () => api.getDashboardStats(),
    refetchInterval: 15000,
    staleTime: 10000,
  });

  // AI Daily Digest
  const { data: digest, isLoading: digestLoading } = useQuery({
    queryKey: ['dashboard', 'digest'],
    queryFn: () => api.getDailyDigest({ period_days: 7 }),
    refetchInterval: 60000,
    staleTime: 30000,
  });

  // System connections
  const { data: connections, isLoading: connectionsLoading } = useQuery({
    queryKey: ['dashboard', 'connections'],
    queryFn: () => api.getConnectionStatus(),
    refetchInterval: 30000,
    staleTime: 20000,
  });

  // Open anomalies
  const { data: openAnomalies, isLoading: anomaliesLoading } = useQuery({
    queryKey: ['dashboard', 'anomalies-open'],
    queryFn: () => api.getAnomalies({ page_size: 20, status: 'open' }),
    refetchInterval: 15000,
    staleTime: 10000,
  });

  // Recent resolved
  const { data: resolvedAnomalies, isLoading: resolvedLoading } = useQuery({
    queryKey: ['dashboard', 'anomalies-resolved'],
    queryFn: () => api.getAnomalies({ page_size: 10, status: 'resolved' }),
    refetchInterval: 30000,
    staleTime: 20000,
  });

  // ML Training status
  const { data: training, isLoading: trainingLoading } = useQuery({
    queryKey: ['dashboard', 'training'],
    queryFn: () => api.getTrainingStatus(),
    refetchInterval: 10000,
    staleTime: 5000,
  });

  // Automation/Scheduler status
  const { data: automation, isLoading: automationLoading } = useQuery({
    queryKey: ['dashboard', 'automation'],
    queryFn: () => api.getAutomationStatus(),
    refetchInterval: 30000,
    staleTime: 20000,
  });

  // Shift readiness
  const { data: shiftReadiness, isLoading: shiftLoading } = useQuery({
    queryKey: ['dashboard', 'shift-readiness', selectedLocation],
    queryFn: () => api.getShiftReadiness(selectedLocation),
    refetchInterval: 60000,
    staleTime: 30000,
  });

  // Network analysis
  const { data: network, isLoading: networkLoading } = useQuery({
    queryKey: ['dashboard', 'network'],
    queryFn: () => api.getNetworkAnalysis(),
    refetchInterval: 60000,
    staleTime: 30000,
  });

  // Device abuse/health
  const { data: deviceAbuse, isLoading: deviceLoading } = useQuery({
    queryKey: ['dashboard', 'device-abuse'],
    queryFn: () => api.getDeviceAbuseAnalysis(),
    refetchInterval: 60000,
    staleTime: 30000,
  });

  // Trends data
  const { data: trends } = useQuery({
    queryKey: ['dashboard', 'trends'],
    queryFn: () => api.getDashboardTrends({ days: 7 }),
    refetchInterval: 60000,
    staleTime: 30000,
  });

  return {
    stats,
    digest,
    connections,
    openAnomalies,
    resolvedAnomalies,
    training,
    automation,
    shiftReadiness,
    network,
    deviceAbuse,
    trends,
    selectedLocation,
    setSelectedLocation,
    isLoading: {
      stats: statsLoading,
      digest: digestLoading,
      connections: connectionsLoading,
      anomalies: anomaliesLoading,
      resolved: resolvedLoading,
      training: trainingLoading,
      automation: automationLoading,
      shift: shiftLoading,
      network: networkLoading,
      device: deviceLoading,
    },
    errors: {
      stats: statsError,
      statsDetail: statsErrorDetail as Error | null,
    },
    refetch: {
      stats: refetchStats,
    },
  };
}

// ============================================================================
// Helper Components
// ============================================================================

function SystemHealthIndicator({
  connections,
  isLoading,
}: {
  connections: Awaited<ReturnType<typeof api.getConnectionStatus>> | undefined;
  isLoading: boolean;
}) {
  if (isLoading || !connections) {
    return (
      <div className="animate-pulse flex items-center gap-2">
        <div className="w-3 h-3 rounded-full bg-slate-600" />
        <div className="h-4 w-20 bg-slate-700 rounded" />
      </div>
    );
  }

  const services = [
    connections.backend_db,
    connections.dw_sql,
    connections.mc_sql,
    connections.mobicontrol_api,
    connections.redis,
    connections.qdrant,
    connections.llm,
  ];
  const connectedCount = services.filter((s) => s.status === 'connected').length;
  const totalCount = services.length;
  const healthPercent = (connectedCount / totalCount) * 100;

  let statusColor = 'bg-emerald-500';
  let statusText = 'Healthy';
  if (healthPercent < 50) {
    statusColor = 'bg-red-500';
    statusText = 'Critical';
  } else if (healthPercent < 100) {
    statusColor = 'bg-amber-500';
    statusText = 'Degraded';
  }

  return (
    <Link
      to="/status"
      className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800/50 border border-slate-700/50 hover:border-slate-600 transition-colors"
    >
      <div className={clsx('w-2.5 h-2.5 rounded-full', statusColor)} />
      <span className="text-sm text-slate-300">
        {statusText} ({connectedCount}/{totalCount})
      </span>
    </Link>
  );
}

function PipelineStatus({
  training,
  automation,
  isLoading,
}: {
  training: Awaited<ReturnType<typeof api.getTrainingStatus>> | undefined;
  automation: Awaited<ReturnType<typeof api.getAutomationStatus>> | undefined;
  isLoading: boolean;
}) {
  if (isLoading) {
    return (
      <div className="animate-pulse flex items-center gap-2">
        <div className="w-3 h-3 rounded-full bg-slate-600" />
        <div className="h-4 w-24 bg-slate-700 rounded" />
      </div>
    );
  }

  const isTraining = training?.status === 'running';
  const lastRun = automation?.last_scoring_result?.timestamp;

  return (
    <Link
      to="/automation"
      className="flex items-center gap-3 px-3 py-1.5 rounded-lg bg-slate-800/50 border border-slate-700/50 hover:border-slate-600 transition-colors"
    >
      {isTraining ? (
        <>
          <div className="w-2.5 h-2.5 rounded-full bg-amber-500 animate-pulse" />
          <span className="text-sm text-amber-400">
            Training {training?.progress ? `${training.progress}%` : '...'}
          </span>
        </>
      ) : (
        <>
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
          <span className="text-sm text-slate-300">ML Active</span>
        </>
      )}
      {lastRun && (
        <span className="text-xs text-slate-500">
          Last: {formatDistanceToNowStrict(new Date(lastRun), { addSuffix: true })}
        </span>
      )}
    </Link>
  );
}

function ActivityFeed({
  anomalies,
  resolved,
  training,
  isLoading,
}: {
  anomalies: Anomaly[] | undefined;
  resolved: Anomaly[] | undefined;
  training: Awaited<ReturnType<typeof api.getTrainingStatus>> | undefined;
  isLoading: boolean;
}) {
  const items = useMemo(() => {
    const feed: Array<{
      id: string;
      type: 'anomaly' | 'resolved' | 'training';
      timestamp: Date;
      content: string;
      link?: string;
    }> = [];

    // Add anomalies
    anomalies?.slice(0, 5).forEach((a) => {
      feed.push({
        id: `anomaly-${a.id}`,
        type: 'anomaly',
        timestamp: new Date(a.timestamp),
        content: `Anomaly detected on ${a.device_name || `Device #${a.device_id}`}`,
        link: `/investigations/${a.id}`,
      });
    });

    // Add resolved
    resolved?.slice(0, 3).forEach((a) => {
      feed.push({
        id: `resolved-${a.id}`,
        type: 'resolved',
        timestamp: new Date(a.updated_at),
        content: `Investigation #${a.id} resolved`,
        link: `/investigations/${a.id}`,
      });
    });

    // Add training if running
    if (training?.status === 'running' && training.started_at) {
      feed.push({
        id: 'training',
        type: 'training',
        timestamp: new Date(training.started_at),
        content: `Model training in progress (${training.progress || 0}%)`,
        link: '/training',
      });
    }

    // Sort by timestamp descending
    return feed.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime()).slice(0, 6);
  }, [anomalies, resolved, training]);

  if (isLoading) {
    return (
      <div className="space-y-2">
        {[1, 2, 3].map((i) => (
          <div key={i} className="flex items-center gap-3 p-2 animate-pulse">
            <div className="w-2 h-2 rounded-full bg-slate-700" />
            <div className="h-4 flex-1 bg-slate-700 rounded" />
          </div>
        ))}
      </div>
    );
  }

  if (items.length === 0) {
    return (
      <div className="flex items-center justify-center py-6 text-slate-500 text-sm">
        No recent activity
      </div>
    );
  }

  const typeStyles = {
    anomaly: { dot: 'bg-red-500', text: 'text-red-400' },
    resolved: { dot: 'bg-emerald-500', text: 'text-emerald-400' },
    training: { dot: 'bg-amber-500', text: 'text-amber-400' },
  };

  return (
    <div className="space-y-1">
      {items.map((item) => {
        const style = typeStyles[item.type];
        const Component = item.link ? Link : 'div';
        return (
          <Component
            key={item.id}
            to={item.link || '#'}
            className={clsx(
              'flex items-center gap-3 p-2 rounded-lg transition-colors',
              item.link && 'hover:bg-slate-800/50 cursor-pointer'
            )}
          >
            <div className={clsx('w-2 h-2 rounded-full flex-shrink-0', style.dot)} />
            <span className="text-sm text-slate-300 flex-1 truncate">{item.content}</span>
            <span className="text-xs text-slate-600">
              {formatDistanceToNowStrict(item.timestamp, { addSuffix: true })}
            </span>
          </Component>
        );
      })}
    </div>
  );
}

function CriticalAlertBanner({
  count,
  onDismiss,
}: {
  count: number;
  onDismiss: () => void;
}) {
  if (count === 0) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-xl bg-gradient-to-r from-red-900/40 to-red-800/20 border border-red-500/30 p-4"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse" />
            <div className="absolute inset-0 w-3 h-3 rounded-full bg-red-500 animate-ping" />
          </div>
          <div>
            <span className="text-red-400 font-semibold">
              {count} Critical Issue{count > 1 ? 's' : ''} Requiring Attention
            </span>
            <p className="text-sm text-slate-400 mt-0.5">
              These anomalies have high severity scores and should be investigated immediately.
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Link
            to="/investigations?severity=critical"
            className="px-4 py-2 rounded-lg bg-red-500/20 text-red-400 text-sm font-medium hover:bg-red-500/30 transition-colors"
          >
            View All
          </Link>
          <button
            onClick={onDismiss}
            className="p-2 rounded-lg hover:bg-slate-800/50 text-slate-500 hover:text-slate-400 transition-colors"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>
    </motion.div>
  );
}

function ExecutiveSummaryCard({
  digest,
  isLoading,
}: {
  digest: Awaited<ReturnType<typeof api.getDailyDigest>> | undefined;
  isLoading: boolean;
}) {
  const [selectedInsight, setSelectedInsight] = useState<CustomerInsightResponse | null>(null);

  if (isLoading) {
    return (
      <Card title="AI Intelligence Briefing" accent="plasma">
        <div className="space-y-3 animate-pulse">
          <div className="h-4 bg-slate-700 rounded w-3/4" />
          <div className="h-4 bg-slate-700 rounded w-full" />
          <div className="h-4 bg-slate-700 rounded w-2/3" />
        </div>
      </Card>
    );
  }

  if (!digest?.executive_summary) {
    return (
      <Card title="AI Intelligence Briefing" accent="plasma">
        <div className="flex items-center justify-center py-8 text-slate-500">
          No intelligence briefing available
        </div>
      </Card>
    );
  }

  const priorityIssues = digest.top_insights?.filter(
    (i) => i.severity === 'critical' || i.severity === 'high'
  ) || [];

  const severityStyles = {
    critical: { bg: 'bg-red-500/10', border: 'border-red-500/20', text: 'text-red-400' },
    high: { bg: 'bg-orange-500/10', border: 'border-orange-500/20', text: 'text-orange-400' },
    medium: { bg: 'bg-amber-500/10', border: 'border-amber-500/20', text: 'text-amber-400' },
    low: { bg: 'bg-slate-700/30', border: 'border-slate-600/30', text: 'text-slate-400' },
    info: { bg: 'bg-blue-500/10', border: 'border-blue-500/20', text: 'text-blue-400' },
  };

  const trendStyles = {
    improving: { icon: '↗', color: 'text-emerald-400' },
    stable: { icon: '→', color: 'text-slate-400' },
    degrading: { icon: '↘', color: 'text-red-400' },
  };

  return (
    <>
      <Card title="AI Intelligence Briefing" accent="plasma">
        <div className="space-y-4">
          {/* Summary text */}
          <div className="p-4 rounded-lg bg-gradient-to-br from-purple-500/5 to-blue-500/5 border border-purple-500/20">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-purple-500/20 flex items-center justify-center flex-shrink-0">
                <svg className="w-4 h-4 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <div>
                <p className="text-slate-300 text-sm leading-relaxed">
                  {digest.executive_summary}
                </p>
                <div className="flex items-center gap-4 mt-3 text-xs text-slate-500">
                  <span>{digest.total_insights} insights generated</span>
                  <span>|</span>
                  <span>{digest.critical_count} critical</span>
                  <span>|</span>
                  <span>{digest.high_count} high priority</span>
                </div>
              </div>
            </div>
          </div>

          {/* Priority issues */}
          {priorityIssues.length > 0 && (
            <div>
              <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wide mb-2">
                Priority Issues
              </h4>
              <div className="space-y-2">
                {priorityIssues.slice(0, 3).map((issue, i) => {
                  const severityStyle = issue.severity === 'critical'
                    ? 'bg-red-500/10 border-red-500/20 text-red-400'
                    : 'bg-orange-500/10 border-orange-500/20 text-orange-400';

                  return (
                    <button
                      key={issue.insight_id}
                      onClick={() => setSelectedInsight(issue)}
                      className="w-full flex items-start gap-3 p-3 rounded-lg bg-slate-800/30 border border-slate-700/30 hover:bg-slate-800/50 hover:border-slate-600/50 transition-all text-left group"
                    >
                      <span className={clsx(
                        'px-2 py-0.5 rounded text-xs font-medium',
                        severityStyle
                      )}>
                        {i + 1}
                      </span>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm text-white font-medium truncate group-hover:text-purple-300 transition-colors">{issue.headline}</p>
                        <p className="text-xs text-slate-500 mt-0.5 truncate">{issue.impact_statement}</p>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-slate-600">
                          {issue.affected_device_count} devices
                        </span>
                        <svg className="w-4 h-4 text-slate-600 group-hover:text-slate-400 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* View all link */}
          {digest.total_insights > 3 && (
            <div className="pt-2 border-t border-slate-700/30">
              <Link
                to="/insights"
                className="text-sm text-purple-400 hover:text-purple-300 transition-colors"
              >
                View all {digest.total_insights} insights
              </Link>
            </div>
          )}
        </div>
      </Card>

      {/* Insight Detail Panel */}
      <SlideOverPanel
        isOpen={!!selectedInsight}
        onClose={() => setSelectedInsight(null)}
        title={selectedInsight?.headline || 'Insight Details'}
        subtitle={selectedInsight?.category?.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}
        width="md"
        footer={
          selectedInsight && (
            <div className="flex gap-2">
              <Link
                to={`/insights?category=${selectedInsight.category}`}
                className="flex-1 py-2 px-4 rounded-lg bg-purple-500 text-white text-sm font-medium text-center hover:bg-purple-400 transition-colors"
              >
                View Related Insights
              </Link>
              <button
                onClick={() => setSelectedInsight(null)}
                className="py-2 px-4 rounded-lg bg-slate-700 text-slate-300 text-sm font-medium hover:bg-slate-600 transition-colors"
              >
                Close
              </button>
            </div>
          )
        }
      >
        {selectedInsight && (
          <div className="space-y-4">
            {/* Severity & Status */}
            <div className="flex items-center gap-3">
              <span className={clsx(
                'px-3 py-1 rounded-full text-xs font-bold uppercase',
                severityStyles[selectedInsight.severity as keyof typeof severityStyles]?.bg,
                severityStyles[selectedInsight.severity as keyof typeof severityStyles]?.text
              )}>
                {selectedInsight.severity}
              </span>
              <span className={clsx(
                'flex items-center gap-1 text-sm',
                trendStyles[selectedInsight.trend_direction as keyof typeof trendStyles]?.color
              )}>
                <span>{trendStyles[selectedInsight.trend_direction as keyof typeof trendStyles]?.icon}</span>
                {selectedInsight.trend_direction}
                {selectedInsight.trend_change_percent != null && (
                  <span className="text-slate-500 ml-1">
                    ({selectedInsight.trend_change_percent > 0 ? '+' : ''}{selectedInsight.trend_change_percent.toFixed(1)}%)
                  </span>
                )}
              </span>
            </div>

            {/* Impact Statement */}
            <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/30">
              <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wide mb-2">Impact</h4>
              <p className="text-white text-sm leading-relaxed">{selectedInsight.impact_statement}</p>
            </div>

            {/* Key Metrics */}
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/30">
                <div className="text-xs text-slate-500 mb-1">Affected Devices</div>
                <div className="text-2xl font-bold font-mono text-white">{selectedInsight.affected_device_count}</div>
              </div>
              <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/30">
                <div className="text-xs text-slate-500 mb-1">Confidence</div>
                <div className="text-2xl font-bold font-mono text-purple-400">{(selectedInsight.confidence_score * 100).toFixed(0)}%</div>
              </div>
            </div>

            {/* Primary Metric */}
            {selectedInsight.primary_metric && (
              <div className="p-3 rounded-lg bg-purple-500/10 border border-purple-500/20">
                <div className="text-xs text-slate-400 mb-1">{selectedInsight.primary_metric}</div>
                <div className="text-xl font-bold font-mono text-purple-400">
                  {typeof selectedInsight.primary_value === 'number'
                    ? selectedInsight.primary_value.toLocaleString()
                    : selectedInsight.primary_value}
                </div>
              </div>
            )}

            {/* Comparison Context */}
            {selectedInsight.comparison_context && (
              <div className="p-3 rounded-lg bg-slate-800/30 border border-slate-700/30">
                <div className="flex items-start gap-2">
                  <svg className="w-4 h-4 text-slate-400 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <p className="text-sm text-slate-400">{selectedInsight.comparison_context}</p>
                </div>
              </div>
            )}

            {/* Entity Info */}
            {selectedInsight.entity_name && (
              <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/30">
                <div className="text-xs text-slate-500 mb-1">{selectedInsight.entity_type?.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}</div>
                <div className="text-white font-medium">{selectedInsight.entity_name}</div>
              </div>
            )}

            {/* Recommended Actions */}
            {selectedInsight.recommended_actions && selectedInsight.recommended_actions.length > 0 && (
              <div>
                <h4 className="text-xs font-medium text-emerald-400 uppercase tracking-wide mb-2">
                  Recommended Actions
                </h4>
                <ul className="space-y-2">
                  {selectedInsight.recommended_actions.map((action: string, i: number) => (
                    <li key={i} className="flex items-start gap-3 p-2 rounded-lg bg-emerald-500/5 border border-emerald-500/20">
                      <span className="w-5 h-5 rounded-full bg-emerald-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                        <span className="text-xs font-bold text-emerald-400">{i + 1}</span>
                      </span>
                      <span className="text-sm text-slate-300">{action}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Detected At */}
            <div className="text-xs text-slate-600 text-center pt-2 border-t border-slate-700/30">
              Detected {formatDistanceToNowStrict(new Date(selectedInsight.detected_at), { addSuffix: true })}
            </div>
          </div>
        )}
      </SlideOverPanel>
    </>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export default function UnifiedDashboard() {
  const navigate = useNavigate();
  const data = useUnifiedDashboardData();
  const [alertDismissed, setAlertDismissed] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date());

  // Live clock update every second
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const criticalCount = data.stats?.critical_issues || 0;
  const showCriticalAlert = criticalCount > 0 && !alertDismissed;

  return (
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      {/* ================================================================== */}
      {/* LEVEL 1: SYSTEM PULSE - Header and KPIs                           */}
      {/* ================================================================== */}

      {/* Header Bar */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Operations Dashboard</h1>
          <p className="text-slate-500 text-sm mt-1">
            Real-time operational intelligence with AI-powered insights
          </p>
        </div>
        <div className="flex items-center gap-3">
          {/* Data Quality */}
          <DataQualityIndicator showTooltip={true} size="sm" />
          {/* System Health */}
          <SystemHealthIndicator
            connections={data.connections}
            isLoading={data.isLoading.connections}
          />
          {/* Pipeline Status */}
          <PipelineStatus
            training={data.training}
            automation={data.automation}
            isLoading={data.isLoading.training || data.isLoading.automation}
          />
          {/* Live Time */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800/50 border border-slate-700/50">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-sm text-slate-300 font-mono">
              {format(currentTime, 'HH:mm:ss')}
            </span>
          </div>
        </div>
      </div>

      {/* Critical Alert Banner */}
      {showCriticalAlert && (
        <CriticalAlertBanner
          count={criticalCount}
          onDismiss={() => setAlertDismissed(true)}
        />
      )}

      {/* Primary KPI Row */}
      <QueryState
        isLoading={data.isLoading.stats}
        isError={data.errors.stats}
        error={data.errors.statsDetail}
        isEmpty={!data.stats}
        emptyMessage="No dashboard data available yet."
        onRetry={data.refetch.stats}
      >
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
          <KPICard
            title="Open Cases"
            value={data.stats?.open_cases ?? '—'}
            color={data.stats?.open_cases && data.stats.open_cases > 0 ? 'warning' : 'stellar'}
            showProgress={false}
            onClick={() => window.location.href = '/investigations?status=open'}
            explainer="Active investigations requiring attention"
          />
          <KPICard
            title="Critical"
            value={criticalCount}
            color="danger"
            showProgress={false}
            isActive={criticalCount > 0}
            onClick={() => window.location.href = '/investigations?severity=critical'}
            explainer="High-severity issues needing immediate action"
          />
          <KPICard
            title="AI Insights"
            value={data.digest?.total_insights ?? '—'}
            color="plasma"
            showProgress={false}
            onClick={() => window.location.href = '/insights'}
            explainer="AI-generated intelligence signals today"
          />
          <KPICard
            title="Resolved Today"
            value={data.stats?.resolved_today ?? '—'}
            color="aurora"
            showProgress={false}
            onClick={() => window.location.href = '/investigations?status=resolved'}
            explainer="Issues closed in the last 24 hours"
          />
          <KPICard
            title="Fleet Size"
            value={data.stats?.devices_monitored ?? '—'}
            color="stellar"
            showProgress={false}
            onClick={() => window.location.href = '/fleet'}
            explainer="Total devices under monitoring"
          />
        </div>
      </QueryState>

      {/* ================================================================== */}
      {/* LEVEL 2: INTELLIGENCE CENTER                                      */}
      {/* ================================================================== */}

      <ExecutiveSummaryCard
        digest={data.digest}
        isLoading={data.isLoading.digest}
      />

      {/* Intelligence Cards Grid - Cross-Device, Correlations, Predictions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <SystemicIssuesCard maxIssues={4} />
        <CorrelationInsightsCard maxInsights={3} />
        <PredictiveWarningsCard maxPredictions={3} />
      </div>

      {/* ================================================================== */}
      {/* LEVEL 3: OPERATIONAL DOMAINS - Expandable Sections                */}
      {/* ================================================================== */}

      <div className="space-y-4">
        {/* Investigations Section */}
        <InvestigationsSection
          anomalies={data.openAnomalies?.anomalies || []}
          stats={data.stats ? {
            open_cases: data.stats.open_cases || 0,
            critical_issues: data.stats.critical_issues || 0,
            detected_today: data.stats.anomalies_today || 0,
            resolved_today: data.stats.resolved_today || 0,
          } : undefined}
          isLoading={data.isLoading.anomalies}
        />

        {/* Shift Readiness Section */}
        <ShiftReadinessSection
          data={data.shiftReadiness}
          isLoading={data.isLoading.shift}
          selectedLocation={data.selectedLocation}
          onLocationChange={data.setSelectedLocation}
        />

        {/* Network Health Section */}
        <NetworkSection
          data={data.network}
          isLoading={data.isLoading.network}
        />

        {/* Device Health Section */}
        <DeviceHealthSection
          data={data.deviceAbuse}
          isLoading={data.isLoading.device}
        />
      </div>

      {/* ================================================================== */}
      {/* LEVEL 4: SYSTEM & AUTOMATION                                      */}
      {/* ================================================================== */}

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* System Services */}
        <Card title="System Services" accent="stellar">
          <ServiceHealthGrid connections={data.connections} isLoading={data.isLoading.connections} />
        </Card>

        {/* ML Pipeline */}
        <Card title="ML Pipeline" accent="aurora">
          <PipelineDetails
            training={data.training}
            automation={data.automation}
            isLoading={data.isLoading.training || data.isLoading.automation}
          />
        </Card>

        {/* Recent Activity */}
        <Card title="Recent Activity" accent="stellar">
          <ActivityFeed
            anomalies={data.openAnomalies?.anomalies}
            resolved={data.resolvedAnomalies?.anomalies}
            training={data.training}
            isLoading={data.isLoading.anomalies || data.isLoading.resolved}
          />
        </Card>

        <LiveAlerts onAlertClick={(alert) => navigate(`/investigations/${alert.id}`)} />
      </div>
    </motion.div>
  );
}

// ============================================================================
// Additional Helper Components
// ============================================================================

function ServiceHealthGrid({
  connections,
  isLoading,
}: {
  connections: Awaited<ReturnType<typeof api.getConnectionStatus>> | undefined;
  isLoading: boolean;
}) {
  if (isLoading || !connections) {
    return (
      <div className="grid grid-cols-4 gap-2">
        {[1, 2, 3, 4, 5, 6, 7].map((i) => (
          <div key={i} className="p-2 rounded bg-slate-800/30 animate-pulse">
            <div className="h-3 w-12 bg-slate-700 rounded mb-1" />
            <div className="w-2 h-2 rounded-full bg-slate-700" />
          </div>
        ))}
      </div>
    );
  }

  const services = [
    { key: 'backend_db', name: 'PostgreSQL', data: connections.backend_db },
    { key: 'redis', name: 'Redis', data: connections.redis },
    { key: 'qdrant', name: 'Qdrant', data: connections.qdrant },
    { key: 'dw_sql', name: 'XSight', data: connections.dw_sql },
    { key: 'mc_sql', name: 'MC SQL', data: connections.mc_sql },
    { key: 'mobicontrol_api', name: 'MC API', data: connections.mobicontrol_api },
    { key: 'llm', name: 'LLM', data: connections.llm },
  ];

  return (
    <div className="grid grid-cols-4 gap-2">
      {services.map((service) => {
        const isConnected = service.data.status === 'connected';
        return (
          <div
            key={service.key}
            className={clsx(
              'p-2 rounded border transition-colors',
              isConnected
                ? 'bg-emerald-500/5 border-emerald-500/20'
                : 'bg-red-500/5 border-red-500/20'
            )}
          >
            <div className="text-xs text-slate-400 truncate">{service.name}</div>
            <div className="flex items-center gap-1 mt-1">
              <div className={clsx(
                'w-2 h-2 rounded-full',
                isConnected ? 'bg-emerald-500' : 'bg-red-500'
              )} />
              <span className={clsx(
                'text-xs',
                isConnected ? 'text-emerald-400' : 'text-red-400'
              )}>
                {isConnected ? 'OK' : 'Down'}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function PipelineDetails({
  training,
  automation,
  isLoading,
}: {
  training: Awaited<ReturnType<typeof api.getTrainingStatus>> | undefined;
  automation: Awaited<ReturnType<typeof api.getAutomationStatus>> | undefined;
  isLoading: boolean;
}) {
  if (isLoading) {
    return (
      <div className="space-y-3 animate-pulse">
        <div className="h-4 bg-slate-700 rounded w-full" />
        <div className="h-4 bg-slate-700 rounded w-3/4" />
        <div className="h-4 bg-slate-700 rounded w-1/2" />
      </div>
    );
  }

  const isTraining = training?.status === 'running';
  const lastScoring = automation?.last_scoring_result;
  const nextScoring = automation?.next_scoring_time;

  return (
    <div className="space-y-3">
      {/* Training Status */}
      <div className="flex items-center justify-between">
        <span className="text-xs text-slate-500">Training</span>
        <div className="flex items-center gap-2">
          {isTraining ? (
            <>
              <div className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
              <span className="text-sm text-amber-400">
                Running {training?.progress ? `(${training.progress}%)` : ''}
              </span>
            </>
          ) : (
            <>
              <div className="w-2 h-2 rounded-full bg-slate-500" />
              <span className="text-sm text-slate-400">Idle</span>
            </>
          )}
        </div>
      </div>

      {/* Last Scoring */}
      <div className="flex items-center justify-between">
        <span className="text-xs text-slate-500">Last Score Run</span>
        <span className="text-sm text-slate-300">
          {lastScoring?.timestamp
            ? formatDistanceToNowStrict(new Date(lastScoring.timestamp), { addSuffix: true })
            : '—'}
        </span>
      </div>

      {/* Last Scoring Results */}
      {lastScoring && (
        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-600">
            {lastScoring.total_scored} scored
          </span>
          <span className="text-slate-600">
            {lastScoring.anomalies_detected} anomalies ({(lastScoring.anomaly_rate * 100).toFixed(1)}%)
          </span>
        </div>
      )}

      {/* Next Run */}
      <div className="flex items-center justify-between">
        <span className="text-xs text-slate-500">Next Score Run</span>
        <span className="text-sm text-slate-300">
          {nextScoring
            ? format(new Date(nextScoring), 'HH:mm')
            : '—'}
        </span>
      </div>

      {/* Actions */}
      <div className="pt-2 border-t border-slate-700/30">
        <Link
          to="/automation"
          className="text-xs text-emerald-400 hover:text-emerald-300 transition-colors"
        >
          Configure Automation
        </Link>
      </div>
    </div>
  );
}
