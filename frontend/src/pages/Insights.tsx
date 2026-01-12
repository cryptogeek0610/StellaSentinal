/**
 * AI Insights Page - Customer-Facing Intelligence
 *
 * Aligned with Carl's vision: "XSight has the data. XSight needs the story."
 * Provides pre-interpreted, contextualized, and actionable insights.
 */

import { useState, useEffect, lazy, Suspense } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { api, CustomerInsightResponse, DailyDigestResponse } from '../api/client';
import { motion } from 'framer-motion';
import { Card } from '../components/Card';
import { KPICard } from '../components/KPICard';
import { SlideOverPanel } from '../components/unified';

// Lazy load tab components for better performance
const SystemHealthTab = lazy(() => import('../components/SystemHealthTab'));
const EventsAlertsTab = lazy(() => import('../components/EventsAlertsTab'));
const LocationIntelligenceTab = lazy(() => import('../components/LocationIntelligenceTab'));
const TemporalAnalysisTab = lazy(() => import('../components/TemporalAnalysisTab'));
const CorrelationsTab = lazy(() => import('../components/CorrelationsTab'));
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie,
} from 'recharts';
import { format } from 'date-fns';
import clsx from 'clsx';

// Severity color mapping
const SEVERITY_COLORS: Record<string, string> = {
  critical: '#ef4444',
  high: '#f97316',
  medium: '#f59e0b',
  low: '#64748b',
  info: '#3b82f6',
};

const SEVERITY_BG: Record<string, string> = {
  critical: 'bg-red-500/20 border-red-500/30',
  high: 'bg-orange-500/20 border-orange-500/30',
  medium: 'bg-amber-500/20 border-amber-500/30',
  low: 'bg-slate-500/20 border-slate-500/30',
  info: 'bg-blue-500/20 border-blue-500/30',
};

const TREND_ICONS: Record<string, string> = {
  degrading: '↗',
  stable: '→',
  improving: '↘',
};

// Category display names
const CATEGORY_LABELS: Record<string, string> = {
  battery_shift_failure: 'Battery Shift Failure',
  battery_rapid_drain: 'Rapid Battery Drain',
  excessive_drops: 'Excessive Drops',
  excessive_reboots: 'Excessive Reboots',
  wifi_ap_hopping: 'WiFi Roaming Issues',
  wifi_dead_zone: 'WiFi Dead Zones',
  app_crash_pattern: 'App Crashes',
  app_power_drain: 'App Power Drain',
  network_disconnect_pattern: 'Network Disconnects',
  device_hidden_pattern: 'Hidden Devices',
};

// Extended tabs for comprehensive intelligence
type TabId = 'digest' | 'devices' | 'correlations' | 'health' | 'events' | 'location' | 'temporal';

const VALID_TABS: TabId[] = ['digest', 'devices', 'correlations', 'health', 'events', 'location', 'temporal'];

function Insights() {
  const [searchParams, setSearchParams] = useSearchParams();
  const tabFromUrl = searchParams.get('tab') as TabId | null;
  const initialTab = tabFromUrl && VALID_TABS.includes(tabFromUrl) ? tabFromUrl : 'digest';

  const [activeTab, setActiveTab] = useState<TabId>(initialTab);

  // Sync tab state with URL params
  useEffect(() => {
    const urlTab = searchParams.get('tab') as TabId | null;
    if (urlTab && VALID_TABS.includes(urlTab) && urlTab !== activeTab) {
      setActiveTab(urlTab);
    }
  }, [searchParams, activeTab]);

  // Update URL when tab changes
  const handleTabChange = (tab: TabId) => {
    setActiveTab(tab);
    if (tab === 'digest') {
      // Remove tab param for default tab to keep URL clean
      searchParams.delete('tab');
    } else {
      searchParams.set('tab', tab);
    }
    setSearchParams(searchParams, { replace: true });
  };

  // Fetch daily digest - auto-refresh every minute for live updates
  const { data: digest, isLoading: digestLoading } = useQuery({
    queryKey: ['insights', 'daily-digest'],
    queryFn: () => api.getDailyDigest({ period_days: 7 }),
    refetchInterval: 60000, // Auto-refresh every minute
  });

  // Fetch device abuse analysis (for Device Health tab)
  const { data: deviceAbuse, isLoading: deviceLoading } = useQuery({
    queryKey: ['insights', 'device-abuse'],
    queryFn: () => api.getDeviceAbuseAnalysis(),
    enabled: activeTab === 'devices',
  });

  // Extended tabs for comprehensive intelligence
  const tabs = [
    {
      id: 'digest',
      label: 'Daily Digest',
      icon: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      ),
    },
    {
      id: 'devices',
      label: 'Device Health',
      icon: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 18h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
        </svg>
      ),
    },
    {
      id: 'correlations',
      label: 'Correlations',
      icon: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
        </svg>
      ),
    },
    {
      id: 'health',
      label: 'System Health',
      icon: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
        </svg>
      ),
    },
    {
      id: 'events',
      label: 'Events & Alerts',
      icon: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
        </svg>
      ),
    },
    {
      id: 'location',
      label: 'Location Intelligence',
      icon: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      ),
    },
    {
      id: 'temporal',
      label: 'Temporal Analysis',
      icon: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      ),
    },
  ] as const;

  return (
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      {/* Header */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">AI Insights</h1>
          <p className="text-slate-500 mt-1">
            Pre-interpreted, contextualized intelligence for your fleet
          </p>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex gap-2 p-1 bg-slate-800/50 rounded-lg border border-slate-700/50 overflow-x-auto">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => handleTabChange(tab.id)}
            className={clsx(
              'flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md transition-all whitespace-nowrap',
              activeTab === tab.id
                ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                : 'text-slate-400 hover:text-white'
            )}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'digest' && (
        <DigestTab digest={digest} isLoading={digestLoading} />
      )}
      {activeTab === 'devices' && (
        <DevicesTab deviceAbuse={deviceAbuse} isLoading={deviceLoading} />
      )}
      {activeTab === 'correlations' && (
        <Suspense fallback={<TabLoadingState message="Loading Correlations..." />}>
          <CorrelationsTab />
        </Suspense>
      )}
      {activeTab === 'health' && (
        <Suspense fallback={<TabLoadingState message="Loading System Health..." />}>
          <SystemHealthTab />
        </Suspense>
      )}
      {activeTab === 'events' && (
        <Suspense fallback={<TabLoadingState message="Loading Events & Alerts..." />}>
          <EventsAlertsTab />
        </Suspense>
      )}
      {activeTab === 'location' && (
        <Suspense fallback={<TabLoadingState message="Loading Location Intelligence..." />}>
          <LocationIntelligenceTab />
        </Suspense>
      )}
      {activeTab === 'temporal' && (
        <Suspense fallback={<TabLoadingState message="Loading Temporal Analysis..." />}>
          <TemporalAnalysisTab />
        </Suspense>
      )}
    </motion.div>
  );
}

// ============================================================================
// Daily Digest Tab
// ============================================================================

type KPIPanelType = 'total' | 'critical' | 'high' | 'digest' | null;

function DigestTab({
  digest,
  isLoading,
}: {
  digest?: DailyDigestResponse;
  isLoading: boolean;
}) {
  const [selectedKPI, setSelectedKPI] = useState<KPIPanelType>(null);

  if (isLoading) {
    return <LoadingState message="Generating daily digest..." />;
  }

  if (!digest) {
    return <EmptyState message="No insights available" />;
  }

  // Prepare severity distribution for chart
  const severityData = [
    { name: 'Critical', value: digest.critical_count, color: SEVERITY_COLORS.critical },
    { name: 'High', value: digest.high_count, color: SEVERITY_COLORS.high },
    { name: 'Medium', value: digest.medium_count, color: SEVERITY_COLORS.medium },
  ];

  // Group insights by category for breakdown
  const insightsByCategory = digest.top_insights.reduce((acc, insight) => {
    const cat = insight.category || 'other';
    if (!acc[cat]) acc[cat] = [];
    acc[cat].push(insight);
    return acc;
  }, {} as Record<string, CustomerInsightResponse[]>);

  // Filter insights by severity
  const criticalInsights = digest.top_insights.filter(i => i.severity === 'critical');
  const highInsights = digest.top_insights.filter(i => i.severity === 'high');

  const getPanelTitle = () => {
    switch (selectedKPI) {
      case 'total': return 'All Insights Breakdown';
      case 'critical': return 'Critical Issues';
      case 'high': return 'High Priority Issues';
      case 'digest': return 'Digest Information';
      default: return '';
    }
  };

  const getPanelSubtitle = () => {
    switch (selectedKPI) {
      case 'total': return `${digest.total_insights} insights by category`;
      case 'critical': return `${digest.critical_count} critical issues requiring immediate attention`;
      case 'high': return `${digest.high_count} high priority issues to address`;
      case 'digest': return `Generated ${format(new Date(digest.generated_at), 'PPpp')}`;
      default: return '';
    }
  };

  return (
    <>
      <div className="space-y-6">
        {/* KPI Cards */}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <KPICard
            title="Total Insights"
            value={digest.total_insights}
            color="stellar"
            icon={
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            }
            onClick={() => setSelectedKPI('total')}
            explainer="Click to see breakdown by category"
          />
          <KPICard
            title="Critical Issues"
            value={digest.critical_count}
            color="danger"
            isActive={digest.critical_count > 0}
            icon={
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            }
            onClick={() => setSelectedKPI('critical')}
            explainer="Click to see critical issues"
          />
          <KPICard
            title="High Priority"
            value={digest.high_count}
            color="warning"
            icon={
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            }
            onClick={() => setSelectedKPI('high')}
            explainer="Click to see high priority issues"
          />
          <KPICard
            title="Digest Date"
            value={format(new Date(digest.digest_date), 'MMM d')}
            color="aurora"
            icon={
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            }
            onClick={() => setSelectedKPI('digest')}
            explainer="Click for digest details"
          />
        </div>

      {/* KPI Detail Panel */}
      <SlideOverPanel
        isOpen={selectedKPI !== null}
        onClose={() => setSelectedKPI(null)}
        title={getPanelTitle()}
        subtitle={getPanelSubtitle()}
        width="md"
      >
        {selectedKPI === 'total' && (
          <div className="space-y-4">
            {/* Category breakdown */}
            <div className="grid grid-cols-2 gap-3 mb-4">
              <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20">
                <div className="text-xs text-slate-500 mb-1">Critical</div>
                <div className="text-2xl font-bold font-mono text-red-400">{digest.critical_count}</div>
              </div>
              <div className="p-3 rounded-lg bg-orange-500/10 border border-orange-500/20">
                <div className="text-xs text-slate-500 mb-1">High</div>
                <div className="text-2xl font-bold font-mono text-orange-400">{digest.high_count}</div>
              </div>
              <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20">
                <div className="text-xs text-slate-500 mb-1">Medium</div>
                <div className="text-2xl font-bold font-mono text-amber-400">{digest.medium_count}</div>
              </div>
              <div className="p-3 rounded-lg bg-slate-700/30 border border-slate-600/30">
                <div className="text-xs text-slate-500 mb-1">Total</div>
                <div className="text-2xl font-bold font-mono text-white">{digest.total_insights}</div>
              </div>
            </div>

            <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wide mb-2">
              By Category
            </h4>
            <div className="space-y-2">
              {Object.entries(insightsByCategory).map(([category, insights]) => (
                <div
                  key={category}
                  className="flex items-center justify-between p-3 rounded-lg bg-slate-800/50 border border-slate-700/30"
                >
                  <span className="text-sm text-white">
                    {CATEGORY_LABELS[category] || category.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}
                  </span>
                  <div className="flex items-center gap-2">
                    <span className="text-lg font-bold font-mono text-amber-400">{insights.length}</span>
                    <span className="text-xs text-slate-500">issues</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {selectedKPI === 'critical' && (
          <div className="space-y-4">
            {criticalInsights.length === 0 ? (
              <div className="p-8 text-center">
                <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-emerald-500/20 flex items-center justify-center">
                  <svg className="w-8 h-8 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <p className="text-emerald-400 font-medium">No Critical Issues</p>
                <p className="text-sm text-slate-500 mt-1">All systems operating normally</p>
              </div>
            ) : (
              <>
                <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 mb-4">
                  <p className="text-sm text-red-300">
                    These issues require immediate attention. Click each to see details and recommended actions.
                  </p>
                </div>
                {criticalInsights.map((insight) => (
                  <InsightDetailCard key={insight.insight_id} insight={insight} />
                ))}
              </>
            )}
          </div>
        )}

        {selectedKPI === 'high' && (
          <div className="space-y-4">
            {highInsights.length === 0 ? (
              <div className="p-8 text-center">
                <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-emerald-500/20 flex items-center justify-center">
                  <svg className="w-8 h-8 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <p className="text-emerald-400 font-medium">No High Priority Issues</p>
                <p className="text-sm text-slate-500 mt-1">All high-priority areas are clear</p>
              </div>
            ) : (
              <>
                <div className="p-3 rounded-lg bg-orange-500/10 border border-orange-500/20 mb-4">
                  <p className="text-sm text-orange-300">
                    High priority issues that should be addressed soon to prevent escalation.
                  </p>
                </div>
                {highInsights.map((insight) => (
                  <InsightDetailCard key={insight.insight_id} insight={insight} />
                ))}
              </>
            )}
          </div>
        )}

        {selectedKPI === 'digest' && (
          <div className="space-y-4">
            <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/30">
              <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wide mb-2">Digest Period</h4>
              <p className="text-white font-medium">{format(new Date(digest.digest_date), 'PPPP')}</p>
            </div>

            <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/30">
              <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wide mb-2">Generated</h4>
              <p className="text-white font-medium">{format(new Date(digest.generated_at), 'PPpp')}</p>
            </div>

            <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/30">
              <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wide mb-2">Tenant</h4>
              <p className="text-white font-medium font-mono">{digest.tenant_id}</p>
            </div>

            <div className="p-4 rounded-lg bg-purple-500/10 border border-purple-500/20">
              <h4 className="text-xs font-medium text-purple-400 uppercase tracking-wide mb-2">Executive Summary</h4>
              <p className="text-slate-300 text-sm leading-relaxed">{digest.executive_summary}</p>
            </div>

            <div className="grid grid-cols-3 gap-3">
              <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-center">
                <div className="text-2xl font-bold font-mono text-emerald-400">{digest.trending_issues.length}</div>
                <div className="text-xs text-slate-500">Trending</div>
              </div>
              <div className="p-3 rounded-lg bg-blue-500/10 border border-blue-500/20 text-center">
                <div className="text-2xl font-bold font-mono text-blue-400">{digest.new_issues.length}</div>
                <div className="text-xs text-slate-500">New Today</div>
              </div>
              <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-center">
                <div className="text-2xl font-bold font-mono text-amber-400">{digest.top_insights.length}</div>
                <div className="text-xs text-slate-500">Top Issues</div>
              </div>
            </div>
          </div>
        )}
      </SlideOverPanel>

      {/* Executive Summary */}
      <Card title="Executive Summary" accent="stellar" glow>
        <p className="text-slate-300 leading-relaxed">{digest.executive_summary}</p>
      </Card>

      {/* Top Insights & Severity Distribution */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
        {/* Top Insights */}
        <Card title="Top Priority Issues" className="xl:col-span-2">
          <div className="space-y-3">
            {digest.top_insights.map((insight, index) => (
              <InsightCard key={insight.insight_id} insight={insight} rank={index + 1} />
            ))}
          </div>
        </Card>

        {/* Severity Distribution */}
        <Card title="Severity Breakdown">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={severityData}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {severityData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(14, 17, 23, 0.95)',
                    border: '1px solid rgba(245, 166, 35, 0.2)',
                    borderRadius: '12px',
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="flex justify-center gap-4 mt-4">
            {severityData.map((item) => (
              <div key={item.name} className="flex items-center gap-2">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: item.color }}
                />
                <span className="text-sm text-slate-400">{item.name}</span>
                <span className="text-sm font-bold text-white">{item.value}</span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Trending & New Issues */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <Card title="Trending Issues" accent="warning">
          <div className="space-y-3">
            {digest.trending_issues.map((insight) => (
              <InsightCard key={insight.insight_id} insight={insight} compact />
            ))}
          </div>
        </Card>

        <Card title="New Issues Today" accent="aurora">
          <div className="space-y-3">
            {digest.new_issues.map((insight) => (
              <InsightCard key={insight.insight_id} insight={insight} compact />
            ))}
          </div>
        </Card>
      </div>
    </div>
    </>
  );
}

// ============================================================================
// Devices Tab
// ============================================================================

function DevicesTab({
  deviceAbuse,
  isLoading,
}: {
  deviceAbuse?: ReturnType<typeof api.getDeviceAbuseAnalysis> extends Promise<infer T> ? T : never;
  isLoading: boolean;
}) {
  if (isLoading) {
    return <LoadingState message="Analyzing device health..." />;
  }

  if (!deviceAbuse) {
    return <EmptyState message="No device data available" />;
  }

  return (
    <div className="space-y-6">
      {/* Device KPIs */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <KPICard
          title="Total Drops"
          value={deviceAbuse.total_drops}
          color="warning"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
            </svg>
          }
        />
        <KPICard
          title="Excessive Drops"
          value={deviceAbuse.devices_with_excessive_drops}
          color="danger"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          }
        />
        <KPICard
          title="Total Reboots"
          value={deviceAbuse.total_reboots}
          color="warning"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          }
        />
        <KPICard
          title="Excessive Reboots"
          value={deviceAbuse.devices_with_excessive_reboots}
          color="danger"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          }
        />
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        {/* Worst Locations */}
        <Card title="Worst Locations (Drops)">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={deviceAbuse.worst_locations} layout="vertical">
                <XAxis type="number" stroke="#475569" fontSize={10} />
                <YAxis
                  dataKey="location_id"
                  type="category"
                  stroke="#475569"
                  fontSize={10}
                  width={80}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(14, 17, 23, 0.95)',
                    border: '1px solid rgba(245, 166, 35, 0.2)',
                    borderRadius: '12px',
                  }}
                />
                <Bar dataKey="drops" fill="#f97316" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Problem Combinations */}
        <Card title="Problem Device Cohorts" accent="danger">
          <div className="space-y-3">
            {deviceAbuse.problem_combinations.map((combo) => (
              <div
                key={combo.cohort_id}
                className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-white">
                    {combo.manufacturer} {combo.model}
                  </span>
                  <span className="text-xs px-2 py-1 rounded-full bg-red-500/20 text-red-400 border border-red-500/30">
                    {combo.severity}
                  </span>
                </div>
                <div className="text-sm text-slate-400">
                  {combo.device_count} devices • {combo.vs_fleet_multiplier.toFixed(1)}x fleet rate
                </div>
                <div className="text-sm text-amber-400 mt-1">
                  Primary issue: {combo.primary_issue.replace(/_/g, ' ')}
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* People with Excessive Drops - Carl's requirement */}
      {deviceAbuse.worst_users && deviceAbuse.worst_users.length > 0 && (
        <Card title="People with Excessive Drops" accent="warning">
          <p className="text-sm text-slate-400 mb-4">
            Users ranked by device drop count. Users with {'>'}2x fleet average are flagged as excessive.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700/50">
                  <th className="text-left py-2 px-3 text-slate-400 font-medium">User</th>
                  <th className="text-right py-2 px-3 text-slate-400 font-medium">Drops</th>
                  <th className="text-right py-2 px-3 text-slate-400 font-medium">Devices</th>
                  <th className="text-right py-2 px-3 text-slate-400 font-medium">Per Device</th>
                  <th className="text-right py-2 px-3 text-slate-400 font-medium">vs Fleet</th>
                  <th className="text-center py-2 px-3 text-slate-400 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {deviceAbuse.worst_users.map((user, index) => (
                  <tr
                    key={user.user_id}
                    className={clsx(
                      'border-b border-slate-800/50',
                      user.is_excessive && 'bg-red-500/5'
                    )}
                  >
                    <td className="py-3 px-3">
                      <div className="flex items-center gap-3">
                        <div
                          className={clsx(
                            'w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold',
                            index === 0
                              ? 'bg-red-500/20 text-red-400'
                              : index === 1
                              ? 'bg-orange-500/20 text-orange-400'
                              : index === 2
                              ? 'bg-amber-500/20 text-amber-400'
                              : 'bg-slate-700/50 text-slate-400'
                          )}
                        >
                          {index + 1}
                        </div>
                        <div>
                          <div className="font-medium text-white">
                            {user.user_name || user.user_id}
                          </div>
                          {user.user_email && (
                            <div className="text-xs text-slate-500">{user.user_email}</div>
                          )}
                        </div>
                      </div>
                    </td>
                    <td className="py-3 px-3 text-right">
                      <span className="font-mono font-bold text-amber-400">{user.total_drops}</span>
                    </td>
                    <td className="py-3 px-3 text-right text-slate-300">{user.device_count}</td>
                    <td className="py-3 px-3 text-right text-slate-300">
                      {user.drops_per_device.toFixed(1)}
                    </td>
                    <td className="py-3 px-3 text-right">
                      <span
                        className={clsx(
                          'font-mono font-bold',
                          user.vs_fleet_multiplier >= 3
                            ? 'text-red-400'
                            : user.vs_fleet_multiplier >= 2
                            ? 'text-orange-400'
                            : 'text-slate-300'
                        )}
                      >
                        {user.vs_fleet_multiplier.toFixed(1)}x
                      </span>
                    </td>
                    <td className="py-3 px-3 text-center">
                      {user.is_excessive ? (
                        <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs bg-red-500/20 text-red-400 border border-red-500/30">
                          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                          </svg>
                          Excessive
                        </span>
                      ) : (
                        <span className="text-xs text-slate-500">Normal</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {deviceAbuse.worst_users.filter(u => u.is_excessive).length > 0 && (
            <div className="mt-4 p-3 rounded-lg bg-amber-500/10 border border-amber-500/20">
              <p className="text-sm text-amber-300">
                <strong>{deviceAbuse.worst_users.filter(u => u.is_excessive).length} user(s)</strong> have
                significantly higher drop rates than fleet average. Consider targeted device handling training.
              </p>
            </div>
          )}
        </Card>
      )}

      {/* Financial Impact */}
      {deviceAbuse.financial_impact && (
        <FinancialImpactCard
          impact={deviceAbuse.financial_impact}
          title="Device Issues - Financial Impact"
        />
      )}

      {/* Recommendations */}
      {deviceAbuse.recommendations.length > 0 && (
        <Card title="Recommendations" accent="aurora">
          <ul className="space-y-2">
            {deviceAbuse.recommendations.map((rec, i) => (
              <li key={i} className="flex items-start gap-3 text-slate-300">
                <span className="text-amber-400">→</span>
                {rec}
              </li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  );
}

// ============================================================================
// Shared Components
// ============================================================================

function InsightCard({
  insight,
  rank,
  compact = false,
}: {
  insight: CustomerInsightResponse;
  rank?: number;
  compact?: boolean;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={clsx(
        'p-4 rounded-lg border',
        SEVERITY_BG[insight.severity] || SEVERITY_BG.medium
      )}
    >
      <div className="flex items-start gap-3">
        {rank && (
          <span
            className={clsx(
              'w-6 h-6 flex items-center justify-center rounded-full text-xs font-bold',
              rank === 1
                ? 'bg-red-500/30 text-red-400'
                : rank === 2
                ? 'bg-orange-500/30 text-orange-400'
                : 'bg-slate-700 text-slate-400'
            )}
          >
            {rank}
          </span>
        )}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span
              className="px-2 py-0.5 text-xs rounded-full font-medium uppercase"
              style={{
                backgroundColor: `${SEVERITY_COLORS[insight.severity]}20`,
                color: SEVERITY_COLORS[insight.severity],
              }}
            >
              {insight.severity}
            </span>
            <span className="text-xs text-slate-500">
              {CATEGORY_LABELS[insight.category] || insight.category}
            </span>
            {insight.trend_direction && (
              <span
                className={clsx(
                  'text-xs',
                  insight.trend_direction === 'degrading'
                    ? 'text-red-400'
                    : insight.trend_direction === 'improving'
                    ? 'text-green-400'
                    : 'text-slate-400'
                )}
              >
                {TREND_ICONS[insight.trend_direction]}
              </span>
            )}
          </div>
          <h4 className="font-medium text-white mb-1">{insight.headline}</h4>
          {!compact && (
            <>
              <p className="text-sm text-slate-400 mb-2">{insight.impact_statement}</p>
              <p className="text-sm text-amber-400/80">{insight.comparison_context}</p>
            </>
          )}
        </div>
        <div className="text-right">
          <p className="text-xs text-slate-500">{insight.affected_device_count} devices</p>
          <p className="text-xs text-slate-500">{insight.entity_name}</p>
        </div>
      </div>
    </motion.div>
  );
}

function InsightDetailCard({ insight }: { insight: CustomerInsightResponse }) {
  return (
    <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/30 space-y-3">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span
              className="px-2 py-0.5 text-xs rounded-full font-bold uppercase"
              style={{
                backgroundColor: `${SEVERITY_COLORS[insight.severity]}20`,
                color: SEVERITY_COLORS[insight.severity],
              }}
            >
              {insight.severity}
            </span>
            <span className="text-xs text-slate-500">
              {CATEGORY_LABELS[insight.category] || insight.category.replace(/_/g, ' ')}
            </span>
          </div>
          <h4 className="font-medium text-white">{insight.headline}</h4>
        </div>
        {insight.trend_direction && (
          <span
            className={clsx(
              'px-2 py-1 rounded text-xs font-medium',
              insight.trend_direction === 'degrading'
                ? 'bg-red-500/20 text-red-400'
                : insight.trend_direction === 'improving'
                ? 'bg-emerald-500/20 text-emerald-400'
                : 'bg-slate-700 text-slate-400'
            )}
          >
            {TREND_ICONS[insight.trend_direction]} {insight.trend_direction}
          </span>
        )}
      </div>

      {/* Impact */}
      <p className="text-sm text-slate-300">{insight.impact_statement}</p>

      {/* Context */}
      {insight.comparison_context && (
        <p className="text-sm text-amber-400/80 italic">{insight.comparison_context}</p>
      )}

      {/* Metrics */}
      <div className="grid grid-cols-2 gap-2">
        <div className="p-2 rounded bg-slate-900/50">
          <div className="text-xs text-slate-500">Affected Devices</div>
          <div className="text-lg font-bold font-mono text-white">{insight.affected_device_count}</div>
        </div>
        <div className="p-2 rounded bg-slate-900/50">
          <div className="text-xs text-slate-500">Confidence</div>
          <div className="text-lg font-bold font-mono text-purple-400">
            {(insight.confidence_score * 100).toFixed(0)}%
          </div>
        </div>
      </div>

      {/* Entity */}
      {insight.entity_name && (
        <div className="p-2 rounded bg-slate-900/50">
          <div className="text-xs text-slate-500">{insight.entity_type || 'Entity'}</div>
          <div className="text-sm text-white">{insight.entity_name}</div>
        </div>
      )}

      {/* Recommended Actions */}
      {insight.recommended_actions && insight.recommended_actions.length > 0 && (
        <div>
          <div className="text-xs font-medium text-emerald-400 uppercase tracking-wide mb-2">
            Recommended Actions
          </div>
          <ul className="space-y-1">
            {insight.recommended_actions.map((action: string, i: number) => (
              <li key={i} className="flex items-start gap-2 text-sm text-slate-300">
                <span className="text-emerald-500 mt-0.5">•</span>
                {action}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function LoadingState({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-16">
      <div className="w-10 h-10 border-4 border-amber-500/30 border-t-amber-500 rounded-full animate-spin mb-4" />
      <p className="text-slate-400">{message}</p>
    </div>
  );
}

// Alias for Suspense fallback
const TabLoadingState = LoadingState;

function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-16">
      <div className="w-16 h-16 mb-4 rounded-full bg-slate-700/50 flex items-center justify-center text-slate-400">
        <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      </div>
      <p className="text-slate-400">{message}</p>
    </div>
  );
}

interface FinancialImpact {
  total_estimated_cost: number;
  cost_breakdown: Array<{ category: string; amount: number; description: string }>;
  potential_savings: number;
  cost_per_incident?: number;
  monthly_trend?: number;
}

function FinancialImpactCard({ impact, title = "Financial Impact" }: { impact: FinancialImpact; title?: string }) {
  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  return (
    <Card title={title} accent="warning">
      <div className="space-y-4">
        {/* Top metrics row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20">
            <p className="text-xs text-slate-400 mb-1">Total Impact</p>
            <p className="text-xl font-bold font-mono text-red-400">
              {formatCurrency(impact.total_estimated_cost)}
            </p>
          </div>
          <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
            <p className="text-xs text-slate-400 mb-1">Potential Savings</p>
            <p className="text-xl font-bold font-mono text-emerald-400">
              {formatCurrency(impact.potential_savings)}
            </p>
          </div>
          {impact.cost_per_incident && (
            <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20">
              <p className="text-xs text-slate-400 mb-1">Cost/Incident</p>
              <p className="text-xl font-bold font-mono text-amber-400">
                {formatCurrency(impact.cost_per_incident)}
              </p>
            </div>
          )}
          {impact.monthly_trend !== undefined && (
            <div className="p-3 rounded-lg bg-slate-700/30 border border-slate-600/30">
              <p className="text-xs text-slate-400 mb-1">Monthly Trend</p>
              <p className={clsx(
                "text-xl font-bold font-mono flex items-center gap-1",
                impact.monthly_trend > 0 ? "text-red-400" : "text-emerald-400"
              )}>
                {impact.monthly_trend > 0 ? '+' : ''}{impact.monthly_trend.toFixed(1)}%
                {impact.monthly_trend > 0 ? (
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 11l5-5m0 0l5 5m-5-5v12" />
                  </svg>
                ) : (
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 13l-5 5m0 0l-5-5m5 5V6" />
                  </svg>
                )}
              </p>
            </div>
          )}
        </div>

        {/* Cost breakdown */}
        <div>
          <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wide mb-2">
            Cost Breakdown
          </h4>
          <div className="space-y-2">
            {impact.cost_breakdown.map((item, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-3 rounded-lg bg-slate-800/50 border border-slate-700/30"
              >
                <div className="flex-1">
                  <p className="text-sm font-medium text-white">{item.category}</p>
                  <p className="text-xs text-slate-500">{item.description}</p>
                </div>
                <p className="text-sm font-bold font-mono text-amber-400 ml-4">
                  {formatCurrency(item.amount)}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </Card>
  );
}

export default Insights;
