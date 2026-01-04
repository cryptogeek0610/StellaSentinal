/**
 * AI Insights Page - Customer-Facing Intelligence
 *
 * Aligned with Carl's vision: "XSight has the data. XSight needs the story."
 * Provides pre-interpreted, contextualized, and actionable insights.
 */

import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { api, CustomerInsightResponse, DailyDigestResponse, ShiftReadinessResponse } from '../api/client';
import { motion } from 'framer-motion';
import { Card } from '../components/Card';
import { KPICard } from '../components/KPICard';
import { SlideOverPanel } from '../components/unified';
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

type TabId = 'digest' | 'battery' | 'network' | 'devices' | 'apps';

const VALID_TABS: TabId[] = ['digest', 'battery', 'network', 'devices', 'apps'];

function Insights() {
  const [searchParams, setSearchParams] = useSearchParams();
  const tabFromUrl = searchParams.get('tab') as TabId | null;
  const initialTab = tabFromUrl && VALID_TABS.includes(tabFromUrl) ? tabFromUrl : 'digest';

  const [activeTab, setActiveTab] = useState<TabId>(initialTab);
  const [selectedLocation, setSelectedLocation] = useState<string>('store-001');

  // Sync tab state with URL params
  useEffect(() => {
    const urlTab = searchParams.get('tab') as TabId | null;
    if (urlTab && VALID_TABS.includes(urlTab) && urlTab !== activeTab) {
      setActiveTab(urlTab);
    }
  }, [searchParams]);

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

  // Fetch daily digest
  const { data: digest, isLoading: digestLoading } = useQuery({
    queryKey: ['insights', 'daily-digest'],
    queryFn: () => api.getDailyDigest({ period_days: 7 }),
  });

  // Fetch shift readiness for selected location
  const { data: shiftReadiness, isLoading: shiftLoading } = useQuery({
    queryKey: ['insights', 'shift-readiness', selectedLocation],
    queryFn: () => api.getShiftReadiness(selectedLocation),
    enabled: activeTab === 'battery',
  });

  // Fetch network analysis
  const { data: networkAnalysis, isLoading: networkLoading } = useQuery({
    queryKey: ['insights', 'network-analysis'],
    queryFn: () => api.getNetworkAnalysis(),
    enabled: activeTab === 'network',
  });

  // Fetch device abuse analysis
  const { data: deviceAbuse, isLoading: deviceLoading } = useQuery({
    queryKey: ['insights', 'device-abuse'],
    queryFn: () => api.getDeviceAbuseAnalysis(),
    enabled: activeTab === 'devices',
  });

  // Fetch app analysis
  const { data: appAnalysis, isLoading: appLoading } = useQuery({
    queryKey: ['insights', 'app-analysis'],
    queryFn: () => api.getAppAnalysis(),
    enabled: activeTab === 'apps',
  });

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
      id: 'battery',
      label: 'Shift Readiness',
      icon: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      ),
    },
    {
      id: 'network',
      label: 'Network',
      icon: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.111 16.404a5.5 5.5 0 017.778 0M12 20h.01m-7.08-7.071c3.904-3.905 10.236-3.905 14.141 0M1.394 9.393c5.857-5.857 15.355-5.857 21.213 0" />
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
      id: 'apps',
      label: 'Apps',
      icon: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
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
      {activeTab === 'battery' && (
        <BatteryTab
          shiftReadiness={shiftReadiness}
          isLoading={shiftLoading}
          selectedLocation={selectedLocation}
          onLocationChange={setSelectedLocation}
        />
      )}
      {activeTab === 'network' && (
        <NetworkTab networkAnalysis={networkAnalysis} isLoading={networkLoading} />
      )}
      {activeTab === 'devices' && (
        <DevicesTab deviceAbuse={deviceAbuse} isLoading={deviceLoading} />
      )}
      {activeTab === 'apps' && (
        <AppsTab appAnalysis={appAnalysis} isLoading={appLoading} />
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
// Battery/Shift Readiness Tab
// ============================================================================

function BatteryTab({
  shiftReadiness,
  isLoading,
  selectedLocation,
  onLocationChange,
}: {
  shiftReadiness?: ShiftReadinessResponse;
  isLoading: boolean;
  selectedLocation: string;
  onLocationChange: (loc: string) => void;
}) {
  const locations = [
    { id: 'store-001', name: 'Downtown Flagship' },
    { id: 'store-002', name: 'Westside Mall' },
    { id: 'store-003', name: 'Harbor Point' },
    { id: 'store-004', name: 'Tech Plaza' },
  ];

  return (
    <div className="space-y-6">
      {/* Location Selector */}
      <div className="flex items-center gap-4">
        <label className="text-sm text-slate-400">Location:</label>
        <select
          value={selectedLocation}
          onChange={(e) => onLocationChange(e.target.value)}
          className="input-stellar px-4 py-2 rounded-lg bg-slate-800/50 border border-slate-700 text-white"
        >
          {locations.map((loc) => (
            <option key={loc.id} value={loc.id}>
              {loc.name}
            </option>
          ))}
        </select>
      </div>

      {isLoading ? (
        <LoadingState message="Analyzing shift readiness..." />
      ) : !shiftReadiness ? (
        <EmptyState message="No shift data available" />
      ) : (
        <>
          {/* Readiness KPIs */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <KPICard
              title="Shift Readiness"
              value={`${shiftReadiness.readiness_percentage.toFixed(0)}%`}
              color={shiftReadiness.readiness_percentage >= 80 ? 'aurora' : 'warning'}
              progressValue={shiftReadiness.readiness_percentage}
              icon={<span className="text-2xl">🔋</span>}
            />
            <KPICard
              title="Devices Ready"
              value={`${shiftReadiness.devices_ready}/${shiftReadiness.total_devices}`}
              color="aurora"
              icon={<span className="text-2xl">✅</span>}
            />
            <KPICard
              title="At Risk"
              value={shiftReadiness.devices_at_risk}
              color="warning"
              isActive={shiftReadiness.devices_at_risk > 0}
              icon={<span className="text-2xl">⚠️</span>}
            />
            <KPICard
              title="Critical"
              value={shiftReadiness.devices_critical}
              color="danger"
              isActive={shiftReadiness.devices_critical > 0}
              icon={<span className="text-2xl">🚨</span>}
            />
          </div>

          {/* Shift Info */}
          <Card
            title={`${shiftReadiness.shift_name} - ${shiftReadiness.location_name}`}
            accent="stellar"
          >
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Stat label="Avg Battery" value={`${shiftReadiness.avg_battery_at_start.toFixed(0)}%`} />
              <Stat label="Avg Drain Rate" value={`${shiftReadiness.avg_drain_rate.toFixed(1)}%/hr`} />
              <Stat label="Not Fully Charged" value={shiftReadiness.devices_not_fully_charged} />
              <Stat
                label="vs Last Week"
                value={
                  shiftReadiness.vs_last_week_readiness !== null
                    ? `${shiftReadiness.vs_last_week_readiness > 0 ? '+' : ''}${shiftReadiness.vs_last_week_readiness.toFixed(1)}%`
                    : '—'
                }
                trend={
                  shiftReadiness.vs_last_week_readiness !== null
                    ? shiftReadiness.vs_last_week_readiness >= 0
                      ? 'up'
                      : 'down'
                    : undefined
                }
              />
            </div>
          </Card>

          {/* Device Details */}
          <Card title="Device Readiness">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-slate-500 border-b border-slate-700">
                    <th className="py-3 px-4">Device</th>
                    <th className="py-3 px-4">Battery</th>
                    <th className="py-3 px-4">Drain Rate</th>
                    <th className="py-3 px-4">End of Shift</th>
                    <th className="py-3 px-4">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {shiftReadiness.device_details.map((device) => (
                    <tr
                      key={device.device_id}
                      className="border-b border-slate-800 hover:bg-slate-800/30"
                    >
                      <td className="py-3 px-4 text-white font-medium">
                        {device.device_name}
                      </td>
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-2 bg-slate-700 rounded-full overflow-hidden">
                            <div
                              className={clsx(
                                'h-full rounded-full',
                                device.current_battery >= 80
                                  ? 'bg-green-500'
                                  : device.current_battery >= 50
                                  ? 'bg-amber-500'
                                  : 'bg-red-500'
                              )}
                              style={{ width: `${device.current_battery}%` }}
                            />
                          </div>
                          <span className="text-slate-300">{device.current_battery}%</span>
                        </div>
                      </td>
                      <td className="py-3 px-4 text-slate-300">
                        {device.drain_rate_per_hour.toFixed(1)}%/hr
                      </td>
                      <td className="py-3 px-4">
                        <span
                          className={clsx(
                            'font-mono',
                            device.projected_end_battery >= 20
                              ? 'text-green-400'
                              : device.projected_end_battery >= 10
                              ? 'text-amber-400'
                              : 'text-red-400'
                          )}
                        >
                          {device.projected_end_battery}%
                        </span>
                      </td>
                      <td className="py-3 px-4">
                        {device.will_complete_shift ? (
                          <span className="px-2 py-1 text-xs rounded-full bg-green-500/20 text-green-400 border border-green-500/30">
                            Ready
                          </span>
                        ) : (
                          <span className="px-2 py-1 text-xs rounded-full bg-red-500/20 text-red-400 border border-red-500/30">
                            At Risk
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Recommendations */}
          {shiftReadiness.recommendations.length > 0 && (
            <Card title="Recommendations" accent="aurora">
              <ul className="space-y-2">
                {shiftReadiness.recommendations.map((rec, i) => (
                  <li key={i} className="flex items-start gap-3 text-slate-300">
                    <span className="text-amber-400">→</span>
                    {rec}
                  </li>
                ))}
              </ul>
            </Card>
          )}
        </>
      )}
    </div>
  );
}

// ============================================================================
// Network Tab
// ============================================================================

function NetworkTab({
  networkAnalysis,
  isLoading,
}: {
  networkAnalysis?: ReturnType<typeof api.getNetworkAnalysis> extends Promise<infer T> ? T : never;
  isLoading: boolean;
}) {
  if (isLoading) {
    return <LoadingState message="Analyzing network patterns..." />;
  }

  if (!networkAnalysis) {
    return <EmptyState message="No network data available" />;
  }

  return (
    <div className="space-y-6">
      {/* Network KPIs */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <KPICard
          title="WiFi Roaming Issues"
          value={networkAnalysis.wifi_summary.devices_with_roaming_issues}
          color="warning"
          icon={<span className="text-2xl">📶</span>}
        />
        <KPICard
          title="Potential Dead Zones"
          value={networkAnalysis.wifi_summary.potential_dead_zones}
          color="danger"
          icon={<span className="text-2xl">❌</span>}
        />
        <KPICard
          title="Total Disconnects"
          value={networkAnalysis.disconnect_summary.total_disconnects}
          color="warning"
          icon={<span className="text-2xl">🔌</span>}
        />
        <KPICard
          title="Hidden Devices"
          value={networkAnalysis.hidden_devices_count}
          color={networkAnalysis.hidden_devices_count > 0 ? 'danger' : 'aurora'}
          icon={<span className="text-2xl">👻</span>}
        />
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        {/* WiFi Summary */}
        <Card title="WiFi Analysis" accent="stellar">
          <div className="space-y-4">
            <Stat label="Total Devices" value={networkAnalysis.wifi_summary.total_devices} />
            <Stat
              label="Roaming Issues"
              value={networkAnalysis.wifi_summary.devices_with_roaming_issues}
            />
            <Stat
              label="AP Stickiness"
              value={networkAnalysis.wifi_summary.devices_with_stickiness}
            />
            <Stat
              label="Avg APs/Device"
              value={networkAnalysis.wifi_summary.avg_aps_per_device.toFixed(1)}
            />
          </div>
        </Card>

        {/* Disconnect Summary */}
        <Card title="Disconnect Patterns" accent="warning">
          <div className="space-y-4">
            <Stat
              label="Avg Disconnects/Device"
              value={networkAnalysis.disconnect_summary.avg_disconnects_per_device.toFixed(2)}
            />
            <Stat
              label="Total Offline Hours"
              value={networkAnalysis.disconnect_summary.total_offline_hours.toFixed(1)}
            />
            <div className="p-3 bg-slate-800/30 rounded-lg">
              <p className="text-sm text-slate-400 mb-1">Pattern Detected:</p>
              <p className="text-slate-300">{networkAnalysis.disconnect_summary.pattern_description}</p>
            </div>
          </div>
        </Card>
      </div>

      {/* Recommendations */}
      {networkAnalysis.recommendations.length > 0 && (
        <Card title="Recommendations" accent="aurora">
          <ul className="space-y-2">
            {networkAnalysis.recommendations.map((rec, i) => (
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
          icon={<span className="text-2xl">📉</span>}
        />
        <KPICard
          title="Excessive Drops"
          value={deviceAbuse.devices_with_excessive_drops}
          color="danger"
          icon={<span className="text-2xl">⚠️</span>}
        />
        <KPICard
          title="Total Reboots"
          value={deviceAbuse.total_reboots}
          color="warning"
          icon={<span className="text-2xl">🔄</span>}
        />
        <KPICard
          title="Excessive Reboots"
          value={deviceAbuse.devices_with_excessive_reboots}
          color="danger"
          icon={<span className="text-2xl">🚨</span>}
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
// Apps Tab
// ============================================================================

function AppsTab({
  appAnalysis,
  isLoading,
}: {
  appAnalysis?: ReturnType<typeof api.getAppAnalysis> extends Promise<infer T> ? T : never;
  isLoading: boolean;
}) {
  if (isLoading) {
    return <LoadingState message="Analyzing app performance..." />;
  }

  if (!appAnalysis) {
    return <EmptyState message="No app data available" />;
  }

  return (
    <div className="space-y-6">
      {/* App KPIs */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <KPICard
          title="Apps Analyzed"
          value={appAnalysis.total_apps_analyzed}
          color="stellar"
          icon={<span className="text-2xl">📦</span>}
        />
        <KPICard
          title="Apps with Issues"
          value={appAnalysis.apps_with_issues}
          color="warning"
          icon={<span className="text-2xl">⚠️</span>}
        />
        <KPICard
          title="Total Crashes"
          value={appAnalysis.total_crashes}
          color="danger"
          icon={<span className="text-2xl">💥</span>}
        />
        <KPICard
          title="ANRs"
          value={appAnalysis.total_anrs}
          color="warning"
          icon={<span className="text-2xl">⏳</span>}
        />
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        {/* Power Consumers */}
        <Card title="Top Power Consumers" accent="warning">
          <div className="space-y-3">
            {appAnalysis.top_power_consumers.map((app, index) => (
              <div
                key={app.package_name}
                className="flex items-center gap-4 p-3 bg-slate-800/30 rounded-lg"
              >
                <span
                  className={clsx(
                    'w-6 h-6 flex items-center justify-center rounded-full text-xs font-bold',
                    index === 0
                      ? 'bg-red-500/20 text-red-400'
                      : index === 1
                      ? 'bg-orange-500/20 text-orange-400'
                      : 'bg-slate-700 text-slate-400'
                  )}
                >
                  {index + 1}
                </span>
                <div className="flex-1">
                  <p className="font-medium text-white">{app.app_name}</p>
                  <p className="text-xs text-slate-500">{app.package_name}</p>
                </div>
                <div className="text-right">
                  <p className="font-mono text-amber-400">{app.battery_drain_percent.toFixed(1)}%</p>
                  <p className="text-xs text-slate-500">{app.drain_per_hour.toFixed(1)}%/hr</p>
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* Top Crashers */}
        <Card title="Top Crashers" accent="danger">
          <div className="space-y-3">
            {appAnalysis.top_crashers.map((app, index) => (
              <div
                key={app.package_name}
                className="flex items-center gap-4 p-3 bg-slate-800/30 rounded-lg"
              >
                <span
                  className={clsx(
                    'w-6 h-6 flex items-center justify-center rounded-full text-xs font-bold',
                    index === 0
                      ? 'bg-red-500/20 text-red-400'
                      : index === 1
                      ? 'bg-orange-500/20 text-orange-400'
                      : 'bg-slate-700 text-slate-400'
                  )}
                >
                  {index + 1}
                </span>
                <div className="flex-1">
                  <p className="font-medium text-white">{app.app_name}</p>
                  <p className="text-xs text-slate-500">{app.devices_affected} devices affected</p>
                </div>
                <div className="text-right">
                  <p className="font-mono text-red-400">{app.crash_count} crashes</p>
                  <p className="text-xs text-slate-500">{app.anr_count} ANRs</p>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Recommendations */}
      {appAnalysis.recommendations.length > 0 && (
        <Card title="Recommendations" accent="aurora">
          <ul className="space-y-2">
            {appAnalysis.recommendations.map((rec, i) => (
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

function Stat({
  label,
  value,
  trend,
}: {
  label: string;
  value: string | number;
  trend?: 'up' | 'down';
}) {
  return (
    <div className="p-3 bg-slate-800/30 rounded-lg">
      <p className="text-xs text-slate-500 mb-1">{label}</p>
      <p className="text-lg font-bold text-white flex items-center gap-1">
        {value}
        {trend === 'up' && <span className="text-green-400 text-sm">↑</span>}
        {trend === 'down' && <span className="text-red-400 text-sm">↓</span>}
      </p>
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

function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-16">
      <span className="text-5xl mb-4">📊</span>
      <p className="text-slate-400">{message}</p>
    </div>
  );
}

export default Insights;
