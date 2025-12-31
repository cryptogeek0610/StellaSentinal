/**
 * Data Overview Page - ML Training Data Discovery
 *
 * Displays telemetry data profiles, metrics distributions,
 * and patterns from SQL Server databases for ML training.
 */

import { useState, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../api/client';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from '../components/Card';
import { useMockMode } from '../hooks/useMockMode';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import type { TableProfile, AvailableMetric, DataDiscoveryStatus } from '../types/training';

// Format large numbers with K, M, B suffixes
function formatNumber(num: number): string {
  if (num >= 1_000_000_000) return (num / 1_000_000_000).toFixed(1) + 'B';
  if (num >= 1_000_000) return (num / 1_000_000).toFixed(1) + 'M';
  if (num >= 1_000) return (num / 1_000).toFixed(1) + 'K';
  return num.toLocaleString();
}

// Reusable select component for consistent styling
function Select({
  value,
  onChange,
  children,
  className = '',
}: {
  value: string;
  onChange: (value: string) => void;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className={`px-3 py-2 bg-slate-800/80 border border-slate-700/50 rounded-lg text-slate-200 text-sm
        focus:border-stellar-500/50 focus:ring-1 focus:ring-stellar-500/20 focus:outline-none
        transition-colors cursor-pointer ${className}`}
    >
      {children}
    </select>
  );
}

// KPI Card with icon
function KpiCard({
  label,
  value,
  icon,
  color = 'stellar',
}: {
  label: string;
  value: string | number;
  icon: React.ReactNode;
  color?: 'stellar' | 'aurora' | 'nebula' | 'cosmic';
}) {
  const colorClasses = {
    stellar: 'text-stellar-400 bg-stellar-500/10',
    aurora: 'text-aurora-400 bg-aurora-500/10',
    nebula: 'text-nebula-400 bg-nebula-500/10',
    cosmic: 'text-cosmic-400 bg-cosmic-500/10',
  };

  return (
    <Card className="p-4">
      <div className="flex items-start gap-3">
        <div className={`p-2 rounded-lg ${colorClasses[color]}`}>{icon}</div>
        <div className="flex-1 min-w-0">
          <p className="text-xs text-slate-500 uppercase tracking-wide">{label}</p>
          <p className={`text-2xl font-bold mt-0.5 ${colorClasses[color].split(' ')[0]}`}>
            {value}
          </p>
        </div>
      </div>
    </Card>
  );
}

// Table profile card component
function TableCard({ profile }: { profile: TableProfile }) {
  const [expanded, setExpanded] = useState(false);
  const columnCount = Object.keys(profile.column_stats).length;
  const numericColumns = Object.values(profile.column_stats).filter(
    (col) => col.mean !== null
  ).length;

  return (
    <Card className="overflow-hidden hover:border-stellar-500/30 transition-colors">
      <button
        className="w-full text-left p-4"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-slate-700/50 rounded-lg">
              <svg className="w-4 h-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
              </svg>
            </div>
            <h3 className="font-semibold text-slate-100">{profile.table_name}</h3>
          </div>
          <motion.div
            animate={{ rotate: expanded ? 180 : 0 }}
            className="text-slate-500"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </motion.div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500">Rows</span>
            <span className="text-lg font-bold text-stellar-400">
              {formatNumber(profile.row_count)}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500">Devices</span>
            <span className="text-lg font-bold text-aurora-400">
              {formatNumber(profile.device_count)}
            </span>
          </div>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-slate-500">
          <span className="px-2 py-0.5 bg-slate-800/50 rounded">{columnCount} columns</span>
          <span className="px-2 py-0.5 bg-slate-800/50 rounded">{numericColumns} numeric</span>
          {profile.date_range[0] && profile.date_range[1] && (
            <span className="px-2 py-0.5 bg-slate-800/50 rounded">
              {profile.date_range[0]} → {profile.date_range[1]}
            </span>
          )}
        </div>
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="border-t border-slate-700/50"
          >
            <div className="p-4">
              <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wide mb-3">
                Column Statistics
              </h4>
              <div className="space-y-1.5 max-h-48 overflow-y-auto">
                {Object.entries(profile.column_stats)
                  .filter(([, col]) => col.mean !== null)
                  .map(([name, col]) => (
                    <div
                      key={name}
                      className="flex items-center justify-between p-2 bg-slate-800/30 rounded-lg text-xs"
                    >
                      <span className="text-slate-300 font-mono truncate mr-4">{name}</span>
                      <div className="flex gap-3 text-slate-500 flex-shrink-0">
                        <span>μ {col.mean?.toFixed(1)}</span>
                        <span>σ {col.std?.toFixed(1)}</span>
                        <span className={col.null_percent && col.null_percent > 5 ? 'text-amber-400' : ''}>
                          ∅ {col.null_percent?.toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </Card>
  );
}

// Metric distribution chart component
function MetricDistributionChart({
  metricName,
  tableName,
}: {
  metricName: string;
  tableName?: string;
}) {
  const { data, isLoading } = useQuery({
    queryKey: ['metric-distribution', metricName, tableName],
    queryFn: () => api.getMetricDistribution(metricName, tableName),
    enabled: !!metricName,
  });

  if (isLoading) {
    return (
      <div className="h-52 flex items-center justify-center">
        <div className="flex items-center gap-2 text-slate-500">
          <div className="w-4 h-4 border-2 border-slate-500 border-t-transparent rounded-full animate-spin" />
          Loading distribution...
        </div>
      </div>
    );
  }

  if (!data || !data.bins.length) {
    return (
      <div className="h-52 flex items-center justify-center text-slate-500">
        No distribution data available
      </div>
    );
  }

  // Transform histogram data for Recharts
  const chartData = data.counts.map((count, i) => ({
    bin: `${data.bins[i].toFixed(1)}`,
    count,
  }));

  return (
    <div>
      <div className="h-52">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
            <XAxis
              dataKey="bin"
              tick={{ fill: '#64748b', fontSize: 10 }}
              axisLine={{ stroke: '#334155' }}
              tickLine={false}
              interval="preserveStartEnd"
            />
            <YAxis
              tick={{ fill: '#64748b', fontSize: 10 }}
              axisLine={false}
              tickLine={false}
              width={35}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1e293b',
                border: '1px solid #334155',
                borderRadius: '8px',
                fontSize: '12px',
              }}
              labelStyle={{ color: '#94a3b8' }}
              itemStyle={{ color: '#a5b4fc' }}
            />
            <Bar dataKey="count" fill="#6366f1" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-3 flex items-center justify-center gap-6 text-xs">
        <div className="flex items-center gap-1.5">
          <span className="text-slate-500">Mean</span>
          <span className="font-medium text-slate-300">{data.stats.mean.toFixed(2)}</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-slate-500">Median</span>
          <span className="font-medium text-slate-300">{data.stats.median.toFixed(2)}</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-slate-500">Std</span>
          <span className="font-medium text-slate-300">{data.stats.std.toFixed(2)}</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-slate-500">Samples</span>
          <span className="font-medium text-slate-300">{formatNumber(data.stats.total_samples)}</span>
        </div>
      </div>
    </div>
  );
}

// Discovery status banner
function DiscoveryStatusBanner({ status }: { status: DataDiscoveryStatus }) {
  if (status.status === 'idle') return null;

  const config = {
    running: {
      bg: 'bg-stellar-900/30 border-stellar-500/30',
      text: 'text-stellar-200',
    },
    completed: {
      bg: 'bg-emerald-900/30 border-emerald-500/30',
      text: 'text-emerald-200',
    },
    failed: {
      bg: 'bg-red-900/30 border-red-500/30',
      text: 'text-red-200',
    },
  }[status.status];

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`p-4 rounded-xl border ${config.bg}`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {status.status === 'running' && (
            <div className="w-5 h-5 border-2 border-stellar-400 border-t-transparent rounded-full animate-spin" />
          )}
          {status.status === 'completed' && (
            <svg className="w-5 h-5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          )}
          {status.status === 'failed' && (
            <svg className="w-5 h-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          )}
          <div>
            <p className={`font-medium ${config.text}`}>
              {status.status === 'running'
                ? 'Data Discovery in Progress'
                : status.status === 'completed'
                ? 'Discovery Completed'
                : 'Discovery Failed'}
            </p>
            {status.message && (
              <p className="text-sm text-slate-400 mt-0.5">{status.message}</p>
            )}
          </div>
        </div>
        {status.status === 'running' && (
          <div className="text-right">
            <p className="text-2xl font-bold text-stellar-400">
              {Math.round(status.progress)}%
            </p>
          </div>
        )}
      </div>
    </motion.div>
  );
}

// Connection error banner component
function ConnectionErrorBanner({ onEnableMockMode }: { onEnableMockMode: () => void }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-5 bg-amber-900/20 border border-amber-500/30 rounded-xl"
    >
      <div className="flex items-start gap-4">
        <div className="flex-shrink-0 p-2 bg-amber-500/20 rounded-lg">
          <svg className="w-5 h-5 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        </div>
        <div className="flex-1">
          <h3 className="font-semibold text-amber-200">Database Connection Unavailable</h3>
          <p className="text-amber-200/60 text-sm mt-1">
            Unable to connect to the XSight SQL Server. This page requires database access to profile tables and discover metrics.
          </p>
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <button
              onClick={onEnableMockMode}
              className="px-4 py-2 bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/40 text-amber-200 rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
              Enable Mock Mode
            </button>
            <span className="text-xs text-amber-200/40">View simulated demo data</span>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

// Metric list item
function MetricListItem({
  metric,
  isSelected,
  onClick,
}: {
  metric: AvailableMetric;
  isSelected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left p-3 rounded-lg text-sm transition-all ${
        isSelected
          ? 'bg-stellar-500/20 border border-stellar-500/40 ring-1 ring-stellar-500/20'
          : 'bg-slate-800/30 border border-transparent hover:bg-slate-800/60 hover:border-slate-700/50'
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className={`font-mono text-sm truncate ${isSelected ? 'text-stellar-300' : 'text-slate-300'}`}>
              {metric.column}
            </span>
          </div>
          <div className="flex items-center gap-1.5 mt-1.5">
            {metric.category && (
              <span className="text-[10px] px-1.5 py-0.5 bg-slate-700/50 rounded text-slate-400">
                {metric.category}
              </span>
            )}
            {metric.domain && (
              <span className="text-[10px] px-1.5 py-0.5 bg-slate-700/50 rounded text-slate-400">
                {metric.domain}
              </span>
            )}
          </div>
        </div>
        <span className="text-[10px] text-slate-600 flex-shrink-0">{metric.table}</span>
      </div>
    </button>
  );
}

export default function DataOverview() {
  const queryClient = useQueryClient();
  const { mockMode, setMockMode } = useMockMode();
  const [selectedMetric, setSelectedMetric] = useState<AvailableMetric | null>(null);
  const [categoryFilter, setCategoryFilter] = useState<string>('all');
  const [domainFilter, setDomainFilter] = useState<string>('all');

  // Fetch table profiles
  const {
    data: tableProfiles,
    isLoading: profilesLoading,
    error: profilesError,
  } = useQuery({
    queryKey: ['table-profiles'],
    queryFn: () => api.getTableProfiles(),
    refetchInterval: 60000,
  });

  // Fetch available metrics
  const { data: metrics } = useQuery({
    queryKey: ['available-metrics'],
    queryFn: () => api.getAvailableMetrics(),
  });

  // Fetch discovery status
  const { data: discoveryStatus } = useQuery({
    queryKey: ['discovery-status'],
    queryFn: () => api.getDiscoveryStatus(),
    refetchInterval: 5000,
  });

  // Run discovery mutation
  const runDiscoveryMutation = useMutation({
    mutationFn: () => api.runDataDiscovery(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['discovery-status'] });
    },
  });

  // Summary stats
  const summary = useMemo(() => {
    if (!tableProfiles) return null;
    return {
      totalTables: tableProfiles.length,
      totalRows: tableProfiles.reduce((sum, t) => sum + t.row_count, 0),
      totalDevices: Math.max(...tableProfiles.map((t) => t.device_count), 0),
      totalMetrics: metrics?.length || 0,
    };
  }, [tableProfiles, metrics]);

  // Filter metrics by category and domain
  const filteredMetrics = useMemo(() => {
    if (!metrics) return [];
    return metrics.filter((m) => {
      if (categoryFilter !== 'all' && m.category !== categoryFilter) return false;
      if (domainFilter !== 'all' && m.domain !== domainFilter) return false;
      return true;
    });
  }, [metrics, categoryFilter, domainFilter]);

  // Get unique categories and domains for filters
  const categories = useMemo(() => {
    const cats = new Set(metrics?.map((m) => m.category).filter((c): c is string => !!c) || []);
    return Array.from(cats).sort();
  }, [metrics]);

  const domains = useMemo(() => {
    const doms = new Set(metrics?.map((m) => m.domain).filter((d): d is string => !!d) || []);
    return Array.from(doms).sort();
  }, [metrics]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Data Overview</h1>
          <p className="text-slate-400 mt-1">
            Explore telemetry data from SQL Server for ML training
          </p>
        </div>
        <button
          onClick={() => runDiscoveryMutation.mutate()}
          disabled={runDiscoveryMutation.isPending || discoveryStatus?.status === 'running'}
          className="px-4 py-2.5 bg-stellar-600 hover:bg-stellar-500 disabled:bg-slate-700 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2 sm:w-auto w-full"
        >
          {runDiscoveryMutation.isPending || discoveryStatus?.status === 'running' ? (
            <>
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              Running...
            </>
          ) : (
            <>
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Run Discovery
            </>
          )}
        </button>
      </div>

      {/* Connection Error Banner */}
      {!mockMode && profilesError && (
        <ConnectionErrorBanner onEnableMockMode={() => setMockMode(true)} />
      )}

      {/* Discovery Status Banner */}
      {discoveryStatus && discoveryStatus.status !== 'idle' && (
        <DiscoveryStatusBanner status={discoveryStatus} />
      )}

      {/* Summary KPIs */}
      {summary && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <KpiCard
            label="Tables Profiled"
            value={summary.totalTables}
            color="stellar"
            icon={
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
              </svg>
            }
          />
          <KpiCard
            label="Total Rows"
            value={formatNumber(summary.totalRows)}
            color="aurora"
            icon={
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
              </svg>
            }
          />
          <KpiCard
            label="Unique Devices"
            value={formatNumber(summary.totalDevices)}
            color="nebula"
            icon={
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 18h.01M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z" />
              </svg>
            }
          />
          <KpiCard
            label="Available Metrics"
            value={summary.totalMetrics}
            color="cosmic"
            icon={
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            }
          />
        </div>
      )}

      {/* Main Content */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Database Tables - Left Column */}
        <div className="xl:col-span-1">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-slate-200">Database Tables</h2>
            {tableProfiles && (
              <span className="text-xs text-slate-500">{tableProfiles.length} tables</span>
            )}
          </div>

          {profilesLoading ? (
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <Card key={i} className="p-4 animate-pulse">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-8 h-8 bg-slate-700 rounded-lg" />
                    <div className="h-4 bg-slate-700 rounded w-32" />
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="h-6 bg-slate-700 rounded w-16" />
                    <div className="h-6 bg-slate-700 rounded w-16" />
                  </div>
                </Card>
              ))}
            </div>
          ) : profilesError ? (
            <Card className="p-5 border-slate-700/50">
              <div className="flex flex-col items-center text-center">
                <div className="p-3 bg-red-500/10 rounded-full mb-3">
                  <svg className="w-6 h-6 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <p className="text-red-400 font-medium">Connection Failed</p>
                <p className="text-slate-500 text-sm mt-1">Unable to load table profiles</p>
                {!mockMode && (
                  <button
                    onClick={() => setMockMode(true)}
                    className="mt-4 text-sm text-stellar-400 hover:text-stellar-300 underline"
                  >
                    Try Mock Mode
                  </button>
                )}
              </div>
            </Card>
          ) : tableProfiles && tableProfiles.length > 0 ? (
            <div className="space-y-3">
              {tableProfiles.map((profile) => (
                <TableCard key={profile.table_name} profile={profile} />
              ))}
            </div>
          ) : (
            <Card className="p-6">
              <div className="flex flex-col items-center text-center">
                <div className="p-3 bg-slate-700/50 rounded-full mb-3">
                  <svg className="w-6 h-6 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                  </svg>
                </div>
                <p className="text-slate-400">No table profiles available</p>
                <p className="text-slate-500 text-sm mt-1">
                  Click "Run Discovery" to analyze database tables
                </p>
              </div>
            </Card>
          )}
        </div>

        {/* Metric Explorer - Right Column (wider) */}
        <div className="xl:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-slate-200">Metric Explorer</h2>
            <span className="text-xs text-slate-500">
              {filteredMetrics.length} of {metrics?.length || 0} metrics
            </span>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
            {/* Metric List */}
            <div className="lg:col-span-2">
              <Card className="p-4 h-full">
                {/* Filters */}
                <div className="space-y-2 mb-4">
                  <div className="grid grid-cols-2 gap-2">
                    <Select value={categoryFilter} onChange={setCategoryFilter}>
                      <option value="all">All Categories</option>
                      {categories.map((cat) => (
                        <option key={cat} value={cat}>
                          {cat.charAt(0).toUpperCase() + cat.slice(1)}
                        </option>
                      ))}
                    </Select>
                    <Select value={domainFilter} onChange={setDomainFilter}>
                      <option value="all">All Domains</option>
                      {domains.map((dom) => (
                        <option key={dom} value={dom}>
                          {dom.charAt(0).toUpperCase() + dom.slice(1)}
                        </option>
                      ))}
                    </Select>
                  </div>
                </div>

                {/* Metric List */}
                <div className="space-y-2 max-h-[420px] overflow-y-auto pr-1">
                  {filteredMetrics.length > 0 ? (
                    filteredMetrics.map((metric) => (
                      <MetricListItem
                        key={`${metric.table}-${metric.column}`}
                        metric={metric}
                        isSelected={selectedMetric?.column === metric.column}
                        onClick={() => setSelectedMetric(metric)}
                      />
                    ))
                  ) : (
                    <div className="flex flex-col items-center justify-center py-8 text-center">
                      <svg className="w-8 h-8 text-slate-600 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      <p className="text-slate-500 text-sm">No metrics match filters</p>
                    </div>
                  )}
                </div>
              </Card>
            </div>

            {/* Distribution Chart & Details */}
            <div className="lg:col-span-3">
              <Card className="p-4 h-full">
                {selectedMetric ? (
                  <div>
                    {/* Selected Metric Header */}
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <h3 className="font-semibold text-slate-200 font-mono">
                          {selectedMetric.column}
                        </h3>
                        <p className="text-xs text-slate-500 mt-0.5">
                          from {selectedMetric.table}
                        </p>
                      </div>
                      <div className="flex gap-1.5">
                        {selectedMetric.category && (
                          <span className="text-xs px-2 py-1 bg-stellar-500/20 text-stellar-300 rounded">
                            {selectedMetric.category}
                          </span>
                        )}
                        {selectedMetric.domain && (
                          <span className="text-xs px-2 py-1 bg-aurora-500/20 text-aurora-300 rounded">
                            {selectedMetric.domain}
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Description */}
                    {selectedMetric.description && (
                      <p className="text-sm text-slate-400 mb-4 italic">
                        {selectedMetric.description}
                      </p>
                    )}

                    {/* Distribution Chart */}
                    <MetricDistributionChart
                      metricName={selectedMetric.column}
                      tableName={selectedMetric.table}
                    />

                    {/* Statistics Grid */}
                    <div className="mt-4 pt-4 border-t border-slate-700/50">
                      <div className="grid grid-cols-4 gap-3">
                        {selectedMetric.mean !== null && (
                          <div className="p-2 bg-slate-800/30 rounded-lg text-center">
                            <p className="text-[10px] text-slate-500 uppercase">Mean</p>
                            <p className="text-sm font-medium text-slate-200 mt-0.5">
                              {selectedMetric.mean.toFixed(2)}
                            </p>
                          </div>
                        )}
                        {selectedMetric.std !== null && (
                          <div className="p-2 bg-slate-800/30 rounded-lg text-center">
                            <p className="text-[10px] text-slate-500 uppercase">Std Dev</p>
                            <p className="text-sm font-medium text-slate-200 mt-0.5">
                              {selectedMetric.std.toFixed(2)}
                            </p>
                          </div>
                        )}
                        {selectedMetric.min !== null && (
                          <div className="p-2 bg-slate-800/30 rounded-lg text-center">
                            <p className="text-[10px] text-slate-500 uppercase">Min</p>
                            <p className="text-sm font-medium text-slate-200 mt-0.5">
                              {selectedMetric.min.toFixed(2)}
                            </p>
                          </div>
                        )}
                        {selectedMetric.max !== null && (
                          <div className="p-2 bg-slate-800/30 rounded-lg text-center">
                            <p className="text-[10px] text-slate-500 uppercase">Max</p>
                            <p className="text-sm font-medium text-slate-200 mt-0.5">
                              {selectedMetric.max.toFixed(2)}
                            </p>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="h-full min-h-[400px] flex flex-col items-center justify-center text-center">
                    <div className="p-4 bg-slate-800/30 rounded-full mb-4">
                      <svg className="w-8 h-8 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                      </svg>
                    </div>
                    <p className="text-slate-400 font-medium">Select a Metric</p>
                    <p className="text-slate-500 text-sm mt-1 max-w-[200px]">
                      Choose a metric from the list to view its distribution and statistics
                    </p>
                  </div>
                )}
              </Card>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
