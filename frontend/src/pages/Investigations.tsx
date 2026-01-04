/**
 * Investigations Page - Stellar Operations
 * 
 * Anomaly case management with list and kanban views,
 * filtering by status and severity
 */

import { useState, useMemo } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { format, formatDistanceToNowStrict } from 'date-fns';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from '../components/Card';

import { Anomaly } from '../types/anomaly';

type ViewMode = 'list' | 'grouped' | 'kanban';
type SortBy = 'score' | 'time' | 'device';

// Helper to generate readable reasons
const getAnomalyReason = (anomaly: Anomaly): string => {
  if (anomaly.total_battery_level_drop && anomaly.total_battery_level_drop > 20) {
    return 'Severe battery drain detected relative to usage';
  }
  if (anomaly.total_free_storage_kb && anomaly.total_free_storage_kb < 500000) {
    return 'Critical low storage space available';
  }
  if (anomaly.offline_time && anomaly.offline_time > 120) {
    return 'Extended period of offline inactivity';
  }
  if (anomaly.wifi_signal_strength && anomaly.wifi_signal_strength < -80) {
    return 'Persistent poor network quality connectivity';
  }
  if (anomaly.download && anomaly.download > 500) {
    return 'Unusual accumulated data consumption';
  }
  if (anomaly.disconnect_count && anomaly.disconnect_count > 10) {
    return 'High frequency of network disconnections';
  }
  return 'Unusual behavioral pattern detected by AI model';
};

// Severity configuration with static classes
const severityConfig = {
  critical: {
    label: 'CRITICAL',
    barColor: 'bg-red-500',
    badgeBg: 'bg-red-500/20',
    badgeText: 'text-red-400',
    badgeBorder: 'border-red-500/30',
  },
  high: {
    label: 'HIGH',
    barColor: 'bg-orange-500',
    badgeBg: 'bg-orange-500/20',
    badgeText: 'text-orange-400',
    badgeBorder: 'border-orange-500/30',
  },
  medium: {
    label: 'MEDIUM',
    barColor: 'bg-amber-500',
    badgeBg: 'bg-amber-500/20',
    badgeText: 'text-amber-400',
    badgeBorder: 'border-amber-500/30',
  },
  low: {
    label: 'LOW',
    barColor: 'bg-slate-500',
    badgeBg: 'bg-slate-700/50',
    badgeText: 'text-slate-400',
    badgeBorder: 'border-slate-600/50',
  },
};

// Status configuration with static classes
const statusConfig = {
  open: {
    label: 'Open',
    badgeBg: 'bg-red-500/20',
    badgeText: 'text-red-400',
    badgeBorder: 'border-red-500/30',
    headerBg: 'bg-red-500/10',
    headerText: 'text-red-400',
  },
  investigating: {
    label: 'Investigating',
    badgeBg: 'bg-orange-500/20',
    badgeText: 'text-orange-400',
    badgeBorder: 'border-orange-500/30',
    headerBg: 'bg-orange-500/10',
    headerText: 'text-orange-400',
  },
  resolved: {
    label: 'Resolved',
    badgeBg: 'bg-emerald-500/20',
    badgeText: 'text-emerald-400',
    badgeBorder: 'border-emerald-500/30',
    headerBg: 'bg-emerald-500/10',
    headerText: 'text-emerald-400',
  },
  false_positive: {
    label: 'False Positive',
    badgeBg: 'bg-slate-700/50',
    badgeText: 'text-slate-400',
    badgeBorder: 'border-slate-600/50',
    headerBg: 'bg-slate-700/30',
    headerText: 'text-slate-400',
  },
};

// Group anomalies by device
interface DeviceGroup {
  device_id: number;
  anomalies: Anomaly[];
  worstScore: number;
  latestTimestamp: string;
  openCount: number;
}

function Investigations() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [viewMode, setViewMode] = useState<ViewMode>('grouped');
  const [sortBy, setSortBy] = useState<SortBy>('score');
  const [page, setPage] = useState(1);
  const [expandedDevices, setExpandedDevices] = useState<Set<number>>(new Set());

  const statusFilter = searchParams.get('status') || '';
  const severityFilter = searchParams.get('severity') || '';
  const deviceFilter = searchParams.get('device') || '';

  const { data, isLoading } = useQuery({
    queryKey: ['anomalies', page, statusFilter, severityFilter, deviceFilter],
    queryFn: () => api.getAnomalies({
      page,
      page_size: 50,
      status: statusFilter || undefined,
      device_id: deviceFilter ? parseInt(deviceFilter) : undefined,
      min_score: severityFilter === 'critical' ? -1.0 : severityFilter === 'high' ? -0.7 : undefined,
      max_score: severityFilter === 'critical' ? -0.7 : severityFilter === 'high' ? -0.5 : undefined,
    }),
    refetchInterval: 30000,
  });

  // Calculate status counts from the current filtered dataset
  const statusCounts = useMemo(() => {
    if (!data) return { open: 0, investigating: 0, resolved: 0, false_positive: 0, total: 0 };
    const counts = data.anomalies.reduce((acc, a) => {
      acc[a.status as keyof typeof acc] = (acc[a.status as keyof typeof acc] || 0) + 1;
      return acc;
    }, { open: 0, investigating: 0, resolved: 0, false_positive: 0 });
    return { ...counts, total: data.anomalies.length };
  }, [data]);

  // Count items requiring attention (open + investigating from current view)
  const requiresAttentionCount = statusCounts.open + statusCounts.investigating;

  // Group anomalies by device for the grouped view
  const deviceGroups = useMemo((): DeviceGroup[] => {
    if (!data) return [];

    const groupMap = new Map<number, DeviceGroup>();

    for (const anomaly of data.anomalies) {
      const existing = groupMap.get(anomaly.device_id);
      if (existing) {
        existing.anomalies.push(anomaly);
        if (anomaly.anomaly_score < existing.worstScore) {
          existing.worstScore = anomaly.anomaly_score;
        }
        if (new Date(anomaly.timestamp) > new Date(existing.latestTimestamp)) {
          existing.latestTimestamp = anomaly.timestamp;
        }
        if (anomaly.status === 'open') {
          existing.openCount++;
        }
      } else {
        groupMap.set(anomaly.device_id, {
          device_id: anomaly.device_id,
          anomalies: [anomaly],
          worstScore: anomaly.anomaly_score,
          latestTimestamp: anomaly.timestamp,
          openCount: anomaly.status === 'open' ? 1 : 0,
        });
      }
    }

    // Sort groups by worst score (most severe first)
    return Array.from(groupMap.values()).sort((a, b) => a.worstScore - b.worstScore);
  }, [data]);

  const toggleDeviceExpanded = (deviceId: number) => {
    setExpandedDevices(prev => {
      const next = new Set(prev);
      if (next.has(deviceId)) {
        next.delete(deviceId);
      } else {
        next.add(deviceId);
      }
      return next;
    });
  };

  const getSeverityKey = (score: number): keyof typeof severityConfig => {
    if (score <= -0.7) return 'critical';
    if (score <= -0.5) return 'high';
    if (score <= -0.3) return 'medium';
    return 'low';
  };

  const updateFilter = (key: string, value: string) => {
    const params = new URLSearchParams(searchParams);
    if (value) {
      params.set(key, value);
    } else {
      params.delete(key);
    }
    setSearchParams(params);
    setPage(1);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="relative w-16 h-16 mx-auto mb-4">
            <div className="absolute inset-0 rounded-full border-2 border-amber-500/20"></div>
            <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-amber-500 animate-spin"></div>
          </div>
          <p className="text-slate-400 font-mono text-sm">Loading investigations...</p>
        </div>
      </div>
    );
  }

  return (
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      {/* Header */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Investigations</h1>
          <p className="text-slate-500 mt-1">
            {statusCounts.total} anomalies {statusFilter || severityFilter ? 'matching filters' : 'detected'} • {requiresAttentionCount} requiring attention
          </p>
        </div>

        {/* View Toggle */}
        <div className="flex items-center gap-2 p-1 bg-slate-800/50 rounded-lg border border-slate-700/50">
          <button
            onClick={() => setViewMode('grouped')}
            title="Grouped by Device"
            className={`px-3 py-1.5 text-sm font-medium rounded-md transition-all ${viewMode === 'grouped'
              ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
              : 'text-slate-400 hover:text-white'
              }`}
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
          </button>
          <button
            onClick={() => setViewMode('list')}
            title="All Anomalies"
            className={`px-3 py-1.5 text-sm font-medium rounded-md transition-all ${viewMode === 'list'
              ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
              : 'text-slate-400 hover:text-white'
              }`}
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          <button
            onClick={() => setViewMode('kanban')}
            title="Kanban View"
            className={`px-3 py-1.5 text-sm font-medium rounded-md transition-all ${viewMode === 'kanban'
              ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
              : 'text-slate-400 hover:text-white'
              }`}
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7" />
            </svg>
          </button>
        </div>
      </div>

      {/* Quick Filters */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2">
          <span className="telemetry-label">Status:</span>
          {(['', 'open', 'investigating', 'resolved'] as const).map((status) => (
            <button
              key={status || 'all'}
              onClick={() => updateFilter('status', status)}
              className={`px-3 py-1.5 text-xs font-medium rounded-lg border transition-all ${statusFilter === status
                ? 'bg-amber-500/20 text-amber-400 border-amber-500/30'
                : 'text-slate-400 border-slate-700/50 hover:border-slate-600'
                }`}
            >
              {status || 'All'} {status && `(${statusCounts[status as keyof typeof statusCounts]})`}
            </button>
          ))}
        </div>

        <div className="w-px h-6 bg-slate-700/50" />

        <div className="flex items-center gap-2">
          <span className="telemetry-label">Severity:</span>
          {(['', 'critical', 'high'] as const).map((severity) => (
            <button
              key={severity || 'all'}
              onClick={() => updateFilter('severity', severity)}
              className={`px-3 py-1.5 text-xs font-medium rounded-lg border transition-all ${severityFilter === severity
                ? severity === 'critical'
                  ? 'bg-red-500/20 text-red-400 border-red-500/30'
                  : severity === 'high'
                    ? 'bg-orange-500/20 text-orange-400 border-orange-500/30'
                    : 'bg-amber-500/20 text-amber-400 border-amber-500/30'
                : 'text-slate-400 border-slate-700/50 hover:border-slate-600'
                }`}
            >
              {severity || 'All'}
            </button>
          ))}
        </div>

        {/* Device Search */}
        <div className="flex-1" />
        <div className="relative w-full sm:w-auto">
          <input
            type="text"
            placeholder="Search device ID..."
            value={deviceFilter}
            onChange={(e) => updateFilter('device', e.target.value)}
            className="input-stellar w-full sm:w-48 pl-9"
          />
          <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>

        {/* Sort */}
        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as SortBy)}
          className="select-field w-full sm:w-auto"
        >
          <option value="score">Sort by Score</option>
          <option value="time">Sort by Time</option>
          <option value="device">Sort by Device</option>
        </select>
      </div>

      {/* Content */}
      {viewMode === 'grouped' ? (
        /* Grouped by Device View */
        <Card noPadding>
          <div className="divide-y divide-slate-800/50">
            <AnimatePresence mode="popLayout">
              {deviceGroups.map((group, groupIndex) => {
                const severityKey = getSeverityKey(group.worstScore);
                const severity = severityConfig[severityKey];
                const isExpanded = expandedDevices.has(group.device_id);
                const hasMultiple = group.anomalies.length > 1;

                // For single anomaly devices, make the whole row a Link
                const DeviceRowWrapper = hasMultiple ? 'div' : Link;
                const wrapperProps = hasMultiple
                  ? {
                      onClick: () => toggleDeviceExpanded(group.device_id),
                      className: "flex items-center gap-6 p-5 transition-colors cursor-pointer hover:bg-slate-800/30 group"
                    }
                  : {
                      to: `/investigations/${group.anomalies[0].id}`,
                      className: "flex items-center gap-6 p-5 transition-colors cursor-pointer hover:bg-slate-800/30 group"
                    };

                return (
                  <motion.div
                    key={group.device_id}
                    layout
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ delay: groupIndex * 0.02 }}
                  >
                    {/* Device Header - clickable for all devices */}
                    <DeviceRowWrapper {...wrapperProps as any}>
                      {/* Severity Bar */}
                      <div className={`w-1.5 h-16 rounded-full ${severity.barColor}`} />

                      {/* Expand/Collapse Icon or View Icon */}
                      <div className="w-6 flex-shrink-0">
                        {hasMultiple ? (
                          <svg
                            className={`w-5 h-5 text-slate-400 transition-transform ${isExpanded ? 'rotate-90' : ''}`}
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                          >
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                          </svg>
                        ) : (
                          <svg
                            className="w-5 h-5 text-slate-500 group-hover:text-amber-400 transition-colors"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                          >
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                          </svg>
                        )}
                      </div>

                      {/* Main Content */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-3 mb-2">
                          <span className="text-lg font-semibold text-white group-hover:text-amber-400 transition-colors">
                            Device #{group.device_id}
                          </span>
                          <span className={`px-2 py-0.5 text-[10px] font-bold rounded ${severity.badgeBg} ${severity.badgeText}`}>
                            {severity.label}
                          </span>
                          {hasMultiple && (
                            <span className="px-2 py-0.5 text-[10px] font-bold rounded bg-amber-500/20 text-amber-400 border border-amber-500/30">
                              {group.anomalies.length} anomalies
                            </span>
                          )}
                          {group.openCount > 0 && (
                            <span className="px-2 py-0.5 text-[10px] font-bold rounded bg-red-500/20 text-red-400">
                              {group.openCount} open
                            </span>
                          )}
                        </div>

                        {/* Summary for grouped */}
                        <div className="mb-2">
                          <p className="text-sm text-slate-300 font-medium">
                            {hasMultiple
                              ? `${group.anomalies.length} anomalies detected - worst score: ${group.worstScore.toFixed(4)}`
                              : getAnomalyReason(group.anomalies[0])
                            }
                          </p>
                        </div>

                        <div className="flex items-center gap-4 text-sm text-slate-500">
                          <span>Latest: {format(new Date(group.latestTimestamp), 'MMM d, HH:mm')}</span>
                          <span>•</span>
                          <span>{formatDistanceToNowStrict(new Date(group.latestTimestamp))} ago</span>
                        </div>
                      </div>

                      {/* Action indicator */}
                      <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                        <span className="text-xs text-amber-400 font-medium">
                          {hasMultiple ? (isExpanded ? 'Collapse' : 'Expand') : 'Investigate'}
                        </span>
                        <svg className="w-4 h-4 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                      </div>
                    </DeviceRowWrapper>

                    {/* Expanded Anomalies List */}
                    <AnimatePresence>
                      {isExpanded && hasMultiple && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          exit={{ opacity: 0, height: 0 }}
                          className="bg-slate-900/50 border-l-2 border-amber-500/30 ml-8"
                        >
                          {group.anomalies.map((anomaly) => {
                            const anomalySeverity = severityConfig[getSeverityKey(anomaly.anomaly_score)];
                            const status = statusConfig[anomaly.status as keyof typeof statusConfig] || statusConfig.open;

                            return (
                              <Link
                                key={anomaly.id}
                                to={`/investigations/${anomaly.id}`}
                                className="flex items-center gap-4 px-5 py-3 hover:bg-slate-800/30 transition-colors group border-b border-slate-800/30 last:border-b-0"
                              >
                                <div className={`w-1 h-8 rounded-full ${anomalySeverity.barColor}`} />
                                <div className="flex-1 min-w-0">
                                  <div className="flex items-center gap-2 mb-1">
                                    <span className={`px-1.5 py-0.5 text-[9px] font-bold rounded ${anomalySeverity.badgeBg} ${anomalySeverity.badgeText}`}>
                                      {anomalySeverity.label}
                                    </span>
                                    <span className={`px-1.5 py-0.5 text-[9px] font-bold rounded ${status.badgeBg} ${status.badgeText}`}>
                                      {status.label}
                                    </span>
                                    <span className="text-xs text-slate-500 font-mono">
                                      Score: {anomaly.anomaly_score.toFixed(4)}
                                    </span>
                                  </div>
                                  <p className="text-xs text-slate-400">{getAnomalyReason(anomaly)}</p>
                                  <p className="text-xs text-slate-600 mt-1">
                                    {format(new Date(anomaly.timestamp), 'MMM d, HH:mm')} • {formatDistanceToNowStrict(new Date(anomaly.timestamp))} ago
                                  </p>
                                </div>
                                <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                  <span className="text-xs text-amber-400 font-medium">View</span>
                                  <svg className="w-3 h-3 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                  </svg>
                                </div>
                              </Link>
                            );
                          })}
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </motion.div>
                );
              })}
            </AnimatePresence>

            {deviceGroups.length === 0 && (
              <div className="p-12 text-center">
                <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-emerald-500/10 flex items-center justify-center">
                  <svg className="w-8 h-8 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <p className="text-xl font-medium text-slate-300">No anomalies found</p>
                <p className="text-slate-500 mt-1">
                  {statusFilter || severityFilter ? 'Try adjusting your filters' : 'All systems operating normally'}
                </p>
              </div>
            )}
          </div>

          {/* Pagination */}
          {data && data.total_pages > 1 && (
            <div className="px-5 py-4 border-t border-slate-800/50 flex items-center justify-between">
              <p className="text-sm text-slate-500 font-mono">
                {deviceGroups.length} devices with {data.total} total anomalies
              </p>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setPage(Math.max(1, page - 1))}
                  disabled={page === 1}
                  className="btn-ghost text-sm disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  Previous
                </button>
                <span className="px-3 py-1.5 text-sm text-slate-500 font-mono">
                  {page} / {data.total_pages}
                </span>
                <button
                  onClick={() => setPage(Math.min(data.total_pages, page + 1))}
                  disabled={page === data.total_pages}
                  className="btn-ghost text-sm disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </Card>
      ) : viewMode === 'list' ? (
        <Card noPadding>
          <div className="divide-y divide-slate-800/50">
            <AnimatePresence mode="popLayout">
              {data?.anomalies.map((anomaly, index) => {
                const severityKey = getSeverityKey(anomaly.anomaly_score);
                const severity = severityConfig[severityKey];
                const status = statusConfig[anomaly.status as keyof typeof statusConfig] || statusConfig.open;

                return (
                  <motion.div
                    key={anomaly.id}
                    layout
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ delay: index * 0.02 }}
                  >
                    <Link
                      to={`/investigations/${anomaly.id}`}
                      className="flex items-center gap-6 p-5 hover:bg-slate-800/30 transition-colors group"
                    >
                      {/* Severity Bar */}
                      <div className={`w-1.5 h-16 rounded-full ${severity.barColor}`} />

                      {/* Main Content */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-3 mb-2">
                          <span className="text-lg font-semibold text-white">
                            Device #{anomaly.device_id}
                          </span>
                          <span className={`px-2 py-0.5 text-[10px] font-bold rounded ${severity.badgeBg} ${severity.badgeText}`}>
                            {severity.label}
                          </span>
                          <span className={`px-2 py-0.5 text-[10px] font-bold rounded ${status.badgeBg} ${status.badgeText}`}>
                            {status.label}
                          </span>
                        </div>

                        {/* Anomaly Reason */}
                        <div className="mb-2">
                          <p className="text-sm text-slate-300 font-medium">
                            {getAnomalyReason(anomaly)}
                          </p>
                        </div>

                        <div className="flex items-center gap-4 text-sm text-slate-500">
                          <span className="font-mono">Score: {anomaly.anomaly_score.toFixed(4)}</span>
                          <span>•</span>
                          <span>{format(new Date(anomaly.timestamp), 'MMM d, HH:mm')}</span>
                          <span>•</span>
                          <span>{formatDistanceToNowStrict(new Date(anomaly.timestamp))} ago</span>
                        </div>
                      </div>

                      {/* Quick Metrics Preview */}
                      <div className="hidden lg:flex items-center gap-6 text-xs">
                        {anomaly.total_battery_level_drop !== null && (
                          <div className="text-center">
                            <p className="font-mono text-slate-300">{anomaly.total_battery_level_drop?.toFixed(1)}%</p>
                            <p className="text-slate-600">Battery</p>
                          </div>
                        )}
                        {anomaly.offline_time !== null && (
                          <div className="text-center">
                            <p className="font-mono text-slate-300">{anomaly.offline_time?.toFixed(0)}m</p>
                            <p className="text-slate-600">Offline</p>
                          </div>
                        )}
                      </div>

                      {/* Action */}
                      <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                        <span className="text-xs text-amber-400 font-medium">Investigate</span>
                        <svg className="w-4 h-4 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                      </div>
                    </Link>
                  </motion.div>
                );
              })}
            </AnimatePresence>

            {(!data || data.anomalies.length === 0) && (
              <div className="p-12 text-center">
                <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-emerald-500/10 flex items-center justify-center">
                  <svg className="w-8 h-8 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <p className="text-xl font-medium text-slate-300">No anomalies found</p>
                <p className="text-slate-500 mt-1">
                  {statusFilter || severityFilter ? 'Try adjusting your filters' : 'All systems operating normally'}
                </p>
              </div>
            )}
          </div>

          {/* Pagination */}
          {data && data.total_pages > 1 && (
            <div className="px-5 py-4 border-t border-slate-800/50 flex items-center justify-between">
              <p className="text-sm text-slate-500 font-mono">
                Showing {((page - 1) * 50) + 1}-{Math.min(page * 50, data.total)} of {data.total}
              </p>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setPage(Math.max(1, page - 1))}
                  disabled={page === 1}
                  className="btn-ghost text-sm disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  Previous
                </button>
                <span className="px-3 py-1.5 text-sm text-slate-500 font-mono">
                  {page} / {data.total_pages}
                </span>
                <button
                  onClick={() => setPage(Math.min(data.total_pages, page + 1))}
                  disabled={page === data.total_pages}
                  className="btn-ghost text-sm disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </Card>
      ) : (
        /* Kanban View */
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
          {(['open', 'investigating', 'resolved', 'false_positive'] as const).map((statusKey) => {
            const status = statusConfig[statusKey];
            const columnItems = data?.anomalies.filter(a => a.status === statusKey) || [];

            return (
              <div key={statusKey} className="space-y-3">
                {/* Column Header */}
                <div className={`flex items-center justify-between p-3 rounded-lg ${status.headerBg} border ${status.badgeBorder}`}>
                  <span className={`text-sm font-semibold ${status.headerText}`}>
                    {status.label}
                  </span>
                  <span className={`px-2 py-0.5 text-xs font-bold ${status.headerText}`}>
                    {columnItems.length}
                  </span>
                </div>

                {/* Column Content */}
                <div className="space-y-2 min-h-[400px]">
                  <AnimatePresence>
                    {columnItems.slice(0, 8).map((anomaly, index) => {
                      const severityKey = getSeverityKey(anomaly.anomaly_score);
                      const severity = severityConfig[severityKey];

                      return (
                        <motion.div
                          key={anomaly.id}
                          layout
                          initial={{ opacity: 0, scale: 0.9 }}
                          animate={{ opacity: 1, scale: 1 }}
                          exit={{ opacity: 0, scale: 0.9 }}
                          transition={{ delay: index * 0.02 }}
                        >
                          <Link
                            to={`/investigations/${anomaly.id}`}
                            className="block p-3 stellar-card rounded-xl hover:border-amber-500/30 transition-all group"
                          >
                            <div className="flex items-center gap-2 mb-2">
                              <div className={`w-2 h-2 rounded-full ${severity.barColor}`} />
                              <span className="text-sm font-semibold text-white">
                                Device #{anomaly.device_id}
                              </span>
                            </div>
                            <div className="flex items-center justify-between text-xs">
                              <span className={`font-mono ${severity.badgeText}`}>
                                {anomaly.anomaly_score.toFixed(3)}
                              </span>
                              <span className="text-slate-500">
                                {formatDistanceToNowStrict(new Date(anomaly.timestamp))}
                              </span>
                            </div>
                          </Link>
                        </motion.div>
                      );
                    })}
                  </AnimatePresence>

                  {columnItems.length > 8 && (
                    <p className="text-center text-xs text-slate-500 py-2">
                      +{columnItems.length - 8} more
                    </p>
                  )}

                  {columnItems.length === 0 && (
                    <div className="p-4 text-center text-xs text-slate-600">
                      No items
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </motion.div>
  );
}

export default Investigations;
