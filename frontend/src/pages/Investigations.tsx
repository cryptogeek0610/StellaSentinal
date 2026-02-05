/**
 * Investigations Page - Steve Jobs Redesign v2
 *
 * Philosophy: "When I open this page, what's the ONE thing I should do?"
 *
 * The answer should be OBVIOUS:
 * - First group auto-expanded (no clicks needed)
 * - Filters hidden by default (most users want "Needs Action")
 * - Hero card IS the interface - single dominant CTA
 * - The page tells you what to do, not asks you to choose
 */

import { useState, useCallback, useEffect, useMemo } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '../api/client';
import { formatDistanceToNowStrict } from 'date-fns';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from '../components/Card';
import { AnomalyGroupCard } from '../components/unified/AnomalyGroupCard';
import { ImpactSummaryCard } from '../components/ImpactSummaryCard';
import { SuccessToast } from '../components/SuccessToast';
import { useKeyboardShortcuts } from '../hooks/useKeyboardShortcuts';
import {
  SEVERITY_CONFIGS,
  STATUS_CONFIGS,
  getSeverityFromScore,
  type SeverityLevel,
  type StatusLevel,
} from '../utils/severity';

import { AnomalyGroup, AnomalyGroupMember } from '../types/anomaly';

// Derive local configs from centralized source for component use
const severityConfig = Object.fromEntries(
  Object.entries(SEVERITY_CONFIGS).map(([key, cfg]) => [
    key,
    { label: cfg.label, barColor: cfg.color.dot, badgeBg: cfg.color.bg, badgeText: cfg.color.text },
  ])
) as Record<SeverityLevel, { label: string; barColor: string; badgeBg: string; badgeText: string }>;

const statusConfig = Object.fromEntries(
  Object.entries(STATUS_CONFIGS).map(([key, cfg]) => [
    key,
    { label: cfg.label, badgeBg: cfg.bg, badgeText: cfg.color },
  ])
) as Record<StatusLevel, { label: string; badgeBg: string; badgeText: string }>;

function Investigations() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());
  const [selectedAnomalies, setSelectedAnomalies] = useState<Map<string, Set<number>>>(new Map());
  const [isProcessingBulk, setIsProcessingBulk] = useState(false);
  const [focusedGroupIndex, setFocusedGroupIndex] = useState(0);
  const [successToast, setSuccessToast] = useState<{ message: string; count?: number } | null>(null);
  const [showFilters, setShowFilters] = useState(false);
  const [hasAutoExpanded, setHasAutoExpanded] = useState(false);

  const queryClient = useQueryClient();

  // Default to 'open' (Needs Action) - no choice needed
  const statusFilter = searchParams.get('status') ?? 'open';
  const severityFilter = searchParams.get('severity') || '';

  // Show filters if user has explicitly filtered
  const hasActiveFilters = statusFilter !== 'open' || severityFilter !== '';

  const { data: groupedData, isLoading: isLoadingGroups } = useQuery({
    queryKey: ['anomalies-grouped', statusFilter, severityFilter],
    queryFn: () => api.getGroupedAnomalies({
      status: statusFilter || undefined,
      min_severity: severityFilter || undefined,
      min_group_size: 2,
    }),
    refetchInterval: 30000,
  });

  // AUTO-EXPAND first group when data loads - the page should be ready to act
  useEffect(() => {
    if (!hasAutoExpanded && groupedData?.groups && groupedData.groups.length > 0) {
      const firstGroup = groupedData.groups[0];
      setExpandedGroups(new Set([firstGroup.group_id]));
      setHasAutoExpanded(true);
    }
  }, [groupedData?.groups, hasAutoExpanded]);

  const groups = useMemo(() => groupedData?.groups || [], [groupedData?.groups]);
  const totalAnomalies = groupedData?.total_anomalies ?? 0;

  // Handlers
  const toggleGroupExpanded = useCallback((groupId: string) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(groupId)) next.delete(groupId);
      else next.add(groupId);
      return next;
    });
  }, []);

  const getSelectedForGroup = useCallback((groupId: string): Set<number> => {
    return selectedAnomalies.get(groupId) || new Set();
  }, [selectedAnomalies]);

  const handleSelectAnomaly = useCallback((groupId: string, anomalyId: number, selected: boolean) => {
    setSelectedAnomalies((prev) => {
      const next = new Map(prev);
      const groupSet = new Set(next.get(groupId) || []);
      if (selected) groupSet.add(anomalyId);
      else groupSet.delete(anomalyId);
      next.set(groupId, groupSet);
      return next;
    });
  }, []);

  const handleSelectAllInGroup = useCallback((groupId: string, group: AnomalyGroup, selected: boolean) => {
    setSelectedAnomalies((prev) => {
      const next = new Map(prev);
      if (selected) {
        next.set(groupId, new Set(group.sample_anomalies.map((a) => a.anomaly_id)));
      } else {
        next.set(groupId, new Set());
      }
      return next;
    });
  }, []);

  const handleBulkAction = useCallback(async (groupId: string, action: 'resolve' | 'investigating' | 'dismiss') => {
    const selectedIds = selectedAnomalies.get(groupId);
    if (!selectedIds || selectedIds.size === 0) return;

    const count = selectedIds.size;
    setIsProcessingBulk(true);
    try {
      await api.bulkAction({ action, anomaly_ids: Array.from(selectedIds) });
      setSelectedAnomalies((prev) => { const next = new Map(prev); next.set(groupId, new Set()); return next; });
      setSuccessToast({ message: action === 'resolve' ? 'Resolved' : action === 'investigating' ? 'Marked for investigation' : 'Dismissed', count });
      queryClient.invalidateQueries({ queryKey: ['anomalies'] });
      queryClient.invalidateQueries({ queryKey: ['anomalies-grouped'] });
      queryClient.invalidateQueries({ queryKey: ['dashboard', 'stats'] });
    } catch (error) {
      console.error('Bulk action failed:', error);
    } finally {
      setIsProcessingBulk(false);
    }
  }, [selectedAnomalies, queryClient]);

  const handleAnomalyClick = useCallback((anomaly: AnomalyGroupMember) => {
    window.location.href = `/investigations/${anomaly.anomaly_id}`;
  }, []);

  const handleStartWithGroup = useCallback((groupId: string) => {
    setExpandedGroups((prev) => new Set(prev).add(groupId));
    setTimeout(() => {
      document.getElementById(`group-${groupId}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  }, []);

  // Keyboard navigation
  const handleKeyboardNext = useCallback(() => {
    if (groups.length === 0) return;
    setFocusedGroupIndex((prev) => {
      const next = Math.min(prev + 1, groups.length - 1);
      document.getElementById(`group-${groups[next]?.group_id}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      return next;
    });
  }, [groups]);

  const handleKeyboardPrevious = useCallback(() => {
    if (groups.length === 0) return;
    setFocusedGroupIndex((prev) => {
      const next = Math.max(prev - 1, 0);
      document.getElementById(`group-${groups[next]?.group_id}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      return next;
    });
  }, [groups]);

  const handleKeyboardExpand = useCallback(() => {
    if (groups.length === 0) return;
    const group = groups[focusedGroupIndex];
    if (group) toggleGroupExpanded(group.group_id);
  }, [groups, focusedGroupIndex, toggleGroupExpanded]);

  const handleKeyboardFix = useCallback(() => {
    if (groups.length === 0) return;
    const group = groups[focusedGroupIndex];
    if (group?.suggested_remediation) {
      handleSelectAllInGroup(group.group_id, group, true);
      handleBulkAction(group.group_id, 'resolve');
    }
  }, [groups, focusedGroupIndex, handleSelectAllInGroup, handleBulkAction]);

  useKeyboardShortcuts({
    onNext: handleKeyboardNext,
    onPrevious: handleKeyboardPrevious,
    onExpand: handleKeyboardExpand,
    onFix: handleKeyboardFix,
    onEscape: () => setExpandedGroups(new Set()),
    enabled: true,
  });

  const getSeverityKey = (score: number): SeverityLevel => {
    return getSeverityFromScore(score).level;
  };

  const updateFilter = (key: string, value: string) => {
    const params = new URLSearchParams(searchParams);
    if (value) params.set(key, value);
    else params.delete(key);
    setSearchParams(params);
    setHasAutoExpanded(false); // Reset auto-expand when filters change
  };

  return (
    <motion.div className="space-y-6" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
      {/* Header - Minimal */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Investigations</h1>
          {totalAnomalies > 0 && (
            <p className="text-sm text-slate-500 mt-0.5">
              {totalAnomalies} anomalies to review
            </p>
          )}
        </div>

        {/* Subtle filter toggle - not in your face */}
        <button
          onClick={() => setShowFilters(!showFilters)}
          className="text-xs text-slate-500 hover:text-slate-300 transition-colors flex items-center gap-1"
        >
          {hasActiveFilters && <span className="w-1.5 h-1.5 bg-amber-400 rounded-full" />}
          {showFilters ? 'Hide filters' : 'Filter'}
          <svg className={`w-3 h-3 transition-transform ${showFilters ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>
      </div>

      {/* Collapsible Filters - Hidden by default */}
      <AnimatePresence>
        {showFilters && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="flex flex-wrap items-center gap-3 pb-2">
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-600">Status:</span>
                {[
                  { value: '', label: 'All' },
                  { value: 'open', label: 'Needs Action' },
                  { value: 'investigating', label: 'In Progress' },
                  { value: 'resolved', label: 'Fixed' },
                ].map((opt) => (
                  <button
                    key={opt.value}
                    onClick={() => updateFilter('status', opt.value)}
                    className={`px-2 py-1 text-xs rounded transition-all ${
                      (statusFilter || '') === opt.value
                        ? 'bg-slate-700 text-white'
                        : 'text-slate-500 hover:text-slate-300'
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
              <div className="w-px h-4 bg-slate-700/50" />
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-600">Severity:</span>
                {[
                  { value: '', label: 'All' },
                  { value: 'critical', label: 'Critical' },
                  { value: 'high', label: 'High' },
                ].map((opt) => (
                  <button
                    key={opt.value}
                    onClick={() => updateFilter('severity', opt.value)}
                    className={`px-2 py-1 text-xs rounded transition-all ${
                      severityFilter === opt.value
                        ? 'bg-slate-700 text-white'
                        : 'text-slate-500 hover:text-slate-300'
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Hero Card - THE Interface */}
      <ImpactSummaryCard
        totalGroups={groupedData?.total_groups ?? 0}
        coveragePercent={groupedData?.coverage_percent ?? 0}
        totalAnomalies={totalAnomalies}
        groupedCount={(groupedData?.total_anomalies ?? 0) - (groupedData?.ungrouped_count ?? 0)}
        topGroupName={groupedData?.top_impact_group_name}
        topGroupId={groupedData?.top_impact_group_id}
        onStartAction={handleStartWithGroup}
      />

      {/* Content */}
      <div className="space-y-3">
        {isLoadingGroups && (
          <Card className="p-8">
            <div className="flex items-center justify-center">
              <div className="relative w-10 h-10 mr-3">
                <div className="absolute inset-0 rounded-full border-2 border-amber-500/20" />
                <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-amber-500 animate-spin" />
              </div>
              <p className="text-slate-400 text-sm">Analyzing patterns...</p>
            </div>
          </Card>
        )}

        {!isLoadingGroups && groupedData && (
          <>
            {/* Groups - First one auto-expanded */}
            <AnimatePresence mode="popLayout">
              {groups.map((group, index) => (
                <motion.div
                  key={group.group_id}
                  id={`group-${group.group_id}`}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ delay: index * 0.02 }}
                  className={index === focusedGroupIndex ? 'ring-2 ring-amber-500/40 rounded-xl' : ''}
                >
                  <AnomalyGroupCard
                    group={group}
                    isExpanded={expandedGroups.has(group.group_id)}
                    onToggle={() => { toggleGroupExpanded(group.group_id); setFocusedGroupIndex(index); }}
                    selectedAnomalies={getSelectedForGroup(group.group_id)}
                    onSelectAnomaly={(id, selected) => handleSelectAnomaly(group.group_id, id, selected)}
                    onSelectAll={(selected) => handleSelectAllInGroup(group.group_id, group, selected)}
                    onBulkAction={(action) => handleBulkAction(group.group_id, action)}
                    onAnomalyClick={handleAnomalyClick}
                    isLoading={isProcessingBulk}
                  />
                </motion.div>
              ))}
            </AnimatePresence>

            {/* Unique Cases - Collapsed by default */}
            {groupedData.ungrouped_anomalies && groupedData.ungrouped_anomalies.length > 0 && (
              <details className="group mt-6">
                <summary className="flex items-center gap-2 cursor-pointer list-none text-slate-500 hover:text-slate-300 transition-colors">
                  <svg className="w-4 h-4 transition-transform group-open:rotate-90" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                  <span className="text-sm">{groupedData.ungrouped_count} unique cases (no pattern match)</span>
                </summary>
                <Card noPadding className="mt-3">
                  <div className="divide-y divide-slate-800/50">
                    {groupedData.ungrouped_anomalies.slice(0, 10).map((anomaly) => {
                      const severityKey = getSeverityKey(anomaly.anomaly_score);
                      const severity = severityConfig[severityKey];
                      const status = statusConfig[anomaly.status as keyof typeof statusConfig] || statusConfig.open;
                      return (
                        <Link
                          key={anomaly.anomaly_id}
                          to={`/investigations/${anomaly.anomaly_id}`}
                          className="flex items-center gap-4 p-4 hover:bg-slate-800/30 transition-colors group"
                        >
                          <div className={`w-1 h-8 rounded-full ${severity.barColor}`} />
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-medium text-white">{anomaly.device_name || `Device #${anomaly.device_id}`}</span>
                              <span className={`px-1.5 py-0.5 text-[9px] font-bold rounded ${severity.badgeBg} ${severity.badgeText}`}>
                                {severity.label}
                              </span>
                              <span className={`px-1.5 py-0.5 text-[9px] font-bold rounded ${status.badgeBg} ${status.badgeText}`}>
                                {status.label}
                              </span>
                            </div>
                            <span className="text-xs text-slate-500">{formatDistanceToNowStrict(new Date(anomaly.timestamp))} ago</span>
                          </div>
                          <svg className="w-4 h-4 text-slate-600 group-hover:text-amber-400 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                          </svg>
                        </Link>
                      );
                    })}
                    {groupedData.ungrouped_anomalies.length > 10 && (
                      <div className="p-3 text-center text-xs text-slate-500">
                        +{groupedData.ungrouped_count - 10} more
                      </div>
                    )}
                  </div>
                </Card>
              </details>
            )}

            {/* Empty state */}
            {groups.length === 0 && groupedData.ungrouped_count === 0 && (
              <Card className="p-12 text-center">
                <div className="w-14 h-14 mx-auto mb-4 rounded-full bg-emerald-500/10 flex items-center justify-center">
                  <svg className="w-7 h-7 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <p className="text-lg font-medium text-slate-300">All clear</p>
                <p className="text-sm text-slate-500 mt-1">No anomalies need attention</p>
              </Card>
            )}
          </>
        )}
      </div>

      {successToast && (
        <SuccessToast message={successToast.message} count={successToast.count} onClose={() => setSuccessToast(null)} />
      )}
    </motion.div>
  );
}

export default Investigations;
