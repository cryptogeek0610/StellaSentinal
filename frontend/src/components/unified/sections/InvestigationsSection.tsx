/**
 * InvestigationsSection - Collapsible section for anomaly investigations
 *
 * Shows open investigations, critical issues, and priority queue.
 * Expands to show full investigation list with filtering.
 * Supports Smart Groups view for better anomaly organization.
 */

import { useState, useCallback } from 'react';
import { Link } from 'react-router-dom';
import clsx from 'clsx';
import { format, formatDistanceToNowStrict } from 'date-fns';
import { CollapsibleSection } from '../CollapsibleSection';
import { SlideOverPanel } from '../SlideOverPanel';
import { AnomalyGroupCard } from '../AnomalyGroupCard';
import type { Anomaly, AnomalyGroup, AnomalyGroupMember } from '../../../types/anomaly';
import {
  SEVERITY_CONFIGS,
  STATUS_CONFIGS,
  getSeverityFromScore as getSeverityConfig,
  type SeverityLevel,
  type StatusLevel,
} from '../../../utils/severity';

type ViewMode = 'groups' | 'list';

interface InvestigationsSectionProps {
  anomalies: Anomaly[];
  groups?: AnomalyGroup[];
  stats: {
    open_cases: number;
    critical_issues: number;
    detected_today: number;
    resolved_today: number;
  } | undefined;
  isLoading?: boolean;
  onBulkAction?: (action: string, anomalyIds: number[]) => Promise<void>;
}

// Derived configs from centralized source for local component use
const severityConfig = Object.fromEntries(
  Object.entries(SEVERITY_CONFIGS).map(([key, cfg]) => [
    key,
    { label: cfg.label, color: cfg.color.text, bg: cfg.color.bg, border: cfg.color.border },
  ])
) as Record<SeverityLevel, { label: string; color: string; bg: string; border: string }>;

const getSeverityFromScore = (score: number): SeverityLevel => {
  return getSeverityConfig(score).level;
};

const statusConfig = Object.fromEntries(
  Object.entries(STATUS_CONFIGS).map(([key, cfg]) => [
    key,
    { label: cfg.label, color: cfg.color, bg: cfg.bg },
  ])
) as Record<StatusLevel, { label: string; color: string; bg: string }>;

function AnomalyRow({ anomaly, onClick }: { anomaly: Anomaly; onClick: () => void }) {
  const severity = getSeverityFromScore(anomaly.anomaly_score);
  const severityStyle = severityConfig[severity];
  const statusStyle = statusConfig[anomaly.status as keyof typeof statusConfig] || statusConfig.open;

  return (
    <button
      onClick={onClick}
      className="w-full flex items-center gap-3 p-3 rounded-lg bg-slate-800/30 hover:bg-slate-800/50 border border-slate-700/30 hover:border-slate-600/50 transition-all group text-left"
    >
      {/* Severity indicator */}
      <div className={clsx('w-1 h-10 rounded-full', severityStyle.bg)} />

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-white font-medium text-sm">
            {anomaly.device_name || `Device #${anomaly.device_id}`}
          </span>
          <span className={clsx(
            'px-1.5 py-0.5 rounded text-xs font-medium',
            severityStyle.bg,
            severityStyle.color
          )}>
            {severityStyle.label}
          </span>
        </div>
        <div className="flex items-center gap-2 mt-1">
          <span className="text-xs text-slate-500">
            {formatDistanceToNowStrict(new Date(anomaly.timestamp), { addSuffix: true })}
          </span>
          <span className="text-slate-700">|</span>
          <span className={clsx('text-xs', statusStyle.color)}>
            {statusStyle.label}
          </span>
        </div>
      </div>

      {/* Score and Cost */}
      <div className="text-right">
        <span className={clsx('font-mono text-sm', severityStyle.color)}>
          {anomaly.anomaly_score.toFixed(3)}
        </span>
        {anomaly.cost_impact && anomaly.cost_impact.hourly_cost > 0 && (
          <div className="text-xs text-amber-400 mt-0.5">
            ${anomaly.cost_impact.hourly_cost.toFixed(0)}/hr
          </div>
        )}
      </div>

      {/* Arrow */}
      <svg className="w-4 h-4 text-slate-500 group-hover:text-slate-400 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
      </svg>
    </button>
  );
}

function LoadingSkeleton() {
  return (
    <div className="space-y-2">
      {[1, 2, 3].map((i) => (
        <div key={i} className="flex items-center gap-3 p-3 rounded-lg bg-slate-800/30 animate-pulse">
          <div className="w-1 h-10 rounded-full bg-slate-700/50" />
          <div className="flex-1 space-y-2">
            <div className="h-4 w-24 bg-slate-700/50 rounded" />
            <div className="h-3 w-32 bg-slate-700/50 rounded" />
          </div>
          <div className="h-4 w-12 bg-slate-700/50 rounded" />
        </div>
      ))}
    </div>
  );
}

export function InvestigationsSection({
  anomalies,
  groups,
  stats,
  isLoading,
  onBulkAction,
}: InvestigationsSectionProps) {
  const [selectedAnomaly, setSelectedAnomaly] = useState<Anomaly | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('groups');
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());
  const [selectedAnomaliesByGroup, setSelectedAnomaliesByGroup] = useState<Map<string, Set<number>>>(new Map());

  const openCount = stats?.open_cases || 0;
  const criticalCount = stats?.critical_issues || 0;

  // Split anomalies by status
  const openAnomalies = anomalies.filter(a => a.status === 'open' || a.status === 'investigating');
  const criticalAnomalies = openAnomalies.filter(a => getSeverityFromScore(a.anomaly_score) === 'critical');

  // Toggle group expansion
  const toggleGroup = useCallback((groupId: string) => {
    setExpandedGroups(prev => {
      const next = new Set(prev);
      if (next.has(groupId)) {
        next.delete(groupId);
      } else {
        next.add(groupId);
      }
      return next;
    });
  }, []);

  // Handle anomaly selection within a group
  const handleSelectAnomaly = useCallback((groupId: string, anomalyId: number, selected: boolean) => {
    setSelectedAnomaliesByGroup(prev => {
      const next = new Map(prev);
      const groupSet = new Set(prev.get(groupId) || []);
      if (selected) {
        groupSet.add(anomalyId);
      } else {
        groupSet.delete(anomalyId);
      }
      next.set(groupId, groupSet);
      return next;
    });
  }, []);

  // Handle select all in a group
  const handleSelectAll = useCallback((groupId: string, group: AnomalyGroup, selected: boolean) => {
    setSelectedAnomaliesByGroup(prev => {
      const next = new Map(prev);
      if (selected) {
        next.set(groupId, new Set(group.sample_anomalies.map(a => a.anomaly_id)));
      } else {
        next.set(groupId, new Set());
      }
      return next;
    });
  }, []);

  // Handle bulk action for a group
  const handleBulkAction = useCallback(async (groupId: string, action: 'resolve' | 'investigating' | 'dismiss') => {
    const selectedIds = selectedAnomaliesByGroup.get(groupId);
    if (!selectedIds || selectedIds.size === 0) return;

    if (onBulkAction) {
      await onBulkAction(action, Array.from(selectedIds));
      // Clear selection after action
      setSelectedAnomaliesByGroup(prev => {
        const next = new Map(prev);
        next.set(groupId, new Set());
        return next;
      });
    }
  }, [selectedAnomaliesByGroup, onBulkAction]);

  // Handle clicking on a group member
  const handleGroupMemberClick = useCallback((member: AnomalyGroupMember) => {
    // Find the corresponding anomaly in the full list
    const anomaly = anomalies.find(a => a.id === member.anomaly_id);
    if (anomaly) {
      setSelectedAnomaly(anomaly);
    }
  }, [anomalies]);

  // Collapsed content - show either group summaries or anomaly list
  const collapsedContent = isLoading ? (
    <LoadingSkeleton />
  ) : openAnomalies.length === 0 && (!groups || groups.length === 0) ? (
    <div className="flex items-center justify-center py-6 text-slate-500">
      <svg className="w-5 h-5 mr-2 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
      </svg>
      No open investigations
    </div>
  ) : groups && groups.length > 0 ? (
    // Show group summaries in collapsed view
    <div className="space-y-2">
      {groups.slice(0, 3).map((group) => (
        <div
          key={group.group_id}
          className="flex items-center gap-3 p-2 rounded-lg bg-slate-800/30 border border-slate-700/30"
        >
          <div className={clsx(
            'w-1 h-8 rounded-full',
            group.severity === 'critical' ? 'bg-red-500' :
            group.severity === 'high' ? 'bg-orange-500' :
            group.severity === 'medium' ? 'bg-amber-500' : 'bg-slate-500'
          )} />
          <div className="flex-1 min-w-0">
            <span className="text-sm text-white truncate block">{group.group_name}</span>
            <span className="text-xs text-slate-500">{group.device_count} devices</span>
          </div>
          {group.suggested_remediation && (
            <span className="px-2 py-0.5 text-xs rounded bg-emerald-500/20 text-emerald-400">
              Fix Available
            </span>
          )}
        </div>
      ))}
      {groups.length > 3 && (
        <Link
          to="/investigations?view=grouped"
          className="block text-center py-2 text-xs text-slate-400 hover:text-amber-400 transition-colors"
        >
          View all {groups.length} groups
        </Link>
      )}
    </div>
  ) : (
    // Fallback to anomaly list
    <div className="space-y-2">
      {openAnomalies.slice(0, 3).map((anomaly) => (
        <AnomalyRow
          key={anomaly.id}
          anomaly={anomaly}
          onClick={() => setSelectedAnomaly(anomaly)}
        />
      ))}
      {openAnomalies.length > 3 && (
        <Link
          to="/investigations?status=open"
          className="block text-center py-2 text-xs text-slate-400 hover:text-amber-400 transition-colors"
        >
          View all {openAnomalies.length} open investigations
        </Link>
      )}
    </div>
  );

  const expandedContent = (
    <div className="space-y-4">
      {/* Metrics row */}
      <div className="grid grid-cols-4 gap-3">
        <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/30">
          <div className="text-2xl font-bold font-mono text-white">{openCount}</div>
          <div className="text-xs text-slate-500">Open Cases</div>
        </div>
        <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20">
          <div className="text-2xl font-bold font-mono text-red-400">{criticalCount}</div>
          <div className="text-xs text-slate-500">Critical</div>
        </div>
        <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/30">
          <div className="text-2xl font-bold font-mono text-amber-400">{stats?.detected_today || 0}</div>
          <div className="text-xs text-slate-500">Today</div>
        </div>
        <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/30">
          <div className="text-2xl font-bold font-mono text-emerald-400">{stats?.resolved_today || 0}</div>
          <div className="text-xs text-slate-500">Resolved</div>
        </div>
      </div>

      {/* View toggle - only show if we have groups */}
      {groups && groups.length > 0 && (
        <div className="flex items-center gap-2 border-b border-slate-700/30 pb-3">
          <button
            onClick={() => setViewMode('groups')}
            className={clsx(
              'px-3 py-1.5 text-xs font-medium rounded transition-colors',
              viewMode === 'groups'
                ? 'bg-amber-500/20 text-amber-400'
                : 'text-slate-400 hover:text-white'
            )}
          >
            <svg className="w-4 h-4 inline-block mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
            Smart Groups
          </button>
          <button
            onClick={() => setViewMode('list')}
            className={clsx(
              'px-3 py-1.5 text-xs font-medium rounded transition-colors',
              viewMode === 'list'
                ? 'bg-amber-500/20 text-amber-400'
                : 'text-slate-400 hover:text-white'
            )}
          >
            <svg className="w-4 h-4 inline-block mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
            </svg>
            All Issues
          </button>
        </div>
      )}

      {/* Smart Groups view */}
      {viewMode === 'groups' && groups && groups.length > 0 ? (
        <div className="space-y-3 max-h-[500px] overflow-y-auto pr-2">
          {groups.map((group) => (
            <AnomalyGroupCard
              key={group.group_id}
              group={group}
              isExpanded={expandedGroups.has(group.group_id)}
              onToggle={() => toggleGroup(group.group_id)}
              selectedAnomalies={selectedAnomaliesByGroup.get(group.group_id) || new Set()}
              onSelectAnomaly={(id, selected) => handleSelectAnomaly(group.group_id, id, selected)}
              onSelectAll={(selected) => handleSelectAll(group.group_id, group, selected)}
              onBulkAction={(action) => handleBulkAction(group.group_id, action)}
              onAnomalyClick={handleGroupMemberClick}
            />
          ))}
        </div>
      ) : (
        <>
          {/* Critical section - only in list view */}
          {criticalAnomalies.length > 0 && (
            <div>
              <h4 className="text-xs font-medium text-red-400 uppercase tracking-wide mb-2">
                Critical Issues
              </h4>
              <div className="space-y-2">
                {criticalAnomalies.map((anomaly) => (
                  <AnomalyRow
                    key={anomaly.id}
                    anomaly={anomaly}
                    onClick={() => setSelectedAnomaly(anomaly)}
                  />
                ))}
              </div>
            </div>
          )}

          {/* All open */}
          <div>
            <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wide mb-2">
              All Open ({openAnomalies.length})
            </h4>
            {isLoading ? (
              <LoadingSkeleton />
            ) : openAnomalies.length === 0 ? (
              <div className="flex items-center justify-center py-8 text-slate-500">
                No open investigations
              </div>
            ) : (
              <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2">
                {openAnomalies.map((anomaly) => (
                  <AnomalyRow
                    key={anomaly.id}
                    anomaly={anomaly}
                    onClick={() => setSelectedAnomaly(anomaly)}
                  />
                ))}
              </div>
            )}
          </div>
        </>
      )}

      {/* Actions */}
      <div className="flex gap-2 pt-2 border-t border-slate-700/30">
        <Link
          to={viewMode === 'groups' ? '/investigations?view=grouped' : '/investigations'}
          className="flex-1 py-2 px-4 rounded-lg bg-amber-500/20 text-amber-400 text-sm font-medium text-center hover:bg-amber-500/30 transition-colors"
        >
          View All Investigations
        </Link>
      </div>
    </div>
  );

  return (
    <>
      <CollapsibleSection
        title="Investigations"
        icon={
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        }
        badge={criticalCount > 0 ? { value: criticalCount, variant: 'danger' } : undefined}
        summaryMetrics={[
          { label: 'Open', value: openCount, variant: openCount > 0 ? 'warning' : 'default' },
          { label: 'Critical', value: criticalCount, variant: criticalCount > 0 ? 'danger' : 'default' },
        ]}
        collapsedContent={collapsedContent}
        expandedContent={expandedContent}
        defaultExpanded={true}
        accent={criticalCount > 0 ? 'red' : 'amber'}
      />

      {/* Detail Panel */}
      <SlideOverPanel
        isOpen={!!selectedAnomaly}
        onClose={() => setSelectedAnomaly(null)}
        title={`Investigation #${selectedAnomaly?.id}`}
        subtitle={selectedAnomaly ? `Device ${selectedAnomaly.device_id}` : undefined}
        footer={
          selectedAnomaly && (
            <div className="flex gap-2">
              <Link
                to={`/investigations/${selectedAnomaly.id}`}
                className="flex-1 py-2 px-4 rounded-lg bg-amber-500 text-slate-900 text-sm font-medium text-center hover:bg-amber-400 transition-colors"
              >
                Open Full Investigation
              </Link>
              <button
                onClick={() => setSelectedAnomaly(null)}
                className="py-2 px-4 rounded-lg bg-slate-700 text-slate-300 text-sm font-medium hover:bg-slate-600 transition-colors"
              >
                Close
              </button>
            </div>
          )
        }
      >
        {selectedAnomaly && (
          <div className="space-y-4">
            {/* Quick stats */}
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/30">
                <div className="text-xs text-slate-500 mb-1">Anomaly Score</div>
                <div className={clsx(
                  'text-2xl font-bold font-mono',
                  severityConfig[getSeverityFromScore(selectedAnomaly.anomaly_score)].color
                )}>
                  {selectedAnomaly.anomaly_score.toFixed(4)}
                </div>
              </div>
              <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/30">
                <div className="text-xs text-slate-500 mb-1">Status</div>
                <div className={clsx(
                  'text-lg font-medium',
                  statusConfig[selectedAnomaly.status as keyof typeof statusConfig]?.color || 'text-slate-300'
                )}>
                  {selectedAnomaly.status}
                </div>
              </div>
            </div>

            {/* Timestamp */}
            <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/30">
              <div className="text-xs text-slate-500 mb-1">Detected</div>
              <div className="text-white">
                {format(new Date(selectedAnomaly.timestamp), 'PPpp')}
              </div>
              <div className="text-xs text-slate-400 mt-1">
                {formatDistanceToNowStrict(new Date(selectedAnomaly.timestamp), { addSuffix: true })}
              </div>
            </div>

            {/* Key metrics */}
            <div>
              <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wide mb-2">
                Key Metrics
              </h4>
              <div className="grid grid-cols-2 gap-2">
                {selectedAnomaly.total_battery_level_drop != null && (
                  <div className="p-2 rounded bg-slate-800/30 border border-slate-700/30">
                    <div className="text-xs text-slate-500">Battery Drop</div>
                    <div className="text-white font-mono">{selectedAnomaly.total_battery_level_drop.toFixed(1)}%</div>
                  </div>
                )}
                {selectedAnomaly.download != null && (
                  <div className="p-2 rounded bg-slate-800/30 border border-slate-700/30">
                    <div className="text-xs text-slate-500">Download</div>
                    <div className="text-white font-mono">{(selectedAnomaly.download / 1000).toFixed(2)} GB</div>
                  </div>
                )}
                {selectedAnomaly.offline_time != null && (
                  <div className="p-2 rounded bg-slate-800/30 border border-slate-700/30">
                    <div className="text-xs text-slate-500">Offline Time</div>
                    <div className="text-white font-mono">{selectedAnomaly.offline_time.toFixed(1)} min</div>
                  </div>
                )}
                {selectedAnomaly.disconnect_count != null && (
                  <div className="p-2 rounded bg-slate-800/30 border border-slate-700/30">
                    <div className="text-xs text-slate-500">Disconnects</div>
                    <div className="text-white font-mono">{selectedAnomaly.disconnect_count}</div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </SlideOverPanel>
    </>
  );
}

export default InvestigationsSection;
