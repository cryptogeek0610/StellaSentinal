/**
 * InvestigationsSection - Collapsible section for anomaly investigations
 *
 * Shows open investigations, critical issues, and priority queue.
 * Expands to show full investigation list with filtering.
 */

import { useState } from 'react';
import { Link } from 'react-router-dom';
import clsx from 'clsx';
import { format, formatDistanceToNowStrict } from 'date-fns';
import { CollapsibleSection } from '../CollapsibleSection';
import { SlideOverPanel } from '../SlideOverPanel';
import type { Anomaly } from '../../../types/anomaly';

interface InvestigationsSectionProps {
  anomalies: Anomaly[];
  stats: {
    open_cases: number;
    critical_issues: number;
    detected_today: number;
    resolved_today: number;
  } | undefined;
  isLoading?: boolean;
}

const severityConfig = {
  critical: { label: 'CRITICAL', color: 'text-red-400', bg: 'bg-red-500/20', border: 'border-red-500/30' },
  high: { label: 'HIGH', color: 'text-orange-400', bg: 'bg-orange-500/20', border: 'border-orange-500/30' },
  medium: { label: 'MEDIUM', color: 'text-amber-400', bg: 'bg-amber-500/20', border: 'border-amber-500/30' },
  low: { label: 'LOW', color: 'text-slate-400', bg: 'bg-slate-700/50', border: 'border-slate-600/50' },
};

const getSeverityFromScore = (score: number): keyof typeof severityConfig => {
  if (score <= -0.7) return 'critical';
  if (score <= -0.5) return 'high';
  if (score <= -0.3) return 'medium';
  return 'low';
};

const statusConfig = {
  open: { label: 'Open', color: 'text-red-400', bg: 'bg-red-500/10' },
  investigating: { label: 'Investigating', color: 'text-orange-400', bg: 'bg-orange-500/10' },
  resolved: { label: 'Resolved', color: 'text-emerald-400', bg: 'bg-emerald-500/10' },
  false_positive: { label: 'False Positive', color: 'text-slate-400', bg: 'bg-slate-700/30' },
};

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
            Device #{anomaly.device_id}
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

      {/* Score */}
      <div className="text-right">
        <span className={clsx('font-mono text-sm', severityStyle.color)}>
          {anomaly.anomaly_score.toFixed(3)}
        </span>
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
  stats,
  isLoading,
}: InvestigationsSectionProps) {
  const [selectedAnomaly, setSelectedAnomaly] = useState<Anomaly | null>(null);

  const openCount = stats?.open_cases || 0;
  const criticalCount = stats?.critical_issues || 0;

  // Split anomalies by status
  const openAnomalies = anomalies.filter(a => a.status === 'open' || a.status === 'investigating');
  const criticalAnomalies = openAnomalies.filter(a => getSeverityFromScore(a.anomaly_score) === 'critical');

  const collapsedContent = isLoading ? (
    <LoadingSkeleton />
  ) : openAnomalies.length === 0 ? (
    <div className="flex items-center justify-center py-6 text-slate-500">
      <svg className="w-5 h-5 mr-2 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
      </svg>
      No open investigations
    </div>
  ) : (
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

      {/* Critical section */}
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

      {/* Actions */}
      <div className="flex gap-2 pt-2 border-t border-slate-700/30">
        <Link
          to="/investigations"
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
        defaultExpanded={criticalCount > 0}
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
