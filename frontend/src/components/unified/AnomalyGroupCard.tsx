/**
 * AnomalyGroupCard - Action-oriented card for anomaly groups
 *
 * Steve Jobs redesign principles:
 * - Fix All button visible WITHOUT expanding (action-first)
 * - Severity gradients communicate urgency visually
 * - Success rate as confidence indicator
 * - Location chips in collapsed view
 */

import { useState } from 'react';
import { Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import clsx from 'clsx';
import { formatDistanceToNowStrict } from 'date-fns';
import type { AnomalyGroup, AnomalyGroupMember, Severity } from '../../types/anomaly';
import { SEVERITY_CONFIGS, STATUS_CONFIGS, type SeverityLevel } from '../../utils/severity';

interface AnomalyGroupCardProps {
  group: AnomalyGroup;
  isExpanded: boolean;
  onToggle: () => void;
  selectedAnomalies: Set<number>;
  onSelectAnomaly: (id: number, selected: boolean) => void;
  onSelectAll: (selected: boolean) => void;
  onBulkAction: (action: 'resolve' | 'investigating' | 'dismiss') => void;
  onAnomalyClick: (anomaly: AnomalyGroupMember) => void;
  onFixAll?: () => void;
  isLoading?: boolean;
}

// Gradient extensions for visual weight (not in centralized config)
const severityGradients: Record<SeverityLevel, string> = {
  critical: 'bg-gradient-to-r from-red-950/30 via-red-950/10 to-transparent',
  high: 'bg-gradient-to-r from-orange-950/25 via-orange-950/10 to-transparent',
  medium: 'bg-gradient-to-r from-amber-950/20 via-amber-950/5 to-transparent',
  low: 'bg-gradient-to-r from-cyan-950/20 via-cyan-950/5 to-transparent',
};

// Derive severity config from centralized source with gradient extensions
const severityConfig: Record<
  Severity,
  {
    label: string;
    color: string;
    bg: string;
    border: string;
    barBg: string;
    gradient: string;
  }
> = Object.fromEntries(
  Object.entries(SEVERITY_CONFIGS).map(([key, cfg]) => [
    key,
    {
      label: cfg.label,
      color: cfg.color.text,
      bg: cfg.color.bg,
      border: cfg.color.border,
      barBg: cfg.color.dot,
      gradient: severityGradients[key as SeverityLevel],
    },
  ])
) as Record<Severity, { label: string; color: string; bg: string; border: string; barBg: string; gradient: string }>;

// Derive status config from centralized source
const statusConfig: Record<string, { label: string; color: string }> = Object.fromEntries(
  Object.entries(STATUS_CONFIGS).map(([key, cfg]) => [
    key,
    { label: cfg.label, color: cfg.color },
  ])
);

// Group anomalies by device_id, preserving order by first occurrence
interface DeviceGroup {
  deviceId: number;
  deviceName: string | null | undefined;
  deviceModel: string | null | undefined;
  location: string | null | undefined;
  anomalies: AnomalyGroupMember[];
  worstSeverity: Severity;
}

function groupAnomaliesByDevice(anomalies: AnomalyGroupMember[]): DeviceGroup[] {
  const severityOrder: Record<Severity, number> = {
    critical: 0,
    high: 1,
    medium: 2,
    low: 3,
  };

  const deviceMap = new Map<number, DeviceGroup>();

  for (const anomaly of anomalies) {
    const existing = deviceMap.get(anomaly.device_id);
    if (existing) {
      existing.anomalies.push(anomaly);
      // Update worst severity if this one is worse
      if (severityOrder[anomaly.severity] < severityOrder[existing.worstSeverity]) {
        existing.worstSeverity = anomaly.severity;
      }
    } else {
      deviceMap.set(anomaly.device_id, {
        deviceId: anomaly.device_id,
        deviceName: anomaly.device_name,
        deviceModel: anomaly.device_model,
        location: anomaly.location,
        anomalies: [anomaly],
        worstSeverity: anomaly.severity,
      });
    }
  }

  return Array.from(deviceMap.values());
}

// Compact anomaly row for nested display within device cards
function CompactAnomalyRow({
  anomaly,
  onClick,
}: {
  anomaly: AnomalyGroupMember;
  onClick: () => void;
}) {
  const severity = severityConfig[anomaly.severity];
  const status = statusConfig[anomaly.status] || statusConfig.open;

  return (
    <button
      onClick={onClick}
      className="w-full flex items-center gap-3 py-1.5 px-2 rounded text-left hover:bg-slate-700/30 transition-colors"
    >
      <span
        className={clsx(
          'px-1.5 py-0.5 text-[10px] font-medium rounded',
          severity.bg,
          severity.color
        )}
      >
        {severity.label}
      </span>
      <span className="text-xs text-slate-500">
        {formatDistanceToNowStrict(new Date(anomaly.timestamp), { addSuffix: false })}
      </span>
      <span className="text-slate-700">·</span>
      <span className={clsx('text-xs', status.color)}>{status.label}</span>
      <span className="ml-auto font-mono text-xs text-slate-500">
        {anomaly.anomaly_score.toFixed(2)}
      </span>
    </button>
  );
}

// Device card that groups multiple anomalies from the same device
function DeviceCard({
  device,
  selectedAnomalyIds,
  onToggleDevice,
  onAnomalyClick,
}: {
  device: DeviceGroup;
  selectedAnomalyIds: Set<number>;
  onToggleDevice: (deviceId: number, anomalyIds: number[], selected: boolean) => void;
  onAnomalyClick: (anomaly: AnomalyGroupMember) => void;
}) {
  const severity = severityConfig[device.worstSeverity];
  const anomalyIds = device.anomalies.map((a) => a.anomaly_id);
  const selectedCount = anomalyIds.filter((id) => selectedAnomalyIds.has(id)).length;
  const isSelected = selectedCount === anomalyIds.length;
  const isPartial = selectedCount > 0 && selectedCount < anomalyIds.length;

  const visibleAnomalies = device.anomalies.slice(0, 2);
  const hiddenCount = device.anomalies.length - visibleAnomalies.length;

  return (
    <div
      className={clsx(
        'rounded-lg border-l-2 bg-slate-800/40 overflow-hidden',
        severity.border
      )}
    >
      {/* Device header */}
      <div className="flex items-center gap-3 p-3">
        <input
          type="checkbox"
          checked={isSelected}
          ref={(el) => {
            if (el) el.indeterminate = isPartial;
          }}
          onChange={(e) => onToggleDevice(device.deviceId, anomalyIds, e.target.checked)}
          className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-amber-500 focus:ring-amber-500 focus:ring-offset-0"
        />

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-white truncate">
              {device.deviceName || `Device #${device.deviceId}`}
            </span>
            <span className="text-xs text-slate-500 bg-slate-700/50 px-1.5 py-0.5 rounded">
              {device.anomalies.length} {device.anomalies.length === 1 ? 'issue' : 'issues'}
            </span>
          </div>
          {(device.deviceModel || device.location) && (
            <div className="text-xs text-slate-500 mt-0.5">
              {[device.deviceModel, device.location].filter(Boolean).join(' · ')}
            </div>
          )}
        </div>
      </div>

      {/* Anomaly list */}
      <div className="border-t border-slate-700/30 px-2 py-1">
        {visibleAnomalies.map((anomaly) => (
          <CompactAnomalyRow
            key={anomaly.anomaly_id}
            anomaly={anomaly}
            onClick={() => onAnomalyClick(anomaly)}
          />
        ))}
        {hiddenCount > 0 && (
          <div className="text-xs text-slate-500 py-1.5 px-2">
            +{hiddenCount} more {hiddenCount === 1 ? 'issue' : 'issues'}
          </div>
        )}
      </div>
    </div>
  );
}

export function AnomalyGroupCard({
  group,
  isExpanded,
  onToggle,
  selectedAnomalies,
  onSelectAnomaly,
  onSelectAll,
  onBulkAction,
  onAnomalyClick,
  onFixAll,
  isLoading: _isLoading,
}: AnomalyGroupCardProps) {
  const [isProcessing, setIsProcessing] = useState(false);
  const severity = severityConfig[group.severity];

  const handleBulkAction = async (action: 'resolve' | 'investigating' | 'dismiss') => {
    setIsProcessing(true);
    try {
      await onBulkAction(action);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleFixAll = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (onFixAll) {
      onFixAll();
    } else {
      // Default: select all and resolve
      onSelectAll(true);
      handleBulkAction('resolve');
    }
  };

  const allSelected = selectedAnomalies.size === group.sample_anomalies.length;
  const someSelected = selectedAnomalies.size > 0;

  // Calculate time range display
  const timeRangeDisplay = (() => {
    const start = new Date(group.time_range_start);
    const end = new Date(group.time_range_end);
    const diffMs = end.getTime() - start.getTime();
    const diffDays = Math.ceil(diffMs / (1000 * 60 * 60 * 24));
    if (diffDays <= 1) return 'Last 24h';
    if (diffDays <= 7) return `Last ${diffDays} days`;
    return `${diffDays} days`;
  })();

  // Get success rate if available
  const successRate = group.suggested_remediation?.historical_success_rate;
  const hasHighSuccessRate = successRate && successRate >= 0.7;

  return (
    <div
      className={clsx(
        'rounded-xl border-l-4 border transition-all overflow-hidden',
        severity.gradient,
        severity.border,
        isExpanded && 'ring-1 ring-slate-600/50'
      )}
    >
      {/* Header - Always visible with Fix All button */}
      <div className="flex items-center gap-4 p-4">
        {/* Expand chevron - clickable area */}
        <button
          onClick={onToggle}
          className="flex items-center gap-4 flex-1 min-w-0 text-left"
        >
          <motion.div
            animate={{ rotate: isExpanded ? 90 : 0 }}
            transition={{ duration: 0.2 }}
            className="text-slate-400 flex-shrink-0"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 5l7 7-7 7"
              />
            </svg>
          </motion.div>

          {/* Group info */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <h3 className="text-white font-medium text-sm">{group.group_name}</h3>
              <span className={clsx('px-1.5 py-0.5 text-xs font-medium rounded', severity.bg, severity.color)}>
                {severity.label}
              </span>
            </div>

            {/* Stats row */}
            <div className="flex items-center gap-3 mt-1 text-xs text-slate-500">
              <span>{group.device_count} devices</span>
              <span className="text-slate-700">&#183;</span>
              <span>{group.open_count} open</span>
              <span className="text-slate-700">&#183;</span>
              <span>{timeRangeDisplay}</span>
              {hasHighSuccessRate && (
                <>
                  <span className="text-slate-700">&#183;</span>
                  <span className="text-emerald-400/80 flex items-center gap-1">
                    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    {Math.round(successRate * 100)}% success
                  </span>
                </>
              )}
            </div>

            {/* Location chip - show in collapsed view */}
            {group.common_location && (
              <div className="flex gap-1 mt-2">
                <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 bg-slate-700/50 rounded-full text-slate-400">
                  <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                  {group.common_location}
                </span>
              </div>
            )}
          </div>
        </button>

        {/* Fix All button - always visible */}
        {group.suggested_remediation && (
          <button
            onClick={handleFixAll}
            disabled={isProcessing}
            className="flex items-center gap-1.5 px-3 py-2 text-sm font-medium rounded-lg bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30 transition-colors disabled:opacity-50 whitespace-nowrap flex-shrink-0"
          >
            Fix All
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 5l7 7-7 7"
              />
            </svg>
          </button>
        )}
      </div>

      {/* Expanded Content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden border-t border-slate-700/30"
          >
            {/* Bulk action toolbar */}
            <div className="p-3 bg-slate-800/50 flex items-center gap-3 flex-wrap">
              <input
                type="checkbox"
                checked={allSelected}
                ref={(el) => {
                  if (el) el.indeterminate = someSelected && !allSelected;
                }}
                onChange={(e) => onSelectAll(e.target.checked)}
                className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-amber-500 focus:ring-amber-500 focus:ring-offset-0"
              />
              <span className="text-xs text-slate-400">
                {selectedAnomalies.size} of {group.sample_anomalies.length} selected
              </span>

              {someSelected && (
                <div className="flex gap-2 ml-auto">
                  <button
                    onClick={() => handleBulkAction('resolve')}
                    disabled={isProcessing}
                    className="px-3 py-1.5 text-xs font-medium rounded bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30 transition-colors disabled:opacity-50"
                  >
                    Resolve All
                  </button>
                  <button
                    onClick={() => handleBulkAction('investigating')}
                    disabled={isProcessing}
                    className="px-3 py-1.5 text-xs font-medium rounded bg-orange-500/20 text-orange-400 hover:bg-orange-500/30 transition-colors disabled:opacity-50"
                  >
                    Investigate
                  </button>
                  <button
                    onClick={() => handleBulkAction('dismiss')}
                    disabled={isProcessing}
                    className="px-3 py-1.5 text-xs font-medium rounded bg-slate-700 text-slate-300 hover:bg-slate-600 transition-colors disabled:opacity-50"
                  >
                    Dismiss
                  </button>
                </div>
              )}
            </div>

            {/* Suggested remediation card */}
            {group.suggested_remediation && (
              <div className="p-3 mx-3 mt-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
                <h4 className="text-sm font-medium text-emerald-400">
                  Suggested Fix: {group.suggested_remediation.title}
                </h4>
                <p className="text-xs text-slate-400 mt-1">
                  {group.suggested_remediation.description}
                </p>
                {group.suggested_remediation.historical_success_rate && (
                  <p className="text-xs text-emerald-400/70 mt-1">
                    {Math.round(group.suggested_remediation.historical_success_rate * 100)}% success
                    rate
                    {group.suggested_remediation.historical_sample_size && (
                      <span className="text-slate-500">
                        {' '}
                        ({group.suggested_remediation.historical_sample_size} cases)
                      </span>
                    )}
                  </p>
                )}
              </div>
            )}

            {/* Device-grouped anomaly list */}
            <div className="p-3 space-y-2">
              {groupAnomaliesByDevice(group.sample_anomalies).map((device) => (
                <DeviceCard
                  key={device.deviceId}
                  device={device}
                  selectedAnomalyIds={selectedAnomalies}
                  onToggleDevice={(_deviceId, anomalyIds, selected) => {
                    anomalyIds.forEach((id) => onSelectAnomaly(id, selected));
                  }}
                  onAnomalyClick={onAnomalyClick}
                />
              ))}

              {group.total_count > group.sample_anomalies.length && (
                <Link
                  to={`/investigations?group=${group.group_id}`}
                  className="block text-center py-2 text-xs text-amber-400 hover:text-amber-300 transition-colors"
                >
                  View all {group.device_count} devices in this group
                </Link>
              )}
            </div>

            {/* Grouping explanation */}
            <div className="p-3 border-t border-slate-700/30 flex items-center gap-2 flex-wrap">
              <span className="text-xs text-slate-500 font-medium">Grouped by:</span>
              {group.grouping_factors.map((factor, i) => (
                <span
                  key={i}
                  className="text-xs text-slate-400 px-2 py-0.5 rounded bg-slate-700/50"
                >
                  {factor}
                </span>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default AnomalyGroupCard;
