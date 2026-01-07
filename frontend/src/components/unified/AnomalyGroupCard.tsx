/**
 * AnomalyGroupCard - Collapsible card for displaying a group of related anomalies
 *
 * Features:
 * - Collapsible header with group summary
 * - Severity color coding
 * - Bulk action toolbar when expanded
 * - Suggested remediation display
 * - List of anomalies with selection
 */

import { useState } from 'react';
import { Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import clsx from 'clsx';
import { formatDistanceToNowStrict } from 'date-fns';
import type { AnomalyGroup, AnomalyGroupMember, Severity } from '../../types/anomaly';

interface AnomalyGroupCardProps {
  group: AnomalyGroup;
  isExpanded: boolean;
  onToggle: () => void;
  selectedAnomalies: Set<number>;
  onSelectAnomaly: (id: number, selected: boolean) => void;
  onSelectAll: (selected: boolean) => void;
  onBulkAction: (action: 'resolve' | 'investigating' | 'dismiss') => void;
  onAnomalyClick: (anomaly: AnomalyGroupMember) => void;
  isLoading?: boolean;
}

const severityConfig: Record<Severity, { label: string; color: string; bg: string; border: string; barBg: string }> = {
  critical: { label: 'CRITICAL', color: 'text-red-400', bg: 'bg-red-500/20', border: 'border-red-500/30', barBg: 'bg-red-500' },
  high: { label: 'HIGH', color: 'text-orange-400', bg: 'bg-orange-500/20', border: 'border-orange-500/30', barBg: 'bg-orange-500' },
  medium: { label: 'MEDIUM', color: 'text-amber-400', bg: 'bg-amber-500/20', border: 'border-amber-500/30', barBg: 'bg-amber-500' },
  low: { label: 'LOW', color: 'text-slate-400', bg: 'bg-slate-700/50', border: 'border-slate-600/50', barBg: 'bg-slate-500' },
};

const statusConfig: Record<string, { label: string; color: string }> = {
  open: { label: 'Open', color: 'text-red-400' },
  investigating: { label: 'Investigating', color: 'text-orange-400' },
  resolved: { label: 'Resolved', color: 'text-emerald-400' },
  false_positive: { label: 'Dismissed', color: 'text-slate-400' },
};

function GroupMemberRow({
  member,
  isSelected,
  onSelect,
  onClick,
}: {
  member: AnomalyGroupMember;
  isSelected: boolean;
  onSelect: (selected: boolean) => void;
  onClick: () => void;
}) {
  const severity = severityConfig[member.severity];
  const status = statusConfig[member.status] || statusConfig.open;

  return (
    <div
      className={clsx(
        'flex items-center gap-3 p-2 rounded-lg transition-colors cursor-pointer',
        'hover:bg-slate-700/30',
        isSelected && 'bg-slate-700/40'
      )}
    >
      {/* Checkbox */}
      <input
        type="checkbox"
        checked={isSelected}
        onChange={(e) => {
          e.stopPropagation();
          onSelect(e.target.checked);
        }}
        className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-amber-500 focus:ring-amber-500 focus:ring-offset-0"
      />

      {/* Severity bar */}
      <div className={clsx('w-1 h-8 rounded-full', severity.barBg)} />

      {/* Content - clickable */}
      <button onClick={onClick} className="flex-1 text-left min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm text-white font-medium">Device #{member.device_id}</span>
          <span className={clsx('px-1.5 py-0.5 rounded text-xs', severity.bg, severity.color)}>
            {severity.label}
          </span>
        </div>
        <div className="flex items-center gap-2 mt-0.5">
          <span className="text-xs text-slate-500">
            {formatDistanceToNowStrict(new Date(member.timestamp), { addSuffix: true })}
          </span>
          {member.location && (
            <>
              <span className="text-slate-700">|</span>
              <span className="text-xs text-slate-500 truncate">{member.location}</span>
            </>
          )}
          <span className="text-slate-700">|</span>
          <span className={clsx('text-xs', status.color)}>{status.label}</span>
        </div>
      </button>

      {/* Score */}
      <span className={clsx('font-mono text-xs', severity.color)}>
        {member.anomaly_score.toFixed(2)}
      </span>
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

  const allSelected = selectedAnomalies.size === group.sample_anomalies.length;
  const someSelected = selectedAnomalies.size > 0;

  return (
    <div className={clsx(
      'rounded-xl border transition-colors',
      'bg-slate-800/30',
      severity.border,
      isExpanded && 'ring-1 ring-slate-600/50'
    )}>
      {/* Header - Always visible */}
      <button
        onClick={onToggle}
        className="w-full p-4 text-left flex items-center gap-4"
      >
        {/* Expand chevron */}
        <motion.div
          animate={{ rotate: isExpanded ? 90 : 0 }}
          transition={{ duration: 0.2 }}
          className="text-slate-400"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </motion.div>

        {/* Severity indicator */}
        <div className={clsx('w-1.5 h-12 rounded-full', severity.barBg)} />

        {/* Group info */}
        <div className="flex-1 min-w-0">
          <h3 className="text-white font-medium text-sm truncate">{group.group_name}</h3>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-xs text-slate-500">{group.device_count} devices</span>
            <span className="text-slate-700">|</span>
            <span className="text-xs text-slate-500">{group.open_count} open</span>
            {group.common_location && (
              <>
                <span className="text-slate-700">|</span>
                <span className="text-xs text-slate-400">{group.common_location}</span>
              </>
            )}
          </div>
        </div>

        {/* Badges */}
        <div className="flex items-center gap-2">
          <span className={clsx('px-2 py-1 text-xs font-medium rounded', severity.bg, severity.color)}>
            {severity.label}
          </span>
          {group.suggested_remediation && (
            <span className="px-2 py-1 text-xs font-medium rounded bg-emerald-500/20 text-emerald-400">
              Fix Available
            </span>
          )}
        </div>
      </button>

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
                    {Math.round(group.suggested_remediation.historical_success_rate * 100)}% success rate
                    {group.suggested_remediation.historical_sample_size && (
                      <span className="text-slate-500"> ({group.suggested_remediation.historical_sample_size} cases)</span>
                    )}
                  </p>
                )}
              </div>
            )}

            {/* Anomaly list */}
            <div className="p-3 space-y-1">
              {group.sample_anomalies.map((member) => (
                <GroupMemberRow
                  key={member.anomaly_id}
                  member={member}
                  isSelected={selectedAnomalies.has(member.anomaly_id)}
                  onSelect={(selected) => onSelectAnomaly(member.anomaly_id, selected)}
                  onClick={() => onAnomalyClick(member)}
                />
              ))}

              {group.total_count > group.sample_anomalies.length && (
                <Link
                  to={`/investigations?group=${group.group_id}`}
                  className="block text-center py-2 text-xs text-amber-400 hover:text-amber-300 transition-colors"
                >
                  View all {group.total_count} anomalies in this group
                </Link>
              )}
            </div>

            {/* Grouping explanation */}
            <div className="p-3 border-t border-slate-700/30 flex items-center gap-2 flex-wrap">
              <span className="text-xs text-slate-500 font-medium">Grouped by:</span>
              {group.grouping_factors.map((factor, i) => (
                <span key={i} className="text-xs text-slate-400 px-2 py-0.5 rounded bg-slate-700/50">
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
