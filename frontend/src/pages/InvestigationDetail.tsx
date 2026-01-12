/**
 * Investigation Detail Page - AIOps Dashboard
 *
 * Comprehensive anomaly investigation with:
 * - Feature contribution breakdown (why the anomaly was detected)
 * - Baseline comparison visualization
 * - LLM-powered root cause analysis
 * - Device telemetry & historical trends
 * - SOTI MobiControl remediation actions
 * - Similar case reference
 */

import { useState } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../api/client';
import { format, formatDistanceToNowStrict } from 'date-fns';
import type { AnomalyImpactResponse } from '../types/cost';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from '../components/Card';
import { AIAnalysisDisplay } from '../components/AIAnalysisDisplay';
import { useMockMode } from '../hooks/useMockMode';
import { showSuccess, showError } from '../utils/toast';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  Cell, CartesianGrid,
} from 'recharts';
import type {
  FeatureContribution,
  BaselineMetric,
  RootCauseHypothesis,
  RemediationSuggestion,
  SimilarCase,
  LearnFromFix,
  InvestigationNote,
} from '../types/anomaly';
import {
  SEVERITY_CONFIGS,
  STATUS_CONFIGS,
  type SeverityLevel,
  type StatusLevel,
} from '../utils/severity';

// Create local configs that match expected shapes (for backwards compatibility)
const severityConfig = Object.fromEntries(
  Object.entries(SEVERITY_CONFIGS).map(([key, cfg]) => [
    key,
    { label: cfg.label, color: cfg.color.text, bg: cfg.color.bg, border: cfg.color.border },
  ])
) as Record<SeverityLevel, { label: string; color: string; bg: string; border: string }>;

const statusConfig = Object.fromEntries(
  Object.entries(STATUS_CONFIGS).map(([key, cfg]) => [
    key,
    { label: cfg.label, color: cfg.color, bg: cfg.bg, icon: cfg.icon },
  ])
) as Record<StatusLevel, { label: string; color: string; bg: string; icon: string }>;

// ============================================================================
// Data Size Formatting Utility
// ============================================================================

/**
 * Formats a data size in MB to the most appropriate unit (MB, GB, or TB)
 * @param megabytes - Value in megabytes
 * @returns Object with formatted value and unit string
 */
function formatDataSize(megabytes: number | null | undefined): { value: string; unit: string } {
  if (megabytes == null) {
    return { value: '0', unit: 'MB' };
  }

  const MB_THRESHOLD = 1000;    // Show as GB if >= 1000 MB
  const GB_THRESHOLD = 1000;    // Show as TB if >= 1000 GB

  if (megabytes >= MB_THRESHOLD * GB_THRESHOLD) {
    // Convert to TB (MB / 1,000,000)
    return { value: (megabytes / 1_000_000).toFixed(2), unit: 'TB' };
  } else if (megabytes >= MB_THRESHOLD) {
    // Convert to GB (MB / 1,000)
    return { value: (megabytes / 1_000).toFixed(2), unit: 'GB' };
  } else {
    // Keep as MB
    return { value: megabytes.toFixed(1), unit: 'MB' };
  }
}

const getSeverityKey = (score: number): keyof typeof severityConfig => {
  if (score <= -0.7) return 'critical';
  if (score <= -0.5) return 'high';
  if (score <= -0.3) return 'medium';
  return 'low';
};

// ============================================================================
// Icon Components
// ============================================================================

const BrainIcon = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
  </svg>
);

const ChartIcon = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
  </svg>
);

const TargetIcon = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
  </svg>
);

const WrenchIcon = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
  </svg>
);

const DeviceIcon = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 18h.01M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z" />
  </svg>
);

// ============================================================================
// Sub-Components
// ============================================================================

// Feature Contribution Bar Chart
const FeatureContributionChart = ({ contributions }: { contributions: FeatureContribution[] }) => {
  if (!contributions || contributions.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-slate-500">
        No feature contribution data available
      </div>
    );
  }

  const data = contributions.slice(0, 8).map((c) => ({
    name: c.feature_display_name,
    contribution: c.contribution_percentage,
    sigma: Math.abs(c.deviation_sigma),
    current: c.current_value_display,
    baseline: c.baseline_value_display,
  }));

  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 100, right: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
          <XAxis type="number" domain={[0, 'dataMax']} stroke="#64748b" fontSize={10} />
          <YAxis type="category" dataKey="name" stroke="#64748b" fontSize={11} width={95} />
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(14, 17, 23, 0.95)',
              border: '1px solid rgba(245, 158, 11, 0.3)',
              borderRadius: '8px',
              fontSize: '11px',
            }}
            formatter={(value: number) => [`${value.toFixed(1)}% contribution`]}
          />
          <Bar dataKey="contribution" radius={[0, 4, 4, 0]}>
            {data.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.sigma >= 3 ? '#ef4444' : entry.sigma >= 2 ? '#f97316' : '#f59e0b'}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

// Baseline Comparison Visualization
const BaselineComparisonViz = ({ metrics }: { metrics: BaselineMetric[] }) => {
  if (!metrics || metrics.length === 0) {
    return (
      <div className="h-48 flex items-center justify-center text-slate-500">
        No baseline comparison data available
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {metrics.filter(m => m.is_anomalous).map((metric) => (
        <div key={metric.metric_name} className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/50">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-white">{metric.metric_display_name}</span>
            <span className={`text-xs px-2 py-0.5 rounded ${
              Math.abs(metric.deviation_sigma) >= 3 ? 'bg-red-500/20 text-red-400' :
              Math.abs(metric.deviation_sigma) >= 2 ? 'bg-orange-500/20 text-orange-400' :
              'bg-amber-500/20 text-amber-400'
            }`}>
              {metric.deviation_sigma > 0 ? '+' : ''}{metric.deviation_sigma.toFixed(1)}σ
            </span>
          </div>

          {/* Visual bar showing position relative to baseline */}
          <div className="relative h-8 bg-slate-700/50 rounded-lg overflow-hidden mb-2">
            {/* Baseline range (mean ± 2σ) */}
            <div
              className="absolute h-full bg-emerald-500/20"
              style={{
                left: `${Math.max(0, 50 - 25)}%`,
                width: '50%',
              }}
            />
            {/* Mean marker */}
            <div className="absolute h-full w-0.5 bg-emerald-500/50" style={{ left: '50%' }} />
            {/* Current value marker */}
            <div
              className="absolute h-full w-1.5 bg-amber-500 rounded"
              style={{
                left: `${Math.min(95, Math.max(5, metric.percentile_rank))}%`,
                transform: 'translateX(-50%)',
              }}
            />
          </div>

          <div className="flex justify-between text-xs text-slate-500">
            <span>Current: <span className="text-amber-400 font-mono">{metric.current_value_display}</span></span>
            <span>Baseline: <span className="text-emerald-400 font-mono">{metric.baseline_mean.toFixed(1)} {metric.metric_unit}</span></span>
          </div>
        </div>
      ))}
    </div>
  );
};

// AI Root Cause Analysis Panel
const AIAnalysisPanel = ({
  hypothesis,
  isLoading,
  onRegenerate,
}: {
  hypothesis?: RootCauseHypothesis;
  isLoading: boolean;
  onRegenerate: () => void;
}) => {
  if (isLoading) {
    return (
      <div className="p-6 rounded-xl bg-gradient-to-br from-indigo-500/10 to-purple-500/10 border border-indigo-500/20">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full border-2 border-indigo-500/30 border-t-indigo-500 animate-spin" />
          <div>
            <p className="text-indigo-400 font-medium">Analyzing anomaly...</p>
            <p className="text-xs text-slate-500">AI is determining root cause</p>
          </div>
        </div>
      </div>
    );
  }

  if (!hypothesis) {
    return (
      <div className="p-6 rounded-xl bg-gradient-to-br from-indigo-500/10 to-purple-500/10 border border-indigo-500/20">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full border-2 border-indigo-500/30 border-t-indigo-500 animate-spin" />
          <div>
            <p className="text-indigo-400 font-medium">Generating AI analysis...</p>
            <p className="text-xs text-slate-500">This may take a moment</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 rounded-xl bg-gradient-to-br from-indigo-500/10 to-purple-500/10 border border-indigo-500/20">
      <AIAnalysisDisplay hypothesis={hypothesis} onRegenerate={onRegenerate} />
    </div>
  );
};

// MobiControl Action Icons
const SyncIcon = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
  </svg>
);

const MessageIcon = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
  </svg>
);

const LocationIcon = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
  </svg>
);

const LockIcon = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
  </svg>
);

const RestartIcon = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 4v5h.582m0 0a8.001 8.001 0 0115.356 2M4.582 9H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
  </svg>
);

const ClearCacheIcon = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
  </svg>
);

const DollarIcon = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
      d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
    />
  </svg>
);

// ============================================================================
// Cost Impact Card Component
// ============================================================================

const formatCurrency = (amount: number): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(amount);
};

const CostImpactCard = ({
  costImpact,
  isLoading,
}: {
  costImpact?: AnomalyImpactResponse;
  isLoading: boolean;
}) => {
  if (isLoading) {
    return (
      <div className="p-6 rounded-xl bg-slate-800/30 border border-slate-700/50">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full border-2 border-amber-500/30 border-t-amber-500 animate-spin" />
          <div>
            <p className="text-amber-400 font-medium text-sm">Calculating impact...</p>
            <p className="text-xs text-slate-500">Analyzing financial data</p>
          </div>
        </div>
      </div>
    );
  }

  if (!costImpact || costImpact.using_defaults) {
    return (
      <div className="p-6 rounded-xl bg-slate-800/30 border border-slate-700/50 text-center">
        <div className="w-10 h-10 mx-auto mb-2 rounded-full bg-slate-700/50 flex items-center justify-center text-slate-400">
          <DollarIcon />
        </div>
        <p className="text-slate-400 text-sm">No cost data configured</p>
        <p className="text-xs text-slate-500 mt-1">Configure your labor rates and device costs to see financial impact</p>
        <Link
          to="/costs"
          className="mt-3 inline-block text-xs text-amber-400 hover:text-amber-300 transition-colors"
        >
          Configure Costs →
        </Link>
      </div>
    );
  }

  const impactColors = {
    high: { bg: 'bg-red-500/10', border: 'border-red-500/30', text: 'text-red-400', badge: 'bg-red-500/20 text-red-300' },
    medium: { bg: 'bg-amber-500/10', border: 'border-amber-500/30', text: 'text-amber-400', badge: 'bg-amber-500/20 text-amber-300' },
    low: { bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-400', badge: 'bg-emerald-500/20 text-emerald-300' },
  };

  const colors = impactColors[costImpact.impact_level] || impactColors.medium;

  return (
    <div className={`p-4 rounded-xl ${colors.bg} border ${colors.border}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${colors.bg} ${colors.text}`}>
            <DollarIcon />
          </div>
          <span className="text-sm font-semibold text-white">Financial Impact</span>
        </div>
        <span className={`px-2 py-0.5 rounded text-xs font-bold uppercase ${colors.badge}`}>
          {costImpact.impact_level}
        </span>
      </div>

      {/* Total Impact */}
      <div className="text-center p-4 rounded-lg bg-slate-800/30 mb-4">
        <p className="text-xs text-slate-500 mb-1">Estimated Total Impact</p>
        <p className={`text-2xl font-bold font-mono ${colors.text}`}>
          {formatCurrency(costImpact.total_estimated_impact)}
        </p>
        <p className="text-[10px] text-slate-600 mt-1">
          {(costImpact.overall_confidence * 100).toFixed(0)}% confidence
        </p>
      </div>

      {/* Cost Breakdown */}
      {costImpact.impact_components && costImpact.impact_components.length > 0 && (
        <div className="space-y-2 mb-4">
          <p className="text-xs font-medium text-slate-400">Cost Breakdown</p>
          {costImpact.impact_components.map((component, idx) => (
            <div
              key={idx}
              className="flex items-center justify-between p-2 rounded-lg bg-slate-800/30"
            >
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-slate-300 truncate">{component.type}</p>
                <p className="text-[10px] text-slate-500 truncate">{component.description}</p>
              </div>
              <span className="text-xs font-mono text-slate-300 ml-2">
                {formatCurrency(component.amount)}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Device Value Context */}
      {(costImpact.device_unit_cost || costImpact.device_depreciated_value) && (
        <div className="pt-3 border-t border-slate-700/30">
          <p className="text-xs font-medium text-slate-400 mb-2">Device Context</p>
          <div className="grid grid-cols-2 gap-2">
            {costImpact.device_unit_cost && (
              <div className="text-center p-2 rounded bg-slate-800/30">
                <p className="text-[10px] text-slate-500">Purchase Cost</p>
                <p className="text-xs font-mono text-slate-300">{formatCurrency(costImpact.device_unit_cost)}</p>
              </div>
            )}
            {costImpact.device_depreciated_value && (
              <div className="text-center p-2 rounded bg-slate-800/30">
                <p className="text-[10px] text-slate-500">Current Value</p>
                <p className="text-xs font-mono text-emerald-400">{formatCurrency(costImpact.device_depreciated_value)}</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Explanation */}
      <div className="mt-3 pt-3 border-t border-slate-700/30">
        <p className="text-[10px] text-slate-500">{costImpact.confidence_explanation}</p>
      </div>
    </div>
  );
};

// MobiControl Action Buttons
const MobiControlActions = ({
  onAction,
  loadingAction
}: {
  deviceId: number;
  onAction: (action: string) => void;
  loadingAction?: string | null;
}) => {
  const actions = [
    { id: 'sync', label: 'Sync Device', icon: <SyncIcon />, description: 'Force sync telemetry data' },
    { id: 'message', label: 'Send Message', icon: <MessageIcon />, description: 'Send notification to device' },
    { id: 'locate', label: 'Locate', icon: <LocationIcon />, description: 'Get current location' },
    { id: 'lock', label: 'Lock Device', icon: <LockIcon />, description: 'Remotely lock device' },
    { id: 'restart', label: 'Restart', icon: <RestartIcon />, description: 'Restart device remotely' },
    { id: 'clearCache', label: 'Clear Cache', icon: <ClearCacheIcon />, description: 'Clear app caches' },
  ];

  return (
    <div className="grid grid-cols-3 gap-2">
      {actions.map((action) => {
        const isLoading = loadingAction === action.id;
        const isDisabled = loadingAction !== null;

        return (
          <button
            key={action.id}
            onClick={() => onAction(action.id)}
            disabled={isDisabled}
            className={`p-3 rounded-xl bg-slate-800/50 border border-slate-700/50 transition-all group text-left
                       ${isDisabled ? 'opacity-50 cursor-not-allowed' : 'hover:border-amber-500/30 hover:bg-amber-500/5'}
                       ${isLoading ? 'border-amber-500/50' : ''}`}
          >
            <span className={`mb-2 block ${isLoading ? 'text-amber-400 animate-pulse' : 'text-slate-400 group-hover:text-amber-400'}`}>
              {isLoading ? (
                <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
                </svg>
              ) : action.icon}
            </span>
            <p className={`text-xs font-medium transition-colors ${isLoading ? 'text-amber-400' : 'text-white group-hover:text-amber-400'}`}>
              {isLoading ? 'Executing...' : action.label}
            </p>
            <p className="text-[10px] text-slate-500 mt-0.5">{action.description}</p>
          </button>
        );
      })}
    </div>
  );
};

// Remediation Card
const RemediationCard = ({
  remediation,
  onApply,
  onMarkSuccess,
}: {
  remediation: RemediationSuggestion;
  onApply: () => void;
  onMarkSuccess: () => void;
}) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/50 hover:border-amber-500/20 transition-colors">
      <div className="flex items-start justify-between">
        <div className="flex items-start gap-3">
          <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
            remediation.confidence_level === 'high' ? 'bg-emerald-500/20 text-emerald-400' :
            remediation.confidence_level === 'medium' ? 'bg-amber-500/20 text-amber-400' :
            'bg-slate-700/50 text-slate-400'
          }`}>
            <span className="text-sm font-bold">#{remediation.priority}</span>
          </div>
          <div className="flex-1">
            <h4 className="text-sm font-medium text-white">{remediation.title}</h4>
            <p className="text-xs text-slate-400 mt-1">{remediation.description}</p>
            {remediation.estimated_impact && (
              <p className="text-xs text-emerald-400 mt-1">Impact: {remediation.estimated_impact}</p>
            )}
          </div>
        </div>
        <span className={`text-xs px-2 py-0.5 rounded ${
          remediation.source === 'policy' ? 'bg-blue-500/20 text-blue-400' :
          remediation.source === 'ai_generated' ? 'bg-purple-500/20 text-purple-400' :
          'bg-slate-700/50 text-slate-400'
        }`}>
          {remediation.source === 'ai_generated' ? 'AI' : remediation.source}
        </span>
      </div>

      {expanded && (
        <div className="mt-4 pt-4 border-t border-slate-700/30">
          <p className="text-xs text-slate-500 mb-2">Steps:</p>
          <ol className="space-y-1">
            {remediation.detailed_steps.map((step, i) => (
              <li key={i} className="text-xs text-slate-300 flex items-start gap-2">
                <span className="text-amber-400 font-mono">{i + 1}.</span>
                {step}
              </li>
            ))}
          </ol>
        </div>
      )}

      <div className="mt-3 flex items-center gap-2">
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-xs text-slate-400 hover:text-white transition-colors"
        >
          {expanded ? 'Show less' : 'Show steps'}
        </button>
        <div className="flex-1" />
        <button
          onClick={onMarkSuccess}
          className="text-xs text-emerald-400 hover:text-emerald-300 transition-colors"
        >
          Mark as fixed
        </button>
        <button
          onClick={onApply}
          className="px-3 py-1 text-xs font-medium text-amber-400 bg-amber-500/10 rounded-lg hover:bg-amber-500/20 transition-colors"
        >
          Apply
        </button>
      </div>
    </div>
  );
};

// Similar Cases List
const SimilarCasesList = ({ cases }: { cases: SimilarCase[] }) => {
  if (!cases || cases.length === 0) {
    return <p className="text-sm text-slate-500 text-center py-4">No similar cases found</p>;
  }

  return (
    <div className="space-y-2">
      {cases.slice(0, 5).map((c) => (
        <Link
          key={c.case_id}
          to={`/investigations/${c.anomaly_id}`}
          className="block p-3 rounded-lg bg-slate-800/30 hover:bg-slate-800/50 transition-colors group"
        >
          <div className="flex items-center justify-between">
            <div>
              <span className="text-xs text-slate-500">Case #{c.anomaly_id}</span>
              <p className="text-sm text-slate-300 group-hover:text-white transition-colors">
                {c.device_name || `Device #${c.device_id}`}
              </p>
            </div>
            <div className="text-right">
              <span className={`text-xs px-2 py-0.5 rounded ${
                c.resolution_status === 'resolved' ? 'bg-emerald-500/20 text-emerald-400' :
                c.resolution_status === 'false_positive' ? 'bg-slate-700/50 text-slate-400' :
                'bg-orange-500/20 text-orange-400'
              }`}>
                {c.resolution_status}
              </span>
              <p className="text-[10px] text-slate-500 mt-1">
                {(c.similarity_score * 100).toFixed(0)}% similar
              </p>
            </div>
          </div>
          {c.resolution_summary && (
            <p className="text-xs text-slate-500 mt-1 truncate">{c.resolution_summary}</p>
          )}
        </Link>
      ))}
    </div>
  );
};

// ============================================================================
// Main Component
// ============================================================================

function InvestigationDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  useMockMode(); // Used for conditional mock data in API client
  const anomalyId = parseInt(id || '0');

  const [noteText, setNoteText] = useState('');
  const [showActions, setShowActions] = useState(false);
  const [activeTab, setActiveTab] = useState<'analysis' | 'telemetry' | 'actions'>('analysis');

  // Fetch anomaly details
  const { data: anomaly, isLoading: anomalyLoading } = useQuery({
    queryKey: ['anomaly', anomalyId],
    queryFn: () => api.getAnomaly(anomalyId),
    enabled: !!anomalyId,
  });

  // Fetch investigation panel data
  const { data: investigation } = useQuery({
    queryKey: ['investigation', anomalyId],
    queryFn: () => api.getInvestigationPanel(anomalyId),
    enabled: !!anomalyId,
  });

  // Fetch AI analysis
  const { data: aiAnalysis, isLoading: aiLoading, refetch: refetchAI } = useQuery({
    queryKey: ['aiAnalysis', anomalyId],
    queryFn: () => api.getAIAnalysis(anomalyId),
    enabled: !!anomalyId,
  });

  // Fetch remediations
  const { data: remediations } = useQuery({
    queryKey: ['remediations', anomalyId],
    queryFn: () => api.getRemediations(anomalyId),
    enabled: !!anomalyId,
  });

  // Fetch similar cases
  const { data: similarCases } = useQuery({
    queryKey: ['similarCases', anomalyId],
    queryFn: () => api.getSimilarCases(anomalyId),
    enabled: !!anomalyId,
  });

  // Fetch cost impact
  const { data: costImpact, isLoading: costImpactLoading } = useQuery({
    queryKey: ['anomalyImpact', anomalyId],
    queryFn: () => api.getAnomalyImpact(anomalyId),
    enabled: !!anomalyId,
  });

  // Status update mutation
  const resolveMutation = useMutation({
    mutationFn: ({ status, notes }: { status: string; notes?: string }) =>
      api.resolveAnomaly(anomalyId, status, notes),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['anomaly', anomalyId] });
      queryClient.invalidateQueries({ queryKey: ['anomalies'] });
      setShowActions(false);
      const statusLabel = variables.status === 'resolved' ? 'Resolved'
        : variables.status === 'investigating' ? 'Investigating'
        : variables.status === 'false_positive' ? 'False Positive'
        : variables.status;
      showSuccess(`Status updated to ${statusLabel}`);
    },
    onError: (error) => {
      showError(`Failed to update status: ${error instanceof Error ? error.message : 'Unknown error'}`);
    },
  });

  // Add note mutation
  const addNoteMutation = useMutation({
    mutationFn: (note: string) => api.addNote(anomalyId, note),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['anomaly', anomalyId] });
      setNoteText('');
      showSuccess('Note added successfully');
    },
    onError: (error) => {
      showError(`Failed to add note: ${error instanceof Error ? error.message : 'Unknown error'}`);
    },
  });

  // Learn from fix mutation
  const learnMutation = useMutation({
    mutationFn: (data: LearnFromFix) => api.learnFromFix(anomalyId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['remediations', anomalyId] });
    },
  });

  // State for action execution
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [messageDialogOpen, setMessageDialogOpen] = useState(false);
  const [messageText, setMessageText] = useState('');

  // Handle MobiControl actions
  const handleMobiControlAction = async (action: string) => {
    if (!anomaly?.device_id) return;

    const deviceId = anomaly.device_id;
    setActionLoading(action);

    try {
      let result;
      const reason = `Triggered from investigation #${anomalyId}`;

      switch (action) {
        case 'sync':
          result = await api.syncDevice(deviceId, reason);
          break;
        case 'lock':
          result = await api.lockDevice(deviceId, reason);
          break;
        case 'restart':
          result = await api.restartDevice(deviceId, reason);
          break;
        case 'locate':
          result = await api.locateDevice(deviceId);
          break;
        case 'clearCache':
          result = await api.clearDeviceCache(deviceId);
          break;
        case 'message':
          // Open message dialog instead
          setMessageDialogOpen(true);
          setActionLoading(null);
          return;
        default:
          console.warn(`Unknown action: ${action}`);
          setActionLoading(null);
          return;
      }

      // Log the action as a note and show success toast
      if (result?.success) {
        showSuccess(`${action.charAt(0).toUpperCase() + action.slice(1)} action completed`);
        addNoteMutation.mutate(`Executed MobiControl action: ${action} - ${result.message}`);
      }
    } catch (error) {
      console.error(`Failed to execute ${action}:`, error);
      showError(`Failed to execute ${action}: ${error instanceof Error ? error.message : 'Unknown error'}`);
      addNoteMutation.mutate(`Failed to execute MobiControl action: ${action} - ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setActionLoading(null);
    }
  };

  // Handle sending message
  const handleSendMessage = async () => {
    if (!anomaly?.device_id || !messageText.trim()) return;

    setActionLoading('message');
    try {
      const result = await api.sendMessageToDevice(anomaly.device_id, messageText.trim());
      if (result?.success) {
        showSuccess('Message sent to device');
        addNoteMutation.mutate(`Sent message to device: "${messageText.substring(0, 50)}..."`);
      }
      setMessageDialogOpen(false);
      setMessageText('');
    } catch (error) {
      console.error('Failed to send message:', error);
      showError(`Failed to send message: ${error instanceof Error ? error.message : 'Unknown error'}`);
      addNoteMutation.mutate(`Failed to send message: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setActionLoading(null);
    }
  };

  if (anomalyLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="relative w-16 h-16 mx-auto mb-4">
            <div className="absolute inset-0 rounded-full border-2 border-amber-500/20"></div>
            <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-amber-500 animate-spin"></div>
          </div>
          <p className="text-slate-400 font-mono text-sm">Loading investigation...</p>
        </div>
      </div>
    );
  }

  if (!anomaly) {
    return (
      <div className="flex flex-col items-center justify-center h-96">
        <div className="w-16 h-16 mb-4 rounded-full bg-red-500/20 flex items-center justify-center">
          <svg className="w-8 h-8 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </div>
        <p className="text-red-400 font-medium mb-4">Investigation not found</p>
        <Link to="/investigations" className="text-amber-400 hover:text-amber-300 text-sm">
          ← Back to Investigations
        </Link>
      </div>
    );
  }

  const severityKey = getSeverityKey(anomaly.anomaly_score);
  const severity = severityConfig[severityKey];
  const status = statusConfig[anomaly.status as keyof typeof statusConfig] || statusConfig.open;

  return (
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate('/investigations')}
            className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-bold text-white">Investigation #{anomaly.id}</h1>
              <span className={`px-2 py-1 text-xs font-bold rounded ${severity.bg} ${severity.color} border ${severity.border}`}>
                {severity.label}
              </span>
            </div>
            <p className="text-slate-500 text-sm mt-1">
              {anomaly.device_name || `Device #${anomaly.device_id}`} • Detected {formatDistanceToNowStrict(new Date(anomaly.timestamp))} ago
            </p>
          </div>
        </div>

        {/* Status Update Dropdown */}
        <div className="relative">
          <button
            onClick={() => setShowActions(!showActions)}
            className="btn-stellar flex items-center gap-2"
          >
            Update Status
            <svg className={`w-4 h-4 transition-transform ${showActions ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          <AnimatePresence>
            {showActions && (
              <motion.div
                initial={{ opacity: 0, y: -10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -10, scale: 0.95 }}
                className="absolute right-0 top-full mt-2 w-64 stellar-card rounded-xl p-2 z-10"
              >
                {[
                  { status: 'investigating', label: 'Mark as Investigating', colorClass: 'hover:bg-orange-500/10 hover:text-orange-400' },
                  { status: 'resolved', label: 'Mark as Resolved', colorClass: 'hover:bg-emerald-500/10 hover:text-emerald-400' },
                  { status: 'false_positive', label: 'Mark as False Positive', colorClass: 'hover:bg-slate-700/50 hover:text-slate-300' },
                ].map((action) => (
                  <button
                    key={action.status}
                    onClick={() => resolveMutation.mutate({ status: action.status })}
                    disabled={anomaly.status === action.status || resolveMutation.isPending}
                    className={`w-full px-4 py-3 text-left text-sm font-medium rounded-lg transition-colors
                              text-slate-300 ${action.colorClass}
                              disabled:opacity-30 disabled:cursor-not-allowed`}
                  >
                    {action.label}
                  </button>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Status Banner */}
      <div className={`p-4 rounded-xl border ${status.bg} border-slate-700/50`}>
        <div className="flex items-center gap-4">
          <span className={`text-2xl ${status.color}`}>{status.icon}</span>
          <div>
            <p className={`text-lg font-bold ${status.color}`}>{status.label}</p>
            <p className="text-sm text-slate-400">
              Last updated {format(new Date(anomaly.updated_at), 'MMM d, HH:mm')}
            </p>
          </div>
          <div className="flex-1" />
          <div className="text-right">
            <p className="text-xs text-slate-500">Anomaly Score</p>
            <p className={`text-2xl font-bold font-mono ${severity.color}`}>
              {anomaly.anomaly_score.toFixed(4)}
            </p>
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex gap-2 border-b border-slate-800">
        {[
          { id: 'analysis', label: 'AI Analysis', icon: <BrainIcon /> },
          { id: 'telemetry', label: 'Telemetry', icon: <ChartIcon /> },
          { id: 'actions', label: 'Actions', icon: <WrenchIcon /> },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as typeof activeTab)}
            className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === tab.id
                ? 'border-amber-500 text-amber-400'
                : 'border-transparent text-slate-400 hover:text-white'
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Left Column - Main Content */}
        <div className="xl:col-span-2 space-y-6">
          {/* Analysis Tab */}
          {activeTab === 'analysis' && (
            <>
              {/* AI Root Cause Analysis */}
              <Card title={<span className="flex items-center gap-2"><BrainIcon /> Root Cause Analysis</span>}>
                <AIAnalysisPanel
                  hypothesis={aiAnalysis?.primary_hypothesis}
                  isLoading={aiLoading}
                  onRegenerate={() => refetchAI()}
                />
              </Card>

              {/* Feature Contributions */}
              <Card title={<span className="flex items-center gap-2"><ChartIcon /> Why This Anomaly Was Detected</span>}>
                <p className="text-sm text-slate-400 mb-4">
                  These metrics deviated most significantly from the device's baseline behavior:
                </p>
                <FeatureContributionChart contributions={investigation?.explanation?.feature_contributions || []} />

                {investigation?.explanation?.feature_contributions && investigation.explanation.feature_contributions.length > 0 && (
                  <div className="mt-6 pt-4 border-t border-slate-700/50">
                    <h4 className="text-sm font-medium text-slate-300 mb-3">Key Findings</h4>
                    <div className="space-y-2">
                      {investigation.explanation.feature_contributions.slice(0, 3).map((fc: FeatureContribution, i: number) => (
                        <div key={i} className="flex items-start gap-2 text-sm">
                          <span className="text-amber-400 mt-0.5">→</span>
                          <span className="text-slate-300">{fc.plain_text_explanation}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </Card>

              {/* Baseline Comparison */}
              <Card title={<span className="flex items-center gap-2"><TargetIcon /> Baseline Comparison</span>}>
                <p className="text-sm text-slate-400 mb-4">
                  Current values compared to 30-day rolling average:
                </p>
                <BaselineComparisonViz metrics={investigation?.baseline_comparison?.metrics || []} />
              </Card>
            </>
          )}

          {/* Telemetry Tab */}
          {activeTab === 'telemetry' && (
            <>
              {/* Device Header Card */}
              <Card title={<span className="flex items-center gap-2"><DeviceIcon /> Device Overview</span>}>
                <div className="flex items-center justify-between p-4 rounded-xl bg-gradient-to-r from-slate-800/50 to-slate-800/30 border border-slate-700/50 mb-4">
                  <div className="flex items-center gap-4">
                    <div className="w-14 h-14 rounded-xl bg-amber-500/20 flex items-center justify-center">
                      <DeviceIcon />
                    </div>
                    <div>
                      <p className="text-lg font-bold text-white">{anomaly.device_name || `Device #${anomaly.device_id}`}</p>
                      <p className="text-sm text-slate-400">Telemetry snapshot at anomaly detection</p>
                    </div>
                  </div>
                  <Link
                    to={`/devices/${anomaly.device_id}`}
                    className="px-4 py-2 text-sm font-medium text-amber-400 bg-amber-500/10 rounded-lg hover:bg-amber-500/20 transition-colors"
                  >
                    View Full Profile
                  </Link>
                </div>

                {/* Quick Health Indicators */}
                <div className="grid grid-cols-4 gap-3">
                  {/* Battery Health */}
                  <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/50">
                    <div className="flex items-center gap-2 mb-2">
                      <svg className="w-4 h-4 text-orange-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8V5a1 1 0 00-1-1H8a1 1 0 00-1 1v3M7 8h10l1 12H6L7 8z" />
                      </svg>
                      <span className="text-xs text-slate-500">Battery</span>
                    </div>
                    <p className={`text-xl font-bold font-mono ${
                      (anomaly.total_battery_level_drop ?? 0) > 30 ? 'text-red-400' :
                      (anomaly.total_battery_level_drop ?? 0) > 15 ? 'text-orange-400' : 'text-emerald-400'
                    }`}>
                      {anomaly.total_battery_level_drop != null ? `-${anomaly.total_battery_level_drop}%` : 'N/A'}
                    </p>
                    <p className="text-[10px] text-slate-600 mt-1">24h drain</p>
                  </div>

                  {/* Storage Health */}
                  <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/50">
                    <div className="flex items-center gap-2 mb-2">
                      <svg className="w-4 h-4 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                      </svg>
                      <span className="text-xs text-slate-500">Storage</span>
                    </div>
                    <p className="text-xl font-bold font-mono text-cyan-400">
                      {anomaly.total_free_storage_kb != null ? `${(anomaly.total_free_storage_kb / 1024 / 1024).toFixed(1)}GB` : 'N/A'}
                    </p>
                    <p className="text-[10px] text-slate-600 mt-1">free space</p>
                  </div>

                  {/* Network Health */}
                  <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/50">
                    <div className="flex items-center gap-2 mb-2">
                      <svg className="w-4 h-4 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.111 16.404a5.5 5.5 0 017.778 0M12 20h.01m-7.08-7.071c3.904-3.905 10.236-3.905 14.141 0M1.394 9.393c5.857-5.857 15.355-5.857 21.213 0" />
                      </svg>
                      <span className="text-xs text-slate-500">WiFi Signal</span>
                    </div>
                    <p className={`text-xl font-bold font-mono ${
                      (anomaly.wifi_signal_strength ?? -100) > -50 ? 'text-emerald-400' :
                      (anomaly.wifi_signal_strength ?? -100) > -70 ? 'text-amber-400' : 'text-red-400'
                    }`}>
                      {anomaly.wifi_signal_strength != null ? `${anomaly.wifi_signal_strength}dBm` : 'N/A'}
                    </p>
                    <p className="text-[10px] text-slate-600 mt-1">signal strength</p>
                  </div>

                  {/* Connectivity Health */}
                  <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/50">
                    <div className="flex items-center gap-2 mb-2">
                      <svg className="w-4 h-4 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 5.636a9 9 0 010 12.728m0 0l-2.829-2.829m2.829 2.829L21 21M15.536 8.464a5 5 0 010 7.072m0 0l-2.829-2.829m-4.243 2.829a5 5 0 01-7.072 0" />
                      </svg>
                      <span className="text-xs text-slate-500">Disconnects</span>
                    </div>
                    <p className={`text-xl font-bold font-mono ${
                      (anomaly.disconnect_count ?? 0) > 10 ? 'text-red-400' :
                      (anomaly.disconnect_count ?? 0) > 3 ? 'text-orange-400' : 'text-emerald-400'
                    }`}>
                      {anomaly.disconnect_count != null ? anomaly.disconnect_count : 'N/A'}
                    </p>
                    <p className="text-[10px] text-slate-600 mt-1">in period</p>
                  </div>
                </div>
              </Card>

              {/* Network & Data Transfer */}
              <Card title={
                <span className="flex items-center gap-2">
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
                  </svg>
                  Network & Data Transfer
                </span>
              }>
                <div className="grid grid-cols-2 gap-4 mb-4">
                  {/* Download */}
                  <div className="p-4 rounded-xl bg-gradient-to-br from-emerald-500/10 to-emerald-500/5 border border-emerald-500/20">
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-xs text-emerald-400 font-medium">DOWNLOAD</span>
                      <svg className="w-4 h-4 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                      </svg>
                    </div>
                    <p className="text-3xl font-bold font-mono text-emerald-400">
                      {formatDataSize(anomaly.download).value}
                    </p>
                    <p className="text-sm text-slate-500 mt-1">{formatDataSize(anomaly.download).unit} transferred</p>
                    {anomaly.download != null && anomaly.download > 100 && (
                      <div className="mt-2 px-2 py-1 rounded bg-emerald-500/10 text-xs text-emerald-400">
                        High data usage detected
                      </div>
                    )}
                  </div>

                  {/* Upload */}
                  <div className="p-4 rounded-xl bg-gradient-to-br from-purple-500/10 to-purple-500/5 border border-purple-500/20">
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-xs text-purple-400 font-medium">UPLOAD</span>
                      <svg className="w-4 h-4 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
                      </svg>
                    </div>
                    <p className="text-3xl font-bold font-mono text-purple-400">
                      {formatDataSize(anomaly.upload).value}
                    </p>
                    <p className="text-sm text-slate-500 mt-1">{formatDataSize(anomaly.upload).unit} transferred</p>
                    {anomaly.upload != null && anomaly.upload > 50 && (
                      <div className="mt-2 px-2 py-1 rounded bg-purple-500/10 text-xs text-purple-400">
                        High upload activity
                      </div>
                    )}
                  </div>
                </div>

                {/* Data Transfer Summary Bar */}
                <div className="p-3 rounded-lg bg-slate-800/30 border border-slate-700/50">
                  <div className="flex items-center justify-between text-xs text-slate-500 mb-2">
                    <span>Total Data Transfer</span>
                    <span className="font-mono text-slate-400">
                      {formatDataSize((anomaly.download ?? 0) + (anomaly.upload ?? 0)).value} {formatDataSize((anomaly.download ?? 0) + (anomaly.upload ?? 0)).unit}
                    </span>
                  </div>
                  <div className="h-2 bg-slate-700/50 rounded-full overflow-hidden flex">
                    <div
                      className="h-full bg-emerald-500/70"
                      style={{
                        width: `${Math.max(5, ((anomaly.download ?? 0) / ((anomaly.download ?? 0) + (anomaly.upload ?? 1))) * 100)}%`
                      }}
                    />
                    <div
                      className="h-full bg-purple-500/70"
                      style={{
                        width: `${Math.max(5, ((anomaly.upload ?? 0) / ((anomaly.download ?? 1) + (anomaly.upload ?? 0))) * 100)}%`
                      }}
                    />
                  </div>
                  <div className="flex justify-between mt-2 text-[10px]">
                    <span className="text-emerald-400">Download</span>
                    <span className="text-purple-400">Upload</span>
                  </div>
                </div>
              </Card>

              {/* Connectivity Status */}
              <Card title={
                <span className="flex items-center gap-2">
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Connectivity & Uptime
                </span>
              }>
                <div className="grid grid-cols-3 gap-4 mb-4">
                  {/* Offline Time */}
                  <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/50">
                    <p className="text-xs text-slate-500 mb-2">Offline Duration</p>
                    <p className={`text-2xl font-bold font-mono ${
                      (anomaly.offline_time ?? 0) > 60 ? 'text-red-400' :
                      (anomaly.offline_time ?? 0) > 15 ? 'text-orange-400' : 'text-emerald-400'
                    }`}>
                      {anomaly.offline_time != null ? anomaly.offline_time.toFixed(0) : '0'}
                    </p>
                    <p className="text-xs text-slate-600">minutes</p>
                  </div>

                  {/* Connection Time */}
                  <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/50">
                    <p className="text-xs text-slate-500 mb-2">Avg Connection Time</p>
                    <p className="text-2xl font-bold font-mono text-cyan-400">
                      {anomaly.connection_time != null ? anomaly.connection_time.toFixed(1) : '0'}
                    </p>
                    <p className="text-xs text-slate-600">seconds</p>
                  </div>

                  {/* Disconnect Events */}
                  <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/50">
                    <p className="text-xs text-slate-500 mb-2">Disconnect Events</p>
                    <p className={`text-2xl font-bold font-mono ${
                      (anomaly.disconnect_count ?? 0) > 10 ? 'text-red-400' :
                      (anomaly.disconnect_count ?? 0) > 3 ? 'text-amber-400' : 'text-slate-300'
                    }`}>
                      {anomaly.disconnect_count ?? 0}
                    </p>
                    <p className="text-xs text-slate-600">occurrences</p>
                  </div>
                </div>

                {/* Uptime Visualization */}
                <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/50">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm font-medium text-slate-300">Uptime Analysis</span>
                    <span className={`text-xs px-2 py-0.5 rounded ${
                      (anomaly.offline_time ?? 0) < 15 ? 'bg-emerald-500/20 text-emerald-400' :
                      (anomaly.offline_time ?? 0) < 60 ? 'bg-amber-500/20 text-amber-400' : 'bg-red-500/20 text-red-400'
                    }`}>
                      {(anomaly.offline_time ?? 0) < 15 ? 'Healthy' :
                       (anomaly.offline_time ?? 0) < 60 ? 'Degraded' : 'Critical'}
                    </span>
                  </div>
                  <div className="h-8 bg-slate-700/30 rounded-lg overflow-hidden flex">
                    <div
                      className="h-full bg-emerald-500/50 flex items-center justify-center text-[10px] text-emerald-200 font-medium"
                      style={{ width: `${Math.max(10, 100 - ((anomaly.offline_time ?? 0) / 1440) * 100)}%` }}
                    >
                      Online
                    </div>
                    {(anomaly.offline_time ?? 0) > 0 && (
                      <div
                        className="h-full bg-red-500/50 flex items-center justify-center text-[10px] text-red-200 font-medium"
                        style={{ width: `${Math.min(90, Math.max(10, ((anomaly.offline_time ?? 0) / 1440) * 100))}%` }}
                      >
                        Offline
                      </div>
                    )}
                  </div>
                  <p className="text-[10px] text-slate-600 mt-2">24-hour period breakdown</p>
                </div>
              </Card>

              {/* Power & Storage */}
              <Card title={
                <span className="flex items-center gap-2">
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  Power & Storage
                </span>
              }>
                <div className="grid grid-cols-2 gap-4">
                  {/* Battery Section */}
                  <div className="p-4 rounded-xl bg-gradient-to-br from-orange-500/10 to-orange-500/5 border border-orange-500/20">
                    <div className="flex items-center justify-between mb-4">
                      <span className="text-sm font-medium text-orange-400">Battery Health</span>
                      <svg className="w-5 h-5 text-orange-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8V5a1 1 0 00-1-1H8a1 1 0 00-1 1v3M7 8h10l1 12H6L7 8z" />
                      </svg>
                    </div>
                    <div className="space-y-3">
                      <div>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-slate-500">Battery Drain (24h)</span>
                          <span className={`font-mono ${
                            (anomaly.total_battery_level_drop ?? 0) > 30 ? 'text-red-400' :
                            (anomaly.total_battery_level_drop ?? 0) > 15 ? 'text-orange-400' : 'text-emerald-400'
                          }`}>
                            {anomaly.total_battery_level_drop ?? 0}%
                          </span>
                        </div>
                        <div className="h-3 bg-slate-700/50 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all ${
                              (anomaly.total_battery_level_drop ?? 0) > 30 ? 'bg-red-500' :
                              (anomaly.total_battery_level_drop ?? 0) > 15 ? 'bg-orange-500' : 'bg-emerald-500'
                            }`}
                            style={{ width: `${Math.min(100, anomaly.total_battery_level_drop ?? 0)}%` }}
                          />
                        </div>
                      </div>
                      <div className="pt-2 border-t border-orange-500/20">
                        <p className="text-[10px] text-slate-500">
                          {(anomaly.total_battery_level_drop ?? 0) > 30
                            ? 'High battery consumption detected - check running apps'
                            : (anomaly.total_battery_level_drop ?? 0) > 15
                            ? 'Moderate battery usage - within expected range'
                            : 'Normal battery consumption'}
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Storage Section */}
                  <div className="p-4 rounded-xl bg-gradient-to-br from-cyan-500/10 to-cyan-500/5 border border-cyan-500/20">
                    <div className="flex items-center justify-between mb-4">
                      <span className="text-sm font-medium text-cyan-400">Storage Status</span>
                      <svg className="w-5 h-5 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                      </svg>
                    </div>
                    <div className="space-y-3">
                      <div>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-slate-500">Free Space</span>
                          <span className="font-mono text-cyan-400">
                            {anomaly.total_free_storage_kb != null
                              ? `${(anomaly.total_free_storage_kb / 1024 / 1024).toFixed(1)} GB`
                              : 'N/A'}
                          </span>
                        </div>
                        <div className="h-3 bg-slate-700/50 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-cyan-500 rounded-full"
                            style={{
                              width: anomaly.total_free_storage_kb != null
                                ? `${Math.min(100, (anomaly.total_free_storage_kb / 1024 / 1024 / 64) * 100)}%`
                                : '0%'
                            }}
                          />
                        </div>
                      </div>
                      <div className="pt-2 border-t border-cyan-500/20">
                        <p className="text-[10px] text-slate-500">
                          {anomaly.total_free_storage_kb != null && anomaly.total_free_storage_kb < 1024 * 1024 * 2
                            ? 'Low storage warning - cleanup recommended'
                            : 'Storage capacity healthy'}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </Card>

              {/* Raw Telemetry Data */}
              <Card title="All Telemetry Metrics">
                <div className="overflow-hidden rounded-lg border border-slate-700/50">
                  <table className="w-full text-sm">
                    <thead className="bg-slate-800/50">
                      <tr>
                        <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">Metric</th>
                        <th className="px-4 py-3 text-right text-xs font-medium text-slate-400 uppercase tracking-wider">Value</th>
                        <th className="px-4 py-3 text-center text-xs font-medium text-slate-400 uppercase tracking-wider">Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-700/30">
                      {[
                        { label: 'Battery Level Drop', value: anomaly.total_battery_level_drop?.toFixed(0), unit: '%', threshold: { warn: 15, crit: 30 } },
                        { label: 'Free Storage', value: anomaly.total_free_storage_kb ? (anomaly.total_free_storage_kb / 1024 / 1024).toFixed(2) : null, unit: 'GB', threshold: { warn: 5, crit: 2 }, inverse: true },
                        { label: 'Data Download', value: formatDataSize(anomaly.download).value, unit: formatDataSize(anomaly.download).unit, threshold: { warn: 100, crit: 500 } },
                        { label: 'Data Upload', value: formatDataSize(anomaly.upload).value, unit: formatDataSize(anomaly.upload).unit, threshold: { warn: 50, crit: 200 } },
                        { label: 'Offline Time', value: anomaly.offline_time?.toFixed(1), unit: 'min', threshold: { warn: 15, crit: 60 } },
                        { label: 'Disconnect Count', value: anomaly.disconnect_count?.toFixed(0), unit: '', threshold: { warn: 3, crit: 10 } },
                        { label: 'WiFi Signal Strength', value: anomaly.wifi_signal_strength?.toFixed(0), unit: 'dBm', threshold: { warn: -70, crit: -80 }, inverse: true },
                        { label: 'Connection Time', value: anomaly.connection_time?.toFixed(2), unit: 's', threshold: { warn: 5, crit: 15 } },
                      ].map((metric, i) => {
                        const numValue = parseFloat(metric.value ?? '0');
                        const getStatus = () => {
                          if (metric.value == null) return 'unknown';
                          if (metric.inverse) {
                            if (numValue <= metric.threshold.crit) return 'critical';
                            if (numValue <= metric.threshold.warn) return 'warning';
                            return 'normal';
                          }
                          if (numValue >= metric.threshold.crit) return 'critical';
                          if (numValue >= metric.threshold.warn) return 'warning';
                          return 'normal';
                        };
                        const status = getStatus();
                        return (
                          <tr key={i} className="hover:bg-slate-800/30 transition-colors">
                            <td className="px-4 py-3 text-slate-300">{metric.label}</td>
                            <td className="px-4 py-3 text-right font-mono text-white">
                              {metric.value != null ? `${metric.value} ${metric.unit}` : 'N/A'}
                            </td>
                            <td className="px-4 py-3 text-center">
                              <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium ${
                                status === 'critical' ? 'bg-red-500/20 text-red-400' :
                                status === 'warning' ? 'bg-amber-500/20 text-amber-400' :
                                status === 'unknown' ? 'bg-slate-700/50 text-slate-400' :
                                'bg-emerald-500/20 text-emerald-400'
                              }`}>
                                {status === 'critical' ? 'Critical' :
                                 status === 'warning' ? 'Warning' :
                                 status === 'unknown' ? 'N/A' : 'Normal'}
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </Card>
            </>
          )}

          {/* Actions Tab */}
          {activeTab === 'actions' && (
            <>
              {/* MobiControl Actions */}
              <Card title={<span className="flex items-center gap-2"><WrenchIcon /> SOTI MobiControl Actions</span>}>
                <p className="text-sm text-slate-400 mb-4">
                  Execute remote actions on this device via MobiControl:
                </p>
                <MobiControlActions deviceId={anomaly.device_id} onAction={handleMobiControlAction} loadingAction={actionLoading} />

                {/* Send Message Dialog */}
                {messageDialogOpen && (
                  <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
                    <div className="bg-slate-800 rounded-xl p-6 w-full max-w-md border border-slate-700">
                      <h3 className="text-lg font-semibold text-white mb-4">Send Message to Device</h3>
                      <textarea
                        value={messageText}
                        onChange={(e) => setMessageText(e.target.value)}
                        placeholder="Enter message to send to device..."
                        className="w-full h-32 p-3 bg-slate-900 border border-slate-700 rounded-lg text-white placeholder-slate-500 resize-none focus:outline-none focus:ring-2 focus:ring-amber-500"
                      />
                      <div className="flex justify-end gap-3 mt-4">
                        <button
                          onClick={() => { setMessageDialogOpen(false); setMessageText(''); }}
                          className="px-4 py-2 text-slate-400 hover:text-white transition-colors"
                        >
                          Cancel
                        </button>
                        <button
                          onClick={handleSendMessage}
                          disabled={!messageText.trim() || actionLoading === 'message'}
                          className="px-4 py-2 bg-amber-500 text-white rounded-lg hover:bg-amber-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          {actionLoading === 'message' ? 'Sending...' : 'Send Message'}
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </Card>

              {/* Remediation Suggestions */}
              <Card title="AI-Powered Remediation Suggestions">
                <div className="space-y-3">
                  {remediations && remediations.length > 0 ? (
                    remediations.map((r: RemediationSuggestion) => (
                      <RemediationCard
                        key={r.remediation_id}
                        remediation={r}
                        onApply={() => handleMobiControlAction('remediation')}
                        onMarkSuccess={() => learnMutation.mutate({ remediation_description: r.title, tags: [] })}
                      />
                    ))
                  ) : (
                    <p className="text-sm text-slate-500 text-center py-4">
                      No remediation suggestions available
                    </p>
                  )}
                </div>
              </Card>
            </>
          )}
        </div>

        {/* Right Column - Sidebar */}
        <div className="space-y-6">
          {/* Cost Impact */}
          <Card title={<span className="flex items-center gap-2"><DollarIcon /> Cost Impact</span>}>
            <CostImpactCard costImpact={costImpact} isLoading={costImpactLoading} />
          </Card>

          {/* Similar Cases */}
          <Card title="Similar Cases">
            <SimilarCasesList cases={similarCases || []} />
          </Card>

          {/* Investigation Notes */}
          <Card title="Investigation Log">
            <div className="space-y-3 max-h-[300px] overflow-y-auto mb-4">
              {anomaly.investigation_notes && anomaly.investigation_notes.length > 0 ? (
                anomaly.investigation_notes.map((note: InvestigationNote) => (
                  <div
                    key={note.id}
                    className="relative pl-4 border-l-2 border-amber-500/30"
                  >
                    <div className="absolute left-0 top-1 w-2 h-2 -translate-x-[5px] rounded-full bg-amber-500" />
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-medium text-amber-400">{note.user}</span>
                      <span className="text-[10px] text-slate-600 font-mono">
                        {format(new Date(note.created_at), 'MMM d, HH:mm')}
                      </span>
                    </div>
                    <p className="text-sm text-slate-300">{note.note}</p>
                  </div>
                ))
              ) : (
                <p className="text-sm text-slate-500 text-center py-4">No notes yet</p>
              )}
            </div>

            {/* Add Note Form */}
            <div className="pt-4 border-t border-slate-700/50">
              <textarea
                value={noteText}
                onChange={(e) => setNoteText(e.target.value)}
                placeholder="Add investigation note..."
                className="input-stellar w-full resize-none"
                rows={3}
                maxLength={1000}
              />
              <div className="flex justify-between items-center mt-2">
                <span className="text-xs text-slate-500">{noteText.length}/1000</span>
                <button
                  onClick={() => noteText.trim() && addNoteMutation.mutate(noteText)}
                  disabled={!noteText.trim() || noteText.trim().length < 5 || addNoteMutation.isPending}
                  className="btn-ghost text-sm disabled:opacity-30"
                >
                  {addNoteMutation.isPending ? 'Adding...' : 'Add Note'}
                </button>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </motion.div>
  );
}

export default InvestigationDetail;
