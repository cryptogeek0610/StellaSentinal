/**
 * Investigation Detail Page - Stella Sentinel
 *
 * Comprehensive anomaly investigation panel with:
 * - Summary banner with severity and key metrics
 * - Detection explanation with feature contributions
 * - Baseline comparison with deviation analysis
 * - Evidence timeline with correlated events
 * - AI-assisted root cause analysis and remediation suggestions
 */

import { useState } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../api/client';
import { format, formatDistanceToNowStrict } from 'date-fns';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from '../components/Card';
import { useMockMode } from '../hooks/useMockMode';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  ReferenceLine,
  ReferenceArea,
} from 'recharts';
import type {
  FeatureContribution,
  BaselineMetric,
  EvidenceEvent,
  RemediationSuggestion,
  SimilarCase,
} from '../types/anomaly';

// Severity configuration
const severityConfig = {
  critical: {
    label: 'CRITICAL',
    description: 'Requires immediate attention',
    textColor: 'text-red-400',
    bgColor: 'bg-red-500/20',
    borderColor: 'border-red-500/30',
    barColor: 'bg-red-500',
    gradientFrom: 'from-red-500/20',
    gradientTo: 'to-red-600/5',
  },
  high: {
    label: 'HIGH',
    description: 'Should be investigated soon',
    textColor: 'text-orange-400',
    bgColor: 'bg-orange-500/20',
    borderColor: 'border-orange-500/30',
    barColor: 'bg-orange-500',
    gradientFrom: 'from-orange-500/20',
    gradientTo: 'to-orange-600/5',
  },
  medium: {
    label: 'MEDIUM',
    description: 'Monitor and investigate',
    textColor: 'text-amber-400',
    bgColor: 'bg-amber-500/20',
    borderColor: 'border-amber-500/30',
    barColor: 'bg-amber-500',
    gradientFrom: 'from-amber-500/20',
    gradientTo: 'to-amber-600/5',
  },
  low: {
    label: 'LOW',
    description: 'Low priority',
    textColor: 'text-slate-400',
    bgColor: 'bg-slate-700/50',
    borderColor: 'border-slate-600/50',
    barColor: 'bg-slate-500',
    gradientFrom: 'from-slate-500/20',
    gradientTo: 'to-slate-600/5',
  },
};

// Status configuration
const statusConfig = {
  open: {
    label: 'Open',
    description: 'Awaiting investigation',
    textColor: 'text-red-400',
    bgColor: 'bg-red-500/10',
    borderColor: 'border-red-500/30',
  },
  investigating: {
    label: 'Investigating',
    description: 'Currently being analyzed',
    textColor: 'text-orange-400',
    bgColor: 'bg-orange-500/10',
    borderColor: 'border-orange-500/30',
  },
  resolved: {
    label: 'Resolved',
    description: 'Issue has been addressed',
    textColor: 'text-emerald-400',
    bgColor: 'bg-emerald-500/10',
    borderColor: 'border-emerald-500/30',
  },
  false_positive: {
    label: 'False Positive',
    description: 'Not a real anomaly',
    textColor: 'text-slate-400',
    bgColor: 'bg-slate-700/30',
    borderColor: 'border-slate-600/50',
  },
};

// Evidence category icons (SVG components)
const CategoryIcons = {
  apps: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 1.5H8.25A2.25 2.25 0 006 3.75v16.5a2.25 2.25 0 002.25 2.25h7.5A2.25 2.25 0 0018 20.25V3.75a2.25 2.25 0 00-2.25-2.25H13.5m-3 0V3h3V1.5m-3 0h3m-3 18.75h3" />
    </svg>
  ),
  storage: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375m16.5 0c0-2.278-3.694-4.125-8.25-4.125S3.75 4.097 3.75 6.375m16.5 0v11.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125V6.375m16.5 0v3.75m-16.5-3.75v3.75m16.5 0v3.75C20.25 16.153 16.556 18 12 18s-8.25-1.847-8.25-4.125v-3.75m16.5 0c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125" />
    </svg>
  ),
  battery: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M21 10.5h.375c.621 0 1.125.504 1.125 1.125v2.25c0 .621-.504 1.125-1.125 1.125H21M4.5 10.5H18V15a2.25 2.25 0 01-2.25 2.25h-9A2.25 2.25 0 014.5 15v-4.5zM4.5 10.5V7.5A2.25 2.25 0 016.75 5.25h9A2.25 2.25 0 0118 7.5v3" />
    </svg>
  ),
  network: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M8.288 15.038a5.25 5.25 0 017.424 0M5.106 11.856c3.807-3.808 9.98-3.808 13.788 0M1.924 8.674c5.565-5.565 14.587-5.565 20.152 0M12.53 18.22l-.53.53-.53-.53a.75.75 0 011.06 0z" />
    </svg>
  ),
  security: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" />
    </svg>
  ),
  system: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.324.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 011.37.49l1.296 2.247a1.125 1.125 0 01-.26 1.431l-1.003.827c-.293.24-.438.613-.431.992a6.759 6.759 0 010 .255c-.007.378.138.75.43.99l1.005.828c.424.35.534.954.26 1.43l-1.298 2.247a1.125 1.125 0 01-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.57 6.57 0 01-.22.128c-.331.183-.581.495-.644.869l-.213 1.28c-.09.543-.56.941-1.11.941h-2.594c-.55 0-1.02-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 01-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 01-1.369-.49l-1.297-2.247a1.125 1.125 0 01.26-1.431l1.004-.827c.292-.24.437-.613.43-.992a6.932 6.932 0 010-.255c.007-.378-.138-.75-.43-.99l-1.004-.828a1.125 1.125 0 01-.26-1.43l1.297-2.247a1.125 1.125 0 011.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.087.22-.128.332-.183.582-.495.644-.869l.214-1.281z" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
    </svg>
  ),
};

const eventCategoryConfig = {
  apps: { color: 'text-purple-400', bgColor: 'bg-purple-500/10' },
  storage: { color: 'text-cyan-400', bgColor: 'bg-cyan-500/10' },
  battery: { color: 'text-orange-400', bgColor: 'bg-orange-500/10' },
  network: { color: 'text-blue-400', bgColor: 'bg-blue-500/10' },
  security: { color: 'text-red-400', bgColor: 'bg-red-500/10' },
  system: { color: 'text-slate-400', bgColor: 'bg-slate-500/10' },
};

// Custom Tooltip for Timeline Chart
const TimelineTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const point = payload[0].payload;
    return (
      <div className="bg-slate-900 border border-slate-700 p-3 rounded-lg shadow-xl">
        <p className="text-slate-400 text-xs mb-1">{label}</p>
        <p className={`font-mono font-bold ${point.is_anomalous ? 'text-red-400' : 'text-amber-400'}`}>
          {payload[0].value.toFixed(2)}
        </p>
        {point.is_anomalous && (
          <p className="text-red-400 text-xs mt-1">Anomalous</p>
        )}
      </div>
    );
  }
  return null;
};

// Feature Contribution Bar Component
function FeatureContributionBar({ contribution, maxContribution }: {
  contribution: FeatureContribution;
  maxContribution: number;
}) {
  const barWidth = (contribution.contribution_percentage / maxContribution) * 100;
  const isPositive = contribution.contribution_direction === 'positive';

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      className="group"
    >
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-white">
            {contribution.feature_display_name}
          </span>
          <span className={`text-xs px-1.5 py-0.5 rounded ${
            isPositive ? 'bg-red-500/20 text-red-400' : 'bg-blue-500/20 text-blue-400'
          }`}>
            {isPositive ? '↑' : '↓'} {contribution.deviation_sigma.toFixed(1)}σ
          </span>
        </div>
        <span className="text-sm font-mono text-amber-400">
          {contribution.contribution_percentage.toFixed(1)}%
        </span>
      </div>

      <div className="relative h-8 bg-slate-800/50 rounded-lg overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${barWidth}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
          className={`absolute h-full ${isPositive ? 'bg-gradient-to-r from-red-500/60 to-red-400/40' : 'bg-gradient-to-r from-blue-500/60 to-blue-400/40'}`}
        />
        <div className="absolute inset-0 flex items-center px-3">
          <span className="text-xs text-white/80">
            {contribution.current_value_display}
          </span>
          <span className="mx-2 text-slate-500">vs</span>
          <span className="text-xs text-slate-400">
            baseline {contribution.baseline_value_display}
          </span>
        </div>
      </div>

      {/* Expandable explanation on hover */}
      <div className="mt-1 text-xs text-slate-500 group-hover:text-slate-400 transition-colors line-clamp-1 group-hover:line-clamp-none">
        {contribution.plain_text_explanation}
      </div>
    </motion.div>
  );
}

// Baseline Metric Row Component
function BaselineMetricRow({ metric }: { metric: BaselineMetric }) {
  const deviationColor = metric.is_anomalous
    ? metric.anomaly_direction === 'above'
      ? 'text-red-400'
      : 'text-blue-400'
    : 'text-slate-400';

  const bgColor = metric.is_anomalous ? 'bg-red-500/5' : '';

  return (
    <tr className={`border-b border-slate-800/50 ${bgColor}`}>
      <td className="py-3 px-4">
        <span className="text-sm font-medium text-white">{metric.metric_display_name}</span>
      </td>
      <td className="py-3 px-4 font-mono text-sm text-white">
        {metric.current_value_display}
      </td>
      <td className="py-3 px-4 font-mono text-sm text-slate-400">
        {metric.baseline_mean.toFixed(1)} ± {metric.baseline_std.toFixed(1)}
      </td>
      <td className={`py-3 px-4 font-mono text-sm ${deviationColor}`}>
        {metric.deviation_sigma > 0 ? '+' : ''}{metric.deviation_sigma.toFixed(1)}σ
      </td>
      <td className="py-3 px-4">
        <div className="flex items-center gap-2">
          <div className="w-16 h-2 bg-slate-800 rounded-full overflow-hidden">
            <div
              className={`h-full ${metric.is_anomalous ? 'bg-red-500' : 'bg-emerald-500'}`}
              style={{ width: `${Math.min(metric.percentile_rank, 100)}%` }}
            />
          </div>
          <span className="text-xs text-slate-500">{metric.percentile_rank.toFixed(0)}%ile</span>
        </div>
      </td>
    </tr>
  );
}

// Evidence Event Card Component
function EvidenceEventCard({ event }: { event: EvidenceEvent }) {
  const category = eventCategoryConfig[event.event_category] || eventCategoryConfig.system;
  const severityColors = {
    critical: 'border-l-red-500',
    high: 'border-l-orange-500',
    medium: 'border-l-amber-500',
    low: 'border-l-slate-500',
    info: 'border-l-blue-500',
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`p-4 bg-slate-800/30 rounded-lg border-l-4 ${severityColors[event.severity]} ${
        event.is_contributing_event ? 'ring-1 ring-amber-500/30' : ''
      }`}
    >
      <div className="flex items-start gap-3">
        <div className={`w-8 h-8 rounded-lg ${category.bgColor} ${category.color} flex items-center justify-center`}>
          {CategoryIcons[event.event_category] || CategoryIcons.system}
        </div>
        <div className="flex-1">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-medium text-white">{event.title}</h4>
            <span className="text-xs text-slate-500 font-mono">
              {format(new Date(event.timestamp), 'HH:mm')}
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">{event.description}</p>
          {event.is_contributing_event && event.contribution_note && (
            <div className="mt-2 px-2 py-1 bg-amber-500/10 rounded text-xs text-amber-400">
              ⚡ {event.contribution_note}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}

// AI Hypothesis Card Component
function HypothesisCard({ hypothesis, isPrimary }: {
  hypothesis: any;
  isPrimary: boolean;
}) {
  const [expanded, setExpanded] = useState(isPrimary);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`p-4 rounded-xl border ${
        isPrimary
          ? 'bg-gradient-to-br from-amber-500/10 to-orange-500/5 border-amber-500/30'
          : 'bg-slate-800/30 border-slate-700/50'
      }`}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            {isPrimary && (
              <span className="px-2 py-0.5 text-xs font-bold bg-amber-500/20 text-amber-400 rounded">
                PRIMARY
              </span>
            )}
            <span className="text-lg font-medium text-white">{hypothesis.title}</span>
          </div>
          <p className="text-sm text-slate-400">{hypothesis.description}</p>
        </div>
        <div className="text-right ml-4">
          <div className="text-2xl font-bold text-amber-400">
            {(hypothesis.likelihood * 100).toFixed(0)}%
          </div>
          <div className="text-xs text-slate-500">Likelihood</div>
        </div>
      </div>

      <button
        onClick={() => setExpanded(!expanded)}
        className="mt-3 text-xs text-amber-400 hover:text-amber-300"
      >
        {expanded ? 'Hide details' : 'Show evidence & actions'}
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="mt-4 space-y-4 overflow-hidden"
          >
            {/* Evidence For */}
            <div>
              <h5 className="text-xs font-bold text-emerald-400 uppercase mb-2">Supporting Evidence</h5>
              <div className="space-y-2">
                {hypothesis.evidence_for.map((ev: any, idx: number) => (
                  <div key={idx} className="flex items-start gap-2 text-sm">
                    <span className={`px-1.5 py-0.5 rounded text-xs ${
                      ev.strength === 'strong' ? 'bg-emerald-500/20 text-emerald-400' :
                      ev.strength === 'moderate' ? 'bg-amber-500/20 text-amber-400' :
                      'bg-slate-500/20 text-slate-400'
                    }`}>
                      {ev.strength}
                    </span>
                    <span className="text-slate-300">{ev.statement}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Evidence Against */}
            {hypothesis.evidence_against?.length > 0 && (
              <div>
                <h5 className="text-xs font-bold text-red-400 uppercase mb-2">Counter Evidence</h5>
                <div className="space-y-2">
                  {hypothesis.evidence_against.map((ev: any, idx: number) => (
                    <div key={idx} className="flex items-start gap-2 text-sm">
                      <span className={`px-1.5 py-0.5 rounded text-xs ${
                        ev.strength === 'strong' ? 'bg-red-500/20 text-red-400' :
                        ev.strength === 'moderate' ? 'bg-orange-500/20 text-orange-400' :
                        'bg-slate-500/20 text-slate-400'
                      }`}>
                        {ev.strength}
                      </span>
                      <span className="text-slate-300">{ev.statement}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Recommended Actions */}
            <div>
              <h5 className="text-xs font-bold text-blue-400 uppercase mb-2">Recommended Actions</h5>
              <ul className="space-y-1">
                {hypothesis.recommended_actions.map((action: string, idx: number) => (
                  <li key={idx} className="text-sm text-slate-300 flex items-start gap-2">
                    <span className="text-blue-400">→</span>
                    {action}
                  </li>
                ))}
              </ul>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// Remediation Suggestion Card
function RemediationCard({ remediation, onApply }: {
  remediation: RemediationSuggestion;
  onApply?: () => void;
}) {
  const [expanded, setExpanded] = useState(false);

  const sourceConfig = {
    learned: { label: 'Learned', color: 'text-emerald-400', bgColor: 'bg-emerald-500/20' },
    ai_generated: { label: 'AI Generated', color: 'text-purple-400', bgColor: 'bg-purple-500/20' },
    policy: { label: 'Policy', color: 'text-blue-400', bgColor: 'bg-blue-500/20' },
  };

  const source = sourceConfig[remediation.source as keyof typeof sourceConfig] || sourceConfig.ai_generated;

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      className="p-4 bg-slate-800/30 rounded-xl border border-slate-700/50 hover:border-slate-600/50 transition-colors"
    >
      <div className="flex items-start justify-between">
        <div className="flex items-start gap-3">
          <div className="w-8 h-8 rounded-lg bg-amber-500/20 flex items-center justify-center text-amber-400 font-bold">
            {remediation.priority}
          </div>
          <div>
            <h4 className="text-sm font-medium text-white">{remediation.title}</h4>
            <p className="text-xs text-slate-400 mt-1">{remediation.description}</p>
          </div>
        </div>
        <div className="flex flex-col items-end gap-1">
          <span className={`px-2 py-0.5 text-xs rounded ${source.bgColor} ${source.color}`}>
            {source.label}
          </span>
          <span className="text-xs text-slate-500">
            {(remediation.confidence_score * 100).toFixed(0)}% confidence
          </span>
        </div>
      </div>

      {remediation.historical_success_rate !== null && (
        <div className="mt-3 flex items-center gap-4 text-xs">
          <div className="flex items-center gap-1">
            <span className="text-slate-500">Success rate:</span>
            <span className="text-emerald-400 font-mono">
              {(remediation.historical_success_rate * 100).toFixed(0)}%
            </span>
          </div>
          <div className="flex items-center gap-1">
            <span className="text-slate-500">from</span>
            <span className="text-slate-300">{remediation.historical_sample_size} cases</span>
          </div>
        </div>
      )}

      <button
        onClick={() => setExpanded(!expanded)}
        className="mt-3 text-xs text-amber-400 hover:text-amber-300"
      >
        {expanded ? 'Hide steps' : 'View steps'}
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="mt-3 overflow-hidden"
          >
            <ol className="space-y-2">
              {remediation.detailed_steps.map((step, idx) => (
                <li key={idx} className="flex items-start gap-2 text-sm text-slate-300">
                  <span className="w-5 h-5 rounded-full bg-slate-700 flex items-center justify-center text-xs text-slate-400">
                    {idx + 1}
                  </span>
                  {step}
                </li>
              ))}
            </ol>

            {remediation.is_automated && (
              <button
                onClick={onApply}
                className="mt-4 w-full py-2 px-4 bg-amber-500/20 hover:bg-amber-500/30 text-amber-400 rounded-lg text-sm font-medium transition-colors"
              >
                Apply Automatically
              </button>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// Similar Case Card
function SimilarCaseCard({ caseData }: { caseData: SimilarCase }) {
  const isResolved = caseData.resolution_status === 'resolved';

  return (
    <div className="p-3 bg-slate-800/30 rounded-lg border border-slate-700/50">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-mono text-slate-500">{caseData.case_id}</span>
        <span className={`px-2 py-0.5 text-xs rounded ${
          isResolved ? 'bg-emerald-500/20 text-emerald-400' : 'bg-orange-500/20 text-orange-400'
        }`}>
          {caseData.resolution_status}
        </span>
      </div>

      <div className="flex items-center gap-2 mb-2">
        <div className="flex-1">
          <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-amber-500 to-orange-500"
              style={{ width: `${caseData.similarity_score * 100}%` }}
            />
          </div>
        </div>
        <span className="text-sm font-mono text-amber-400">
          {(caseData.similarity_score * 100).toFixed(0)}%
        </span>
      </div>

      <div className="flex flex-wrap gap-1">
        {caseData.similarity_factors.map((factor, idx) => (
          <span key={idx} className="px-1.5 py-0.5 text-xs bg-slate-700/50 text-slate-400 rounded">
            {factor}
          </span>
        ))}
      </div>

      {isResolved && caseData.successful_remediation && (
        <div className="mt-2 pt-2 border-t border-slate-700/50">
          <p className="text-xs text-slate-500">Resolved with:</p>
          <p className="text-xs text-emerald-400">{caseData.successful_remediation}</p>
          {caseData.time_to_resolution_hours && (
            <p className="text-xs text-slate-500 mt-1">
              in {caseData.time_to_resolution_hours}h
            </p>
          )}
        </div>
      )}
    </div>
  );
}

// Main Investigation Detail Component
function InvestigationDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  useMockMode(); // Initialize mock mode
  const anomalyId = parseInt(id || '0');

  const [noteText, setNoteText] = useState('');
  const [showActions, setShowActions] = useState(false);
  const [activeTab, setActiveTab] = useState<'explanation' | 'baseline' | 'evidence' | 'ai'>('explanation');
  const [selectedMetric, setSelectedMetric] = useState('total_battery_level_drop');
  const [aiAnalysisLoading, setAiAnalysisLoading] = useState(false);

  // Fetch investigation panel data
  const { data: investigation, isLoading: investigationLoading } = useQuery({
    queryKey: ['investigation', anomalyId],
    queryFn: () => api.getInvestigationPanel(anomalyId),
    enabled: !!anomalyId,
  });

  // Fetch historical timeline for selected metric
  const { data: timeline, isLoading: timelineLoading } = useQuery({
    queryKey: ['timeline', anomalyId, selectedMetric],
    queryFn: () => api.getHistoricalTimeline(anomalyId, selectedMetric, 7),
    enabled: !!anomalyId && activeTab === 'baseline',
  });

  // Fetch basic anomaly data for status/notes
  const { data: anomaly, isLoading: anomalyLoading } = useQuery({
    queryKey: ['anomaly', anomalyId],
    queryFn: () => api.getAnomaly(anomalyId),
    enabled: !!anomalyId,
  });

  const resolveMutation = useMutation({
    mutationFn: ({ status, notes }: { status: string; notes?: string }) =>
      api.resolveAnomaly(anomalyId, status, notes),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['anomaly', anomalyId] });
      queryClient.invalidateQueries({ queryKey: ['investigation', anomalyId] });
      queryClient.invalidateQueries({ queryKey: ['anomalies'] });
      setShowActions(false);
    },
  });

  const addNoteMutation = useMutation({
    mutationFn: (note: string) => api.addNote(anomalyId, note),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['anomaly', anomalyId] });
      setNoteText('');
    },
  });

  const feedbackMutation = useMutation({
    mutationFn: (feedback: { rating: 'helpful' | 'not_helpful' }) =>
      api.submitAIFeedback(anomalyId, feedback),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['investigation', anomalyId] });
    },
  });

  // Format timeline data for chart
  const chartData = timeline?.data_points.map((point) => ({
    time: format(new Date(point.timestamp), 'MM/dd HH:mm'),
    value: point.value,
    is_anomalous: point.is_anomalous,
  })) || [];

  const isLoading = investigationLoading || anomalyLoading;

  if (isLoading) {
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

  if (!investigation || !anomaly) {
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

  const severity = severityConfig[investigation.severity] || severityConfig.medium;
  const status = statusConfig[anomaly.status as keyof typeof statusConfig] || statusConfig.open;
  const maxContribution = Math.max(...investigation.explanation.feature_contributions.map(f => f.contribution_percentage));

  return (
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      {/* Header with back button and actions */}
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
            <h1 className="text-2xl font-bold text-white">Investigation #{anomaly.id}</h1>
            <p className="text-slate-500 text-sm mt-1">
              Device #{investigation.device_id} • Detected {formatDistanceToNowStrict(new Date(investigation.detected_at))} ago
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

      {/* Summary Banner */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className={`p-6 rounded-xl border bg-gradient-to-br ${severity.gradientFrom} ${severity.gradientTo} ${severity.borderColor}`}
      >
        <div className="grid grid-cols-4 gap-6">
          {/* Severity & Score */}
          <div>
            <span className={`inline-block px-3 py-1 text-sm font-bold rounded-lg ${severity.bgColor} ${severity.textColor} border ${severity.borderColor} mb-2`}>
              {severity.label}
            </span>
            <div className="flex items-baseline gap-2">
              <span className={`text-4xl font-bold font-mono ${severity.textColor}`}>
                {investigation.anomaly_score.toFixed(3)}
              </span>
              <span className="text-sm text-slate-500">score</span>
            </div>
            <p className="text-xs text-slate-500 mt-1">{severity.description}</p>
          </div>

          {/* Confidence */}
          <div>
            <p className="text-xs text-slate-500 uppercase tracking-wider mb-2">Detection Confidence</p>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-bold text-white">
                {(investigation.confidence_score * 100).toFixed(0)}%
              </span>
            </div>
            <div className="mt-2 h-2 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-amber-500 to-orange-500"
                style={{ width: `${investigation.confidence_score * 100}%` }}
              />
            </div>
          </div>

          {/* Top Contributors */}
          <div>
            <p className="text-xs text-slate-500 uppercase tracking-wider mb-2">Top Contributing Factors</p>
            <div className="space-y-1">
              {investigation.explanation.top_contributing_features.slice(0, 3).map((feature, idx) => (
                <div key={idx} className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${idx === 0 ? 'bg-red-400' : idx === 1 ? 'bg-orange-400' : 'bg-amber-400'}`} />
                  <span className="text-sm text-white">{feature}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Status */}
          <div>
            <p className="text-xs text-slate-500 uppercase tracking-wider mb-2">Current Status</p>
            <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg ${status.bgColor} border ${status.borderColor}`}>
              <div className={`w-2 h-2 rounded-full ${status.textColor.replace('text-', 'bg-')}`} />
              <span className={`text-sm font-medium ${status.textColor}`}>{status.label}</span>
            </div>
            <p className="text-xs text-slate-500 mt-2">
              Updated {format(new Date(anomaly.updated_at), 'MMM d, HH:mm')}
            </p>
          </div>
        </div>

        {/* Summary Text */}
        <div className="mt-4 pt-4 border-t border-slate-700/50">
          <p className="text-sm text-slate-300 leading-relaxed">
            {investigation.explanation.summary_text}
          </p>
        </div>
      </motion.div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-12 gap-6">
        {/* Left Column - Main Analysis */}
        <div className="col-span-8 space-y-4">
          {/* Tab Navigation */}
          <div className="flex gap-1 p-1 bg-slate-800/50 rounded-lg">
            {[
              { key: 'explanation', label: 'Detection Explanation', icon: (
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5" />
                </svg>
              )},
              { key: 'baseline', label: 'Baseline', icon: (
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
                </svg>
              )},
              { key: 'evidence', label: 'Evidence', icon: (
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              )},
              { key: 'ai', label: 'AI Analysis', icon: (
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25 2.25 0 00-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 001.423 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z" />
                </svg>
              )},
            ].map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key as any)}
                className={`flex-1 py-2.5 px-3 text-sm font-medium rounded-lg transition-all flex items-center justify-center gap-2 ${
                  activeTab === tab.key
                    ? 'bg-amber-500/20 text-amber-400 shadow-lg shadow-amber-500/10'
                    : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
                }`}
              >
                {tab.icon}
                <span className="hidden sm:inline">{tab.label}</span>
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <AnimatePresence mode="wait">
            {activeTab === 'explanation' && (
              <motion.div
                key="explanation"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="grid grid-cols-1 lg:grid-cols-2 gap-4"
              >
                {/* Feature Contributions */}
                <Card title={<span className="telemetry-label">Feature Contributions</span>}>
                  <p className="text-xs text-slate-500 mb-4">
                    Features contributing to anomaly detection
                  </p>
                  <div className="space-y-4">
                    {investigation.explanation.feature_contributions.map((contribution) => (
                      <FeatureContributionBar
                        key={contribution.feature_name}
                        contribution={contribution}
                        maxContribution={maxContribution}
                      />
                    ))}
                  </div>
                  <div className="mt-4 pt-3 border-t border-slate-700/50">
                    <p className="text-[10px] text-slate-600">
                      Method: {investigation.explanation.explanation_method} •
                      {investigation.explanation.explanation_generated_at
                        ? format(new Date(investigation.explanation.explanation_generated_at), ' MMM d, HH:mm')
                        : ' N/A'}
                    </p>
                  </div>
                </Card>

                {/* Detailed Explanation + Quick Stats */}
                <div className="space-y-4">
                  <Card title={<span className="telemetry-label">Analysis Summary</span>}>
                    <p className="text-sm text-slate-300 leading-relaxed">
                      {investigation.explanation.detailed_explanation}
                    </p>
                  </Card>

                  {/* Quick Baseline Stats */}
                  {investigation.baseline_comparison && (
                    <Card title={<span className="telemetry-label">Quick Baseline Stats</span>}>
                      <div className="grid grid-cols-2 gap-3">
                        {investigation.baseline_comparison.metrics.slice(0, 4).map((metric) => (
                          <div key={metric.metric_name} className={`p-3 rounded-lg ${metric.is_anomalous ? 'bg-red-500/10 border border-red-500/20' : 'bg-slate-800/50'}`}>
                            <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">{metric.metric_display_name}</p>
                            <div className="flex items-baseline justify-between">
                              <span className="text-lg font-bold text-white font-mono">{metric.current_value_display}</span>
                              <span className={`text-xs font-mono ${metric.is_anomalous ? 'text-red-400' : 'text-slate-400'}`}>
                                {metric.deviation_sigma > 0 ? '+' : ''}{metric.deviation_sigma.toFixed(1)}σ
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </Card>
                  )}
                </div>
              </motion.div>
            )}

            {activeTab === 'baseline' && (
              <motion.div
                key="baseline"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="grid grid-cols-1 lg:grid-cols-3 gap-4"
              >
                {/* Chart - Spans 2 columns */}
                <div className="lg:col-span-2">
                  <Card title={
                    <div className="flex items-center justify-between w-full">
                      <span className="telemetry-label">Historical Timeline</span>
                      <select
                        value={selectedMetric}
                        onChange={(e) => setSelectedMetric(e.target.value)}
                        className="input-stellar text-xs py-1 px-2"
                      >
                        {investigation.baseline_comparison?.metrics.map((m) => (
                          <option key={m.metric_name} value={m.metric_name}>
                            {m.metric_display_name}
                          </option>
                        ))}
                      </select>
                    </div>
                  }>
                    <div className="h-52">
                      {timelineLoading ? (
                        <div className="h-full flex items-center justify-center">
                          <div className="animate-spin w-8 h-8 border-2 border-amber-500/20 border-t-amber-500 rounded-full" />
                        </div>
                      ) : (
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={chartData}>
                            <defs>
                              <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} />
                                <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                              </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
                            <XAxis dataKey="time" stroke="#64748b" fontSize={10} tickLine={false} axisLine={false} />
                            <YAxis stroke="#64748b" fontSize={10} tickLine={false} axisLine={false} />
                            <Tooltip content={<TimelineTooltip />} />
                            {timeline && (
                              <>
                                <ReferenceArea y1={timeline.baseline_lower} y2={timeline.baseline_upper} fill="#22c55e" fillOpacity={0.1} />
                                <ReferenceLine y={timeline.baseline_mean} stroke="#22c55e" strokeDasharray="3 3" label={{ value: 'Baseline', fill: '#22c55e', fontSize: 10 }} />
                              </>
                            )}
                            <Area type="monotone" dataKey="value" stroke="#f59e0b" fillOpacity={1} fill="url(#colorValue)" />
                          </AreaChart>
                        </ResponsiveContainer>
                      )}
                    </div>
                  </Card>
                </div>

                {/* Overall Score Card */}
                <div className="space-y-4">
                  {investigation.baseline_comparison && (
                    <>
                      <Card>
                        <div className="text-center py-4">
                          <p className="text-xs text-slate-500 uppercase tracking-wider mb-2">Overall Deviation</p>
                          <span className="text-4xl font-bold text-amber-400 font-mono">
                            {investigation.baseline_comparison.overall_deviation_score.toFixed(1)}σ
                          </span>
                          <p className="text-xs text-slate-500 mt-2">
                            {investigation.baseline_comparison.baseline_config.peer_group_size} peer devices
                          </p>
                        </div>
                      </Card>
                      <Card title={<span className="telemetry-label text-xs">Baseline Config</span>}>
                        <div className="space-y-2 text-xs">
                          <div className="flex justify-between">
                            <span className="text-slate-500">Peer Group</span>
                            <span className="text-white">{investigation.baseline_comparison.baseline_config.peer_group_name}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-500">Period</span>
                            <span className="text-white">{investigation.baseline_comparison.baseline_config.baseline_period_days} days</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-500">Method</span>
                            <span className="text-white">{investigation.baseline_comparison.baseline_config.statistical_method}</span>
                          </div>
                        </div>
                      </Card>
                    </>
                  )}
                </div>

                {/* Table - Full width below */}
                <div className="lg:col-span-3">
                  <Card title={<span className="telemetry-label">Deviation Analysis</span>}>
                    {investigation.baseline_comparison && (
                      <div className="overflow-x-auto">
                        <table className="w-full">
                          <thead>
                            <tr className="text-xs text-slate-500 uppercase border-b border-slate-700/50">
                              <th className="py-2 px-3 text-left">Metric</th>
                              <th className="py-2 px-3 text-left">Current</th>
                              <th className="py-2 px-3 text-left">Baseline</th>
                              <th className="py-2 px-3 text-left">Deviation</th>
                              <th className="py-2 px-3 text-left">Percentile</th>
                            </tr>
                          </thead>
                          <tbody>
                            {investigation.baseline_comparison.metrics.map((metric) => (
                              <BaselineMetricRow key={metric.metric_name} metric={metric} />
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </Card>
                </div>
              </motion.div>
            )}

            {activeTab === 'evidence' && (
              <motion.div
                key="evidence"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="grid grid-cols-1 lg:grid-cols-2 gap-4"
              >
                {/* Contributing Events */}
                <Card title={
                  <div className="flex items-center justify-between w-full">
                    <span className="telemetry-label">Contributing Events</span>
                    <span className="text-xs text-amber-400">
                      {investigation.evidence_events.filter(e => e.is_contributing_event).length} key events
                    </span>
                  </div>
                }>
                  <div className="space-y-3 max-h-[400px] overflow-y-auto">
                    {investigation.evidence_events.filter(e => e.is_contributing_event).length > 0 ? (
                      investigation.evidence_events
                        .filter(e => e.is_contributing_event)
                        .map((event) => (
                          <EvidenceEventCard key={event.event_id} event={event} />
                        ))
                    ) : (
                      <div className="text-center py-6 text-slate-500 text-sm">
                        No key contributing events identified
                      </div>
                    )}
                  </div>
                </Card>

                {/* All Events Timeline */}
                <Card title={
                  <div className="flex items-center justify-between w-full">
                    <span className="telemetry-label">Full Timeline</span>
                    <span className="text-xs text-slate-500">
                      {investigation.evidence_event_count} total events
                    </span>
                  </div>
                }>
                  <div className="space-y-3 max-h-[400px] overflow-y-auto">
                    {investigation.evidence_events.length > 0 ? (
                      investigation.evidence_events.map((event) => (
                        <EvidenceEventCard key={event.event_id} event={event} />
                      ))
                    ) : (
                      <div className="text-center py-6 text-slate-500 text-sm">
                        No correlated events found
                      </div>
                    )}
                  </div>
                </Card>
              </motion.div>
            )}

            {activeTab === 'ai' && (
              <motion.div
                key="ai"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="grid grid-cols-1 lg:grid-cols-2 gap-4"
              >
                {/* AI Analysis - Left */}
                <Card title={
                  <div className="flex items-center justify-between w-full">
                    <span className="telemetry-label">Root Cause Analysis</span>
                    {investigation.ai_analysis && (
                      <span className="text-xs text-slate-500">
                        {investigation.ai_analysis.model_used}
                      </span>
                    )}
                  </div>
                }>
                  {investigation.ai_analysis ? (
                    <div className="space-y-4">
                      {/* Confidence Banner */}
                      <div className={`p-3 rounded-lg ${
                        investigation.ai_analysis.confidence_level === 'high' ? 'bg-emerald-500/10 border border-emerald-500/20' :
                        investigation.ai_analysis.confidence_level === 'medium' ? 'bg-amber-500/10 border border-amber-500/20' :
                        'bg-slate-700/30 border border-slate-600/30'
                      }`}>
                        <div className="flex items-center justify-between">
                          <span className={`text-sm font-medium ${
                            investigation.ai_analysis.confidence_level === 'high' ? 'text-emerald-400' :
                            investigation.ai_analysis.confidence_level === 'medium' ? 'text-amber-400' :
                            'text-slate-400'
                          }`}>
                            {investigation.ai_analysis.confidence_level.toUpperCase()} CONFIDENCE
                          </span>
                          <span className="text-sm font-mono text-white">
                            {(investigation.ai_analysis.confidence_score * 100).toFixed(0)}%
                          </span>
                        </div>
                        <p className="text-xs text-slate-400 mt-1">
                          {investigation.ai_analysis.confidence_explanation}
                        </p>
                      </div>

                      {/* Primary Hypothesis */}
                      <HypothesisCard
                        hypothesis={investigation.ai_analysis.primary_hypothesis}
                        isPrimary={true}
                      />

                      {/* Feedback */}
                      {!investigation.ai_analysis.feedback_received && (
                        <div className="pt-3 border-t border-slate-700/50">
                          <p className="text-xs text-slate-400 mb-2">Was this helpful?</p>
                          <div className="flex gap-2">
                            <button
                              onClick={() => feedbackMutation.mutate({ rating: 'helpful' })}
                              disabled={feedbackMutation.isPending}
                              className="flex-1 py-1.5 px-3 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 rounded-lg text-xs transition-colors flex items-center justify-center gap-1"
                            >
                              <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" /></svg>
                              Helpful
                            </button>
                            <button
                              onClick={() => feedbackMutation.mutate({ rating: 'not_helpful' })}
                              disabled={feedbackMutation.isPending}
                              className="flex-1 py-1.5 px-3 bg-slate-700/30 hover:bg-slate-700/50 text-slate-400 rounded-lg text-xs transition-colors flex items-center justify-center gap-1"
                            >
                              <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14H5.236a2 2 0 01-1.789-2.894l3.5-7A2 2 0 018.736 3h4.018a2 2 0 01.485.06l3.76.94m-7 10v5a2 2 0 002 2h.096c.5 0 .905-.405.905-.904 0-.715.211-1.413.608-2.008L17 13V4m-7 10h2m5-10h2a2 2 0 012 2v6a2 2 0 01-2 2h-2.5" /></svg>
                              Not Helpful
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="text-center py-6">
                      <div className="w-12 h-12 mx-auto mb-3 rounded-xl bg-slate-800/50 flex items-center justify-center">
                        <svg className="w-6 h-6 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
                        </svg>
                      </div>
                      <p className="text-slate-400 text-sm mb-3">AI analysis not available</p>
                      <button
                        onClick={() => {
                          setAiAnalysisLoading(true);
                          api.getAIAnalysis(anomalyId, true)
                            .then(() => queryClient.invalidateQueries({ queryKey: ['investigation', anomalyId] }))
                            .finally(() => setAiAnalysisLoading(false));
                        }}
                        disabled={aiAnalysisLoading}
                        className="btn-stellar text-sm"
                      >
                        {aiAnalysisLoading ? 'Generating...' : 'Generate Analysis'}
                      </button>
                    </div>
                  )}
                </Card>

                {/* Right - Alternative Hypotheses + Explanation */}
                <div className="space-y-4">
                  {/* Alternative Hypotheses */}
                  {investigation.ai_analysis?.alternative_hypotheses && investigation.ai_analysis.alternative_hypotheses.length > 0 && (
                    <Card title={<span className="telemetry-label">Alternative Hypotheses</span>}>
                      <div className="space-y-3 max-h-[250px] overflow-y-auto">
                        {investigation.ai_analysis.alternative_hypotheses.map((hyp) => (
                          <HypothesisCard
                            key={hyp.hypothesis_id}
                            hypothesis={hyp}
                            isPrimary={false}
                          />
                        ))}
                      </div>
                    </Card>
                  )}

                  {/* Detailed Explanation */}
                  <Card title={<span className="telemetry-label">Detailed Explanation</span>}>
                    <p className="text-sm text-slate-300 leading-relaxed whitespace-pre-wrap max-h-[200px] overflow-y-auto">
                      {investigation.explanation.detailed_explanation}
                    </p>
                  </Card>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Right Column - Remediation & Similar Cases */}
        <div className="col-span-4 space-y-4">
          {/* Device Quick Link */}
          <Link
            to={`/devices/${investigation.device_id}`}
            className="block p-3 stellar-card-hover rounded-xl group"
          >
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-amber-500/20 to-orange-500/20 flex items-center justify-center">
                <svg className="w-5 h-5 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                    d="M12 18h.01M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z" />
                </svg>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-white group-hover:text-amber-400 transition-colors truncate">
                  Device #{investigation.device_id}
                </p>
                <p className="text-xs text-slate-500">View details</p>
              </div>
              <svg className="w-4 h-4 text-slate-600 group-hover:text-amber-400 transition-colors flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </div>
          </Link>

          {/* Suggested Remediations */}
          <Card title={<span className="telemetry-label">Suggested Remediations</span>}>
            <div className="space-y-3 max-h-[300px] overflow-y-auto">
              {investigation.suggested_remediations.length > 0 ? (
                investigation.suggested_remediations.map((rem) => (
                  <RemediationCard key={rem.remediation_id} remediation={rem} />
                ))
              ) : (
                <p className="text-sm text-slate-500 text-center py-3">
                  No remediation suggestions
                </p>
              )}
            </div>
          </Card>

          {/* Similar Cases */}
          <Card title={
            <div className="flex items-center justify-between w-full">
              <span className="telemetry-label">Similar Cases</span>
              <span className="text-xs text-slate-500">{investigation.similar_cases.length}</span>
            </div>
          }>
            <div className="space-y-2 max-h-[250px] overflow-y-auto">
              {investigation.similar_cases.length > 0 ? (
                investigation.similar_cases.map((c) => (
                  <SimilarCaseCard key={c.case_id} caseData={c} />
                ))
              ) : (
                <p className="text-sm text-slate-500 text-center py-3">
                  No similar cases
                </p>
              )}
            </div>
          </Card>

          {/* Investigation Notes */}
          <Card title={<span className="telemetry-label">Investigation Log</span>}>
            <div className="space-y-3 max-h-[200px] overflow-y-auto">
              {anomaly.investigation_notes && anomaly.investigation_notes.length > 0 ? (
                anomaly.investigation_notes.map((note, index) => (
                  <motion.div
                    key={note.id}
                    initial={{ opacity: 0, x: 10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.03 }}
                    className="relative pl-3 border-l-2 border-amber-500/30"
                  >
                    <div className="absolute left-0 top-1 w-1.5 h-1.5 -translate-x-[4px] rounded-full bg-amber-500" />
                    <div className="flex items-center justify-between mb-0.5">
                      <span className="text-xs font-medium text-amber-400">{note.user}</span>
                      <span className="text-[10px] text-slate-600 font-mono">
                        {format(new Date(note.created_at), 'MMM d')}
                      </span>
                    </div>
                    <p className="text-xs text-slate-300 line-clamp-2">{note.note}</p>
                  </motion.div>
                ))
              ) : (
                <p className="text-xs text-slate-500 text-center py-2">No notes yet</p>
              )}
            </div>

            {/* Add Note Form */}
            <div className="mt-3 pt-3 border-t border-slate-800/50">
              <textarea
                value={noteText}
                onChange={(e) => setNoteText(e.target.value)}
                placeholder="Add note..."
                className="input-stellar w-full resize-none text-sm"
                rows={2}
                maxLength={1000}
              />
              <button
                onClick={() => noteText.trim() && addNoteMutation.mutate(noteText)}
                disabled={!noteText.trim() || noteText.trim().length < 5 || addNoteMutation.isPending}
                className="mt-2 w-full btn-ghost text-xs disabled:opacity-30 disabled:cursor-not-allowed"
              >
                {addNoteMutation.isPending ? 'Adding...' : 'Add Note'}
              </button>
            </div>
          </Card>
        </div>
      </div>
    </motion.div>
  );
}

export default InvestigationDetail;
