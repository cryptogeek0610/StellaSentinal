/**
 * Action Center - The Command Hub
 *
 * Philosophy: "Show what needs fixing, then guide to deeper analysis."
 *
 * This is the ONE screen that matters, but it's also a gateway:
 * - Shows problems ranked by $ impact with one-click fixes
 * - Each category links to its specialized dashboard
 * - Fleet Pulse section provides at-a-glance health with navigation
 * - Issue cards have deep links for investigation
 */

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useNavigate, Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { clsx } from 'clsx';
import { api } from '../api/client';
import type {
  ActionCenterSummary,
  ActionCenterIssue,
} from '../api/client';

// ============================================================================
// Types (use imported types from api/client.ts, local alias for convenience)
// ============================================================================

type Issue = ActionCenterIssue;

// Fleet pulse metrics for the health overview
interface FleetPulseMetrics {
  total_devices: number;
  healthy_devices: number;
  devices_with_issues: number;
  security_score: number | null;
  network_score: number | null;
  open_investigations: number;
  anomalies_today: number;
  avg_battery_health: number | null;
}


// ============================================================================
// Category Configuration with Navigation
// ============================================================================

interface CategoryConfig {
  label: string;
  color: string;
  bgColor: string;
  borderColor: string;
  icon: React.ReactNode;
  dashboard: string;
  dashboardLabel: string;
}

const categoryConfig: Record<string, CategoryConfig> = {
  security_risk: {
    label: 'Security Risk',
    color: 'text-red-400',
    bgColor: 'bg-red-500/10',
    borderColor: 'border-red-500/30',
    dashboard: '/security',
    dashboardLabel: 'Security Posture',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
        />
      </svg>
    ),
  },
  productivity_loss: {
    label: 'Productivity',
    color: 'text-orange-400',
    bgColor: 'bg-orange-500/10',
    borderColor: 'border-orange-500/30',
    dashboard: '/network',
    dashboardLabel: 'Network Intelligence',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M8.111 16.404a5.5 5.5 0 017.778 0M12 20h.01m-7.08-7.071c3.904-3.905 10.236-3.905 14.141 0M1.394 9.393c5.857-5.857 15.355-5.857 21.213 0"
        />
      </svg>
    ),
  },
  impending_failure: {
    label: 'Hardware',
    color: 'text-amber-400',
    bgColor: 'bg-amber-500/10',
    borderColor: 'border-amber-500/30',
    dashboard: '/fleet',
    dashboardLabel: 'Fleet Management',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M12 18h.01M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z"
        />
      </svg>
    ),
  },
  cost_waste: {
    label: 'Cost',
    color: 'text-emerald-400',
    bgColor: 'bg-emerald-500/10',
    borderColor: 'border-emerald-500/30',
    dashboard: '/costs',
    dashboardLabel: 'Cost Analysis',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
        />
      </svg>
    ),
  },
};

// ============================================================================
// Fleet Pulse Component - Quick Health Overview with Navigation
// ============================================================================

function FleetPulse({ metrics }: { metrics: FleetPulseMetrics }) {
  const healthPct = Math.round((metrics.healthy_devices / metrics.total_devices) * 100);

  const pulseItems = [
    {
      label: 'Fleet Health',
      value: `${healthPct}%`,
      subtext: `${metrics.healthy_devices}/${metrics.total_devices} healthy`,
      color: healthPct >= 90 ? 'text-emerald-400' : healthPct >= 70 ? 'text-amber-400' : 'text-red-400',
      link: '/fleet',
      linkLabel: 'View Fleet',
    },
    {
      label: 'Security Score',
      value: metrics.security_score != null ? `${metrics.security_score}` : '--',
      subtext: metrics.security_score != null ? 'out of 100' : 'Not yet available',
      color: metrics.security_score == null ? 'text-slate-500' : metrics.security_score >= 80 ? 'text-emerald-400' : metrics.security_score >= 60 ? 'text-amber-400' : 'text-red-400',
      link: '/security',
      linkLabel: 'Security Details',
    },
    {
      label: 'Network Score',
      value: metrics.network_score != null ? `${metrics.network_score}` : '--',
      subtext: metrics.network_score != null ? 'out of 100' : 'Not yet available',
      color: metrics.network_score == null ? 'text-slate-500' : metrics.network_score >= 80 ? 'text-emerald-400' : metrics.network_score >= 60 ? 'text-amber-400' : 'text-red-400',
      link: '/network',
      linkLabel: 'Network Details',
    },
    {
      label: 'Investigations',
      value: `${metrics.open_investigations}`,
      subtext: `${metrics.anomalies_today} anomalies today`,
      color: metrics.open_investigations === 0 ? 'text-emerald-400' : metrics.open_investigations <= 5 ? 'text-amber-400' : 'text-red-400',
      link: '/investigations',
      linkLabel: 'View All',
    },
  ];

  return (
    <div className="grid grid-cols-4 gap-4 mb-6">
      {pulseItems.map((item, idx) => (
        <Link
          key={idx}
          to={item.link}
          className="stellar-glass rounded-xl p-4 border border-slate-700/50 hover:border-slate-600/50 transition-all group"
        >
          <div className="flex items-start justify-between mb-2">
            <p className="text-xs text-slate-500 uppercase tracking-wider">{item.label}</p>
            <svg className="w-4 h-4 text-slate-600 group-hover:text-slate-400 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </div>
          <div className={clsx('text-3xl font-bold', item.color)}>{item.value}</div>
          <p className="text-xs text-slate-500 mt-1">{item.subtext}</p>
          <p className="text-xs text-slate-600 mt-2 group-hover:text-amber-400 transition-colors">
            {item.linkLabel} →
          </p>
        </Link>
      ))}
    </div>
  );
}

// ============================================================================
// Category Cards - Clickable Navigation to Dashboards
// ============================================================================

function CategoryCards({
  summary,
  onCategoryClick
}: {
  summary: ActionCenterSummary;
  onCategoryClick: (category: string) => void;
}) {
  const navigate = useNavigate();

  return (
    <div className="grid grid-cols-4 gap-4 mb-6">
      {Object.entries(summary.by_category).map(([cat, data]) => {
        const config = categoryConfig[cat];
        if (!config) return null;

        return (
          <motion.div
            key={cat}
            whileHover={{ scale: 1.02 }}
            className={clsx(
              'rounded-xl p-4 border cursor-pointer transition-all',
              config.bgColor, config.borderColor,
              'hover:shadow-lg'
            )}
            onClick={() => onCategoryClick(cat)}
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <span className={config.color}>{config.icon}</span>
                <span className={clsx('text-sm font-medium', config.color)}>{config.label}</span>
              </div>
              <span className={clsx('text-2xl font-bold', config.color)}>{data.count}</span>
            </div>

            <div className="text-xs text-slate-400 mb-3">
              {data.devices} devices &middot; ${data.hourly_cost.toFixed(0)}/hr
            </div>

            <button
              onClick={(e) => {
                e.stopPropagation();
                navigate(config.dashboard);
              }}
              className={clsx(
                'w-full text-xs py-1.5 rounded-lg transition-colors',
                'bg-slate-800/50 hover:bg-slate-700/50',
                'text-slate-400 hover:text-white',
                'flex items-center justify-center gap-1'
              )}
            >
              <span>{config.dashboardLabel}</span>
              <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
              </svg>
            </button>
          </motion.div>
        );
      })}
    </div>
  );
}

// ============================================================================
// Executive Summary with Navigation
// ============================================================================

function ExecutiveSummary({ summary }: { summary: ActionCenterSummary }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="stellar-glass rounded-2xl p-6 border border-slate-700/50 mb-6"
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-slate-400 mb-1">Issues Costing You</p>
          <div className="flex items-baseline gap-2">
            <span className="text-5xl font-bold text-white">${summary.total_hourly_cost.toLocaleString()}</span>
            <span className="text-xl text-slate-400">/hour</span>
          </div>
          <p className="text-slate-500 mt-2">
            ${summary.daily_cost.toLocaleString()}/day &middot; ${Math.round(summary.monthly_cost).toLocaleString()}/month
          </p>
        </div>

        <div className="text-right">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-emerald-500/20 border border-emerald-500/30">
            <svg className="w-5 h-5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            <span className="text-emerald-400 font-semibold">
              {summary.automatable_count} can auto-fix
            </span>
          </div>
          <p className="text-slate-500 text-sm mt-2">
            Save ${summary.automatable_savings.toLocaleString()}/hr with one click
          </p>
        </div>
      </div>

      {/* Recommended Action */}
      <div className="mt-6 p-4 rounded-xl bg-gradient-to-r from-amber-500/20 to-orange-500/20 border border-amber-500/30">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-amber-500/30 flex items-center justify-center">
            <svg className="w-5 h-5 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
              />
            </svg>
          </div>
          <div className="flex-1">
            <p className="text-xs text-amber-400/80 uppercase tracking-wider font-semibold">Recommended Action</p>
            <p className="text-white font-medium">{summary.recommended_action}</p>
          </div>
          <Link
            to="/insights"
            className="text-xs text-amber-400/70 hover:text-amber-400 transition-colors flex items-center gap-1"
          >
            <span>AI Insights</span>
            <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
            </svg>
          </Link>
        </div>
      </div>
    </motion.div>
  );
}

// ============================================================================
// Issue Card with Deep Links
// ============================================================================

function IssueCard({
  issue,
  onFix,
  isFixing,
}: {
  issue: Issue;
  onFix: (id: string) => void;
  isFixing: boolean;
}) {
  const navigate = useNavigate();
  const config = categoryConfig[issue.category] || categoryConfig.cost_waste;
  const [expanded, setExpanded] = useState(false);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, x: -100 }}
      className={clsx(
        'stellar-glass rounded-xl border overflow-hidden transition-all',
        config.borderColor
      )}
    >
      {/* Main Row */}
      <div
        className="p-4 cursor-pointer hover:bg-slate-800/30 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-start gap-4">
          {/* Category Icon */}
          <div className={clsx('p-2 rounded-lg', config.bgColor)}>
            <span className={config.color}>{config.icon}</span>
          </div>

          {/* Issue Info */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <span className={clsx('text-xs font-medium px-2 py-0.5 rounded-full', config.bgColor, config.color)}>
                {config.label}
              </span>
              {issue.remediation.automated && (
                <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400">
                  Auto-fix
                </span>
              )}
            </div>
            <h3 className="text-white font-medium mb-1">{issue.title}</h3>
            <p className="text-slate-400 text-sm">{issue.one_liner}</p>
          </div>

          {/* Cost & Action */}
          <div className="text-right flex-shrink-0">
            <div className="text-2xl font-bold text-white">
              ${issue.impact.hourly_cost < 1 ? issue.impact.hourly_cost.toFixed(2) : Math.round(issue.impact.hourly_cost)}
              <span className="text-sm text-slate-400">/hr</span>
            </div>
            <div className="text-xs text-slate-500 mb-2">
              {issue.impact.affected_devices} devices
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onFix(issue.id);
              }}
              disabled={isFixing}
              className={clsx(
                'px-4 py-2 rounded-lg font-medium text-sm transition-all',
                issue.remediation.automated
                  ? 'bg-emerald-500 hover:bg-emerald-400 text-white'
                  : 'bg-slate-700 hover:bg-slate-600 text-slate-300',
                isFixing && 'opacity-50 cursor-not-allowed'
              )}
            >
              {isFixing ? (
                <span className="flex items-center gap-2">
                  <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Fixing...
                </span>
              ) : issue.remediation.automated ? (
                'Fix Now'
              ) : (
                'Create Ticket'
              )}
            </button>
          </div>

          {/* Expand Icon */}
          <motion.div
            animate={{ rotate: expanded ? 180 : 0 }}
            className="text-slate-500"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </motion.div>
        </div>
      </div>

      {/* Expanded Details with Navigation */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="border-t border-slate-700/50"
          >
            <div className="p-4 bg-slate-800/20">
              <div className="grid grid-cols-2 gap-6 mb-4">
                <div>
                  <h4 className="text-xs text-slate-500 uppercase tracking-wider mb-2">Root Cause</h4>
                  <p className="text-slate-300">{issue.root_cause}</p>
                </div>
                <div>
                  <h4 className="text-xs text-slate-500 uppercase tracking-wider mb-2">Remediation</h4>
                  <p className="text-slate-300">{issue.remediation.description}</p>
                  <div className="flex items-center gap-4 mt-2 text-xs text-slate-500">
                    <span>~{issue.remediation.estimated_minutes} min</span>
                    <span>{Math.round(issue.remediation.success_rate * 100)}% success rate</span>
                  </div>
                </div>
              </div>

              {/* Navigation Links */}
              <div className="flex items-center gap-3 pt-3 border-t border-slate-700/30">
                <span className="text-xs text-slate-500">Explore:</span>

                {issue.related_dashboard && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      navigate(issue.related_dashboard!);
                    }}
                    className="text-xs px-3 py-1.5 rounded-lg bg-slate-700/50 text-slate-300 hover:bg-slate-600/50 hover:text-white transition-colors flex items-center gap-1"
                  >
                    <span>{config.dashboardLabel}</span>
                    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                    </svg>
                  </button>
                )}

                {issue.device_ids.length > 0 && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      navigate(`/devices/${issue.device_ids[0]}`);
                    }}
                    className="text-xs px-3 py-1.5 rounded-lg bg-slate-700/50 text-slate-300 hover:bg-slate-600/50 hover:text-white transition-colors flex items-center gap-1"
                  >
                    <span>View Device</span>
                    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                    </svg>
                  </button>
                )}

                {issue.investigation_id && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      navigate(`/investigations/${issue.investigation_id}`);
                    }}
                    className="text-xs px-3 py-1.5 rounded-lg bg-amber-500/20 text-amber-400 hover:bg-amber-500/30 transition-colors flex items-center gap-1"
                  >
                    <span>Investigation</span>
                    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                  </button>
                )}

                {issue.device_ids.length > 1 && (
                  <span className="text-xs text-slate-500 ml-auto">
                    +{issue.device_ids.length - 1} more devices affected
                  </span>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export default function ActionCenter() {
  const [categoryFilter, setCategoryFilter] = useState<string | null>(null);
  const [fixingIds, setFixingIds] = useState<Set<string>>(new Set());
  const queryClient = useQueryClient();

  // Data queries - now using real API (backend handles mock mode via X-Mock-Mode header)
  const { data: summary, isLoading: summaryLoading } = useQuery({
    queryKey: ['action-center', 'summary'],
    queryFn: () => api.getActionCenterSummary(),
    refetchInterval: 30000,
  });

  // Fleet pulse - get from dashboard stats for real data integration
  const { data: dashboardStats } = useQuery({
    queryKey: ['dashboard', 'stats'],
    queryFn: () => api.getDashboardStats(),
    refetchInterval: 30000,
  });

  // Derive fleet pulse from real dashboard stats when available
  const fleetPulse: FleetPulseMetrics | undefined = dashboardStats
    ? {
        total_devices: dashboardStats.devices_monitored,
        healthy_devices: dashboardStats.devices_monitored - (dashboardStats.open_cases || 0),
        devices_with_issues: dashboardStats.open_cases || 0,
        security_score: null, // Not yet available from API
        network_score: null, // Not yet available from API
        open_investigations: dashboardStats.open_cases || 0,
        anomalies_today: dashboardStats.anomalies_today,
        avg_battery_health: null, // Not yet available from API
      }
    : undefined;

  const { data: issueList, isLoading: issuesLoading } = useQuery({
    queryKey: ['action-center', 'issues', categoryFilter],
    queryFn: () => api.getActionCenterIssues({
      category: categoryFilter || undefined,
    }),
    refetchInterval: 30000,
  });

  // Fix mutation - using real API
  const fixMutation = useMutation({
    mutationFn: async (issueId: string) => {
      setFixingIds((prev) => new Set(prev).add(issueId));
      return api.fixIssue(issueId);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['action-center'] });
    },
    onSettled: (_, __, issueId) => {
      setFixingIds((prev) => {
        const next = new Set(prev);
        next.delete(issueId);
        return next;
      });
    },
  });

  const fixAllMutation = useMutation({
    mutationFn: () => api.fixAllAutomated(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['action-center'] });
    },
  });

  const isLoading = summaryLoading || issuesLoading;

  if (isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-amber-500/30 border-t-amber-500 rounded-full animate-spin mx-auto mb-4" />
          <p className="text-slate-400">Scanning fleet for issues...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 p-6 overflow-y-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white">Action Center</h1>
          <p className="text-slate-400">Problems ranked by business impact. Fix them, or explore deeper.</p>
        </div>

        <div className="flex items-center gap-3">
          <Link
            to="/dashboard"
            className="px-4 py-2 rounded-lg text-sm font-medium bg-slate-800/50 text-slate-300 hover:bg-slate-700/50 hover:text-white transition-colors flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
              />
            </svg>
            <span>Command Center</span>
          </Link>

          {summary && summary.automatable_count > 0 && (
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => fixAllMutation.mutate()}
              disabled={fixAllMutation.isPending}
              className={clsx(
                'flex items-center gap-3 px-6 py-2.5 rounded-xl font-semibold',
                'bg-gradient-to-r from-emerald-500 to-emerald-600',
                'hover:from-emerald-400 hover:to-emerald-500',
                'text-white shadow-lg shadow-emerald-500/25',
                'transition-all',
                fixAllMutation.isPending && 'opacity-50 cursor-not-allowed'
              )}
            >
              {fixAllMutation.isPending ? (
                <>
                  <svg className="animate-spin w-5 h-5" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  <span>Fixing All...</span>
                </>
              ) : (
                <>
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  <span>Fix All ({summary.automatable_count})</span>
                </>
              )}
            </motion.button>
          )}
        </div>
      </div>

      {/* Fleet Pulse - Quick Health Overview */}
      {fleetPulse && <FleetPulse metrics={fleetPulse} />}

      {/* Executive Summary */}
      {summary && <ExecutiveSummary summary={summary} />}

      {/* Category Cards with Navigation */}
      {summary && (
        <CategoryCards
          summary={summary}
          onCategoryClick={(cat) => setCategoryFilter(categoryFilter === cat ? null : cat)}
        />
      )}

      {/* Filter Bar */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="text-sm text-slate-500">Showing:</span>
          <button
            onClick={() => setCategoryFilter(null)}
            className={clsx(
              'px-3 py-1.5 rounded-lg text-sm font-medium transition-colors',
              categoryFilter === null
                ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                : 'bg-slate-800/50 text-slate-400 hover:text-white'
            )}
          >
            All Issues ({issueList?.total_count || 0})
          </button>
          {categoryFilter && (
            <span className={clsx(
              'px-3 py-1.5 rounded-lg text-sm font-medium',
              categoryConfig[categoryFilter]?.bgColor,
              categoryConfig[categoryFilter]?.color,
              'border',
              categoryConfig[categoryFilter]?.borderColor
            )}>
              {categoryConfig[categoryFilter]?.label}
              <button
                onClick={() => setCategoryFilter(null)}
                className="ml-2 hover:text-white"
              >
                ×
              </button>
            </span>
          )}
        </div>

        <Link
          to="/investigations"
          className="text-sm text-slate-400 hover:text-amber-400 transition-colors flex items-center gap-1"
        >
          <span>View all investigations</span>
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
          </svg>
        </Link>
      </div>

      {/* Issues List */}
      <div className="space-y-4">
        <AnimatePresence mode="popLayout">
          {issueList?.issues.map((issue) => (
            <IssueCard
              key={issue.id}
              issue={issue}
              onFix={(id) => fixMutation.mutate(id)}
              isFixing={fixingIds.has(issue.id)}
            />
          ))}
        </AnimatePresence>

        {issueList?.issues.length === 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-16"
          >
            <div className="w-16 h-16 rounded-full bg-emerald-500/20 flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">All Clear!</h3>
            <p className="text-slate-400 mb-4">No issues detected. Your fleet is operating optimally.</p>
            <div className="flex items-center justify-center gap-4">
              <Link
                to="/dashboard"
                className="text-sm text-amber-400 hover:text-amber-300 transition-colors"
              >
                View Command Center →
              </Link>
              <Link
                to="/insights"
                className="text-sm text-slate-400 hover:text-white transition-colors"
              >
                AI Insights →
              </Link>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
