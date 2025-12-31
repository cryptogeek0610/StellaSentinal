/**
 * Dashboard Analysis View
 *
 * AI insights panel and analysis summary for data-driven decisions.
 * Balanced layout for both insights-rich and empty states.
 */

import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { AIInsightsPanel } from '../../components/AIInsightsPanel';
import { InfoTooltip } from '../../components/ui/InfoTooltip';
import type { AIInsight } from '../../types/anomaly';

interface DashboardAnalysisProps {
  insights: AIInsight[];
  resolutionRate: number;
}

export function DashboardAnalysis({ insights, resolutionRate }: DashboardAnalysisProps) {
  const navigate = useNavigate();
  const hasInsights = insights.length > 0;

  return (
    <motion.div
      key="analysis"
      className="space-y-6"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
    >
      {/* Top Stats Row */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <AnalysisStatCard
          label="Resolution Rate"
          value={`${resolutionRate}%`}
          status={resolutionRate >= 80 ? 'good' : resolutionRate >= 50 ? 'warning' : 'critical'}
          explainer="Percentage of anomalies successfully investigated and resolved within SLA targets"
          icon={
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          }
        />
        <AnalysisStatCard
          label="Total Insights"
          value={insights.length}
          status="neutral"
          explainer="Number of AI-generated insights from pattern analysis and anomaly correlation"
          icon={
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          }
        />
        <AnalysisStatCard
          label="Actionable"
          value={insights.filter(i => i.actionable).length}
          status={insights.filter(i => i.actionable).length > 0 ? 'warning' : 'neutral'}
          explainer="Insights requiring immediate action or configuration changes to prevent future issues"
          icon={
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" />
            </svg>
          }
        />
        <AnalysisStatCard
          label="Critical"
          value={insights.filter(i => i.severity === 'critical').length}
          status={insights.filter(i => i.severity === 'critical').length > 0 ? 'critical' : 'neutral'}
          explainer="High-severity insights indicating widespread issues or significant performance degradation"
          icon={
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          }
        />
      </div>

      {/* Main Content */}
      {hasInsights ? (
        <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
          {/* AI Insights Panel */}
          <div className="xl:col-span-2">
            <AIInsightsPanel
              title="AI Analysis"
              insights={insights}
              onInsightAction={(insight, action) => {
                if (action === 'apply') {
                  if (insight.id === 'critical-alert') {
                    navigate('/investigations?severity=critical');
                  } else if (insight.id === 'system-degraded') {
                    navigate('/system');
                  }
                }
              }}
            />
          </div>

          {/* Analysis Summary */}
          <div className="space-y-4">
            <PerformanceCard resolutionRate={resolutionRate} />
            <DeepAnalyticsCard />
          </div>
        </div>
      ) : (
        /* Empty State - No Insights */
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 xl:grid-cols-3">
          {/* All Clear Card */}
          <div className="md:col-span-2 xl:col-span-2">
            <div className="stellar-card rounded-xl p-8 text-center">
              <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-emerald-500/20 to-emerald-600/10 flex items-center justify-center border border-emerald-500/20">
                <svg className="w-8 h-8 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">No Active Insights</h3>
              <p className="text-slate-400 max-w-md mx-auto mb-6">
                The AI analysis engine hasn't detected any patterns or issues requiring attention.
                Your fleet is operating within normal parameters.
              </p>
              <div className="flex flex-wrap justify-center gap-3">
                <Link
                  to="/insights"
                  className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-indigo-400 bg-indigo-500/10 border border-indigo-500/20 rounded-lg hover:bg-indigo-500/20 transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                  View Analytics
                </Link>
                <Link
                  to="/investigations"
                  className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-slate-300 bg-slate-800/50 border border-slate-700/50 rounded-lg hover:bg-slate-700/50 transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                  </svg>
                  Browse Investigations
                </Link>
              </div>
            </div>
          </div>

          {/* Performance Card */}
          <PerformanceCard resolutionRate={resolutionRate} />
        </div>
      )}
    </motion.div>
  );
}

// Analysis Stat Card Component
interface AnalysisStatCardProps {
  label: string;
  value: string | number;
  status: 'good' | 'warning' | 'critical' | 'neutral';
  icon: React.ReactNode;
  explainer?: string;
}

function AnalysisStatCard({ label, value, status, icon, explainer }: AnalysisStatCardProps) {
  const colorClasses = {
    good: {
      border: 'border-emerald-500/20',
      bg: 'bg-emerald-500/5',
      icon: 'bg-emerald-500/20 text-emerald-400',
      value: 'text-emerald-400',
    },
    warning: {
      border: 'border-amber-500/20',
      bg: 'bg-amber-500/5',
      icon: 'bg-amber-500/20 text-amber-400',
      value: 'text-amber-400',
    },
    critical: {
      border: 'border-red-500/20',
      bg: 'bg-red-500/5',
      icon: 'bg-red-500/20 text-red-400',
      value: 'text-red-400',
    },
    neutral: {
      border: 'border-slate-700/50',
      bg: 'bg-slate-800/30',
      icon: 'bg-indigo-500/20 text-indigo-400',
      value: 'text-slate-300',
    },
  };

  const colors = colorClasses[status];

  return (
    <div className={`stellar-card rounded-xl p-4 border ${colors.border} ${colors.bg}`}>
      <div className="flex items-start gap-3">
        <div className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${colors.icon}`} aria-hidden="true">
          {icon}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5 mb-1">
            <p className="text-xs text-slate-500">{label}</p>
            {explainer && <InfoTooltip content={explainer} />}
          </div>
          <p className={`text-xl font-bold font-mono ${colors.value}`}>{value}</p>
        </div>
      </div>
    </div>
  );
}

// Performance Card Component
function PerformanceCard({ resolutionRate }: { resolutionRate: number }) {
  const status = resolutionRate >= 80 ? 'good' : resolutionRate >= 50 ? 'warning' : 'critical';
  const statusColor = {
    good: 'text-emerald-400',
    warning: 'text-amber-400',
    critical: 'text-red-400',
  };
  const barColor = {
    good: 'bg-emerald-500',
    warning: 'bg-amber-500',
    critical: 'bg-red-500',
  };

  return (
    <div className="stellar-card rounded-xl p-5">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-600 to-purple-600 flex items-center justify-center" aria-hidden="true">
          <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
        </div>
        <div>
          <p className="text-sm font-semibold text-white">Performance Score</p>
          <p className="text-xs text-slate-500">Overall fleet health</p>
        </div>
      </div>
      <div className="text-center py-3">
        <p
          className={`text-4xl font-bold font-mono ${statusColor[status]}`}
          role="status"
          aria-label={`Resolution rate: ${resolutionRate} percent`}
        >
          {resolutionRate}%
        </p>
        <p className="text-xs text-slate-500 mt-1">Resolution Rate</p>
      </div>

      {/* Visual progress indicator */}
      <div className="mt-3">
        <div className="h-2 bg-slate-800 rounded-full overflow-hidden" role="progressbar" aria-valuenow={resolutionRate} aria-valuemin={0} aria-valuemax={100}>
          <motion.div
            className={`h-full rounded-full ${barColor[status]}`}
            initial={{ width: 0 }}
            animate={{ width: `${resolutionRate}%` }}
            transition={{ duration: 1, ease: 'easeOut' }}
          />
        </div>
      </div>
    </div>
  );
}

// Deep Analytics Card Component
function DeepAnalyticsCard() {
  return (
    <Link
      to="/insights"
      className="block stellar-card rounded-xl p-5 hover:border-amber-500/30 transition-colors group"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-amber-500/20 flex items-center justify-center" aria-hidden="true">
            <svg className="w-5 h-5 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
            </svg>
          </div>
          <div>
            <p className="text-sm font-semibold text-white">Deep Analytics</p>
            <p className="text-xs text-slate-500">View detailed insights</p>
          </div>
        </div>
        <svg
          className="w-5 h-5 text-amber-400 transform group-hover:translate-x-1 transition-transform"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          aria-hidden="true"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
      </div>
    </Link>
  );
}
