/**
 * Command Center Dashboard - Unified View
 *
 * Single-page mission control with all key information visible at once:
 * KPIs, priority queue, AI insights, trends, and system telemetry
 */

import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate, Link } from 'react-router-dom';
import { api } from '../api/client';
import { format } from 'date-fns';
import { KPICard } from '../components/KPICard';
import { motion } from 'framer-motion';
import { AIInsightsPanel } from '../components/AIInsightsPanel';
import { InfoTooltip } from '../components/ui/InfoTooltip';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer
} from 'recharts';
import type { AIInsight, Anomaly, ConnectionStatus } from '../types/anomaly';

function Dashboard() {
  const navigate = useNavigate();

  // Data queries
  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['dashboard', 'stats'],
    queryFn: () => api.getDashboardStats(),
    refetchInterval: 15000,
  });

  const { data: trends } = useQuery({
    queryKey: ['dashboard', 'trends', 7],
    queryFn: () => api.getDashboardTrends({ days: 7 }),
    refetchInterval: 60000,
  });

  const { data: anomalies } = useQuery({
    queryKey: ['anomalies', 'recent'],
    queryFn: () => api.getAnomalies({ page: 1, page_size: 10, status: 'open' }),
    refetchInterval: 15000,
  });

  const { data: connections } = useQuery({
    queryKey: ['dashboard', 'connections'],
    queryFn: () => api.getConnectionStatus(),
    refetchInterval: 30000,
  });

  // Calculate system health
  const systemHealth = useMemo(() => {
    if (!connections) return { percentage: 0, status: 'offline' as const, connected: 0 };
    const systems = [
      connections.dw_sql,
      connections.mc_sql,
      connections.mobicontrol_api,
      connections.llm,
    ];
    const connectedCount = systems.filter((s) => s.connected).length;
    const percentage = Math.round((connectedCount / systems.length) * 100);
    const status = percentage === 100 ? 'healthy' : percentage >= 50 ? 'degraded' : 'offline';
    return { percentage, status, connected: connectedCount };
  }, [connections]);

  // Calculate metrics
  const openAnomalies = stats?.open_cases ?? 0;
  const resolutionRate = (stats?.anomalies_today && stats.anomalies_today > 0)
    ? Math.round(((stats.resolved_today ?? 0) / stats.anomalies_today) * 100)
    : 0;

  // Generate AI insights from anomaly data
  const aiInsights: AIInsight[] = useMemo(() => {
    const insights: AIInsight[] = [];

    if (stats?.critical_issues && stats.critical_issues > 0) {
      insights.push({
        id: 'critical-alert',
        type: 'warning',
        severity: 'critical',
        title: `${stats.critical_issues} Critical Anomalies Detected`,
        description: `There are ${stats.critical_issues} critical anomalies requiring immediate attention.`,
        recommendation: 'Prioritize investigating critical anomalies immediately.',
        why: 'Critical anomalies indicate significant deviation from baseline behavior.',
        whatToDo: 'Navigate to Investigations and filter by critical severity.',
        impact: { deviceCount: stats.critical_issues, confidence: 95 },
        createdAt: new Date().toISOString(),
        status: 'pending',
        actionable: true,
        actionLabel: 'View Critical Cases',
      });
    }

    if (openAnomalies > 5) {
      insights.push({
        id: 'backlog-warning',
        type: 'efficiency',
        severity: 'medium',
        title: 'Investigation Backlog Growing',
        description: `${openAnomalies} open cases are pending investigation.`,
        recommendation: 'Review and triage open cases.',
        impact: { confidence: 80 },
        createdAt: new Date().toISOString(),
        status: 'pending',
      });
    }

    if (resolutionRate >= 80 && stats?.resolved_today && stats.resolved_today > 0) {
      insights.push({
        id: 'resolution-good',
        type: 'pattern',
        severity: 'info',
        title: 'Strong Resolution Performance',
        description: `Today's ${resolutionRate}% resolution rate indicates efficient case handling.`,
        impact: { metric: 'Resolution Rate', value: `${resolutionRate}%`, confidence: 90 },
        createdAt: new Date().toISOString(),
        status: 'pending',
      });
    }

    if (systemHealth.status === 'degraded') {
      insights.push({
        id: 'system-degraded',
        type: 'warning',
        severity: 'high',
        title: 'System Health Degraded',
        description: `Only ${systemHealth.connected}/4 services are online.`,
        recommendation: 'Check System settings and restore offline connections.',
        whatToDo: 'Navigate to System page and check connection status.',
        createdAt: new Date().toISOString(),
        status: 'pending',
        actionable: true,
        actionLabel: 'Check Systems',
      });
    }

    return insights;
  }, [stats, openAnomalies, resolutionRate, systemHealth]);

  // Loading state
  if (statsLoading) {
    return (
      <div className="flex items-center justify-center h-96" role="status" aria-label="Loading dashboard">
        <motion.div
          className="text-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <div className="relative w-20 h-20 mx-auto mb-6">
            <motion.div
              className="absolute inset-0 rounded-full border-2 border-amber-500/30"
              animate={{ scale: [1, 1.2, 1], opacity: [0.3, 0.1, 0.3] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
            <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-amber-500 animate-spin" />
            <div className="absolute inset-2 rounded-full border-2 border-transparent border-t-indigo-500 animate-spin" style={{ animationDirection: 'reverse', animationDuration: '1.5s' }} />
          </div>
          <p className="text-slate-400 font-mono text-sm">Initializing Command Center...</p>
        </motion.div>
      </div>
    );
  }

  return (
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4 }}
    >
      {/* Header Section */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <motion.h1
            className="text-3xl font-bold text-white"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            Command Center
          </motion.h1>
          <motion.p
            className="text-slate-500 mt-1"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            AI-Powered Fleet Intelligence • {format(new Date(), 'EEEE, MMMM d')}
          </motion.p>
        </div>

        {/* System Health Ring */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
        >
          <Link
            to="/system"
            className="relative group flex items-center gap-3 px-4 py-2 rounded-xl bg-slate-800/50 border border-slate-700/50 hover:border-slate-600/50 transition-colors"
            aria-label={`System health: ${systemHealth.percentage}%. Click to view system settings.`}
          >
            <svg className="w-10 h-10 -rotate-90" viewBox="0 0 36 36" aria-hidden="true">
              <circle cx="18" cy="18" r="15.5" fill="none" stroke="currentColor" strokeWidth="2" className="text-slate-800" />
              <motion.circle
                cx="18" cy="18" r="15.5" fill="none" strokeWidth="2"
                strokeDasharray={`${systemHealth.percentage} 100`}
                strokeLinecap="round"
                className={
                  systemHealth.status === 'healthy' ? 'text-emerald-400'
                    : systemHealth.status === 'degraded' ? 'text-orange-400'
                      : 'text-red-400'
                }
                initial={{ strokeDasharray: '0 100' }}
                animate={{ strokeDasharray: `${systemHealth.percentage} 100` }}
                transition={{ duration: 1, ease: 'easeOut' }}
              />
            </svg>
            <div>
              <p className="text-sm font-semibold text-white">{systemHealth.percentage}% Online</p>
              <p className="text-xs text-slate-500">{systemHealth.connected}/4 Services</p>
            </div>
          </Link>
        </motion.div>
      </div>

      {/* KPI Cards Row */}
      <motion.div
        className="grid grid-cols-2 gap-3 xl:grid-cols-4"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15 }}
      >
        <KPICard
          title="Detected Today"
          value={stats?.anomalies_today || 0}
          change={anomalies?.anomalies.length ? `+${anomalies.anomalies.length}` : undefined}
          trend={anomalies?.anomalies.length ? 'up' : 'neutral'}
          color="stellar"
          onClick={() => navigate('/investigations')}
          explainer="Total anomalies detected by the AI engine in the last 24 hours"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          }
        />
        <KPICard
          title="Open Cases"
          value={openAnomalies}
          trend={openAnomalies > 5 ? 'up' : 'neutral'}
          color={openAnomalies > 5 ? 'warning' : 'aurora'}
          onClick={() => navigate('/investigations?status=open')}
          isActive={openAnomalies > 0}
          explainer="Cases currently pending investigation and resolution"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
            </svg>
          }
        />
        <KPICard
          title="Critical"
          value={stats?.critical_issues || 0}
          trend={stats?.critical_issues ? 'up' : 'neutral'}
          color={stats?.critical_issues ? 'danger' : 'aurora'}
          onClick={() => navigate('/investigations?severity=critical')}
          isActive={(stats?.critical_issues || 0) > 0}
          explainer="High-severity anomalies requiring immediate attention"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          }
        />
        <KPICard
          title="Resolved Today"
          value={stats?.resolved_today || 0}
          change={`${resolutionRate}%`}
          trend="up"
          color="aurora"
          onClick={() => navigate('/investigations?status=resolved')}
          explainer="Cases investigated and closed today with resolution rate"
          context="resolution rate"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          }
        />
      </motion.div>

      {/* Main Content - Responsive Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
        {/* Priority Queue */}
        <motion.div
          className="stellar-card rounded-xl p-4 xl:col-span-2"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-amber-500/20 flex items-center justify-center">
                <svg className="w-4 h-4 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
              </div>
              <h2 className="text-sm font-semibold text-white">Priority Queue</h2>
              <span className="text-xs text-slate-500 hidden sm:inline">Open cases requiring attention</span>
            </div>
            <Link to="/investigations" className="text-xs text-amber-400 hover:text-amber-300 transition-colors">
              View All →
            </Link>
          </div>

          {anomalies?.anomalies && anomalies.anomalies.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {anomalies.anomalies.slice(0, 6).map((anomaly) => (
                <PriorityQueueItem key={anomaly.id} anomaly={anomaly} />
              ))}
            </div>
          ) : (
            <div className="text-center py-6">
              <div className="w-10 h-10 mx-auto mb-2 rounded-xl bg-emerald-500/10 flex items-center justify-center">
                <svg className="w-5 h-5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <p className="text-sm text-slate-400">All clear! No open cases.</p>
            </div>
          )}
        </motion.div>

        {/* AI Insights Panel */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25 }}
          className="row-span-2"
        >
          {aiInsights.length > 0 ? (
            <AIInsightsPanel
              title="AI Insights"
              insights={aiInsights}
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
          ) : (
            <div className="stellar-card rounded-xl p-4 h-full">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-8 h-8 rounded-lg bg-emerald-500/20 flex items-center justify-center">
                  <svg className="w-4 h-4 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
                <h2 className="text-sm font-semibold text-white">AI Insights</h2>
              </div>
              <div className="text-center py-4">
                <p className="text-sm text-emerald-400 font-medium">All Systems Normal</p>
                <p className="text-xs text-slate-500 mt-1">No issues detected</p>
              </div>
            </div>
          )}
        </motion.div>

        {/* 7-Day Trend */}
        <motion.div
          className="stellar-card rounded-xl p-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-indigo-500/20 flex items-center justify-center">
                <svg className="w-4 h-4 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <h2 className="text-sm font-semibold text-white">7-Day Trend</h2>
            </div>
            <Link to="/locations" className="text-xs text-indigo-400 hover:text-indigo-300 transition-colors">
              Locations →
            </Link>
          </div>

          {trends && trends.length > 0 ? (
            <div className="h-36">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={trends}>
                  <defs>
                    <linearGradient id="trendGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#6366f1" stopOpacity={0.3} />
                      <stop offset="100%" stopColor="#6366f1" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis
                    dataKey="date"
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: '#64748b', fontSize: 10 }}
                    tickFormatter={(value) => format(new Date(value), 'EEE')}
                  />
                  <YAxis
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: '#64748b', fontSize: 10 }}
                    width={24}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1e293b',
                      border: '1px solid #334155',
                      borderRadius: '8px',
                      color: '#f8fafc',
                      fontSize: '12px',
                    }}
                    labelFormatter={(value) => format(new Date(value), 'MMM d, yyyy')}
                  />
                  <Area
                    type="monotone"
                    dataKey="anomaly_count"
                    stroke="#6366f1"
                    strokeWidth={2}
                    fill="url(#trendGradient)"
                    name="Anomalies"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-36 flex items-center justify-center text-slate-500 text-sm">
              No trend data available
            </div>
          )}
        </motion.div>

        {/* System Status */}
        <motion.div
          className="stellar-card rounded-xl p-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35 }}
        >
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-violet-500/20 flex items-center justify-center">
                <svg className="w-4 h-4 text-violet-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2" />
                </svg>
              </div>
              <h2 className="text-sm font-semibold text-white">System Status</h2>
            </div>
            <Link to="/system" className="text-xs text-violet-400 hover:text-violet-300 transition-colors">
              Manage →
            </Link>
          </div>

          <div className="space-y-2">
            <ConnectionStatusRow name="XSight SQL" connection={connections?.dw_sql} />
            <ConnectionStatusRow name="MobiControl SQL" connection={connections?.mc_sql} />
            <ConnectionStatusRow name="MobiControl API" connection={connections?.mobicontrol_api} />
            <ConnectionStatusRow name="LLM Service" connection={connections?.llm} />
          </div>
        </motion.div>

        {/* Performance Stats */}
        <motion.div
          className="stellar-card rounded-xl p-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <div className="flex items-center gap-2 mb-3">
            <div className="w-8 h-8 rounded-lg bg-cyan-500/20 flex items-center justify-center">
              <svg className="w-4 h-4 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <h2 className="text-sm font-semibold text-white">Performance</h2>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <AnalysisStat
              label="Resolution Rate"
              value={`${resolutionRate}%`}
              status={resolutionRate >= 80 ? 'good' : resolutionRate >= 50 ? 'warning' : 'critical'}
              tooltip="Percentage of anomalies resolved within SLA"
            />
            <AnalysisStat
              label="Insights"
              value={aiInsights.length}
              status="neutral"
              tooltip="AI-generated insights from pattern analysis"
            />
            <AnalysisStat
              label="Actionable"
              value={aiInsights.filter(i => i.actionable).length}
              status={aiInsights.filter(i => i.actionable).length > 0 ? 'warning' : 'neutral'}
              tooltip="Insights requiring immediate action"
            />
            <AnalysisStat
              label="Devices"
              value={stats?.devices_monitored || 0}
              status="neutral"
              tooltip="Total devices being monitored"
            />
          </div>
        </motion.div>
      </div>
    </motion.div>
  );
}

// Priority Queue Item Component
function PriorityQueueItem({ anomaly }: { anomaly: Anomaly }) {
  const navigate = useNavigate();
  const severity = anomaly.anomaly_score < -0.7 ? 'critical' : anomaly.anomaly_score < -0.5 ? 'high' : 'medium';
  const severityColors = {
    critical: 'bg-red-500/20 text-red-400 border-red-500/30',
    high: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
    medium: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  };

  return (
    <button
      onClick={() => navigate(`/investigations/${anomaly.id}`)}
      className="w-full flex items-center gap-3 p-2 rounded-lg bg-slate-800/30 hover:bg-slate-800/50 border border-slate-700/30 hover:border-slate-600/50 transition-all text-left"
    >
      <div className={`px-2 py-0.5 rounded text-[10px] font-semibold uppercase border ${severityColors[severity]}`}>
        {severity}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-xs text-white truncate">Device #{anomaly.device_id}</p>
        <p className="text-[10px] text-slate-500">{format(new Date(anomaly.timestamp), 'MMM d, h:mm a')}</p>
      </div>
      <div className="text-xs font-mono text-slate-400">
        {anomaly.anomaly_score.toFixed(3)}
      </div>
    </button>
  );
}

// Connection Status Row Component
function ConnectionStatusRow({ name, connection }: { name: string; connection?: ConnectionStatus }) {
  const isConnected = connection?.connected ?? false;

  return (
    <div className="flex items-center justify-between py-1.5 px-2 rounded-lg bg-slate-800/30">
      <span className="text-xs text-slate-400">{name}</span>
      <div className="flex items-center gap-2">
        <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-emerald-400' : 'bg-red-400'}`} />
        <span className={`text-xs ${isConnected ? 'text-emerald-400' : 'text-red-400'}`}>
          {isConnected ? 'Online' : 'Offline'}
        </span>
      </div>
    </div>
  );
}

// Analysis Stat Component
interface AnalysisStatProps {
  label: string;
  value: string | number;
  status: 'good' | 'warning' | 'critical' | 'neutral';
  tooltip: string;
}

function AnalysisStat({ label, value, status, tooltip }: AnalysisStatProps) {
  const colorClasses = {
    good: 'text-emerald-400',
    warning: 'text-amber-400',
    critical: 'text-red-400',
    neutral: 'text-slate-300',
  };

  return (
    <div className="p-2 rounded-lg bg-slate-800/30">
      <div className="flex items-center gap-1 mb-1">
        <p className="text-[10px] text-slate-500">{label}</p>
        <InfoTooltip content={tooltip} />
      </div>
      <p className={`text-lg font-bold font-mono ${colorClasses[status]}`}>{value}</p>
    </div>
  );
}

export default Dashboard;
