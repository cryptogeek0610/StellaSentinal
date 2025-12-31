/**
 * Command Center - Enterprise NOC-Style Anomaly Detection Dashboard
 *
 * A professional mission control interface following industry NOC/SOC best practices:
 * - Real-time operational awareness with live event stream
 * - SLA/MTTR metrics for operational excellence tracking
 * - ML model health and performance visualization
 * - Geographic fleet distribution with anomaly hotspots
 * - Tiered alert prioritization and incident management
 * - AI-powered pattern recognition and predictive insights
 */

import { useMemo, useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate, Link } from 'react-router-dom';
import { api } from '../api/client';
import { format, formatDistanceToNowStrict, differenceInMinutes } from 'date-fns';
import { motion, AnimatePresence } from 'framer-motion';
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts';
import { Card } from '../components/Card';
import { useLocationAttribute } from '../hooks/useLocationAttribute';
import { getSeverityFromScore } from '../utils/severity';
import type { AIInsight, Anomaly } from '../types/anomaly';

// ============================================================================
// Main Dashboard Component
// ============================================================================

function Dashboard() {
  const navigate = useNavigate();
  const [attributeName] = useLocationAttribute();
  const [currentTime, setCurrentTime] = useState(new Date());

  // Update clock every minute
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 60000);
    return () => clearInterval(timer);
  }, []);

  // ---- Data Queries ----
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
    queryFn: () => api.getAnomalies({ page: 1, page_size: 20, status: 'open' }),
    refetchInterval: 15000,
  });

  const { data: recentResolved } = useQuery({
    queryKey: ['anomalies', 'recent-resolved'],
    queryFn: () => api.getAnomalies({ page: 1, page_size: 10, status: 'resolved' }),
    refetchInterval: 30000,
  });

  const { data: connections } = useQuery({
    queryKey: ['dashboard', 'connections'],
    queryFn: () => api.getConnectionStatus(),
    refetchInterval: 30000,
  });

  const { data: locationHeatmap } = useQuery({
    queryKey: ['dashboard', 'location-heatmap', attributeName],
    queryFn: () => api.getLocationHeatmap(attributeName),
    refetchInterval: 60000,
  });

  const { data: trainingStatus } = useQuery({
    queryKey: ['training', 'status'],
    queryFn: () => api.getTrainingStatus(),
    refetchInterval: 10000,
  });

  const { data: automationStatus } = useQuery({
    queryKey: ['automation', 'status'],
    queryFn: () => api.getAutomationStatus(),
    refetchInterval: 15000,
  });

  const { data: ifStats } = useQuery({
    queryKey: ['isolation-forest', 'stats'],
    queryFn: () => api.getIsolationForestStats(7),
    refetchInterval: 60000,
  });

  // ---- Computed Values ----
  const systemHealth = useMemo(() => {
    if (!connections) return { percentage: 0, status: 'offline' as const, connected: 0, total: 7 };
    const systems = [
      connections.backend_db,
      connections.redis,
      connections.qdrant,
      connections.dw_sql,
      connections.mc_sql,
      connections.mobicontrol_api,
      connections.llm,
    ];
    const connectedCount = systems.filter((s) => s?.connected).length;
    const percentage = Math.round((connectedCount / systems.length) * 100);
    const status = percentage === 100 ? 'healthy' : percentage >= 50 ? 'degraded' : 'offline';
    return { percentage, status, connected: connectedCount, total: systems.length };
  }, [connections]);

  const openAnomalies = stats?.open_cases ?? 0;
  const criticalCount = stats?.critical_issues ?? 0;
  const resolutionRate = (stats?.anomalies_today && stats.anomalies_today > 0)
    ? Math.round(((stats.resolved_today ?? 0) / stats.anomalies_today) * 100)
    : 0;

  // Calculate MTTR (Mean Time To Resolution) from resolved anomalies
  const mttrMinutes = useMemo(() => {
    if (!recentResolved?.anomalies?.length) return null;
    const resolvedAnomalies = recentResolved.anomalies.filter(a => a.status === 'resolved');
    if (resolvedAnomalies.length === 0) return null;
    // Estimate ~30 min average resolution time based on updated_at vs timestamp
    const times = resolvedAnomalies
      .filter(a => a.updated_at && a.timestamp)
      .map(a => differenceInMinutes(new Date(a.updated_at), new Date(a.timestamp)));
    if (times.length === 0) return null;
    return Math.round(times.reduce((a, b) => a + b, 0) / times.length);
  }, [recentResolved]);

  // Anomaly trend direction (comparing today vs yesterday)
  const trendDirection = useMemo(() => {
    if (!trends || trends.length < 2) return null;
    const today = trends[trends.length - 1]?.anomaly_count ?? 0;
    const yesterday = trends[trends.length - 2]?.anomaly_count ?? 0;
    if (today > yesterday) return 'up';
    if (today < yesterday) return 'down';
    return 'stable';
  }, [trends]);

  // AI Insights generation
  const aiInsights: AIInsight[] = useMemo(() => {
    const insights: AIInsight[] = [];

    if (criticalCount > 0) {
      insights.push({
        id: 'critical-alert',
        type: 'warning',
        severity: 'critical',
        title: `${criticalCount} Critical ${criticalCount === 1 ? 'Anomaly' : 'Anomalies'}`,
        description: `Devices showing significant deviation from baseline behavior.`,
        recommendation: 'Investigate critical cases immediately.',
        createdAt: new Date().toISOString(),
        status: 'pending',
        actionable: true,
        actionLabel: 'View Critical',
      });
    }

    if (openAnomalies > 10) {
      insights.push({
        id: 'backlog-warning',
        type: 'efficiency',
        severity: 'medium',
        title: 'Growing Investigation Backlog',
        description: `${openAnomalies} cases pending review.`,
        recommendation: 'Allocate time to resolve open cases.',
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
        description: `${systemHealth.connected}/${systemHealth.total} services online.`,
        recommendation: 'Check System settings for offline services.',
        createdAt: new Date().toISOString(),
        status: 'pending',
        actionable: true,
        actionLabel: 'Check Systems',
      });
    }

    if (trainingStatus?.status === 'running') {
      insights.push({
        id: 'training-active',
        type: 'pattern',
        severity: 'info',
        title: 'Model Training in Progress',
        description: `Training ${Math.round(trainingStatus.progress || 0)}% complete.`,
        createdAt: new Date().toISOString(),
        status: 'pending',
      });
    }

    if (trendDirection === 'up' && (stats?.anomalies_today ?? 0) > 5) {
      insights.push({
        id: 'trend-warning',
        type: 'pattern',
        severity: 'medium',
        title: 'Anomaly Rate Increasing',
        description: 'Detection rate is higher than yesterday.',
        recommendation: 'Monitor for potential fleet-wide issues.',
        createdAt: new Date().toISOString(),
        status: 'pending',
      });
    }

    return insights;
  }, [criticalCount, openAnomalies, systemHealth, trainingStatus, trendDirection, stats?.anomalies_today]);

  // Location data for heatmap
  const topLocations = useMemo(() => {
    if (!locationHeatmap?.locations) return [];
    return [...locationHeatmap.locations]
      .sort((a, b) => (b.anomalyCount ?? 0) - (a.anomalyCount ?? 0))
      .slice(0, 6);
  }, [locationHeatmap]);

  // Activity feed from anomalies
  const activityFeed = useMemo(() => {
    const events: { id: string; type: 'anomaly' | 'resolved' | 'training' | 'system'; message: string; timestamp: string; severity?: string }[] = [];

    // Add open anomalies
    anomalies?.anomalies?.slice(0, 5).forEach(a => {
      const severity = getSeverityFromScore(a.anomaly_score);
      events.push({
        id: `anomaly-${a.id}`,
        type: 'anomaly',
        message: `Device #${a.device_id} flagged`,
        timestamp: a.timestamp,
        severity: severity.label,
      });
    });

    // Add recent resolutions
    recentResolved?.anomalies?.slice(0, 3).forEach(a => {
      events.push({
        id: `resolved-${a.id}`,
        type: 'resolved',
        message: `Device #${a.device_id} resolved`,
        timestamp: a.timestamp,
      });
    });

    // Add training events
    if (trainingStatus?.status === 'running') {
      events.push({
        id: 'training',
        type: 'training',
        message: `Model training ${Math.round(trainingStatus.progress || 0)}%`,
        timestamp: new Date().toISOString(),
      });
    }

    // Sort by timestamp
    return events.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()).slice(0, 8);
  }, [anomalies, recentResolved, trainingStatus]);

  // Score distribution for visualization
  const scoreDistribution = useMemo(() => {
    if (!ifStats?.score_distribution?.bins) return [];
    return ifStats.score_distribution.bins.map((bin) => ({
      range: `${bin.bin_start.toFixed(2)}`,
      count: bin.count,
      isAnomaly: bin.is_anomaly,
    }));
  }, [ifStats]);

  // ---- Loading State ----
  if (statsLoading) {
    return (
      <div className="flex items-center justify-center h-96" role="status">
        <motion.div className="text-center" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
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
      className="space-y-4"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4 }}
    >
      {/* ================================================================== */}
      {/* HEADER BAR - NOC Style */}
      {/* ================================================================== */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between bg-slate-900/50 rounded-xl p-4 border border-slate-800/50">
        <div className="flex items-center gap-6">
          {/* Title + Timestamp */}
          <div>
            <motion.h1
              className="text-2xl font-bold text-white tracking-tight"
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              Command Center
            </motion.h1>
            <motion.div
              className="flex items-center gap-3 text-slate-500 text-sm mt-1"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.1 }}
            >
              <span className="font-mono">{format(currentTime, 'HH:mm')}</span>
              <span className="text-slate-700">|</span>
              <span>{format(currentTime, 'EEE, MMM d')}</span>
            </motion.div>
          </div>

          {/* Live Status Indicator */}
          <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20">
            <motion.div
              className="w-2 h-2 rounded-full bg-emerald-400"
              animate={{ opacity: [1, 0.4, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
            <span className="text-xs font-medium text-emerald-400">LIVE</span>
          </div>
        </div>

        {/* Right Side Controls */}
        <div className="flex items-center gap-4">
          {/* ML Pipeline Status */}
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-slate-800/50 border border-slate-700/50">
            <div className={`w-2 h-2 rounded-full ${
              trainingStatus?.status === 'running'
                ? 'bg-amber-400 animate-pulse'
                : automationStatus?.is_running
                  ? 'bg-emerald-400'
                  : 'bg-slate-500'
            }`} />
            <span className="text-xs text-slate-400">
              {trainingStatus?.status === 'running'
                ? `Training ${Math.round(trainingStatus.progress || 0)}%`
                : automationStatus?.is_running
                  ? 'ML Active'
                  : 'ML Idle'}
            </span>
          </div>

          {/* System Health Ring */}
          <Link to="/system" className="relative group" title="System Health">
            <svg className="w-12 h-12 -rotate-90" viewBox="0 0 36 36">
              <circle cx="18" cy="18" r="15.5" fill="none" stroke="currentColor" strokeWidth="2" className="text-slate-800" />
              <motion.circle
                cx="18" cy="18" r="15.5" fill="none" strokeWidth="2.5"
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
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-[10px] font-bold text-white font-mono">{systemHealth.connected}/{systemHealth.total}</span>
            </div>
          </Link>
        </div>
      </div>

      {/* ================================================================== */}
      {/* CRITICAL ALERT BANNER */}
      {/* ================================================================== */}
      <AnimatePresence>
        {criticalCount > 0 && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
          >
            <Link
              to="/investigations?severity=critical"
              className="flex items-center justify-between p-4 bg-red-500/10 border border-red-500/30 rounded-xl hover:bg-red-500/15 transition-colors group"
            >
              <div className="flex items-center gap-4">
                <motion.div
                  className="w-3 h-3 bg-red-500 rounded-full shadow-[0_0_12px_rgba(239,68,68,0.6)]"
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 1, repeat: Infinity }}
                />
                <div>
                  <p className="font-semibold text-red-400">
                    {criticalCount} Critical {criticalCount === 1 ? 'Alert' : 'Alerts'} Require Immediate Attention
                  </p>
                  <p className="text-xs text-slate-500">Devices showing severe behavioral deviation</p>
                </div>
              </div>
              <svg className="w-5 h-5 text-red-400 group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </Link>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ================================================================== */}
      {/* PRIMARY METRICS ROW - NOC Style Big Numbers */}
      {/* ================================================================== */}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-6">
        <MetricCard
          label="Open Incidents"
          value={openAnomalies}
          icon={<CaseIcon />}
          variant={openAnomalies > 10 ? 'warning' : openAnomalies > 0 ? 'alert' : 'normal'}
          onClick={() => navigate('/investigations?status=open')}
          pulse={openAnomalies > 10}
        />
        <MetricCard
          label="Critical"
          value={criticalCount}
          icon={<AlertIcon />}
          variant={criticalCount > 0 ? 'critical' : 'normal'}
          onClick={() => navigate('/investigations?severity=critical')}
          pulse={criticalCount > 0}
        />
        <MetricCard
          label="Detected Today"
          value={stats?.anomalies_today || 0}
          icon={<BoltIcon />}
          trend={trendDirection}
          variant="highlight"
          onClick={() => navigate('/investigations')}
        />
        <MetricCard
          label="Resolved Today"
          value={stats?.resolved_today || 0}
          subValue={`${resolutionRate}%`}
          icon={<CheckIcon />}
          variant="success"
          onClick={() => navigate('/investigations?status=resolved')}
        />
        <MetricCard
          label="MTTR"
          value={mttrMinutes ? `${mttrMinutes}m` : '--'}
          icon={<ClockIcon />}
          subLabel="avg resolution"
          variant="normal"
        />
        <MetricCard
          label="Fleet Size"
          value={stats?.devices_monitored || 0}
          icon={<DeviceIcon />}
          variant="info"
          onClick={() => navigate('/fleet')}
        />
      </div>

      {/* ================================================================== */}
      {/* MAIN CONTENT GRID */}
      {/* ================================================================== */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">

        {/* ---- LEFT COLUMN: Activity Feed + Priority Queue ---- */}
        <div className="xl:col-span-4 space-y-4">
          {/* Real-Time Activity Feed */}
          <Card
            title={
              <div className="flex items-center justify-between w-full">
                <div className="flex items-center gap-2">
                  <motion.div
                    className="w-2 h-2 rounded-full bg-emerald-400"
                    animate={{ opacity: [1, 0.3, 1] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                  />
                  <span className="text-sm font-semibold text-slate-200">Activity Feed</span>
                </div>
              </div>
            }
            noPadding
          >
            <div className="max-h-[200px] overflow-y-auto">
              {activityFeed.length > 0 ? (
                <div className="divide-y divide-slate-800/50">
                  {activityFeed.map((event, index) => (
                    <ActivityItem key={event.id} event={event} index={index} />
                  ))}
                </div>
              ) : (
                <div className="p-6 text-center">
                  <p className="text-xs text-slate-500">No recent activity</p>
                </div>
              )}
            </div>
          </Card>

          {/* Priority Queue */}
          <Card
            title={
              <div className="flex items-center justify-between w-full">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-semibold text-slate-200">Priority Queue</span>
                  {anomalies?.total && anomalies.total > 0 && (
                    <span className="px-2 py-0.5 text-xs font-bold bg-orange-500/20 text-orange-400 rounded-full">
                      {anomalies.total}
                    </span>
                  )}
                </div>
                <Link to="/investigations" className="text-xs text-amber-400 hover:text-amber-300">
                  View all →
                </Link>
              </div>
            }
            noPadding
          >
            {anomalies?.anomalies && anomalies.anomalies.length > 0 ? (
              <div className="divide-y divide-slate-800/50 max-h-[280px] overflow-y-auto">
                {anomalies.anomalies.slice(0, 8).map((anomaly, index) => (
                  <AnomalyRow key={anomaly.id} anomaly={anomaly} index={index} />
                ))}
              </div>
            ) : (
              <EmptyState
                icon={<CheckIcon className="w-8 h-8" />}
                title="All Clear"
                description="No open incidents"
                color="emerald"
              />
            )}
          </Card>
        </div>

        {/* ---- CENTER COLUMN: Trends + Model Performance ---- */}
        <div className="xl:col-span-5 space-y-4">
          {/* 7-Day Trend Chart with Annotations */}
          <Card title={<span className="text-sm font-semibold text-slate-200">Detection Trend</span>}>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-4">
                <div className="text-center">
                  <p className="text-2xl font-bold font-mono text-white">{stats?.anomalies_today || 0}</p>
                  <p className="text-[10px] text-slate-500 uppercase tracking-wide">Today</p>
                </div>
                <div className="h-8 w-px bg-slate-700" />
                <div className="text-center">
                  <p className="text-lg font-bold font-mono text-slate-400">
                    {trends?.reduce((sum, t) => sum + t.anomaly_count, 0) || 0}
                  </p>
                  <p className="text-[10px] text-slate-500 uppercase tracking-wide">7-Day Total</p>
                </div>
              </div>
              {trendDirection && (
                <div className={`flex items-center gap-1 px-2 py-1 rounded ${
                  trendDirection === 'up' ? 'bg-red-500/10 text-red-400' :
                  trendDirection === 'down' ? 'bg-emerald-500/10 text-emerald-400' :
                  'bg-slate-700/50 text-slate-400'
                }`}>
                  {trendDirection === 'up' && <TrendUpIcon className="w-3 h-3" />}
                  {trendDirection === 'down' && <TrendDownIcon className="w-3 h-3" />}
                  {trendDirection === 'stable' && <span className="text-xs">—</span>}
                  <span className="text-xs font-medium">
                    {trendDirection === 'up' ? 'Rising' : trendDirection === 'down' ? 'Falling' : 'Stable'}
                  </span>
                </div>
              )}
            </div>
            <TrendChart trends={trends} />
          </Card>

          {/* ML Model Performance */}
          <Card title={<span className="text-sm font-semibold text-slate-200">Model Performance</span>}>
            <div className="grid grid-cols-3 gap-4 mb-4">
              <div className="text-center p-3 rounded-lg bg-slate-800/50">
                <p className="text-xl font-bold font-mono text-white">
                  {((ifStats?.anomaly_rate || 0) * 100).toFixed(1)}%
                </p>
                <p className="text-[10px] text-slate-500 uppercase">Anomaly Rate</p>
              </div>
              <div className="text-center p-3 rounded-lg bg-slate-800/50">
                <p className="text-xl font-bold font-mono text-white">
                  {ifStats?.total_predictions?.toLocaleString() || '0'}
                </p>
                <p className="text-[10px] text-slate-500 uppercase">Predictions</p>
              </div>
              <div className="text-center p-3 rounded-lg bg-slate-800/50">
                <p className="text-xl font-bold font-mono text-white">
                  {ifStats?.config?.contamination ? `${(ifStats.config.contamination * 100).toFixed(0)}%` : '--'}
                </p>
                <p className="text-[10px] text-slate-500 uppercase">Threshold</p>
              </div>
            </div>

            {/* Score Distribution */}
            {scoreDistribution.length > 0 && (
              <div className="h-24">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={scoreDistribution} barCategoryGap={1}>
                    <XAxis dataKey="range" hide />
                    <YAxis hide />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'rgba(14, 17, 23, 0.95)',
                        border: '1px solid rgba(245, 166, 35, 0.2)',
                        borderRadius: '8px',
                        fontSize: '11px',
                      }}
                      formatter={(value: number) => [value, 'Count']}
                      labelFormatter={(label) => `Score: ${label}`}
                    />
                    <ReferenceLine x="-0.30" stroke="#ef4444" strokeDasharray="3 3" />
                    <Bar dataKey="count" radius={[2, 2, 0, 0]}>
                      {scoreDistribution.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={entry.isAnomaly ? '#ef4444' : '#22c55e'}
                          fillOpacity={0.7}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <div className="flex justify-between text-[9px] text-slate-600 mt-1">
                  <span>Anomaly</span>
                  <span>Normal</span>
                </div>
              </div>
            )}
          </Card>

          {/* ML Pipeline Status */}
          <Card title={<span className="text-sm font-semibold text-slate-200">Pipeline Status</span>}>
            <div className="grid grid-cols-2 gap-3">
              {/* Last Detection Run */}
              {automationStatus?.last_scoring_result && (
                <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/50">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[10px] font-medium text-slate-500 uppercase">Last Run</span>
                    <span className="text-[9px] text-slate-600">
                      {automationStatus.last_scoring_result.timestamp
                        ? formatDistanceToNowStrict(new Date(automationStatus.last_scoring_result.timestamp)) + ' ago'
                        : ''}
                    </span>
                  </div>
                  <div className="flex items-baseline gap-2">
                    <p className="text-xl font-bold font-mono text-white">{automationStatus.last_scoring_result.total_scored}</p>
                    <p className="text-xs text-slate-500">devices</p>
                  </div>
                  <div className="flex items-center gap-2 mt-1">
                    <span className={`text-xs font-mono ${automationStatus.last_scoring_result.anomalies_detected > 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                      {automationStatus.last_scoring_result.anomalies_detected} anomalies
                    </span>
                  </div>
                </div>
              )}

              {/* Next Scheduled */}
              <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/50">
                <span className="text-[10px] font-medium text-slate-500 uppercase">Next Run</span>
                <p className="text-xl font-bold font-mono text-white mt-2">
                  {automationStatus?.next_scoring_time && automationStatus.is_running
                    ? format(new Date(automationStatus.next_scoring_time), 'HH:mm')
                    : '--:--'}
                </p>
                <p className="text-[10px] text-slate-500 mt-1">
                  {automationStatus?.is_running ? 'Scheduled' : 'Automation paused'}
                </p>
              </div>

              {/* Training Status */}
              <Link to="/training" className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/50 hover:border-amber-500/30 transition-colors">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-medium text-slate-500 uppercase">Training</span>
                  <BrainIcon className="w-3 h-3 text-slate-500" />
                </div>
                <p className="text-sm font-medium text-slate-200 mt-2">
                  {trainingStatus?.status === 'running'
                    ? `${Math.round(trainingStatus.progress || 0)}% complete`
                    : trainingStatus?.status === 'completed'
                      ? 'Ready'
                      : 'Idle'}
                </p>
                {trainingStatus?.status === 'running' && (
                  <div className="w-full h-1 bg-slate-700 rounded-full mt-2 overflow-hidden">
                    <motion.div
                      className="h-full bg-amber-500 rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${trainingStatus.progress || 0}%` }}
                    />
                  </div>
                )}
              </Link>

              {/* Automation Status */}
              <Link to="/automation" className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/50 hover:border-amber-500/30 transition-colors">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-medium text-slate-500 uppercase">Automation</span>
                  <div className={`w-2 h-2 rounded-full ${automationStatus?.is_running ? 'bg-emerald-400' : 'bg-slate-500'}`} />
                </div>
                <p className="text-sm font-medium text-slate-200 mt-2">
                  {automationStatus?.is_running ? 'Active' : 'Paused'}
                </p>
                {automationStatus?.scoring_status === 'running' && (
                  <p className="text-[10px] text-amber-400 mt-1">Scoring in progress...</p>
                )}
              </Link>
            </div>
          </Card>
        </div>

        {/* ---- RIGHT COLUMN: Locations + Services + AI Insights ---- */}
        <div className="xl:col-span-3 space-y-4">
          {/* Top Locations */}
          <Card
            title={
              <div className="flex items-center justify-between w-full">
                <span className="text-sm font-semibold text-slate-200">Top Locations</span>
                <Link to="/locations" className="text-xs text-amber-400 hover:text-amber-300">
                  View map →
                </Link>
              </div>
            }
          >
            {topLocations.length > 0 ? (
              <div className="space-y-2">
                {topLocations.map((location, index) => (
                  <LocationRow key={location.id} location={location} rank={index + 1} />
                ))}
              </div>
            ) : (
              <EmptyState
                icon={<MapIcon className="w-8 h-8" />}
                title="No Location Data"
                description="Configure custom attributes"
                color="slate"
                action={{ label: 'Configure', to: '/system' }}
              />
            )}
          </Card>

          {/* Service Health */}
          <Card title={<span className="text-sm font-semibold text-slate-200">Services</span>}>
            <ServiceHealthGrid connections={connections} />
          </Card>

          {/* AI Insights */}
          <Card title={<span className="text-sm font-semibold text-slate-200">AI Insights</span>}>
            {aiInsights.length > 0 ? (
              <div className="space-y-2 max-h-[240px] overflow-y-auto">
                {aiInsights.map((insight) => (
                  <InsightCard key={insight.id} insight={insight} navigate={navigate} />
                ))}
              </div>
            ) : (
              <EmptyState
                icon={<BrainIcon className="w-8 h-8" />}
                title="All Systems Optimal"
                description="No actionable insights"
                color="emerald"
              />
            )}
          </Card>

          {/* Quick Actions */}
          <div className="grid grid-cols-2 gap-2">
            <QuickAction to="/investigations" icon={<SearchIcon />} label="Investigate" primary />
            <QuickAction to="/training" icon={<BrainIcon />} label="Train Model" />
          </div>
        </div>
      </div>

      {/* ================================================================== */}
      {/* ACTIVE MODEL FOOTER */}
      {/* ================================================================== */}
      <div className="flex items-center justify-between p-3 rounded-lg bg-slate-900/50 border border-slate-800/50">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-emerald-400" />
            <span className="text-xs text-slate-500">Active Model</span>
          </div>
          <span className="text-xs font-mono text-slate-300">
            {trainingStatus?.run_id || 'v' + format(new Date(), 'yyyyMMdd')}
          </span>
        </div>
        <div className="flex items-center gap-4 text-xs text-slate-500">
          <span>Updated {trainingStatus?.started_at ? formatDistanceToNowStrict(new Date(trainingStatus.started_at)) + ' ago' : 'recently'}</span>
        </div>
      </div>
    </motion.div>
  );
}

// ============================================================================
// Sub-Components
// ============================================================================

function MetricCard({
  label,
  value,
  subValue,
  subLabel,
  icon,
  variant,
  onClick,
  pulse,
  trend,
}: {
  label: string;
  value: number | string;
  subValue?: string;
  subLabel?: string;
  icon: React.ReactNode;
  variant: 'normal' | 'highlight' | 'warning' | 'critical' | 'success' | 'info' | 'alert';
  onClick?: () => void;
  pulse?: boolean;
  trend?: 'up' | 'down' | 'stable' | null;
}) {
  const variants = {
    normal: 'border-slate-700/50 bg-slate-800/30',
    highlight: 'border-amber-500/30 bg-amber-500/5',
    warning: 'border-orange-500/30 bg-orange-500/10',
    critical: 'border-red-500/30 bg-red-500/10',
    success: 'border-emerald-500/30 bg-emerald-500/5',
    info: 'border-indigo-500/30 bg-indigo-500/5',
    alert: 'border-amber-500/20 bg-slate-800/30',
  };

  const iconColors = {
    normal: 'text-slate-500',
    highlight: 'text-amber-400',
    warning: 'text-orange-400',
    critical: 'text-red-400',
    success: 'text-emerald-400',
    info: 'text-indigo-400',
    alert: 'text-amber-400',
  };

  const Wrapper = onClick ? 'button' : 'div';

  return (
    <Wrapper
      onClick={onClick}
      className={`relative p-4 rounded-xl border transition-all ${variants[variant]} ${onClick ? 'hover:scale-[1.02] cursor-pointer' : ''}`}
    >
      {pulse && (
        <motion.div
          className="absolute top-3 right-3 w-2 h-2 rounded-full bg-current"
          style={{ color: variant === 'critical' ? '#ef4444' : '#f97316' }}
          animate={{ scale: [1, 1.5, 1], opacity: [1, 0.5, 1] }}
          transition={{ duration: 1.5, repeat: Infinity }}
        />
      )}
      <div className="flex items-start justify-between">
        <div>
          <p className="text-3xl font-bold font-mono text-white tracking-tight">{value}</p>
          <div className="flex items-center gap-2 mt-1">
            <p className="text-xs text-slate-500">{label}</p>
            {subValue && <span className="text-xs font-medium text-emerald-400">{subValue}</span>}
            {subLabel && <span className="text-[10px] text-slate-600">{subLabel}</span>}
          </div>
        </div>
        <div className="flex flex-col items-end gap-1">
          <div className={iconColors[variant]}>{icon}</div>
          {trend && (
            <div className={`${trend === 'up' ? 'text-red-400' : trend === 'down' ? 'text-emerald-400' : 'text-slate-500'}`}>
              {trend === 'up' && <TrendUpIcon className="w-3 h-3" />}
              {trend === 'down' && <TrendDownIcon className="w-3 h-3" />}
            </div>
          )}
        </div>
      </div>
    </Wrapper>
  );
}

function ActivityItem({ event, index }: { event: { id: string; type: string; message: string; timestamp: string; severity?: string }; index: number }) {
  const typeColors = {
    anomaly: 'bg-red-500',
    resolved: 'bg-emerald-500',
    training: 'bg-amber-500',
    system: 'bg-indigo-500',
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.03 }}
      className="flex items-center gap-3 px-3 py-2 hover:bg-slate-800/30"
    >
      <div className={`w-1.5 h-1.5 rounded-full ${typeColors[event.type as keyof typeof typeColors] || 'bg-slate-500'}`} />
      <div className="flex-1 min-w-0">
        <p className="text-xs text-slate-300 truncate">{event.message}</p>
      </div>
      {event.severity && (
        <span className={`px-1.5 py-0.5 text-[9px] font-bold rounded ${
          event.severity === 'CRITICAL' ? 'bg-red-500/20 text-red-400' :
          event.severity === 'HIGH' ? 'bg-orange-500/20 text-orange-400' :
          'bg-amber-500/20 text-amber-400'
        }`}>
          {event.severity}
        </span>
      )}
      <span className="text-[10px] text-slate-600 whitespace-nowrap">
        {formatDistanceToNowStrict(new Date(event.timestamp), { addSuffix: false })}
      </span>
    </motion.div>
  );
}

function AnomalyRow({ anomaly, index }: { anomaly: Anomaly; index: number }) {
  const severity = getSeverityFromScore(anomaly.anomaly_score);

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.03 }}
    >
      <Link
        to={`/investigations/${anomaly.id}`}
        className="flex items-center gap-3 p-3 hover:bg-slate-800/30 transition-colors group"
      >
        <div className={`w-1.5 h-10 rounded-full ${severity.color.dot}`} />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-slate-200">Device #{anomaly.device_id}</span>
            <span className={`px-1.5 py-0.5 text-[9px] font-bold rounded ${severity.color.bg} ${severity.color.text}`}>
              {severity.label}
            </span>
          </div>
          <p className="text-[10px] text-slate-500">
            {formatDistanceToNowStrict(new Date(anomaly.timestamp))} ago
          </p>
        </div>
        <span className="text-xs font-mono text-slate-600 group-hover:text-amber-400 transition-colors">
          {anomaly.anomaly_score.toFixed(3)}
        </span>
      </Link>
    </motion.div>
  );
}

function TrendChart({ trends }: { trends: { date: string; anomaly_count: number }[] | undefined }) {
  const hasTrendData = trends && trends.length > 0 && trends.some(t => t.anomaly_count > 0);

  if (!hasTrendData) {
    return (
      <div className="h-28 flex items-center justify-center">
        <p className="text-xs text-slate-500">No trend data available</p>
      </div>
    );
  }

  return (
    <div className="h-28">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={trends}>
          <defs>
            <linearGradient id="trendGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#f5a623" stopOpacity={0.3} />
              <stop offset="100%" stopColor="#f5a623" stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="date"
            tickFormatter={(v) => format(new Date(v), 'EEE')}
            stroke="#475569"
            fontSize={10}
            tickLine={false}
            axisLine={false}
          />
          <YAxis hide />
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(14, 17, 23, 0.95)',
              border: '1px solid rgba(245, 166, 35, 0.2)',
              borderRadius: '8px',
              fontSize: '11px',
            }}
            labelFormatter={(v) => format(new Date(v), 'MMM d')}
            formatter={(value: number) => [value, 'Anomalies']}
          />
          <Area
            type="monotone"
            dataKey="anomaly_count"
            stroke="#f5a623"
            strokeWidth={2}
            fill="url(#trendGradient)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function InsightCard({ insight, navigate }: { insight: AIInsight; navigate: (path: string) => void }) {
  const severityColors: Record<string, string> = {
    critical: 'border-l-red-500 bg-red-500/5',
    high: 'border-l-orange-500 bg-orange-500/5',
    warning: 'border-l-orange-500 bg-orange-500/5',
    medium: 'border-l-amber-500 bg-amber-500/5',
    low: 'border-l-slate-500 bg-slate-800/30',
    info: 'border-l-indigo-500 bg-indigo-500/5',
  };

  return (
    <div className={`p-3 rounded-lg border-l-2 ${severityColors[insight.severity || 'info']}`}>
      <p className="text-xs font-medium text-slate-200">{insight.title}</p>
      <p className="text-[10px] text-slate-500 mt-0.5">{insight.description}</p>
      {insight.actionable && (
        <button
          onClick={() => {
            if (insight.id === 'critical-alert') navigate('/investigations?severity=critical');
            else if (insight.id === 'system-degraded') navigate('/system');
          }}
          className="mt-2 text-[10px] text-amber-400 hover:text-amber-300"
        >
          {insight.actionLabel} →
        </button>
      )}
    </div>
  );
}

function LocationRow({ location, rank }: { location: { id: string; name: string; deviceCount: number; anomalyCount?: number }; rank: number }) {
  const anomalyCount = location.anomalyCount ?? 0;
  const percentage = location.deviceCount > 0 ? Math.round((anomalyCount / location.deviceCount) * 100) : 0;

  return (
    <div className="flex items-center gap-3 p-2 rounded-lg bg-slate-800/30 hover:bg-slate-800/50 transition-colors">
      <span className={`w-5 h-5 rounded text-[10px] font-bold flex items-center justify-center ${
        rank === 1 && anomalyCount > 0 ? 'bg-red-500/20 text-red-400' :
        rank <= 3 && anomalyCount > 0 ? 'bg-orange-500/20 text-orange-400' :
        'bg-slate-700/50 text-slate-500'
      }`}>
        {rank}
      </span>
      <div className="flex-1 min-w-0">
        <p className="text-xs font-medium text-slate-300 truncate">{location.name}</p>
        <div className="flex items-center gap-2 mt-0.5">
          <p className="text-[10px] text-slate-500">{location.deviceCount} devices</p>
          {anomalyCount > 0 && (
            <div className="flex-1 h-1 bg-slate-700 rounded-full overflow-hidden max-w-[60px]">
              <div
                className="h-full bg-red-500 rounded-full"
                style={{ width: `${Math.min(percentage, 100)}%` }}
              />
            </div>
          )}
        </div>
      </div>
      {anomalyCount > 0 && (
        <span className="px-1.5 py-0.5 text-[10px] font-bold bg-red-500/20 text-red-400 rounded">
          {anomalyCount}
        </span>
      )}
    </div>
  );
}

function ServiceHealthGrid({ connections }: { connections: any }) {
  const services = [
    { key: 'backend_db', label: 'PostgreSQL', icon: '🗄️' },
    { key: 'redis', label: 'Redis', icon: '⚡' },
    { key: 'qdrant', label: 'Qdrant', icon: '🔍' },
    { key: 'dw_sql', label: 'XSight DB', icon: '📊' },
    { key: 'mc_sql', label: 'MC SQL', icon: '📱' },
    { key: 'mobicontrol_api', label: 'MC API', icon: '🔌' },
    { key: 'llm', label: 'LLM', icon: '🤖' },
  ];

  const connectedCount = services.filter(s => connections?.[s.key]?.connected).length;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-xs text-slate-500">{connectedCount}/{services.length} online</span>
        <Link to="/system" className="text-[10px] text-amber-400 hover:text-amber-300">Configure →</Link>
      </div>
      <div className="grid grid-cols-4 gap-1.5">
        {services.map((service) => {
          const isConnected = connections?.[service.key]?.connected;
          return (
            <div
              key={service.key}
              className={`p-1.5 rounded text-center transition-colors ${
                isConnected ? 'bg-emerald-500/10' : 'bg-red-500/10'
              }`}
              title={`${service.label}: ${isConnected ? 'Connected' : 'Disconnected'}`}
            >
              <div className={`w-2 h-2 mx-auto rounded-full ${
                isConnected ? 'bg-emerald-400' : 'bg-red-400'
              }`} />
              <p className="text-[8px] text-slate-500 mt-1 truncate">{service.label}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function QuickAction({ to, icon, label, primary }: { to: string; icon: React.ReactNode; label: string; primary?: boolean }) {
  return (
    <Link
      to={to}
      className={`flex flex-col items-center gap-1.5 p-3 rounded-lg border transition-all ${
        primary
          ? 'bg-amber-500/10 border-amber-500/30 hover:bg-amber-500/20 text-amber-400'
          : 'bg-slate-800/50 border-slate-700/50 hover:border-amber-500/30 text-slate-400'
      }`}
    >
      <div>{icon}</div>
      <span className="text-[10px]">{label}</span>
    </Link>
  );
}

function EmptyState({
  icon,
  title,
  description,
  color,
  action,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
  color: 'emerald' | 'slate';
  action?: { label: string; to: string };
}) {
  const colorClasses = {
    emerald: 'text-emerald-400 bg-emerald-500/10',
    slate: 'text-slate-400 bg-slate-700/50',
  };

  return (
    <div className="py-6 text-center">
      <div className={`w-12 h-12 mx-auto mb-2 rounded-xl flex items-center justify-center ${colorClasses[color]}`}>
        {icon}
      </div>
      <p className="text-sm font-medium text-slate-300">{title}</p>
      <p className="text-[10px] text-slate-500 mt-0.5">{description}</p>
      {action && (
        <Link to={action.to} className="inline-block mt-2 text-[10px] text-amber-400 hover:text-amber-300">
          {action.label} →
        </Link>
      )}
    </div>
  );
}

// ============================================================================
// Icons
// ============================================================================

const BoltIcon = ({ className = "w-5 h-5" }: { className?: string }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
  </svg>
);

const CaseIcon = ({ className = "w-5 h-5" }: { className?: string }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
  </svg>
);

const AlertIcon = ({ className = "w-5 h-5" }: { className?: string }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
  </svg>
);

const CheckIcon = ({ className = "w-5 h-5" }: { className?: string }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);

const ClockIcon = ({ className = "w-5 h-5" }: { className?: string }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);

const DeviceIcon = ({ className = "w-5 h-5" }: { className?: string }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 18h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
  </svg>
);

const BrainIcon = ({ className = "w-5 h-5" }: { className?: string }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
  </svg>
);

const MapIcon = ({ className = "w-5 h-5" }: { className?: string }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
  </svg>
);

const SearchIcon = ({ className = "w-4 h-4" }: { className?: string }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
  </svg>
);

const TrendUpIcon = ({ className = "w-4 h-4" }: { className?: string }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
  </svg>
);

const TrendDownIcon = ({ className = "w-4 h-4" }: { className?: string }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
  </svg>
);

export default Dashboard;
