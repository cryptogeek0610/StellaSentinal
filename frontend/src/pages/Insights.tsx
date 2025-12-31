/**
 * Insights Page - Stellar Operations Analytics
 * 
 * Advanced analytics, model performance metrics,
 * and trend analysis with Isolation Forest visualization
 */

import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { motion } from 'framer-motion';
import { Card } from '../components/Card';
import { KPICard } from '../components/KPICard';
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell,
} from 'recharts';
import { format } from 'date-fns';
import { IsolationForestViz } from '../components/IsolationForestViz';

function Insights() {
  const [timeRange, setTimeRange] = useState(30);

  const { data: trends } = useQuery({
    queryKey: ['dashboard', 'trends', timeRange],
    queryFn: () => api.getDashboardTrends({ days: timeRange }),
  });

  const { data: anomalies } = useQuery({
    queryKey: ['anomalies', 'all', timeRange],
    queryFn: () => api.getAnomalies({ page: 1, page_size: 1000 }),
  });

  // Calculate severity distribution
  const severityDistribution = useMemo(() => {
    if (!anomalies) return [];

    const distribution = {
      critical: 0,
      high: 0,
      medium: 0,
      low: 0,
    };

    anomalies.anomalies.forEach((a) => {
      if (a.anomaly_score <= -0.7) distribution.critical++;
      else if (a.anomaly_score <= -0.5) distribution.high++;
      else if (a.anomaly_score <= -0.3) distribution.medium++;
      else distribution.low++;
    });

    return [
      { name: 'Critical', value: distribution.critical, color: '#ef4444' },
      { name: 'High', value: distribution.high, color: '#f97316' },
      { name: 'Medium', value: distribution.medium, color: '#f59e0b' },
      { name: 'Low', value: distribution.low, color: '#64748b' },
    ];
  }, [anomalies]);

  // Calculate status distribution
  const statusDistribution = useMemo(() => {
    if (!anomalies) return [];

    const distribution = {
      open: 0,
      investigating: 0,
      resolved: 0,
      false_positive: 0,
    };

    anomalies.anomalies.forEach((a) => {
      distribution[a.status as keyof typeof distribution]++;
    });

    return [
      { name: 'Open', value: distribution.open, color: '#ef4444' },
      { name: 'Investigating', value: distribution.investigating, color: '#f97316' },
      { name: 'Resolved', value: distribution.resolved, color: '#10b981' },
      { name: 'False Positive', value: distribution.false_positive, color: '#64748b' },
    ];
  }, [anomalies]);

  // Top affected devices
  const topDevices = useMemo(() => {
    if (!anomalies) return [];

    const deviceCounts = new Map<number, number>();
    anomalies.anomalies.forEach((a) => {
      deviceCounts.set(a.device_id, (deviceCounts.get(a.device_id) || 0) + 1);
    });

    return Array.from(deviceCounts.entries())
      .map(([device_id, count]) => ({ device_id, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5);
  }, [anomalies]);

  // Resolution rate
  const resolutionRate = useMemo(() => {
    if (!anomalies || anomalies.total === 0) return 0;
    const resolved = anomalies.anomalies.filter(a => a.status === 'resolved' || a.status === 'false_positive').length;
    return Math.round((resolved / anomalies.total) * 100);
  }, [anomalies]);

  // Average score
  const avgScore = useMemo(() => {
    if (!anomalies || anomalies.anomalies.length === 0) return 0;
    const sum = anomalies.anomalies.reduce((acc, a) => acc + a.anomaly_score, 0);
    return sum / anomalies.anomalies.length;
  }, [anomalies]);

  // Critical rate
  const criticalRate = useMemo(() => {
    if (!anomalies?.total) return 0;
    return Math.round((severityDistribution.find(s => s.name === 'Critical')?.value || 0) / anomalies.total * 100);
  }, [anomalies, severityDistribution]);

  return (
    <motion.div
      className="space-y-8"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      {/* Header */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">AI Insights</h1>
          <p className="text-slate-500 mt-1">
            Analytics and model performance metrics
          </p>
        </div>

        {/* Time Range Selector */}
        <div className="flex items-center gap-2 p-1 bg-slate-800/50 rounded-lg border border-slate-700/50">
          {[7, 14, 30, 90].map((days) => (
            <button
              key={days}
              onClick={() => setTimeRange(days)}
              className={`px-4 py-2 text-sm font-medium rounded-md transition-all ${
                timeRange === days
                  ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              {days}d
            </button>
          ))}
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <KPICard
          title="Total Anomalies"
          value={anomalies?.total || 0}
          color="stellar"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          }
        />
        <KPICard
          title="Resolution Rate"
          value={`${resolutionRate}%`}
          color="aurora"
          progressValue={resolutionRate}
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          }
        />
        <KPICard
          title="Avg. Score"
          value={avgScore.toFixed(3)}
          color="warning"
          showProgress={false}
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          }
        />
        <KPICard
          title="Critical Rate"
          value={`${criticalRate}%`}
          color={criticalRate > 20 ? 'danger' : 'stellar'}
          isActive={criticalRate > 20}
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          }
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
        {/* Trend Chart */}
        <Card title={<span className="telemetry-label">{timeRange}-Day Trend</span>} className="xl:col-span-2">
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trends || []}>
                <defs>
                  <linearGradient id="insightTrendGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#f5a623" stopOpacity={0.4} />
                    <stop offset="100%" stopColor="#f5a623" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.1)" vertical={false} />
                <XAxis
                  dataKey="date"
                  tickFormatter={(v) => format(new Date(v), 'MMM d')}
                  stroke="#475569"
                  fontSize={10}
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis stroke="#475569" fontSize={10} tickLine={false} axisLine={false} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(14, 17, 23, 0.95)',
                    border: '1px solid rgba(245, 166, 35, 0.2)',
                    borderRadius: '12px',
                  }}
                  labelFormatter={(v) => format(new Date(v), 'MMM d, yyyy')}
                  formatter={(value: number) => [value, 'Anomalies']}
                />
                <Area
                  type="monotone"
                  dataKey="anomaly_count"
                  stroke="#f5a623"
                  strokeWidth={2}
                  fill="url(#insightTrendGradient)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Severity Distribution */}
        <Card title={<span className="telemetry-label">Severity Distribution</span>}>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={severityDistribution} layout="vertical">
                <XAxis type="number" stroke="#475569" fontSize={10} tickLine={false} axisLine={false} />
                <YAxis dataKey="name" type="category" stroke="#475569" fontSize={10} tickLine={false} axisLine={false} width={70} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(14, 17, 23, 0.95)',
                    border: '1px solid rgba(245, 166, 35, 0.2)',
                    borderRadius: '12px',
                  }}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {severityDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      {/* Second Row */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
        {/* Status Distribution */}
        <Card title={<span className="telemetry-label">Status Breakdown</span>}>
          <div className="space-y-4">
            {statusDistribution.map((status, index) => (
              <motion.div
                key={status.name}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="flex items-center gap-4"
              >
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: status.color }} />
                <span className="flex-1 text-sm text-slate-400">{status.name}</span>
                <span className="text-sm font-mono font-bold text-white">{status.value}</span>
                <div className="w-24 h-2 bg-slate-800 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full rounded-full"
                    style={{ backgroundColor: status.color }}
                    initial={{ width: 0 }}
                    animate={{ width: `${anomalies?.total ? (status.value / anomalies.total) * 100 : 0}%` }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                  />
                </div>
              </motion.div>
            ))}
          </div>
        </Card>

        {/* Top Affected Devices */}
        <Card title={<span className="telemetry-label">Most Affected Devices</span>}>
          <div className="space-y-3">
            {topDevices.map((device, index) => (
              <motion.div
                key={device.device_id}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="flex items-center gap-4 p-3 bg-slate-800/30 rounded-lg"
              >
                <span className={`w-6 h-6 flex items-center justify-center rounded-full text-xs font-bold ${
                  index === 0 ? 'bg-red-500/20 text-red-400' :
                  index === 1 ? 'bg-orange-500/20 text-orange-400' :
                  'bg-slate-700 text-slate-400'
                }`}>
                  {index + 1}
                </span>
                <span className="flex-1 font-medium text-white">Device #{device.device_id}</span>
                <span className="text-sm font-mono text-amber-400">{device.count}</span>
              </motion.div>
            ))}
            {topDevices.length === 0 && (
              <p className="text-center text-sm text-slate-500 py-4">No data available</p>
            )}
          </div>
        </Card>

        {/* Quick Stats */}
        <Card title={<span className="telemetry-label">Performance Summary</span>}>
          <div className="space-y-4">
            <StatRow label="Avg. Time to Resolution" value="—" subtext="Not enough data" />
            <StatRow
              label="Detection Accuracy"
              value={`${100 - (anomalies?.total ? Math.round((statusDistribution.find(s => s.name === 'False Positive')?.value || 0) / anomalies.total * 100) : 0)}%`}
              subtext="Based on false positive rate"
            />
            <StatRow
              label="Open Rate"
              value={`${anomalies?.total ? Math.round((statusDistribution.find(s => s.name === 'Open')?.value || 0) / anomalies.total * 100) : 0}%`}
              subtext="Anomalies awaiting review"
            />
          </div>
        </Card>
      </div>

      {/* Isolation Forest Section */}
      <div>
        <div className="mb-4">
          <h2 className="text-xl font-bold text-white">Model Analysis</h2>
          <p className="text-sm text-slate-500 mt-1">
            Isolation Forest performance and score distribution
          </p>
        </div>
        <IsolationForestViz days={timeRange} />
      </div>
    </motion.div>
  );
}

// Stat Row Component
function StatRow({
  label,
  value,
  subtext,
}: {
  label: string;
  value: string;
  subtext: string;
}) {
  return (
    <div className="flex items-center justify-between p-3 bg-slate-800/30 rounded-lg">
      <div>
        <p className="text-sm font-medium text-slate-300">{label}</p>
        <p className="text-xs text-slate-500">{subtext}</p>
      </div>
      <p className="text-xl font-bold text-white font-mono">{value}</p>
    </div>
  );
}

export default Insights;
