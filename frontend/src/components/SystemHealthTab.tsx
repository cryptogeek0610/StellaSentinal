/**
 * System Health Tab Component
 *
 * Displays fleet health metrics including CPU, RAM, storage, and temperature.
 * Provides storage forecasting and cohort-level health breakdowns.
 */

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { Card } from './Card';
import { KPICard } from './KPICard';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
} from 'recharts';
import { format } from 'date-fns';
import clsx from 'clsx';

const METRIC_OPTIONS = [
  { id: 'cpu_usage', label: 'CPU Usage' },
  { id: 'memory_usage', label: 'Memory Usage' },
  { id: 'storage_available', label: 'Storage Available' },
  { id: 'device_temp', label: 'Temperature' },
];

export function SystemHealthTab() {
  const [selectedMetric, setSelectedMetric] = useState('cpu_usage');

  // Fetch system health summary
  const { data: healthSummary, isLoading: summaryLoading } = useQuery({
    queryKey: ['insights', 'system-health', 'summary'],
    queryFn: () => api.getSystemHealthSummary(7),
  });

  // Fetch health trends for selected metric
  const { data: healthTrends, isLoading: trendsLoading } = useQuery({
    queryKey: ['insights', 'system-health', 'trends', selectedMetric],
    queryFn: () => api.getHealthTrends(selectedMetric, 7),
  });

  // Fetch storage forecast
  const { data: storageForecast, isLoading: forecastLoading } = useQuery({
    queryKey: ['insights', 'system-health', 'storage-forecast'],
    queryFn: () => api.getStorageForecast(30, 10),
  });

  if (summaryLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-pulse text-slate-400">Loading system health data...</div>
      </div>
    );
  }

  if (!healthSummary) {
    return (
      <div className="flex items-center justify-center h-64 text-slate-400">
        No health data available
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* KPI Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <KPICard
          title="Fleet Health Score"
          value={`${healthSummary.fleet_health_score.toFixed(0)}%`}
          color={healthSummary.fleet_health_score >= 80 ? 'aurora' : healthSummary.fleet_health_score >= 60 ? 'warning' : 'danger'}
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          }
          explainer={`${healthSummary.health_trend} trend`}
        />
        <KPICard
          title="Devices High CPU"
          value={healthSummary.metrics.devices_high_cpu}
          color={healthSummary.metrics.devices_high_cpu > 10 ? 'warning' : 'stellar'}
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
            </svg>
          }
          explainer={`Avg CPU: ${healthSummary.metrics.avg_cpu_usage.toFixed(1)}%`}
        />
        <KPICard
          title="Low Storage"
          value={healthSummary.metrics.devices_low_storage}
          color={healthSummary.metrics.devices_low_storage > 5 ? 'danger' : 'stellar'}
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
            </svg>
          }
          explainer={`<20% storage remaining`}
        />
        <KPICard
          title="High Temperature"
          value={healthSummary.metrics.devices_high_temp}
          color={healthSummary.metrics.devices_high_temp > 3 ? 'danger' : 'aurora'}
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M17.657 18.657A8 8 0 016.343 7.343S7 9 9 10c0-2 .5-5 2.986-7C14 5 16.09 5.777 17.656 7.343A7.975 7.975 0 0120 13a7.975 7.975 0 01-2.343 5.657z" />
            </svg>
          }
          explainer={`Avg temp: ${healthSummary.metrics.avg_device_temp.toFixed(1)}°C`}
        />
      </div>

      {/* Trends Chart */}
      <Card
        title={
          <div className="flex items-center justify-between w-full">
            <span>Resource Trends <span className="text-slate-500 text-sm font-normal ml-2">7-day view</span></span>
            <div className="flex gap-1">
              {METRIC_OPTIONS.map((metric) => (
                <button
                  key={metric.id}
                  onClick={() => setSelectedMetric(metric.id)}
                  className={clsx(
                    'px-3 py-1 text-xs font-medium rounded-md transition-all',
                    selectedMetric === metric.id
                      ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                      : 'text-slate-400 hover:text-white'
                  )}
                >
                  {metric.label}
                </button>
              ))}
            </div>
          </div>
        }
      >
        {trendsLoading ? (
          <div className="h-64 flex items-center justify-center text-slate-400">
            Loading trends...
          </div>
        ) : healthTrends?.trends ? (
          <ResponsiveContainer width="100%" height={240}>
            <AreaChart data={healthTrends.trends}>
              <defs>
                <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis
                dataKey="timestamp"
                stroke="#64748b"
                fontSize={11}
                tickLine={false}
                tickFormatter={(val) => format(new Date(val), 'MMM d')}
              />
              <YAxis
                stroke="#64748b"
                fontSize={11}
                tickLine={false}
                axisLine={false}
                tickFormatter={(val) => `${val.toFixed(0)}%`}
              />
              <Tooltip
                contentStyle={{
                  background: '#1e293b',
                  border: '1px solid #334155',
                  borderRadius: '8px',
                  fontSize: '12px',
                }}
                labelFormatter={(val) => format(new Date(val), 'PPP')}
                formatter={(value: number) => [`${value.toFixed(1)}%`, METRIC_OPTIONS.find(m => m.id === selectedMetric)?.label]}
              />
              <Area
                type="monotone"
                dataKey="value"
                stroke="#f59e0b"
                strokeWidth={2}
                fill="url(#colorValue)"
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-64 flex items-center justify-center text-slate-400">
            No trend data available
          </div>
        )}
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Storage Forecast */}
        <Card
          title={<>Storage Forecast <span className="text-slate-500 text-sm font-normal ml-2">{storageForecast?.total_at_risk_count || 0} devices at risk</span></>}
        >
          {forecastLoading ? (
            <div className="h-48 flex items-center justify-center text-slate-400">
              Loading forecast...
            </div>
          ) : storageForecast?.devices_at_risk && storageForecast.devices_at_risk.length > 0 ? (
            <div className="space-y-3">
              {storageForecast.devices_at_risk.slice(0, 5).map((device) => (
                <div
                  key={device.device_id}
                  className="flex items-center justify-between p-3 rounded-lg bg-slate-800/50 border border-slate-700/30"
                >
                  <div>
                    <div className="text-sm font-medium text-white">{device.device_name}</div>
                    <div className="text-xs text-slate-500">
                      {device.current_storage_pct.toFixed(1)}% remaining
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={clsx(
                      'text-lg font-bold font-mono',
                      device.days_until_full && device.days_until_full <= 7 ? 'text-red-400' : 'text-amber-400'
                    )}>
                      {device.days_until_full ? `${device.days_until_full}d` : '-'}
                    </div>
                    <div className="text-xs text-slate-500">until full</div>
                  </div>
                </div>
              ))}
              {storageForecast.recommendations && storageForecast.recommendations.length > 0 && (
                <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20">
                  <div className="text-xs font-medium text-amber-400 mb-1">Recommendation</div>
                  <div className="text-sm text-slate-300">{storageForecast.recommendations[0]}</div>
                </div>
              )}
            </div>
          ) : (
            <div className="h-48 flex items-center justify-center text-slate-400">
              No devices at risk
            </div>
          )}
        </Card>

        {/* Cohort Health */}
        <Card
          title={<>Health by Device Cohort <span className="text-slate-500 text-sm font-normal ml-2">Comparative analysis</span></>}
        >
          {healthSummary.cohort_breakdown && healthSummary.cohort_breakdown.length > 0 ? (
            <ResponsiveContainer width="100%" height={240}>
              <BarChart
                data={healthSummary.cohort_breakdown}
                layout="vertical"
                margin={{ left: 100 }}
              >
                <XAxis type="number" domain={[0, 100]} stroke="#64748b" fontSize={11} tickLine={false} />
                <YAxis
                  type="category"
                  dataKey="cohort_name"
                  stroke="#64748b"
                  fontSize={11}
                  tickLine={false}
                  width={100}
                />
                <Tooltip
                  contentStyle={{
                    background: '#1e293b',
                    border: '1px solid #334155',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                  formatter={(value: number) => [`${value.toFixed(0)}%`, 'Health Score']}
                />
                <Bar dataKey="health_score" radius={[0, 4, 4, 0]}>
                  {healthSummary.cohort_breakdown.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.health_score >= 80 ? '#10b981' : entry.health_score >= 60 ? '#f59e0b' : '#ef4444'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-48 flex items-center justify-center text-slate-400">
              No cohort data available
            </div>
          )}
        </Card>
      </div>

      {/* Recommendations */}
      {healthSummary.recommendations && healthSummary.recommendations.length > 0 && (
        <Card title="AI Recommendations">
          <div className="space-y-2">
            {healthSummary.recommendations.map((rec, index) => (
              <div
                key={index}
                className="flex items-start gap-3 p-3 rounded-lg bg-slate-800/50 border border-slate-700/30"
              >
                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-amber-500/20 flex items-center justify-center">
                  <span className="text-xs font-bold text-amber-400">{index + 1}</span>
                </div>
                <p className="text-sm text-slate-300">{rec}</p>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}

export default SystemHealthTab;
