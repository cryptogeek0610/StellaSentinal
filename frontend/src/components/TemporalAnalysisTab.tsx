/**
 * Temporal Analysis Tab Component
 *
 * Provides time-based analytics including hourly breakdowns, peak detection,
 * day-over-day comparisons, and week-over-week trends.
 */

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { Card } from './Card';
import { KPICard } from './KPICard';
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { format } from 'date-fns';
import clsx from 'clsx';

const METRIC_OPTIONS = [
  { id: 'data_usage', label: 'Data Usage' },
  { id: 'battery_drain', label: 'Battery Drain' },
  { id: 'app_usage', label: 'App Usage' },
];

// Hour-of-day heatmap colors
const getHourColor = (value: number, max: number): string => {
  const ratio = value / max;
  if (ratio >= 0.8) return '#ef4444'; // High usage
  if (ratio >= 0.6) return '#f97316';
  if (ratio >= 0.4) return '#f59e0b';
  if (ratio >= 0.2) return '#84cc16';
  return '#22c55e'; // Low usage
};

export function TemporalAnalysisTab() {
  const [selectedMetric, setSelectedMetric] = useState('data_usage');
  const [selectedView, setSelectedView] = useState<'hourly' | 'peaks' | 'dod' | 'wow'>('hourly');

  // Fetch hourly breakdown
  const { data: hourlyData, isLoading: hourlyLoading } = useQuery({
    queryKey: ['insights', 'temporal', 'hourly', selectedMetric],
    queryFn: () => api.getHourlyBreakdown(selectedMetric, 7),
    enabled: selectedView === 'hourly',
  });

  // Fetch peak detection
  const { data: peakData, isLoading: peakLoading } = useQuery({
    queryKey: ['insights', 'temporal', 'peaks', selectedMetric],
    queryFn: () => api.getPeakDetection(selectedMetric, 7, 2.0),
    enabled: selectedView === 'peaks',
  });

  // Fetch day-over-day
  const { data: dodData, isLoading: dodLoading } = useQuery({
    queryKey: ['insights', 'temporal', 'dod', selectedMetric],
    queryFn: () => api.getDayOverDay(selectedMetric, 7),
    enabled: selectedView === 'dod',
  });

  // Fetch week-over-week
  const { data: wowData, isLoading: wowLoading } = useQuery({
    queryKey: ['insights', 'temporal', 'wow', selectedMetric],
    queryFn: () => api.getWeekOverWeek(selectedMetric, 4),
    enabled: selectedView === 'wow',
  });

  // Calculate max value for hourly heatmap
  const maxHourlyValue = hourlyData?.hourly_data
    ? Math.max(...hourlyData.hourly_data.map((h) => h.avg_value))
    : 1;

  return (
    <div className="space-y-6">
      {/* KPI Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <KPICard
          title="Peak Hours"
          value={hourlyData?.peak_hours?.length || '-'}
          color="warning"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
            </svg>
          }
          explainer={hourlyData?.peak_hours?.length ? `Hours: ${hourlyData.peak_hours.slice(0, 3).join(', ')}...` : 'N/A'}
        />
        <KPICard
          title="Low Activity Hours"
          value={hourlyData?.low_hours?.length || '-'}
          color="stellar"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
            </svg>
          }
          explainer={hourlyData?.low_hours?.length ? `Hours: ${hourlyData.low_hours.slice(0, 3).join(', ')}...` : 'N/A'}
        />
        <KPICard
          title="Day/Night Ratio"
          value={hourlyData?.day_night_ratio?.toFixed(1) || '-'}
          color="stellar"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
            </svg>
          }
          explainer="Daytime vs nighttime activity"
        />
        <KPICard
          title="Detected Peaks"
          value={peakData?.total_peaks || '-'}
          color={peakData && peakData.total_peaks > 5 ? 'danger' : 'aurora'}
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          }
          explainer="Significant spikes detected"
        />
      </div>

      {/* Metric Selector */}
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex gap-1 p-1 bg-slate-800/50 rounded-lg border border-slate-700/50">
          {METRIC_OPTIONS.map((metric) => (
            <button
              key={metric.id}
              onClick={() => setSelectedMetric(metric.id)}
              className={clsx(
                'px-4 py-2 text-sm font-medium rounded-md transition-all',
                selectedMetric === metric.id
                  ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                  : 'text-slate-400 hover:text-white'
              )}
            >
              {metric.label}
            </button>
          ))}
        </div>

        <div className="flex gap-1 p-1 bg-slate-800/50 rounded-lg border border-slate-700/50">
          {[
            { id: 'hourly', label: 'Hourly Pattern' },
            { id: 'peaks', label: 'Peak Detection' },
            { id: 'dod', label: 'Day over Day' },
            { id: 'wow', label: 'Week over Week' },
          ].map((view) => (
            <button
              key={view.id}
              onClick={() => setSelectedView(view.id as 'hourly' | 'peaks' | 'dod' | 'wow')}
              className={clsx(
                'px-4 py-2 text-sm font-medium rounded-md transition-all',
                selectedView === view.id
                  ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                  : 'text-slate-400 hover:text-white'
              )}
            >
              {view.label}
            </button>
          ))}
        </div>
      </div>

      {/* Hourly Breakdown View */}
      {selectedView === 'hourly' && (
        <div className="space-y-6">
          <Card title={<>Hour-of-Day Pattern <span className="text-slate-500 text-sm font-normal ml-2">Average values by hour</span></>}>
            {hourlyLoading ? (
              <div className="h-64 flex items-center justify-center text-slate-400">
                Loading hourly data...
              </div>
            ) : hourlyData?.hourly_data ? (
              <ResponsiveContainer width="100%" height={280}>
                <AreaChart data={hourlyData.hourly_data}>
                  <defs>
                    <linearGradient id="hourlyGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis
                    dataKey="hour"
                    stroke="#64748b"
                    fontSize={11}
                    tickLine={false}
                    tickFormatter={(val) => `${val}:00`}
                  />
                  <YAxis stroke="#64748b" fontSize={11} tickLine={false} axisLine={false} />
                  <Tooltip
                    contentStyle={{
                      background: '#1e293b',
                      border: '1px solid #334155',
                      borderRadius: '8px',
                      fontSize: '12px',
                    }}
                    formatter={(value: number, name: string) => {
                      if (name === 'avg_value') return [value.toFixed(2), 'Average'];
                      return [value.toFixed(2), name];
                    }}
                    labelFormatter={(val) => `${val}:00`}
                  />
                  <Area
                    type="monotone"
                    dataKey="avg_value"
                    stroke="#f59e0b"
                    strokeWidth={2}
                    fill="url(#hourlyGradient)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-64 flex items-center justify-center text-slate-400">
                No hourly data available
              </div>
            )}
          </Card>

          {/* Hour Heatmap Grid */}
          <Card title="24-Hour Activity Heatmap">
            {hourlyData?.hourly_data ? (
              <div className="space-y-4">
                <div className="grid grid-cols-12 gap-1">
                  {hourlyData.hourly_data.map((hour) => (
                    <div
                      key={hour.hour}
                      className="aspect-square rounded-md cursor-pointer transition-transform hover:scale-105 flex items-center justify-center"
                      style={{ backgroundColor: getHourColor(hour.avg_value, maxHourlyValue) }}
                      title={`${hour.hour}:00 - Avg: ${hour.avg_value.toFixed(2)}, Samples: ${hour.sample_count}`}
                    >
                      <span className="text-xs font-mono text-white/80">{hour.hour}</span>
                    </div>
                  ))}
                </div>
                <div className="flex items-center justify-center gap-4 text-xs text-slate-400">
                  <span className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded" style={{ backgroundColor: '#22c55e' }} />
                    Low
                  </span>
                  <span className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded" style={{ backgroundColor: '#84cc16' }} />
                  </span>
                  <span className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded" style={{ backgroundColor: '#f59e0b' }} />
                    Medium
                  </span>
                  <span className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded" style={{ backgroundColor: '#f97316' }} />
                  </span>
                  <span className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded" style={{ backgroundColor: '#ef4444' }} />
                    High
                  </span>
                </div>
              </div>
            ) : (
              <div className="h-32 flex items-center justify-center text-slate-400">
                No data available
              </div>
            )}
          </Card>
        </div>
      )}

      {/* Peak Detection View */}
      {selectedView === 'peaks' && (
        <Card title={<>Detected Usage Peaks <span className="text-slate-500 text-sm font-normal ml-2">Statistically significant spikes</span></>}>
          {peakLoading ? (
            <div className="h-64 flex items-center justify-center text-slate-400">
              Loading peak data...
            </div>
          ) : peakData?.peaks && peakData.peaks.length > 0 ? (
            <div className="space-y-4">
              {peakData.peaks.map((peak, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-4 rounded-lg bg-slate-800/50 border border-slate-700/30"
                >
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-lg bg-red-500/20 flex items-center justify-center">
                      <span className="text-lg font-bold text-red-400">#{index + 1}</span>
                    </div>
                    <div>
                      <div className="text-sm font-medium text-white">
                        {format(new Date(peak.timestamp), 'PPP p')}
                      </div>
                      <div className="text-xs text-slate-500">
                        Z-Score: {peak.z_score.toFixed(2)} | {peak.is_significant ? 'Significant' : 'Normal'}
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-bold font-mono text-amber-400">
                      {peak.value.toFixed(1)}
                    </div>
                    <div className="text-xs text-slate-500">Value</div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="h-64 flex flex-col items-center justify-center text-slate-400">
              <div className="w-16 h-16 mb-4 rounded-full bg-emerald-500/20 flex items-center justify-center">
                <svg className="w-8 h-8 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <p className="text-emerald-400 font-medium">No Significant Peaks Detected</p>
              <p className="text-sm mt-1">Usage patterns are stable</p>
            </div>
          )}
        </Card>
      )}

      {/* Day-over-Day View */}
      {selectedView === 'dod' && (
        <Card title={<>Day-over-Day Comparison <span className="text-slate-500 text-sm font-normal ml-2">Daily trend with change percentage</span></>}>
          {dodLoading ? (
            <div className="h-64 flex items-center justify-center text-slate-400">
              Loading daily data...
            </div>
          ) : dodData?.comparisons ? (
            <div className="space-y-6">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={dodData.comparisons}>
                  <XAxis
                    dataKey="date"
                    stroke="#64748b"
                    fontSize={11}
                    tickLine={false}
                    tickFormatter={(val) => format(new Date(val), 'MMM d')}
                  />
                  <YAxis stroke="#64748b" fontSize={11} tickLine={false} axisLine={false} />
                  <Tooltip
                    contentStyle={{
                      background: '#1e293b',
                      border: '1px solid #334155',
                      borderRadius: '8px',
                      fontSize: '12px',
                    }}
                    formatter={(value: number, name: string) => {
                      if (name === 'change_percent') return [`${value.toFixed(1)}%`, 'Change'];
                      return [value.toFixed(2), 'Value'];
                    }}
                    labelFormatter={(val) => format(new Date(val), 'PPP')}
                  />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {dodData.comparisons.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.change_percent > 0 ? '#ef4444' : entry.change_percent < 0 ? '#22c55e' : '#f59e0b'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>

              {/* Change Summary */}
              <div className="grid grid-cols-7 gap-2">
                {dodData.comparisons.map((day) => (
                  <div
                    key={day.date}
                    className="p-2 rounded-lg bg-slate-800/50 border border-slate-700/30 text-center"
                  >
                    <div className="text-xs text-slate-500 mb-1">
                      {format(new Date(day.date), 'EEE')}
                    </div>
                    <div
                      className={clsx(
                        'text-sm font-bold font-mono',
                        day.change_percent > 5 ? 'text-red-400' :
                        day.change_percent < -5 ? 'text-emerald-400' :
                        'text-slate-300'
                      )}
                    >
                      {day.change_percent > 0 ? '+' : ''}{day.change_percent.toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="h-64 flex items-center justify-center text-slate-400">
              No daily comparison data available
            </div>
          )}
        </Card>
      )}

      {/* Week-over-Week View */}
      {selectedView === 'wow' && (
        <Card title={<>Week-over-Week Comparison <span className="text-slate-500 text-sm font-normal ml-2">Weekly trend analysis</span></>}>
          {wowLoading ? (
            <div className="h-64 flex items-center justify-center text-slate-400">
              Loading weekly data...
            </div>
          ) : wowData?.comparisons ? (
            <div className="space-y-6">
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={wowData.comparisons}>
                  <XAxis
                    dataKey="week"
                    stroke="#64748b"
                    fontSize={11}
                    tickLine={false}
                    tickFormatter={(_val, index) => {
                      const entry = wowData.comparisons[index];
                      return entry ? `W${entry.week}` : '';
                    }}
                  />
                  <YAxis stroke="#64748b" fontSize={11} tickLine={false} axisLine={false} />
                  <Tooltip
                    contentStyle={{
                      background: '#1e293b',
                      border: '1px solid #334155',
                      borderRadius: '8px',
                      fontSize: '12px',
                    }}
                    formatter={(value: number, name: string) => {
                      if (name === 'change_percent') return [`${value.toFixed(1)}%`, 'WoW Change'];
                      return [value.toFixed(2), 'Value'];
                    }}
                    labelFormatter={(_val, payload) => {
                      const entry = payload?.[0]?.payload;
                      return entry ? `${entry.year} Week ${entry.week}` : '';
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#f59e0b"
                    strokeWidth={2}
                    dot={{ fill: '#f59e0b', strokeWidth: 2 }}
                  />
                </LineChart>
              </ResponsiveContainer>

              {/* Weekly Cards */}
              <div className="grid grid-cols-4 gap-4">
                {wowData.comparisons.map((week, index) => (
                  <div
                    key={`${week.year}-${week.week}`}
                    className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/30 text-center"
                  >
                    <div className="text-xs text-slate-500 mb-1">
                      {week.year} W{week.week}
                    </div>
                    <div className="text-xl font-bold font-mono text-white mb-1">
                      {week.value.toFixed(0)}
                    </div>
                    <div
                      className={clsx(
                        'text-sm font-mono',
                        week.change_percent > 5 ? 'text-red-400' :
                        week.change_percent < -5 ? 'text-emerald-400' :
                        'text-slate-400'
                      )}
                    >
                      {index === 0 ? '-' : `${week.change_percent > 0 ? '+' : ''}${week.change_percent.toFixed(1)}%`}
                    </div>
                    <div className="text-xs text-slate-600 mt-1">
                      {week.sample_count.toLocaleString()} samples
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="h-64 flex items-center justify-center text-slate-400">
              No weekly comparison data available
            </div>
          )}
        </Card>
      )}
    </div>
  );
}

export default TemporalAnalysisTab;
