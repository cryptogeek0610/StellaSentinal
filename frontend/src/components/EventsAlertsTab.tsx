/**
 * Events & Alerts Tab Component
 *
 * Displays system event timeline, alert dashboard, and event statistics.
 */

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { Card } from './Card';
import { KPICard } from './KPICard';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { format, formatDistanceToNow } from 'date-fns';
import clsx from 'clsx';

// Severity colors
const SEVERITY_COLORS: Record<string, string> = {
  Critical: '#ef4444',
  High: '#f97316',
  Error: '#ef4444',
  Warning: '#f59e0b',
  Medium: '#f59e0b',
  Info: '#3b82f6',
  Low: '#64748b',
};

const SEVERITY_BG: Record<string, string> = {
  Critical: 'bg-red-500/20 border-red-500/30 text-red-400',
  High: 'bg-orange-500/20 border-orange-500/30 text-orange-400',
  Error: 'bg-red-500/20 border-red-500/30 text-red-400',
  Warning: 'bg-amber-500/20 border-amber-500/30 text-amber-400',
  Medium: 'bg-amber-500/20 border-amber-500/30 text-amber-400',
  Info: 'bg-blue-500/20 border-blue-500/30 text-blue-400',
  Low: 'bg-slate-500/20 border-slate-500/30 text-slate-400',
};

export function EventsAlertsTab() {
  const [selectedView, setSelectedView] = useState<'timeline' | 'alerts' | 'stats'>('timeline');
  const [severityFilter, setSeverityFilter] = useState<string>('');
  const [eventClassFilter, setEventClassFilter] = useState<string>('');

  // Fetch event timeline
  const { data: eventTimeline, isLoading: timelineLoading, refetch: refetchTimeline } = useQuery({
    queryKey: ['insights', 'events', 'timeline', severityFilter, eventClassFilter],
    queryFn: () => api.getEventTimeline({
      page: 1,
      page_size: 50,
      severity: severityFilter || undefined,
      event_class: eventClassFilter || undefined,
      hours_back: 24,
    }),
    enabled: selectedView === 'timeline',
  });

  // Fetch alert summary
  const { data: alertSummary, isLoading: alertsLoading } = useQuery({
    queryKey: ['insights', 'alerts', 'summary'],
    queryFn: () => api.getAlertSummary(24),
    enabled: selectedView === 'alerts',
  });

  // Fetch alert trends
  const { data: alertTrends, isLoading: trendsLoading } = useQuery({
    queryKey: ['insights', 'alerts', 'trends'],
    queryFn: () => api.getAlertTrends(7, 'daily'),
    enabled: selectedView === 'alerts',
  });

  // Fetch event statistics
  const { data: eventStats, isLoading: statsLoading } = useQuery({
    queryKey: ['insights', 'events', 'statistics'],
    queryFn: () => api.getEventStatistics(24),
    enabled: selectedView === 'stats',
  });

  // Prepare alert pie chart data
  const alertPieData = alertSummary?.by_severity
    ? Object.entries(alertSummary.by_severity).map(([severity, count]) => ({
        name: severity,
        value: count,
        color: SEVERITY_COLORS[severity] || '#64748b',
      }))
    : [];

  return (
    <div className="space-y-6">
      {/* KPI Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <KPICard
          title="Active Alerts"
          value={alertSummary?.total_active || 0}
          color={alertSummary && alertSummary.total_active > 20 ? 'danger' : alertSummary && alertSummary.total_active > 10 ? 'warning' : 'stellar'}
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
            </svg>
          }
          explainer="Requiring attention"
        />
        <KPICard
          title="Acknowledged"
          value={alertSummary?.total_acknowledged || 0}
          color="stellar"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          }
          explainer="Being worked on"
        />
        <KPICard
          title="Resolved (24h)"
          value={alertSummary?.total_resolved || 0}
          color="aurora"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 13l4 4L19 7" />
            </svg>
          }
          explainer="Successfully closed"
        />
        <KPICard
          title="Avg Resolution"
          value={`${alertSummary?.avg_resolution_time_minutes?.toFixed(0) || '-'} min`}
          color="stellar"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          }
          explainer="Time to resolve"
        />
      </div>

      {/* View Selector */}
      <div className="flex gap-2 p-1 bg-slate-800/50 rounded-lg border border-slate-700/50">
        {[
          { id: 'timeline', label: 'Event Timeline', icon: (
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
            </svg>
          ) },
          { id: 'alerts', label: 'Alert Dashboard', icon: (
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
            </svg>
          ) },
          { id: 'stats', label: 'Statistics', icon: (
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          ) },
        ].map((view) => (
          <button
            key={view.id}
            onClick={() => setSelectedView(view.id as 'timeline' | 'alerts' | 'stats')}
            className={clsx(
              'flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md transition-all',
              selectedView === view.id
                ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                : 'text-slate-400 hover:text-white'
            )}
          >
            {view.icon}
            {view.label}
          </button>
        ))}
      </div>

      {/* Timeline View */}
      {selectedView === 'timeline' && (
        <Card
          title={
            <div className="flex items-center justify-between w-full">
              <span>Event Timeline <span className="text-slate-500 text-sm font-normal ml-2">{eventTimeline?.total || 0} events in last 24 hours</span></span>
              <div className="flex items-center gap-2">
                <select
                  value={severityFilter}
                  onChange={(e) => setSeverityFilter(e.target.value)}
                  className="px-3 py-1.5 text-xs rounded-md bg-slate-800 border border-slate-700 text-white"
                >
                  <option value="">All Severities</option>
                  <option value="Critical">Critical</option>
                  <option value="Error">Error</option>
                  <option value="Warning">Warning</option>
                  <option value="Info">Info</option>
                </select>
                <select
                  value={eventClassFilter}
                  onChange={(e) => setEventClassFilter(e.target.value)}
                  className="px-3 py-1.5 text-xs rounded-md bg-slate-800 border border-slate-700 text-white"
                >
                  <option value="">All Classes</option>
                  <option value="Device">Device</option>
                  <option value="Network">Network</option>
                  <option value="Application">Application</option>
                  <option value="Security">Security</option>
                  <option value="System">System</option>
                </select>
                <button
                  onClick={() => refetchTimeline()}
                  className="px-3 py-1.5 text-xs rounded-md bg-amber-500/20 text-amber-400 border border-amber-500/30 hover:bg-amber-500/30"
                >
                  Refresh
                </button>
              </div>
            </div>
          }
        >
          {timelineLoading ? (
            <div className="h-96 flex items-center justify-center text-slate-400">
              Loading events...
            </div>
          ) : eventTimeline?.events && eventTimeline.events.length > 0 ? (
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {eventTimeline.events.map((event) => (
                <div
                  key={event.log_id}
                  className="flex items-start gap-3 p-3 rounded-lg bg-slate-800/50 border border-slate-700/30 hover:border-slate-600/50 transition-colors"
                >
                  <div className="flex-shrink-0">
                    <span
                      className={clsx(
                        'inline-block px-2 py-0.5 text-xs font-medium rounded border',
                        SEVERITY_BG[event.severity] || SEVERITY_BG.Info
                      )}
                    >
                      {event.severity}
                    </span>
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm text-white line-clamp-2">{event.message}</div>
                    <div className="flex items-center gap-4 mt-1 text-xs text-slate-500">
                      <span className="px-1.5 py-0.5 rounded bg-slate-700/50">{event.event_class}</span>
                      {event.device_id && <span>Device: {event.device_id}</span>}
                      <span>{formatDistanceToNow(new Date(event.timestamp), { addSuffix: true })}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="h-64 flex items-center justify-center text-slate-400">
              No events found
            </div>
          )}
        </Card>
      )}

      {/* Alerts View */}
      {selectedView === 'alerts' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Alert Breakdown Pie */}
          <Card title="Alert Severity Distribution">
            {alertsLoading ? (
              <div className="h-64 flex items-center justify-center text-slate-400">
                Loading alerts...
              </div>
            ) : alertPieData.length > 0 ? (
              <ResponsiveContainer width="100%" height={240}>
                <PieChart>
                  <Pie
                    data={alertPieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={90}
                    paddingAngle={2}
                    dataKey="value"
                    label={({ name, value }) => `${name}: ${value}`}
                    labelLine={false}
                  >
                    {alertPieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      background: '#1e293b',
                      border: '1px solid #334155',
                      borderRadius: '8px',
                      fontSize: '12px',
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-64 flex items-center justify-center text-slate-400">
                No alert data available
              </div>
            )}
          </Card>

          {/* Alert Trends */}
          <Card title={<>Alert Trends <span className="text-slate-500 text-sm font-normal ml-2">Last 7 days</span></>}>
            {trendsLoading ? (
              <div className="h-64 flex items-center justify-center text-slate-400">
                Loading trends...
              </div>
            ) : alertTrends?.trends ? (
              <ResponsiveContainer width="100%" height={240}>
                <LineChart
                  data={alertTrends.trends.filter(t => t.severity === 'Critical' || t.severity === 'High')}
                >
                  <XAxis
                    dataKey="timestamp"
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
                    labelFormatter={(val) => format(new Date(val), 'PPP')}
                  />
                  <Line
                    type="monotone"
                    dataKey="count"
                    stroke="#ef4444"
                    strokeWidth={2}
                    dot={false}
                    name="Critical/High"
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-64 flex items-center justify-center text-slate-400">
                No trend data available
              </div>
            )}
          </Card>

          {/* Top Alerts */}
          <Card title="Top Alert Types" className="lg:col-span-2">
            {alertSummary?.by_alert_name && alertSummary.by_alert_name.length > 0 ? (
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                {alertSummary.by_alert_name.slice(0, 5).map((alert) => (
                  <div
                    key={alert.name}
                    className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/30 text-center"
                  >
                    <div className="text-2xl font-bold font-mono text-amber-400 mb-1">
                      {alert.count}
                    </div>
                    <div className="text-sm text-slate-400">{alert.name}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="h-32 flex items-center justify-center text-slate-400">
                No alert types available
              </div>
            )}
          </Card>

          {/* Recent Alerts */}
          <Card title="Recent Alerts" className="lg:col-span-2">
            {alertSummary?.recent_alerts && alertSummary.recent_alerts.length > 0 ? (
              <div className="space-y-2">
                {alertSummary.recent_alerts.map((alert) => (
                  <div
                    key={alert.alert_id}
                    className="flex items-center justify-between p-3 rounded-lg bg-slate-800/50 border border-slate-700/30"
                  >
                    <div className="flex items-center gap-3">
                      <span
                        className={clsx(
                          'px-2 py-0.5 text-xs font-medium rounded border',
                          SEVERITY_BG[alert.severity] || SEVERITY_BG.Medium
                        )}
                      >
                        {alert.severity}
                      </span>
                      <div>
                        <div className="text-sm font-medium text-white">{alert.alert_name}</div>
                        <div className="text-xs text-slate-500">Device: {alert.device_id || 'N/A'}</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <span
                        className={clsx(
                          'px-2 py-0.5 text-xs font-medium rounded',
                          alert.status === 'Active' ? 'bg-red-500/20 text-red-400' :
                          alert.status === 'Acknowledged' ? 'bg-amber-500/20 text-amber-400' :
                          'bg-emerald-500/20 text-emerald-400'
                        )}
                      >
                        {alert.status}
                      </span>
                      {alert.set_datetime && (
                        <div className="text-xs text-slate-500 mt-1">
                          {formatDistanceToNow(new Date(alert.set_datetime), { addSuffix: true })}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="h-32 flex items-center justify-center text-slate-400">
                No recent alerts
              </div>
            )}
          </Card>
        </div>
      )}

      {/* Statistics View */}
      {selectedView === 'stats' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card title={<>Event Statistics <span className="text-slate-500 text-sm font-normal ml-2">Last 24 hours</span></>}>
            {statsLoading ? (
              <div className="h-48 flex items-center justify-center text-slate-400">
                Loading statistics...
              </div>
            ) : eventStats ? (
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/30 text-center">
                  <div className="text-3xl font-bold font-mono text-white mb-1">
                    {eventStats.total_events.toLocaleString()}
                  </div>
                  <div className="text-sm text-slate-400">Total Events</div>
                </div>
                <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/30 text-center">
                  <div className="text-3xl font-bold font-mono text-amber-400 mb-1">
                    {eventStats.events_per_day.toLocaleString()}
                  </div>
                  <div className="text-sm text-slate-400">Per Day Avg</div>
                </div>
                <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/30 text-center col-span-2">
                  <div className="text-3xl font-bold font-mono text-blue-400 mb-1">
                    {eventStats.unique_devices}
                  </div>
                  <div className="text-sm text-slate-400">Unique Devices</div>
                </div>
              </div>
            ) : (
              <div className="h-48 flex items-center justify-center text-slate-400">
                No statistics available
              </div>
            )}
          </Card>

          <Card title="Events by Class">
            {eventStats?.top_event_classes && eventStats.top_event_classes.length > 0 ? (
              <div className="space-y-3">
                {eventStats.top_event_classes.map((eventClass) => {
                  const maxCount = eventStats.top_event_classes[0]?.count || 1;
                  const percentage = (eventClass.count / maxCount) * 100;
                  return (
                    <div key={eventClass.class}>
                      <div className="flex items-center justify-between text-sm mb-1">
                        <span className="text-white">{eventClass.class}</span>
                        <span className="text-slate-400 font-mono">{eventClass.count.toLocaleString()}</span>
                      </div>
                      <div className="h-2 bg-slate-700/50 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-amber-500 rounded-full transition-all"
                          style={{ width: `${percentage}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="h-48 flex items-center justify-center text-slate-400">
                No event class data available
              </div>
            )}
          </Card>

          {/* Severity Distribution from Timeline */}
          {eventTimeline?.severity_distribution && (
            <Card title="Severity Distribution" className="lg:col-span-2">
              <div className="grid grid-cols-4 gap-4">
                {Object.entries(eventTimeline.severity_distribution).map(([severity, count]) => (
                  <div
                    key={severity}
                    className={clsx(
                      'p-4 rounded-lg border text-center',
                      SEVERITY_BG[severity]?.replace('text-', 'border-') || 'bg-slate-700/30 border-slate-600/30'
                    )}
                  >
                    <div className="text-2xl font-bold font-mono mb-1">{count}</div>
                    <div className="text-sm">{severity}</div>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      )}
    </div>
  );
}

export default EventsAlertsTab;
