import { useEffect, useState } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { format } from 'date-fns';
import { Card } from '../components/Card';
import { motion } from 'framer-motion';

function DeviceDetail() {
  const { id } = useParams<{ id: string }>();
  const deviceId = parseInt(id || '0');
  const refreshIntervalSeconds = 10;
  const [refreshIn, setRefreshIn] = useState(refreshIntervalSeconds);
  const navigate = useNavigate();

  const { data: device, isLoading, isFetching } = useQuery({
    queryKey: ['device', deviceId],
    queryFn: () => api.getDevice(deviceId),
    enabled: !!deviceId,
    refetchInterval: refreshIntervalSeconds * 1000,
  });

  useEffect(() => {
    setRefreshIn(refreshIntervalSeconds);
    const interval = setInterval(() => {
      setRefreshIn((prev) => (prev <= 1 ? refreshIntervalSeconds : prev - 1));
    }, 1000);
    return () => clearInterval(interval);
  }, [deviceId, refreshIntervalSeconds]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="relative w-16 h-16 mx-auto mb-4">
            <div className="absolute inset-0 rounded-full border-2 border-cyan-400/20"></div>
            <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-cyan-400 animate-spin"></div>
          </div>
          <p className="text-slate-400 font-mono text-sm">Loading device details...</p>
        </div>
      </div>
    );
  }

  if (!device) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-red-500/20 flex items-center justify-center">
            <svg className="w-8 h-8 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </div>
          <p className="text-red-500 font-medium">Device not found</p>
        </div>
      </div>
    );
  }

  const InfoRow = ({ label, value }: { label: string; value: string | number }) => (
    <div className="flex justify-between py-2 border-b border-slate-700/30 last:border-0">
      <span className="text-sm text-slate-500">{label}</span>
      <span className="text-sm font-mono text-slate-200">{value}</span>
    </div>
  );

  const getScoreSeverity = (score: number) => {
    if (score <= -0.7) return { color: 'text-red-500', badge: 'badge-critical' };
    if (score <= -0.5) return { color: 'text-orange-400', badge: 'badge-warning' };
    return { color: 'text-cyan-400', badge: 'badge-info' };
  };

  const statusBadgeClass = (() => {
    const status = (device.status || '').toLowerCase();
    if (!status) return 'badge-neutral';
    if (status.includes('offline') || status.includes('error')) return 'badge-critical';
    if (status.includes('idle') || status.includes('warning')) return 'badge-warning';
    return 'badge-success';
  })();

  return (
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      {/* Device Header */}
      <div className="glass-panel p-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <Link
              to="/investigations"
              className="p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded-lg transition-colors"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
            </Link>
            <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-emerald-500/20 to-cyan-400/20 flex items-center justify-center">
              <svg
                className="h-8 w-8 text-emerald-500"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M12 18h.01M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z"
                />
              </svg>
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">
                {device.device_name || `Device ${device.device_id}`}
              </h1>
              <div className="flex items-center gap-3 mt-2">
                <span className={statusBadgeClass}>
                  {device.status || 'Unknown'}
                </span>
              </div>
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            <button
              className="btn-secondary text-sm opacity-50 cursor-not-allowed"
              disabled
              title="Remote control functionality coming soon"
            >
              <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                />
              </svg>
              Remote Control
            </button>
            <button
              className="btn-secondary text-sm opacity-50 cursor-not-allowed"
              disabled
              title="Check-in functionality coming soon"
            >
              <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                />
              </svg>
              Check-in
            </button>
            <button
              className="btn-primary text-sm opacity-50 cursor-not-allowed"
              disabled
              title="Actions menu coming soon"
            >
              Actions
            </button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Device Info */}
        <Card title="Device Information" accent="cyan">
          <div className="space-y-1">
            <InfoRow label="Device Name" value={device.device_name || 'Unknown'} />
            <InfoRow label="Device ID" value={device.device_id} />
            <InfoRow label="Model" value={device.device_model || 'Unknown'} />
            <InfoRow label="Location" value={device.location || 'Unknown'} />
            <InfoRow label="Status" value={device.status || 'Unknown'} />
            <InfoRow
              label="Last Seen"
              value={
                device.last_seen
                  ? format(new Date(device.last_seen), 'yyyy-MM-dd HH:mm:ss')
                  : 'Unknown'
              }
            />
            <InfoRow label="Total Anomalies" value={device.anomaly_count} />
            <InfoRow label="OS Version" value={device.os_version || 'N/A'} />
            <InfoRow label="Agent Version" value={device.agent_version || 'N/A'} />
          </div>
        </Card>

        {/* Telemetry Placeholder */}
        {/* Telemetry Snapshot */}
        <Card
          title={
            <div className="flex items-center justify-between w-full">
              <span className="telemetry-label">Telemetry Snapshot</span>
              <span className="flex items-center gap-1 text-[10px] text-slate-500 font-mono">
                <span className={`h-1.5 w-1.5 rounded-full ${isFetching ? 'bg-emerald-400 animate-pulse' : 'bg-slate-500'}`} />
                refresh in {refreshIn}s
              </span>
            </div>
          }
          accent="green"
        >
          {device.battery_level !== undefined ? (
            <div className="space-y-4">
              {/* Battery & Signal Header */}
              <div className="flex items-center justify-between pb-4 border-b border-slate-700/30">
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-lg ${device.is_charging ? 'bg-green-500/20 text-green-400' : 'bg-slate-700/50 text-slate-400'}`}>
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                  <div>
                    <div className="text-sm text-slate-400">Battery</div>
                    <div className="font-mono font-medium text-white">{device.battery_level}% {device.is_charging && '(Charging)'}</div>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-lg ${(device.wifi_signal || -100) > -70 ? 'bg-blue-500/20 text-blue-400' : 'bg-amber-500/20 text-amber-400'
                    }`}>
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.111 16.404a5.5 5.5 0 017.778 0M12 20h.01m-7.08-7.071c3.904-3.905 10.236-3.905 14.141 0M1.394 9.393c5.857-5.857 15.355-5.857 21.213 0" />
                    </svg>
                  </div>
                  <div className="text-right">
                    <div className="text-sm text-slate-400">Signal</div>
                    <div className="font-mono font-medium text-white">{device.wifi_signal} dBm</div>
                  </div>
                </div>
              </div>

              {/* Resource Usage Bars */}
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-500">Storage Used</span>
                    <span className="text-slate-300 font-mono">{device.storage_used}%</span>
                  </div>
                  <div className="h-1.5 bg-slate-700/30 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }} animate={{ width: `${device.storage_used}%` }}
                      className={`h-full rounded-full ${(device.storage_used || 0) > 90 ? 'bg-red-500' : 'bg-purple-500'}`}
                    />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-500">Memory Usage</span>
                    <span className="text-slate-300 font-mono">{device.memory_usage}%</span>
                  </div>
                  <div className="h-1.5 bg-slate-700/30 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }} animate={{ width: `${device.memory_usage}%` }}
                      className={`h-full rounded-full ${(device.memory_usage || 0) > 85 ? 'bg-orange-500' : 'bg-blue-500'}`}
                    />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-500">CPU Load</span>
                    <span className="text-slate-300 font-mono">{device.cpu_load}%</span>
                  </div>
                  <div className="h-1.5 bg-slate-700/30 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }} animate={{ width: `${device.cpu_load}%` }}
                      className={`h-full rounded-full ${(device.cpu_load || 0) > 80 ? 'bg-amber-500' : 'bg-cyan-500'}`}
                    />
                  </div>
                </div>
              </div>

              {/* Extra Telemetry Points */}
              <div className="grid grid-cols-2 gap-3 pt-3 border-t border-slate-700/30">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-500">Charging</span>
                  <span className="font-mono text-slate-200">{device.is_charging ? 'Yes' : 'No'}</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-500">Signal Quality</span>
                  <span className="font-mono text-slate-200">
                    {(device.wifi_signal || -100) > -60 ? 'Strong' : (device.wifi_signal || -100) > -75 ? 'Moderate' : 'Weak'}
                  </span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-500">Storage Free</span>
                  <span className="font-mono text-slate-200">
                    {Math.max(0, 100 - (device.storage_used || 0))}%
                  </span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-500">Memory Free</span>
                  <span className="font-mono text-slate-200">
                    {Math.max(0, 100 - (device.memory_usage || 0))}%
                  </span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-500">CPU Headroom</span>
                  <span className="font-mono text-slate-200">
                    {Math.max(0, 100 - (device.cpu_load || 0))}%
                  </span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-500">Last Telemetry</span>
                  <span className="font-mono text-slate-200">
                    {device.last_seen ? format(new Date(device.last_seen), 'HH:mm:ss') : 'N/A'}
                  </span>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-sm text-slate-500">
              Hardware and performance telemetry are not available for this device view yet.
            </div>
          )}
        </Card>
      </div>

      {/* Recent Anomalies */}
      <Card title="Recent Anomalies (Last 30 days)" noPadding>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-700/50">
                <th className="table-header">Time</th>
                <th className="table-header">Score</th>
                <th className="table-header">Status</th>
                <th className="table-header text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {device.recent_anomalies.length > 0 ? (
                device.recent_anomalies.map((anomaly, index) => {
                  const severity = getScoreSeverity(anomaly.anomaly_score);
                  const anomalyDetailPath = `/investigations/${anomaly.id}`;
                  return (
                    <motion.tr
                      key={anomaly.id}
                      className="table-row cursor-pointer"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.05 }}
                      role="link"
                      tabIndex={0}
                      onClick={(event) => {
                        const target = event.target as HTMLElement;
                        if (target.closest('a')) return;
                        navigate(anomalyDetailPath);
                      }}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter' || event.key === ' ') {
                          event.preventDefault();
                          navigate(anomalyDetailPath);
                        }
                      }}
                    >
                      <td className="table-cell font-mono text-slate-300">
                        {format(new Date(anomaly.timestamp), 'MMM dd, HH:mm')}
                      </td>
                      <td className="table-cell">
                        <span className={`font-mono font-bold ${severity.color}`}>
                          {anomaly.anomaly_score.toFixed(3)}
                        </span>
                      </td>
                      <td className="table-cell">
                        <span
                          className={
                            anomaly.status === 'open'
                              ? 'badge-critical'
                              : anomaly.status === 'resolved'
                                ? 'badge-success'
                                : 'badge-warning'
                          }
                        >
                          {anomaly.status}
                        </span>
                      </td>
                      <td className="table-cell text-right">
                        <Link
                          to={anomalyDetailPath}
                          className="text-cyan-400 hover:text-cyan-300 text-sm font-medium transition-colors"
                        >
                          View Details →
                        </Link>
                      </td>
                    </motion.tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={4} className="px-6 py-12 text-center">
                    <div className="flex flex-col items-center">
                      <div className="w-12 h-12 mb-3 rounded-full bg-emerald-500/10 flex items-center justify-center">
                        <svg
                          className="w-6 h-6 text-emerald-500"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M5 13l4 4L19 7"
                          />
                        </svg>
                      </div>
                      <p className="text-slate-400 font-medium">No anomalies found</p>
                      <p className="text-sm text-slate-600 mt-1">
                        This device is operating normally
                      </p>
                    </div>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </motion.div>
  );
}

export default DeviceDetail;
