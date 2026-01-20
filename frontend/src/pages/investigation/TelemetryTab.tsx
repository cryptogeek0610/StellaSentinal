/**
 * Telemetry Tab content component.
 */

import { Link } from 'react-router-dom';
import { Card } from '../../components/Card';
import { DeviceIcon } from './Icons';
import { formatDataSize } from './investigationUtils';

interface TelemetryTabProps {
  anomaly: {
    device_id: number;
    device_name?: string | null;
    total_battery_level_drop?: number | null;
    total_free_storage_kb?: number | null;
    wifi_signal_strength?: number | null;
    disconnect_count?: number | null;
    download?: number | null;
    upload?: number | null;
    offline_time?: number | null;
    connection_time?: number | null;
  };
}

export function TelemetryTab({ anomaly }: TelemetryTabProps) {
  return (
    <>
      {/* Device Header Card */}
      <Card title={<span className="flex items-center gap-2"><DeviceIcon /> Device Overview</span>}>
        <div className="flex items-center justify-between p-4 rounded-xl bg-gradient-to-r from-slate-800/50 to-slate-800/30 border border-slate-700/50 mb-4">
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 rounded-xl bg-amber-500/20 flex items-center justify-center">
              <DeviceIcon />
            </div>
            <div>
              <p className="text-lg font-bold text-white">{anomaly.device_name || `Device #${anomaly.device_id}`}</p>
              <p className="text-sm text-slate-400">Telemetry snapshot at anomaly detection</p>
            </div>
          </div>
          <Link
            to={`/devices/${anomaly.device_id}`}
            className="px-4 py-2 text-sm font-medium text-amber-400 bg-amber-500/10 rounded-lg hover:bg-amber-500/20 transition-colors"
          >
            View Full Profile
          </Link>
        </div>

        {/* Quick Health Indicators */}
        <div className="grid grid-cols-4 gap-3">
          {/* Battery Health */}
          <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/50">
            <div className="flex items-center gap-2 mb-2">
              <svg className="w-4 h-4 text-orange-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8V5a1 1 0 00-1-1H8a1 1 0 00-1 1v3M7 8h10l1 12H6L7 8z" />
              </svg>
              <span className="text-xs text-slate-500">Battery</span>
            </div>
            <p className={`text-xl font-bold font-mono ${
              (anomaly.total_battery_level_drop ?? 0) > 30 ? 'text-red-400' :
              (anomaly.total_battery_level_drop ?? 0) > 15 ? 'text-orange-400' : 'text-emerald-400'
            }`}>
              {anomaly.total_battery_level_drop != null ? `-${anomaly.total_battery_level_drop}%` : 'N/A'}
            </p>
            <p className="text-[10px] text-slate-600 mt-1">24h drain</p>
          </div>

          {/* Storage Health */}
          <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/50">
            <div className="flex items-center gap-2 mb-2">
              <svg className="w-4 h-4 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
              </svg>
              <span className="text-xs text-slate-500">Storage</span>
            </div>
            <p className="text-xl font-bold font-mono text-cyan-400">
              {anomaly.total_free_storage_kb != null ? `${(anomaly.total_free_storage_kb / 1024 / 1024).toFixed(1)}GB` : 'N/A'}
            </p>
            <p className="text-[10px] text-slate-600 mt-1">free space</p>
          </div>

          {/* Network Health */}
          <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/50">
            <div className="flex items-center gap-2 mb-2">
              <svg className="w-4 h-4 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.111 16.404a5.5 5.5 0 017.778 0M12 20h.01m-7.08-7.071c3.904-3.905 10.236-3.905 14.141 0M1.394 9.393c5.857-5.857 15.355-5.857 21.213 0" />
              </svg>
              <span className="text-xs text-slate-500">WiFi Signal</span>
            </div>
            <p className={`text-xl font-bold font-mono ${
              (anomaly.wifi_signal_strength ?? -100) > -50 ? 'text-emerald-400' :
              (anomaly.wifi_signal_strength ?? -100) > -70 ? 'text-amber-400' : 'text-red-400'
            }`}>
              {anomaly.wifi_signal_strength != null ? `${anomaly.wifi_signal_strength}dBm` : 'N/A'}
            </p>
            <p className="text-[10px] text-slate-600 mt-1">signal strength</p>
          </div>

          {/* Connectivity Health */}
          <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/50">
            <div className="flex items-center gap-2 mb-2">
              <svg className="w-4 h-4 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 5.636a9 9 0 010 12.728m0 0l-2.829-2.829m2.829 2.829L21 21M15.536 8.464a5 5 0 010 7.072m0 0l-2.829-2.829m-4.243 2.829a5 5 0 01-7.072 0" />
              </svg>
              <span className="text-xs text-slate-500">Disconnects</span>
            </div>
            <p className={`text-xl font-bold font-mono ${
              (anomaly.disconnect_count ?? 0) > 10 ? 'text-red-400' :
              (anomaly.disconnect_count ?? 0) > 3 ? 'text-orange-400' : 'text-emerald-400'
            }`}>
              {anomaly.disconnect_count != null ? anomaly.disconnect_count : 'N/A'}
            </p>
            <p className="text-[10px] text-slate-600 mt-1">in period</p>
          </div>
        </div>
      </Card>

      {/* Network & Data Transfer */}
      <Card title={
        <span className="flex items-center gap-2">
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
          </svg>
          Network & Data Transfer
        </span>
      }>
        <div className="grid grid-cols-2 gap-4 mb-4">
          {/* Download */}
          <div className="p-4 rounded-xl bg-gradient-to-br from-emerald-500/10 to-emerald-500/5 border border-emerald-500/20">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs text-emerald-400 font-medium">DOWNLOAD</span>
              <svg className="w-4 h-4 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
              </svg>
            </div>
            <p className="text-3xl font-bold font-mono text-emerald-400">
              {formatDataSize(anomaly.download).value}
            </p>
            <p className="text-sm text-slate-500 mt-1">{formatDataSize(anomaly.download).unit} transferred</p>
            {anomaly.download != null && anomaly.download > 100 && (
              <div className="mt-2 px-2 py-1 rounded bg-emerald-500/10 text-xs text-emerald-400">
                High data usage detected
              </div>
            )}
          </div>

          {/* Upload */}
          <div className="p-4 rounded-xl bg-gradient-to-br from-purple-500/10 to-purple-500/5 border border-purple-500/20">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs text-purple-400 font-medium">UPLOAD</span>
              <svg className="w-4 h-4 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
              </svg>
            </div>
            <p className="text-3xl font-bold font-mono text-purple-400">
              {formatDataSize(anomaly.upload).value}
            </p>
            <p className="text-sm text-slate-500 mt-1">{formatDataSize(anomaly.upload).unit} transferred</p>
            {anomaly.upload != null && anomaly.upload > 50 && (
              <div className="mt-2 px-2 py-1 rounded bg-purple-500/10 text-xs text-purple-400">
                High upload activity
              </div>
            )}
          </div>
        </div>

        {/* Data Transfer Summary Bar */}
        <div className="p-3 rounded-lg bg-slate-800/30 border border-slate-700/50">
          <div className="flex items-center justify-between text-xs text-slate-500 mb-2">
            <span>Total Data Transfer</span>
            <span className="font-mono text-slate-400">
              {formatDataSize((anomaly.download ?? 0) + (anomaly.upload ?? 0)).value} {formatDataSize((anomaly.download ?? 0) + (anomaly.upload ?? 0)).unit}
            </span>
          </div>
          <div className="h-2 bg-slate-700/50 rounded-full overflow-hidden flex">
            <div
              className="h-full bg-emerald-500/70"
              style={{
                width: `${Math.max(5, ((anomaly.download ?? 0) / ((anomaly.download ?? 0) + (anomaly.upload ?? 1))) * 100)}%`
              }}
            />
            <div
              className="h-full bg-purple-500/70"
              style={{
                width: `${Math.max(5, ((anomaly.upload ?? 0) / ((anomaly.download ?? 1) + (anomaly.upload ?? 0))) * 100)}%`
              }}
            />
          </div>
          <div className="flex justify-between mt-2 text-[10px]">
            <span className="text-emerald-400">Download</span>
            <span className="text-purple-400">Upload</span>
          </div>
        </div>
      </Card>

      {/* Connectivity Status */}
      <Card title={
        <span className="flex items-center gap-2">
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          Connectivity & Uptime
        </span>
      }>
        <div className="grid grid-cols-3 gap-4 mb-4">
          {/* Offline Time */}
          <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/50">
            <p className="text-xs text-slate-500 mb-2">Offline Duration</p>
            <p className={`text-2xl font-bold font-mono ${
              (anomaly.offline_time ?? 0) > 60 ? 'text-red-400' :
              (anomaly.offline_time ?? 0) > 15 ? 'text-orange-400' : 'text-emerald-400'
            }`}>
              {anomaly.offline_time != null ? anomaly.offline_time.toFixed(0) : '0'}
            </p>
            <p className="text-xs text-slate-600">minutes</p>
          </div>

          {/* Connection Time */}
          <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/50">
            <p className="text-xs text-slate-500 mb-2">Avg Connection Time</p>
            <p className="text-2xl font-bold font-mono text-cyan-400">
              {anomaly.connection_time != null ? anomaly.connection_time.toFixed(1) : '0'}
            </p>
            <p className="text-xs text-slate-600">seconds</p>
          </div>

          {/* Disconnect Events */}
          <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/50">
            <p className="text-xs text-slate-500 mb-2">Disconnect Events</p>
            <p className={`text-2xl font-bold font-mono ${
              (anomaly.disconnect_count ?? 0) > 10 ? 'text-red-400' :
              (anomaly.disconnect_count ?? 0) > 3 ? 'text-amber-400' : 'text-slate-300'
            }`}>
              {anomaly.disconnect_count ?? 0}
            </p>
            <p className="text-xs text-slate-600">occurrences</p>
          </div>
        </div>

        {/* Uptime Visualization */}
        <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/50">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-slate-300">Uptime Analysis</span>
            <span className={`text-xs px-2 py-0.5 rounded ${
              (anomaly.offline_time ?? 0) < 15 ? 'bg-emerald-500/20 text-emerald-400' :
              (anomaly.offline_time ?? 0) < 60 ? 'bg-amber-500/20 text-amber-400' : 'bg-red-500/20 text-red-400'
            }`}>
              {(anomaly.offline_time ?? 0) < 15 ? 'Healthy' :
               (anomaly.offline_time ?? 0) < 60 ? 'Degraded' : 'Critical'}
            </span>
          </div>
          <div className="h-8 bg-slate-700/30 rounded-lg overflow-hidden flex">
            <div
              className="h-full bg-emerald-500/50 flex items-center justify-center text-[10px] text-emerald-200 font-medium"
              style={{ width: `${Math.max(10, 100 - ((anomaly.offline_time ?? 0) / 1440) * 100)}%` }}
            >
              Online
            </div>
            {(anomaly.offline_time ?? 0) > 0 && (
              <div
                className="h-full bg-red-500/50 flex items-center justify-center text-[10px] text-red-200 font-medium"
                style={{ width: `${Math.min(90, Math.max(10, ((anomaly.offline_time ?? 0) / 1440) * 100))}%` }}
              >
                Offline
              </div>
            )}
          </div>
          <p className="text-[10px] text-slate-600 mt-2">24-hour period breakdown</p>
        </div>
      </Card>

      {/* Power & Storage */}
      <Card title={
        <span className="flex items-center gap-2">
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          Power & Storage
        </span>
      }>
        <div className="grid grid-cols-2 gap-4">
          {/* Battery Section */}
          <div className="p-4 rounded-xl bg-gradient-to-br from-orange-500/10 to-orange-500/5 border border-orange-500/20">
            <div className="flex items-center justify-between mb-4">
              <span className="text-sm font-medium text-orange-400">Battery Health</span>
              <svg className="w-5 h-5 text-orange-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8V5a1 1 0 00-1-1H8a1 1 0 00-1 1v3M7 8h10l1 12H6L7 8z" />
              </svg>
            </div>
            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-500">Battery Drain (24h)</span>
                  <span className={`font-mono ${
                    (anomaly.total_battery_level_drop ?? 0) > 30 ? 'text-red-400' :
                    (anomaly.total_battery_level_drop ?? 0) > 15 ? 'text-orange-400' : 'text-emerald-400'
                  }`}>
                    {anomaly.total_battery_level_drop ?? 0}%
                  </span>
                </div>
                <div className="h-3 bg-slate-700/50 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all ${
                      (anomaly.total_battery_level_drop ?? 0) > 30 ? 'bg-red-500' :
                      (anomaly.total_battery_level_drop ?? 0) > 15 ? 'bg-orange-500' : 'bg-emerald-500'
                    }`}
                    style={{ width: `${Math.min(100, anomaly.total_battery_level_drop ?? 0)}%` }}
                  />
                </div>
              </div>
              <div className="pt-2 border-t border-orange-500/20">
                <p className="text-[10px] text-slate-500">
                  {(anomaly.total_battery_level_drop ?? 0) > 30
                    ? 'High battery consumption detected - check running apps'
                    : (anomaly.total_battery_level_drop ?? 0) > 15
                    ? 'Moderate battery usage - within expected range'
                    : 'Normal battery consumption'}
                </p>
              </div>
            </div>
          </div>

          {/* Storage Section */}
          <div className="p-4 rounded-xl bg-gradient-to-br from-cyan-500/10 to-cyan-500/5 border border-cyan-500/20">
            <div className="flex items-center justify-between mb-4">
              <span className="text-sm font-medium text-cyan-400">Storage Status</span>
              <svg className="w-5 h-5 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
              </svg>
            </div>
            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-500">Free Space</span>
                  <span className="font-mono text-cyan-400">
                    {anomaly.total_free_storage_kb != null
                      ? `${(anomaly.total_free_storage_kb / 1024 / 1024).toFixed(1)} GB`
                      : 'N/A'}
                  </span>
                </div>
                <div className="h-3 bg-slate-700/50 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-cyan-500 rounded-full"
                    style={{
                      width: anomaly.total_free_storage_kb != null
                        ? `${Math.min(100, (anomaly.total_free_storage_kb / 1024 / 1024 / 64) * 100)}%`
                        : '0%'
                    }}
                  />
                </div>
              </div>
              <div className="pt-2 border-t border-cyan-500/20">
                <p className="text-[10px] text-slate-500">
                  {anomaly.total_free_storage_kb != null && anomaly.total_free_storage_kb < 1024 * 1024 * 2
                    ? 'Low storage warning - cleanup recommended'
                    : 'Storage capacity healthy'}
                </p>
              </div>
            </div>
          </div>
        </div>
      </Card>

      {/* Raw Telemetry Data */}
      <Card title="All Telemetry Metrics">
        <div className="overflow-hidden rounded-lg border border-slate-700/50">
          <table className="w-full text-sm">
            <thead className="bg-slate-800/50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">Metric</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-slate-400 uppercase tracking-wider">Value</th>
                <th className="px-4 py-3 text-center text-xs font-medium text-slate-400 uppercase tracking-wider">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700/30">
              {[
                { label: 'Battery Level Drop', value: anomaly.total_battery_level_drop?.toFixed(0), unit: '%', threshold: { warn: 15, crit: 30 } },
                { label: 'Free Storage', value: anomaly.total_free_storage_kb ? (anomaly.total_free_storage_kb / 1024 / 1024).toFixed(2) : null, unit: 'GB', threshold: { warn: 5, crit: 2 }, inverse: true },
                { label: 'Data Download', value: formatDataSize(anomaly.download).value, unit: formatDataSize(anomaly.download).unit, threshold: { warn: 100, crit: 500 } },
                { label: 'Data Upload', value: formatDataSize(anomaly.upload).value, unit: formatDataSize(anomaly.upload).unit, threshold: { warn: 50, crit: 200 } },
                { label: 'Offline Time', value: anomaly.offline_time?.toFixed(1), unit: 'min', threshold: { warn: 15, crit: 60 } },
                { label: 'Disconnect Count', value: anomaly.disconnect_count?.toFixed(0), unit: '', threshold: { warn: 3, crit: 10 } },
                { label: 'WiFi Signal Strength', value: anomaly.wifi_signal_strength?.toFixed(0), unit: 'dBm', threshold: { warn: -70, crit: -80 }, inverse: true },
                { label: 'Connection Time', value: anomaly.connection_time?.toFixed(2), unit: 's', threshold: { warn: 5, crit: 15 } },
              ].map((metric, i) => {
                const numValue = parseFloat(metric.value ?? '0');
                const getStatus = () => {
                  if (metric.value == null) return 'unknown';
                  if (metric.inverse) {
                    if (numValue <= metric.threshold.crit) return 'critical';
                    if (numValue <= metric.threshold.warn) return 'warning';
                    return 'normal';
                  }
                  if (numValue >= metric.threshold.crit) return 'critical';
                  if (numValue >= metric.threshold.warn) return 'warning';
                  return 'normal';
                };
                const status = getStatus();
                return (
                  <tr key={i} className="hover:bg-slate-800/30 transition-colors">
                    <td className="px-4 py-3 text-slate-300">{metric.label}</td>
                    <td className="px-4 py-3 text-right font-mono text-white">
                      {metric.value != null ? `${metric.value} ${metric.unit}` : 'N/A'}
                    </td>
                    <td className="px-4 py-3 text-center">
                      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium ${
                        status === 'critical' ? 'bg-red-500/20 text-red-400' :
                        status === 'warning' ? 'bg-amber-500/20 text-amber-400' :
                        status === 'unknown' ? 'bg-slate-700/50 text-slate-400' :
                        'bg-emerald-500/20 text-emerald-400'
                      }`}>
                        {status === 'critical' ? 'Critical' :
                         status === 'warning' ? 'Warning' :
                         status === 'unknown' ? 'N/A' : 'Normal'}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </>
  );
}
