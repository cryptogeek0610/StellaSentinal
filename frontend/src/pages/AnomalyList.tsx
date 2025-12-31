import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useAnomalies } from '../hooks/useAnomalies';
import { format } from 'date-fns';
import { motion } from 'framer-motion';
import { Card } from '../components/Card';

function AnomalyList() {
  const [page, setPage] = useState(1);
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [deviceIdFilter, setDeviceIdFilter] = useState<string>('');

  const { data, isLoading, error } = useAnomalies({
    page,
    page_size: 50,
    status: statusFilter || undefined,
    device_id: deviceIdFilter ? parseInt(deviceIdFilter) : undefined,
  });

  const getScoreSeverity = (score: number) => {
    if (score <= -0.7) return 'critical';
    if (score <= -0.5) return 'high';
    if (score <= -0.3) return 'medium';
    return 'low';
  };

  const getSeverityBadge = (severity: string) => {
    const classes = {
      critical: 'badge-critical',
      high: 'badge-warning',
      medium: 'badge-info',
      low: 'badge-neutral',
    };
    return classes[severity as keyof typeof classes] || 'badge-neutral';
  };

  const getStatusBadge = (status: string) => {
    const classes: Record<string, string> = {
      open: 'badge-critical',
      investigating: 'badge-warning',
      resolved: 'badge-success',
      false_positive: 'badge-neutral',
    };
    return classes[status] || 'badge-neutral';
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="relative w-16 h-16 mx-auto mb-4">
            <div className="absolute inset-0 rounded-full border-2 border-cyber-blue/20"></div>
            <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-cyber-blue animate-spin"></div>
          </div>
          <p className="text-slate-400 font-mono text-sm">Loading devices...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-cyber-red/20 flex items-center justify-center">
            <svg className="w-8 h-8 text-cyber-red" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </div>
          <p className="text-cyber-red font-medium">Error loading devices</p>
        </div>
      </div>
    );
  }

  return (
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Devices with Anomalies</h1>
          <p className="text-sm text-slate-500 mt-1">Monitor and investigate device anomalies</p>
        </div>
        <div className="flex items-center gap-2">
          <span className="badge-info">
            {data?.total || 0} Total
          </span>
        </div>
      </div>

      {/* Filters */}
      <Card noPadding>
        <div className="p-4 border-b border-slate-700/50">
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex-1 min-w-[200px]">
              <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
                Device ID
              </label>
              <input
                type="text"
                value={deviceIdFilter}
                onChange={(e) => {
                  setDeviceIdFilter(e.target.value);
                  setPage(1);
                }}
                placeholder="Search by device ID..."
                className="input-field"
              />
            </div>
            <div className="w-48">
              <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
                Status
              </label>
              <select
                value={statusFilter}
                onChange={(e) => {
                  setStatusFilter(e.target.value);
                  setPage(1);
                }}
                className="select-field"
              >
                <option value="">All Statuses</option>
                <option value="open">Open</option>
                <option value="investigating">Investigating</option>
                <option value="resolved">Resolved</option>
                <option value="false_positive">False Positive</option>
              </select>
            </div>
          </div>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-700/50">
                <th className="table-header w-12">
                  <input
                    type="checkbox"
                    className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-cyber-blue focus:ring-cyber-blue/50"
                  />
                </th>
                <th className="table-header">Device</th>
                <th className="table-header">Last Anomaly</th>
                <th className="table-header">Score</th>
                <th className="table-header">Severity</th>
                <th className="table-header">Status</th>
                <th className="table-header text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {data?.anomalies.map((anomaly, index) => {
                const severity = getScoreSeverity(anomaly.anomaly_score);
                return (
                  <motion.tr
                    key={anomaly.id}
                    className="table-row"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.03 }}
                  >
                    <td className="table-cell">
                      <input
                        type="checkbox"
                        className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-cyber-blue focus:ring-cyber-blue/50"
                      />
                    </td>
                    <td className="table-cell">
                      <div className="flex items-center gap-3">
                        <div
                          className={`w-2 h-8 rounded-full ${
                            severity === 'critical'
                              ? 'bg-cyber-red'
                              : severity === 'high'
                              ? 'bg-cyber-orange'
                              : severity === 'medium'
                              ? 'bg-cyber-blue'
                              : 'bg-cyber-green'
                          }`}
                        />
                        <div>
                          <Link
                            to={`/devices/${anomaly.device_id}`}
                            className="font-semibold text-slate-200 hover:text-cyber-blue transition-colors"
                          >
                            Device {anomaly.device_id}
                          </Link>
                          <p className="text-xs text-slate-500 font-mono">
                            ID: {anomaly.device_id}
                          </p>
                        </div>
                      </div>
                    </td>
                    <td className="table-cell">
                      <span className="font-mono text-slate-300">
                        {format(new Date(anomaly.timestamp), 'MMM dd, HH:mm')}
                      </span>
                    </td>
                    <td className="table-cell">
                      <span
                        className={`font-mono font-bold ${
                          severity === 'critical'
                            ? 'text-cyber-red glow-text-red'
                            : severity === 'high'
                            ? 'text-cyber-orange'
                            : severity === 'medium'
                            ? 'text-cyber-blue'
                            : 'text-slate-400'
                        }`}
                      >
                        {anomaly.anomaly_score.toFixed(3)}
                      </span>
                    </td>
                    <td className="table-cell">
                      <span className={getSeverityBadge(severity)}>
                        {severity}
                      </span>
                    </td>
                    <td className="table-cell">
                      <span className={getStatusBadge(anomaly.status)}>
                        {anomaly.status}
                      </span>
                    </td>
                    <td className="table-cell text-right">
                      <Link
                        to={`/investigations/${anomaly.id}`}
                        className="btn-secondary text-xs py-1.5"
                      >
                        Investigate
                      </Link>
                    </td>
                  </motion.tr>
                );
              })}
              {(!data || data.anomalies.length === 0) && (
                <tr>
                  <td colSpan={7} className="px-6 py-12 text-center">
                    <div className="flex flex-col items-center">
                      <div className="w-16 h-16 mb-4 rounded-full bg-slate-800 flex items-center justify-center">
                        <svg
                          className="w-8 h-8 text-slate-600"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={1.5}
                            d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                          />
                        </svg>
                      </div>
                      <p className="text-slate-400 font-medium">No devices with anomalies found</p>
                      <p className="text-sm text-slate-600 mt-1">
                        All systems operating normally
                      </p>
                    </div>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {data && data.total_pages > 1 && (
          <div className="px-4 py-3 border-t border-slate-700/50 flex items-center justify-between">
            <div className="text-sm text-slate-500 font-mono">
              Page {page} of {data.total_pages}
              <span className="ml-2 text-slate-600">
                ({((page - 1) * 50) + 1}-{Math.min(page * 50, data.total)} of {data.total})
              </span>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setPage(Math.max(1, page - 1))}
                disabled={page === 1}
                className="btn-secondary text-xs py-1.5 disabled:opacity-30"
              >
                ← Previous
              </button>
              <button
                onClick={() => setPage(Math.min(data.total_pages, page + 1))}
                disabled={page === data.total_pages}
                className="btn-secondary text-xs py-1.5 disabled:opacity-30"
              >
                Next →
              </button>
            </div>
          </div>
        )}
      </Card>
    </motion.div>
  );
}

export default AnomalyList;
