import { useParams, Link, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { Card } from '../components/Card';
import { format } from 'date-fns';
import { motion } from 'framer-motion';

const KpiDetail = () => {
  const { type } = useParams<{ type: string }>();
  const navigate = useNavigate();

  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['dashboard', 'stats'],
    queryFn: () => api.getDashboardStats(),
  });

  const { data: anomaliesData } = useQuery({
    queryKey: ['anomalies', { page: 1, page_size: 100 }],
    queryFn: () => api.getAnomalies({ page: 1, page_size: 100 }),
    enabled: type === 'anomalies' || type === 'critical' || type === 'devices',
  });

  const { data: connectionStatus, isLoading: connectionLoading } = useQuery({
    queryKey: ['dashboard', 'connections'],
    queryFn: () => api.getConnectionStatus(),
    enabled: type === 'uptime',
    refetchInterval: 30000,
  });

  const getTitle = () => {
    switch (type) {
      case 'devices':
        return 'Devices Overview';
      case 'anomalies':
        return 'Recent Anomalies';
      case 'critical':
        return 'Critical Issues';
      case 'uptime':
        return 'System Connectivity Status';
      default:
        return 'KPI Detail';
    }
  };

  const getColor = () => {
    switch (type) {
      case 'devices':
        return { text: 'text-cyber-blue', glow: 'glow-text-cyan', accent: 'cyan' as const };
      case 'anomalies':
        return { text: 'text-cyber-red', glow: 'glow-text-red', accent: 'red' as const };
      case 'critical':
        return { text: 'text-cyber-orange', glow: '', accent: 'orange' as const };
      case 'uptime':
        return { text: 'text-cyber-green', glow: 'glow-text-green', accent: 'green' as const };
      default:
        return { text: 'text-slate-200', glow: '', accent: 'cyan' as const };
    }
  };

  const getValue = () => {
    if (statsLoading) return '...';
    switch (type) {
      case 'devices':
        return stats?.devices_monitored || 0;
      case 'anomalies':
        return stats?.anomalies_today || 0;
      case 'critical':
        return stats?.critical_issues || 0;
      case 'uptime': {
        if (!connectionStatus) return '...';
        const systems = [
          connectionStatus.dw_sql,
          connectionStatus.mc_sql,
          connectionStatus.mobicontrol_api,
          connectionStatus.llm,
        ];
        const connectedCount = systems.filter((s) => s.connected).length;
        return `${Math.round((connectedCount / systems.length) * 100)}%`;
      }
      default:
        return 0;
    }
  };

  const colors = getColor();

  return (
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate(-1)}
            className="p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded-lg transition-colors"
          >
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
          <div>
            <h1 className="text-2xl font-bold text-white">{getTitle()}</h1>
            <p className="text-sm text-slate-500 mt-1">Detailed view of selected metric</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Main Stat Card */}
        <Card accent={colors.accent}>
          <div className="text-center py-8">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
              {type === 'uptime' ? 'Overall Connectivity' : 'Current Count'}
            </p>
            <p className={`text-5xl font-bold font-mono ${colors.text} ${colors.glow}`}>
              {getValue()}
            </p>
          </div>
        </Card>

        {/* Details Card */}
        <Card className="md:col-span-2" title="Details" noPadding>
          <div className="p-5">
            {type === 'devices' && (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-slate-700/50">
                      <th className="table-header">Device ID</th>
                      <th className="table-header">Latest Anomaly</th>
                      <th className="table-header">Last Seen</th>
                    </tr>
                  </thead>
                  <tbody>
                    {anomaliesData?.anomalies
                      .reduce((acc: any[], current) => {
                        if (!acc.find((a) => a.device_id === current.device_id)) {
                          acc.push(current);
                        }
                        return acc;
                      }, [])
                      .slice(0, 10)
                      .map((device, index) => (
                        <motion.tr
                          key={device.device_id}
                          className="table-row"
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: index * 0.05 }}
                        >
                          <td className="table-cell">
                            <Link
                              to={`/devices/${device.device_id}`}
                              className="font-semibold text-cyber-blue hover:text-cyan-300 transition-colors"
                            >
                              Device {device.device_id}
                            </Link>
                          </td>
                          <td className="table-cell font-mono text-slate-300">
                            {device.anomaly_score.toFixed(3)}
                          </td>
                          <td className="table-cell font-mono text-slate-500">
                            {format(new Date(device.timestamp), 'MMM dd, HH:mm')}
                          </td>
                        </motion.tr>
                      ))}
                    {(!anomaliesData || anomaliesData.anomalies.length === 0) && (
                      <tr>
                        <td colSpan={3} className="px-4 py-8 text-center text-sm text-slate-500">
                          No devices found in recent logs.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            )}

            {(type === 'anomalies' || type === 'critical') && (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-slate-700/50">
                      <th className="table-header">Device</th>
                      <th className="table-header">Score</th>
                      <th className="table-header">Status</th>
                      <th className="table-header">Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(type === 'critical'
                      ? anomaliesData?.anomalies.filter((a) => a.anomaly_score <= -0.7)
                      : anomaliesData?.anomalies
                    )
                      ?.slice(0, 15)
                      .map((anomaly, index) => (
                        <motion.tr
                          key={anomaly.id}
                          className="table-row"
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: index * 0.03 }}
                        >
                          <td className="table-cell">
                            <Link
                              to={`/devices/${anomaly.device_id}`}
                              className="font-semibold text-cyber-blue hover:text-cyan-300 transition-colors"
                            >
                              Device {anomaly.device_id}
                            </Link>
                          </td>
                          <td className="table-cell">
                            <span className="font-mono font-bold text-cyber-red glow-text-red">
                              {anomaly.anomaly_score.toFixed(3)}
                            </span>
                          </td>
                          <td className="table-cell">
                            <span className="badge-critical">{anomaly.status}</span>
                          </td>
                          <td className="table-cell font-mono text-slate-500">
                            {format(new Date(anomaly.timestamp), 'HH:mm:ss')}
                          </td>
                        </motion.tr>
                      ))}
                    {(!anomaliesData ||
                      (type === 'critical'
                        ? anomaliesData.anomalies.filter((a) => a.anomaly_score <= -0.7).length === 0
                        : anomaliesData.anomalies.length === 0)) && (
                      <tr>
                        <td colSpan={4} className="px-4 py-8 text-center text-sm text-slate-500">
                          No recent anomalies found.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            )}

            {type === 'uptime' && (
              <div className="space-y-3">
                {connectionLoading ? (
                  <div className="flex items-center justify-center py-8">
                    <div className="relative w-10 h-10">
                      <div className="absolute inset-0 rounded-full border-2 border-cyber-blue/20"></div>
                      <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-cyber-blue animate-spin"></div>
                    </div>
                  </div>
                ) : connectionStatus ? (
                  <>
                    {[
                      { label: 'DW SQL Server', status: connectionStatus.dw_sql },
                      { label: 'MC SQL Server', status: connectionStatus.mc_sql },
                      { label: 'MobiControl API', status: connectionStatus.mobicontrol_api },
                      { label: 'LLM Service', status: connectionStatus.llm },
                    ].map((item, index) => (
                      <motion.div
                        key={item.label}
                        className={`flex justify-between items-center p-4 rounded-lg border ${
                          item.status.connected
                            ? 'bg-cyber-green/5 border-cyber-green/20'
                            : 'bg-cyber-red/5 border-cyber-red/20'
                        }`}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1 }}
                      >
                        <div>
                          <span className="text-sm font-semibold text-slate-200 block">
                            {item.label}
                          </span>
                          <span className="text-xs font-mono text-slate-500 truncate block max-w-[200px]">
                            {item.status.server}
                          </span>
                        </div>
                        <div className="flex flex-col items-end">
                          <span
                            className={
                              item.status.connected ? 'badge-success' : 'badge-critical'
                            }
                          >
                            {item.status.status}
                          </span>
                          {!item.status.connected && item.status.error && (
                            <span className="text-[10px] text-cyber-red mt-2 max-w-[150px] text-right">
                              {item.status.error}
                            </span>
                          )}
                        </div>
                      </motion.div>
                    ))}
                    <p className="text-[10px] font-mono text-slate-600 text-right mt-4">
                      Last checked:{' '}
                      {format(new Date(connectionStatus.last_checked), 'yyyy-MM-dd HH:mm:ss')}
                    </p>
                  </>
                ) : (
                  <p className="text-cyber-red text-sm text-center py-8">
                    Failed to load connection status.
                  </p>
                )}
              </div>
            )}
          </div>
        </Card>
      </div>
    </motion.div>
  );
};

export default KpiDetail;
