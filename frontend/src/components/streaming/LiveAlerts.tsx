import { useEffect, useMemo, useState } from 'react';
import { clsx } from 'clsx';
import { Card } from '../Card';
import { wsClient, StreamingAlert } from '../../lib/websocket';

interface LiveAlertsProps {
  maxAlerts?: number;
  onAlertClick?: (alert: StreamingAlert) => void;
}

const severityStyles: Record<StreamingAlert['severity'], string> = {
  low: 'bg-cyan-500/10 text-cyan-300 border-cyan-500/30',
  medium: 'bg-amber-500/10 text-amber-300 border-amber-500/30',
  high: 'bg-orange-500/10 text-orange-300 border-orange-500/30',
  critical: 'bg-red-500/10 text-red-300 border-red-500/30',
};

const normalizeAlert = (payload: Record<string, unknown>): StreamingAlert => {
  const deviceId = String(payload.device_id ?? payload.deviceId ?? 'unknown');
  const severity = (payload.severity ?? 'low') as StreamingAlert['severity'];
  const timestamp = String(payload.timestamp ?? new Date().toISOString());
  return {
    id: String(payload.id ?? payload.anomaly_id ?? `${deviceId}-${timestamp}`),
    device_id: deviceId,
    device_name: String(payload.device_name ?? payload.deviceName ?? `Device ${deviceId}`),
    anomaly_score: Number(payload.anomaly_score ?? payload.score ?? 0),
    severity,
    message: String(payload.message ?? payload.summary ?? 'Anomaly detected'),
    timestamp,
  };
};

export function LiveAlerts({ maxAlerts = 50, onAlertClick }: LiveAlertsProps) {
  const [alerts, setAlerts] = useState<StreamingAlert[]>([]);
  const [connected, setConnected] = useState(false);
  const [paused, setPaused] = useState(false);

  const header = useMemo(() => (
    <div className="flex items-center justify-between w-full">
      <span className="telemetry-label">Live Alerts</span>
      <div className="flex items-center gap-3 text-xs text-slate-400">
        <button
          type="button"
          onClick={() => setPaused((prev) => !prev)}
          className="hover:text-white transition-colors"
        >
          {paused ? 'Resume' : 'Pause'}
        </button>
        <span
          className={clsx(
            'w-2 h-2 rounded-full',
            connected ? 'bg-emerald-400 animate-pulse' : 'bg-red-400'
          )}
        />
      </div>
    </div>
  ), [connected, paused]);

  useEffect(() => {
    wsClient.connect();

    const handleAlert = (payload: unknown) => {
      if (paused || !payload || typeof payload !== 'object') {
        return;
      }
      const message = normalizeAlert(payload as Record<string, unknown>);
      setAlerts((prev) => [message, ...prev].slice(0, maxAlerts));
    };

    const unsubscribeAnomaly = wsClient.subscribe('anomaly', handleAlert);
    const unsubscribeAlert = wsClient.subscribe('alert', handleAlert);

    const timer = setInterval(() => {
      setConnected(wsClient.isConnected());
    }, 1000);

    return () => {
      unsubscribeAnomaly();
      unsubscribeAlert();
      clearInterval(timer);
    };
  }, [maxAlerts, paused]);

  return (
    <Card title={header} accent="danger" noPadding>
      <div className="max-h-80 overflow-y-auto p-5 space-y-3">
        {alerts.length === 0 ? (
          <div className="text-center text-slate-500 py-6">
            <div className="text-sm">No alerts yet</div>
            <div className="text-xs mt-1">Waiting for real-time data...</div>
          </div>
        ) : (
          alerts.map((alert) => (
            <button
              key={alert.id}
              type="button"
              onClick={() => onAlertClick?.(alert)}
              className="w-full text-left"
            >
              <div className="p-3 rounded-lg border border-slate-700/50 hover:border-slate-600/70 hover:bg-slate-800/40 transition-colors">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className={clsx('px-2 py-0.5 rounded-full text-xs border', severityStyles[alert.severity])}>
                      {alert.severity.toUpperCase()}
                    </span>
                    <span className="text-sm text-white truncate">
                      {alert.device_name}
                    </span>
                  </div>
                  <span className="text-xs text-slate-500">
                    {new Date(alert.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <div className="text-xs text-slate-400 mt-2">
                  {alert.message}
                </div>
              </div>
            </button>
          ))
        )}
      </div>
    </Card>
  );
}
