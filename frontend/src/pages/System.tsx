/**
 * System Page - Stellar Operations Configuration
 *
 * Service connections, location attributes,
 * and application settings
 */

import { useQuery, useMutation } from '@tanstack/react-query';
import { api } from '../api/client';
import { formatDistanceToNow } from 'date-fns';
import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from '../components/Card';
import { LocationAttributeSettings, ConnectionCards } from '../components/settings';
import { useMockMode } from '../hooks/useMockMode';

// Auto-refresh interval options
const INTERVAL_OPTIONS = [
  { value: 5000, label: '5 seconds' },
  { value: 10000, label: '10 seconds' },
  { value: 15000, label: '15 seconds' },
  { value: 30000, label: '30 seconds' },
  { value: 60000, label: '1 minute' },
  { value: 300000, label: '5 minutes' },
  { value: 0, label: 'Manual only' },
];

// Loading spinner component
function LoadingSpinner() {
  return (
    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" aria-hidden="true">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
    </svg>
  );
}

function System() {
  const { mockMode } = useMockMode();
  const [pollingInterval, setPollingInterval] = useState(() => {
    const saved = localStorage.getItem('connectionPollingInterval');
    return saved ? parseInt(saved, 10) : 300000;
  });
  const [retryCountdown, setRetryCountdown] = useState(pollingInterval / 1000);

  const {
    data: connections,
    isFetching: isCheckingConnections,
    dataUpdatedAt,
    refetch,
  } = useQuery({
    queryKey: ['dashboard', 'connections'],
    queryFn: () => api.getConnectionStatus(),
    refetchInterval: pollingInterval || false,
    retry: true,
    retryDelay: 5000,
  });

  const [troubleshootingAdvice, setTroubleshootingAdvice] = useState<string | null>(null);
  const [troubleshootingError, setTroubleshootingError] = useState<string | null>(null);
  const [showTroubleshooting, setShowTroubleshooting] = useState(false);

  const troubleshootMutation = useMutation({
    mutationFn: (connectionStatus: typeof connections) => {
      if (!connectionStatus) throw new Error('No connection status available');
      return api.getTroubleshootingAdvice(connectionStatus);
    },
    onSuccess: (data) => {
      setTroubleshootingAdvice(data.advice);
      setTroubleshootingError(null);
    },
    onError: (error: Error) => {
      setTroubleshootingError(error.message || 'Failed to get troubleshooting advice. Please try again.');
      setTroubleshootingAdvice(null);
    },
  });

  const hasFailedConnections =
    connections &&
    (!connections.backend_db?.connected ||
      !connections.redis?.connected ||
      !connections.qdrant?.connected ||
      !connections.dw_sql?.connected ||
      !connections.mc_sql?.connected ||
      !connections.mobicontrol_api?.connected ||
      !connections.llm?.connected);

  const connectedCount = connections
    ? [connections.backend_db, connections.redis, connections.qdrant, connections.dw_sql, connections.mc_sql, connections.mobicontrol_api, connections.llm].filter(s => s?.connected).length
    : 0;

  const handleIntervalChange = (newInterval: number) => {
    setPollingInterval(newInterval);
    localStorage.setItem('connectionPollingInterval', String(newInterval));
    setRetryCountdown(newInterval / 1000);
  };

  // Update countdown timer
  useEffect(() => {
    if (pollingInterval === 0) return;

    const interval = setInterval(() => {
      if (dataUpdatedAt) {
        const elapsed = Math.floor((Date.now() - dataUpdatedAt) / 1000);
        const intervalSeconds = pollingInterval / 1000;
        const remaining = Math.max(0, intervalSeconds - elapsed);
        setRetryCountdown(remaining);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [dataUpdatedAt, pollingInterval]);

  return (
    <motion.div
      className="space-y-8"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      {/* Header */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">System</h1>
          <p className="text-slate-500 mt-1">
            Configuration, connections, and application settings
          </p>
        </div>

        {/* Overall Status */}
        <div
          className="flex items-center gap-4 px-5 py-3 stellar-card rounded-xl"
          role="status"
          aria-label={`System status: ${connectedCount} of 7 services connected`}
        >
          <div
            className={`w-3 h-3 rounded-full ${connectedCount === 7
              ? 'bg-emerald-400 shadow-[0_0_8px_rgba(16,185,129,0.6)]'
              : connectedCount >= 4
                ? 'bg-orange-400 animate-pulse shadow-[0_0_8px_rgba(251,146,60,0.6)]'
                : 'bg-red-400 animate-pulse shadow-[0_0_8px_rgba(248,113,113,0.6)]'
              }`}
            aria-hidden="true"
          />
          <div>
            <p className="text-sm font-medium text-white">
              {connectedCount === 7
                ? 'All Systems Operational'
                : connectedCount >= 4
                  ? 'Partial Connectivity'
                  : 'Critical: Services Offline'}
            </p>
            <p className="text-xs text-slate-500">{connectedCount}/7 services connected</p>
          </div>
        </div>
      </div>

      {/* Connection Grid */}
      <section aria-labelledby="connections-heading">
        <div className="flex items-center justify-between mb-4">
          <h2 id="connections-heading" className="text-xl font-bold text-white">
            Service Connections
          </h2>
          <div className="flex items-center gap-4">
            {/* Auto-refresh Selector */}
            <div className="flex items-center gap-3">
              <label htmlFor="refresh-interval" className="telemetry-label">
                Auto-refresh:
              </label>
              <select
                id="refresh-interval"
                value={pollingInterval}
                onChange={(e) => handleIntervalChange(parseInt(e.target.value, 10))}
                className="select-field w-auto"
              >
                {INTERVAL_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Refresh Button */}
            <button
              onClick={() => refetch()}
              disabled={isCheckingConnections}
              className="btn-ghost text-sm flex items-center gap-2"
              aria-label="Refresh connection status"
            >
              {isCheckingConnections ? (
                <LoadingSpinner />
              ) : (
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
              )}
              Refresh
            </button>
          </div>
        </div>

        {/* Status indicators */}
        {pollingInterval > 0 && hasFailedConnections && !isCheckingConnections && (
          <div className="mb-4 flex items-center gap-2 text-xs text-slate-500" role="status">
            <svg className="w-4 h-4 text-orange-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>Next retry in {retryCountdown}s</span>
            {dataUpdatedAt && (
              <span className="text-slate-600">
                • Last checked {formatDistanceToNow(new Date(dataUpdatedAt), { addSuffix: true })}
              </span>
            )}
          </div>
        )}

        {/* Service Cards */}
        <ConnectionCards connections={connections} isChecking={isCheckingConnections} />
      </section>

      {/* Troubleshooting Section */}
      <AnimatePresence>
        {hasFailedConnections && (
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            aria-labelledby="troubleshooting-heading"
          >
            <Card>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-orange-500/10" aria-hidden="true">
                    <svg className="w-5 h-5 text-orange-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                        d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                  </div>
                  <div>
                    <h3 id="troubleshooting-heading" className="font-semibold text-white">
                      Connection Issues Detected
                    </h3>
                    <p className="text-xs text-slate-500">Get AI-powered troubleshooting advice</p>
                  </div>
                </div>
                <button
                  onClick={() => {
                    setShowTroubleshooting(!showTroubleshooting);
                    if (!troubleshootingAdvice && connections) {
                      troubleshootMutation.mutate(connections);
                    }
                  }}
                  disabled={troubleshootMutation.isPending}
                  className="btn-stellar text-sm flex items-center gap-2"
                  aria-expanded={showTroubleshooting}
                  aria-controls="troubleshooting-content"
                >
                  {troubleshootMutation.isPending ? (
                    <>
                      <LoadingSpinner />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                      </svg>
                      Analyze with Stella AI
                      <svg
                        className={`w-4 h-4 transition-transform ${showTroubleshooting ? 'rotate-180' : ''}`}
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        aria-hidden="true"
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </>
                  )}
                </button>
              </div>

              <AnimatePresence>
                {showTroubleshooting && (troubleshootingAdvice || troubleshootingError) && (
                  <motion.div
                    id="troubleshooting-content"
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="overflow-hidden"
                  >
                    {troubleshootingError ? (
                      <div className="p-4 bg-red-500/10 rounded-xl border border-red-500/30">
                        <div className="flex items-center gap-2 text-red-400">
                          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          <span className="text-sm font-medium">{troubleshootingError}</span>
                        </div>
                      </div>
                    ) : (
                      <div className="p-4 bg-slate-800/50 rounded-xl border border-slate-700/50">
                        <pre className="whitespace-pre-wrap text-sm text-slate-300 font-mono leading-relaxed">
                          {troubleshootingAdvice}
                        </pre>
                      </div>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </Card>
          </motion.section>
        )}
      </AnimatePresence>

      {/* Location Attribute Configuration */}
      <LocationAttributeSettings />

      {/* Application Info */}
      <section aria-labelledby="app-info-heading">
        <Card title={<span id="app-info-heading" className="telemetry-label">Application Info</span>}>
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 xl:grid-cols-4">
            <InfoItem label="Version" value="0.3.0" />
            <InfoItem label="Environment" value={mockMode ? 'Demo' : 'Development'} />
            <InfoItem label="Database" value="PostgreSQL" />
            <InfoItem label="Model" value="Isolation Forest" />
          </div>
        </Card>
      </section>
    </motion.div>
  );
}

function InfoItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="telemetry-label mb-1">{label}</p>
      <p className="text-lg font-medium text-white">{value}</p>
    </div>
  );
}

export default System;
