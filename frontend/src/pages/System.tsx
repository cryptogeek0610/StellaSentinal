/**
 * System Page - Consolidated Admin Dashboard
 *
 * Combines:
 * - Connections & diagnostics (original System)
 * - ML Pipeline (Training, Automation)
 * - Configuration (Setup, Baselines)
 *
 * Simplified per first-principles redesign: 14 pages → 3
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../api/client';
import { formatDistanceToNow } from 'date-fns';
import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from '../components/Card';
import { LocationAttributeSettings, ConnectionCards } from '../components/settings';
import { useMockMode } from '../hooks/useMockMode';
import { ToggleSwitch } from '../components/ui';
import type { SchedulerConfig, ScheduleInterval } from '../types/automation';

// Tab definitions
type TabId = 'overview' | 'ml-pipeline' | 'configuration';

const tabs: { id: TabId; label: string; description: string }[] = [
  { id: 'overview', label: 'Overview', description: 'Connections & diagnostics' },
  { id: 'ml-pipeline', label: 'ML Pipeline', description: 'Training & scoring' },
  { id: 'configuration', label: 'Configuration', description: 'Settings & baselines' },
];

// Loading spinner
function LoadingSpinner({ className = 'h-4 w-4' }: { className?: string }) {
  return (
    <svg className={`animate-spin ${className}`} viewBox="0 0 24 24" aria-hidden="true">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
    </svg>
  );
}

// Format relative time
function formatRelativeTime(dateStr: string | null | undefined): string {
  if (!dateStr) return 'Never';
  const date = new Date(dateStr);
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  if (diff < 0) {
    const futureMinutes = Math.abs(minutes);
    const futureHours = Math.abs(hours);
    if (futureMinutes < 60) return `in ${futureMinutes}m`;
    if (futureHours < 24) return `in ${futureHours}h`;
    return date.toLocaleString();
  }
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  return date.toLocaleDateString();
}

// Overview Tab - Connections & Diagnostics
function OverviewTab() {
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
  });

  const { data: llmDiagnostics, isFetching: isLoadingDiagnostics } = useQuery({
    queryKey: ['dashboard', 'llm-diagnostics'],
    queryFn: () => api.getLLMDiagnostics(),
    refetchInterval: 60000,
  });

  const { data: cacheStats, isFetching: isLoadingCacheStats } = useQuery({
    queryKey: ['dashboard', 'cache-stats'],
    queryFn: () => api.getTroubleshootingCacheStats(),
    refetchInterval: 60000,
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
      setTroubleshootingError(error.message || 'Failed to get troubleshooting advice.');
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

  const handleIntervalChange = (newInterval: number) => {
    setPollingInterval(newInterval);
    localStorage.setItem('connectionPollingInterval', String(newInterval));
    setRetryCountdown(newInterval / 1000);
  };

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

  const INTERVAL_OPTIONS = [
    { value: 5000, label: '5 seconds' },
    { value: 10000, label: '10 seconds' },
    { value: 30000, label: '30 seconds' },
    { value: 60000, label: '1 minute' },
    { value: 300000, label: '5 minutes' },
    { value: 0, label: 'Manual only' },
  ];

  return (
    <div className="space-y-6">
      {/* Connection Grid */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-white">Service Connections</h2>
          <div className="flex items-center gap-4">
            <select
              value={pollingInterval}
              onChange={(e) => handleIntervalChange(parseInt(e.target.value, 10))}
              className="select-field w-auto text-sm"
            >
              {INTERVAL_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
            <button
              onClick={() => refetch()}
              disabled={isCheckingConnections}
              className="btn-ghost text-sm flex items-center gap-2"
            >
              {isCheckingConnections ? <LoadingSpinner /> : (
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
              )}
              Refresh
            </button>
          </div>
        </div>

        {pollingInterval > 0 && hasFailedConnections && !isCheckingConnections && (
          <div className="mb-4 flex items-center gap-2 text-xs text-slate-500">
            <svg className="w-4 h-4 text-orange-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
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

        <ConnectionCards connections={connections} isChecking={isCheckingConnections} />
      </section>

      {/* Troubleshooting */}
      <AnimatePresence>
        {hasFailedConnections && (
          <motion.section initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }}>
            <Card>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-orange-500/10">
                    <svg className="w-5 h-5 text-orange-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                        d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                  </div>
                  <div>
                    <h3 className="font-semibold text-white">Connection Issues Detected</h3>
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
                >
                  {troubleshootMutation.isPending ? (
                    <><LoadingSpinner />Analyzing...</>
                  ) : (
                    <>Analyze with Stella AI</>
                  )}
                </button>
              </div>

              <AnimatePresence>
                {showTroubleshooting && (troubleshootingAdvice || troubleshootingError) && (
                  <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} className="overflow-hidden">
                    {troubleshootingError ? (
                      <div className="p-4 bg-red-500/10 rounded-xl border border-red-500/30">
                        <span className="text-sm text-red-400">{troubleshootingError}</span>
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

      {/* System Diagnostics */}
      <section>
        <Card title={<span className="telemetry-label">System Diagnostics</span>}>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <div className="space-y-3">
              <div className="text-sm text-slate-400">LLM Diagnostics</div>
              <div className="bg-slate-800/50 rounded-lg border border-slate-700/50 p-4 text-xs text-slate-300 font-mono max-h-48 overflow-auto">
                {isLoadingDiagnostics ? 'Loading...' : JSON.stringify(llmDiagnostics, null, 2)}
              </div>
            </div>
            <div className="space-y-3">
              <div className="text-sm text-slate-400">Cache Stats</div>
              <div className="bg-slate-800/50 rounded-lg border border-slate-700/50 p-4 text-xs text-slate-300 font-mono max-h-48 overflow-auto">
                {isLoadingCacheStats ? 'Loading...' : JSON.stringify(cacheStats, null, 2)}
              </div>
            </div>
          </div>
        </Card>
      </section>

      {/* App Info */}
      <section>
        <Card title={<span className="telemetry-label">Application Info</span>}>
          <div className="grid grid-cols-2 gap-6 sm:grid-cols-4">
            <div><p className="telemetry-label mb-1">Version</p><p className="text-lg font-medium text-white">0.3.0</p></div>
            <div><p className="telemetry-label mb-1">Environment</p><p className="text-lg font-medium text-white">{mockMode ? 'Demo' : 'Development'}</p></div>
            <div><p className="telemetry-label mb-1">Database</p><p className="text-lg font-medium text-white">PostgreSQL</p></div>
            <div><p className="telemetry-label mb-1">Model</p><p className="text-lg font-medium text-white">Isolation Forest</p></div>
          </div>
        </Card>
      </section>
    </div>
  );
}

// ML Pipeline Tab - Training & Automation
function MLPipelineTab() {
  const queryClient = useQueryClient();

  const { data: trainingStatus } = useQuery({
    queryKey: ['training-status'],
    queryFn: () => api.getTrainingStatus(),
    refetchInterval: (query) => {
      const data = query.state.data;
      return data?.status === 'running' || data?.status === 'pending' ? 2000 : 30000;
    },
  });

  const { data: automationStatus } = useQuery({
    queryKey: ['automation-status'],
    queryFn: () => api.getAutomationStatus(),
    refetchInterval: 5000,
  });

  const { data: queueStatus } = useQuery({
    queryKey: ['training-queue'],
    queryFn: () => api.getTrainingQueueStatus(),
    refetchInterval: 10000,
  });

  const { data: history } = useQuery({
    queryKey: ['training-history'],
    queryFn: () => api.getTrainingHistory(5),
  });

  const triggerJobMutation = useMutation({
    mutationFn: (jobType: 'training' | 'scoring') => api.triggerAutomationJob({ job_type: jobType }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['automation-status'] });
      queryClient.invalidateQueries({ queryKey: ['training-status'] });
    },
  });

  const isTrainingRunning = trainingStatus?.status === 'running' || trainingStatus?.status === 'pending';
  const isScoringRunning = automationStatus?.scoring_status === 'running';

  return (
    <div className="space-y-6">
      {/* Status Overview */}
      <div className="grid grid-cols-2 gap-6">
        <Card className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">Training Status</h3>
            <span className={`px-3 py-1 text-xs font-semibold rounded-full ${
              isTrainingRunning
                ? 'bg-stellar-900/50 text-stellar-400 border border-stellar-500/30'
                : 'bg-slate-800 text-slate-400 border border-slate-600'
            }`}>
              {trainingStatus?.status || 'idle'}
            </span>
          </div>
          {isTrainingRunning && (
            <div className="mb-4">
              <div className="flex justify-between text-sm mb-2">
                <span className="text-slate-400">{trainingStatus?.message}</span>
                <span className="text-stellar-400 font-bold">{Math.round(trainingStatus?.progress || 0)}%</span>
              </div>
              <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                <div className="h-full bg-stellar-500" style={{ width: `${trainingStatus?.progress || 0}%` }} />
              </div>
            </div>
          )}
          <div className="text-sm text-slate-500">
            Last trained: {formatRelativeTime(history?.runs?.[0]?.completed_at)}
          </div>
          <button
            onClick={() => triggerJobMutation.mutate('training')}
            disabled={isTrainingRunning || triggerJobMutation.isPending}
            className="mt-4 w-full py-2 bg-stellar-600 hover:bg-stellar-500 disabled:bg-slate-700 text-white rounded-lg text-sm font-medium transition-colors"
          >
            {isTrainingRunning ? 'Training in Progress...' : 'Start Training'}
          </button>
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">Scoring Status</h3>
            <span className={`px-3 py-1 text-xs font-semibold rounded-full ${
              isScoringRunning
                ? 'bg-aurora-900/50 text-aurora-400 border border-aurora-500/30'
                : 'bg-slate-800 text-slate-400 border border-slate-600'
            }`}>
              {isScoringRunning ? 'running' : 'idle'}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p className="text-slate-500">Devices Scored</p>
              <p className="text-xl font-bold text-white">{automationStatus?.last_scoring_result?.total_scored ?? 0}</p>
            </div>
            <div>
              <p className="text-slate-500">Anomalies Found</p>
              <p className="text-xl font-bold text-cosmic-400">{automationStatus?.last_scoring_result?.anomalies_detected ?? 0}</p>
            </div>
          </div>
          <button
            onClick={() => triggerJobMutation.mutate('scoring')}
            disabled={isScoringRunning || triggerJobMutation.isPending}
            className="mt-4 w-full py-2 bg-aurora-600 hover:bg-aurora-500 disabled:bg-slate-700 text-white rounded-lg text-sm font-medium transition-colors"
          >
            {isScoringRunning ? 'Scoring in Progress...' : 'Score All Devices'}
          </button>
        </Card>
      </div>

      {/* Queue Status */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Training Queue</h3>
        <div className="grid grid-cols-4 gap-4 text-sm">
          <div className="bg-slate-800/50 rounded-lg p-3">
            <p className="text-slate-500">Queue Length</p>
            <p className="text-xl font-bold text-white">{queueStatus?.queue_length ?? 0}</p>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-3">
            <p className="text-slate-500">Worker Status</p>
            <p className={`text-lg font-semibold ${queueStatus?.worker_available ? 'text-emerald-400' : 'text-amber-400'}`}>
              {queueStatus?.worker_available ? 'Ready' : 'Busy'}
            </p>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-3">
            <p className="text-slate-500">Next Scheduled</p>
            <p className="text-sm text-white">{formatRelativeTime(queueStatus?.next_scheduled)}</p>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-3">
            <p className="text-slate-500">Last Completed</p>
            <p className="text-sm text-white">{formatRelativeTime(queueStatus?.last_job_completed_at)}</p>
          </div>
        </div>
      </Card>

      {/* Recent Training Runs */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Recent Training Runs</h3>
        {history?.runs && history.runs.length > 0 ? (
          <div className="space-y-3">
            {history.runs.slice(0, 5).map((run) => (
              <div key={run.run_id} className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
                <div className="flex items-center gap-3">
                  <span className={`px-2 py-1 text-xs rounded-full ${
                    run.status === 'completed' ? 'bg-emerald-900/50 text-emerald-400' :
                    run.status === 'failed' ? 'bg-red-900/50 text-red-400' :
                    'bg-slate-700 text-slate-400'
                  }`}>
                    {run.status}
                  </span>
                  <span className="text-sm text-slate-300">{run.run_id.slice(0, 8)}</span>
                </div>
                <div className="text-right">
                  {run.metrics?.validation_auc && (
                    <span className="text-sm text-slate-300 mr-4">AUC: {(run.metrics.validation_auc * 100).toFixed(1)}%</span>
                  )}
                  <span className="text-xs text-slate-500">{formatRelativeTime(run.completed_at || run.started_at)}</span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-slate-500 text-center py-8">No training history yet</p>
        )}
      </Card>
    </div>
  );
}

// Configuration Tab - Settings & Baselines
function ConfigurationTab() {
  const queryClient = useQueryClient();
  const [showSuccess, setShowSuccess] = useState(false);

  const { data: config } = useQuery({
    queryKey: ['automation-config'],
    queryFn: () => api.getAutomationConfig(),
  });

  const { data: baselineSuggestions } = useQuery({
    queryKey: ['baselines', 'suggestions', 30],
    queryFn: () => api.getBaselineSuggestions(undefined, 30),
    refetchInterval: 60000,
  });

  const [localConfig, setLocalConfig] = useState<SchedulerConfig | null>(null);

  useEffect(() => {
    if (config) setLocalConfig(config);
  }, [config]);

  const updateConfigMutation = useMutation({
    mutationFn: (configUpdate: Partial<SchedulerConfig>) => api.updateAutomationConfig(configUpdate),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['automation-config'] });
      setShowSuccess(true);
      setTimeout(() => setShowSuccess(false), 3000);
    },
  });

  const handleSave = () => {
    if (localConfig) updateConfigMutation.mutate(localConfig);
  };

  const scheduleLabels: Record<ScheduleInterval, string> = {
    hourly: 'Every Hour',
    every_6_hours: 'Every 6 Hours',
    every_12_hours: 'Every 12 Hours',
    daily: 'Daily',
    weekly: 'Weekly',
    manual: 'Manual Only',
  };

  if (!localConfig) {
    return <div className="animate-pulse h-96 bg-slate-800 rounded-xl" />;
  }

  return (
    <div className="space-y-6">
      {/* Automation Settings */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-white">Automation Settings</h3>
          <button
            onClick={handleSave}
            disabled={updateConfigMutation.isPending}
            className="px-4 py-2 bg-stellar-600 hover:bg-stellar-500 disabled:bg-slate-700 text-white rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
          >
            {updateConfigMutation.isPending && <LoadingSpinner />}
            Save Changes
          </button>
        </div>

        <div className="space-y-6">
          {/* Training */}
          <div className="p-4 bg-slate-800/40 rounded-xl border border-slate-700/50">
            <div className="flex items-center justify-between mb-4">
              <div>
                <p className="text-sm font-medium text-white">Scheduled Training</p>
                <p className="text-xs text-slate-500">Automatically retrain on schedule</p>
              </div>
              <ToggleSwitch
                enabled={localConfig.training_enabled}
                onChange={() => setLocalConfig({ ...localConfig, training_enabled: !localConfig.training_enabled })}
                variant="stellar"
              />
            </div>
            {localConfig.training_enabled && (
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Schedule</label>
                  <select
                    value={localConfig.training_interval}
                    onChange={(e) => setLocalConfig({ ...localConfig, training_interval: e.target.value as ScheduleInterval })}
                    className="w-full px-3 py-2 bg-slate-900/50 border border-slate-600 rounded-lg text-slate-200 text-sm"
                  >
                    {Object.entries(scheduleLabels).map(([value, label]) => (
                      <option key={value} value={value}>{label}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Lookback Days</label>
                  <input
                    type="number"
                    value={localConfig.training_lookback_days}
                    onChange={(e) => setLocalConfig({ ...localConfig, training_lookback_days: parseInt(e.target.value) || 90 })}
                    className="w-full px-3 py-2 bg-slate-900/50 border border-slate-600 rounded-lg text-slate-200 text-sm"
                  />
                </div>
              </div>
            )}
          </div>

          {/* Scoring */}
          <div className="p-4 bg-slate-800/40 rounded-xl border border-slate-700/50">
            <div className="flex items-center justify-between mb-4">
              <div>
                <p className="text-sm font-medium text-white">Real-time Scoring</p>
                <p className="text-xs text-slate-500">Continuously score device data</p>
              </div>
              <ToggleSwitch
                enabled={localConfig.scoring_enabled}
                onChange={() => setLocalConfig({ ...localConfig, scoring_enabled: !localConfig.scoring_enabled })}
                variant="stellar"
              />
            </div>
            {localConfig.scoring_enabled && (
              <div>
                <label className="block text-xs text-slate-400 mb-1">Interval (minutes)</label>
                <input
                  type="number"
                  value={localConfig.scoring_interval_minutes}
                  onChange={(e) => setLocalConfig({ ...localConfig, scoring_interval_minutes: parseInt(e.target.value) || 15 })}
                  className="w-full px-3 py-2 bg-slate-900/50 border border-slate-600 rounded-lg text-slate-200 text-sm"
                />
              </div>
            )}
          </div>

          {/* Auto-Retrain */}
          <div className="p-4 bg-slate-800/40 rounded-xl border border-slate-700/50">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-white">Auto-Retrain on Feedback</p>
                <p className="text-xs text-slate-500">Retrain when false positive rate exceeds threshold</p>
              </div>
              <ToggleSwitch
                enabled={localConfig.auto_retrain_enabled}
                onChange={() => setLocalConfig({ ...localConfig, auto_retrain_enabled: !localConfig.auto_retrain_enabled })}
                variant="stellar"
              />
            </div>
          </div>
        </div>
      </Card>

      {/* Baseline Drift */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Baseline Drift Alerts</h3>
        {baselineSuggestions && baselineSuggestions.length > 0 ? (
          <div className="space-y-3">
            {baselineSuggestions.slice(0, 5).map((suggestion, i) => (
              <div key={i} className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
                <div>
                  <p className="text-sm text-white">{suggestion.feature}</p>
                  <p className="text-xs text-slate-500">{suggestion.rationale}</p>
                </div>
                <div className="text-right">
                  <p className="text-sm text-slate-300">
                    {suggestion.baseline_median?.toFixed(2)} → {suggestion.proposed_new_median?.toFixed(2)}
                  </p>
                </div>
              </div>
            ))}
            <p className="text-xs text-slate-500 text-center pt-2">
              {baselineSuggestions.length} features with suggested adjustments
            </p>
          </div>
        ) : (
          <p className="text-slate-500 text-center py-8">No baseline drift detected</p>
        )}
      </Card>

      {/* Location Settings */}
      <LocationAttributeSettings />

      {/* Success Toast */}
      <AnimatePresence>
        {showSuccess && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="fixed bottom-6 right-6 px-4 py-3 bg-emerald-900/90 border border-emerald-500/30 rounded-lg shadow-lg"
          >
            <p className="text-sm text-emerald-400">Configuration saved successfully</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// Main System Component
function System() {
  const [activeTab, setActiveTab] = useState<TabId>('overview');

  const { data: connections } = useQuery({
    queryKey: ['dashboard', 'connections'],
    queryFn: () => api.getConnectionStatus(),
    refetchInterval: 30000,
  });

  const connectedCount = connections
    ? [connections.backend_db, connections.redis, connections.qdrant, connections.dw_sql, connections.mc_sql, connections.mobicontrol_api, connections.llm].filter(s => s?.connected).length
    : 0;

  return (
    <motion.div className="space-y-6" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
      {/* Header */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">System</h1>
          <p className="text-slate-500 mt-1">Connections, ML pipeline, and configuration</p>
        </div>

        {/* Overall Status */}
        <div className="flex items-center gap-4 px-5 py-3 stellar-card rounded-xl">
          <div className={`w-3 h-3 rounded-full ${
            connectedCount === 7
              ? 'bg-emerald-400 shadow-[0_0_8px_rgba(16,185,129,0.6)]'
              : connectedCount >= 4
                ? 'bg-orange-400 animate-pulse shadow-[0_0_8px_rgba(251,146,60,0.6)]'
                : 'bg-red-400 animate-pulse shadow-[0_0_8px_rgba(248,113,113,0.6)]'
          }`} />
          <div>
            <p className="text-sm font-medium text-white">
              {connectedCount === 7 ? 'All Systems Operational' : connectedCount >= 4 ? 'Partial Connectivity' : 'Critical: Services Offline'}
            </p>
            <p className="text-xs text-slate-500">{connectedCount}/7 services connected</p>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-slate-700/50 pb-2">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2.5 rounded-lg text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? 'bg-stellar-600/20 text-stellar-400 border border-stellar-500/30'
                : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.2 }}
        >
          {activeTab === 'overview' && <OverviewTab />}
          {activeTab === 'ml-pipeline' && <MLPipelineTab />}
          {activeTab === 'configuration' && <ConfigurationTab />}
        </motion.div>
      </AnimatePresence>
    </motion.div>
  );
}

export default System;
