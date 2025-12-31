/**
 * Automation Dashboard - ML Scheduler & Real-time Scoring Control
 *
 * Comprehensive UI for automated ML operations:
 * - Scheduler status with live indicators
 * - Configuration for training schedules, scoring intervals
 * - Auto-retrain settings based on feedback
 * - Alert monitoring and job history
 */

import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../api/client';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from '../components/Card';
import { ToggleSwitch } from '../components/ui';
import type {
  SchedulerStatus,
  SchedulerConfig,
  SchedulerConfigUpdate,
  ScheduleInterval,
  AutomationAlert,
  AutomationJob,
} from '../types/automation';

// Format relative time
function formatRelativeTime(dateStr: string | null | undefined): string {
  if (!dateStr) return 'Never';
  const date = new Date(dateStr);
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  const days = Math.floor(diff / 86400000);

  if (diff < 0) {
    // Future time
    const futureMinutes = Math.abs(minutes);
    const futureHours = Math.abs(hours);
    if (futureMinutes < 60) return `in ${futureMinutes}m`;
    if (futureHours < 24) return `in ${futureHours}h`;
    return date.toLocaleString();
  }

  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days < 7) return `${days}d ago`;
  return date.toLocaleDateString();
}

// Schedule interval display names
const scheduleLabels: Record<ScheduleInterval, string> = {
  hourly: 'Every Hour',
  every_6_hours: 'Every 6 Hours',
  every_12_hours: 'Every 12 Hours',
  daily: 'Daily',
  weekly: 'Weekly',
  manual: 'Manual Only',
};

// Status indicator component
function StatusIndicator({
  active,
  label,
  description,
  pulsing = false,
}: {
  active: boolean;
  label: string;
  description?: string;
  pulsing?: boolean;
}) {
  return (
    <div className="flex items-center gap-3">
      <motion.div
        className={`w-3 h-3 rounded-full ${
          active
            ? 'bg-emerald-400 shadow-[0_0_10px_rgba(16,185,129,0.6)]'
            : 'bg-slate-600'
        }`}
        animate={pulsing && active ? { scale: [1, 1.2, 1] } : {}}
        transition={{ duration: 1.5, repeat: Infinity }}
      />
      <div>
        <p className="text-sm font-medium text-slate-200">{label}</p>
        {description && (
          <p className="text-xs text-slate-500">{description}</p>
        )}
      </div>
    </div>
  );
}

// Main status card with live indicators
function AutomationStatusCard({ status }: { status: SchedulerStatus }) {
  const trainingRunning = status.training_status === 'running';
  const scoringRunning = status.scoring_status === 'running';
  const devicesScored = status.last_scoring_result?.total_scored ?? 0;
  const anomaliesDetected = status.last_scoring_result?.anomalies_detected ?? 0;

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-lg font-semibold text-slate-200">Automation Status</h2>
          <p className="text-sm text-slate-500">Real-time ML operations monitoring</p>
        </div>
        <div
          className={`px-4 py-2 rounded-full text-sm font-semibold ${
            status.is_running
              ? 'bg-emerald-900/50 text-emerald-400 border border-emerald-500/30'
              : 'bg-slate-800 text-slate-400 border border-slate-600'
          }`}
        >
          {status.is_running ? 'ACTIVE' : 'STOPPED'}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6 mb-6">
        <StatusIndicator
          active={trainingRunning}
          label="Training Loop"
          description={
            status.last_training_result?.timestamp
              ? `Last: ${formatRelativeTime(status.last_training_result.timestamp)}`
              : 'Not started'
          }
          pulsing={trainingRunning}
        />
        <StatusIndicator
          active={scoringRunning}
          label="Scoring Loop"
          description={
            status.last_scoring_result?.timestamp
              ? `Last: ${formatRelativeTime(status.last_scoring_result.timestamp)}`
              : 'Not started'
          }
          pulsing={scoringRunning}
        />
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-4 gap-4 pt-4 border-t border-slate-700">
        <div className="text-center">
          <p className="text-2xl font-bold text-stellar-400">
            {devicesScored}
          </p>
          <p className="text-xs text-slate-500">Devices Scored</p>
        </div>
        <div className="text-center">
          <p className="text-2xl font-bold text-cosmic-400">
            {anomaliesDetected}
          </p>
          <p className="text-xs text-slate-500">Anomalies Found</p>
        </div>
        <div className="text-center">
          <p className="text-2xl font-bold text-aurora-400">
            {status.total_anomalies_detected}
          </p>
          <p className="text-xs text-slate-500">Total Anomalies</p>
        </div>
        <div className="text-center">
          <p className="text-sm font-medium text-slate-300">
            {status.next_training_time
              ? formatRelativeTime(status.next_training_time)
              : 'Not scheduled'}
          </p>
          <p className="text-xs text-slate-500">Next Training</p>
        </div>
      </div>

      {/* Errors if any */}
      {status.errors.length > 0 && (
        <div className="mt-4 p-3 bg-red-900/20 border border-red-500/30 rounded-lg">
          <p className="text-xs text-red-400 font-medium mb-1">Recent Errors:</p>
          <p className="text-xs text-red-300">{status.errors[status.errors.length - 1]}</p>
        </div>
      )}
    </Card>
  );
}

// Consistent setting row with toggle
function SettingRow({
  label,
  description,
  enabled,
  onToggle,
  children,
}: {
  label: string;
  description: string;
  enabled: boolean;
  onToggle: () => void;
  children?: React.ReactNode;
}) {
  return (
    <div className="p-4 bg-slate-800/40 rounded-xl border border-slate-700/50">
      <div className="flex items-center justify-between">
        <div className="flex-1 min-w-0 mr-4">
          <p className="text-sm font-medium text-slate-200">{label}</p>
          <p className="text-xs text-slate-500 mt-0.5">{description}</p>
        </div>
        <ToggleSwitch enabled={enabled} onChange={() => onToggle()} variant="stellar" />
      </div>
      {children && (
        <div className={`mt-4 pt-4 border-t border-slate-700/50 ${!enabled ? 'opacity-50' : ''}`}>
          {children}
        </div>
      )}
    </div>
  );
}

// Configuration form component
function ConfigurationForm({
  config,
  onUpdate,
  isUpdating,
}: {
  config: SchedulerConfig;
  onUpdate: (config: SchedulerConfigUpdate) => void;
  isUpdating: boolean;
}) {
  const [localConfig, setLocalConfig] = useState(config);
  const [hasChanges, setHasChanges] = useState(false);

  // Update local config when prop changes
  useEffect(() => {
    setLocalConfig(config);
  }, [config]);

  const handleChange = <K extends keyof SchedulerConfig>(
    key: K,
    value: SchedulerConfig[K]
  ) => {
    setLocalConfig({ ...localConfig, [key]: value });
    setHasChanges(true);
  };

  const handleSave = () => {
    onUpdate(localConfig);
    setHasChanges(false);
  };

  const handleReset = () => {
    setLocalConfig(config);
    setHasChanges(false);
  };

  const inputClasses = "w-full px-3 py-2.5 bg-slate-900/50 border border-slate-600 rounded-lg text-slate-200 text-sm focus:border-stellar-500 focus:outline-none focus:ring-1 focus:ring-stellar-500 disabled:opacity-50 disabled:cursor-not-allowed";

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-lg font-semibold text-slate-200">Configuration</h2>
          <p className="text-sm text-slate-500">Customize automation behavior</p>
        </div>
        <div className="flex gap-2">
          {hasChanges && (
            <button
              onClick={handleReset}
              className="px-4 py-2 text-sm text-slate-400 hover:text-slate-200 transition-colors"
            >
              Reset
            </button>
          )}
          <button
            onClick={handleSave}
            disabled={isUpdating || !hasChanges}
            className="px-4 py-2 bg-stellar-600 hover:bg-stellar-500 disabled:bg-slate-700 disabled:text-slate-500 text-white rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
          >
            {isUpdating && (
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            )}
            Save Changes
          </button>
        </div>
      </div>

      <div className="space-y-4">
        {/* Training Settings */}
        <SettingRow
          label="Scheduled Training"
          description="Automatically retrain the model on a schedule"
          enabled={localConfig.training_enabled}
          onToggle={() => handleChange('training_enabled', !localConfig.training_enabled)}
        >
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">
                Schedule
              </label>
              <select
                value={localConfig.training_interval}
                onChange={(e) =>
                  handleChange('training_interval', e.target.value as ScheduleInterval)
                }
                disabled={!localConfig.training_enabled}
                className={inputClasses}
              >
                {Object.entries(scheduleLabels).map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">
                Lookback Days
              </label>
              <input
                type="number"
                value={localConfig.training_lookback_days}
                onChange={(e) =>
                  handleChange('training_lookback_days', parseInt(e.target.value) || 90)
                }
                min={7}
                max={365}
                disabled={!localConfig.training_enabled}
                className={inputClasses}
              />
            </div>
          </div>
        </SettingRow>

        {/* Scoring Settings */}
        <SettingRow
          label="Real-time Scoring"
          description="Continuously score device data for anomalies"
          enabled={localConfig.scoring_enabled}
          onToggle={() => handleChange('scoring_enabled', !localConfig.scoring_enabled)}
        >
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1.5">
              Scoring Interval (minutes)
            </label>
            <input
              type="number"
              value={localConfig.scoring_interval_minutes}
              onChange={(e) =>
                handleChange('scoring_interval_minutes', parseInt(e.target.value) || 15)
              }
              min={5}
              max={1440}
              disabled={!localConfig.scoring_enabled}
              className={inputClasses}
            />
            <p className="text-xs text-slate-500 mt-1.5">
              How often to check for new data and score devices
            </p>
          </div>
        </SettingRow>

        {/* Auto-Retrain Settings */}
        <SettingRow
          label="Auto-Retrain on Feedback"
          description="Automatically retrain when false positive rate exceeds threshold"
          enabled={localConfig.auto_retrain_enabled}
          onToggle={() => handleChange('auto_retrain_enabled', !localConfig.auto_retrain_enabled)}
        >
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">
                FP Threshold (%)
              </label>
              <input
                type="number"
                value={Math.round(localConfig.auto_retrain_fp_threshold * 100)}
                onChange={(e) =>
                  handleChange(
                    'auto_retrain_fp_threshold',
                    (parseInt(e.target.value) || 15) / 100
                  )
                }
                min={1}
                max={50}
                disabled={!localConfig.auto_retrain_enabled}
                className={inputClasses}
              />
              <p className="text-xs text-slate-500 mt-1.5">
                Trigger retrain above this rate
              </p>
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">
                Min Feedback Items
              </label>
              <input
                type="number"
                value={localConfig.auto_retrain_min_feedback}
                onChange={(e) =>
                  handleChange(
                    'auto_retrain_min_feedback',
                    parseInt(e.target.value) || 50
                  )
                }
                min={10}
                max={1000}
                disabled={!localConfig.auto_retrain_enabled}
                className={inputClasses}
              />
              <p className="text-xs text-slate-500 mt-1.5">
                Required before evaluation
              </p>
            </div>
          </div>
        </SettingRow>

        {/* Alert Settings */}
        <SettingRow
          label="High Anomaly Rate Alerts"
          description="Get notified when anomaly rate spikes"
          enabled={localConfig.alert_on_high_anomaly_rate}
          onToggle={() =>
            handleChange('alert_on_high_anomaly_rate', !localConfig.alert_on_high_anomaly_rate)
          }
        >
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1.5">
              Alert Threshold (%)
            </label>
            <input
              type="number"
              value={Math.round(localConfig.high_anomaly_rate_threshold * 100)}
              onChange={(e) =>
                handleChange(
                  'high_anomaly_rate_threshold',
                  (parseFloat(e.target.value) || 10) / 100
                )
              }
              min={1}
              max={50}
              step={1}
              disabled={!localConfig.alert_on_high_anomaly_rate}
              className={inputClasses}
            />
            <p className="text-xs text-slate-500 mt-1.5">
              Alert when anomaly rate exceeds this percentage
            </p>
          </div>
        </SettingRow>
      </div>
    </Card>
  );
}

// Quick actions panel
function QuickActionsPanel({
  onTrigger,
  isTriggering,
}: {
  onTrigger: (jobType: 'training' | 'scoring') => void;
  isTriggering: boolean;
}) {
  return (
    <Card className="p-6">
      <h2 className="text-lg font-semibold text-slate-200 mb-4">Quick Actions</h2>
      <div className="grid grid-cols-2 gap-4">
        <button
          onClick={() => onTrigger('training')}
          disabled={isTriggering}
          className="p-4 bg-gradient-to-br from-stellar-600/20 to-stellar-700/20 hover:from-stellar-600/30 hover:to-stellar-700/30 border border-stellar-500/30 rounded-xl transition-all disabled:opacity-50"
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-stellar-500/20 flex items-center justify-center">
              <svg
                className="w-5 h-5 text-stellar-400"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                />
              </svg>
            </div>
            <div className="text-left">
              <p className="text-sm font-medium text-slate-200">Trigger Training</p>
              <p className="text-xs text-slate-500">Start model training now</p>
            </div>
          </div>
        </button>
        <button
          onClick={() => onTrigger('scoring')}
          disabled={isTriggering}
          className="p-4 bg-gradient-to-br from-aurora-600/20 to-aurora-700/20 hover:from-aurora-600/30 hover:to-aurora-700/30 border border-aurora-500/30 rounded-xl transition-all disabled:opacity-50"
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-aurora-500/20 flex items-center justify-center">
              <svg
                className="w-5 h-5 text-aurora-400"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                />
              </svg>
            </div>
            <div className="text-left">
              <p className="text-sm font-medium text-slate-200">Score All Devices</p>
              <p className="text-xs text-slate-500">Run anomaly detection now</p>
            </div>
          </div>
        </button>
      </div>
    </Card>
  );
}

// Alert list component
function AlertsList({ alerts }: { alerts: AutomationAlert[] }) {
  const queryClient = useQueryClient();
  const acknowledgeMutation = useMutation({
    mutationFn: (alertIndex: string) => api.acknowledgeAlert(alertIndex),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['automation-alerts'] });
    },
  });

  if (alerts.length === 0) {
    return (
      <div className="p-8 text-center">
        <svg
          className="w-12 h-12 text-slate-600 mx-auto mb-4"
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
        <p className="text-slate-400">No alerts</p>
        <p className="text-sm text-slate-500">All systems operating normally</p>
      </div>
    );
  }

  return (
    <div className="divide-y divide-slate-700/50">
      {alerts.map((alert, index) => (
        <div
          key={`${alert.timestamp}-${index}`}
          className={`p-4 ${alert.acknowledged ? 'opacity-60' : ''}`}
        >
          <div className="flex items-start gap-3">
            <div className="p-2 rounded-lg bg-orange-900/30 border border-orange-500/30 text-orange-400">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                />
              </svg>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-slate-200">{alert.message}</p>
              <p className="text-xs text-slate-500 mt-1">
                {formatRelativeTime(alert.timestamp)}
              </p>
            </div>
            {!alert.acknowledged && (
              <button
                onClick={() => acknowledgeMutation.mutate(String(index))}
                disabled={acknowledgeMutation.isPending}
                className="px-3 py-1 text-xs text-slate-400 hover:text-slate-200 hover:bg-slate-700 rounded transition-colors"
              >
                Dismiss
              </button>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

// Job history component
function JobHistory({ jobs }: { jobs: AutomationJob[] }) {
  const typeLabels: Record<string, string> = {
    training: 'Training',
    scoring: 'Scoring',
    auto_retrain: 'Auto-Retrain',
  };

  if (jobs.length === 0) {
    return (
      <div className="p-8 text-center">
        <svg
          className="w-12 h-12 text-slate-600 mx-auto mb-4"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
        <p className="text-slate-400">No job history</p>
        <p className="text-sm text-slate-500">Jobs will appear here once executed</p>
      </div>
    );
  }

  return (
    <div className="divide-y divide-slate-700/50">
      {jobs.map((job, index) => {
        const isSuccess = job.success !== false;
        const statusColor = isSuccess
          ? 'bg-emerald-900/50 text-emerald-400 border-emerald-500/30'
          : 'bg-red-900/50 text-red-400 border-red-500/30';

        return (
          <div key={`${job.timestamp}-${index}`} className="p-4 hover:bg-slate-800/30 transition-colors">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span
                  className={`px-2.5 py-1 text-xs font-medium rounded-full border ${statusColor}`}
                >
                  {isSuccess ? 'Completed' : 'Failed'}
                </span>
                <div>
                  <p className="text-sm font-medium text-slate-200">
                    {typeLabels[job.type] || job.type}
                  </p>
                  <p className="text-xs text-slate-500">
                    {job.triggered_by === 'manual' ? 'Manual trigger' : 'Scheduled'}
                  </p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-xs text-slate-400">
                  {formatRelativeTime(job.timestamp)}
                </p>
                {job.error && (
                  <p className="text-xs text-red-400 mt-1 max-w-[200px] truncate">
                    {job.error}
                  </p>
                )}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// Main component
export default function Automation() {
  const queryClient = useQueryClient();

  // Fetch automation status with polling
  const { data: status, isLoading: statusLoading } = useQuery({
    queryKey: ['automation-status'],
    queryFn: () => api.getAutomationStatus(),
    refetchInterval: 5000, // Poll every 5 seconds
  });

  // Fetch automation config
  const { data: config, isLoading: configLoading } = useQuery({
    queryKey: ['automation-config'],
    queryFn: () => api.getAutomationConfig(),
  });

  // Fetch alerts
  const { data: alerts } = useQuery({
    queryKey: ['automation-alerts'],
    queryFn: () => api.getAutomationAlerts(20),
    refetchInterval: 10000,
  });

  // Fetch job history
  const { data: history } = useQuery({
    queryKey: ['automation-history'],
    queryFn: () => api.getAutomationHistory(10),
    refetchInterval: 10000,
  });

  // Update config mutation
  const updateConfigMutation = useMutation({
    mutationFn: (configUpdate: SchedulerConfigUpdate) => api.updateAutomationConfig(configUpdate),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['automation-config'] });
      queryClient.invalidateQueries({ queryKey: ['automation-status'] });
    },
  });

  // Trigger job mutation
  const triggerJobMutation = useMutation({
    mutationFn: (jobType: 'training' | 'scoring') =>
      api.triggerAutomationJob({ job_type: jobType }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['automation-status'] });
      queryClient.invalidateQueries({ queryKey: ['automation-history'] });
    },
  });

  if (statusLoading || configLoading || !status || !config) {
    return (
      <div className="space-y-6">
        <div className="animate-pulse">
          <div className="h-8 w-48 bg-slate-700 rounded mb-2" />
          <div className="h-4 w-64 bg-slate-700 rounded" />
        </div>
        <div className="grid grid-cols-2 gap-6">
          <div className="h-64 bg-slate-800 rounded-xl animate-pulse" />
          <div className="h-64 bg-slate-800 rounded-xl animate-pulse" />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-100">ML Automation</h1>
        <p className="text-slate-400 mt-1">
          Configure automated training, scoring, and model management
        </p>
      </div>

      {/* Status and Quick Actions */}
      <div className="grid grid-cols-3 gap-6">
        <div className="col-span-2">
          <AutomationStatusCard status={status} />
        </div>
        <QuickActionsPanel
          onTrigger={(type) => triggerJobMutation.mutate(type)}
          isTriggering={triggerJobMutation.isPending}
        />
      </div>

      {/* Configuration */}
      <ConfigurationForm
        config={config}
        onUpdate={(configUpdate) => updateConfigMutation.mutate(configUpdate)}
        isUpdating={updateConfigMutation.isPending}
      />

      {/* Alerts and History */}
      <div className="grid grid-cols-2 gap-6">
        <Card className="overflow-hidden">
          <div className="p-4 border-b border-slate-700">
            <h2 className="text-lg font-semibold text-slate-200">Recent Alerts</h2>
          </div>
          <AlertsList alerts={alerts || []} />
        </Card>

        <Card className="overflow-hidden">
          <div className="p-4 border-b border-slate-700">
            <h2 className="text-lg font-semibold text-slate-200">Job History</h2>
          </div>
          <JobHistory jobs={history || []} />
        </Card>
      </div>

      {/* Success/Error notifications */}
      <AnimatePresence>
        {updateConfigMutation.isSuccess && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="fixed bottom-6 right-6 px-4 py-3 bg-emerald-900/90 border border-emerald-500/30 rounded-lg shadow-lg"
          >
            <p className="text-sm text-emerald-400">Configuration saved successfully</p>
          </motion.div>
        )}
        {triggerJobMutation.isSuccess && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="fixed bottom-6 right-6 px-4 py-3 bg-stellar-900/90 border border-stellar-500/30 rounded-lg shadow-lg"
          >
            <p className="text-sm text-stellar-400">Job triggered successfully</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
