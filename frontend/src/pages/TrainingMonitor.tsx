/**
 * Training Monitor Page - ML Model Training & Observability
 *
 * Comprehensive UI for ML training with:
 * - Current training status with detailed stage progress
 * - Training history with expandable metrics
 * - Queue status for scheduled jobs
 * - Feature importance visualization
 * - Model performance metrics
 */

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../api/client';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from '../components/Card';
import { Link } from 'react-router-dom';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import type {
  TrainingRun,
  TrainingConfigRequest,
  TrainingMetrics,
  TrainingStage,
} from '../types/training';
import { formatAUC, formatAvgAUC } from '../lib/formatters';

// Format large numbers with K, M, B suffixes
function formatNumber(num: number): string {
  if (num >= 1_000_000_000) return (num / 1_000_000_000).toFixed(1) + 'B';
  if (num >= 1_000_000) return (num / 1_000_000).toFixed(1) + 'M';
  if (num >= 1_000) return (num / 1_000).toFixed(1) + 'K';
  return num.toLocaleString();
}

// Format date string to relative time
function formatRelativeTime(dateStr: string | null | undefined): string {
  if (!dateStr) return 'N/A';
  const date = new Date(dateStr);
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  const days = Math.floor(diff / 86400000);

  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days < 7) return `${days}d ago`;
  return date.toLocaleDateString();
}

// Format duration
function formatDuration(startStr: string | null | undefined, endStr: string | null | undefined): string {
  if (!startStr || !endStr) return 'N/A';
  const start = new Date(startStr);
  const end = new Date(endStr);
  const diff = end.getTime() - start.getTime();
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);

  if (minutes < 60) return `${minutes}m`;
  return `${hours}h ${minutes % 60}m`;
}

// Training status badge with animation
function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    pending: 'bg-yellow-900/50 text-yellow-400 border-yellow-500/30',
    running: 'bg-stellar-900/50 text-stellar-400 border-stellar-500/30',
    completed: 'bg-emerald-900/50 text-emerald-400 border-emerald-500/30',
    failed: 'bg-red-900/50 text-red-400 border-red-500/30',
    idle: 'bg-slate-800 text-slate-400 border-slate-600',
  };

  return (
    <span
      className={`px-3 py-1.5 text-xs font-semibold rounded-full border ${colors[status] || colors.idle} flex items-center gap-2`}
    >
      {status === 'running' && (
        <span className="w-2 h-2 bg-stellar-400 rounded-full animate-pulse" />
      )}
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
}

// Stage progress component with detailed view
function StageProgress({ stages }: { stages?: TrainingStage[] | null; currentStage?: string | null }) {
  const defaultStages = [
    { name: 'Initialize', status: 'pending' as const },
    { name: 'Load Data', status: 'pending' as const },
    { name: 'Features', status: 'pending' as const },
    { name: 'Training', status: 'pending' as const },
    { name: 'Validation', status: 'pending' as const },
    { name: 'Export', status: 'pending' as const },
  ];

  const displayStages = stages && stages.length > 0 ? stages : defaultStages;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        {displayStages.map((stage, i) => (
          <div key={stage.name} className="flex flex-col items-center flex-1">
            {/* Connector line */}
            {i > 0 && (
              <div
                className={`absolute h-0.5 -translate-y-3 ${
                  stage.status === 'completed' || stage.status === 'running'
                    ? 'bg-stellar-500'
                    : 'bg-slate-700'
                }`}
                style={{
                  width: `calc(100% / ${displayStages.length} - 24px)`,
                  left: `calc(${(i - 0.5) * (100 / displayStages.length)}%)`,
                }}
              />
            )}

            {/* Stage indicator */}
            <motion.div
              initial={{ scale: 0.8 }}
              animate={{
                scale: stage.status === 'running' ? [1, 1.2, 1] : 1,
              }}
              transition={{
                repeat: stage.status === 'running' ? Infinity : 0,
                duration: 1.5,
              }}
              className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold z-10 ${
                stage.status === 'completed'
                  ? 'bg-stellar-500 text-white'
                  : stage.status === 'running'
                  ? 'bg-stellar-400 text-white ring-4 ring-stellar-500/30'
                  : stage.status === 'failed'
                  ? 'bg-red-500 text-white'
                  : 'bg-slate-700 text-slate-400'
              }`}
            >
              {stage.status === 'completed' ? (
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                </svg>
              ) : stage.status === 'failed' ? (
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M6 18L18 6M6 6l12 12" />
                </svg>
              ) : stage.status === 'running' ? (
                <div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : (
                i + 1
              )}
            </motion.div>

            {/* Stage name */}
            <span
              className={`mt-2 text-xs font-medium ${
                stage.status === 'completed' || stage.status === 'running'
                  ? 'text-slate-200'
                  : 'text-slate-500'
              }`}
            >
              {stage.name}
            </span>

            {/* Stage message */}
            {'message' in stage && stage.message && stage.status === 'running' && (
              <span className="text-[10px] text-slate-400 mt-1 max-w-[80px] truncate text-center">
                {stage.message}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// Training progress card with live updates
function TrainingProgressCard({ run }: { run: TrainingRun }) {
  const isRunning = run.status === 'running';
  const isPending = run.status === 'pending';

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-lg font-semibold text-slate-200">
            {isRunning ? 'Training in Progress' : isPending ? 'Training Queued' : 'Current Status'}
          </h2>
          <p className="text-sm text-slate-500">
            Run ID: {run.run_id}
          </p>
        </div>
        <StatusBadge status={run.status} />
      </div>

      {/* Progress bar */}
      {(isRunning || isPending) && (
        <div className="mb-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-slate-400">{run.message}</span>
            <span className="text-lg font-bold text-stellar-400">
              {Math.round(run.progress)}%
            </span>
          </div>
          <div className="h-3 bg-slate-800 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-stellar-600 via-stellar-500 to-stellar-400"
              initial={{ width: 0 }}
              animate={{ width: `${run.progress}%` }}
              transition={{ duration: 0.5, ease: 'easeOut' }}
            />
          </div>
          {run.estimated_completion && isRunning && (
            <p className="text-xs text-slate-500 mt-2">
              Estimated completion: {new Date(run.estimated_completion).toLocaleTimeString()}
            </p>
          )}
        </div>
      )}

      {/* Stage progress */}
      {(isRunning || run.status === 'completed') && (
        <div className="relative pt-2">
          <StageProgress stages={run.stages} currentStage={run.stage} />
        </div>
      )}

      {/* Error display */}
      {run.status === 'failed' && (
        <div className="p-4 bg-red-900/20 border border-red-500/30 rounded-lg mt-4">
          <div className="flex items-start gap-3">
            <svg className="w-5 h-5 text-red-400 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div>
              <p className="text-red-400 font-medium">Training Failed</p>
              <p className="text-sm text-red-300 mt-1">
                {run.error || run.message || 'Unknown error occurred'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Timing info */}
      <div className="flex items-center gap-6 mt-6 pt-4 border-t border-slate-700 text-sm">
        <div>
          <span className="text-slate-500">Started: </span>
          <span className="text-slate-300">{formatRelativeTime(run.started_at)}</span>
        </div>
        {run.completed_at && (
          <>
            <div>
              <span className="text-slate-500">Duration: </span>
              <span className="text-slate-300">{formatDuration(run.started_at, run.completed_at)}</span>
            </div>
          </>
        )}
      </div>
    </Card>
  );
}

// Feature importance chart
function FeatureImportanceChart({ metrics }: { metrics: TrainingMetrics }) {
  if (!metrics.feature_importance || Object.keys(metrics.feature_importance).length === 0) {
    return (
      <div className="h-48 flex items-center justify-center text-slate-500">
        No feature importance data
      </div>
    );
  }

  const data = Object.entries(metrics.feature_importance)
    .map(([name, value]) => ({ name, importance: value * 100 }))
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 10);

  return (
    <div className="h-72">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 120 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis
            type="number"
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            tickFormatter={(v) => `${v.toFixed(0)}%`}
          />
          <YAxis
            dataKey="name"
            type="category"
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            width={110}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1e293b',
              border: '1px solid #334155',
              borderRadius: '8px',
            }}
            formatter={(value: number) => [`${value.toFixed(1)}%`, 'Importance']}
          />
          <Bar dataKey="importance" fill="#6366f1" radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// Training metrics display with visual indicators
function MetricsDisplay({ metrics }: { metrics: TrainingMetrics }) {
  const anomalyData = [
    { name: 'Normal', value: 100 - metrics.anomaly_rate_train * 100, color: '#10b981' },
    { name: 'Anomaly', value: metrics.anomaly_rate_train * 100, color: '#f43f5e' },
  ];

  // AUC gauge data
  const aucValue = metrics.validation_auc || 0;
  const aucColor = aucValue >= 0.85 ? '#10b981' : aucValue >= 0.7 ? '#f59e0b' : '#ef4444';

  return (
    <div className="space-y-6">
      {/* Key metrics row */}
      <div className="grid grid-cols-4 gap-4">
        <Card className="p-4 bg-gradient-to-br from-slate-800 to-slate-900">
          <p className="text-xs text-slate-500 uppercase tracking-wide">Training Rows</p>
          <p className="text-2xl font-bold text-stellar-400 mt-1">
            {formatNumber(metrics.train_rows)}
          </p>
          <p className="text-xs text-slate-500 mt-1">samples processed</p>
        </Card>
        <Card className="p-4 bg-gradient-to-br from-slate-800 to-slate-900">
          <p className="text-xs text-slate-500 uppercase tracking-wide">Validation Rows</p>
          <p className="text-2xl font-bold text-aurora-400 mt-1">
            {formatNumber(metrics.validation_rows)}
          </p>
          <p className="text-xs text-slate-500 mt-1">holdout samples</p>
        </Card>
        <Card className="p-4 bg-gradient-to-br from-slate-800 to-slate-900">
          <p className="text-xs text-slate-500 uppercase tracking-wide">Features</p>
          <p className="text-2xl font-bold text-nebula-400 mt-1">
            {metrics.feature_count}
          </p>
          <p className="text-xs text-slate-500 mt-1">engineered</p>
        </Card>
        <Card className="p-4 bg-gradient-to-br from-slate-800 to-slate-900">
          <p className="text-xs text-slate-500 uppercase tracking-wide">Anomaly Rate</p>
          <p className="text-2xl font-bold text-cosmic-400 mt-1">
            {(metrics.anomaly_rate_train * 100).toFixed(1)}%
          </p>
          <p className="text-xs text-slate-500 mt-1">in training data</p>
        </Card>
      </div>

      {/* Model performance section */}
      <div className="grid grid-cols-2 gap-6">
        {/* Left: Performance metrics */}
        <Card className="p-6">
          <h3 className="text-sm font-semibold text-slate-300 mb-4 uppercase tracking-wide">
            Model Performance
          </h3>

          {/* AUC display */}
          <div className="flex items-center gap-6 mb-6">
            <div className="relative w-24 h-24">
              <svg className="w-24 h-24 transform -rotate-90">
                <circle
                  cx="48"
                  cy="48"
                  r="40"
                  stroke="#1e293b"
                  strokeWidth="8"
                  fill="none"
                />
                <circle
                  cx="48"
                  cy="48"
                  r="40"
                  stroke={aucColor}
                  strokeWidth="8"
                  fill="none"
                  strokeDasharray={`${aucValue * 251.2} 251.2`}
                  strokeLinecap="round"
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-xl font-bold text-white">
                  {formatAUC(metrics.validation_auc)}
                </span>
              </div>
            </div>
            <div>
              <p className="text-lg font-semibold text-slate-200">Validation AUC</p>
              <p className="text-sm text-slate-500">
                {aucValue >= 0.85 ? 'Excellent' : aucValue >= 0.7 ? 'Good' : 'Needs improvement'}
              </p>
            </div>
          </div>

          {/* Additional metrics */}
          {metrics.precision_at_recall_80 !== undefined && metrics.precision_at_recall_80 !== null && (
            <div className="p-4 bg-slate-800/50 rounded-lg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-400">Precision @ 80% Recall</p>
                  <p className="text-xl font-bold text-slate-200 mt-1">
                    {(metrics.precision_at_recall_80 * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="w-20 h-20">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={anomalyData}
                        dataKey="value"
                        cx="50%"
                        cy="50%"
                        innerRadius={25}
                        outerRadius={35}
                        paddingAngle={2}
                      >
                        {anomalyData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}
        </Card>

        {/* Right: Feature importance */}
        <Card className="p-6">
          <h3 className="text-sm font-semibold text-slate-300 mb-4 uppercase tracking-wide">
            Top Features
          </h3>
          <FeatureImportanceChart metrics={metrics} />
        </Card>
      </div>
    </div>
  );
}

// Training configuration form
function TrainingConfigForm({
  onSubmit,
  isSubmitting,
  onCancel,
}: {
  onSubmit: (config: TrainingConfigRequest) => void;
  isSubmitting: boolean;
  onCancel: () => void;
}) {
  const [config, setConfig] = useState<TrainingConfigRequest>({
    start_date: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    end_date: new Date().toISOString().split('T')[0],
    validation_days: 7,
    contamination: 0.03,
    n_estimators: 300,
    export_onnx: true,
  });

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit(config);
      }}
      className="space-y-6"
    >
      {/* Date range */}
      <div>
        <h3 className="text-sm font-medium text-slate-300 mb-3">Training Data Range</h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-slate-500 mb-1">Start Date</label>
            <input
              type="date"
              value={config.start_date}
              onChange={(e) => setConfig({ ...config, start_date: e.target.value })}
              className="w-full p-3 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 focus:border-stellar-500 focus:outline-none focus:ring-1 focus:ring-stellar-500"
            />
          </div>
          <div>
            <label className="block text-xs text-slate-500 mb-1">End Date</label>
            <input
              type="date"
              value={config.end_date}
              onChange={(e) => setConfig({ ...config, end_date: e.target.value })}
              className="w-full p-3 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 focus:border-stellar-500 focus:outline-none focus:ring-1 focus:ring-stellar-500"
            />
          </div>
        </div>
      </div>

      {/* Model parameters */}
      <div>
        <h3 className="text-sm font-medium text-slate-300 mb-3">Model Parameters</h3>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <label className="block text-xs text-slate-500 mb-1">Validation Days</label>
            <input
              type="number"
              value={config.validation_days}
              onChange={(e) =>
                setConfig({ ...config, validation_days: parseInt(e.target.value) || 7 })
              }
              min={1}
              max={30}
              className="w-full p-3 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 focus:border-stellar-500 focus:outline-none focus:ring-1 focus:ring-stellar-500"
            />
            <p className="text-[10px] text-slate-600 mt-1">Days held for validation</p>
          </div>
          <div>
            <label className="block text-xs text-slate-500 mb-1">Contamination</label>
            <input
              type="number"
              value={config.contamination}
              onChange={(e) =>
                setConfig({ ...config, contamination: parseFloat(e.target.value) || 0.03 })
              }
              step={0.01}
              min={0.001}
              max={0.1}
              className="w-full p-3 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 focus:border-stellar-500 focus:outline-none focus:ring-1 focus:ring-stellar-500"
            />
            <p className="text-[10px] text-slate-600 mt-1">Expected anomaly rate</p>
          </div>
          <div>
            <label className="block text-xs text-slate-500 mb-1">Estimators</label>
            <input
              type="number"
              value={config.n_estimators}
              onChange={(e) =>
                setConfig({ ...config, n_estimators: parseInt(e.target.value) || 300 })
              }
              min={50}
              max={1000}
              className="w-full p-3 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 focus:border-stellar-500 focus:outline-none focus:ring-1 focus:ring-stellar-500"
            />
            <p className="text-[10px] text-slate-600 mt-1">Trees in forest</p>
          </div>
        </div>
      </div>

      {/* Export options */}
      <div className="flex items-center gap-4">
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={config.export_onnx}
            onChange={(e) => setConfig({ ...config, export_onnx: e.target.checked })}
            className="w-5 h-5 rounded border-slate-600 bg-slate-800 text-stellar-500 focus:ring-stellar-500"
          />
          <span className="text-sm text-slate-300">Export ONNX model for edge deployment</span>
        </label>
      </div>

      {/* Action buttons */}
      <div className="flex gap-4 pt-4">
        <button
          type="button"
          onClick={onCancel}
          className="flex-1 py-3 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg font-medium transition-colors"
        >
          Cancel
        </button>
        <button
          type="submit"
          disabled={isSubmitting}
          className="flex-1 py-3 bg-stellar-600 hover:bg-stellar-500 disabled:bg-slate-700 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
        >
          {isSubmitting ? (
            <>
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              Starting...
            </>
          ) : (
            <>
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Start Training
            </>
          )}
        </button>
      </div>
    </form>
  );
}

// Training history row
function HistoryRow({ run, isFirst }: { run: TrainingRun; isFirst: boolean }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className={`border-b border-slate-700/50 last:border-b-0 ${isFirst ? 'bg-slate-800/30' : ''}`}>
      <div
        className="p-4 hover:bg-slate-800/50 cursor-pointer transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <StatusBadge status={run.status} />
            <div>
              <p className="text-sm font-medium text-slate-200">
                {run.model_version || run.run_id}
              </p>
              <p className="text-xs text-slate-500">
                {formatRelativeTime(run.started_at)}
                {run.completed_at && ` • ${formatDuration(run.started_at, run.completed_at)}`}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-6">
            {run.metrics && (
              <div className="text-right">
                <p className="text-sm font-semibold text-slate-300">
                  AUC: {formatAUC(run.metrics.validation_auc)}
                </p>
                <p className="text-xs text-slate-500">
                  {formatNumber(run.metrics.train_rows)} rows
                </p>
              </div>
            )}
            <motion.span
              animate={{ rotate: expanded ? 180 : 0 }}
              className="text-slate-400"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </motion.span>
          </div>
        </div>
      </div>

      <AnimatePresence>
        {expanded && run.metrics && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4">
              <div className="p-4 bg-slate-800/50 rounded-lg">
                <MetricsDisplay metrics={run.metrics} />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function TrainingMonitor() {
  const queryClient = useQueryClient();
  const [showConfig, setShowConfig] = useState(false);

  // Fetch current training status with dynamic polling
  const { data: currentRun } = useQuery({
    queryKey: ['training-status'],
    queryFn: () => api.getTrainingStatus(),
    refetchInterval: (query) => {
      const data = query.state.data;
      return data?.status === 'running' || data?.status === 'pending' ? 2000 : 30000;
    },
  });

  // Fetch training history
  const { data: history, isLoading: historyLoading } = useQuery({
    queryKey: ['training-history'],
    queryFn: () => api.getTrainingHistory(10),
  });

  // Start training mutation
  const startTrainingMutation = useMutation({
    mutationFn: (config: TrainingConfigRequest) => api.startTraining(config),
    onSuccess: () => {
      setShowConfig(false);
      queryClient.invalidateQueries({ queryKey: ['training-status'] });
      queryClient.invalidateQueries({ queryKey: ['training-history'] });
    },
  });

  const isRunning = currentRun?.status === 'running' || currentRun?.status === 'pending';

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Model Training</h1>
          <p className="text-slate-400 mt-1">
            Train and monitor IsolationForest anomaly detection models
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Link to="/baselines" className="btn-ghost flex items-center gap-2">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            Baselines
          </Link>
          {!isRunning && !showConfig && (
            <button
              onClick={() => setShowConfig(true)}
              className="px-5 py-2.5 bg-stellar-600 hover:bg-stellar-500 text-white rounded-lg font-medium transition-colors flex items-center gap-2 shadow-lg shadow-stellar-500/20"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
              </svg>
              New Training
            </button>
          )}
        </div>
      </div>

      {/* Training Config Form */}
      <AnimatePresence>
        {showConfig && !isRunning && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
          >
            <Card className="p-6">
              <h2 className="text-lg font-semibold text-slate-200 mb-6">
                Training Configuration
              </h2>
              <TrainingConfigForm
                onSubmit={(config) => startTrainingMutation.mutate(config)}
                isSubmitting={startTrainingMutation.isPending}
                onCancel={() => setShowConfig(false)}
              />
              {startTrainingMutation.isError && (
                <div className="mt-4 p-4 bg-red-900/20 border border-red-500/30 rounded-lg">
                  <p className="text-sm text-red-400">
                    Failed to start training:{' '}
                    {(startTrainingMutation.error as Error)?.message || 'Unknown error'}
                  </p>
                </div>
              )}
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Current Training Status */}
      {currentRun && currentRun.status !== 'idle' && (
        <TrainingProgressCard run={currentRun} />
      )}

      {/* Completed Training Metrics */}
      {currentRun?.status === 'completed' && currentRun.metrics && (
        <Card className="p-6">
          <h2 className="text-lg font-semibold text-slate-200 mb-6">
            Training Results
          </h2>
          <MetricsDisplay metrics={currentRun.metrics} />

          {/* Artifacts */}
          {currentRun.artifacts && (
            <div className="mt-6 pt-6 border-t border-slate-700">
              <h3 className="text-sm font-semibold text-slate-300 mb-4 uppercase tracking-wide">
                Model Artifacts
              </h3>
              <div className="grid grid-cols-2 gap-3">
                {currentRun.artifacts.model_path && (
                  <div className="flex items-center gap-3 p-3 bg-slate-800/50 rounded-lg">
                    <div className="w-10 h-10 rounded-lg bg-stellar-900/50 flex items-center justify-center">
                      <svg className="w-5 h-5 text-stellar-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-slate-300">Model (sklearn)</p>
                      <p className="text-xs text-slate-500 font-mono truncate">{currentRun.artifacts.model_path}</p>
                    </div>
                  </div>
                )}
                {currentRun.artifacts.onnx_path && (
                  <div className="flex items-center gap-3 p-3 bg-slate-800/50 rounded-lg">
                    <div className="w-10 h-10 rounded-lg bg-aurora-900/50 flex items-center justify-center">
                      <svg className="w-5 h-5 text-aurora-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-slate-300">ONNX Runtime</p>
                      <p className="text-xs text-slate-500 font-mono truncate">{currentRun.artifacts.onnx_path}</p>
                    </div>
                  </div>
                )}
                {currentRun.artifacts.baselines_path && (
                  <div className="flex items-center gap-3 p-3 bg-slate-800/50 rounded-lg">
                    <div className="w-10 h-10 rounded-lg bg-nebula-900/50 flex items-center justify-center">
                      <svg className="w-5 h-5 text-nebula-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-slate-300">Baselines</p>
                      <p className="text-xs text-slate-500 font-mono truncate">{currentRun.artifacts.baselines_path}</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </Card>
      )}

      {/* Training History */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-slate-200">Training History</h2>
          {history && history.total > 0 && (
            <span className="text-sm text-slate-500">{history.total} total runs</span>
          )}
        </div>
        <Card className="divide-y divide-slate-700/50 overflow-hidden">
          {historyLoading ? (
            <div className="p-6">
              <div className="animate-pulse space-y-4">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="flex items-center gap-4">
                    <div className="h-7 w-24 bg-slate-700 rounded-full" />
                    <div className="flex-1">
                      <div className="h-4 w-40 bg-slate-700 rounded" />
                      <div className="h-3 w-24 bg-slate-700 rounded mt-2" />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : history && history.runs.length > 0 ? (
            history.runs.map((run, i) => (
              <HistoryRow key={run.run_id} run={run} isFirst={i === 0} />
            ))
          ) : (
            <div className="p-8 text-center">
              <svg className="w-12 h-12 text-slate-600 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
              </svg>
              <p className="text-slate-400 font-medium">No training history yet</p>
              <p className="text-sm text-slate-500 mt-1">
                Start your first training run to see results here
              </p>
            </div>
          )}
        </Card>
      </div>

      {/* Summary Stats */}
      {history && history.runs.length > 0 && (
        <div className="grid grid-cols-4 gap-4">
          <Card className="p-5 bg-gradient-to-br from-slate-800 to-slate-900">
            <p className="text-xs text-slate-500 uppercase tracking-wide">Total Runs</p>
            <p className="text-3xl font-bold text-stellar-400 mt-1">{history.total}</p>
          </Card>
          <Card className="p-5 bg-gradient-to-br from-slate-800 to-slate-900">
            <p className="text-xs text-slate-500 uppercase tracking-wide">Successful</p>
            <p className="text-3xl font-bold text-emerald-400 mt-1">
              {history.runs.filter((r) => r.status === 'completed').length}
            </p>
          </Card>
          <Card className="p-5 bg-gradient-to-br from-slate-800 to-slate-900">
            <p className="text-xs text-slate-500 uppercase tracking-wide">Failed</p>
            <p className="text-3xl font-bold text-red-400 mt-1">
              {history.runs.filter((r) => r.status === 'failed').length}
            </p>
          </Card>
          <Card className="p-5 bg-gradient-to-br from-slate-800 to-slate-900">
            <p className="text-xs text-slate-500 uppercase tracking-wide">Avg AUC</p>
            <p className="text-3xl font-bold text-cosmic-400 mt-1">
              {formatAvgAUC(history.runs.map((r) => r.metrics?.validation_auc))}
            </p>
          </Card>
        </div>
      )}
    </div>
  );
}
