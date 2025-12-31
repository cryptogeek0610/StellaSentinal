import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { Card, StatCard } from './Card';
import {
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ComposedChart,
} from 'recharts';
import { motion } from 'framer-motion';
import { useState } from 'react';

interface IsolationForestVizProps {
  days?: number;
}

export function IsolationForestViz({ days = 30 }: IsolationForestVizProps) {
  const [activeDetail, setActiveDetail] = useState<'feedback' | 'review' | 'training' | 'retrain' | null>('feedback');
  const { data: stats, isLoading } = useQuery({
    queryKey: ['isolation-forest', 'stats', days],
    queryFn: () => api.getIsolationForestStats(days),
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="relative w-12 h-12 mx-auto mb-3">
            <div className="absolute inset-0 rounded-full border-2 border-cyber-blue/20"></div>
            <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-cyber-blue animate-spin"></div>
          </div>
          <p className="text-slate-500 font-mono text-sm">Loading model statistics...</p>
        </div>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="text-center py-8 text-slate-500">No data available</div>
    );
  }

  // Prepare histogram data
  const histogramData = stats.score_distribution.bins.map((bin) => ({
    bin_center: (bin.bin_start + bin.bin_end) / 2,
    bin_start: bin.bin_start,
    bin_end: bin.bin_end,
    normal: bin.is_anomaly ? 0 : bin.count,
    anomaly: bin.is_anomaly ? bin.count : 0,
    label: bin.is_anomaly ? 'Anomaly' : 'Normal',
  }));

  // Group by bin center
  const groupedData = histogramData.reduce(
    (acc, item) => {
      const key = item.bin_center.toFixed(3);
      if (!acc[key]) {
        acc[key] = {
          bin_center: item.bin_center,
          normal: 0,
          anomaly: 0,
        };
      }
      acc[key].normal += item.normal;
      acc[key].anomaly += item.anomaly;
      return acc;
    },
    {} as Record<string, { bin_center: number; normal: number; anomaly: number }>
  );

  const chartData = Object.values(groupedData).sort((a, b) => a.bin_center - b.bin_center);

  const formatScore = (score: number) => score.toFixed(3);

  return (
    <div className="space-y-6">
      {/* Model Configuration */}
      <Card title="Model Configuration">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">
              Trees
            </p>
            <p className="text-2xl font-bold font-mono text-cyber-blue">
              {stats.config.n_estimators}
            </p>
          </motion.div>
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">
              Contamination
            </p>
            <p className="text-2xl font-bold font-mono text-cyber-orange">
              {(stats.config.contamination * 100).toFixed(1)}%
            </p>
          </motion.div>
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">
              Features Scaled
            </p>
            <p className="text-2xl font-bold text-slate-200">
              {stats.config.scale_features ? 'Yes' : 'No'}
            </p>
          </motion.div>
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">
              Model Type
            </p>
            <p className="text-lg font-semibold text-slate-200 capitalize">
              {stats.config.model_type.replace('_', ' ')}
            </p>
          </motion.div>
        </div>
      </Card>

      {/* Score Distribution Histogram */}
      <Card
        title={
          <div className="flex items-center justify-between w-full">
            <span className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
              Anomaly Score Distribution
            </span>
            <div className="flex items-center gap-4 text-xs font-mono text-slate-500">
              <span>Mean: {formatScore(stats.score_distribution.mean_score)}</span>
              <span>Median: {formatScore(stats.score_distribution.median_score)}</span>
            </div>
          </div>
        }
      >
        <div className="h-80 relative">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData}>
              <defs>
                <linearGradient id="normalGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#00ff88" stopOpacity={0.8} />
                  <stop offset="100%" stopColor="#00ff88" stopOpacity={0.3} />
                </linearGradient>
                <linearGradient id="anomalyGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#ff3366" stopOpacity={0.8} />
                  <stop offset="100%" stopColor="#ff3366" stopOpacity={0.3} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.1)" vertical={false} />
              <XAxis
                dataKey="bin_center"
                tickFormatter={(v) => formatScore(v)}
                stroke="#475569"
                fontSize={10}
                tickLine={false}
                axisLine={false}
              />
              <YAxis stroke="#475569" fontSize={10} tickLine={false} axisLine={false} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(15, 18, 25, 0.95)',
                  border: '1px solid rgba(148, 163, 184, 0.2)',
                  borderRadius: '8px',
                  boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)',
                }}
                labelStyle={{ color: '#94a3b8', fontFamily: 'JetBrains Mono', fontSize: 11 }}
                formatter={(value: number, name: string) => [
                  value,
                  name === 'normal' ? 'Normal' : 'Anomaly',
                ]}
                labelFormatter={(label) => `Score: ${formatScore(label)}`}
              />
              <Legend
                wrapperStyle={{ paddingTop: '20px' }}
                formatter={(value) => (
                  <span className="text-slate-400 text-xs">
                    {value === 'normal' ? 'Normal' : 'Anomaly'}
                  </span>
                )}
              />
              <Area
                type="monotone"
                dataKey="normal"
                stackId="1"
                stroke="#00ff88"
                fill="url(#normalGradient)"
                strokeWidth={2}
                name="normal"
              />
              <Area
                type="monotone"
                dataKey="anomaly"
                stackId="1"
                stroke="#ff3366"
                fill="url(#anomalyGradient)"
                strokeWidth={2}
                name="anomaly"
              />
            </ComposedChart>
          </ResponsiveContainer>

          {/* Decision boundary indicator */}
          <div className="absolute top-2 right-2 glass-panel px-3 py-2 text-xs">
            <div className="flex items-center gap-2">
              <div className="w-4 h-0.5 border-t-2 border-dashed border-cyber-purple"></div>
              <span className="text-slate-400">Decision Boundary (score ≈ 0)</span>
            </div>
          </div>
        </div>

        <div className="mt-4 pt-4 border-t border-slate-700/50">
          <p className="text-xs text-slate-500">
            <strong className="text-slate-400">Interpretation:</strong> Lower scores (negative)
            indicate anomalies. Scores closer to 0 are near the decision boundary. The model
            expects ~{(stats.config.contamination * 100).toFixed(1)}% of data to be anomalous.
          </p>
        </div>
      </Card>

      {/* Feedback Loop Visualization */}
      {stats.feedback_stats && (
        <Card title="Model Adaptation & Feedback Loop" accent="purple">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-center">
            {/* Stats */}
            <div className="lg:col-span-1 space-y-4">
              <button
                type="button"
                className={`w-full text-left p-4 rounded-xl border transition-colors ${activeDetail === 'feedback'
                  ? 'border-purple-500/50 bg-purple-500/10'
                  : 'border-slate-800 bg-slate-900/50 hover:border-slate-700/60'
                  }`}
                onClick={() => setActiveDetail('feedback')}
              >
                <p className="text-xs text-slate-400 uppercase tracking-wider mb-1">Total Feedback</p>
                <p className="text-2xl font-bold text-white">{stats.feedback_stats.total_feedback}</p>
              </button>
              <div className="grid grid-cols-2 gap-4">
                <button
                  type="button"
                  className={`text-left p-4 rounded-xl border transition-colors ${activeDetail === 'review'
                    ? 'border-rose-500/50 bg-rose-500/10'
                    : 'border-slate-800 bg-slate-900/50 hover:border-slate-700/60'
                    }`}
                  onClick={() => setActiveDetail('review')}
                >
                  <p className="text-xs text-rose-400 uppercase tracking-wider mb-1">False Positives</p>
                  <p className="text-xl font-bold text-white">{stats.feedback_stats.false_positives}</p>
                </button>
                <button
                  type="button"
                  className={`text-left p-4 rounded-xl border transition-colors ${activeDetail === 'training'
                    ? 'border-emerald-500/50 bg-emerald-500/10'
                    : 'border-slate-800 bg-slate-900/50 hover:border-slate-700/60'
                    }`}
                  onClick={() => setActiveDetail('training')}
                >
                  <p className="text-xs text-emerald-400 uppercase tracking-wider mb-1">Confirmed</p>
                  <p className="text-xl font-bold text-white">{stats.feedback_stats.confirmed_anomalies}</p>
                </button>
              </div>
            </div>

            {/* Visual Flow */}
            <div className="lg:col-span-2 relative h-full min-h-[160px] bg-slate-900/30 rounded-xl border border-slate-800 p-6 flex flex-col justify-between gap-4">
              <div className="flex items-center justify-between">
                {/* Step 1: User Action */}
                <button
                  type="button"
                  aria-pressed={activeDetail === 'review'}
                  onClick={() => setActiveDetail('review')}
                  className={`text-center z-10 rounded-lg px-2 py-1 transition-colors ${activeDetail === 'review'
                    ? 'bg-purple-500/10 ring-1 ring-purple-500/60'
                    : 'hover:bg-slate-800/40'
                    }`}
                >
                  <div className="w-12 h-12 mx-auto bg-purple-500/20 text-purple-400 rounded-full flex items-center justify-center mb-2 border border-purple-500/50">
                    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
                    </svg>
                  </div>
                  <p className="text-xs font-semibold text-slate-300">Operator Review</p>
                </button>

                {/* Arrow 1 */}
                <div className="flex-1 h-0.5 bg-slate-700 mx-4 relative overflow-hidden">
                  <div className="absolute inset-0 bg-purple-500/50 animate-shimmer" />
                </div>

                {/* Step 2: Labelled Data */}
                <button
                  type="button"
                  aria-pressed={activeDetail === 'training'}
                  onClick={() => setActiveDetail('training')}
                  className={`text-center z-10 rounded-lg px-2 py-1 transition-colors ${activeDetail === 'training'
                    ? 'bg-blue-500/10 ring-1 ring-blue-500/60'
                    : 'hover:bg-slate-800/40'
                    }`}
                >
                  <div className="w-12 h-12 mx-auto bg-blue-500/20 text-blue-400 rounded-full flex items-center justify-center mb-2 border border-blue-500/50">
                    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                    </svg>
                  </div>
                  <p className="text-xs font-semibold text-slate-300">Training Set Update</p>
                </button>

                {/* Arrow 2 */}
                <div className="flex-1 h-0.5 bg-slate-700 mx-4 relative overflow-hidden">
                  <div className="absolute inset-0 bg-blue-500/50 animate-shimmer" style={{ animationDelay: '0.5s' }} />
                </div>

                {/* Step 3: Model */}
                <button
                  type="button"
                  aria-pressed={activeDetail === 'retrain'}
                  onClick={() => setActiveDetail('retrain')}
                  className={`text-center z-10 rounded-lg px-2 py-1 transition-colors ${activeDetail === 'retrain'
                    ? 'bg-emerald-500/10 ring-1 ring-emerald-500/60'
                    : 'hover:bg-slate-800/40'
                    }`}
                >
                  <div className="w-12 h-12 mx-auto bg-emerald-500/20 text-emerald-400 rounded-full flex items-center justify-center mb-2 border border-emerald-500/50">
                    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                    </svg>
                  </div>
                  <p className="text-xs font-semibold text-slate-300">Retraining</p>
                  <p className="text-[10px] text-emerald-400 mt-1">+{stats.feedback_stats.projected_accuracy_gain}% Accuracy</p>
                </button>
              </div>

              <div className="rounded-lg border border-slate-800 bg-slate-950/40 px-4 py-3 text-xs text-slate-300">
                {activeDetail === 'feedback' && (
                  <div className="space-y-1">
                    <p className="font-semibold text-slate-200">Feedback volume</p>
                    <p>
                      {stats.feedback_stats.total_feedback} reviews in the last 30 days are feeding the model
                      adaptation cycle.
                    </p>
                    <p className="text-slate-400">Click a stage to see how feedback is used.</p>
                  </div>
                )}
                {activeDetail === 'review' && (
                  <div className="space-y-1">
                    <p className="font-semibold text-slate-200">Operator review</p>
                    <p>
                      {stats.feedback_stats.false_positives} anomalies were marked as false positives and are
                      queued for label correction.
                    </p>
                    <p className="text-slate-400">Manual review reduces alert fatigue and improves precision.</p>
                  </div>
                )}
                {activeDetail === 'training' && (
                  <div className="space-y-1">
                    <p className="font-semibold text-slate-200">Training set update</p>
                    <p>
                      {stats.feedback_stats.confirmed_anomalies} confirmed anomalies are added to the training
                      pool for the next refresh.
                    </p>
                    <p className="text-slate-400">Fresh labels keep baselines aligned with field behavior.</p>
                  </div>
                )}
                {activeDetail === 'retrain' && (
                  <div className="space-y-1">
                    <p className="font-semibold text-slate-200">Retraining impact</p>
                    <p>
                      Expected lift of +{stats.feedback_stats.projected_accuracy_gain}% accuracy once the model
                      is retrained.
                    </p>
                    <p className="text-slate-400">Deployment occurs after validation checks pass.</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Statistics Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard
          label="Total Predictions"
          value={stats.total_predictions.toLocaleString()}
          color="cyan"
          icon={
            <svg className="w-6 h-6 text-cyber-blue" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
              />
            </svg>
          }
        />
        <StatCard
          label="Anomaly Rate"
          value={`${(stats.anomaly_rate * 100).toFixed(2)}%`}
          color="red"
          icon={
            <svg className="w-6 h-6 text-cyber-red" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
          }
        />
        <StatCard
          label="Normal Observations"
          value={stats.score_distribution.total_normal.toLocaleString()}
          color="green"
          icon={
            <svg className="w-6 h-6 text-cyber-green" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          }
        />
      </div>

      {/* Score Statistics */}
      <Card title="Score Statistics">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {[
            { label: 'Mean', value: stats.score_distribution.mean_score },
            { label: 'Median', value: stats.score_distribution.median_score },
            { label: 'Std Dev', value: stats.score_distribution.std_score },
            {
              label: 'Range',
              value: stats.score_distribution.max_score - stats.score_distribution.min_score,
            },
          ].map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">
                {stat.label}
              </p>
              <p className="text-xl font-bold font-mono text-slate-200">{formatScore(stat.value)}</p>
            </motion.div>
          ))}
        </div>
      </Card>
    </div>
  );
}
