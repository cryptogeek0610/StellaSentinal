/**
 * Baselines Page - Feature Baseline Management & Drift Analysis
 *
 * Comprehensive view of all feature baselines with drift detection,
 * time-series visualization, and adjustment suggestions.
 */

import { useState, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../api/client';
import { Card } from '../components/Card';
import type { BaselineSuggestion } from '../types/anomaly';
import { motion, AnimatePresence } from 'framer-motion';
import { ToggleSwitch } from '../components/ui';
import { Link } from 'react-router-dom';

// Loading spinner component
function LoadingSpinner({ className = '' }: { className?: string }) {
  return (
    <svg className={`animate-spin ${className}`} viewBox="0 0 24 24" aria-hidden="true">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
    </svg>
  );
}

// Feature icons mapping
const featureIcons: Record<string, JSX.Element> = {
  BatteryDrop: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 13l4 4L19 7" />
    </svg>
  ),
  OfflineTime: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M18.364 5.636a9 9 0 010 12.728m0 0l-2.829-2.829m2.829 2.829L21 21M15.536 8.464a5 5 0 010 7.072m0 0l-2.829-2.829m-4.243 2.829a4.978 4.978 0 01-1.414-2.83m-1.414 5.658a9 9 0 01-2.167-9.238m7.824 2.167a1 1 0 111.414 1.414m-1.414-1.414L3 3m8.293 8.293l1.414 1.414" />
    </svg>
  ),
  UploadSize: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
    </svg>
  ),
  DownloadSize: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10" />
    </svg>
  ),
  StorageFree: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
    </svg>
  ),
  AppCrashes: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
    </svg>
  ),
};

function formatRelativeTime(date: Date | string): string {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  const now = new Date();
  const diffMs = now.getTime() - dateObj.getTime();
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffHours < 1) return 'Just now';
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays === 1) return 'Yesterday';
  if (diffDays < 7) return `${diffDays} days ago`;
  return dateObj.toLocaleDateString();
}

export default function Baselines() {
  const [source, setSource] = useState('dw');
  const [days, setDays] = useState(30);
  const [daysError, setDaysError] = useState<string | null>(null);
  const [autoCorrectionEnabled, setAutoCorrectionEnabled] = useState(false);
  const [selectedFeature, setSelectedFeature] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'suggestions' | 'history'>('overview');
  const queryClient = useQueryClient();

  // Fetch baseline features from backend
  const { data: baselineFeatures = [], isLoading: featuresLoading } = useQuery({
    queryKey: ['baselines', 'features', source, days],
    queryFn: () => api.getBaselineFeatures(source, days),
  });

  // Fetch baseline history from backend
  const { data: adjustmentHistory = [], isLoading: historyLoading } = useQuery({
    queryKey: ['baselines', 'history', source],
    queryFn: () => api.getBaselineHistory(source, 50),
  });

  const { data: suggestions, isLoading } = useQuery({
    queryKey: ['baselines', 'suggestions', source, days],
    queryFn: () => api.getBaselineSuggestions(source, days),
  });

  const {
    data: llmSuggestions,
    isLoading: llmLoading,
    refetch: refetchLLM,
  } = useQuery({
    queryKey: ['baselines', 'llm-suggestions', source, days],
    queryFn: () => api.analyzeBaselinesWithLLM(source, days),
    enabled: false,
  });

  const applyMutation = useMutation({
    mutationFn: (suggestion: BaselineSuggestion) => {
      let groupKey: Record<string, unknown>;
      try {
        groupKey =
          typeof suggestion.group_key === 'string'
            ? JSON.parse(suggestion.group_key)
            : suggestion.group_key;
      } catch {
        groupKey = { group: suggestion.group_key };
      }

      return api.applyBaselineAdjustment(
        {
          level: suggestion.level,
          group_key: groupKey,
          feature: suggestion.feature,
          adjustment: suggestion.proposed_new_median - suggestion.baseline_median,
          reason: suggestion.rationale,
          auto_retrain: true,
        },
        source
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['baselines'] });
      queryClient.invalidateQueries({ queryKey: ['isolation-forest'] });
      queryClient.invalidateQueries({ queryKey: ['dashboard'] });
    },
  });

  const handleDaysChange = (value: string) => {
    const numValue = parseInt(value);
    if (isNaN(numValue) || numValue < 1) {
      setDaysError('Please enter a valid number (1-365)');
      setDays(1);
    } else if (numValue > 365) {
      setDaysError('Maximum 365 days allowed');
      setDays(365);
    } else {
      setDaysError(null);
      setDays(numValue);
    }
  };

  const displaySuggestions = llmSuggestions && llmSuggestions.length > 0
    ? llmSuggestions
    : suggestions || [];

  // Calculate stats from fetched data
  const stats = useMemo(() => {
    const driftCount = baselineFeatures.filter(f => f.status === 'drift').length;
    const warningCount = baselineFeatures.filter(f => f.status === 'warning').length;
    const stableCount = baselineFeatures.filter(f => f.status === 'stable').length;
    const recentAdjustments = adjustmentHistory.filter(
      h => Date.now() - new Date(h.date).getTime() < 7 * 24 * 60 * 60 * 1000
    ).length;
    return { driftCount, warningCount, stableCount, recentAdjustments, total: baselineFeatures.length };
  }, [baselineFeatures, adjustmentHistory]);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Baseline Management</h1>
          <p className="text-slate-400 mt-1">
            Monitor feature baselines, detect drift, and apply adjustments
          </p>
        </div>

        {/* Quick Actions */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => refetchLLM()}
            disabled={llmLoading || isLoading}
            className="btn-primary flex items-center gap-2"
          >
            {llmLoading ? (
              <>
                <LoadingSpinner className="h-4 w-4" />
                Analyzing...
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
                Analyze with AI
              </>
            )}
          </button>
          <Link to="/status" className="btn-ghost flex items-center gap-2">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
            </svg>
            Model Status
          </Link>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        <Card className="!p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-slate-700/50 flex items-center justify-center">
              <svg className="w-5 h-5 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <div>
              <p className="text-2xl font-bold text-white font-mono">{stats.total}</p>
              <p className="text-xs text-slate-500">Total Features</p>
            </div>
          </div>
        </Card>

        <Card className="!p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-emerald-500/10 flex items-center justify-center">
              <svg className="w-5 h-5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <div>
              <p className="text-2xl font-bold text-emerald-400 font-mono">{stats.stableCount}</p>
              <p className="text-xs text-slate-500">Stable</p>
            </div>
          </div>
        </Card>

        <Card className="!p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-amber-500/10 flex items-center justify-center">
              <svg className="w-5 h-5 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
            <div>
              <p className="text-2xl font-bold text-amber-400 font-mono">{stats.warningCount}</p>
              <p className="text-xs text-slate-500">Warning</p>
            </div>
          </div>
        </Card>

        <Card className="!p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-red-500/10 flex items-center justify-center">
              <svg className="w-5 h-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
              </svg>
            </div>
            <div>
              <p className="text-2xl font-bold text-red-400 font-mono">{stats.driftCount}</p>
              <p className="text-xs text-slate-500">Drift Detected</p>
            </div>
          </div>
        </Card>

        <Card className="!p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-indigo-500/10 flex items-center justify-center">
              <svg className="w-5 h-5 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div>
              <p className="text-2xl font-bold text-indigo-400 font-mono">{stats.recentAdjustments}</p>
              <p className="text-xs text-slate-500">Recent (7d)</p>
            </div>
          </div>
        </Card>
      </div>

      {/* Configuration Bar */}
      <Card className="!p-4">
        <div className="flex flex-wrap items-center gap-6">
          <div className="flex items-center gap-3">
            <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
              Data Source
            </label>
            <select
              value={source}
              onChange={(e) => setSource(e.target.value)}
              className="select-field w-auto"
            >
              <option value="dw">XSight Database</option>
              <option value="synthetic">Synthetic</option>
            </select>
          </div>

          <div className="flex items-center gap-3">
            <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
              Analysis Period
            </label>
            <input
              type="number"
              value={days}
              onChange={(e) => handleDaysChange(e.target.value)}
              min={1}
              max={365}
              className={`input-field w-20 ${daysError ? 'border-red-500' : ''}`}
            />
            <span className="text-sm text-slate-500">days</span>
          </div>

          <div className="flex items-center gap-3 ml-auto">
            <span className="text-xs text-slate-400">Auto-Correction</span>
            <ToggleSwitch
              enabled={autoCorrectionEnabled}
              onChange={setAutoCorrectionEnabled}
              size="sm"
              variant="emerald"
            />
          </div>
        </div>
        {daysError && (
          <p className="mt-2 text-xs text-red-400">{daysError}</p>
        )}
      </Card>

      {/* Tab Navigation */}
      <div className="flex gap-1 p-1 bg-slate-800/50 rounded-xl w-fit">
        {[
          { id: 'overview' as const, label: 'Feature Overview', count: stats.total },
          { id: 'suggestions' as const, label: 'Suggestions', count: displaySuggestions.length },
          { id: 'history' as const, label: 'Adjustment History', count: adjustmentHistory.length },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2 ${
              activeTab === tab.id
                ? 'bg-amber-500/20 text-amber-400 shadow-lg'
                : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
            }`}
          >
            {tab.label}
            {tab.count > 0 && (
              <span className={`px-1.5 py-0.5 rounded text-xs font-mono ${
                activeTab === tab.id ? 'bg-amber-500/30 text-amber-300' : 'bg-slate-700 text-slate-400'
              }`}>
                {tab.count}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <AnimatePresence mode="wait">
        {activeTab === 'overview' && (
          <motion.div
            key="overview"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
          >
            {featuresLoading ? (
              <Card>
                <div className="flex items-center justify-center h-32">
                  <div className="text-center">
                    <LoadingSpinner className="w-8 h-8 text-amber-400 mx-auto mb-3" />
                    <p className="text-slate-500 text-sm">Loading baseline features...</p>
                  </div>
                </div>
              </Card>
            ) : baselineFeatures.length === 0 ? (
              <Card>
                <div className="text-center py-16">
                  <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-slate-800 flex items-center justify-center">
                    <svg className="w-8 h-8 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">No Baseline Data</h3>
                  <p className="text-slate-500 max-w-md mx-auto mb-4">
                    Baselines will appear here once model training has been completed and feature baselines are established.
                  </p>
                  <Link to="/training" className="btn-primary inline-flex items-center gap-2">
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                    </svg>
                    Go to Training
                  </Link>
                </div>
              </Card>
            ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {baselineFeatures.map((feature) => {
              const driftPercent = Number.isFinite(feature.drift_percent) ? feature.drift_percent : 0;
              const baselineValue = feature.baseline;
              const observedValue = feature.observed;
              const barWidth = baselineValue > 0
                ? Math.min(100, Math.max(0, (observedValue / baselineValue) * 100))
                : 0;
              const isSelected = selectedFeature === feature.feature;

              return (
                <div
                  key={feature.feature}
                  onClick={() => setSelectedFeature(isSelected ? null : feature.feature)}
                  className={`stellar-card rounded-xl p-5 cursor-pointer transition-all ${
                    isSelected ? 'ring-2 ring-amber-500/50' : 'hover:bg-slate-800/30'
                  }`}
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                        feature.status === 'stable' ? 'bg-emerald-500/10 text-emerald-400' :
                        feature.status === 'warning' ? 'bg-amber-500/10 text-amber-400' :
                        'bg-red-500/10 text-red-400'
                      }`}>
                        {featureIcons[feature.feature] || (
                          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                          </svg>
                        )}
                      </div>
                      <div>
                        <h3 className="font-semibold text-white">{feature.feature}</h3>
                        <p className="text-xs text-slate-500">{feature.unit}</p>
                      </div>
                    </div>

                    <div className={`px-2 py-1 rounded-full text-xs font-medium ${
                      feature.status === 'stable' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30' :
                      feature.status === 'warning' ? 'bg-amber-500/10 text-amber-400 border border-amber-500/30' :
                      'bg-red-500/10 text-red-400 border border-red-500/30'
                    }`}>
                      {feature.status === 'stable' ? 'Stable' : feature.status === 'warning' ? 'Warning' : 'Drift'}
                    </div>
                  </div>

                  {/* Baseline vs Observed Comparison */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-slate-500">Baseline</span>
                      <span className="font-mono text-slate-300">{feature.baseline} {feature.unit}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-slate-500">Observed ({days}d avg)</span>
                      <span className={`font-mono font-medium ${
                        Math.abs(driftPercent) < 10 ? 'text-emerald-400' :
                        Math.abs(driftPercent) < 25 ? 'text-amber-400' : 'text-red-400'
                      }`}>
                        {observedValue} {feature.unit}
                      </span>
                    </div>

                    {/* Visual Bar */}
                    <div className="relative h-2 bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className="absolute h-full bg-slate-600 rounded-full"
                        style={{ width: '100%' }}
                      />
                      <div
                        className={`absolute h-full rounded-full transition-all ${
                          feature.status === 'stable' ? 'bg-emerald-500' :
                          feature.status === 'warning' ? 'bg-amber-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${barWidth}%` }}
                      />
                      {/* Baseline marker */}
                      <div
                        className="absolute top-0 w-0.5 h-full bg-white/50"
                        style={{ left: '100%', transform: 'translateX(-50%)' }}
                      />
                    </div>

                    <div className="flex items-center justify-between text-xs">
                      <span className="text-slate-600">0</span>
                      <span className={`font-medium ${
                        driftPercent > 0 ? 'text-red-400' : driftPercent < 0 ? 'text-emerald-400' : 'text-slate-400'
                      }`}>
                        {driftPercent > 0 ? '+' : ''}{driftPercent.toFixed(1)}% drift
                      </span>
                      <span className="text-slate-600">{(baselineValue * 1.5).toFixed(0)}</span>
                    </div>
                  </div>

                  {/* Expanded Details */}
                  <AnimatePresence>
                    {isSelected && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="overflow-hidden"
                      >
                        <div className="mt-4 pt-4 border-t border-slate-700/50 space-y-3">
                          <div className="grid grid-cols-2 gap-3 text-sm">
                            <div className="p-3 bg-slate-800/50 rounded-lg">
                              <p className="text-xs text-slate-500 mb-1">Min (30d)</p>
                              <p className="font-mono text-white">{(feature.baseline * 0.7).toFixed(1)}</p>
                            </div>
                            <div className="p-3 bg-slate-800/50 rounded-lg">
                              <p className="text-xs text-slate-500 mb-1">Max (30d)</p>
                              <p className="font-mono text-white">{(feature.baseline * 1.4).toFixed(1)}</p>
                            </div>
                            <div className="p-3 bg-slate-800/50 rounded-lg">
                              <p className="text-xs text-slate-500 mb-1">Std Dev</p>
                              <p className="font-mono text-white">{(feature.baseline * 0.15).toFixed(2)}</p>
                            </div>
                            <div className="p-3 bg-slate-800/50 rounded-lg">
                              <p className="text-xs text-slate-500 mb-1">Last Adjusted</p>
                              <p className="font-mono text-white">12d ago</p>
                            </div>
                          </div>

                          {feature.status !== 'stable' && (
                            <button className="w-full btn-stellar text-sm">
                              View Adjustment Suggestion
                            </button>
                          )}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              );
            })}
            </div>
            )}
          </motion.div>
        )}

        {activeTab === 'suggestions' && (
          <motion.div
            key="suggestions"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
          >
            <Card>
              {isLoading ? (
                <div className="flex items-center justify-center h-32">
                  <div className="text-center">
                    <LoadingSpinner className="w-8 h-8 text-amber-400 mx-auto mb-3" />
                    <p className="text-slate-500 text-sm">Loading suggestions...</p>
                  </div>
                </div>
              ) : displaySuggestions && displaySuggestions.length > 0 ? (
                <div className="space-y-4">
                  {llmSuggestions && (
                    <div className="flex items-center gap-2 text-sm text-emerald-400 mb-4 p-3 bg-emerald-500/10 rounded-lg border border-emerald-500/20">
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      AI analysis complete - {llmSuggestions.length} intelligent suggestions
                    </div>
                  )}

                  {displaySuggestions.map((suggestion, idx) => {
                    const maxVal = Math.max(suggestion.baseline_median, suggestion.observed_median, suggestion.proposed_new_median);
                    const scaledMax = maxVal > 0 ? maxVal * 1.2 : 1;
                    const getPercent = (val: number) => Math.min(100, (val / scaledMax) * 100);

                    return (
                      <motion.div
                        key={idx}
                        className="p-5 rounded-xl bg-slate-800/30 border border-slate-700/50 hover:border-indigo-500/30 transition-colors"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: idx * 0.05 }}
                      >
                        <div className="flex justify-between items-start mb-4">
                          <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-lg bg-indigo-500/10 flex items-center justify-center text-indigo-400">
                              {featureIcons[suggestion.feature] || (
                                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                                </svg>
                              )}
                            </div>
                            <div>
                              <h4 className="font-semibold text-white">{suggestion.feature}</h4>
                              <p className="text-xs text-slate-500">
                                Level: {suggestion.level} | Group: {suggestion.group_key}
                              </p>
                            </div>
                          </div>
                          <button
                            onClick={() => applyMutation.mutate(suggestion)}
                            disabled={applyMutation.isPending}
                            className="btn-success text-sm"
                          >
                            {applyMutation.isPending ? 'Applying...' : 'Apply'}
                          </button>
                        </div>

                        {/* Drift Visualization */}
                        <div className="mb-4 space-y-2 bg-slate-900/50 p-4 rounded-lg">
                          <div className="flex items-center gap-3 text-xs">
                            <div className="w-20 text-slate-400 text-right">Baseline</div>
                            <div className="flex-1 h-3 bg-slate-800 rounded-full overflow-hidden">
                              <div style={{ width: `${getPercent(suggestion.baseline_median)}%` }} className="h-full bg-slate-600 rounded-full" />
                            </div>
                            <div className="w-16 font-mono text-slate-500">{suggestion.baseline_median.toFixed(1)}</div>
                          </div>
                          <div className="flex items-center gap-3 text-xs">
                            <div className="w-20 text-indigo-400 font-bold text-right">Observed</div>
                            <div className="flex-1 h-3 bg-slate-800 rounded-full overflow-hidden">
                              <div style={{ width: `${getPercent(suggestion.observed_median)}%` }} className="h-full bg-indigo-500 rounded-full" />
                            </div>
                            <div className="w-16 font-mono text-indigo-400 font-bold">{suggestion.observed_median.toFixed(1)}</div>
                          </div>
                          <div className="flex items-center gap-3 text-xs">
                            <div className="w-20 text-emerald-400 text-right">Proposed</div>
                            <div className="flex-1 h-3 bg-slate-800 rounded-full overflow-hidden">
                              <div style={{ width: `${getPercent(suggestion.proposed_new_median)}%` }} className="h-full bg-emerald-500/50 border-2 border-emerald-500/30 rounded-full" />
                            </div>
                            <div className="w-16 font-mono text-emerald-400">{suggestion.proposed_new_median.toFixed(1)}</div>
                          </div>
                        </div>

                        <div className="flex items-start gap-2 p-3 bg-slate-800/30 rounded-lg">
                          <svg className="w-4 h-4 text-indigo-400 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          <p className="text-sm text-slate-300">{suggestion.rationale}</p>
                        </div>
                      </motion.div>
                    );
                  })}
                </div>
              ) : (
                <div className="text-center py-16">
                  <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-emerald-500/10 flex items-center justify-center">
                    <svg className="w-8 h-8 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">All Baselines Healthy</h3>
                  <p className="text-slate-500 max-w-md mx-auto">
                    No significant drift detected in the last {days} days. Baselines are performing well.
                  </p>
                  <button
                    onClick={() => refetchLLM()}
                    disabled={llmLoading}
                    className="mt-6 btn-ghost"
                  >
                    Run AI Analysis Anyway
                  </button>
                </div>
              )}
            </Card>

            {/* Success/Error Messages */}
            <AnimatePresence>
              {applyMutation.isSuccess && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="mt-4 p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-lg"
                >
                  <div className="flex items-center gap-3">
                    <svg className="w-5 h-5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    <span className="text-emerald-400 font-medium">
                      {applyMutation.data?.message || 'Baseline adjustment applied successfully'}
                    </span>
                  </div>
                </motion.div>
              )}

              {applyMutation.isError && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="mt-4 p-4 bg-red-500/10 border border-red-500/30 rounded-lg"
                >
                  <div className="flex items-center gap-3">
                    <svg className="w-5 h-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                    <span className="text-red-400 font-medium">
                      Failed to apply adjustment. Please try again.
                    </span>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        )}

        {activeTab === 'history' && (
          <motion.div
            key="history"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
          >
            <Card>
              {historyLoading ? (
                <div className="flex items-center justify-center h-32">
                  <div className="text-center">
                    <LoadingSpinner className="w-8 h-8 text-amber-400 mx-auto mb-3" />
                    <p className="text-slate-500 text-sm">Loading adjustment history...</p>
                  </div>
                </div>
              ) : adjustmentHistory.length === 0 ? (
                <div className="text-center py-16">
                  <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-slate-800 flex items-center justify-center">
                    <svg className="w-8 h-8 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">No Adjustment History</h3>
                  <p className="text-slate-500 max-w-md mx-auto">
                    Baseline adjustments will be recorded here once you start applying changes to feature baselines.
                  </p>
                </div>
              ) : (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-slate-700/50">
                      <th className="text-left py-3 px-4 text-xs font-semibold text-slate-500 uppercase tracking-wider">When</th>
                      <th className="text-left py-3 px-4 text-xs font-semibold text-slate-500 uppercase tracking-wider">Feature</th>
                      <th className="text-left py-3 px-4 text-xs font-semibold text-slate-500 uppercase tracking-wider">Change</th>
                      <th className="text-left py-3 px-4 text-xs font-semibold text-slate-500 uppercase tracking-wider">Type</th>
                      <th className="text-left py-3 px-4 text-xs font-semibold text-slate-500 uppercase tracking-wider">Reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {adjustmentHistory.map((item) => (
                      <tr key={item.id} className="border-b border-slate-800/50 hover:bg-slate-800/20 transition-colors">
                        <td className="py-4 px-4">
                          <span className="text-sm text-slate-400">{formatRelativeTime(item.date)}</span>
                        </td>
                        <td className="py-4 px-4">
                          <div className="flex items-center gap-2">
                            <div className="w-8 h-8 rounded-lg bg-slate-800 flex items-center justify-center text-slate-400">
                              {featureIcons[item.feature] || (
                                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2z" />
                                </svg>
                              )}
                            </div>
                            <span className="font-medium text-white">{item.feature}</span>
                          </div>
                        </td>
                        <td className="py-4 px-4">
                          <div className="flex items-center gap-2 font-mono text-sm">
                            <span className="text-slate-500 line-through">{item.old_value}</span>
                            <svg className="w-4 h-4 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                            </svg>
                            <span className={item.new_value > item.old_value ? 'text-amber-400' : 'text-emerald-400'}>
                              {item.new_value}
                            </span>
                          </div>
                        </td>
                        <td className="py-4 px-4">
                          <span className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium ${
                            item.type === 'auto'
                              ? 'bg-indigo-500/10 text-indigo-400 border border-indigo-500/30'
                              : 'bg-slate-700/50 text-slate-400 border border-slate-600/30'
                          }`}>
                            {item.type === 'auto' ? (
                              <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                              </svg>
                            ) : (
                              <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                              </svg>
                            )}
                            {item.type === 'auto' ? 'Auto' : 'Manual'}
                          </span>
                        </td>
                        <td className="py-4 px-4">
                          <p className="text-sm text-slate-400 max-w-xs truncate" title={item.reason ?? undefined}>
                            {item.reason}
                          </p>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              )}
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
