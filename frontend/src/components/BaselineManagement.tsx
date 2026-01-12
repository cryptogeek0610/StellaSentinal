import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../api/client';
import { Card } from './Card';
import type { BaselineSuggestion } from '../types/anomaly';
import { motion, AnimatePresence } from 'framer-motion';
import { ToggleSwitch } from './ui';

// Loading spinner component
function LoadingSpinner({ className = '' }: { className?: string }) {
  return (
    <svg className={`animate-spin ${className}`} viewBox="0 0 24 24" aria-hidden="true">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
    </svg>
  );
}

export function BaselineManagement() {
  const [source, setSource] = useState('dw');
  const [days, setDays] = useState(30);
  const [daysError, setDaysError] = useState<string | null>(null);
  const [autoCorrectionEnabled, setAutoCorrectionEnabled] = useState(false);
  const queryClient = useQueryClient();

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

  const analyzeWithLLM = () => {
    refetchLLM();
  };

  const displaySuggestions = llmSuggestions && llmSuggestions.length > 0
    ? llmSuggestions
    : suggestions || [];

  return (
    <div className="space-y-6">
      {/* Configuration */}
      <Card title={<span className="telemetry-label">Configuration</span>}>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
          <div>
            <label htmlFor="data-source" className="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
              Data Source
            </label>
            <select
              id="data-source"
              value={source}
              onChange={(e) => setSource(e.target.value)}
              className="select-field"
            >
              <option value="dw">XSight Database</option>
              <option value="synthetic">Synthetic</option>
            </select>
          </div>
          <div>
            <label htmlFor="analysis-days" className="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
              Analysis Period (days)
            </label>
            <input
              id="analysis-days"
              type="number"
              value={days}
              onChange={(e) => handleDaysChange(e.target.value)}
              min={1}
              max={365}
              className={`input-field ${daysError ? 'border-red-500 focus:border-red-500 focus:ring-red-500/50' : ''}`}
              aria-describedby={daysError ? 'days-error' : undefined}
              aria-invalid={daysError ? 'true' : 'false'}
            />
            {daysError && (
              <p id="days-error" className="mt-1 text-xs text-red-400" role="alert">
                {daysError}
              </p>
            )}
          </div>
          <div>
            <button
              onClick={analyzeWithLLM}
              disabled={llmLoading || isLoading}
              className="btn-primary w-full flex items-center justify-center"
              aria-label="Analyze baselines with AI"
            >
              {llmLoading ? (
                <>
                  <LoadingSpinner className="h-4 w-4 mr-2" />
                  Analyzing...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                    />
                  </svg>
                  Analyze with LLM
                </>
              )}
            </button>
          </div>
        </div>
        {llmSuggestions && (
          <div className="mt-4 flex items-center gap-2 text-sm text-emerald-400" role="status">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
            LLM analysis complete ({llmSuggestions.length} suggestions)
          </div>
        )}
      </Card>

      {/* Suggestions */}
      <Card
        title={
          <div className="flex items-center justify-between w-full">
            <span className="telemetry-label">Baseline Adjustment Suggestions</span>
            <div className="flex items-center gap-3">
              <span className="text-xs text-slate-400">Auto-Correction</span>
              <ToggleSwitch
                enabled={autoCorrectionEnabled}
                onChange={setAutoCorrectionEnabled}
                size="sm"
                variant="emerald"
                aria-label="Enable auto-correction of baselines"
              />
            </div>
          </div>
        }
      >
        {isLoading ? (
          <div className="flex items-center justify-center h-32" role="status" aria-label="Loading suggestions">
            <div className="text-center">
              <div className="relative w-10 h-10 mx-auto mb-3">
                <div className="absolute inset-0 rounded-full border-2 border-indigo-500/20" />
                <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-indigo-500 animate-spin" />
              </div>
              <p className="text-slate-500 text-sm">Loading suggestions...</p>
            </div>
          </div>
        ) : displaySuggestions && displaySuggestions.length > 0 ? (
          <div className="space-y-4" role="list" aria-label="Baseline suggestions">
            <AnimatePresence>
              {displaySuggestions.map((suggestion, idx) => {
                const maxVal = Math.max(suggestion.baseline_median, suggestion.observed_median, suggestion.proposed_new_median);
                const scaledMax = maxVal > 0 ? maxVal * 1.2 : 1;
                const getPercent = (val: number) => Math.min(100, (val / scaledMax) * 100);

                return (
                  <motion.div
                    key={idx}
                    className="stellar-card p-4 rounded-xl border-l-2 border-l-indigo-500"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ delay: idx * 0.05 }}
                    role="listitem"
                  >
                    <div className="flex justify-between items-start mb-4">
                      <div className="flex-1">
                        <h4 className="font-semibold text-slate-200">{suggestion.feature}</h4>
                        <div className="mt-1 text-xs text-slate-500">
                          <span className="font-mono">Level:</span> {suggestion.level} |{' '}
                          <span className="font-mono">Group:</span> {suggestion.group_key}
                        </div>
                      </div>
                      <button
                        onClick={() => applyMutation.mutate(suggestion)}
                        disabled={applyMutation.isPending}
                        className="btn-success text-xs py-1.5"
                        aria-label={`Apply baseline adjustment for ${suggestion.feature}`}
                      >
                        {applyMutation.isPending ? 'Applying...' : 'Apply Adjustment'}
                      </button>
                    </div>

                    {/* Drift Visualization */}
                    <div className="mb-6 space-y-2 bg-slate-900/50 p-4 rounded-lg">
                      <div className="flex items-center gap-3 text-xs">
                        <div className="w-20 text-slate-400 text-right">Baseline</div>
                        <div className="flex-1 h-4 bg-slate-800 rounded-sm relative overflow-hidden">
                          <div style={{ width: `${getPercent(suggestion.baseline_median)}%` }} className="h-full bg-slate-500/50 rounded-sm" />
                        </div>
                        <div className="w-16 font-mono text-slate-500">{suggestion.baseline_median.toFixed(1)}</div>
                      </div>
                      <div className="flex items-center gap-3 text-xs">
                        <div className="w-20 text-indigo-400 font-bold text-right">Observed</div>
                        <div className="flex-1 h-4 bg-slate-800 rounded-sm relative overflow-hidden">
                          <div style={{ width: `${getPercent(suggestion.observed_median)}%` }} className="h-full bg-indigo-500 rounded-sm" />
                        </div>
                        <div className="w-16 font-mono text-indigo-400 font-bold">{suggestion.observed_median.toFixed(1)}</div>
                      </div>
                      <div className="flex items-center gap-3 text-xs">
                        <div className="w-20 text-emerald-400 text-right">Proposed</div>
                        <div className="flex-1 h-4 bg-slate-800 rounded-sm relative overflow-hidden">
                          <div style={{ width: `${getPercent(suggestion.proposed_new_median)}%` }} className="h-full bg-emerald-500/50 border-2 border-emerald-500/30 rounded-sm box-border" />
                        </div>
                        <div className="w-16 font-mono text-emerald-400">{suggestion.proposed_new_median.toFixed(1)}</div>
                      </div>
                    </div>

                    <div className="mt-4 pt-3 border-t border-slate-700/50">
                      <div className="flex items-start gap-2">
                        <svg className="w-5 h-5 text-indigo-400 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <div>
                          <p className="text-xs text-slate-500 uppercase font-bold mb-1">Reason for Adjustment</p>
                          <p className="text-sm text-slate-300 leading-relaxed">{suggestion.rationale}</p>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </AnimatePresence>
          </div>
        ) : (
          <div className="text-center py-12">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-slate-800 flex items-center justify-center" aria-hidden="true">
              <svg className="w-8 h-8 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
            </div>
            <p className="text-slate-400 font-medium">No baseline adjustments suggested</p>
            <p className="text-xs text-slate-600 mt-2">
              Suggestions appear when systematic drift is detected in anomaly patterns.
            </p>
          </div>
        )}
      </Card>

      {/* History Section (Mock) */}
      <Card title={<span className="telemetry-label">Adjustment History</span>}>
        <div className="space-y-4">
          {[
            { date: 'Today, 10:23 AM', feature: 'BatteryDrop', old: 12, new: 15, auto: true },
            { date: 'Yesterday, 4:15 PM', feature: 'OfflineTime', old: 30, new: 45, auto: false },
            { date: 'Dec 24, 09:00 AM', feature: 'UploadSize', old: 500, new: 600, auto: true },
          ].map((item, i) => (
            <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-slate-800/30 border border-slate-700/30">
              <div className="flex items-center gap-3">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center ${item.auto ? 'bg-indigo-500/20 text-indigo-400' : 'bg-slate-700/50 text-slate-400'}`}>
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={item.auto ? "M13 10V3L4 14h7v7l9-11h-7z" : "M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"} />
                  </svg>
                </div>
                <div>
                  <p className="text-sm font-medium text-slate-200">{item.feature}</p>
                  <p className="text-xs text-slate-500">{item.date} • {item.auto ? 'Auto-applied' : 'Manual'}</p>
                </div>
              </div>
              <div className="text-right">
                <div className="flex items-center gap-2 text-sm font-mono">
                  <span className="text-slate-500 line-through">{item.old}</span>
                  <svg className="w-3 h-3 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" /></svg>
                  <span className="text-emerald-400">{item.new}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Success Message */}
      <AnimatePresence>
        {applyMutation.isSuccess && (
          <motion.div
            className="p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-lg"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            role="alert"
          >
            <div className="flex items-center gap-3">
              <svg className="w-5 h-5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              <span className="text-emerald-400 font-medium">
                {applyMutation.data?.message || 'Baseline adjustment applied successfully'}
              </span>
            </div>
            {applyMutation.data?.model_retrained && (
              <p className="text-sm text-slate-400 mt-2 ml-8">
                Model retraining has been triggered. Results will update shortly.
              </p>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error Message */}
      <AnimatePresence>
        {applyMutation.isError && (
          <motion.div
            className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            role="alert"
          >
            <div className="flex items-center gap-3">
              <svg className="w-5 h-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
              <span className="text-red-400 font-medium">
                Failed to apply baseline adjustment. Please try again.
              </span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
