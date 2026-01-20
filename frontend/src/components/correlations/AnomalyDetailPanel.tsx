/**
 * Anomaly detail panel for clicked scatter plot points.
 */

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import clsx from 'clsx';
import { api } from '../../api/client';

interface AnomalyDetailPanelProps {
  point: {
    deviceId: number;
    x: number;
    y: number;
    cohort: string | null;
  } | null;
  metricX: string;
  metricY: string;
  onClose: () => void;
}

export function AnomalyDetailPanel({ point, metricX, metricY, onClose }: AnomalyDetailPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  // Fetch explanation when panel opens
  const { data: explanation, isLoading: explanationLoading } = useQuery({
    queryKey: ['scatter-explanation', point?.deviceId, metricX, metricY, point?.x, point?.y],
    queryFn: () => api.explainScatterAnomaly({
      deviceId: point!.deviceId,
      metricX,
      metricY,
      xValue: point!.x,
      yValue: point!.y,
    }),
    enabled: !!point,
    staleTime: 5 * 60 * 1000, // Cache for 5 minutes
  });

  if (!point) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-full max-w-lg mx-4 overflow-hidden max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 bg-red-500/10 border-b border-red-500/20 sticky top-0 bg-slate-900/95 backdrop-blur-sm z-10">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse" />
            <span className="font-medium text-red-400">Anomaly Detected</span>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg hover:bg-slate-700 transition-colors"
          >
            <svg className="w-5 h-5 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4">
          {/* Device ID */}
          <div className="flex items-center justify-between p-3 rounded-lg bg-slate-800/50 border border-slate-700/30">
            <span className="text-sm text-slate-400">Device ID</span>
            <span className="font-mono font-medium text-white">{point.deviceId}</span>
          </div>

          {/* Metric Values */}
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/30">
              <div className="text-xs text-slate-500 mb-1 truncate" title={metricX}>{metricX}</div>
              <div className="text-lg font-mono font-bold text-blue-400">{point.x.toFixed(2)}</div>
            </div>
            <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/30">
              <div className="text-xs text-slate-500 mb-1 truncate" title={metricY}>{metricY}</div>
              <div className="text-lg font-mono font-bold text-blue-400">{point.y.toFixed(2)}</div>
            </div>
          </div>

          {/* Cohort */}
          {point.cohort && (
            <div className="flex items-center justify-between p-3 rounded-lg bg-slate-800/50 border border-slate-700/30">
              <span className="text-sm text-slate-400">Cohort</span>
              <span className="px-2 py-1 text-xs font-medium rounded-full bg-purple-500/20 text-purple-400 border border-purple-500/30">
                {point.cohort}
              </span>
            </div>
          )}

          {/* AI Explanation Section */}
          <div className="rounded-lg bg-gradient-to-br from-purple-500/10 to-blue-500/10 border border-purple-500/20">
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="w-full flex items-center justify-between p-3 hover:bg-white/5 transition-colors rounded-t-lg"
            >
              <div className="flex items-center gap-2">
                <svg className="w-5 h-5 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
                <span className="font-medium text-purple-300">AI Explanation</span>
              </div>
              <svg
                className={clsx(
                  'w-5 h-5 text-slate-400 transition-transform',
                  isExpanded && 'rotate-180'
                )}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>

            {/* Expandable Explanation Content */}
            <div className={clsx(
              'overflow-hidden transition-all duration-300',
              isExpanded ? 'max-h-[500px] opacity-100' : 'max-h-0 opacity-0'
            )}>
              <div className="px-3 pb-3 space-y-3">
                {explanationLoading ? (
                  <div className="flex items-center justify-center py-6 gap-2 text-slate-400">
                    <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    <span className="text-sm">Analyzing anomaly...</span>
                  </div>
                ) : explanation ? (
                  <>
                    {/* What Happened */}
                    <div className="p-3 rounded-lg bg-slate-800/50">
                      <div className="text-xs font-medium text-amber-400 mb-1 uppercase tracking-wider">What Happened</div>
                      <p className="text-sm text-slate-300">{explanation.what_happened}</p>
                    </div>

                    {/* Key Concerns */}
                    <div className="p-3 rounded-lg bg-slate-800/50">
                      <div className="text-xs font-medium text-red-400 mb-2 uppercase tracking-wider">Key Concerns</div>
                      <ul className="space-y-1">
                        {explanation.key_concerns.map((concern, idx) => (
                          <li key={idx} className="flex items-start gap-2 text-sm text-slate-300">
                            <span className="text-red-400 mt-0.5">•</span>
                            {concern}
                          </li>
                        ))}
                      </ul>
                    </div>

                    {/* Likely Explanation */}
                    <div className="p-3 rounded-lg bg-slate-800/50">
                      <div className="text-xs font-medium text-blue-400 mb-1 uppercase tracking-wider">Likely Explanation</div>
                      <p className="text-sm text-slate-300">{explanation.likely_explanation}</p>
                    </div>

                    {/* Suggested Action */}
                    <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
                      <div className="text-xs font-medium text-emerald-400 mb-1 uppercase tracking-wider">Suggested Action</div>
                      <p className="text-sm text-slate-300">{explanation.suggested_action}</p>
                    </div>
                  </>
                ) : (
                  <div className="py-4 text-center text-slate-500 text-sm">
                    Unable to generate explanation
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-2 pt-2">
            <a
              href={`/devices/${point.deviceId}`}
              className="flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-blue-500/20 text-blue-400 border border-blue-500/30 hover:bg-blue-500/30 transition-colors"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
              </svg>
              View Device Profile
            </a>
            <button
              onClick={onClose}
              className="px-4 py-2 rounded-lg bg-slate-700 text-slate-300 hover:bg-slate-600 transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
