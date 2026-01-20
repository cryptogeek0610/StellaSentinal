/**
 * Correlation Matrix heatmap view component.
 */

import React from 'react';
import clsx from 'clsx';
import { Card } from '../Card';
import { ErrorState } from './ErrorState';
import {
  getCorrelationColor,
  getCorrelationTextColor,
  formatCorrelation,
  truncateMetric,
  formatPValue,
} from './correlationUtils';
import type { CorrelationMatrixResponse } from '../../types/correlations';

interface CorrelationMatrixViewProps {
  data?: CorrelationMatrixResponse;
  isLoading: boolean;
  error?: unknown;
  onRetry: () => void;
  onCellClick: (metricX: string, metricY: string) => void;
}

export function CorrelationMatrixView({ data, isLoading, error, onRetry, onCellClick }: CorrelationMatrixViewProps) {
  if (isLoading) {
    return (
      <Card title="Correlation Matrix">
        <div className="h-96 flex items-center justify-center text-slate-400">
          Loading correlation matrix...
        </div>
      </Card>
    );
  }

  if (error) {
    return <ErrorState title="Correlation Matrix" error={error} onRetry={onRetry} />;
  }

  if (!data?.metrics || !data?.matrix || data.metrics.length === 0) {
    return (
      <Card title="Correlation Matrix">
        <div className="h-96 flex flex-col items-center justify-center text-slate-400">
          <svg className="w-12 h-12 text-slate-600 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
          </svg>
          <p>No correlation data available</p>
          <p className="text-sm text-slate-500 mt-1">Try selecting a different domain or check data sync status</p>
        </div>
      </Card>
    );
  }

  const { metrics, matrix, strong_correlations, p_values, filter_stats, date_range, computed_at } = data;

  return (
    <div className="space-y-6">
      <Card title={<>Correlation Matrix <span className="text-slate-500 text-sm font-normal ml-2">{data.method} correlation ({metrics.length} metrics)</span></>}>
        {/* Filter Stats Info Panel */}
        {filter_stats && (
          <div className="mb-4 p-3 rounded-lg bg-slate-800/30 border border-slate-700/30">
            <div className="flex flex-wrap items-center gap-4 text-xs">
              <span className="text-slate-400">
                <span className="font-medium text-white">{filter_stats.passed}</span> of {filter_stats.total_input} metrics analyzed
              </span>
              {filter_stats.high_null > 0 && (
                <span className="text-amber-400">
                  {filter_stats.high_null} high-null filtered
                </span>
              )}
              {filter_stats.low_variance > 0 && (
                <span className="text-blue-400">
                  {filter_stats.low_variance} low-variance filtered
                </span>
              )}
              {filter_stats.low_cardinality > 0 && (
                <span className="text-purple-400">
                  {filter_stats.low_cardinality} low-cardinality filtered
                </span>
              )}
              {date_range && (
                <span className="ml-auto text-slate-500">
                  {date_range.start} to {date_range.end}
                </span>
              )}
            </div>
          </div>
        )}
        <div className="overflow-x-auto">
          <div className="inline-block min-w-full">
            {/* Matrix grid */}
            <div className="grid gap-0.5" style={{ gridTemplateColumns: `80px repeat(${metrics.length}, minmax(40px, 60px))` }}>
              {/* Header row */}
              <div className="h-20" /> {/* Empty corner */}
              {metrics.map((metric) => (
                <div
                  key={`header-${metric}`}
                  className="h-20 flex items-end justify-center pb-1"
                  title={metric}
                >
                  <span className="text-xs text-slate-400 transform -rotate-45 origin-bottom-left whitespace-nowrap">
                    {truncateMetric(metric)}
                  </span>
                </div>
              ))}

              {/* Data rows */}
              {metrics.map((rowMetric, rowIdx) => (
                <React.Fragment key={rowMetric}>
                  {/* Row label */}
                  <div
                    className="h-10 flex items-center justify-end pr-2"
                    title={rowMetric}
                  >
                    <span className="text-xs text-slate-400 truncate max-w-[75px]">
                      {truncateMetric(rowMetric)}
                    </span>
                  </div>
                  {/* Cells */}
                  {metrics.map((colMetric, colIdx) => {
                    const value = matrix[rowIdx][colIdx];
                    const pValue = p_values?.[rowIdx]?.[colIdx];
                    const isDiagonal = rowIdx === colIdx;
                    const tooltipText = isDiagonal
                      ? `${rowMetric}`
                      : `${rowMetric} vs ${colMetric}: r=${value.toFixed(3)}${pValue !== undefined ? `, p=${formatPValue(pValue)}` : ''}`;
                    return (
                      <div
                        key={`cell-${rowMetric}-${colMetric}`}
                        className={clsx(
                          'h-10 flex items-center justify-center text-xs font-mono rounded cursor-pointer transition-all',
                          isDiagonal ? 'bg-slate-700/50' : 'hover:ring-2 hover:ring-white/30 hover:scale-105'
                        )}
                        style={{ backgroundColor: isDiagonal ? undefined : getCorrelationColor(value) }}
                        onClick={() => !isDiagonal && onCellClick(rowMetric, colMetric)}
                        title={tooltipText}
                      >
                        <span style={{ color: isDiagonal ? '#64748b' : getCorrelationTextColor(value) }}>
                          {formatCorrelation(value)}
                        </span>
                      </div>
                    );
                  })}
                </React.Fragment>
              ))}
            </div>
          </div>
        </div>

        {/* Color Legend */}
        <div className="mt-6 flex items-center justify-center gap-6 text-xs text-slate-400">
          <span className="flex items-center gap-1">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: '#dc2626' }} />
            -1.0
          </span>
          <span className="flex items-center gap-1">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: '#ef4444' }} />
            -0.7
          </span>
          <span className="flex items-center gap-1">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: '#f97316' }} />
            -0.4
          </span>
          <span className="flex items-center gap-1">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: '#64748b' }} />
            0
          </span>
          <span className="flex items-center gap-1">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: '#eab308' }} />
            +0.4
          </span>
          <span className="flex items-center gap-1">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: '#84cc16' }} />
            +0.7
          </span>
          <span className="flex items-center gap-1">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: '#22c55e' }} />
            +1.0
          </span>
        </div>
      </Card>

      {/* Strong Correlations List */}
      {strong_correlations && strong_correlations.length > 0 && (
        <Card title={<>Strong Correlations <span className="text-slate-500 text-sm font-normal ml-2">|r| &gt; 0.6</span></>}>
          <div className="space-y-2">
            {strong_correlations.slice(0, 10).map((corr, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between p-3 rounded-lg bg-slate-800/50 border border-slate-700/30 hover:border-slate-600 cursor-pointer transition-all"
                onClick={() => onCellClick(corr.metric_x, corr.metric_y)}
              >
                <div className="flex items-center gap-3">
                  <div
                    className="w-2 h-8 rounded"
                    style={{ backgroundColor: getCorrelationColor(corr.correlation) }}
                  />
                  <div>
                    <div className="text-sm text-white font-medium">
                      {corr.metric_x} <span className="text-slate-500">↔</span> {corr.metric_y}
                    </div>
                    <div className="flex items-center gap-2 text-xs text-slate-500">
                      <span>{corr.sample_count.toLocaleString()} samples</span>
                      {corr.p_value !== null && (
                        <span className="text-slate-600">p={formatPValue(corr.p_value)}</span>
                      )}
                      {corr.is_significant && (
                        <span className="px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                          significant
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div
                    className="text-lg font-bold font-mono"
                    style={{ color: getCorrelationColor(corr.correlation) }}
                  >
                    r={formatCorrelation(corr.correlation)}
                  </div>
                  <div className="text-xs text-slate-500">
                    {corr.correlation > 0 ? 'Positive' : 'Negative'}
                  </div>
                </div>
              </div>
            ))}
          </div>
          {/* Footer with computed timestamp */}
          {computed_at && (
            <div className="mt-4 pt-3 border-t border-slate-700/30 text-xs text-slate-600 text-right">
              Computed: {new Date(computed_at).toLocaleString()}
            </div>
          )}
        </Card>
      )}
    </div>
  );
}
