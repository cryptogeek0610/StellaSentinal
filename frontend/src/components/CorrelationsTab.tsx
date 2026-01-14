/**
 * Correlations Tab Component
 *
 * Provides correlation intelligence including matrix heatmaps, scatter plots,
 * causal graphs, auto-discovered insights, cohort patterns, and time-lagged correlations.
 */

import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { Card } from './Card';
import { KPICard } from './KPICard';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ZAxis,
  Cell,
} from 'recharts';
import clsx from 'clsx';
import type {
  CorrelationMatrixResponse,
  ScatterPlotResponse,
  CausalGraphResponse,
  CorrelationInsightsResponse,
  CohortCorrelationPatternsResponse,
  TimeLagCorrelationsResponse,
  CausalNode,
} from '../types/correlations';

// Domain filter options
const DOMAIN_OPTIONS = [
  { id: undefined, label: 'All Domains' },
  { id: 'battery', label: 'Battery' },
  { id: 'rf', label: 'RF/Signal' },
  { id: 'throughput', label: 'Throughput' },
  { id: 'app', label: 'App Usage' },
  { id: 'system', label: 'System' },
];

// Sub-view types
type SubView = 'matrix' | 'scatter' | 'causal' | 'insights' | 'cohort' | 'lagged';

// Color scale for correlation values (-1 to +1)
const getCorrelationColor = (value: number): string => {
  if (value >= 0.7) return '#22c55e';  // Strong positive - green
  if (value >= 0.4) return '#84cc16';  // Moderate positive - lime
  if (value >= 0.1) return '#eab308';  // Weak positive - yellow
  if (value >= -0.1) return '#64748b'; // Near zero - slate
  if (value >= -0.4) return '#f97316'; // Weak negative - orange
  if (value >= -0.7) return '#ef4444'; // Moderate negative - red
  return '#dc2626';                     // Strong negative - dark red
};

// Get text color for contrast on correlation cell
const getCorrelationTextColor = (value: number): string => {
  const absValue = Math.abs(value);
  return absValue >= 0.5 ? '#ffffff' : '#1e293b';
};

// Format correlation value for display
const formatCorrelation = (value: number): string => {
  if (value === 1) return '1.00';
  if (value === -1) return '-1.0';
  return value.toFixed(2);
};

export function CorrelationsTab() {
  const [selectedDomain, setSelectedDomain] = useState<string | undefined>(undefined);
  const [selectedView, setSelectedView] = useState<SubView>('matrix');
  const [selectedMetricX, setSelectedMetricX] = useState<string>('');
  const [selectedMetricY, setSelectedMetricY] = useState<string>('');

  // Fetch correlation matrix
  const { data: matrixData, isLoading: matrixLoading } = useQuery({
    queryKey: ['correlations', 'matrix', selectedDomain],
    queryFn: () => api.getCorrelationMatrix({ domain: selectedDomain }),
  });

  // Fetch scatter data when metrics selected (increased limit from 500 to 2000 for fuller visualization)
  const { data: scatterData, isLoading: scatterLoading } = useQuery({
    queryKey: ['correlations', 'scatter', selectedMetricX, selectedMetricY],
    queryFn: () => api.getScatterData(selectedMetricX, selectedMetricY, 'anomaly', 2000),
    enabled: selectedView === 'scatter' && !!selectedMetricX && !!selectedMetricY,
  });

  // Fetch causal graph
  const { data: causalData, isLoading: causalLoading } = useQuery({
    queryKey: ['correlations', 'causal'],
    queryFn: () => api.getCausalGraph(),
    enabled: selectedView === 'causal',
  });

  // Fetch correlation insights
  const { data: insightsData, isLoading: insightsLoading } = useQuery({
    queryKey: ['correlations', 'insights'],
    queryFn: () => api.getCorrelationInsights(),
    enabled: selectedView === 'insights',
  });

  // Fetch cohort patterns
  const { data: cohortData, isLoading: cohortLoading } = useQuery({
    queryKey: ['correlations', 'cohort'],
    queryFn: () => api.getCohortCorrelationPatterns(),
    enabled: selectedView === 'cohort',
  });

  // Fetch time-lagged correlations
  const { data: laggedData, isLoading: laggedLoading } = useQuery({
    queryKey: ['correlations', 'lagged'],
    queryFn: () => api.getTimeLaggedCorrelations(),
    enabled: selectedView === 'lagged',
  });

  // Handle matrix cell click to open scatter view
  const handleMatrixCellClick = (metricX: string, metricY: string) => {
    if (metricX !== metricY) {
      setSelectedMetricX(metricX);
      setSelectedMetricY(metricY);
      setSelectedView('scatter');
    }
  };

  // Set default metrics for scatter view from matrix data
  useMemo(() => {
    if (matrixData?.metrics && matrixData.metrics.length >= 2 && !selectedMetricX && !selectedMetricY) {
      setSelectedMetricX(matrixData.metrics[0]);
      setSelectedMetricY(matrixData.metrics[1]);
    }
  }, [matrixData, selectedMetricX, selectedMetricY]);

  return (
    <div className="space-y-6">
      {/* KPI Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <KPICard
          title="Strong Correlations"
          value={matrixData?.strong_correlations?.length || '-'}
          color="aurora"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          }
          explainer="|r| > 0.6 threshold"
        />
        <KPICard
          title="Metrics Analyzed"
          value={matrixData?.metrics?.length || '-'}
          color="stellar"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          }
          explainer="Unique metrics in matrix"
        />
        <KPICard
          title="Causal Links"
          value={causalData?.edges?.length || '26'}
          color="warning"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
            </svg>
          }
          explainer="Domain knowledge relationships"
        />
        <KPICard
          title="Predictive Lags"
          value={laggedData?.correlations?.length || '-'}
          color="stellar"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          }
          explainer="Cross-time correlations"
        />
      </div>

      {/* Domain Filter & View Selector */}
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex gap-1 p-1 bg-slate-800/50 rounded-lg border border-slate-700/50">
          {DOMAIN_OPTIONS.map((domain) => (
            <button
              key={domain.id || 'all'}
              onClick={() => setSelectedDomain(domain.id)}
              className={clsx(
                'px-3 py-2 text-sm font-medium rounded-md transition-all',
                selectedDomain === domain.id
                  ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                  : 'text-slate-400 hover:text-white'
              )}
            >
              {domain.label}
            </button>
          ))}
        </div>

        <div className="flex gap-1 p-1 bg-slate-800/50 rounded-lg border border-slate-700/50">
          {[
            { id: 'matrix', label: 'Matrix' },
            { id: 'scatter', label: 'Scatter' },
            { id: 'causal', label: 'Causal' },
            { id: 'insights', label: 'Insights' },
            { id: 'cohort', label: 'Cohort' },
            { id: 'lagged', label: 'Lagged' },
          ].map((view) => (
            <button
              key={view.id}
              onClick={() => setSelectedView(view.id as SubView)}
              className={clsx(
                'px-4 py-2 text-sm font-medium rounded-md transition-all',
                selectedView === view.id
                  ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                  : 'text-slate-400 hover:text-white'
              )}
            >
              {view.label}
            </button>
          ))}
        </div>
      </div>

      {/* Correlation Matrix View */}
      {selectedView === 'matrix' && (
        <CorrelationMatrixView
          data={matrixData}
          isLoading={matrixLoading}
          onCellClick={handleMatrixCellClick}
        />
      )}

      {/* Scatter Plot View */}
      {selectedView === 'scatter' && (
        <ScatterExplorerView
          data={scatterData}
          isLoading={scatterLoading}
          metrics={matrixData?.metrics || []}
          selectedMetricX={selectedMetricX}
          selectedMetricY={selectedMetricY}
          onMetricXChange={setSelectedMetricX}
          onMetricYChange={setSelectedMetricY}
        />
      )}

      {/* Causal Graph View */}
      {selectedView === 'causal' && (
        <CausalGraphView data={causalData} isLoading={causalLoading} />
      )}

      {/* Insights View */}
      {selectedView === 'insights' && (
        <InsightsView data={insightsData} isLoading={insightsLoading} />
      )}

      {/* Cohort Patterns View */}
      {selectedView === 'cohort' && (
        <CohortPatternsView data={cohortData} isLoading={cohortLoading} />
      )}

      {/* Time-Lagged View */}
      {selectedView === 'lagged' && (
        <TimeLaggedView data={laggedData} isLoading={laggedLoading} />
      )}
    </div>
  );
}

// =============================================================================
// Correlation Matrix View
// =============================================================================

interface CorrelationMatrixViewProps {
  data?: CorrelationMatrixResponse;
  isLoading: boolean;
  onCellClick: (metricX: string, metricY: string) => void;
}

function CorrelationMatrixView({ data, isLoading, onCellClick }: CorrelationMatrixViewProps) {
  if (isLoading) {
    return (
      <Card title="Correlation Matrix">
        <div className="h-96 flex items-center justify-center text-slate-400">
          Loading correlation matrix...
        </div>
      </Card>
    );
  }

  if (!data?.metrics || !data?.matrix) {
    return (
      <Card title="Correlation Matrix">
        <div className="h-96 flex items-center justify-center text-slate-400">
          No correlation data available
        </div>
      </Card>
    );
  }

  const { metrics, matrix, strong_correlations } = data;
  // Truncate long metric names for display
  const truncateMetric = (name: string, maxLen: number = 12): string => {
    if (name.length <= maxLen) return name;
    return name.substring(0, maxLen - 2) + '..';
  };

  return (
    <div className="space-y-6">
      <Card title={<>Correlation Matrix <span className="text-slate-500 text-sm font-normal ml-2">{data.method} correlation ({metrics.length} metrics)</span></>}>
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
                <>
                  {/* Row label */}
                  <div
                    key={`label-${rowMetric}`}
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
                    const isDiagonal = rowIdx === colIdx;
                    return (
                      <div
                        key={`cell-${rowMetric}-${colMetric}`}
                        className={clsx(
                          'h-10 flex items-center justify-center text-xs font-mono rounded cursor-pointer transition-all',
                          isDiagonal ? 'bg-slate-700/50' : 'hover:ring-2 hover:ring-white/30 hover:scale-105'
                        )}
                        style={{ backgroundColor: isDiagonal ? undefined : getCorrelationColor(value) }}
                        onClick={() => !isDiagonal && onCellClick(rowMetric, colMetric)}
                        title={`${rowMetric} vs ${colMetric}: r=${value.toFixed(3)}`}
                      >
                        <span style={{ color: isDiagonal ? '#64748b' : getCorrelationTextColor(value) }}>
                          {formatCorrelation(value)}
                        </span>
                      </div>
                    );
                  })}
                </>
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
                    <div className="text-xs text-slate-500">
                      {corr.sample_count.toLocaleString()} samples
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
        </Card>
      )}
    </div>
  );
}

// =============================================================================
// Scatter Explorer View
// =============================================================================

// Anomaly detail panel for clicked points
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

function AnomalyDetailPanel({ point, metricX, metricY, onClose }: AnomalyDetailPanelProps) {
  if (!point) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-full max-w-md mx-4 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 bg-red-500/10 border-b border-red-500/20">
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

interface ScatterExplorerViewProps {
  data?: ScatterPlotResponse;
  isLoading: boolean;
  metrics: string[];
  selectedMetricX: string;
  selectedMetricY: string;
  onMetricXChange: (metric: string) => void;
  onMetricYChange: (metric: string) => void;
}

function ScatterExplorerView({
  data,
  isLoading,
  metrics,
  selectedMetricX,
  selectedMetricY,
  onMetricXChange,
  onMetricYChange,
}: ScatterExplorerViewProps) {
  const [selectedAnomaly, setSelectedAnomaly] = useState<{
    deviceId: number;
    x: number;
    y: number;
    cohort: string | null;
  } | null>(null);

  // Prepare scatter data for Recharts
  const chartData = useMemo(() => {
    if (!data?.points) return [];
    return data.points.map((p) => ({
      x: p.x_value,
      y: p.y_value,
      z: 50, // constant size for now
      isAnomaly: p.is_anomaly,
      deviceId: p.device_id,
      cohort: p.cohort,
    }));
  }, [data]);

  const normalData = chartData.filter((p) => !p.isAnomaly);
  const anomalyData = chartData.filter((p) => p.isAnomaly);

  return (
    <div className="space-y-6">
      {/* Metric Selectors */}
      <Card title="Select Metrics to Compare">
        <div className="flex flex-wrap gap-4">
          <div className="flex-1 min-w-[200px]">
            <label className="block text-xs text-slate-500 mb-1">X-Axis Metric</label>
            <select
              value={selectedMetricX}
              onChange={(e) => onMetricXChange(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              {metrics.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>
          <div className="flex-1 min-w-[200px]">
            <label className="block text-xs text-slate-500 mb-1">Y-Axis Metric</label>
            <select
              value={selectedMetricY}
              onChange={(e) => onMetricYChange(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              {metrics.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>
        </div>
      </Card>

      {/* Scatter Chart */}
      <Card title={
        <>
          {selectedMetricX} vs {selectedMetricY}
          {data && (
            <span className="text-slate-500 text-sm font-normal ml-2">
              r={formatCorrelation(data.correlation)} | R²={data.r_squared?.toFixed(3) || 'N/A'}
            </span>
          )}
        </>
      }>
        {isLoading ? (
          <div className="h-96 flex items-center justify-center text-slate-400">
            Loading scatter data...
          </div>
        ) : chartData.length > 0 ? (
          <div className="space-y-4">
            <ResponsiveContainer width="100%" height={400}>
              <ScatterChart margin={{ top: 20, right: 20, bottom: 60, left: 60 }}>
                <XAxis
                  type="number"
                  dataKey="x"
                  name={selectedMetricX}
                  stroke="#64748b"
                  fontSize={11}
                  tickLine={false}
                  label={{ value: selectedMetricX, position: 'bottom', offset: 40, fill: '#64748b', fontSize: 12 }}
                />
                <YAxis
                  type="number"
                  dataKey="y"
                  name={selectedMetricY}
                  stroke="#64748b"
                  fontSize={11}
                  tickLine={false}
                  axisLine={false}
                  label={{ value: selectedMetricY, angle: -90, position: 'left', offset: 40, fill: '#64748b', fontSize: 12 }}
                />
                <ZAxis type="number" dataKey="z" range={[30, 100]} />
                <Tooltip
                  contentStyle={{
                    background: '#1e293b',
                    border: '1px solid #334155',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                  formatter={(value: number, name: string) => [value.toFixed(2), name]}
                  labelFormatter={() => ''}
                />
                {/* Regression line approximation using reference line */}
                {data?.regression_slope != null && data?.regression_intercept != null && (
                  <ReferenceLine
                    segment={[
                      { x: Math.min(...chartData.map(p => p.x)), y: data.regression_intercept! + data.regression_slope! * Math.min(...chartData.map(p => p.x)) },
                      { x: Math.max(...chartData.map(p => p.x)), y: data.regression_intercept! + data.regression_slope! * Math.max(...chartData.map(p => p.x)) },
                    ]}
                    stroke="#f59e0b"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                  />
                )}
                {/* Normal points */}
                <Scatter name="Normal" data={normalData} fill="#3b82f6" fillOpacity={0.6} />
                {/* Anomaly points - clickable with Cell components */}
                <Scatter name="Anomaly" data={anomalyData} fill="#ef4444" fillOpacity={0.8}>
                  {anomalyData.map((entry, index) => (
                    <Cell
                      key={`anomaly-${index}`}
                      cursor="pointer"
                      onClick={() => {
                        setSelectedAnomaly({
                          deviceId: entry.deviceId,
                          x: entry.x,
                          y: entry.y,
                          cohort: entry.cohort,
                        });
                      }}
                    />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>

            {/* Anomaly Detail Panel */}
            <AnomalyDetailPanel
              point={selectedAnomaly}
              metricX={selectedMetricX}
              metricY={selectedMetricY}
              onClose={() => setSelectedAnomaly(null)}
            />

            {/* Stats Summary */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/30 text-center">
                <div className="text-2xl font-bold font-mono text-blue-400">{formatCorrelation(data?.correlation || 0)}</div>
                <div className="text-xs text-slate-500">Correlation (r)</div>
              </div>
              <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/30 text-center">
                <div className="text-2xl font-bold font-mono text-amber-400">{data?.r_squared?.toFixed(3) || 'N/A'}</div>
                <div className="text-xs text-slate-500">R² Score</div>
              </div>
              <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/30 text-center">
                <div className="text-2xl font-bold font-mono text-white">{data?.total_points?.toLocaleString() || 0}</div>
                <div className="text-xs text-slate-500">Total Points</div>
              </div>
              <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/30 text-center">
                <div className="text-2xl font-bold font-mono text-red-400">{data?.anomaly_count || 0}</div>
                <div className="text-xs text-slate-500">Anomalies</div>
              </div>
            </div>

            {/* Legend */}
            <div className="flex items-center justify-center gap-6 text-xs text-slate-400">
              <span className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-blue-500" />
                Normal ({normalData.length})
              </span>
              <span className="flex items-center gap-2 cursor-pointer group" title="Click any red point for details">
                <div className="w-3 h-3 rounded-full bg-red-500 group-hover:ring-2 group-hover:ring-red-400/50" />
                <span className="group-hover:text-red-400 transition-colors">Anomaly ({anomalyData.length})</span>
                <span className="text-slate-600 group-hover:text-slate-400">- click for details</span>
              </span>
              <span className="flex items-center gap-2">
                <div className="w-6 h-0.5 bg-amber-500" style={{ borderStyle: 'dashed' }} />
                Regression Line
              </span>
            </div>
          </div>
        ) : (
          <div className="h-96 flex items-center justify-center text-slate-400">
            Select metrics to view scatter plot
          </div>
        )}
      </Card>
    </div>
  );
}

// =============================================================================
// Causal Graph View
// =============================================================================

interface CausalGraphViewProps {
  data?: CausalGraphResponse;
  isLoading: boolean;
}

function CausalGraphView({ data, isLoading }: CausalGraphViewProps) {
  if (isLoading) {
    return (
      <Card title="Causal Relationship Graph">
        <div className="h-96 flex items-center justify-center text-slate-400">
          Loading causal graph...
        </div>
      </Card>
    );
  }

  if (!data?.nodes || !data?.edges) {
    return (
      <Card title="Causal Relationship Graph">
        <div className="h-96 flex items-center justify-center text-slate-400">
          No causal data available
        </div>
      </Card>
    );
  }

  const { nodes, edges } = data;

  // Position nodes: causes on left, effects on right, both in center
  const getNodePosition = (node: CausalNode): { x: number; y: number } => {
    const padding = 60;
    const width = 800;
    const height = 500;

    if (node.is_cause && !node.is_effect) {
      // Pure causes on the left
      const causesOnly = nodes.filter(n => n.is_cause && !n.is_effect);
      const idx = causesOnly.indexOf(node);
      return {
        x: padding,
        y: padding + (idx / Math.max(causesOnly.length - 1, 1)) * (height - 2 * padding),
      };
    } else if (!node.is_cause && node.is_effect) {
      // Pure effects on the right
      const effectsOnly = nodes.filter(n => !n.is_cause && n.is_effect);
      const idx = effectsOnly.indexOf(node);
      return {
        x: width - padding,
        y: padding + (idx / Math.max(effectsOnly.length - 1, 1)) * (height - 2 * padding),
      };
    } else {
      // Both cause and effect in center
      const both = nodes.filter(n => n.is_cause && n.is_effect);
      const idx = both.indexOf(node);
      return {
        x: width / 2,
        y: padding + (idx / Math.max(both.length - 1, 1)) * (height - 2 * padding),
      };
    }
  };

  // Create positioned nodes
  const positionedNodes = nodes.map((node) => ({
    ...node,
    position: getNodePosition(node),
  }));

  // Get node color by domain
  const getDomainColor = (domain: string): string => {
    const colors: Record<string, string> = {
      battery: '#22c55e',
      rf: '#3b82f6',
      throughput: '#8b5cf6',
      app: '#f59e0b',
      system: '#ef4444',
      network: '#06b6d4',
      storage: '#ec4899',
    };
    return colors[domain.toLowerCase()] || '#64748b';
  };

  return (
    <div className="space-y-6">
      <Card title={<>Causal Relationship Graph <span className="text-slate-500 text-sm font-normal ml-2">{nodes.length} metrics, {edges.length} relationships</span></>}>
        <div className="overflow-x-auto">
          <svg width={800} height={500} className="mx-auto">
            {/* Draw edges */}
            {edges.map((edge, idx) => {
              const sourceNode = positionedNodes.find(n => n.metric === edge.source);
              const targetNode = positionedNodes.find(n => n.metric === edge.target);
              if (!sourceNode || !targetNode) return null;

              const dx = targetNode.position.x - sourceNode.position.x;
              const dy = targetNode.position.y - sourceNode.position.y;
              const dist = Math.sqrt(dx * dx + dy * dy);
              const offsetX = (dx / dist) * 35;
              const offsetY = (dy / dist) * 35;

              return (
                <g key={`edge-${idx}`}>
                  <defs>
                    <marker
                      id={`arrow-${idx}`}
                      markerWidth="8"
                      markerHeight="8"
                      refX="8"
                      refY="4"
                      orient="auto"
                    >
                      <path
                        d="M0,0 L8,4 L0,8 Z"
                        fill={edge.relationship === 'causes' ? '#f59e0b' : '#64748b'}
                      />
                    </marker>
                  </defs>
                  <line
                    x1={sourceNode.position.x + offsetX}
                    y1={sourceNode.position.y + offsetY}
                    x2={targetNode.position.x - offsetX}
                    y2={targetNode.position.y - offsetY}
                    stroke={edge.relationship === 'causes' ? '#f59e0b' : '#64748b'}
                    strokeWidth={Math.max(1, edge.strength * 3)}
                    strokeOpacity={0.6}
                    markerEnd={`url(#arrow-${idx})`}
                  />
                </g>
              );
            })}

            {/* Draw nodes */}
            {positionedNodes.map((node) => (
              <g key={node.metric} transform={`translate(${node.position.x}, ${node.position.y})`}>
                <circle
                  r={30}
                  fill={getDomainColor(node.domain)}
                  fillOpacity={0.2}
                  stroke={getDomainColor(node.domain)}
                  strokeWidth={2}
                />
                <text
                  textAnchor="middle"
                  dy="4"
                  className="text-xs fill-white font-medium"
                  style={{ fontSize: '9px' }}
                >
                  {node.metric.length > 12 ? node.metric.substring(0, 10) + '..' : node.metric}
                </text>
              </g>
            ))}
          </svg>
        </div>

        {/* Domain Legend */}
        <div className="mt-4 flex flex-wrap items-center justify-center gap-4 text-xs text-slate-400">
          {['Battery', 'RF', 'Throughput', 'App', 'System', 'Network'].map((domain) => (
            <span key={domain} className="flex items-center gap-1">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: getDomainColor(domain) }} />
              {domain}
            </span>
          ))}
        </div>
      </Card>

      {/* Causal Edges List */}
      <Card title="Causal Relationships">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-h-96 overflow-y-auto">
          {edges.filter(e => e.relationship === 'causes').map((edge, idx) => (
            <div
              key={idx}
              className="flex items-center gap-2 p-2 rounded-lg bg-slate-800/30 border border-slate-700/30"
            >
              <span className="text-xs font-medium text-amber-400">{edge.source}</span>
              <svg className="w-4 h-4 text-amber-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
              <span className="text-xs font-medium text-white">{edge.target}</span>
              <span className="ml-auto text-xs text-slate-500">{(edge.strength * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

// =============================================================================
// Insights View
// =============================================================================

interface InsightsViewProps {
  data?: CorrelationInsightsResponse;
  isLoading: boolean;
}

function InsightsView({ data, isLoading }: InsightsViewProps) {
  if (isLoading) {
    return (
      <Card title="Correlation Insights">
        <div className="h-96 flex items-center justify-center text-slate-400">
          Loading insights...
        </div>
      </Card>
    );
  }

  if (!data?.insights || data.insights.length === 0) {
    return (
      <Card title="Correlation Insights">
        <div className="h-96 flex items-center justify-center text-slate-400">
          No insights available
        </div>
      </Card>
    );
  }

  const getStrengthColor = (strength: string): string => {
    switch (strength) {
      case 'strong': return 'bg-green-500/20 text-green-400 border-green-500/30';
      case 'moderate': return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
      default: return 'bg-slate-500/20 text-slate-400 border-slate-500/30';
    }
  };

  const getDirectionIcon = (direction: string) => {
    if (direction === 'positive') {
      return (
        <svg className="w-5 h-5 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
        </svg>
      );
    }
    return (
      <svg className="w-5 h-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
      </svg>
    );
  };

  return (
    <Card title={<>Auto-Discovered Insights <span className="text-slate-500 text-sm font-normal ml-2">{data.total_correlations_analyzed} correlations analyzed</span></>}>
      <div className="space-y-4">
        {data.insights.map((insight) => (
          <div
            key={insight.insight_id}
            className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/30 hover:border-slate-600 transition-all"
          >
            <div className="flex items-start justify-between gap-4">
              <div className="flex items-start gap-3">
                {getDirectionIcon(insight.direction)}
                <div>
                  <h4 className="text-white font-medium">{insight.headline}</h4>
                  <p className="text-sm text-slate-400 mt-1">{insight.description}</p>
                  {insight.recommendation && (
                    <p className="text-sm text-blue-400 mt-2">
                      <span className="font-medium">Recommendation:</span> {insight.recommendation}
                    </p>
                  )}
                  <div className="flex flex-wrap gap-2 mt-3">
                    {insight.metrics_involved.map((metric) => (
                      <span
                        key={metric}
                        className="px-2 py-0.5 text-xs rounded-full bg-slate-700/50 text-slate-300"
                      >
                        {metric}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
              <div className="flex flex-col items-end gap-2 shrink-0">
                <span
                  className={clsx(
                    'px-2 py-1 text-xs font-medium rounded-full border',
                    getStrengthColor(insight.strength)
                  )}
                >
                  {insight.strength}
                </span>
                <span className="text-lg font-bold font-mono" style={{ color: getCorrelationColor(insight.correlation_value) }}>
                  r={formatCorrelation(insight.correlation_value)}
                </span>
                <span className="text-xs text-slate-500">
                  {(insight.confidence * 100).toFixed(0)}% confidence
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}

// =============================================================================
// Cohort Patterns View
// =============================================================================

interface CohortPatternsViewProps {
  data?: CohortCorrelationPatternsResponse;
  isLoading: boolean;
}

function CohortPatternsView({ data, isLoading }: CohortPatternsViewProps) {
  if (isLoading) {
    return (
      <Card title="Cohort Correlation Patterns">
        <div className="h-96 flex items-center justify-center text-slate-400">
          Loading cohort patterns...
        </div>
      </Card>
    );
  }

  if (!data?.patterns || data.patterns.length === 0) {
    return (
      <Card title="Cohort Correlation Patterns">
        <div className="h-96 flex items-center justify-center text-slate-400">
          No cohort patterns available
        </div>
      </Card>
    );
  }

  return (
    <Card title={<>Cohort Correlation Patterns <span className="text-slate-500 text-sm font-normal ml-2">{data.anomalous_cohorts} anomalous cohorts found</span></>}>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-slate-500 border-b border-slate-700/50">
              <th className="pb-3 font-medium">Cohort</th>
              <th className="pb-3 font-medium">Metric Pair</th>
              <th className="pb-3 font-medium text-right">Cohort r</th>
              <th className="pb-3 font-medium text-right">Fleet r</th>
              <th className="pb-3 font-medium text-right">Deviation</th>
              <th className="pb-3 font-medium text-right">Devices</th>
              <th className="pb-3 font-medium">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-700/30">
            {data.patterns.map((pattern) => (
              <tr key={pattern.cohort_id} className="hover:bg-slate-800/30 transition-colors">
                <td className="py-3">
                  <span className="font-medium text-white">{pattern.cohort_name}</span>
                </td>
                <td className="py-3">
                  <span className="text-slate-400">
                    {pattern.metric_pair[0]} <span className="text-slate-600">↔</span> {pattern.metric_pair[1]}
                  </span>
                </td>
                <td className="py-3 text-right">
                  <span className="font-mono" style={{ color: getCorrelationColor(pattern.cohort_correlation) }}>
                    {formatCorrelation(pattern.cohort_correlation)}
                  </span>
                </td>
                <td className="py-3 text-right">
                  <span className="font-mono text-slate-400">
                    {formatCorrelation(pattern.fleet_correlation)}
                  </span>
                </td>
                <td className="py-3 text-right">
                  <span
                    className={clsx(
                      'font-mono font-medium',
                      Math.abs(pattern.deviation) > 0.3 ? 'text-red-400' : 'text-slate-400'
                    )}
                  >
                    {pattern.deviation > 0 ? '+' : ''}{formatCorrelation(pattern.deviation)}
                  </span>
                </td>
                <td className="py-3 text-right text-slate-400">
                  {pattern.device_count.toLocaleString()}
                </td>
                <td className="py-3">
                  {pattern.is_anomalous ? (
                    <span className="px-2 py-0.5 text-xs font-medium rounded-full bg-red-500/20 text-red-400 border border-red-500/30">
                      Anomalous
                    </span>
                  ) : (
                    <span className="px-2 py-0.5 text-xs font-medium rounded-full bg-slate-500/20 text-slate-400 border border-slate-500/30">
                      Normal
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Insight cards for anomalous patterns */}
      {data.patterns.filter(p => p.is_anomalous && p.insight).length > 0 && (
        <div className="mt-6 space-y-3">
          <h4 className="text-sm font-medium text-slate-400">Anomalous Pattern Insights</h4>
          {data.patterns.filter(p => p.is_anomalous && p.insight).map((pattern) => (
            <div
              key={pattern.cohort_id + '-insight'}
              className="p-3 rounded-lg bg-red-500/10 border border-red-500/20"
            >
              <div className="flex items-start gap-2">
                <svg className="w-4 h-4 text-red-400 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                <div>
                  <span className="text-sm font-medium text-red-400">{pattern.cohort_name}</span>
                  <p className="text-sm text-slate-300 mt-1">{pattern.insight}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}

// =============================================================================
// Time-Lagged Correlations View
// =============================================================================

interface TimeLaggedViewProps {
  data?: TimeLagCorrelationsResponse;
  isLoading: boolean;
}

function TimeLaggedView({ data, isLoading }: TimeLaggedViewProps) {
  if (isLoading) {
    return (
      <Card title="Time-Lagged Correlations">
        <div className="h-96 flex items-center justify-center text-slate-400">
          Loading time-lagged correlations...
        </div>
      </Card>
    );
  }

  if (!data?.correlations || data.correlations.length === 0) {
    return (
      <Card title="Time-Lagged Correlations">
        <div className="h-96 flex items-center justify-center text-slate-400">
          No time-lagged correlations found
        </div>
      </Card>
    );
  }

  return (
    <Card title={<>Predictive Correlations <span className="text-slate-500 text-sm font-normal ml-2">Cross-time relationships (up to {data.max_lag_analyzed} day lag)</span></>}>
      <div className="space-y-4">
        {data.correlations.map((corr, idx) => (
          <div
            key={idx}
            className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/30"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-lg bg-purple-500/20 flex items-center justify-center">
                  <span className="text-lg font-bold text-purple-400">D{corr.lag_days}</span>
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-white">{corr.metric_a}</span>
                    <svg className="w-5 h-5 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                    </svg>
                    <span className="font-medium text-white">{corr.metric_b}</span>
                  </div>
                  <p className="text-sm text-slate-400 mt-1">{corr.insight}</p>
                </div>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold font-mono" style={{ color: getCorrelationColor(corr.correlation) }}>
                  r={formatCorrelation(corr.correlation)}
                </div>
                <div className="text-xs text-slate-500 mt-1">
                  {corr.lag_days} day{corr.lag_days > 1 ? 's' : ''} lag
                </div>
                {corr.p_value !== null && (
                  <div className="text-xs text-slate-600">
                    p={corr.p_value.toFixed(4)}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}

        {/* Explanation */}
        <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
          <div className="flex items-start gap-2">
            <svg className="w-5 h-5 text-blue-400 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div>
              <span className="text-sm font-medium text-blue-400">What are time-lagged correlations?</span>
              <p className="text-sm text-slate-300 mt-1">
                These correlations show predictive relationships where changes in one metric today predict
                changes in another metric in the future. For example, a drop in signal strength today may
                predict increased battery drain tomorrow.
              </p>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
}

export default CorrelationsTab;
