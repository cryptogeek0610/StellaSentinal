/**
 * Scatter plot explorer view with metric selection and anomaly detection.
 */

import { useState, useMemo } from 'react';
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
import { Card } from '../Card';
import { ErrorState } from './ErrorState';
import { AnomalyDetailPanel } from './AnomalyDetailPanel';
import { formatCorrelation } from './correlationUtils';
import type { ScatterPlotResponse } from '../../types/correlations';

interface ScatterExplorerViewProps {
  data?: ScatterPlotResponse;
  isLoading: boolean;
  error?: unknown;
  onRetry: () => void;
  metrics: string[];
  selectedMetricX: string;
  selectedMetricY: string;
  onMetricXChange: (metric: string) => void;
  onMetricYChange: (metric: string) => void;
}

export function ScatterExplorerView({
  data,
  isLoading,
  error,
  onRetry,
  metrics,
  selectedMetricX,
  selectedMetricY,
  onMetricXChange,
  onMetricYChange,
}: ScatterExplorerViewProps) {
  // State for selected anomaly - moved BEFORE any early returns to fix React hooks violation
  const [selectedAnomaly, setSelectedAnomaly] = useState<{
    deviceId: number;
    x: number;
    y: number;
    cohort: string | null;
  } | null>(null);

  // Prepare scatter data for Recharts - also moved before early returns
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

  // Handle error state after hooks
  if (error) {
    return <ErrorState title="Scatter Plot" error={error} onRetry={onRetry} />;
  }

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
