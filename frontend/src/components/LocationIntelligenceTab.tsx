/**
 * Location Intelligence Tab Component
 *
 * Displays WiFi coverage heatmaps, dead zones, and device mobility analytics.
 */

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { Card } from './Card';
import { KPICard } from './KPICard';
import clsx from 'clsx';

// Signal strength color scale
const getSignalColor = (signal: number): string => {
  if (signal >= -50) return '#10b981'; // Excellent - green
  if (signal >= -60) return '#22c55e'; // Good - lighter green
  if (signal >= -70) return '#f59e0b'; // Fair - amber
  if (signal >= -80) return '#f97316'; // Poor - orange
  return '#ef4444'; // Bad - red
};

const getSignalLabel = (signal: number): string => {
  if (signal >= -50) return 'Excellent';
  if (signal >= -60) return 'Good';
  if (signal >= -70) return 'Fair';
  if (signal >= -80) return 'Poor';
  return 'Very Poor';
};

export function LocationIntelligenceTab() {
  const [selectedView, setSelectedView] = useState<'heatmap' | 'deadzones' | 'dwell'>('heatmap');

  // Fetch WiFi heatmap
  const { data: heatmap, isLoading: heatmapLoading } = useQuery({
    queryKey: ['insights', 'location', 'heatmap'],
    queryFn: () => api.getWiFiHeatmap(7),
    enabled: selectedView === 'heatmap',
  });

  // Fetch dead zones
  const { data: deadZones, isLoading: deadZonesLoading } = useQuery({
    queryKey: ['insights', 'location', 'dead-zones'],
    queryFn: () => api.getDeadZones(-75, 50),
    enabled: selectedView === 'deadzones',
  });

  // Fetch dwell time analysis
  const { data: dwellTime, isLoading: dwellLoading } = useQuery({
    queryKey: ['insights', 'location', 'dwell-time'],
    queryFn: () => api.getDwellTime(7),
    enabled: selectedView === 'dwell',
  });

  // Fetch coverage summary
  const { data: coverage } = useQuery({
    queryKey: ['insights', 'location', 'coverage'],
    queryFn: () => api.getCoverageSummary(7),
  });

  return (
    <div className="space-y-6">
      {/* KPI Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <KPICard
          title="Coverage"
          value={`${coverage?.coverage_percentage || 0}%`}
          color={coverage && coverage.coverage_percentage >= 90 ? 'aurora' : coverage && coverage.coverage_percentage >= 75 ? 'warning' : 'danger'}
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8.111 16.404a5.5 5.5 0 017.778 0M12 20h.01m-7.08-7.071c3.904-3.905 10.236-3.905 14.141 0M1.394 9.393c5.857-5.857 15.355-5.857 21.213 0" />
            </svg>
          }
          explainer="WiFi coverage area"
        />
        <KPICard
          title="Avg Signal"
          value={`${coverage?.avg_signal?.toFixed(0) || '-'} dBm`}
          color={coverage && coverage.avg_signal >= -60 ? 'aurora' : coverage && coverage.avg_signal >= -75 ? 'warning' : 'danger'}
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          }
          explainer={coverage ? getSignalLabel(coverage.avg_signal) : '-'}
        />
        <KPICard
          title="Dead Zones"
          value={heatmap?.dead_zone_count || deadZones?.total_count || 0}
          color={heatmap && heatmap.dead_zone_count > 5 ? 'danger' : heatmap && heatmap.dead_zone_count > 0 ? 'warning' : 'aurora'}
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M18.364 5.636a9 9 0 010 12.728m0 0l-2.829-2.829m2.829 2.829L21 21M15.536 8.464a5 5 0 010 7.072m0 0l-2.829-2.829m-4.243 2.829a4.978 4.978 0 01-1.414-2.83m-1.414 5.658a9 9 0 01-2.167-9.238m7.824 2.167a1 1 0 111.414 1.414m-1.414-1.414L3 3m8.293 8.293l1.414 1.414" />
            </svg>
          }
          explainer="Areas with poor signal"
        />
        <KPICard
          title="Total Readings"
          value={coverage?.total_readings?.toLocaleString() || '0'}
          color="stellar"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          }
          explainer="Last 7 days"
        />
      </div>

      {/* View Selector */}
      <div className="flex gap-2 p-1 bg-slate-800/50 rounded-lg border border-slate-700/50">
        {[
          { id: 'heatmap', label: 'Signal Heatmap', icon: (
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          ) },
          { id: 'deadzones', label: 'Dead Zones', icon: (
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
            </svg>
          ) },
          { id: 'dwell', label: 'Dwell Analysis', icon: (
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          ) },
        ].map((view) => (
          <button
            key={view.id}
            onClick={() => setSelectedView(view.id as 'heatmap' | 'deadzones' | 'dwell')}
            className={clsx(
              'flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md transition-all',
              selectedView === view.id
                ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                : 'text-slate-400 hover:text-white'
            )}
          >
            {view.icon}
            {view.label}
          </button>
        ))}
      </div>

      {/* Content Views */}
      {selectedView === 'heatmap' && (
        <Card title={<>WiFi Signal Heatmap <span className="text-slate-500 text-sm font-normal ml-2">Signal strength by area</span></>}>
          {heatmapLoading ? (
            <div className="h-64 flex items-center justify-center text-slate-400">
              Loading heatmap...
            </div>
          ) : heatmap?.grid_cells && heatmap.grid_cells.length > 0 ? (
            <div className="space-y-4">
              {/* Simple grid visualization */}
              <div className="grid grid-cols-10 gap-1 p-4">
                {heatmap.grid_cells.slice(0, 100).map((cell, index) => (
                  <div
                    key={index}
                    className="w-full aspect-square rounded-sm cursor-pointer transition-transform hover:scale-110"
                    style={{ backgroundColor: getSignalColor(cell.signal_strength) }}
                    title={`Signal: ${cell.signal_strength.toFixed(0)} dBm\nReadings: ${cell.reading_count}`}
                  />
                ))}
              </div>
              {/* Legend */}
              <div className="flex items-center justify-center gap-4 text-xs">
                {[
                  { label: 'Excellent', color: '#10b981', range: '> -50' },
                  { label: 'Good', color: '#22c55e', range: '-50 to -60' },
                  { label: 'Fair', color: '#f59e0b', range: '-60 to -70' },
                  { label: 'Poor', color: '#f97316', range: '-70 to -80' },
                  { label: 'Bad', color: '#ef4444', range: '< -80' },
                ].map((item) => (
                  <div key={item.label} className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded" style={{ backgroundColor: item.color }} />
                    <span className="text-slate-400">{item.label}</span>
                    <span className="text-slate-600">({item.range})</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="h-64 flex items-center justify-center text-slate-400">
              No heatmap data available
            </div>
          )}
        </Card>
      )}

      {selectedView === 'deadzones' && (
        <Card title={<>WiFi Dead Zones <span className="text-slate-500 text-sm font-normal ml-2">Areas with poor connectivity</span></>}>
          {deadZonesLoading ? (
            <div className="h-64 flex items-center justify-center text-slate-400">
              Loading dead zones...
            </div>
          ) : deadZones?.dead_zones && deadZones.dead_zones.length > 0 ? (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {deadZones.dead_zones.map((zone) => (
                  <div
                    key={zone.zone_id}
                    className="p-4 rounded-lg bg-red-500/10 border border-red-500/20"
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <div className="text-sm font-medium text-white">{zone.zone_id}</div>
                        <div className="text-xs text-slate-500">
                          Lat: {zone.lat.toFixed(4)}, Long: {zone.long.toFixed(4)}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-lg font-bold font-mono text-red-400">
                          {zone.avg_signal.toFixed(0)} dBm
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-4 text-xs text-slate-400">
                      <span>{zone.affected_devices} devices affected</span>
                      <span>{zone.total_readings} readings</span>
                    </div>
                    {zone.last_detected && (
                      <div className="text-xs text-slate-500 mt-2">
                        Last detected: {new Date(zone.last_detected).toLocaleDateString()}
                      </div>
                    )}
                  </div>
                ))}
              </div>
              {deadZones.recommendations && deadZones.recommendations.length > 0 && (
                <div className="space-y-2">
                  <h4 className="text-sm font-medium text-slate-300">Recommendations</h4>
                  {deadZones.recommendations.map((rec, index) => (
                    <div
                      key={index}
                      className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-sm text-slate-300"
                    >
                      {rec}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="h-64 flex flex-col items-center justify-center text-slate-400">
              <div className="w-16 h-16 mb-4 rounded-full bg-emerald-500/20 flex items-center justify-center">
                <svg className="w-8 h-8 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <p className="text-emerald-400 font-medium">No Dead Zones Detected</p>
              <p className="text-sm mt-1">WiFi coverage is healthy across all areas</p>
            </div>
          )}
        </Card>
      )}

      {selectedView === 'dwell' && (
        <Card title={<>Dwell Time Analysis <span className="text-slate-500 text-sm font-normal ml-2">Where devices spend time</span></>}>
          {dwellLoading ? (
            <div className="h-64 flex items-center justify-center text-slate-400">
              Loading dwell analysis...
            </div>
          ) : dwellTime?.dwell_zones && dwellTime.dwell_zones.length > 0 ? (
            <div className="space-y-4">
              {dwellTime.dwell_zones.map((zone) => (
                <div
                  key={zone.zone_id}
                  className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/30"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <div className="text-sm font-medium text-white">{zone.zone_id}</div>
                      <div className="text-xs text-slate-500">
                        {zone.device_count} devices | {zone.visit_count} visits
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold font-mono text-amber-400">
                        {zone.avg_dwell_minutes.toFixed(0)}
                      </div>
                      <div className="text-xs text-slate-500">avg minutes</div>
                    </div>
                  </div>
                  {zone.peak_hours && zone.peak_hours.length > 0 && (
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-slate-500">Peak hours:</span>
                      <div className="flex gap-1">
                        {zone.peak_hours.map((hour) => (
                          <span
                            key={hour}
                            className="px-2 py-0.5 text-xs rounded bg-amber-500/20 text-amber-400"
                          >
                            {hour}:00
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
              {dwellTime.recommendations && dwellTime.recommendations.length > 0 && (
                <div className="space-y-2">
                  <h4 className="text-sm font-medium text-slate-300">Insights</h4>
                  {dwellTime.recommendations.map((rec, index) => (
                    <div
                      key={index}
                      className="p-3 rounded-lg bg-blue-500/10 border border-blue-500/20 text-sm text-slate-300"
                    >
                      {rec}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="h-64 flex items-center justify-center text-slate-400">
              No dwell data available
            </div>
          )}
        </Card>
      )}

      {/* Coverage Distribution */}
      {coverage?.coverage_distribution && (
        <Card title="Coverage Quality Distribution">
          <div className="grid grid-cols-4 gap-4">
            {Object.entries(coverage.coverage_distribution).map(([quality, percentage]) => (
              <div
                key={quality}
                className="text-center p-4 rounded-lg bg-slate-800/50 border border-slate-700/30"
              >
                <div className="text-2xl font-bold font-mono text-white mb-1">
                  {percentage}%
                </div>
                <div className="text-sm text-slate-400 capitalize">{quality}</div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}

export default LocationIntelligenceTab;
