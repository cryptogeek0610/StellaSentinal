/**
 * ShiftReadinessSection - Collapsible section for battery/shift readiness
 *
 * Shows fleet battery health, shift readiness percentages, and at-risk devices.
 * Expands to show full device breakdown and recommendations.
 */

import { useState } from 'react';
import clsx from 'clsx';
import { CollapsibleSection } from '../CollapsibleSection';
import { SlideOverPanel } from '../SlideOverPanel';
import type { ShiftReadinessResponse } from '../../../api/client';

interface ShiftReadinessSectionProps {
  data: ShiftReadinessResponse | undefined;
  isLoading?: boolean;
  selectedLocation: string;
  onLocationChange: (location: string) => void;
  locations?: Array<{ id: string; name: string }>;
}

function ProgressBar({ value, max = 100, variant = 'default' }: { value: number; max?: number; variant?: 'default' | 'warning' | 'danger' | 'success' }) {
  const percentage = Math.min(100, (value / max) * 100);
  const variantStyles = {
    default: 'bg-slate-500',
    warning: 'bg-amber-500',
    danger: 'bg-red-500',
    success: 'bg-emerald-500',
  };

  return (
    <div className="h-2 bg-slate-700/50 rounded-full overflow-hidden">
      <div
        className={clsx('h-full transition-all duration-500', variantStyles[variant])}
        style={{ width: `${percentage}%` }}
      />
    </div>
  );
}

function DeviceRow({ device, onClick }: { device: ShiftReadinessResponse['device_details'][0]; onClick: () => void }) {
  const batteryVariant = device.current_battery > 70 ? 'success' : device.current_battery > 30 ? 'warning' : 'danger';

  return (
    <button
      onClick={onClick}
      className="w-full flex items-center gap-3 p-3 rounded-lg bg-slate-800/30 hover:bg-slate-800/50 border border-slate-700/30 hover:border-slate-600/50 transition-all group text-left"
    >
      {/* Battery indicator */}
      <div className="w-10 flex-shrink-0">
        <div className="relative w-6 h-10 border-2 border-slate-500 rounded-sm mx-auto">
          <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-1 bg-slate-500 rounded-t-sm" />
          <div
            className={clsx(
              'absolute bottom-0.5 left-0.5 right-0.5 rounded-sm transition-all',
              batteryVariant === 'success' && 'bg-emerald-500',
              batteryVariant === 'warning' && 'bg-amber-500',
              batteryVariant === 'danger' && 'bg-red-500'
            )}
            style={{ height: `${Math.max(5, (device.current_battery / 100) * 32)}px` }}
          />
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-white font-medium text-sm truncate">
            {device.device_name || `Device #${device.device_id}`}
          </span>
          {!device.will_complete_shift && (
            <span className="px-1.5 py-0.5 rounded text-xs font-medium bg-red-500/20 text-red-400">
              At Risk
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 mt-1">
          <span className="text-xs text-slate-500">
            {device.current_battery}% battery
          </span>
          <span className="text-slate-700">|</span>
          <span className="text-xs text-slate-500">
            -{device.drain_rate_per_hour.toFixed(1)}%/hr
          </span>
        </div>
      </div>

      {/* End projection */}
      <div className="text-right">
        <span className={clsx(
          'text-sm font-mono',
          device.projected_end_battery > 20 ? 'text-emerald-400' : device.projected_end_battery > 0 ? 'text-amber-400' : 'text-red-400'
        )}>
          {device.projected_end_battery}%
        </span>
        <div className="text-xs text-slate-600">end of shift</div>
      </div>

      {/* Arrow */}
      <svg className="w-4 h-4 text-slate-500 group-hover:text-slate-400 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
      </svg>
    </button>
  );
}

function LoadingSkeleton() {
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-4 gap-3">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="p-3 rounded-lg bg-slate-800/30 animate-pulse">
            <div className="h-6 w-16 bg-slate-700/50 rounded mb-2" />
            <div className="h-3 w-12 bg-slate-700/50 rounded" />
          </div>
        ))}
      </div>
    </div>
  );
}

export function ShiftReadinessSection({
  data,
  isLoading,
  selectedLocation,
  onLocationChange,
  locations,
}: ShiftReadinessSectionProps) {
  const [selectedDevice, setSelectedDevice] = useState<ShiftReadinessResponse['device_details'][0] | null>(null);

  const readinessPercent = data?.readiness_percentage || 0;
  const readinessVariant = readinessPercent >= 80 ? 'success' : readinessPercent >= 50 ? 'warning' : 'danger';

  const collapsedContent = isLoading ? (
    <LoadingSkeleton />
  ) : !data ? (
    <div className="flex items-center justify-center py-6 text-slate-500">
      No shift readiness data available
    </div>
  ) : (
    <div className="space-y-3">
      {/* Progress bar */}
      <div>
        <div className="flex justify-between text-xs mb-1">
          <span className="text-slate-400">Shift Readiness</span>
          <span className={clsx(
            readinessVariant === 'success' && 'text-emerald-400',
            readinessVariant === 'warning' && 'text-amber-400',
            readinessVariant === 'danger' && 'text-red-400'
          )}>
            {readinessPercent.toFixed(0)}%
          </span>
        </div>
        <ProgressBar value={readinessPercent} variant={readinessVariant} />
      </div>

      {/* Quick stats */}
      <div className="grid grid-cols-3 gap-2">
        <div className="p-2 rounded bg-slate-800/50 text-center">
          <div className="text-lg font-bold font-mono text-emerald-400">{data.devices_ready}</div>
          <div className="text-xs text-slate-500">Ready</div>
        </div>
        <div className="p-2 rounded bg-slate-800/50 text-center">
          <div className="text-lg font-bold font-mono text-amber-400">{data.devices_at_risk}</div>
          <div className="text-xs text-slate-500">At Risk</div>
        </div>
        <div className="p-2 rounded bg-slate-800/50 text-center">
          <div className="text-lg font-bold font-mono text-red-400">{data.devices_critical}</div>
          <div className="text-xs text-slate-500">Critical</div>
        </div>
      </div>
    </div>
  );

  const expandedContent = (
    <div className="space-y-4">
      {/* Location selector */}
      {locations && locations.length > 0 && (
        <div>
          <label className="text-xs text-slate-500 mb-1 block">Location</label>
          <select
            value={selectedLocation}
            onChange={(e) => onLocationChange(e.target.value)}
            className="w-full px-3 py-2 rounded-lg bg-slate-800/50 border border-slate-700/30 text-white text-sm focus:border-amber-500/50 focus:outline-none"
          >
            {locations.map((loc) => (
              <option key={loc.id} value={loc.id}>{loc.name}</option>
            ))}
          </select>
        </div>
      )}

      {/* Metrics grid */}
      <div className="grid grid-cols-4 gap-3">
        <div className={clsx(
          'p-3 rounded-lg border',
          readinessVariant === 'success' && 'bg-emerald-500/10 border-emerald-500/20',
          readinessVariant === 'warning' && 'bg-amber-500/10 border-amber-500/20',
          readinessVariant === 'danger' && 'bg-red-500/10 border-red-500/20'
        )}>
          <div className={clsx(
            'text-2xl font-bold font-mono',
            readinessVariant === 'success' && 'text-emerald-400',
            readinessVariant === 'warning' && 'text-amber-400',
            readinessVariant === 'danger' && 'text-red-400'
          )}>
            {readinessPercent.toFixed(0)}%
          </div>
          <div className="text-xs text-slate-500">Readiness</div>
        </div>
        <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/30">
          <div className="text-2xl font-bold font-mono text-white">
            {data?.devices_ready || 0}/{data?.total_devices || 0}
          </div>
          <div className="text-xs text-slate-500">Devices Ready</div>
        </div>
        <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20">
          <div className="text-2xl font-bold font-mono text-amber-400">{data?.devices_at_risk || 0}</div>
          <div className="text-xs text-slate-500">At Risk</div>
        </div>
        <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20">
          <div className="text-2xl font-bold font-mono text-red-400">{data?.devices_critical || 0}</div>
          <div className="text-xs text-slate-500">Critical</div>
        </div>
      </div>

      {/* Additional stats */}
      <div className="grid grid-cols-4 gap-2 text-center">
        <div className="p-2 rounded bg-slate-800/30">
          <div className="text-sm font-mono text-white">{data?.avg_battery_at_start.toFixed(0) || 0}%</div>
          <div className="text-xs text-slate-600">Avg Start</div>
        </div>
        <div className="p-2 rounded bg-slate-800/30">
          <div className="text-sm font-mono text-white">-{data?.avg_drain_rate.toFixed(1) || 0}%/hr</div>
          <div className="text-xs text-slate-600">Drain Rate</div>
        </div>
        <div className="p-2 rounded bg-slate-800/30">
          <div className="text-sm font-mono text-amber-400">{data?.devices_not_fully_charged || 0}</div>
          <div className="text-xs text-slate-600">Not Charged</div>
        </div>
        <div className="p-2 rounded bg-slate-800/30">
          <div className={clsx(
            'text-sm font-mono',
            (data?.vs_last_week_readiness || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'
          )}>
            {(data?.vs_last_week_readiness || 0) >= 0 ? '+' : ''}{data?.vs_last_week_readiness?.toFixed(1) || 0}%
          </div>
          <div className="text-xs text-slate-600">vs Last Week</div>
        </div>
      </div>

      {/* Device list */}
      {data?.device_details && data.device_details.length > 0 && (
        <div>
          <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wide mb-2">
            Device Status ({data.device_details.length})
          </h4>
          <div className="space-y-2 max-h-[300px] overflow-y-auto pr-2">
            {data.device_details
              .sort((a, b) => a.readiness_score - b.readiness_score)
              .map((device) => (
                <DeviceRow
                  key={device.device_id}
                  device={device}
                  onClick={() => setSelectedDevice(device)}
                />
              ))}
          </div>
        </div>
      )}

      {/* Recommendations */}
      {data?.recommendations && data.recommendations.length > 0 && (
        <div className="p-3 rounded-lg bg-amber-500/5 border border-amber-500/20">
          <h4 className="text-xs font-medium text-amber-400 uppercase tracking-wide mb-2">
            Recommendations
          </h4>
          <ul className="space-y-1">
            {data.recommendations.map((rec, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-slate-300">
                <span className="text-amber-500 mt-0.5">-</span>
                {rec}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );

  return (
    <>
      <CollapsibleSection
        title="Shift Readiness"
        icon={
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        }
        badge={
          (data?.devices_critical || 0) > 0
            ? { value: data?.devices_critical || 0, variant: 'danger' }
            : undefined
        }
        summaryMetrics={[
          {
            label: 'Ready',
            value: `${readinessPercent.toFixed(0)}%`,
            variant: readinessVariant,
          },
          {
            label: 'At Risk',
            value: data?.devices_at_risk || 0,
            variant: (data?.devices_at_risk || 0) > 0 ? 'warning' : 'default',
          },
        ]}
        collapsedContent={collapsedContent}
        expandedContent={expandedContent}
        accent={readinessVariant === 'danger' ? 'red' : readinessVariant === 'warning' ? 'amber' : 'emerald'}
      />

      {/* Device Detail Panel */}
      <SlideOverPanel
        isOpen={!!selectedDevice}
        onClose={() => setSelectedDevice(null)}
        title={selectedDevice?.device_name || `Device #${selectedDevice?.device_id}`}
        subtitle="Battery & Shift Details"
        width="md"
      >
        {selectedDevice && (
          <div className="space-y-4">
            {/* Battery visual */}
            <div className="flex items-center justify-center py-4">
              <div className="relative w-20 h-32 border-4 border-slate-500 rounded-lg">
                <div className="absolute -top-2 left-1/2 -translate-x-1/2 w-8 h-3 bg-slate-500 rounded-t-lg" />
                <div
                  className={clsx(
                    'absolute bottom-1 left-1 right-1 rounded transition-all',
                    selectedDevice.current_battery > 70 && 'bg-emerald-500',
                    selectedDevice.current_battery > 30 && selectedDevice.current_battery <= 70 && 'bg-amber-500',
                    selectedDevice.current_battery <= 30 && 'bg-red-500'
                  )}
                  style={{ height: `${Math.max(5, (selectedDevice.current_battery / 100) * 112)}px` }}
                />
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-2xl font-bold text-white drop-shadow-lg">
                    {selectedDevice.current_battery}%
                  </span>
                </div>
              </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/30">
                <div className="text-xs text-slate-500 mb-1">Drain Rate</div>
                <div className="text-xl font-bold font-mono text-amber-400">
                  -{selectedDevice.drain_rate_per_hour.toFixed(1)}%/hr
                </div>
              </div>
              <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/30">
                <div className="text-xs text-slate-500 mb-1">End of Shift</div>
                <div className={clsx(
                  'text-xl font-bold font-mono',
                  selectedDevice.projected_end_battery > 20 ? 'text-emerald-400' : selectedDevice.projected_end_battery > 0 ? 'text-amber-400' : 'text-red-400'
                )}>
                  {selectedDevice.projected_end_battery}%
                </div>
              </div>
            </div>

            {/* Status */}
            <div className={clsx(
              'p-3 rounded-lg border',
              selectedDevice.will_complete_shift
                ? 'bg-emerald-500/10 border-emerald-500/20'
                : 'bg-red-500/10 border-red-500/20'
            )}>
              <div className="flex items-center gap-2">
                {selectedDevice.will_complete_shift ? (
                  <>
                    <svg className="w-5 h-5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    <span className="text-emerald-400 font-medium">Will complete shift</span>
                  </>
                ) : (
                  <>
                    <svg className="w-5 h-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                    <span className="text-red-400 font-medium">
                      {selectedDevice.estimated_dead_time
                        ? `Battery depletes at ${selectedDevice.estimated_dead_time}`
                        : 'May not complete shift'}
                    </span>
                  </>
                )}
              </div>
            </div>

            {/* Recommendations */}
            {selectedDevice.recommendations && selectedDevice.recommendations.length > 0 && (
              <div>
                <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wide mb-2">
                  Recommendations
                </h4>
                <ul className="space-y-1">
                  {selectedDevice.recommendations.map((rec, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-slate-300">
                      <span className="text-amber-500 mt-0.5">-</span>
                      {rec}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </SlideOverPanel>
    </>
  );
}

export default ShiftReadinessSection;
