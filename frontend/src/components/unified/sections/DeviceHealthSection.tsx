/**
 * DeviceHealthSection - Collapsible section for device abuse/health analysis
 *
 * Shows drops, reboots, problem device cohorts, and abuse patterns.
 * Expands to show full analysis and worst locations.
 */

import clsx from 'clsx';
import { CollapsibleSection } from '../CollapsibleSection';
import type { DeviceAbuseResponse } from '../../../api/client';

interface DeviceHealthSectionProps {
  data: DeviceAbuseResponse | undefined;
  isLoading?: boolean;
}

function StatCard({ label, value, subValue, variant = 'default' }: {
  label: string;
  value: string | number;
  subValue?: string;
  variant?: 'default' | 'warning' | 'danger' | 'success';
}) {
  const variantStyles = {
    default: 'text-white',
    warning: 'text-amber-400',
    danger: 'text-red-400',
    success: 'text-emerald-400',
  };

  return (
    <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/30">
      <div className="text-xs text-slate-500 mb-1">{label}</div>
      <div className={clsx('text-xl font-bold font-mono', variantStyles[variant])}>
        {value}
      </div>
      {subValue && (
        <div className="text-xs text-slate-600 mt-1">{subValue}</div>
      )}
    </div>
  );
}

function CohortCard({ cohort }: { cohort: DeviceAbuseResponse['problem_combinations'][0] }) {
  const severityStyles = {
    critical: { bg: 'bg-red-500/10', border: 'border-red-500/20', text: 'text-red-400' },
    high: { bg: 'bg-orange-500/10', border: 'border-orange-500/20', text: 'text-orange-400' },
    medium: { bg: 'bg-amber-500/10', border: 'border-amber-500/20', text: 'text-amber-400' },
    low: { bg: 'bg-slate-700/30', border: 'border-slate-600/30', text: 'text-slate-400' },
  };
  const style = severityStyles[cohort.severity as keyof typeof severityStyles] || severityStyles.medium;

  return (
    <div className={clsx('p-3 rounded-lg border', style.bg, style.border)}>
      <div className="flex items-start justify-between mb-2">
        <div>
          <div className="text-sm font-medium text-white">
            {cohort.manufacturer} {cohort.model}
          </div>
          <div className="text-xs text-slate-500">{cohort.os_version}</div>
        </div>
        <span className={clsx('px-2 py-0.5 rounded text-xs font-medium', style.bg, style.text)}>
          {cohort.severity.toUpperCase()}
        </span>
      </div>
      <div className="flex items-center gap-3 text-xs">
        <span className="text-slate-400">
          <span className="font-mono text-white">{cohort.device_count}</span> devices
        </span>
        <span className="text-slate-700">|</span>
        <span className={clsx(style.text)}>
          <span className="font-mono">{cohort.vs_fleet_multiplier.toFixed(1)}x</span> fleet rate
        </span>
        <span className="text-slate-700">|</span>
        <span className="text-slate-400">{cohort.primary_issue}</span>
      </div>
    </div>
  );
}

function LocationBar({ location, maxDrops }: {
  location: DeviceAbuseResponse['worst_locations'][0];
  maxDrops: number;
}) {
  const percentage = (location.drops / maxDrops) * 100;

  return (
    <div className="flex items-center gap-3">
      <div className="w-24 text-xs text-slate-400 truncate" title={location.location_id}>
        {location.location_id}
      </div>
      <div className="flex-1 h-4 bg-slate-700/30 rounded overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-amber-500 to-red-500"
          style={{ width: `${percentage}%` }}
        />
      </div>
      <div className="w-16 text-right">
        <span className="text-sm font-mono text-white">{location.drops}</span>
        <span className="text-xs text-slate-500 ml-1">drops</span>
      </div>
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="grid grid-cols-4 gap-3">
      {[1, 2, 3, 4].map((i) => (
        <div key={i} className="p-3 rounded-lg bg-slate-800/30 animate-pulse">
          <div className="h-3 w-16 bg-slate-700/50 rounded mb-2" />
          <div className="h-6 w-12 bg-slate-700/50 rounded" />
        </div>
      ))}
    </div>
  );
}

export function DeviceHealthSection({ data, isLoading }: DeviceHealthSectionProps) {
  const totalDrops = data?.total_drops || 0;
  const excessiveDrops = data?.devices_with_excessive_drops || 0;
  const totalReboots = data?.total_reboots || 0;
  const excessiveReboots = data?.devices_with_excessive_reboots || 0;

  const hasIssues = excessiveDrops > 0 || excessiveReboots > 0;
  const problemCohorts = data?.problem_combinations || [];
  const worstLocations = data?.worst_locations || [];
  const maxDrops = Math.max(...worstLocations.map(l => l.drops), 1);

  const collapsedContent = isLoading ? (
    <LoadingSkeleton />
  ) : !data ? (
    <div className="flex items-center justify-center py-6 text-slate-500">
      No device health data available
    </div>
  ) : (
    <div className="grid grid-cols-4 gap-3">
      <StatCard
        label="Total Drops"
        value={totalDrops}
        variant={totalDrops > 100 ? 'warning' : 'default'}
      />
      <StatCard
        label="Excessive Drops"
        value={excessiveDrops}
        variant={excessiveDrops > 0 ? 'danger' : 'default'}
        subValue="devices"
      />
      <StatCard
        label="Total Reboots"
        value={totalReboots}
        variant={totalReboots > 50 ? 'warning' : 'default'}
      />
      <StatCard
        label="Excessive Reboots"
        value={excessiveReboots}
        variant={excessiveReboots > 0 ? 'danger' : 'default'}
        subValue="devices"
      />
    </div>
  );

  const expandedContent = (
    <div className="space-y-4">
      {/* Stats Grid */}
      <div className="grid grid-cols-4 gap-3">
        <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/30">
          <div className="text-xs text-slate-500 mb-1">Total Devices</div>
          <div className="text-2xl font-bold font-mono text-white">{data?.total_devices || 0}</div>
        </div>
        <div className={clsx(
          'p-3 rounded-lg border',
          totalDrops > 100 ? 'bg-amber-500/10 border-amber-500/20' : 'bg-slate-800/50 border-slate-700/30'
        )}>
          <div className="text-xs text-slate-500 mb-1">Total Drops</div>
          <div className={clsx('text-2xl font-bold font-mono', totalDrops > 100 ? 'text-amber-400' : 'text-white')}>
            {totalDrops}
          </div>
        </div>
        <div className={clsx(
          'p-3 rounded-lg border',
          excessiveDrops > 0 ? 'bg-red-500/10 border-red-500/20' : 'bg-slate-800/50 border-slate-700/30'
        )}>
          <div className="text-xs text-slate-500 mb-1">Excessive Drops</div>
          <div className={clsx('text-2xl font-bold font-mono', excessiveDrops > 0 ? 'text-red-400' : 'text-white')}>
            {excessiveDrops}
          </div>
          <div className="text-xs text-slate-600">devices</div>
        </div>
        <div className={clsx(
          'p-3 rounded-lg border',
          excessiveReboots > 0 ? 'bg-red-500/10 border-red-500/20' : 'bg-slate-800/50 border-slate-700/30'
        )}>
          <div className="text-xs text-slate-500 mb-1">Excessive Reboots</div>
          <div className={clsx('text-2xl font-bold font-mono', excessiveReboots > 0 ? 'text-red-400' : 'text-white')}>
            {excessiveReboots}
          </div>
          <div className="text-xs text-slate-600">devices</div>
        </div>
      </div>

      {/* Two column layout */}
      <div className="grid grid-cols-2 gap-4">
        {/* Worst Locations */}
        <div>
          <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wide mb-2">
            Worst Locations (Drops)
          </h4>
          {worstLocations.length === 0 ? (
            <div className="p-4 rounded-lg bg-slate-800/30 text-center text-slate-500 text-sm">
              No location data
            </div>
          ) : (
            <div className="space-y-2">
              {worstLocations.slice(0, 5).map((location, i) => (
                <LocationBar key={i} location={location} maxDrops={maxDrops} />
              ))}
            </div>
          )}
        </div>

        {/* Problem Cohorts */}
        <div>
          <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wide mb-2">
            Problem Device Cohorts
          </h4>
          {problemCohorts.length === 0 ? (
            <div className="p-4 rounded-lg bg-slate-800/30 text-center text-slate-500 text-sm">
              No problem cohorts detected
            </div>
          ) : (
            <div className="space-y-2 max-h-[200px] overflow-y-auto pr-1">
              {problemCohorts.map((cohort, i) => (
                <CohortCard key={i} cohort={cohort} />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Summary alert if issues exist */}
      {hasIssues && (
        <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20">
          <div className="flex items-center gap-2 mb-2">
            <svg className="w-5 h-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <span className="text-red-400 font-medium">
              {excessiveDrops + excessiveReboots} device{excessiveDrops + excessiveReboots > 1 ? 's' : ''} showing abuse patterns
            </span>
          </div>
          <p className="text-sm text-slate-400">
            Devices with excessive drops or reboots may indicate hardware issues, user mishandling, or environmental factors.
          </p>
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
    <CollapsibleSection
      title="Device Health"
      icon={
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 18h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
        </svg>
      }
      badge={
        hasIssues
          ? { value: excessiveDrops + excessiveReboots, variant: 'danger' }
          : undefined
      }
      summaryMetrics={[
        { label: 'Drops', value: totalDrops, variant: excessiveDrops > 0 ? 'warning' : 'default' },
        { label: 'At Risk', value: excessiveDrops + excessiveReboots, variant: hasIssues ? 'danger' : 'default' },
      ]}
      collapsedContent={collapsedContent}
      expandedContent={expandedContent}
      defaultExpanded={true}
      accent={hasIssues ? 'red' : 'slate'}
    />
  );
}

export default DeviceHealthSection;
