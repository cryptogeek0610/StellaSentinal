/**
 * NetworkSection - Collapsible section for network health analysis
 *
 * Shows WiFi roaming issues, dead zones, disconnects, and connectivity patterns.
 * Expands to show full network analysis and recommendations.
 */

import clsx from 'clsx';
import { CollapsibleSection } from '../CollapsibleSection';
import type { NetworkAnalysisResponse } from '../../../api/client';

interface NetworkSectionProps {
  data: NetworkAnalysisResponse | undefined;
  isLoading?: boolean;
}

function StatCard({ label, value, variant = 'default', icon }: {
  label: string;
  value: string | number;
  variant?: 'default' | 'warning' | 'danger' | 'success';
  icon?: React.ReactNode;
}) {
  const variantStyles = {
    default: 'text-white',
    warning: 'text-amber-400',
    danger: 'text-red-400',
    success: 'text-emerald-400',
  };

  return (
    <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/30">
      <div className="flex items-center gap-2 mb-1">
        {icon && <span className="text-slate-400">{icon}</span>}
        <span className="text-xs text-slate-500">{label}</span>
      </div>
      <div className={clsx('text-xl font-bold font-mono', variantStyles[variant])}>
        {value}
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

export function NetworkSection({ data, isLoading }: NetworkSectionProps) {
  const roamingIssues = data?.wifi_summary?.devices_with_roaming_issues || 0;
  const deadZones = data?.wifi_summary?.potential_dead_zones || 0;
  const totalDisconnects = data?.disconnect_summary?.total_disconnects || 0;
  const hiddenDevices = data?.hidden_devices_count || 0;

  const hasIssues = roamingIssues > 0 || deadZones > 0 || hiddenDevices > 0;

  const collapsedContent = isLoading ? (
    <LoadingSkeleton />
  ) : !data ? (
    <div className="flex items-center justify-center py-6 text-slate-500">
      No network data available
    </div>
  ) : (
    <div className="grid grid-cols-4 gap-3">
      <StatCard
        label="Roaming Issues"
        value={roamingIssues}
        variant={roamingIssues > 0 ? 'warning' : 'default'}
        icon={
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8.111 16.404a5.5 5.5 0 017.778 0M12 20h.01m-7.08-7.071c3.904-3.905 10.236-3.905 14.141 0M1.394 9.393c5.857-5.857 15.355-5.857 21.213 0" />
          </svg>
        }
      />
      <StatCard
        label="Dead Zones"
        value={deadZones}
        variant={deadZones > 0 ? 'danger' : 'default'}
        icon={
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M18.364 5.636a9 9 0 010 12.728m0 0l-2.829-2.829m2.829 2.829L21 21M15.536 8.464a5 5 0 010 7.072m0 0l-2.829-2.829m-4.243 2.829a4.978 4.978 0 01-1.414-2.83m-1.414 5.658a9 9 0 01-2.167-9.238m7.824 2.167a1 1 0 111.414 1.414m-1.414-1.414L3 3m8.293 8.293l1.414 1.414" />
          </svg>
        }
      />
      <StatCard
        label="Disconnects"
        value={totalDisconnects}
        variant={totalDisconnects > 100 ? 'warning' : 'default'}
        icon={
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
          </svg>
        }
      />
      <StatCard
        label="Hidden"
        value={hiddenDevices}
        variant={hiddenDevices > 0 ? 'warning' : 'default'}
        icon={
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        }
      />
    </div>
  );

  const expandedContent = (
    <div className="space-y-4">
      {/* WiFi Summary */}
      <div>
        <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wide mb-2">
          WiFi Analysis
        </h4>
        <div className="grid grid-cols-4 gap-3">
          <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/30">
            <div className="text-xs text-slate-500 mb-1">Total Devices</div>
            <div className="text-xl font-bold font-mono text-white">
              {data?.wifi_summary?.total_devices || 0}
            </div>
          </div>
          <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/30">
            <div className="text-xs text-slate-500 mb-1">Roaming Issues</div>
            <div className={clsx(
              'text-xl font-bold font-mono',
              roamingIssues > 0 ? 'text-amber-400' : 'text-white'
            )}>
              {roamingIssues}
            </div>
          </div>
          <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/30">
            <div className="text-xs text-slate-500 mb-1">AP Stickiness</div>
            <div className={clsx(
              'text-xl font-bold font-mono',
              (data?.wifi_summary?.devices_with_stickiness || 0) > 0 ? 'text-amber-400' : 'text-white'
            )}>
              {data?.wifi_summary?.devices_with_stickiness || 0}
            </div>
          </div>
          <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/30">
            <div className="text-xs text-slate-500 mb-1">Avg APs/Device</div>
            <div className="text-xl font-bold font-mono text-white">
              {data?.wifi_summary?.avg_aps_per_device?.toFixed(1) || 0}
            </div>
          </div>
        </div>
      </div>

      {/* Disconnect Analysis */}
      <div>
        <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wide mb-2">
          Disconnect Patterns
        </h4>
        <div className="grid grid-cols-2 gap-3">
          <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/30">
            <div className="text-xs text-slate-500 mb-1">Total Disconnects</div>
            <div className="text-xl font-bold font-mono text-white">
              {totalDisconnects}
            </div>
            <div className="text-xs text-slate-600 mt-1">
              {data?.disconnect_summary?.avg_disconnects_per_device?.toFixed(1) || 0} avg per device
            </div>
          </div>
          <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/30">
            <div className="text-xs text-slate-500 mb-1">Total Offline Hours</div>
            <div className="text-xl font-bold font-mono text-white">
              {data?.disconnect_summary?.total_offline_hours?.toFixed(1) || 0}
            </div>
          </div>
        </div>

        {/* Pattern description */}
        {data?.disconnect_summary?.pattern_description && (
          <div className="mt-2 p-3 rounded-lg bg-slate-800/30 border border-slate-700/30">
            <div className="flex items-start gap-2">
              <svg className="w-4 h-4 text-slate-400 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="text-sm text-slate-400">
                {data.disconnect_summary.pattern_description}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Dead Zones */}
      {deadZones > 0 && (
        <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20">
          <div className="flex items-center gap-2 mb-2">
            <svg className="w-5 h-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <span className="text-red-400 font-medium">
              {deadZones} potential dead zone{deadZones > 1 ? 's' : ''} detected
            </span>
          </div>
          <p className="text-sm text-slate-400">
            Areas where devices consistently lose connectivity. Consider infrastructure review.
          </p>
        </div>
      )}

      {/* Hidden Devices */}
      {hiddenDevices > 0 && (
        <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20">
          <div className="flex items-center gap-2 mb-2">
            <svg className="w-5 h-5 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span className="text-amber-400 font-medium">
              {hiddenDevices} device{hiddenDevices > 1 ? 's' : ''} with hidden patterns
            </span>
          </div>
          <p className="text-sm text-slate-400">
            Devices exhibiting unusual or obscured connectivity behavior requiring investigation.
          </p>
        </div>
      )}

      {/* Recommendations */}
      {data?.recommendations && data.recommendations.length > 0 && (
        <div className="p-3 rounded-lg bg-purple-500/5 border border-purple-500/20">
          <h4 className="text-xs font-medium text-purple-400 uppercase tracking-wide mb-2">
            Recommendations
          </h4>
          <ul className="space-y-1">
            {data.recommendations.map((rec, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-slate-300">
                <span className="text-purple-500 mt-0.5">-</span>
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
      title="Network Health"
      icon={
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8.111 16.404a5.5 5.5 0 017.778 0M12 20h.01m-7.08-7.071c3.904-3.905 10.236-3.905 14.141 0M1.394 9.393c5.857-5.857 15.355-5.857 21.213 0" />
        </svg>
      }
      badge={
        hasIssues
          ? { value: roamingIssues + deadZones, variant: deadZones > 0 ? 'danger' : 'warning' }
          : undefined
      }
      summaryMetrics={[
        { label: 'Roaming', value: roamingIssues, variant: roamingIssues > 0 ? 'warning' : 'default' },
        { label: 'Dead Zones', value: deadZones, variant: deadZones > 0 ? 'danger' : 'default' },
      ]}
      collapsedContent={collapsedContent}
      expandedContent={expandedContent}
      defaultExpanded={true}
      accent={deadZones > 0 ? 'red' : hasIssues ? 'purple' : 'slate'}
    />
  );
}

export default NetworkSection;
