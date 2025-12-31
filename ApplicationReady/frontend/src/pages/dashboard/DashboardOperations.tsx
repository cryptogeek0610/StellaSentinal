/**
 * Dashboard Operations View
 *
 * Location heatmap and system telemetry for operations monitoring.
 * Designed with balanced layout for data-rich and empty states.
 */

import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { formatDistanceToNowStrict } from 'date-fns';
import { Card } from '../../components/Card';
import { LocationHeatmap } from '../../components/LocationHeatmap';
import { InfoTooltip } from '../../components/ui/InfoTooltip';
import type { AllConnectionsStatus, ConnectionStatus, DashboardStats, LocationHeatmapResponse } from '../../types/anomaly';

// System configuration for telemetry display
const SYSTEM_CONFIG = [
  {
    name: 'XSight Database',
    key: 'dw_sql' as const,
    description: 'Historical telemetry storage',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
      </svg>
    ),
  },
  {
    name: 'MobiControl DB',
    key: 'mc_sql' as const,
    description: 'Device management data',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01" />
      </svg>
    ),
  },
  {
    name: 'MobiControl API',
    key: 'mobicontrol_api' as const,
    description: 'Real-time device communication',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
      </svg>
    ),
  },
  {
    name: 'Stella AI (LLM)',
    key: 'llm' as const,
    description: 'AI-powered analysis engine',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    ),
  },
];

interface DashboardOperationsProps {
  locationHeatmap: LocationHeatmapResponse | undefined;
  attributeName: string;
  connections: AllConnectionsStatus | undefined;
  stats: DashboardStats | undefined;
  systemHealthPercentage: number;
}

export function DashboardOperations({
  locationHeatmap,
  attributeName,
  connections,
  stats,
  systemHealthPercentage,
}: DashboardOperationsProps) {
  const navigate = useNavigate();

  // Helper to get connection status by key
  const getConnectionStatus = (key: keyof Omit<AllConnectionsStatus, 'last_checked'>): ConnectionStatus | undefined => {
    return connections?.[key];
  };

  const hasLocationData = locationHeatmap && locationHeatmap.locations.length > 0;
  const connectedCount = connections
    ? [connections.dw_sql, connections.mc_sql, connections.mobicontrol_api, connections.llm].filter(s => s.connected).length
    : 0;

  return (
    <motion.div
      key="operations"
      className="space-y-6"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
    >
      {/* Top Row: Stats Overview */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatCard
          label="Devices Monitored"
          value={stats?.devices_monitored || 0}
          explainer="Total number of devices actively monitored for anomaly detection across all locations"
          icon={
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 18h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
            </svg>
          }
          color="amber"
        />
        <StatCard
          label="System Health"
          value={`${systemHealthPercentage}%`}
          explainer="Overall health score based on connection status of all critical backend services"
          icon={
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          }
          color={systemHealthPercentage === 100 ? 'emerald' : systemHealthPercentage >= 50 ? 'amber' : 'red'}
        />
        <StatCard
          label="Services Online"
          value={`${connectedCount}/4`}
          explainer="Number of connected backend services (DW SQL, MC SQL, API, LLM) out of total required"
          icon={
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2" />
            </svg>
          }
          color={connectedCount === 4 ? 'emerald' : connectedCount >= 2 ? 'amber' : 'red'}
        />
        <StatCard
          label="Last Sync"
          value={
            connections?.last_checked
              ? formatDistanceToNowStrict(new Date(connections.last_checked), { addSuffix: false })
              : '—'
          }
          suffix={connections?.last_checked ? ' ago' : ''}
          explainer="Time since last successful data synchronization with backend services"
          icon={
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          }
          color="slate"
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
        {/* Location Heatmap or Empty State */}
        <div className="xl:col-span-2">
          {hasLocationData ? (
            <LocationHeatmap
              locations={locationHeatmap.locations}
              attributeName={locationHeatmap.attributeName}
              onLocationClick={(location) => {
                navigate(`/fleet?${attributeName.toLowerCase()}=${location.name}`);
              }}
            />
          ) : (
            <Card
              title={
                <span className="telemetry-label">
                  {attributeName} Performance Map
                </span>
              }
            >
              <div className="p-8 text-center">
                <div className="w-14 h-14 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-slate-700/50 to-slate-800/50 flex items-center justify-center border border-slate-700/50" aria-hidden="true">
                  <svg className="w-7 h-7 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                  </svg>
                </div>
                <p className="text-slate-300 font-medium mb-2">No Location Data Available</p>
                <p className="text-sm text-slate-500 max-w-md mx-auto mb-4">
                  Configure the custom attribute name in Settings and ensure devices have this attribute set to see location-based anomaly distribution.
                </p>
                <Link
                  to="/system"
                  className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded-lg hover:bg-amber-500/20 transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                  Configure Settings
                </Link>
              </div>
            </Card>
          )}
        </div>

        {/* System Status */}
        <div>
          <Card
            title={
              <div className="flex items-center justify-between w-full">
                <span className="telemetry-label">System Telemetry</span>
                <Link
                  to="/system"
                  className="text-xs text-amber-400 hover:text-amber-300 font-medium transition-colors"
                >
                  Configure →
                </Link>
              </div>
            }
          >
            <div className="space-y-3" role="list" aria-label="System connections">
              {SYSTEM_CONFIG.map((system) => {
                const status = getConnectionStatus(system.key);
                const isConnected = status?.connected;

                return (
                  <div
                    key={system.key}
                    className={`p-3 rounded-xl border transition-all ${isConnected
                        ? 'border-emerald-500/20 bg-emerald-500/5'
                        : 'border-red-500/20 bg-red-500/5'
                      }`}
                    role="listitem"
                    aria-label={`${system.name}: ${isConnected ? 'Connected' : 'Offline'}`}
                  >
                    <div className="flex items-center gap-3">
                      <div
                        className={`w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0 ${isConnected ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'
                          }`}
                        aria-hidden="true"
                      >
                        {system.icon}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between">
                          <p className="text-sm font-medium text-slate-200 truncate">{system.name}</p>
                          <span
                            className={`flex items-center gap-1.5 text-xs font-medium ${isConnected ? 'text-emerald-400' : 'text-red-400'
                              }`}
                          >
                            <span
                              className={`w-1.5 h-1.5 rounded-full ${isConnected ? 'bg-emerald-400' : 'bg-red-400 animate-pulse'
                                }`}
                              aria-hidden="true"
                            />
                            {isConnected ? 'Online' : 'Offline'}
                          </span>
                        </div>
                        <p className="text-xs text-slate-500 truncate">{system.description}</p>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </Card>
        </div>
      </div>
    </motion.div>
  );
}

// Stat Card Component
interface StatCardProps {
  label: string;
  value: string | number;
  suffix?: string;
  icon: React.ReactNode;
  color: 'amber' | 'emerald' | 'red' | 'slate' | 'indigo';
  explainer?: string;
}

function StatCard({ label, value, suffix = '', icon, color, explainer }: StatCardProps) {
  const colorClasses = {
    amber: 'border-amber-500/20 bg-amber-500/5',
    emerald: 'border-emerald-500/20 bg-emerald-500/5',
    red: 'border-red-500/20 bg-red-500/5',
    slate: 'border-slate-700/50 bg-slate-800/30',
    indigo: 'border-indigo-500/20 bg-indigo-500/5',
  };

  const iconColorClasses = {
    amber: 'text-amber-400 bg-amber-500/20',
    emerald: 'text-emerald-400 bg-emerald-500/20',
    red: 'text-red-400 bg-red-500/20',
    slate: 'text-slate-400 bg-slate-700/50',
    indigo: 'text-indigo-400 bg-indigo-500/20',
  };

  const valueColorClasses = {
    amber: 'text-amber-400',
    emerald: 'text-emerald-400',
    red: 'text-red-400',
    slate: 'text-slate-300',
    indigo: 'text-indigo-400',
  };

  return (
    <div className={`stellar-card rounded-xl p-4 border ${colorClasses[color]}`}>
      <div className="flex items-start gap-3">
        <div className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${iconColorClasses[color]}`} aria-hidden="true">
          {icon}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5 mb-1">
            <p className="text-xs text-slate-500">{label}</p>
            {explainer && <InfoTooltip content={explainer} />}
          </div>
          <p className={`text-xl font-bold font-mono ${valueColorClasses[color]}`}>
            {value}
            {suffix && <span className="text-sm font-normal text-slate-500">{suffix}</span>}
          </p>
        </div>
      </div>
    </div>
  );
}
