/**
 * Connection Cards Component
 *
 * Display service connection status cards with shimmer loading effect.
 * Extracted from System.tsx for better maintainability.
 */

import { motion } from 'framer-motion';
import type { AllConnectionsStatus, ConnectionStatus } from '../../types/anomaly';

interface ServiceConfig {
  key: keyof Omit<AllConnectionsStatus, 'last_checked'>;
  name: string;
  description: string;
  icon: JSX.Element;
  category: 'core' | 'data' | 'integration' | 'ai';
}

interface ConnectionCardsProps {
  connections: AllConnectionsStatus | undefined;
  isChecking: boolean;
}

// Service configuration with icons - organized by category
const SERVICES: ServiceConfig[] = [
  // Core Infrastructure
  {
    key: 'backend_db',
    name: 'PostgreSQL',
    description: 'Backend database for anomalies and baselines',
    category: 'core',
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
      </svg>
    ),
  },
  {
    key: 'redis',
    name: 'Redis',
    description: 'Real-time streaming and caching',
    category: 'core',
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
  },
  {
    key: 'qdrant',
    name: 'Qdrant',
    description: 'Vector database for semantic search',
    category: 'core',
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
      </svg>
    ),
  },
  // Data Sources
  {
    key: 'dw_sql',
    name: 'XSight Database',
    description: 'SQL Server for device metrics and historical data',
    category: 'data',
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01" />
      </svg>
    ),
  },
  {
    key: 'mc_sql',
    name: 'MobiControl Database',
    description: 'SQL Server for device management data',
    category: 'data',
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 18h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
      </svg>
    ),
  },
  // Integrations
  {
    key: 'mobicontrol_api',
    name: 'MobiControl API',
    description: 'REST API for real-time device information',
    category: 'integration',
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
      </svg>
    ),
  },
  // AI Services
  {
    key: 'llm',
    name: 'Stella AI (LLM)',
    description: 'AI model for anomaly explanation and troubleshooting',
    category: 'ai',
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    ),
  },
];

// Helper to get status colors
function getStatusColors(status: ConnectionStatus | undefined, isChecking: boolean) {
  if (isChecking) {
    return {
      border: 'border-amber-500/20',
      bg: 'bg-slate-700/30',
      text: 'text-slate-400',
      badge: 'bg-slate-700/50 text-slate-400',
      dot: 'bg-slate-400',
    };
  }

  const statusValue = status?.status;

  if (status?.connected) {
    return {
      border: 'border-emerald-500/30',
      bg: 'bg-emerald-500/10',
      text: 'text-emerald-400',
      badge: 'bg-emerald-500/20 text-emerald-400',
      dot: 'bg-emerald-400',
    };
  }

  if (statusValue === 'disabled') {
    return {
      border: 'border-slate-600/30',
      bg: 'bg-slate-700/30',
      text: 'text-slate-500',
      badge: 'bg-slate-700/50 text-slate-500',
      dot: 'bg-slate-500',
    };
  }

  if (statusValue === 'not_configured') {
    return {
      border: 'border-amber-500/30',
      bg: 'bg-amber-500/10',
      text: 'text-amber-400',
      badge: 'bg-amber-500/20 text-amber-400',
      dot: 'bg-amber-400',
    };
  }

  // Error or offline
  return {
    border: 'border-red-500/30',
    bg: 'bg-red-500/10',
    text: 'text-red-400',
    badge: 'bg-red-500/20 text-red-400',
    dot: 'bg-red-400',
  };
}

// Helper to get status label
function getStatusLabel(status: ConnectionStatus | undefined, isChecking: boolean): string {
  if (isChecking) return 'CHECKING';
  if (!status) return 'UNKNOWN';

  switch (status.status) {
    case 'connected':
      return 'CONNECTED';
    case 'disabled':
      return 'DISABLED';
    case 'not_configured':
      return 'NOT CONFIGURED';
    case 'error':
      return 'ERROR';
    case 'disconnected':
    default:
      return 'OFFLINE';
  }
}

export function ConnectionCards({ connections, isChecking }: ConnectionCardsProps) {
  // Count connected services
  const connectedCount = SERVICES.filter(s => connections?.[s.key]?.connected).length;
  const totalCount = SERVICES.length;

  return (
    <div className="space-y-4">
      {/* Summary header */}
      <div className="flex items-center justify-between text-sm">
        <span className="text-slate-500">
          {isChecking ? 'Checking connections...' : `${connectedCount}/${totalCount} services connected`}
        </span>
      </div>

      {/* Service cards grid */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2 xl:grid-cols-3" role="list" aria-label="Service connections">
        {SERVICES.map((service, index) => {
          const status = connections?.[service.key];
          const colors = getStatusColors(status, isChecking);
          const statusLabel = getStatusLabel(status, isChecking);

          return (
            <motion.div
              key={service.key}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
              className={`relative p-5 rounded-xl stellar-card border transition-all overflow-hidden ${colors.border}`}
              role="listitem"
              aria-label={`${service.name}: ${statusLabel}`}
            >
              {/* Shimmer overlay when loading */}
              {isChecking && (
                <div className="absolute inset-0 -translate-x-full animate-shimmer" aria-hidden="true" />
              )}

              <div className={`flex items-start gap-4 ${isChecking ? 'opacity-60' : ''}`}>
                <div
                  className={`p-3 rounded-xl transition-colors ${colors.bg} ${colors.text}`}
                  aria-hidden="true"
                >
                  {service.icon}
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-1">
                    <h3 className="font-semibold text-white">{service.name}</h3>
                    <span
                      className={`px-2 py-0.5 text-[10px] font-bold rounded flex items-center gap-1 ${colors.badge}`}
                      role="status"
                    >
                      <span
                        className={`w-1.5 h-1.5 rounded-full ${colors.dot}`}
                        aria-hidden="true"
                      />
                      {statusLabel}
                    </span>
                  </div>
                  <p className="text-xs text-slate-500 mb-2">{service.description}</p>
                  <p
                    className="text-xs font-mono text-slate-600 truncate"
                    title={status?.server}
                  >
                    {status?.server || 'Not configured'}
                  </p>
                  {status?.error && !isChecking && (
                    <p className="text-xs text-red-400 mt-2 leading-relaxed" role="alert">
                      {status.error}
                    </p>
                  )}
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}

// Export services for use in other components
export { SERVICES };
export type { ConnectionStatus, ServiceConfig };
