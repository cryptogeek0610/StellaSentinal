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
}

interface ConnectionCardsProps {
  connections: AllConnectionsStatus | undefined;
  isChecking: boolean;
}

// Service configuration with icons - Infrastructure services first, then data sources
const SERVICES: ServiceConfig[] = [
  // Infrastructure Services
  {
    key: 'backend_db',
    name: 'PostgreSQL',
    description: 'Primary database for anomaly results and application data',
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
      </svg>
    ),
  },
  {
    key: 'redis',
    name: 'Redis',
    description: 'In-memory cache for job queues and real-time data',
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
  },
  {
    key: 'qdrant',
    name: 'Qdrant',
    description: 'Vector database for semantic search and embeddings',
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
      </svg>
    ),
  },
  // Data Source Services
  {
    key: 'dw_sql',
    name: 'XSight DB',
    description: 'SQL Server for device metrics and historical data',
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
      </svg>
    ),
  },
  {
    key: 'mc_sql',
    name: 'MobiControl Database',
    description: 'SQL Server for device management data',
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01" />
      </svg>
    ),
  },
  {
    key: 'mobicontrol_api',
    name: 'MobiControl API',
    description: 'REST API for real-time device information',
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
      </svg>
    ),
  },
  {
    key: 'llm',
    name: 'Stella AI (LLM)',
    description: 'AI model for anomaly explanation and troubleshooting',
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    ),
  },
];

export function ConnectionCards({ connections, isChecking }: ConnectionCardsProps) {
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2" role="list" aria-label="Service connections">
      {SERVICES.map((service, index) => {
        const status = connections?.[service.key];
        const isConnected = status?.connected;

        return (
          <motion.div
            key={service.key}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className={`relative p-5 rounded-xl stellar-card border transition-all overflow-hidden ${
              isChecking
                ? 'border-amber-500/20'
                : isConnected
                  ? 'border-emerald-500/30'
                  : 'border-red-500/30'
            }`}
            role="listitem"
            aria-label={`${service.name}: ${isChecking ? 'Checking' : isConnected ? 'Connected' : 'Offline'}`}
          >
            {/* Shimmer overlay when loading */}
            {isChecking && (
              <div className="absolute inset-0 -translate-x-full animate-shimmer" aria-hidden="true" />
            )}

            <div className={`flex items-start gap-4 ${isChecking ? 'opacity-60' : ''}`}>
              <div
                className={`p-3 rounded-xl transition-colors ${
                  isChecking
                    ? 'bg-slate-700/30 text-slate-400'
                    : isConnected
                      ? 'bg-emerald-500/10 text-emerald-400'
                      : 'bg-red-500/10 text-red-400'
                }`}
                aria-hidden="true"
              >
                {service.icon}
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between mb-1">
                  <h3 className="font-semibold text-white">{service.name}</h3>
                  {isChecking ? (
                    <span
                      className="px-2 py-0.5 text-[10px] font-bold rounded bg-slate-700/50 text-slate-400"
                      role="status"
                    >
                      CHECKING
                    </span>
                  ) : (
                    <span
                      className={`px-2 py-0.5 text-[10px] font-bold rounded flex items-center gap-1 ${
                        isConnected
                          ? 'bg-emerald-500/20 text-emerald-400'
                          : 'bg-red-500/20 text-red-400'
                      }`}
                      role="status"
                    >
                      {/* Accessible status indicator */}
                      <span
                        className={`w-1.5 h-1.5 rounded-full ${
                          isConnected ? 'bg-emerald-400' : 'bg-red-400'
                        }`}
                        aria-hidden="true"
                      />
                      {isConnected ? 'CONNECTED' : 'OFFLINE'}
                    </span>
                  )}
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
  );
}

// Export services for use in other components
export { SERVICES };
export type { ConnectionStatus, ServiceConfig };
