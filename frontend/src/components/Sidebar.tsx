/**
 * Stellar Sidebar - Mission Control Navigation
 * 
 * Premium space agency inspired navigation with
 * AI status indicators and system health monitoring
 */

import { Link, useLocation } from 'react-router-dom';
import { clsx } from 'clsx';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import { api } from '../api/client';
import { LLMSettingsModal } from './LLMSettingsModal';

const navSections = [
  {
    title: 'MONITOR',
    items: [
      {
        name: 'Dashboard',
        path: '/dashboard',
        description: 'Operations & intelligence',
        icon: (
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
            />
          </svg>
        ),
        badge: null,
      },
      {
        name: 'Investigations',
        path: '/investigations',
        description: 'Active anomalies',
        icon: (
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
        ),
        badge: 'anomalies',
      },
      {
        name: 'Fleet',
        path: '/fleet',
        description: 'Device health',
        icon: (
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M12 18h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z"
            />
          </svg>
        ),
        badge: null,
      },
      {
        name: 'AI Insights',
        path: '/insights',
        description: 'Deep intelligence analysis',
        icon: (
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
            />
          </svg>
        ),
        badge: null,
      },
      {
        name: 'Locations',
        path: '/locations',
        description: 'Geographic analytics',
        icon: (
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"
            />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"
            />
          </svg>
        ),
        badge: null,
      },
    ],
  },
  {
    title: 'ANALYZE',
    items: [
      {
        name: 'Data Overview',
        path: '/data',
        description: 'ML training data',
        icon: (
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4"
            />
          </svg>
        ),
        badge: null,
      },
      {
        name: 'Training',
        path: '/training',
        description: 'Model training & metrics',
        icon: (
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
            />
          </svg>
        ),
        badge: null,
      },
      {
        name: 'Baselines',
        path: '/baselines',
        description: 'Drift detection & tuning',
        icon: (
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
            />
          </svg>
        ),
        badge: 'baselines',
      },
      {
        name: 'Automation',
        path: '/automation',
        description: 'ML scheduling & scoring',
        icon: (
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        ),
        badge: null,
      },
    ],
  },
  {
    title: 'CONFIGURE',
    items: [
      {
        name: 'Cost Intelligence',
        path: '/costs',
        description: 'Hardware & operational costs',
        icon: (
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
        ),
        badge: null,
      },
      {
        name: 'System',
        path: '/system',
        description: 'Connections & settings',
        icon: (
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
            />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        ),
        badge: 'health',
      },
      {
        name: 'Model Status',
        path: '/status',
        description: 'Pipeline architecture',
        icon: (
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
          </svg>
        ),
        badge: null,
      },
      {
        name: 'Setup Wizard',
        path: '/setup',
        description: 'Configure environment',
        icon: (
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"
            />
          </svg>
        ),
        badge: null,
      },
    ],
  },
];

interface SidebarProps {
  onClose?: () => void;
}

export const Sidebar = ({ onClose }: SidebarProps) => {
  const location = useLocation();
  const [showLLMModal, setShowLLMModal] = useState(false);

  const { data: stats } = useQuery({
    queryKey: ['dashboard', 'stats'],
    queryFn: () => api.getDashboardStats(),
    refetchInterval: 30000,
  });

  const { data: connections } = useQuery({
    queryKey: ['dashboard', 'connections'],
    queryFn: () => api.getConnectionStatus(),
    refetchInterval: 30000,
  });

  const { data: baselineSuggestions } = useQuery({
    queryKey: ['baselines', 'suggestions', 30],
    queryFn: () => api.getBaselineSuggestions(undefined, 30),
    refetchInterval: 60000,
  });

  const openAnomalies = stats?.open_cases ?? 0;
  const criticalCount = stats?.critical_issues || 0;
  const baselineDriftCount = baselineSuggestions?.length ?? 0;

  const connectedCount = connections
    ? [connections.backend_db, connections.redis, connections.qdrant, connections.dw_sql, connections.mc_sql, connections.mobicontrol_api, connections.llm].filter(s => s?.connected).length
    : 0;
  const healthStatus = connectedCount === 7 ? 'healthy' : connectedCount >= 4 ? 'degraded' : 'critical';
  const llmConnected = connections?.llm?.connected ?? false;

  const getBadgeContent = (badgeType: string | null) => {
    if (!badgeType) return null;
    if (badgeType === 'anomalies' && openAnomalies > 0) {
      return {
        count: openAnomalies,
        critical: criticalCount > 0,
      };
    }
    if (badgeType === 'health') {
      return {
        status: healthStatus,
      };
    }
    if (badgeType === 'baselines' && baselineDriftCount > 0) {
      return {
        count: baselineDriftCount,
        drift: true,
      };
    }
    return null;
  };

  return (
    <div className="flex flex-col w-72 stellar-glass h-full border-r border-slate-700/30">
      {/* Logo */}
      <div className="flex items-center justify-between h-20 px-6 border-b border-slate-700/30">
        <Link to="/" className="flex items-center gap-4 group" onClick={onClose}>
          <div className="relative">
            {/* Orbital ring effect */}
            <div className="absolute inset-0 rounded-2xl opacity-50 blur-sm animate-pulse"
              style={{
                background: 'linear-gradient(135deg, rgba(245, 166, 35, 0.4), rgba(255, 199, 95, 0.2))'
              }}
            />
            <div className="relative w-12 h-12 rounded-2xl bg-gradient-to-br from-amber-500 via-amber-400 to-orange-500 flex items-center justify-center shadow-stellar">
              <svg className="w-7 h-7 text-slate-900" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
                />
              </svg>
            </div>
            {/* Online status beacon */}
            <motion.div
              className="absolute -top-0.5 -right-0.5 w-3.5 h-3.5 bg-emerald-400 rounded-full border-2 border-slate-900 shadow-[0_0_10px_rgba(16,185,129,0.5)]"
              animate={{ scale: [1, 1.15, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white tracking-tight">Stella Sentinel</h1>
            <p className="text-[10px] font-mono text-amber-400/80 uppercase tracking-[0.2em]">
              SOTI INTELLIGENCE
            </p>
          </div>
        </Link>

        {/* Mobile Close Button */}
        {onClose && (
          <button
            onClick={onClose}
            className="lg:hidden p-2 -mr-2 text-slate-400 hover:text-white hover:bg-slate-800/50 rounded-lg transition-colors"
            aria-label="Close navigation menu"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>

      {/* Alert Banner */}
      <AnimatePresence>
        {criticalCount > 0 && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="mx-4 mt-4"
          >
            <Link
              to="/investigations?severity=critical"
              className="block p-3 bg-red-500/10 border border-red-500/30 rounded-xl hover:bg-red-500/20 transition-colors shadow-danger"
            >
              <div className="flex items-center gap-3">
                <motion.div
                  className="w-3 h-3 bg-red-400 rounded-full shadow-[0_0_10px_rgba(239,68,68,0.6)]"
                  animate={{ opacity: [1, 0.4, 1] }}
                  transition={{ duration: 1, repeat: Infinity }}
                />
                <div className="flex-1">
                  <p className="text-sm font-semibold text-red-400">
                    {criticalCount} Critical {criticalCount === 1 ? 'Alert' : 'Alerts'}
                  </p>
                  <p className="text-[10px] text-slate-500">Requires immediate attention</p>
                </div>
                <svg className="w-4 h-4 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </div>
            </Link>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Navigation */}
      <nav className="flex-1 px-4 py-6 space-y-6 overflow-y-auto">
        {navSections.map((section, sectionIndex) => (
          <div key={section.title}>
            <p className="px-4 mb-3 text-[10px] font-bold text-slate-600 uppercase tracking-[0.2em]">
              {section.title}
            </p>
            <div className="space-y-1">
              {section.items.map((item, index) => {
                const isActive =
                  location.pathname.startsWith(item.path) ||
                  (item.path === '/dashboard' && location.pathname === '/');
                const badge = getBadgeContent(item.badge);

                return (
                  <motion.div
                    key={item.name}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: (sectionIndex * 3 + index) * 0.05 }}
                  >
                    <Link
                      to={item.path}
                      className={clsx(
                        'nav-item group relative flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200',
                        isActive
                          ? 'bg-gradient-to-r from-amber-500/15 via-amber-500/10 to-transparent text-white'
                          : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
                      )}
                    >
                      {/* Active indicator */}
                      {isActive && (
                        <motion.div
                          layoutId="activeNavIndicator"
                          className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-gradient-to-b from-amber-400 to-amber-500 rounded-r-full shadow-stellar"
                          transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                        />
                      )}

                      <span className={clsx(
                        'transition-colors duration-200',
                        isActive ? 'text-amber-400' : 'text-slate-500 group-hover:text-slate-300'
                      )}>
                        {item.icon}
                      </span>

                      <div className="flex-1 min-w-0">
                        <span className={clsx(
                          'block text-sm font-medium',
                          isActive ? 'text-white' : ''
                        )}>
                          {item.name}
                        </span>
                        <span className="block text-[10px] text-slate-600 truncate">
                          {item.description}
                        </span>
                      </div>

                      {/* Badge */}
                      {badge && 'count' in badge && (badge.count ?? 0) > 0 && (
                        <motion.span
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          className={clsx(
                            'min-w-[24px] h-6 flex items-center justify-center px-2 rounded-full text-xs font-bold',
                            (badge as { count: number; critical?: boolean; drift?: boolean }).critical
                              ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                              : (badge as { count: number; drift?: boolean }).drift
                                ? 'bg-purple-500/20 text-purple-400 border border-purple-500/30'
                                : 'bg-orange-500/20 text-orange-400 border border-orange-500/30'
                          )}
                        >
                          {(badge as { count: number }).count}
                        </motion.span>
                      )}

                      {badge && 'status' in badge && (
                        <div className={clsx(
                          'w-2.5 h-2.5 rounded-full',
                          badge.status === 'healthy' && 'bg-emerald-400 shadow-[0_0_8px_rgba(16,185,129,0.6)]',
                          badge.status === 'degraded' && 'bg-orange-400 shadow-[0_0_8px_rgba(251,146,60,0.6)]',
                          badge.status === 'critical' && 'bg-red-400 shadow-[0_0_8px_rgba(248,113,113,0.6)] animate-pulse'
                        )} />
                      )}
                    </Link>
                  </motion.div>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      {/* Footer - AI Status & Stats */}
      <div className="p-4 border-t border-slate-700/30 space-y-3">
        {/* AI Status Card */}
        <button
          type="button"
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            setShowLLMModal(true);
          }}
          className="stellar-card rounded-xl p-3 w-full text-left hover:bg-slate-800/50 transition-colors cursor-pointer active:scale-[0.98]"
          aria-label="Open LLM configuration"
        >
          <div className="flex items-center gap-3">
            <div className={clsx(
              'w-9 h-9 rounded-lg flex items-center justify-center transition-all overflow-hidden',
              llmConnected
                ? 'bg-gradient-to-br from-indigo-600/20 to-purple-600/20 shadow-[0_0_15px_rgba(99,102,241,0.3)] border border-indigo-500/30'
                : 'bg-slate-700/50 border border-slate-600/30'
            )}>
              <img
                src="/assets/stella-ai-logo.png"
                alt="Stella AI"
                className={clsx('w-6 h-6 object-contain', !llmConnected && 'opacity-50 grayscale')}
              />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold text-white">Stella AI</span>
                <span className={clsx(
                  'status-dot w-1.5 h-1.5 rounded-full',
                  llmConnected ? 'bg-emerald-400 shadow-[0_0_6px_rgba(16,185,129,0.6)]' : 'bg-slate-500'
                )} />
              </div>
              <p className="text-[10px] text-slate-500 truncate font-mono">
                {llmConnected ? 'LLM Connected' : 'Not Connected'}
              </p>
            </div>
            {llmConnected && (
              <motion.div
                animate={{ opacity: [0.5, 1, 0.5] }}
                transition={{ duration: 2, repeat: Infinity }}
                className="text-[10px] font-semibold text-indigo-400 pointer-events-none"
              >
                ACTIVE
              </motion.div>
            )}
          </div>
        </button>

        {/* Quick Stats */}
        <div className="stellar-card rounded-xl p-4">
          <div className="flex items-center justify-between mb-3">
            <span className="telemetry-label">Today's Activity</span>
            <span className={clsx(
              'w-2 h-2 rounded-full',
              healthStatus === 'healthy' && 'bg-emerald-400 shadow-[0_0_6px_rgba(16,185,129,0.6)]',
              healthStatus === 'degraded' && 'bg-orange-400',
              healthStatus === 'critical' && 'bg-red-400 animate-pulse'
            )} />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="text-center p-2 rounded-lg bg-slate-800/50">
              <p className="text-2xl font-bold text-white font-mono">{stats?.anomalies_today || 0}</p>
              <p className="text-[10px] text-slate-500 uppercase tracking-wide">Detected</p>
            </div>
            <div className="text-center p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
              <p className="text-2xl font-bold text-emerald-400 font-mono">{stats?.resolved_today || 0}</p>
              <p className="text-[10px] text-slate-500 uppercase tracking-wide">Resolved</p>
            </div>
          </div>
        </div>

        {/* System Health Mini Bar */}
        <div className="flex items-center justify-between px-2">
          <span className="text-[9px] text-slate-600 uppercase tracking-wider">System</span>
          <div className="flex items-center gap-1">
            {[connections?.backend_db, connections?.redis, connections?.qdrant, connections?.dw_sql, connections?.mc_sql, connections?.mobicontrol_api, connections?.llm].map((conn, i) => (
              <div
                key={i}
                className={clsx(
                  'w-1 h-4 rounded-full transition-all',
                  conn?.connected ? 'bg-emerald-400/60' : 'bg-slate-700'
                )}
                title={['PostgreSQL', 'Redis', 'Qdrant', 'DW SQL', 'MC SQL', 'API', 'LLM'][i]}
              />
            ))}
          </div>
        </div>
      </div>

      {/* LLM Settings Modal */}
      <LLMSettingsModal isOpen={showLLMModal} onClose={() => setShowLLMModal(false)} />
    </div>
  );
};
