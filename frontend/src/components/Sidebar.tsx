/**
 * Sidebar - Steve Jobs Redesign
 *
 * Philosophy: "The navigation should be invisible until needed."
 * - Clean navigation only
 * - Removed: alert banner (redundant with nav badge), AI status card, Today's stats
 * - Kept: system health mini bar (single source of truth)
 */

import { Link, useLocation } from 'react-router-dom';
import { clsx } from 'clsx';
import { motion } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { useMockMode } from '../hooks/useMockMode';

/**
 * Apple-like Navigation Philosophy:
 *
 * 1. OBSERVE - What's happening now? (Dashboards, real-time views)
 * 2. INVESTIGATE - What needs attention? (Anomalies, investigations)
 * 3. UNDERSTAND - Why is it happening? (Intelligence, insights)
 * 4. CONTROL - How do I manage it? (ML operations, automation)
 * 5. CONFIGURE - How do I set it up? (System settings)
 *
 * Each section flows naturally into the next, like chapters in a story.
 */
const navSections = [
  {
    title: 'ACT',
    items: [
      {
        name: 'Action Center',
        path: '/action-center',
        description: 'Fix issues now',
        icon: (
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M13 10V3L4 14h7v7l9-11h-7z"
            />
          </svg>
        ),
        badge: 'issues',
      },
    ],
  },
  {
    title: 'OBSERVE',
    items: [
      {
        name: 'Command Center',
        path: '/dashboard',
        description: 'Unified overview',
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
        name: 'Live Operations',
        path: '/noc',
        description: 'Real-time monitoring',
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
        name: 'Fleet',
        path: '/fleet',
        description: 'All devices',
        icon: (
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M12 18h.01M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z"
            />
          </svg>
        ),
        badge: null,
      },
      {
        name: 'Locations',
        path: '/locations',
        description: 'Geographic view',
        icon: (
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"
            />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        ),
        badge: null,
      },
      {
        name: 'Network',
        path: '/network',
        description: 'WiFi & cellular',
        icon: (
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M8.111 16.404a5.5 5.5 0 017.778 0M12 20h.01m-7.08-7.071c3.904-3.905 10.236-3.905 14.141 0M1.394 9.393c5.857-5.857 15.355-5.857 21.213 0"
            />
          </svg>
        ),
        badge: null,
      },
    ],
  },
  {
    title: 'INVESTIGATE',
    items: [
      {
        name: 'Investigations',
        path: '/investigations',
        description: 'Active cases',
        icon: (
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
        ),
        badge: 'anomalies',
      },
    ],
  },
  {
    title: 'UNDERSTAND',
    items: [
      {
        name: 'AI Insights',
        path: '/insights',
        description: 'Intelligence briefing',
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
        name: 'Cost Analysis',
        path: '/costs',
        description: 'Financial impact',
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
        name: 'Security',
        path: '/security',
        description: 'Posture & compliance',
        icon: (
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
            />
          </svg>
        ),
        badge: null,
      },
    ],
  },
  {
    title: 'CONTROL',
    items: [
      {
        name: 'Training',
        path: '/training',
        description: 'ML training status',
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
        description: 'Feature baselines',
        icon: (
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z"
            />
          </svg>
        ),
        badge: null,
      },
      {
        name: 'Automation',
        path: '/automation',
        description: 'Scheduled jobs',
        icon: (
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
            />
          </svg>
        ),
        badge: null,
      },
      {
        name: 'Model Health',
        path: '/model-status',
        description: 'Pipeline status',
        icon: (
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
            />
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
        name: 'System',
        path: '/system',
        description: 'Settings & connections',
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
    ],
  },
];

interface SidebarProps {
  onClose?: () => void;
}

export const Sidebar = ({ onClose }: SidebarProps) => {
  const location = useLocation();
  const { mockMode, toggleMockMode } = useMockMode();

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

  const openAnomalies = stats?.open_cases ?? 0;
  const criticalCount = stats?.critical_issues || 0;

  const connectedCount = connections
    ? [connections.backend_db, connections.redis, connections.qdrant, connections.dw_sql, connections.mc_sql, connections.mobicontrol_api, connections.llm].filter(s => s?.connected).length
    : 0;
  const healthStatus = connectedCount === 7 ? 'healthy' : connectedCount >= 4 ? 'degraded' : 'critical';

  const getBadgeContent = (badgeType: string | null) => {
    if (!badgeType) return null;
    if (badgeType === 'anomalies' && openAnomalies > 0) {
      return { count: openAnomalies, critical: criticalCount > 0 };
    }
    if (badgeType === 'issues') {
      // Show issue count from action center (using open_cases as proxy for now)
      const issueCount = openAnomalies || 6; // Default to 6 for demo
      return { count: issueCount, critical: criticalCount > 0, isAction: true };
    }
    if (badgeType === 'health') {
      return { status: healthStatus };
    }
    return null;
  };

  return (
    <div className="flex flex-col w-72 stellar-glass h-full border-r border-slate-700/30">
      {/* Logo */}
      <div className="flex items-center justify-between h-20 px-6 border-b border-slate-700/30">
        <Link to="/" className="flex items-center gap-4 group" onClick={onClose}>
          <div className="relative">
            <div className="absolute inset-0 rounded-2xl opacity-50 blur-sm animate-pulse"
              style={{ background: 'linear-gradient(135deg, rgba(245, 166, 35, 0.4), rgba(255, 199, 95, 0.2))' }}
            />
            <div className="relative w-12 h-12 rounded-2xl bg-gradient-to-br from-amber-500 via-amber-400 to-orange-500 flex items-center justify-center shadow-stellar">
              <svg className="w-7 h-7 text-slate-900" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
                />
              </svg>
            </div>
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
                      onClick={onClose}
                      className={clsx(
                        'nav-item group relative flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200',
                        isActive
                          ? 'bg-gradient-to-r from-amber-500/15 via-amber-500/10 to-transparent text-white'
                          : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
                      )}
                    >
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
                        <span className={clsx('block text-sm font-medium', isActive ? 'text-white' : '')}>
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
                          animate={(badge as { isAction?: boolean }).isAction ? { scale: [1, 1.05, 1] } : { scale: 1 }}
                          transition={(badge as { isAction?: boolean }).isAction ? { duration: 2, repeat: Infinity } : undefined}
                          className={clsx(
                            'min-w-[24px] h-6 flex items-center justify-center px-2 rounded-full text-xs font-bold',
                            (badge as { isAction?: boolean }).isAction
                              ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 shadow-[0_0_10px_rgba(16,185,129,0.3)]'
                              : (badge as { count: number; critical?: boolean }).critical
                                ? 'bg-red-500/20 text-red-400 border border-red-500/30'
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

      {/* Footer - System Health & Mock Mode Toggle */}
      <div className="p-4 border-t border-slate-700/30 space-y-3">
        {/* Mock Mode Toggle */}
        <button
          onClick={toggleMockMode}
          className={clsx(
            'w-full flex items-center justify-between px-3 py-2 rounded-lg transition-all duration-200',
            mockMode
              ? 'bg-purple-500/20 border border-purple-500/40 hover:bg-purple-500/30'
              : 'bg-slate-800/50 border border-slate-700/50 hover:bg-slate-700/50'
          )}
        >
          <div className="flex items-center gap-2">
            <svg
              className={clsx(
                'w-4 h-4 transition-colors',
                mockMode ? 'text-purple-400' : 'text-slate-500'
              )}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"
              />
            </svg>
            <span className={clsx(
              'text-xs font-medium',
              mockMode ? 'text-purple-300' : 'text-slate-400'
            )}>
              Mock Mode
            </span>
          </div>
          <div className={clsx(
            'relative w-8 h-4 rounded-full transition-colors duration-200',
            mockMode ? 'bg-purple-500' : 'bg-slate-600'
          )}>
            <motion.div
              className="absolute top-0.5 w-3 h-3 bg-white rounded-full shadow-sm"
              animate={{ left: mockMode ? '18px' : '2px' }}
              transition={{ type: 'spring', stiffness: 500, damping: 30 }}
            />
          </div>
        </button>

        {/* System Health Bar */}
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
    </div>
  );
};
