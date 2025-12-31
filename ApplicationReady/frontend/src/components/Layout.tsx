import React, { useState, useEffect } from 'react';
import { Sidebar } from './Sidebar';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { motion, AnimatePresence } from 'framer-motion';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useMockMode } from '../hooks/useMockMode';
import { ToggleSwitch } from './ui';

interface LayoutProps {
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const { mockMode, setMockMode } = useMockMode();
  const [searchQuery, setSearchQuery] = useState('');
  const [showSearch, setShowSearch] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [showMobileSearch, setShowMobileSearch] = useState(false);

  const { data: connectionStatus } = useQuery({
    queryKey: ['dashboard', 'connections'],
    queryFn: () => api.getConnectionStatus(),
    refetchInterval: 30000,
  });

  const { data: stats } = useQuery({
    queryKey: ['dashboard', 'stats'],
    queryFn: () => api.getDashboardStats(),
    refetchInterval: 30000,
  });

  const connectedCount = connectionStatus
    ? [
      connectionStatus.dw_sql,
      connectionStatus.mc_sql,
      connectionStatus.mobicontrol_api,
      connectionStatus.llm,
    ].filter((s) => s.connected).length
    : 0;

  const totalConnections = 4;
  const healthStatus = connectedCount === 4 ? 'healthy' : connectedCount >= 2 ? 'degraded' : 'critical';

  const criticalCount = stats?.critical_issues || 0;

  // Close mobile menu on route change
  useEffect(() => {
    setIsMobileMenuOpen(false);
  }, [location.pathname]);

  // Close mobile menu on escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setIsMobileMenuOpen(false);
      }
    };
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, []);

  // Prevent body scroll when mobile menu is open
  useEffect(() => {
    if (isMobileMenuOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isMobileMenuOpen]);

  // Get page title from path
  const getPageTitle = () => {
    const path = location.pathname;
    if (path === '/' || path === '/dashboard') return 'Command Center';
    if (path.startsWith('/investigations/')) return 'Investigation Details';
    if (path === '/investigations') return 'Investigations';
    if (path.startsWith('/devices/')) return 'Device Details';
    if (path === '/fleet') return 'Fleet Overview';
    if (path === '/insights') return 'AI Insights';
    if (path === '/system') return 'System';
    return 'Stella Sentinel';
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      // If it looks like a device ID, go to that device
      const deviceId = parseInt(searchQuery);
      if (!isNaN(deviceId)) {
        navigate(`/devices/${deviceId}`);
      } else {
        navigate(`/investigations?search=${encodeURIComponent(searchQuery)}`);
      }
      setSearchQuery('');
      setShowSearch(false);
    }
  };

  // Handler to set mock mode - query invalidation is handled by MockModeProvider
  const handleMockModeToggle = (enabled: boolean) => {
    setMockMode(enabled);
  };

  return (
    <div className="flex h-screen overflow-hidden flex-col">
      {/* Mock Mode Banner */}
      <AnimatePresence>
        {mockMode && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-gradient-to-r from-orange-500 via-amber-500 to-orange-500 text-white text-center py-2 px-4 text-sm font-medium flex items-center justify-center gap-3"
            role="alert"
          >
            <span className="flex items-center gap-2">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
              </svg>
              <span>Mock Mode Active — Data is simulated for demonstration purposes</span>
            </span>
            <button
              onClick={() => handleMockModeToggle(false)}
              className="ml-2 px-2 py-0.5 bg-white/20 hover:bg-white/30 rounded text-xs font-medium transition-colors"
            >
              Disable
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="flex flex-1 overflow-hidden">
        {/* Desktop Sidebar */}
        <div className="hidden lg:block">
          <Sidebar />
        </div>

        {/* Mobile Sidebar Overlay */}
        <AnimatePresence>
          {isMobileMenuOpen && (
            <>
              {/* Backdrop */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40 lg:hidden"
                onClick={() => setIsMobileMenuOpen(false)}
                aria-hidden="true"
              />

              {/* Mobile Sidebar Drawer */}
              <motion.div
                initial={{ x: '-100%' }}
                animate={{ x: 0 }}
                exit={{ x: '-100%' }}
                transition={{ type: 'spring', damping: 25, stiffness: 300 }}
                className="fixed inset-y-0 left-0 z-50 lg:hidden"
              >
                <Sidebar onClose={() => setIsMobileMenuOpen(false)} />
              </motion.div>
            </>
          )}
        </AnimatePresence>

        <div className="flex flex-col flex-1 overflow-hidden">
          {/* Header */}
          <header className="h-16 stellar-glass border-b border-slate-700/30 flex items-center px-4 lg:px-6 justify-between relative z-10">
            {/* Left - Mobile Menu Button & Breadcrumb/Title */}
            <div className="flex items-center gap-3">
              {/* Mobile Menu Button */}
              <button
                onClick={() => setIsMobileMenuOpen(true)}
                className="lg:hidden p-2 -ml-2 text-slate-400 hover:text-white hover:bg-slate-800/50 rounded-lg transition-colors"
                aria-label="Open navigation menu"
                aria-expanded={isMobileMenuOpen}
              >
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>

              <h2 className="text-lg font-semibold text-white truncate">{getPageTitle()}</h2>
            </div>

            {/* Right - Actions */}
            <div className="flex items-center gap-2 sm:gap-4">
              {/* Critical Alert Badge */}
              <AnimatePresence>
                {criticalCount > 0 && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    className="hidden sm:block"
                  >
                    <Link
                      to="/investigations?severity=critical"
                      className="flex items-center gap-2 px-3 py-1.5 bg-red-500/10 border border-red-500/30 rounded-lg hover:bg-red-500/20 transition-colors shadow-danger"
                      aria-label={`${criticalCount} critical alerts, click to view`}
                    >
                      <motion.div
                        className="w-2 h-2 bg-red-400 rounded-full shadow-[0_0_8px_rgba(239,68,68,0.6)]"
                        animate={{ opacity: [1, 0.4, 1] }}
                        transition={{ duration: 1, repeat: Infinity }}
                        aria-hidden="true"
                      />
                      <span className="text-xs font-bold text-red-400">
                        {criticalCount} Critical
                      </span>
                    </Link>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Mobile Critical Badge (compact) */}
              <AnimatePresence>
                {criticalCount > 0 && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    className="sm:hidden"
                  >
                    <Link
                      to="/investigations?severity=critical"
                      className="flex items-center justify-center w-8 h-8 bg-red-500/10 border border-red-500/30 rounded-lg"
                      aria-label={`${criticalCount} critical alerts`}
                    >
                      <motion.div
                        className="w-2 h-2 bg-red-400 rounded-full shadow-[0_0_8px_rgba(239,68,68,0.6)]"
                        animate={{ opacity: [1, 0.4, 1] }}
                        transition={{ duration: 1, repeat: Infinity }}
                        aria-hidden="true"
                      />
                    </Link>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Mock Mode Toggle */}
              <div
                className={`flex items-center gap-2 px-2 sm:px-3 py-1.5 rounded-lg border transition-colors ${mockMode
                  ? 'border-orange-500/30 bg-orange-500/10'
                  : 'border-slate-600/30 bg-slate-700/30'
                  }`}
                title={mockMode ? 'Mock Mode Enabled' : 'Mock Mode Disabled'}
              >
                <svg
                  className={`w-4 h-4 ${mockMode ? 'text-orange-400' : 'text-slate-400'}`}
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  aria-hidden="true"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"
                  />
                </svg>
                <ToggleSwitch
                  enabled={mockMode}
                  onChange={handleMockModeToggle}
                  size="sm"
                  variant="stellar"
                  aria-label={`Mock mode is ${mockMode ? 'enabled' : 'disabled'}. Click to toggle.`}
                />
                <span className={`text-xs font-medium hidden sm:inline ${mockMode ? 'text-orange-400' : 'text-slate-400'}`}>
                  {mockMode ? 'Mock' : 'Live'}
                </span>
              </div>

              {/* System Health */}
              <Link
                to="/system"
                className={`flex items-center gap-2 px-2 sm:px-3 py-1.5 rounded-lg border transition-colors ${healthStatus === 'healthy'
                  ? 'border-emerald-500/30 bg-emerald-500/5 hover:bg-emerald-500/10'
                  : healthStatus === 'degraded'
                    ? 'border-orange-500/30 bg-orange-500/5 hover:bg-orange-500/10'
                    : 'border-red-500/30 bg-red-500/5 hover:bg-red-500/10'
                  }`}
                aria-label={`System health: ${connectedCount} of ${totalConnections} services connected. Click to view system settings.`}
              >
                <div
                  className={`w-2 h-2 rounded-full ${healthStatus === 'healthy' ? 'bg-emerald-400 shadow-[0_0_6px_rgba(16,185,129,0.6)]' :
                    healthStatus === 'degraded' ? 'bg-orange-400 animate-pulse shadow-[0_0_6px_rgba(251,146,60,0.6)]' :
                      'bg-red-400 animate-pulse shadow-[0_0_6px_rgba(248,113,113,0.6)]'
                    }`}
                  aria-hidden="true"
                />
                <span className="text-xs font-mono text-slate-400">
                  {connectedCount}/{totalConnections}
                </span>
              </Link>

              {/* Search - Hidden on mobile, shown on larger screens */}
              <div className="relative hidden md:block">
                <form onSubmit={handleSearch}>
                  <label htmlFor="header-search" className="sr-only">
                    Search devices
                  </label>
                  <input
                    id="header-search"
                    type="text"
                    placeholder="Search devices..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onFocus={() => setShowSearch(true)}
                    onBlur={() => setTimeout(() => setShowSearch(false), 200)}
                    className="input-stellar w-44 lg:w-56 pl-10"
                  />
                  <svg
                    className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    aria-hidden="true"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                    />
                  </svg>
                </form>

                {/* Search Hint */}
                <AnimatePresence>
                  {showSearch && searchQuery && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className="absolute top-full left-0 right-0 mt-2 p-3 stellar-card rounded-xl text-xs text-slate-400"
                      role="status"
                    >
                      Press Enter to search for "{searchQuery}"
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Mobile Search Button */}
              <button
                onClick={() => setShowMobileSearch(true)}
                className="md:hidden p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 rounded-lg transition-colors"
                aria-label="Search devices"
              >
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                  />
                </svg>
              </button>

              {/* Notifications */}
              <button
                onClick={() => navigate('/investigations')}
                className="relative p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 rounded-lg transition-colors"
                aria-label={`Notifications${(stats?.anomalies_today || 0) > 0 ? `, ${stats?.anomalies_today} new anomalies today` : ''}`}
              >
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"
                  />
                </svg>
                {(stats?.anomalies_today || 0) > 0 && (
                  <span className="absolute top-1 right-1 w-2 h-2 bg-amber-400 rounded-full" aria-hidden="true" />
                )}
              </button>
            </div>
          </header>

          {/* Main Content */}
          <main className="flex-1 overflow-y-auto p-4 lg:p-6">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
            >
              {children}
            </motion.div>
          </main>
        </div>

        {/* Mobile Search Modal */}
        <AnimatePresence>
          {showMobileSearch && (
            <>
              {/* Backdrop */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={() => setShowMobileSearch(false)}
                className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 md:hidden"
              />
              {/* Modal */}
              <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="fixed top-4 left-4 right-4 z-50 md:hidden"
              >
                <div className="stellar-card p-4 rounded-2xl shadow-2xl">
                  <form onSubmit={(e) => {
                    handleSearch(e);
                    setShowMobileSearch(false);
                  }}>
                    <div className="relative">
                      <input
                        type="text"
                        placeholder="Search devices..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        autoFocus
                        className="input-stellar w-full pl-10 pr-10"
                      />
                      <svg
                        className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-500"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                        />
                      </svg>
                      <button
                        type="button"
                        onClick={() => setShowMobileSearch(false)}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-200"
                      >
                        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                    {searchQuery && (
                      <p className="mt-3 text-xs text-slate-400">
                        Press Enter to search for "{searchQuery}"
                      </p>
                    )}
                  </form>
                </div>
              </motion.div>
            </>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};
