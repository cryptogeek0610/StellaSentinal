/**
 * Layout Component - Steve Jobs Redesign
 *
 * Philosophy: "Simple can be harder than complex."
 * - Clean header: title, search, system health only
 * - Mock mode banner remains (important user feedback)
 * - Removed: duplicate critical badges, mock toggle, notifications bell
 *
 * Accessibility Features:
 * - Skip to main content link
 * - ARIA live region for announcements
 * - Focus management for mobile menu
 */

import React, { useState, useEffect, useCallback } from 'react';
import { Sidebar } from './Sidebar';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { motion, AnimatePresence } from 'framer-motion';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useMockMode } from '../hooks/useMockMode';
import { usePageTitle } from '../hooks/usePageTitle';
import { useFocusTrap } from '../hooks/useFocusTrap';

interface LayoutProps {
  children: React.ReactNode;
}

// ARIA Live Region Announcer for screen readers
const Announcer: React.FC<{ message: string }> = ({ message }) => (
  <div
    role="status"
    aria-live="polite"
    aria-atomic="true"
    className="sr-only"
  >
    {message}
  </div>
);

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const { mockMode, setMockMode } = useMockMode();
  const [searchQuery, setSearchQuery] = useState('');
  const [showSearch, setShowSearch] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [showMobileSearch, setShowMobileSearch] = useState(false);
  const [announcement, setAnnouncement] = useState('');

  // Focus trap for mobile menu
  const mobileMenuRef = useFocusTrap<HTMLDivElement>({ isActive: isMobileMenuOpen });

  // Announce page changes to screen readers
  const announce = useCallback((message: string) => {
    setAnnouncement('');
    // Small delay to ensure the announcement is picked up
    setTimeout(() => setAnnouncement(message), 100);
  }, []);

  const { data: connectionStatus } = useQuery({
    queryKey: ['dashboard', 'connections'],
    queryFn: () => api.getConnectionStatus(),
    refetchInterval: 30000,
  });

  const connectedCount = connectionStatus
    ? [
      connectionStatus.backend_db,
      connectionStatus.redis,
      connectionStatus.qdrant,
      connectionStatus.dw_sql,
      connectionStatus.mc_sql,
      connectionStatus.mobicontrol_api,
      connectionStatus.llm,
    ].filter((s) => s?.connected).length
    : 0;

  const healthStatus = connectedCount === 7 ? 'healthy' : connectedCount >= 4 ? 'degraded' : 'critical';

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
    if (path === '/locations') return 'Location Center';
    if (path === '/data') return 'Data Overview';
    if (path === '/training') return 'Model Training';
    if (path === '/baselines') return 'Baselines';
    if (path === '/automation') return 'Automation';
    if (path === '/system') return 'System';
    if (path === '/status') return 'Model Status';
    if (path === '/setup') return 'Setup Wizard';
    return 'Stella Sentinel';
  };

  // Update browser tab title
  const pageTitle = getPageTitle();
  usePageTitle(pageTitle);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
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

  // Announce page navigation
  useEffect(() => {
    announce(`Navigated to ${pageTitle}`);
  }, [pageTitle, announce]);

  return (
    <div className="flex h-screen overflow-hidden flex-col">
      {/* Skip to Main Content Link - Accessibility */}
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-[100] focus:px-4 focus:py-2 focus:bg-amber-500 focus:text-slate-900 focus:font-semibold focus:rounded-lg focus:shadow-lg focus:outline-none"
      >
        Skip to main content
      </a>

      {/* ARIA Live Region for Announcements */}
      <Announcer message={announcement} />

      {/* Mock Mode Banner - Keep this, it's important user feedback */}
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
              <span>Mock Mode Active — Data is simulated</span>
            </span>
            <button
              onClick={() => setMockMode(false)}
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
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40 lg:hidden"
                onClick={() => setIsMobileMenuOpen(false)}
                aria-hidden="true"
              />
              <motion.div
                ref={mobileMenuRef}
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
          {/* Header - Simplified: title, search, system health */}
          <header className="h-16 stellar-glass border-b border-slate-700/30 flex items-center px-4 lg:px-6 justify-between relative z-10">
            {/* Left - Mobile Menu Button & Title */}
            <div className="flex items-center gap-3">
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
              <h2 className="text-lg font-semibold text-white truncate">{pageTitle}</h2>
            </div>

            {/* Right - Search & System Health (just a dot) */}
            <div className="flex items-center gap-3">
              {/* System Health - Just a dot. Users care about "is it working", not "7/7" */}
              <Link
                to="/system"
                className="p-2 rounded-lg hover:bg-slate-800/50 transition-colors"
                aria-label={`System ${healthStatus}. Click to view details.`}
                title={healthStatus === 'healthy' ? 'All systems operational' : healthStatus === 'degraded' ? 'Some services degraded' : 'Critical issues'}
              >
                <div
                  className={`w-2.5 h-2.5 rounded-full ${
                    healthStatus === 'healthy'
                      ? 'bg-emerald-400 shadow-[0_0_8px_rgba(16,185,129,0.6)]'
                      : healthStatus === 'degraded'
                        ? 'bg-orange-400 animate-pulse shadow-[0_0_8px_rgba(251,146,60,0.6)]'
                        : 'bg-red-400 animate-pulse shadow-[0_0_8px_rgba(248,113,113,0.6)]'
                  }`}
                />
              </Link>

              {/* Search - Desktop */}
              <div className="relative hidden md:block">
                <form onSubmit={handleSearch}>
                  <label htmlFor="header-search" className="sr-only">Search devices</label>
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
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                </form>
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
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </button>
            </div>
          </header>

          {/* Main Content */}
          <main
            id="main-content"
            className="flex-1 overflow-y-auto p-4 lg:p-6"
            tabIndex={-1}
          >
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
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={() => setShowMobileSearch(false)}
                className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 md:hidden"
              />
              <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="fixed top-4 left-4 right-4 z-50 md:hidden"
              >
                <div className="stellar-card p-4 rounded-2xl shadow-2xl">
                  <form onSubmit={(e) => { handleSearch(e); setShowMobileSearch(false); }}>
                    <div className="relative">
                      <input
                        type="text"
                        placeholder="Search devices..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        autoFocus
                        className="input-stellar w-full pl-10 pr-10"
                      />
                      <svg className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
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
                      <p className="mt-3 text-xs text-slate-400">Press Enter to search for "{searchQuery}"</p>
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
