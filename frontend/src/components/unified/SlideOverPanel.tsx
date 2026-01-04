/**
 * SlideOverPanel - Side panel for drill-down details
 *
 * Opens from the right side of the screen to show detail views
 * without navigating away from the dashboard. Preserves context.
 */

import { ReactNode, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import clsx from 'clsx';

interface SlideOverPanelProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  subtitle?: string;
  children: ReactNode;
  /** Width of the panel */
  width?: 'sm' | 'md' | 'lg' | 'xl';
  /** Optional footer actions */
  footer?: ReactNode;
}

const widthStyles = {
  sm: 'max-w-md',
  md: 'max-w-lg',
  lg: 'max-w-2xl',
  xl: 'max-w-4xl',
};

export function SlideOverPanel({
  isOpen,
  onClose,
  title,
  subtitle,
  children,
  width = 'lg',
  footer,
}: SlideOverPanelProps) {
  // Handle escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen, onClose]);

  // Prevent body scroll when open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40"
          />

          {/* Panel */}
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 30, stiffness: 300 }}
            className={clsx(
              'fixed inset-y-0 right-0 z-50 flex flex-col w-full',
              widthStyles[width],
              'bg-slate-900 border-l border-slate-700/50 shadow-2xl'
            )}
          >
            {/* Header */}
            <div className="flex items-start justify-between p-4 border-b border-slate-700/50">
              <div>
                <h2 className="text-lg font-semibold text-white">{title}</h2>
                {subtitle && (
                  <p className="text-sm text-slate-400 mt-1">{subtitle}</p>
                )}
              </div>
              <button
                onClick={onClose}
                className="p-2 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-4">
              {children}
            </div>

            {/* Footer */}
            {footer && (
              <div className="p-4 border-t border-slate-700/50 bg-slate-800/50">
                {footer}
              </div>
            )}
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

export default SlideOverPanel;
