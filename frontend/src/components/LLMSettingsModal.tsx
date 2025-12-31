/**
 * LLM Settings Modal
 * 
 * Modal wrapper for LLM Settings configuration
 */

import { motion, AnimatePresence } from 'framer-motion';
import { createPortal } from 'react-dom';
import { LLMSettings } from './settings/LLMSettings';

interface LLMSettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function LLMSettingsModal({ isOpen, onClose }: LLMSettingsModalProps) {
  const modalContent = (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[100] flex items-center justify-center p-4 overflow-y-auto"
          onClick={onClose}
          role="dialog"
          aria-modal="true"
          aria-labelledby="llm-settings-modal-title"
        >
          <motion.div
            initial={{ scale: 0.95, opacity: 0, y: 20 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0.95, opacity: 0, y: 20 }}
            className="bg-slate-900 border border-slate-700 rounded-xl w-full max-w-4xl my-8 max-h-[90vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div className="sticky top-0 z-10 flex items-center justify-between p-6 border-b border-slate-700/50 bg-slate-900/95 backdrop-blur-sm">
              <div>
                <h2 id="llm-settings-modal-title" className="text-xl font-bold text-white">
                  LLM Configuration
                </h2>
                <p className="text-sm text-slate-500 mt-1">
                  Configure AI model service and select models for anomaly explanations
                </p>
              </div>
              <button
                type="button"
                onClick={onClose}
                className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
                aria-label="Close LLM settings modal"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Modal Content */}
            <div className="p-6">
              <LLMSettings />
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );

  // Render modal in a portal to document.body to ensure proper z-index stacking
  return typeof document !== 'undefined' 
    ? createPortal(modalContent, document.body)
    : null;
}

