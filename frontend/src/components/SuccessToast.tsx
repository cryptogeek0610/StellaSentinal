/**
 * SuccessToast - Celebration feedback for completed actions
 *
 * Steve Jobs principle: Delight in the details.
 * Shows satisfying feedback when users complete bulk actions.
 */

import { motion, AnimatePresence } from 'framer-motion';
import { useEffect, useState } from 'react';

interface SuccessToastProps {
  message: string;
  count?: number;
  onClose: () => void;
  duration?: number;
}

export function SuccessToast({ message, count, onClose, duration = 3000 }: SuccessToastProps) {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(false);
      setTimeout(onClose, 300); // Wait for exit animation
    }, duration);

    return () => clearTimeout(timer);
  }, [duration, onClose]);

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ opacity: 0, y: 50, scale: 0.9 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 20, scale: 0.95 }}
          className="fixed bottom-6 right-6 z-50"
        >
          <div className="flex items-center gap-3 px-5 py-3 bg-emerald-500/20 backdrop-blur-xl rounded-xl border border-emerald-500/30 shadow-2xl shadow-emerald-500/10">
            {/* Success icon with animation */}
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: 'spring', stiffness: 500, damping: 15, delay: 0.1 }}
              className="w-8 h-8 rounded-full bg-emerald-500/30 flex items-center justify-center"
            >
              <motion.svg
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ duration: 0.4, delay: 0.2 }}
                className="w-5 h-5 text-emerald-400"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={3}
              >
                <motion.path
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 0.4, delay: 0.2 }}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M5 13l4 4L19 7"
                />
              </motion.svg>
            </motion.div>

            <div>
              <p className="text-sm font-medium text-white">{message}</p>
              {count && (
                <p className="text-xs text-emerald-400/80">
                  {count} anomal{count === 1 ? 'y' : 'ies'} resolved
                </p>
              )}
            </div>

            {/* Close button */}
            <button
              onClick={() => setIsVisible(false)}
              className="ml-2 p-1 text-slate-400 hover:text-white transition-colors"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

export default SuccessToast;
