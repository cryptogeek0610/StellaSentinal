/**
 * InfoTooltip Component
 * 
 * Reusable info icon with hover tooltip for providing contextual help
 * without cluttering the UI. Uses portal-based positioning and stellar theme.
 */

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface InfoTooltipProps {
  content: string | React.ReactNode;
  position?: 'top' | 'right' | 'bottom' | 'left';
  maxWidth?: string;
  className?: string;
}

export const InfoTooltip: React.FC<InfoTooltipProps> = ({
  content,
  position = 'top',
  maxWidth = '240px',
  className = '',
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [coords, setCoords] = useState({ x: 0, y: 0 });
  const iconRef = useRef<HTMLDivElement>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout>>();

  const handleMouseEnter = () => {
    // Small delay before showing tooltip
    timeoutRef.current = setTimeout(() => {
      if (iconRef.current) {
        const rect = iconRef.current.getBoundingClientRect();
        setCoords({
          x: rect.left + rect.width / 2,
          y: rect.top,
        });
      }
      setIsVisible(true);
    }, 150);
  };

  const handleMouseLeave = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    setIsVisible(false);
  };

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  const getPositionStyles = () => {
    const offset = 8;
    switch (position) {
      case 'top':
        return {
          left: `${coords.x}px`,
          top: `${coords.y - offset}px`,
          transform: 'translate(-50%, -100%)',
        };
      case 'bottom':
        return {
          left: `${coords.x}px`,
          top: `${coords.y + offset}px`,
          transform: 'translate(-50%, 0)',
        };
      case 'left':
        return {
          left: `${coords.x - offset}px`,
          top: `${coords.y}px`,
          transform: 'translate(-100%, -50%)',
        };
      case 'right':
        return {
          left: `${coords.x + offset}px`,
          top: `${coords.y}px`,
          transform: 'translate(0, -50%)',
        };
      default:
        return {};
    }
  };

  return (
    <>
      <div
        ref={iconRef}
        className={`inline-flex items-center justify-center ${className}`}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        aria-label="More information"
        role="button"
        tabIndex={0}
      >
        <svg
          className="info-icon"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          aria-hidden="true"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
      </div>

      <AnimatePresence>
        {isVisible && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className="fixed z-50 pointer-events-none"
            style={{
              ...getPositionStyles(),
              maxWidth,
            }}
          >
            <div className="stellar-card rounded-lg p-3 border border-amber-500/20 shadow-lg">
              <div className="text-xs text-slate-300 leading-relaxed">
                {content}
              </div>
              {/* Tooltip arrow */}
              <div
                className="absolute w-2 h-2 bg-slate-800 border-amber-500/20 transform rotate-45"
                style={{
                  ...(position === 'top' && {
                    bottom: '-4px',
                    left: '50%',
                    marginLeft: '-4px',
                    borderRight: '1px solid',
                    borderBottom: '1px solid',
                  }),
                  ...(position === 'bottom' && {
                    top: '-4px',
                    left: '50%',
                    marginLeft: '-4px',
                    borderLeft: '1px solid',
                    borderTop: '1px solid',
                  }),
                  ...(position === 'left' && {
                    right: '-4px',
                    top: '50%',
                    marginTop: '-4px',
                    borderRight: '1px solid',
                    borderTop: '1px solid',
                  }),
                  ...(position === 'right' && {
                    left: '-4px',
                    top: '50%',
                    marginTop: '-4px',
                    borderLeft: '1px solid',
                    borderBottom: '1px solid',
                  }),
                }}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

export default InfoTooltip;
