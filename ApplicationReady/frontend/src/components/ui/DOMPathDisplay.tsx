/**
 * DOM Path Display Component
 * 
 * Displays DOM path information for debugging and development.
 * Shows path, position, and React component info.
 */

import React, { useRef, useEffect, useState } from 'react';
import { getDOMPathInfo, formatDOMPathInfo, type DOMPathInfo } from '../../utils/domPath';
import { motion, AnimatePresence } from 'framer-motion';

interface DOMPathDisplayProps {
  element: HTMLElement | null;
  showOnHover?: boolean;
  showOnClick?: boolean;
  position?: 'top' | 'right' | 'bottom' | 'left';
  className?: string;
}

export const DOMPathDisplay: React.FC<DOMPathDisplayProps> = ({
  element,
  showOnHover = false,
  showOnClick = true,
  position = 'bottom',
  className = '',
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [pathInfo, setPathInfo] = useState<DOMPathInfo | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (element) {
      const info = getDOMPathInfo(element);
      setPathInfo(info);
    }
  }, [element]);

  const handleClick = (e: React.MouseEvent) => {
    if (showOnClick) {
      e.stopPropagation();
      setIsVisible(!isVisible);
      if (element && !pathInfo) {
        const info = getDOMPathInfo(element);
        setPathInfo(info);
        if (info) {
          console.log(formatDOMPathInfo(info));
        }
      }
    }
  };

  const handleMouseEnter = () => {
    if (showOnHover && element) {
      const info = getDOMPathInfo(element);
      setPathInfo(info);
      setIsVisible(true);
    }
  };

  const handleMouseLeave = () => {
    if (showOnHover) {
      setIsVisible(false);
    }
  };

  if (!element) return null;

  const getPositionStyles = () => {
    const rect = element.getBoundingClientRect();
    const offset = 8;
    switch (position) {
      case 'top':
        return {
          left: `${rect.left + rect.width / 2}px`,
          top: `${rect.top - offset}px`,
          transform: 'translate(-50%, -100%)',
        };
      case 'bottom':
        return {
          left: `${rect.left + rect.width / 2}px`,
          top: `${rect.bottom + offset}px`,
          transform: 'translate(-50%, 0)',
        };
      case 'left':
        return {
          left: `${rect.left - offset}px`,
          top: `${rect.top + rect.height / 2}px`,
          transform: 'translate(-100%, -50%)',
        };
      case 'right':
        return {
          left: `${rect.right + offset}px`,
          top: `${rect.top + rect.height / 2}px`,
          transform: 'translate(0, -50%)',
        };
      default:
        return {};
    }
  };

  return (
    <>
      <div
        ref={containerRef}
        className={`dom-path-trigger ${className}`}
        onClick={handleClick}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        style={{ display: 'inline-block' }}
      >
        {/* Invisible overlay to capture events */}
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            cursor: showOnClick ? 'help' : 'default',
            zIndex: 1,
          }}
        />
      </div>

      <AnimatePresence>
        {isVisible && pathInfo && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className="fixed z-[9999] pointer-events-none"
            style={getPositionStyles()}
          >
            <div className="stellar-card rounded-lg p-4 border border-amber-500/30 shadow-xl max-w-md bg-slate-900/95 backdrop-blur-sm">
              <div className="space-y-2">
                <div className="flex items-center gap-2 mb-2">
                  <svg
                    className="w-4 h-4 text-amber-400"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                  <span className="text-xs font-bold text-amber-400 uppercase tracking-wider">
                    DOM Path Info
                  </span>
                </div>
                <div className="text-xs font-mono text-slate-300 space-y-1">
                  <div>
                    <span className="text-slate-500">Path:</span>
                    <div className="text-amber-300 break-all mt-0.5">
                      {pathInfo.path}
                    </div>
                  </div>
                  <div>
                    <span className="text-slate-500">Position:</span>
                    <div className="text-slate-300 mt-0.5">
                      top={pathInfo.position.top}px, left={pathInfo.position.left}px,
                      <br />
                      width={pathInfo.position.width}px, height={pathInfo.position.height}px
                    </div>
                  </div>
                  {pathInfo.reactComponent && (
                    <div>
                      <span className="text-slate-500">React Component:</span>
                      <div className="text-cyan-300 mt-0.5">{pathInfo.reactComponent}</div>
                    </div>
                  )}
                  <div>
                    <span className="text-slate-500">HTML Element:</span>
                    <div className="text-slate-300 break-all mt-0.5">{pathInfo.htmlElement}</div>
                  </div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    navigator.clipboard.writeText(formatDOMPathInfo(pathInfo));
                  }}
                  className="mt-2 text-xs px-2 py-1 bg-amber-500/20 text-amber-400 border border-amber-500/30 rounded hover:bg-amber-500/30 transition-colors"
                >
                  Copy to Clipboard
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

// Higher-order component removed - use DOMPathDisplay directly with refs

