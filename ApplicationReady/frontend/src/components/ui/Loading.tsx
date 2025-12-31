/**
 * Loading Components
 *
 * Consistent loading states for the application.
 */

import { motion } from 'framer-motion';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export function LoadingSpinner({ size = 'md', className = '' }: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: 'h-4 w-4',
    md: 'h-6 w-6',
    lg: 'h-8 w-8',
  };

  return (
    <svg
      className={`animate-spin ${sizeClasses[size]} ${className}`}
      viewBox="0 0 24 24"
      aria-hidden="true"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
        fill="none"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );
}

interface LoadingDotsProps {
  className?: string;
}

export function LoadingDots({ className = '' }: LoadingDotsProps) {
  return (
    <div className={`flex items-center gap-1 ${className}`} aria-hidden="true">
      {[0, 1, 2].map((i) => (
        <motion.div
          key={i}
          className="w-1.5 h-1.5 bg-current rounded-full"
          animate={{ opacity: [0.3, 1, 0.3] }}
          transition={{ duration: 1, repeat: Infinity, delay: i * 0.2 }}
        />
      ))}
    </div>
  );
}

interface LoadingCardProps {
  lines?: number;
  className?: string;
}

export function LoadingCard({ lines = 3, className = '' }: LoadingCardProps) {
  return (
    <div
      className={`stellar-card rounded-xl p-4 animate-pulse ${className}`}
      role="status"
      aria-label="Loading"
    >
      <div className="space-y-3">
        <div className="h-4 bg-slate-700/50 rounded w-3/4" />
        {Array.from({ length: lines - 1 }).map((_, i) => (
          <div
            key={i}
            className="h-3 bg-slate-700/30 rounded"
            style={{ width: `${100 - (i + 1) * 15}%` }}
          />
        ))}
      </div>
    </div>
  );
}

interface LoadingSkeletonProps {
  variant?: 'text' | 'circular' | 'rectangular';
  width?: string | number;
  height?: string | number;
  className?: string;
}

export function LoadingSkeleton({
  variant = 'rectangular',
  width,
  height,
  className = '',
}: LoadingSkeletonProps) {
  const baseClasses = 'animate-shimmer bg-slate-700/50';
  const variantClasses = {
    text: 'rounded h-4',
    circular: 'rounded-full',
    rectangular: 'rounded-lg',
  };

  const style: React.CSSProperties = {};
  if (width) style.width = typeof width === 'number' ? `${width}px` : width;
  if (height) style.height = typeof height === 'number' ? `${height}px` : height;

  return (
    <div
      className={`${baseClasses} ${variantClasses[variant]} ${className}`}
      style={style}
      role="status"
      aria-label="Loading"
    />
  );
}

interface LoadingOverlayProps {
  message?: string;
}

export function LoadingOverlay({ message = 'Loading...' }: LoadingOverlayProps) {
  return (
    <div
      className="fixed inset-0 bg-void/80 backdrop-blur-sm z-50 flex items-center justify-center"
      role="status"
      aria-label={message}
    >
      <motion.div
        className="text-center"
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
      >
        <div className="relative w-16 h-16 mx-auto mb-4">
          <motion.div
            className="absolute inset-0 rounded-full border-2 border-amber-500/30"
            animate={{ scale: [1, 1.2, 1], opacity: [0.3, 0.1, 0.3] }}
            transition={{ duration: 2, repeat: Infinity }}
          />
          <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-amber-500 animate-spin" />
          <div
            className="absolute inset-2 rounded-full border-2 border-transparent border-t-indigo-500 animate-spin"
            style={{ animationDirection: 'reverse', animationDuration: '1.5s' }}
          />
        </div>
        <p className="text-slate-400 font-mono text-sm">{message}</p>
      </motion.div>
    </div>
  );
}

interface LoadingPageProps {
  message?: string;
}

export function LoadingPage({ message = 'Initializing...' }: LoadingPageProps) {
  return (
    <div
      className="flex items-center justify-center h-96"
      role="status"
      aria-label={message}
    >
      <motion.div
        className="text-center"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <div className="relative w-20 h-20 mx-auto mb-6">
          <motion.div
            className="absolute inset-0 rounded-full border-2 border-amber-500/30"
            animate={{ scale: [1, 1.2, 1], opacity: [0.3, 0.1, 0.3] }}
            transition={{ duration: 2, repeat: Infinity }}
          />
          <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-amber-500 animate-spin" />
          <div
            className="absolute inset-2 rounded-full border-2 border-transparent border-t-indigo-500 animate-spin"
            style={{ animationDirection: 'reverse', animationDuration: '1.5s' }}
          />
        </div>
        <p className="text-slate-400 font-mono text-sm">{message}</p>
      </motion.div>
    </div>
  );
}
