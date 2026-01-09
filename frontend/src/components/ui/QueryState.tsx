import React from 'react';
import { Card } from '../Card';

interface QueryStateProps {
  isLoading?: boolean;
  isError?: boolean;
  error?: Error | null;
  isEmpty?: boolean;
  emptyMessage?: string;
  onRetry?: () => void;
  children: React.ReactNode;
}

export function QueryState({
  isLoading,
  isError,
  error,
  isEmpty,
  emptyMessage = 'No data available yet.',
  onRetry,
  children,
}: QueryStateProps) {
  if (isLoading) {
    return (
      <Card className="p-6">
        <div className="flex items-center justify-center text-slate-500">
          <div className="w-5 h-5 border-2 border-slate-600 border-t-transparent rounded-full animate-spin" />
        </div>
      </Card>
    );
  }

  if (isError) {
    return (
      <Card className="p-6 border border-red-500/30">
        <div className="text-center text-slate-300">
          <div className="text-sm font-semibold text-red-400">Failed to load data</div>
          <div className="text-xs text-slate-500 mt-1">
            {error?.message || 'Unexpected error'}
          </div>
          {onRetry && (
            <button
              type="button"
              onClick={onRetry}
              className="mt-4 px-3 py-1.5 text-xs rounded-md border border-slate-600/60 text-slate-300 hover:text-white hover:border-slate-400 transition-colors"
            >
              Retry
            </button>
          )}
        </div>
      </Card>
    );
  }

  if (isEmpty) {
    return (
      <Card className="p-6">
        <div className="text-center text-slate-500 text-sm">{emptyMessage}</div>
      </Card>
    );
  }

  return <>{children}</>;
}
