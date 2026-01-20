/**
 * Error state component for correlation views.
 */

import clsx from 'clsx';
import { Card } from '../Card';

interface ErrorStateProps {
  title: string;
  error: unknown;
  onRetry: () => void;
}

export function ErrorState({ title, error, onRetry }: ErrorStateProps) {
  // Extract error details from API error response
  const errorObj = error as { message?: string; status?: number; body?: string };
  let errorMessage = 'An unexpected error occurred';
  let recommendations: string[] = [];

  try {
    // Try to parse structured error from backend
    if (errorObj?.body) {
      const parsed = JSON.parse(errorObj.body);
      if (parsed?.detail) {
        const detail = typeof parsed.detail === 'string' ? JSON.parse(parsed.detail) : parsed.detail;
        errorMessage = detail.message || errorMessage;
        recommendations = detail.recommendations || [];
      }
    } else if (errorObj?.message) {
      errorMessage = errorObj.message;
    }
  } catch {
    // If parsing fails, use the raw error message
    if (errorObj?.message) {
      errorMessage = errorObj.message;
    }
  }

  const statusCode = errorObj?.status;
  const isServiceUnavailable = statusCode === 503;
  const isInsufficientData = statusCode === 422;

  return (
    <Card title={title}>
      <div className="h-96 flex flex-col items-center justify-center text-center p-6">
        <div className={clsx(
          'w-16 h-16 rounded-full flex items-center justify-center mb-4',
          isServiceUnavailable ? 'bg-amber-500/20' : isInsufficientData ? 'bg-blue-500/20' : 'bg-red-500/20'
        )}>
          {isServiceUnavailable ? (
            <svg className="w-8 h-8 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          ) : isInsufficientData ? (
            <svg className="w-8 h-8 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
            </svg>
          ) : (
            <svg className="w-8 h-8 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          )}
        </div>

        <h3 className={clsx(
          'text-lg font-medium mb-2',
          isServiceUnavailable ? 'text-amber-400' : isInsufficientData ? 'text-blue-400' : 'text-red-400'
        )}>
          {isServiceUnavailable ? 'Data Unavailable' : isInsufficientData ? 'Insufficient Data' : 'Error Loading Data'}
        </h3>

        <p className="text-slate-400 mb-4 max-w-md">
          {errorMessage}
        </p>

        {recommendations.length > 0 && (
          <ul className="text-sm text-slate-500 mb-4 text-left">
            {recommendations.map((rec, i) => (
              <li key={i} className="flex items-start gap-2">
                <span className="text-slate-600">•</span>
                <span>{rec}</span>
              </li>
            ))}
          </ul>
        )}

        <button
          onClick={onRetry}
          className="px-4 py-2 bg-blue-500/20 text-blue-400 rounded-lg hover:bg-blue-500/30 transition-colors"
        >
          Try Again
        </button>
      </div>
    </Card>
  );
}
