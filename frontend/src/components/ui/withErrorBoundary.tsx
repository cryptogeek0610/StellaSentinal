/**
 * withErrorBoundary HOC
 *
 * Wraps a component with an ErrorBoundary.
 */

import React from 'react';
import { ErrorBoundary } from './ErrorBoundary';

interface ErrorBoundaryHOCProps {
  fallback?: React.ReactNode;
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
  onRetry?: () => void;
}

export function withErrorBoundary<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  errorBoundaryProps?: ErrorBoundaryHOCProps
) {
  const displayName = WrappedComponent.displayName || WrappedComponent.name || 'Component';

  const ComponentWithErrorBoundary = (props: P) => (
    <ErrorBoundary {...errorBoundaryProps}>
      <WrappedComponent {...props} />
    </ErrorBoundary>
  );

  ComponentWithErrorBoundary.displayName = `withErrorBoundary(${displayName})`;

  return ComponentWithErrorBoundary;
}
