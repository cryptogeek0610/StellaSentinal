/**
 * ErrorBoundary Component
 *
 * Catches JavaScript errors anywhere in the child component tree,
 * logs those errors, and displays a fallback UI.
 */

import { Component, ErrorInfo, ReactNode } from 'react';
import { EmptyState } from './EmptyState';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  onRetry?: () => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    this.props.onError?.(error, errorInfo);
  }

  handleRetry = (): void => {
    this.setState({ hasError: false, error: null });
    this.props.onRetry?.();
  };

  render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="min-h-[300px] flex items-center justify-center">
          <EmptyState
            variant="error"
            title="Something went wrong"
            description={
              import.meta.env.DEV
                ? this.state.error?.message
                : 'An unexpected error occurred. Please try again.'
            }
            action={{
              label: 'Try Again',
              onClick: this.handleRetry,
            }}
          />
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
