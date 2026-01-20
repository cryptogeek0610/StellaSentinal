/**
 * Test connection button component for Setup page.
 * Tests connectivity to various services (databases, APIs, etc.)
 */

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { clsx } from 'clsx';
import toast from 'react-hot-toast';
import { api } from '../../api/client';
import type { EnvironmentConfig, TestConnectionRequest } from '../../types/setup';

export interface TestConnectionButtonProps {
  type: TestConnectionRequest['type'];
  config: Partial<EnvironmentConfig>;
  label?: string;
}

export function TestConnectionButton({ type, config, label = 'Test Connection' }: TestConnectionButtonProps) {
  const [status, setStatus] = useState<'idle' | 'testing' | 'success' | 'error'>('idle');
  const [message, setMessage] = useState('');

  const testMutation = useMutation({
    mutationFn: () => api.testSetupConnection({ type, config }),
    onMutate: () => {
      setStatus('testing');
      setMessage('');
    },
    onSuccess: (data) => {
      setStatus(data.success ? 'success' : 'error');
      setMessage(data.message);
      if (data.success) {
        toast.success(data.message);
      } else {
        toast.error(data.message);
      }
    },
    onError: (error: Error) => {
      setStatus('error');
      setMessage(error.message);
      toast.error(error.message);
    },
  });

  return (
    <div className="flex items-center gap-3">
      <button
        type="button"
        onClick={() => testMutation.mutate()}
        disabled={status === 'testing'}
        className={clsx(
          'px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200',
          'flex items-center gap-2',
          status === 'testing' && 'opacity-50 cursor-not-allowed',
          status === 'success' && 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30',
          status === 'error' && 'bg-red-500/20 text-red-400 border border-red-500/30',
          status === 'idle' && 'bg-slate-700/50 text-slate-300 hover:bg-slate-700 border border-slate-600/50'
        )}
      >
        {status === 'testing' ? (
          <>
            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            Testing...
          </>
        ) : status === 'success' ? (
          <>
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
            Connected
          </>
        ) : status === 'error' ? (
          <>
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
            Failed
          </>
        ) : (
          <>
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            {label}
          </>
        )}
      </button>
      {message && status !== 'idle' && (
        <span className={clsx(
          'text-xs',
          status === 'success' && 'text-emerald-400',
          status === 'error' && 'text-red-400'
        )}>
          {message}
        </span>
      )}
    </div>
  );
}
