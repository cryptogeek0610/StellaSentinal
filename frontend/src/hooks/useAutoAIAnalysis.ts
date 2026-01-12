/**
 * useAutoAIAnalysis Hook
 *
 * Manages automatic AI analysis triggering with:
 * - Page load detection (first view)
 * - Data change detection via hash comparison
 * - Session storage tracking to prevent redundant triggers
 * - Mock mode support
 */

import { useEffect, useRef, useCallback, useMemo } from 'react';
import { useMockMode } from './useMockMode';

export interface UseAutoAIAnalysisOptions {
  /** Type of analysis to perform */
  analysisType: 'anomaly' | 'dashboard-summary' | 'digest';
  /** Anomaly ID for anomaly-specific analysis */
  anomalyId?: number;
  /** Data to monitor for changes (will be hashed to detect changes) */
  watchData?: unknown;
  /** Enable/disable auto-trigger (default: true) */
  enabled?: boolean;
  /** Auto-trigger on first render (default: true) */
  triggerOnMount?: boolean;
  /** Skip if already triggered this session (default: true) */
  skipIfTriggeredThisSession?: boolean;
}

export interface UseAutoAIAnalysisResult {
  /** Whether analysis should be fetched */
  shouldFetch: boolean;
  /** Mark this analysis as triggered (call after successful fetch) */
  markTriggered: () => void;
  /** Whether analysis was already triggered this session */
  wasTriggeredThisSession: boolean;
  /** Whether data has changed since last check */
  dataChanged: boolean;
  /** Clear the session trigger (allows re-triggering) */
  clearSessionTrigger: () => void;
}

/**
 * Simple hash function for change detection.
 * Uses djb2 algorithm for fast, reasonable distribution.
 */
function hashData(data: unknown): string {
  const str = JSON.stringify(data);
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) + hash) + str.charCodeAt(i);
    hash |= 0; // Convert to 32bit integer
  }
  return hash.toString(16);
}

/**
 * Hook to manage automatic AI analysis triggering.
 *
 * @example
 * ```tsx
 * const { shouldFetch, markTriggered } = useAutoAIAnalysis({
 *   analysisType: 'anomaly',
 *   anomalyId: 123,
 *   enabled: !!anomalyId,
 * });
 *
 * const { data } = useQuery({
 *   queryKey: ['ai-analysis', anomalyId],
 *   queryFn: () => api.getAIAnalysis(anomalyId),
 *   enabled: shouldFetch,
 *   onSuccess: () => markTriggered(),
 * });
 * ```
 */
export function useAutoAIAnalysis(options: UseAutoAIAnalysisOptions): UseAutoAIAnalysisResult {
  const {
    analysisType,
    anomalyId,
    watchData,
    enabled = true,
    triggerOnMount = true,
    skipIfTriggeredThisSession = true,
  } = options;

  const { mockMode } = useMockMode();
  const lastDataHash = useRef<string>('');
  const dataChangedRef = useRef(false);

  // Session storage key for "first view" tracking
  const sessionKey = useMemo(() => {
    const id = anomalyId ?? 'global';
    return `ai_triggered_${analysisType}_${id}`;
  }, [analysisType, anomalyId]);

  // Check if already triggered this session
  const wasTriggeredThisSession = useMemo(() => {
    if (typeof window === 'undefined') return false;
    return sessionStorage.getItem(sessionKey) === 'true';
  }, [sessionKey]);

  // Track data changes
  useEffect(() => {
    if (watchData !== undefined) {
      const newHash = hashData(watchData);
      if (lastDataHash.current && newHash !== lastDataHash.current) {
        // Data changed significantly
        dataChangedRef.current = true;
      }
      lastDataHash.current = newHash;
    }
  }, [watchData]);

  // Determine if we should fetch
  const shouldFetch = useMemo(() => {
    // Disabled - don't fetch
    if (!enabled) return false;

    // Mock mode - always allow (mock data will be returned)
    // Real mode - check conditions
    if (!mockMode) {
      // Skip if triggered this session and not configured to trigger on mount
      if (skipIfTriggeredThisSession && wasTriggeredThisSession && !dataChangedRef.current) {
        return false;
      }
    }

    // Trigger on mount or when data changed
    return triggerOnMount || dataChangedRef.current;
  }, [enabled, mockMode, skipIfTriggeredThisSession, wasTriggeredThisSession, triggerOnMount]);

  // Mark as triggered
  const markTriggered = useCallback(() => {
    if (typeof window !== 'undefined') {
      sessionStorage.setItem(sessionKey, 'true');
    }
    dataChangedRef.current = false;
  }, [sessionKey]);

  // Clear session trigger (for manual refresh)
  const clearSessionTrigger = useCallback(() => {
    if (typeof window !== 'undefined') {
      sessionStorage.removeItem(sessionKey);
    }
    dataChangedRef.current = false;
  }, [sessionKey]);

  return {
    shouldFetch,
    markTriggered,
    wasTriggeredThisSession,
    dataChanged: dataChangedRef.current,
    clearSessionTrigger,
  };
}

export default useAutoAIAnalysis;
