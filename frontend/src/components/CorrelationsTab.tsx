/**
 * Correlations Tab Component
 *
 * Provides correlation intelligence including matrix heatmaps, scatter plots,
 * causal graphs, auto-discovered insights, cohort patterns, and time-lagged correlations.
 */

import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import clsx from 'clsx';
import { api } from '../api/client';
import { KPICard } from './KPICard';
import {
  DOMAIN_OPTIONS,
  VIEW_OPTIONS,
  CorrelationMatrixView,
  ScatterExplorerView,
  CausalGraphView,
  InsightsView,
  CohortPatternsView,
  TimeLaggedView,
} from './correlations';
import type { SubView } from './correlations';

export function CorrelationsTab() {
  const [selectedDomain, setSelectedDomain] = useState<string | undefined>(undefined);
  const [selectedView, setSelectedView] = useState<SubView>('matrix');
  const [selectedMetricX, setSelectedMetricX] = useState<string>('');
  const [selectedMetricY, setSelectedMetricY] = useState<string>('');

  // Fetch correlation matrix
  const { data: matrixData, isLoading: matrixLoading, error: matrixError, refetch: refetchMatrix } = useQuery({
    queryKey: ['correlations', 'matrix', selectedDomain],
    queryFn: () => api.getCorrelationMatrix({ domain: selectedDomain }),
    retry: 1, // Only retry once on failure
  });

  // Fetch scatter data when metrics selected (increased limit from 500 to 2000 for fuller visualization)
  const { data: scatterData, isLoading: scatterLoading, error: scatterError, refetch: refetchScatter } = useQuery({
    queryKey: ['correlations', 'scatter', selectedMetricX, selectedMetricY],
    queryFn: () => api.getScatterData(selectedMetricX, selectedMetricY, 'anomaly', 2000),
    enabled: selectedView === 'scatter' && !!selectedMetricX && !!selectedMetricY,
    retry: 1,
  });

  // Fetch causal graph
  const { data: causalData, isLoading: causalLoading, error: causalError, refetch: refetchCausal } = useQuery({
    queryKey: ['correlations', 'causal'],
    queryFn: () => api.getCausalGraph(),
    enabled: selectedView === 'causal',
    retry: 1,
  });

  // Fetch correlation insights
  const { data: insightsData, isLoading: insightsLoading, error: insightsError, refetch: refetchInsights } = useQuery({
    queryKey: ['correlations', 'insights'],
    queryFn: () => api.getCorrelationInsights(),
    enabled: selectedView === 'insights',
    retry: 1,
  });

  // Fetch cohort patterns
  const { data: cohortData, isLoading: cohortLoading, error: cohortError, refetch: refetchCohort } = useQuery({
    queryKey: ['correlations', 'cohort'],
    queryFn: () => api.getCohortCorrelationPatterns(),
    enabled: selectedView === 'cohort',
    retry: 1,
  });

  // Fetch time-lagged correlations
  const { data: laggedData, isLoading: laggedLoading, error: laggedError, refetch: refetchLagged } = useQuery({
    queryKey: ['correlations', 'lagged'],
    queryFn: () => api.getTimeLaggedCorrelations(),
    enabled: selectedView === 'lagged',
    retry: 1,
  });

  // Handle matrix cell click to open scatter view
  const handleMatrixCellClick = (metricX: string, metricY: string) => {
    if (metricX !== metricY) {
      setSelectedMetricX(metricX);
      setSelectedMetricY(metricY);
      setSelectedView('scatter');
    }
  };

  // Set default metrics for scatter view from matrix data
  useMemo(() => {
    if (matrixData?.metrics && matrixData.metrics.length >= 2 && !selectedMetricX && !selectedMetricY) {
      setSelectedMetricX(matrixData.metrics[0]);
      setSelectedMetricY(matrixData.metrics[1]);
    }
  }, [matrixData, selectedMetricX, selectedMetricY]);

  return (
    <div className="space-y-6">
      {/* KPI Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <KPICard
          title="Strong Correlations"
          value={matrixData?.strong_correlations?.length || '-'}
          color="aurora"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          }
          explainer="|r| > 0.6 threshold"
        />
        <KPICard
          title="Metrics Analyzed"
          value={matrixData?.metrics?.length || '-'}
          color="stellar"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          }
          explainer="Unique metrics in matrix"
        />
        <KPICard
          title="Causal Links"
          value={causalData?.edges?.length || '26'}
          color="warning"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
            </svg>
          }
          explainer="Domain knowledge relationships"
        />
        <KPICard
          title="Predictive Lags"
          value={laggedData?.correlations?.length || '-'}
          color="stellar"
          icon={
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          }
          explainer="Cross-time correlations"
        />
      </div>

      {/* Domain Filter & View Selector */}
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex gap-1 p-1 bg-slate-800/50 rounded-lg border border-slate-700/50">
          {DOMAIN_OPTIONS.map((domain) => (
            <button
              key={domain.id || 'all'}
              onClick={() => setSelectedDomain(domain.id)}
              className={clsx(
                'px-3 py-2 text-sm font-medium rounded-md transition-all',
                selectedDomain === domain.id
                  ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                  : 'text-slate-400 hover:text-white'
              )}
            >
              {domain.label}
            </button>
          ))}
        </div>

        <div className="flex gap-1 p-1 bg-slate-800/50 rounded-lg border border-slate-700/50">
          {VIEW_OPTIONS.map((view) => (
            <button
              key={view.id}
              onClick={() => setSelectedView(view.id as SubView)}
              className={clsx(
                'px-4 py-2 text-sm font-medium rounded-md transition-all',
                selectedView === view.id
                  ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                  : 'text-slate-400 hover:text-white'
              )}
            >
              {view.label}
            </button>
          ))}
        </div>
      </div>

      {/* Correlation Matrix View */}
      {selectedView === 'matrix' && (
        <CorrelationMatrixView
          data={matrixData}
          isLoading={matrixLoading}
          error={matrixError}
          onRetry={() => refetchMatrix()}
          onCellClick={handleMatrixCellClick}
        />
      )}

      {/* Scatter Plot View */}
      {selectedView === 'scatter' && (
        <ScatterExplorerView
          data={scatterData}
          isLoading={scatterLoading}
          error={scatterError}
          onRetry={() => refetchScatter()}
          metrics={matrixData?.metrics || []}
          selectedMetricX={selectedMetricX}
          selectedMetricY={selectedMetricY}
          onMetricXChange={setSelectedMetricX}
          onMetricYChange={setSelectedMetricY}
        />
      )}

      {/* Causal Graph View */}
      {selectedView === 'causal' && (
        <CausalGraphView data={causalData} isLoading={causalLoading} error={causalError} onRetry={() => refetchCausal()} />
      )}

      {/* Insights View */}
      {selectedView === 'insights' && (
        <InsightsView data={insightsData} isLoading={insightsLoading} error={insightsError} onRetry={() => refetchInsights()} />
      )}

      {/* Cohort Patterns View */}
      {selectedView === 'cohort' && (
        <CohortPatternsView data={cohortData} isLoading={cohortLoading} error={cohortError} onRetry={() => refetchCohort()} />
      )}

      {/* Time-Lagged View */}
      {selectedView === 'lagged' && (
        <TimeLaggedView data={laggedData} isLoading={laggedLoading} error={laggedError} onRetry={() => refetchLagged()} />
      )}
    </div>
  );
}

export default CorrelationsTab;
