import type {
  AllConnectionsStatus,
  AnomalyDetail,
  AnomalyListResponse,
  DashboardStats,
  DashboardTrend,
  DashboardAISummary,
  DeviceDetail,
  DeviceListResponse,
  TroubleshootingAdvice,
  IsolationForestStats,
  BaselineSuggestion,
  BaselineAdjustmentRequest,
  BaselineAdjustmentResponse,
  BaselineFeature,
  BaselineHistoryEntry,
  LLMConfig,
  LLMConfigUpdate,
  LLMModelsResponse,
  LLMTestResult,
  OllamaPullResponse,
  PopularModel,
  LocationHeatmapResponse,
  Anomaly,
  InvestigationPanel,
  AnomalyExplanation,
  BaselineComparison,
  HistoricalTimeline,
  AIAnalysis,
  AIAnalysisFeedback,
  RemediationSuggestion,
  RemediationOutcome,
  SimilarCase,
  LearnFromFix,
  GroupedAnomaliesResponse,
  BulkActionRequest,
  BulkActionResponse,
} from '../types/anomaly';
import type {
  TableProfile,
  TemporalPattern,
  MetricDistribution,
  AvailableMetric,
  DiscoverySummary,
  DataDiscoveryStatus,
  TrainingConfigRequest,
  TrainingRun,
  TrainingMetrics,
  TrainingArtifacts,
  TrainingQueueStatus,
  TrainingHistoryResponse,
} from '../types/training';
import type {
  SchedulerConfig,
  SchedulerConfigUpdate,
  SchedulerStatus,
  AutomationAlert,
  AutomationHistory,
  ScoreRequest,
  ScoreResponse,
  TriggerJobRequest,
  TriggerJobResponse,
} from '../types/automation';
import type {
  HardwareCostCreate,
  HardwareCostUpdate,
  HardwareCostResponse,
  HardwareCostListResponse,
  DeviceModelsResponse,
  OperationalCostCreate,
  OperationalCostUpdate,
  OperationalCostResponse,
  OperationalCostListResponse,
  CostSummaryResponse,
  AnomalyImpactResponse,
  DeviceImpactResponse,
  CostHistoryResponse,
  BatteryForecastResponse,
  CostAlert,
  CostAlertCreate,
  CostAlertListResponse,
  NFFSummary,
} from '../types/cost';
import { getMockModeFromStorage } from '../hooks/useMockMode';
import {
  getMockDashboardStats,
  getMockDashboardTrends,
  getMockConnectionStatus,
  getMockAnomalies,
  getMockAnomalyDetail,
  getMockDeviceDetail,
  getMockIsolationForestStats,
  getMockBaselineSuggestions,
  getMockLocationHeatmap,
  getMockCustomAttributes,
  getMockLLMConfig,
  getMockLLMModels,
  getMockLLMTestResult,
  getMockBaselineAdjustmentResponse,
  getMockOllamaPullResponse,
  getMockDevices,
  getMockTableProfiles,
  getMockAvailableMetrics,
  getMockMetricDistribution,
  getMockDiscoveryStatus,
  getMockTemporalPatterns,
  getMockDiscoverySummary,
  getMockAutomationConfig,
  getMockAutomationStatus,
  getMockAutomationAlerts,
  getMockAutomationHistory,
  getMockTriggerJobResponse,
  getMockTrainingStatus,
  getMockTrainingHistory,
  getMockStartTrainingResponse,
  getMockTrainingMetrics,
  // Insights mock data
  getMockDailyDigest,
  getMockLocationInsights,
  getMockShiftReadiness,
  getMockTrendingInsights,
  getMockNetworkAnalysis,
  getMockDeviceAbuseAnalysis,
  getMockAppAnalysis,
  getMockInsightsByCategory,
  getMockLocationComparison,
  // Smart Grouping mock data
  getMockGroupedAnomalies,
  // System Health mock data
  getMockSystemHealthSummary,
  getMockHealthTrends,
  getMockStorageForecast,
  getMockCohortHealthBreakdown,
  // Location Intelligence mock data
  getMockWiFiHeatmap,
  getMockDeadZones,
  getMockDeviceMovements,
  getMockDwellTime,
  getMockCoverageSummary,
  // Events & Alerts mock data
  getMockEventTimeline,
  getMockAlertSummary,
  getMockAlertTrends,
  getMockEventCorrelation,
  getMockEventStatistics,
  // Temporal Analysis mock data
  getMockHourlyBreakdown,
  getMockPeakDetection,
  getMockTemporalComparison,
  getMockDayOverDay,
  getMockWeekOverWeek,
  // Correlation Intelligence mock data
  getMockCorrelationMatrix,
  getMockScatterData,
  getMockCausalGraph,
  getMockCorrelationInsights,
  getMockCohortCorrelationPatterns,
  getMockTimeLaggedCorrelations,
} from './mockData';

import type {
  CorrelationMatrixResponse,
  ScatterPlotResponse,
  CausalGraphResponse,
  CorrelationInsightsResponse,
  CohortCorrelationPatternsResponse,
  TimeLagCorrelationsResponse,
  CorrelationMatrixParams,
} from '../types/correlations';
import {
  parseAnomalyList,
  parseBaselineFeatures,
  parseBaselineSuggestions,
  parseGroupedAnomalies,
  parseIsolationForestStats,
} from './contracts';

// Environment-based API URL configuration
// Set VITE_API_URL in .env for different environments
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';
let lastConnectionStatus: AllConnectionsStatus | null = null;

const getStoredValue = (key: string, fallback?: string): string | undefined => {
  try {
    const stored = localStorage.getItem(key);
    if (stored) {
      return stored;
    }
  } catch {
    // Ignore storage errors (e.g., private mode).
  }
  return fallback;
};

const getAuthHeaders = (): Record<string, string> => {
  const headers: Record<string, string> = {};
  const tenantId = getStoredValue('tenantId', import.meta.env.VITE_TENANT_ID || 'default');
  const apiKey = getStoredValue('apiKey', import.meta.env.VITE_API_KEY);
  const userId = getStoredValue('userId', import.meta.env.VITE_USER_ID);
  const userRole = getStoredValue('userRole', import.meta.env.VITE_USER_ROLE);

  if (tenantId) {
    headers['X-Tenant-Id'] = tenantId;
  }
  if (apiKey) {
    headers['X-Api-Key'] = apiKey;
  }
  if (userId) {
    headers['X-User-Id'] = userId;
  }
  if (userRole) {
    headers['X-User-Role'] = userRole;
  }

  return headers;
};

const hasSqlConnection = (connections: AllConnectionsStatus | null): boolean => {
  if (!connections) {
    return false;
  }
  return connections.dw_sql.connected || connections.mc_sql.connected;
};

const shouldReturnEmptyLiveData = (): boolean => {
  if (getMockModeFromStorage()) {
    return false;
  }
  return !hasSqlConnection(lastConnectionStatus);
};

const emptyDashboardStats = (): DashboardStats => ({
  anomalies_today: 0,
  devices_monitored: 0,
  critical_issues: 0,
  resolved_today: 0,
  open_cases: 0,
});

const emptyAnomalyList = (page: number, pageSize: number): AnomalyListResponse => ({
  anomalies: [],
  total: 0,
  page,
  page_size: pageSize,
  total_pages: 0,
});

const emptyIsolationForestStats: IsolationForestStats = {
  config: {
    n_estimators: 0,
    contamination: 0,
    random_state: 0,
    scale_features: true,
    min_variance: 0,
    feature_count: 0,
    model_type: 'isolation_forest',
  },
  score_distribution: {
    bins: [],
    total_normal: 0,
    total_anomalies: 0,
    mean_score: 0,
    median_score: 0,
    min_score: 0,
    max_score: 0,
    std_score: 0,
  },
  total_predictions: 0,
  anomaly_rate: 0,
};

// Custom error class for API errors with detailed information
export class APIError extends Error {
  status: number;
  statusText: string;
  body: string;

  constructor(status: number, statusText: string, body: string) {
    super(`API error (${status}): ${body || statusText}`);
    this.name = 'APIError';
    this.status = status;
    this.statusText = statusText;
    this.body = body;
  }
}

async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options?.headers as Record<string, string> || {}),
  };

  // Add X-Mock-Mode header if mock mode is enabled
  // Note: In production, mock mode should be disabled server-side
  if (getMockModeFromStorage()) {
    headers['X-Mock-Mode'] = 'true';
  }

  Object.assign(headers, getAuthHeaders());

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    // Attempt to read response body for detailed error information
    let errorBody = '';
    try {
      errorBody = await response.text();
      // Try to parse as JSON for structured error messages
      const jsonError = JSON.parse(errorBody);
      errorBody = jsonError.detail || jsonError.message || errorBody;
    } catch {
      // Keep raw text if not JSON
    }
    throw new APIError(response.status, response.statusText, errorBody);
  }

  if (response.status === 204) {
    // Some endpoints return 204 with no body (e.g., deletes); avoid JSON parse errors.
    return undefined as T;
  }

  const contentType = response.headers.get('content-type') || '';
  const text = await response.text();
  if (!text) {
    return undefined as T;
  }
  if (!contentType.includes('application/json')) {
    return text as unknown as T;
  }
  return JSON.parse(text) as T;
}

export const api = {
  // Dashboard
  getDashboardStats: (): Promise<DashboardStats> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockDashboardStats());
    }
    if (shouldReturnEmptyLiveData()) {
      return Promise.resolve(emptyDashboardStats());
    }
    return fetchAPI<DashboardStats>('/dashboard/stats');
  },

  getDashboardTrends: (params?: {
    days?: number;
    start_date?: string;
    end_date?: string;
  }): Promise<DashboardTrend[]> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return Promise.resolve(
        getMockDashboardTrends(
          params?.days || 7,
          params?.start_date,
          params?.end_date
        )
      );
    }
    if (shouldReturnEmptyLiveData()) {
      return Promise.resolve([]);
    }

    const queryParams = new URLSearchParams();
    if (params?.start_date) queryParams.append('start_date', params.start_date);
    if (params?.end_date) queryParams.append('end_date', params.end_date);
    if (params?.days && !params.start_date && !params.end_date)
      queryParams.append('days', params.days.toString());
    if (queryParams.toString()) {
      return fetchAPI<DashboardTrend[]>(
        `/dashboard/trends?${queryParams.toString()}`
      );
    }
    return fetchAPI<DashboardTrend[]>('/dashboard/trends?days=7');
  },

  getConnectionStatus: (): Promise<AllConnectionsStatus> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      lastConnectionStatus = null;
      return Promise.resolve(getMockConnectionStatus());
    }
    return fetchAPI<AllConnectionsStatus>('/dashboard/connections').then((data) => {
      lastConnectionStatus = data;
      return data;
    });
  },

  getTroubleshootingAdvice: (
    connectionStatus: AllConnectionsStatus
  ): Promise<TroubleshootingAdvice> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        advice:
          'All systems are operational in Mock Mode. No troubleshooting needed.',
        summary: 'All connections healthy (Mock Mode)',
      });
    }
    return fetchAPI<TroubleshootingAdvice>('/dashboard/connections/troubleshoot', {
      method: 'POST',
      body: JSON.stringify(connectionStatus),
    });
  },

  getLLMDiagnostics: (): Promise<Record<string, unknown>> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        status: 'mock',
        message: 'LLM diagnostics unavailable in mock mode',
      });
    }
    return fetchAPI<Record<string, unknown>>('/dashboard/llm/diagnostics');
  },

  getTroubleshootingCacheStats: (): Promise<Record<string, unknown>> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        total_cached_advice: 0,
        total_times_reused: 0,
        average_reuses_per_advice: 0,
        most_used_advice: [],
      });
    }
    return fetchAPI<Record<string, unknown>>('/dashboard/connections/troubleshoot/cache/stats');
  },

  getDashboardAISummary: (params?: {
    forceRegenerate?: boolean;
  }): Promise<DashboardAISummary> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        summary: 'Fleet health is stable with 3 critical issues requiring attention. Battery drain patterns suggest optimization opportunities at 2 locations.',
        priority_actions: [
          'Investigate battery drain at Westside Mall (12 devices affected)',
          'Review 3 critical anomalies flagged in the last hour',
          'Schedule firmware update for devices showing connectivity issues'
        ],
        health_status: 'degraded' as const,
        generated_at: new Date().toISOString(),
        cached: false,
        based_on: {
          open_cases: 15,
          critical_issues: 3,
          anomalies_today: 7,
          resolved_today: 4,
          devices_monitored: 250
        }
      });
    }
    if (shouldReturnEmptyLiveData()) {
      return Promise.resolve({
        summary: 'No data available. Connect to data sources to see AI-generated insights.',
        priority_actions: [],
        health_status: 'healthy' as const,
        generated_at: new Date().toISOString(),
        cached: false,
        based_on: {
          open_cases: 0,
          critical_issues: 0,
          anomalies_today: 0,
          resolved_today: 0,
          devices_monitored: 0
        }
      });
    }
    const query = params?.forceRegenerate ? '?force_regenerate=true' : '';
    return fetchAPI<DashboardAISummary>(`/dashboard/ai-summary${query}`);
  },

  // Anomalies
  getAnomalies: (params: {
    device_id?: number;
    start_date?: string;
    end_date?: string;
    status?: string;
    min_score?: number;
    max_score?: number;
    page?: number;
    page_size?: number;
  }): Promise<AnomalyListResponse> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return Promise.resolve(
        parseAnomalyList(getMockAnomalies({
          device_id: params.device_id,
          status: params.status,
          page: params.page,
          page_size: params.page_size,
        }))
      );
    }
    if (shouldReturnEmptyLiveData()) {
      const page = params.page || 1;
      const pageSize = params.page_size || 50;
      return Promise.resolve(parseAnomalyList(emptyAnomalyList(page, pageSize)));
    }

    const queryParams = new URLSearchParams();
    if (params.device_id)
      queryParams.append('device_id', params.device_id.toString());
    if (params.start_date) queryParams.append('start_date', params.start_date);
    if (params.end_date) queryParams.append('end_date', params.end_date);
    if (params.status) queryParams.append('status', params.status);
    if (params.min_score !== undefined)
      queryParams.append('min_score', params.min_score.toString());
    if (params.max_score !== undefined)
      queryParams.append('max_score', params.max_score.toString());
    queryParams.append('page', (params.page || 1).toString());
    queryParams.append('page_size', (params.page_size || 50).toString());

    return fetchAPI<AnomalyListResponse>(`/anomalies?${queryParams.toString()}`).then(parseAnomalyList);
  },

  getAnomaly: (id: number): Promise<AnomalyDetail> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      const detail = getMockAnomalyDetail(id);
      if (detail) {
        return Promise.resolve(detail);
      }
      return Promise.reject(new Error('Anomaly not found'));
    }
    return fetchAPI<AnomalyDetail>(`/anomalies/${id}`);
  },

  resolveAnomaly: (
    id: number,
    status: string,
    notes?: string
  ): Promise<Anomaly> => {
    // In mock mode, return a mock response
    if (getMockModeFromStorage()) {
      const detail = getMockAnomalyDetail(id);
      if (detail) {
        return Promise.resolve({
          ...detail,
          status,
        } as Anomaly);
      }
      return Promise.reject(new Error('Anomaly not found'));
    }
    return fetchAPI<Anomaly>(`/anomalies/${id}/resolve`, {
      method: 'POST',
      body: JSON.stringify({ status, notes }),
    });
  },

  addNote: (
    id: number,
    note: string,
    action_type?: string
  ): Promise<{ id: number; message: string }> => {
    // In mock mode, return a mock response
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        id,
        message: 'Note added successfully (mock mode)',
      });
    }
    return fetchAPI<{ id: number; message: string }>(`/anomalies/${id}/notes`, {
      method: 'POST',
      body: JSON.stringify({ note, action_type }),
    });
  },

  // Grouped Anomalies
  getGroupedAnomalies: (params?: {
    status?: string;
    min_severity?: string;
    min_group_size?: number;
    temporal_window_hours?: number;
  }): Promise<GroupedAnomaliesResponse> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return Promise.resolve(parseGroupedAnomalies(getMockGroupedAnomalies(params)));
    }
    if (shouldReturnEmptyLiveData()) {
      return Promise.resolve(parseGroupedAnomalies({
        groups: [],
        total_anomalies: 0,
        total_groups: 0,
        ungrouped_count: 0,
        ungrouped_anomalies: [],
        grouping_method: 'smart_auto',
        computed_at: new Date().toISOString(),
      }));
    }

    const queryParams = new URLSearchParams();
    if (params?.status) queryParams.append('status', params.status);
    if (params?.min_severity) queryParams.append('min_severity', params.min_severity);
    if (params?.min_group_size !== undefined)
      queryParams.append('min_group_size', params.min_group_size.toString());
    if (params?.temporal_window_hours !== undefined)
      queryParams.append('temporal_window_hours', params.temporal_window_hours.toString());

    return fetchAPI<GroupedAnomaliesResponse>(`/anomalies/grouped?${queryParams.toString()}`)
      .then(parseGroupedAnomalies);
  },

  bulkAction: (request: BulkActionRequest): Promise<BulkActionResponse> => {
    // In mock mode, simulate a successful response
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        success: true,
        affected_count: request.anomaly_ids.length,
        failed_ids: [],
        message: `Successfully updated ${request.anomaly_ids.length} anomalies (mock mode)`,
      });
    }
    return fetchAPI<BulkActionResponse>('/anomalies/bulk-action', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  },

  // Devices
  getDevices: (params?: {
    page?: number;
    page_size?: number;
    search?: string;
    group_by?: string;
    group_value?: string;
  }): Promise<DeviceListResponse> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      let devices = getMockDevices();

      if (params?.search) {
        const lowerSearch = params.search.toLowerCase();
        devices = devices.filter((d: DeviceDetail) =>
          d.device_name?.toLowerCase().includes(lowerSearch) ||
          d.device_id.toString().includes(lowerSearch)
        );
      }

      if (params?.group_by && params?.group_value) {
        devices = devices.filter((d: DeviceDetail) => {
          // Direct match on custom attribute
          const attrMatch = d.custom_attributes?.[params.group_by!] === params.group_value;
          // Fallback: If grouping by store-like ID, check store_id directly
          // (This handles case where user renamed 'Store' attribute to something else like 'Zone' but IDs are still store-ids)
          const idMatch = d.store_id === params.group_value;
          return attrMatch || idMatch;
        });
      }

      const page = params?.page || 1;
      const pageSize = params?.page_size || 50;
      const start = (page - 1) * pageSize;
      const end = start + pageSize;

      // Include pagination metadata so list responses align with backend contracts.
      return Promise.resolve({
        devices: devices.slice(start, end),
        total: devices.length,
        page,
        page_size: pageSize,
        total_pages: Math.ceil(devices.length / pageSize),
      });
    }
    if (shouldReturnEmptyLiveData()) {
      const page = params?.page || 1;
      const pageSize = params?.page_size || 50;
      // Include pagination metadata so list responses align with backend contracts.
      return Promise.resolve({ devices: [], total: 0, page, page_size: pageSize, total_pages: 0 });
    }

    const queryParams = new URLSearchParams();
    if (params?.page) queryParams.append('page', params.page.toString());
    if (params?.page_size) queryParams.append('page_size', params.page_size.toString());
    if (params?.search) queryParams.append('search', params.search);
    if (params?.group_by) queryParams.append('group_by', params.group_by);
    if (params?.group_value) queryParams.append('group_value', params.group_value);

    return fetchAPI<DeviceListResponse>(`/devices?${queryParams.toString()}`);
  },

  getDevice: (id: number): Promise<DeviceDetail> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      const detail = getMockDeviceDetail(id);
      if (detail) {
        return Promise.resolve(detail);
      }
      return Promise.reject(new Error('Device not found'));
    }
    return fetchAPI<DeviceDetail>(`/devices/${id}`);
  },

  // Isolation Forest
  getIsolationForestStats: (days?: number): Promise<IsolationForestStats> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return Promise.resolve(parseIsolationForestStats(getMockIsolationForestStats()));
    }
    if (shouldReturnEmptyLiveData()) {
      return Promise.resolve(parseIsolationForestStats(emptyIsolationForestStats));
    }

    const queryParams = new URLSearchParams();
    if (days) queryParams.append('days', days.toString());
    if (queryParams.toString()) {
      return fetchAPI<IsolationForestStats>(
        `/dashboard/isolation-forest/stats?${queryParams.toString()}`
      ).then(parseIsolationForestStats);
    }
    return fetchAPI<IsolationForestStats>('/dashboard/isolation-forest/stats')
      .then(parseIsolationForestStats);
  },

  // Baselines
  getBaselineSuggestions: (
    source?: string,
    days?: number
  ): Promise<BaselineSuggestion[]> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return Promise.resolve(parseBaselineSuggestions(getMockBaselineSuggestions()));
    }

    const queryParams = new URLSearchParams();
    if (source) queryParams.append('source', source);
    if (days) queryParams.append('days', days.toString());
    if (queryParams.toString()) {
      return fetchAPI<BaselineSuggestion[]>(
        `/baselines/suggestions?${queryParams.toString()}`
      ).then(parseBaselineSuggestions);
    }
    return fetchAPI<BaselineSuggestion[]>('/baselines/suggestions')
      .then(parseBaselineSuggestions);
  },

  analyzeBaselinesWithLLM: (
    source?: string,
    days?: number
  ): Promise<BaselineSuggestion[]> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return Promise.resolve(parseBaselineSuggestions(getMockBaselineSuggestions()));
    }

    const queryParams = new URLSearchParams();
    if (source) queryParams.append('source', source);
    if (days) queryParams.append('days', days.toString());
    // Backend expects POST for LLM analysis even when using query params.
    if (queryParams.toString()) {
      return fetchAPI<BaselineSuggestion[]>(
        `/baselines/analyze-with-llm?${queryParams.toString()}`,
        { method: 'POST' }
      ).then(parseBaselineSuggestions);
    }
    return fetchAPI<BaselineSuggestion[]>('/baselines/analyze-with-llm', { method: 'POST' })
      .then(parseBaselineSuggestions);
  },

  applyBaselineAdjustment: (
    request: BaselineAdjustmentRequest,
    source?: string
  ): Promise<BaselineAdjustmentResponse> => {
    // In mock mode, return a mock response
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockBaselineAdjustmentResponse());
    }

    const queryParams = new URLSearchParams();
    if (source) queryParams.append('source', source);
    const url = queryParams.toString()
      ? `/baselines/apply-adjustment?${queryParams.toString()}`
      : '/baselines/apply-adjustment';
    return fetchAPI<BaselineAdjustmentResponse>(url, {
      method: 'POST',
      body: JSON.stringify(request),
    });
  },

  getBaselineFeatures: (
    source?: string,
    days?: number
  ): Promise<BaselineFeature[]> => {
    // In mock mode, return mock baseline features
    if (getMockModeFromStorage()) {
      return Promise.resolve(parseBaselineFeatures([
        { feature: 'BatteryDrop', baseline: 12, observed: 14.2, unit: '%/day', status: 'stable' as const, drift_percent: 18.3, mad: 2.5, sample_count: 1250, last_updated: null },
        { feature: 'OfflineTime', baseline: 30, observed: 42, unit: 'min/day', status: 'drift' as const, drift_percent: 40.0, mad: 8.0, sample_count: 1250, last_updated: null },
        { feature: 'UploadSize', baseline: 500, observed: 520, unit: 'MB/day', status: 'stable' as const, drift_percent: 4.0, mad: 45.0, sample_count: 1250, last_updated: null },
        { feature: 'DownloadSize', baseline: 1200, observed: 1180, unit: 'MB/day', status: 'stable' as const, drift_percent: -1.7, mad: 120.0, sample_count: 1250, last_updated: null },
        { feature: 'StorageFree', baseline: 8.5, observed: 7.2, unit: 'GB', status: 'warning' as const, drift_percent: -15.3, mad: 0.8, sample_count: 1250, last_updated: null },
        { feature: 'AppCrashes', baseline: 0.5, observed: 1.8, unit: '/device/day', status: 'drift' as const, drift_percent: 260.0, mad: 0.3, sample_count: 1250, last_updated: null },
      ]));
    }

    const queryParams = new URLSearchParams();
    if (source) queryParams.append('source', source);
    if (days) queryParams.append('days', days.toString());
    const query = queryParams.toString();
    return fetchAPI<BaselineFeature[]>(`/baselines/features${query ? '?' + query : ''}`)
      .then(parseBaselineFeatures);
  },

  getBaselineHistory: (
    source?: string,
    limit?: number
  ): Promise<BaselineHistoryEntry[]> => {
    // In mock mode, return mock history
    if (getMockModeFromStorage()) {
      const now = new Date();
      return Promise.resolve([
        { id: 1, date: new Date(now.getTime() - 2 * 60 * 60 * 1000).toISOString(), feature: 'BatteryDrop', old_value: 10, new_value: 12, type: 'auto' as const, reason: 'Seasonal adjustment for summer heat' },
        { id: 2, date: new Date(now.getTime() - 24 * 60 * 60 * 1000).toISOString(), feature: 'OfflineTime', old_value: 25, new_value: 30, type: 'manual' as const, reason: 'Network infrastructure upgrade rollout' },
        { id: 3, date: new Date(now.getTime() - 5 * 24 * 60 * 60 * 1000).toISOString(), feature: 'UploadSize', old_value: 450, new_value: 500, type: 'auto' as const, reason: 'New app version increased sync frequency' },
        { id: 4, date: new Date(now.getTime() - 12 * 24 * 60 * 60 * 1000).toISOString(), feature: 'StorageFree', old_value: 10, new_value: 8.5, type: 'manual' as const, reason: 'Lowered threshold per fleet team request' },
      ]);
    }

    const queryParams = new URLSearchParams();
    if (source) queryParams.append('source', source);
    if (limit) queryParams.append('limit', limit.toString());
    const query = queryParams.toString();
    return fetchAPI<BaselineHistoryEntry[]>(`/baselines/history${query ? '?' + query : ''}`);
  },

  // LLM Settings
  getLLMConfig: (): Promise<LLMConfig> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockLLMConfig());
    }
    return fetchAPI<LLMConfig>('/llm/config');
  },

  updateLLMConfig: (update: LLMConfigUpdate): Promise<LLMConfig> => {
    // In mock mode, return a mock response
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        ...getMockLLMConfig(),
        ...update,
      });
    }
    return fetchAPI<LLMConfig>('/llm/config', {
      method: 'POST',
      body: JSON.stringify(update),
    });
  },

  getLLMModels: (): Promise<LLMModelsResponse> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockLLMModels());
    }
    return fetchAPI<LLMModelsResponse>('/llm/models');
  },

  testLLMConnection: (): Promise<LLMTestResult> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockLLMTestResult());
    }
    return fetchAPI<LLMTestResult>('/llm/test', {
      method: 'POST',
    });
  },

  pullOllamaModel: (modelName: string): Promise<OllamaPullResponse> => {
    // In mock mode, return a mock response
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockOllamaPullResponse(modelName));
    }
    return fetchAPI<OllamaPullResponse>('/llm/ollama/pull', {
      method: 'POST',
      body: JSON.stringify({ model_name: modelName }),
    });
  },

  getPopularModels: (): Promise<{ models: PopularModel[] }> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        models: [
          {
            id: 'deepseek/deepseek-r1-0528-qwen3-8b',
            name: 'DeepSeek R1 Qwen 8B',
            description: 'High-quality reasoning model',
            size: '8B',
            category: 'reasoning',
          },
          {
            id: 'llama2:7b',
            name: 'Llama 2 7B',
            description: 'General-purpose language model',
            size: '7B',
            category: 'general',
          },
          {
            id: 'mistral:7b',
            name: 'Mistral 7B',
            description: 'Fast and efficient model',
            size: '7B',
            category: 'general',
          },
        ],
      });
    }
    return fetchAPI<{ models: PopularModel[] }>('/llm/popular-models');
  },

  // Location Heatmap
  getLocationHeatmap: (attributeName?: string): Promise<LocationHeatmapResponse> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockLocationHeatmap(attributeName));
    }
    if (shouldReturnEmptyLiveData()) {
      return Promise.resolve({
        locations: [],
        attributeName: attributeName || 'Location',
        totalLocations: 0,
        totalDevices: 0,
      });
    }

    const queryParams = new URLSearchParams();
    if (attributeName) queryParams.append('attribute_name', attributeName);
    if (queryParams.toString()) {
      return fetchAPI<LocationHeatmapResponse>(
        `/dashboard/location-heatmap?${queryParams.toString()}`
      );
    }
    return fetchAPI<LocationHeatmapResponse>('/dashboard/location-heatmap');
  },

  // Custom Attributes
  getCustomAttributes: (): Promise<{
    custom_attributes: string[];
    error?: string;
  }> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockCustomAttributes());
    }
    if (shouldReturnEmptyLiveData()) {
      return Promise.resolve({ custom_attributes: [] });
    }
    return fetchAPI<{ custom_attributes: string[]; error?: string }>(
      '/dashboard/custom-attributes'
    );
  },

  // ============================================================================
  // Data Discovery
  // ============================================================================

  getTableProfiles: (): Promise<TableProfile[]> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockTableProfiles());
    }
    return fetchAPI<TableProfile[]>('/data-discovery/tables');
  },

  getTableStats: (tableName: string): Promise<TableProfile> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      const profiles = getMockTableProfiles();
      const profile = profiles.find((p) => p.table_name === tableName);
      if (profile) return Promise.resolve(profile);
      return Promise.reject(new Error(`Table '${tableName}' not found`));
    }
    return fetchAPI<TableProfile>(`/data-discovery/tables/${tableName}/stats`);
  },

  getAvailableMetrics: (): Promise<AvailableMetric[]> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockAvailableMetrics());
    }
    return fetchAPI<AvailableMetric[]>('/data-discovery/metrics');
  },

  getMetricDistribution: (
    metricName: string,
    tableName?: string,
    bins?: number
  ): Promise<MetricDistribution> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockMetricDistribution(metricName));
    }
    const queryParams = new URLSearchParams();
    if (tableName) queryParams.append('table_name', tableName);
    if (bins) queryParams.append('bins', bins.toString());
    const query = queryParams.toString();
    return fetchAPI<MetricDistribution>(
      `/data-discovery/metrics/${metricName}/distribution${query ? '?' + query : ''}`
    );
  },

  getTemporalPatterns: (): Promise<TemporalPattern[]> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockTemporalPatterns());
    }
    return fetchAPI<TemporalPattern[]>('/data-discovery/temporal-patterns');
  },

  getDiscoverySummary: (): Promise<DiscoverySummary> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockDiscoverySummary());
    }
    return fetchAPI<DiscoverySummary>('/data-discovery/summary');
  },

  getDiscoveryStatus: (): Promise<DataDiscoveryStatus> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockDiscoveryStatus());
    }
    return fetchAPI<DataDiscoveryStatus>('/data-discovery/status');
  },

  runDataDiscovery: (params?: {
    start_date?: string;
    end_date?: string;
    include_mc?: boolean;
    analyze_patterns?: boolean;
  }): Promise<DataDiscoveryStatus> => {
    // In mock mode, return a simulated running status
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        status: 'running',
        progress: 0,
        message: 'Discovery started (mock mode)',
        started_at: new Date().toISOString(),
        completed_at: null,
        results_available: false,
      });
    }
    const queryParams = new URLSearchParams();
    if (params?.start_date) queryParams.append('start_date', params.start_date);
    if (params?.end_date) queryParams.append('end_date', params.end_date);
    if (params?.include_mc !== undefined)
      queryParams.append('include_mc', params.include_mc.toString());
    if (params?.analyze_patterns !== undefined)
      queryParams.append('analyze_patterns', params.analyze_patterns.toString());
    const query = queryParams.toString();
    return fetchAPI<DataDiscoveryStatus>(`/data-discovery/run${query ? '?' + query : ''}`, {
      method: 'POST',
    });
  },

  // ============================================================================
  // Training
  // ============================================================================

  startTraining: (config: TrainingConfigRequest): Promise<TrainingRun> => {
    // In mock mode, return a simulated pending training run
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockStartTrainingResponse(config));
    }
    return fetchAPI<TrainingRun>('/training/start', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  },

  getTrainingStatus: (): Promise<TrainingRun> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockTrainingStatus());
    }
    return fetchAPI<TrainingRun>('/training/status');
  },

  getTrainingHistory: (limit?: number): Promise<TrainingHistoryResponse> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockTrainingHistory(limit));
    }
    const queryParams = new URLSearchParams();
    if (limit) queryParams.append('limit', limit.toString());
    const query = queryParams.toString();
    return fetchAPI<TrainingHistoryResponse>(
      `/training/history${query ? '?' + query : ''}`
    );
  },

  getTrainingMetrics: (runId: string): Promise<TrainingMetrics> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockTrainingMetrics(runId));
    }
    return fetchAPI<TrainingMetrics>(`/training/${runId}/metrics`);
  },

  getTrainingQueueStatus: (): Promise<TrainingQueueStatus> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        queue_length: 0,
        worker_available: true,
        last_job_completed_at: new Date().toISOString(),
        next_scheduled: null,
      });
    }
    return fetchAPI<TrainingQueueStatus>('/training/queue');
  },

  getTrainingArtifacts: (runId: string): Promise<TrainingArtifacts> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        model_path: 'models/production/isolation_forest.pkl',
        onnx_path: 'models/production/isolation_forest.onnx',
        baselines_path: 'models/production/baselines.json',
        metadata_path: 'models/production/training_metadata.json',
      });
    }
    return fetchAPI<TrainingArtifacts>(`/training/${runId}/artifacts`);
  },

  // ============================================================================
  // Automation
  // ============================================================================

  getAutomationConfig: (): Promise<SchedulerConfig> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockAutomationConfig());
    }
    return fetchAPI<SchedulerConfig>('/automation/config');
  },

  updateAutomationConfig: (config: SchedulerConfigUpdate): Promise<SchedulerConfig> => {
    // In mock mode, return updated mock config
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        ...getMockAutomationConfig(),
        ...config,
      });
    }
    return fetchAPI<SchedulerConfig>('/automation/config', {
      method: 'PUT',
      body: JSON.stringify(config),
    });
  },

  getAutomationStatus: (): Promise<SchedulerStatus> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockAutomationStatus());
    }
    return fetchAPI<SchedulerStatus>('/automation/status');
  },

  triggerAutomationJob: (request: TriggerJobRequest): Promise<TriggerJobResponse> => {
    // In mock mode, return mock response
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockTriggerJobResponse(request.job_type));
    }
    return fetchAPI<TriggerJobResponse>('/automation/trigger', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  },

  scoreDevices: (request: ScoreRequest): Promise<ScoreResponse> => {
    // In mock mode, return mock scoring response
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        success: true,
        total_scored: 248,
        anomalies_detected: 12,
        anomaly_rate: 0.048,
        results: null,
      });
    }
    return fetchAPI<ScoreResponse>('/automation/score', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  },

  getAutomationAlerts: (limit?: number, unacknowledged_only?: boolean): Promise<AutomationAlert[]> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      let alerts = getMockAutomationAlerts(limit);
      if (unacknowledged_only) {
        alerts = alerts.filter(a => !a.acknowledged);
      }
      return Promise.resolve(alerts);
    }
    const queryParams = new URLSearchParams();
    if (limit) queryParams.append('limit', limit.toString());
    // Backend expects an acknowledged flag; map the UI filter to acknowledged=false.
    if (unacknowledged_only) queryParams.append('acknowledged', 'false');
    const query = queryParams.toString();
    return fetchAPI<AutomationAlert[]>(`/automation/alerts${query ? '?' + query : ''}`);
  },

  acknowledgeAlert: (alertId: string): Promise<{ success: boolean }> => {
    // In mock mode, return success
    if (getMockModeFromStorage()) {
      return Promise.resolve({ success: true });
    }
    return fetchAPI<{ success: boolean }>(`/automation/alerts/${alertId}/acknowledge`, {
      method: 'POST',
    });
  },

  syncDeviceMetadata: (): Promise<{
    success: boolean;
    synced_count: number;
    duration_seconds: number;
    message: string;
    errors: string[];
  }> => {
    // In mock mode, return mock response
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        success: true,
        synced_count: 0,
        duration_seconds: 0,
        message: 'Mock mode enabled - sync skipped',
        errors: [],
      });
    }
    return fetchAPI<{
      success: boolean;
      synced_count: number;
      duration_seconds: number;
      message: string;
      errors: string[];
    }>('/automation/sync-device-metadata', {
      method: 'POST',
    });
  },

  getAutomationHistory: (limit?: number, job_type?: string): Promise<AutomationHistory> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      let history = getMockAutomationHistory(limit);
      if (job_type) {
        history = history.filter(j => j.type === job_type);
      }
      return Promise.resolve(history);
    }
    const queryParams = new URLSearchParams();
    if (limit) queryParams.append('limit', limit.toString());
    if (job_type) queryParams.append('job_type', job_type);
    const query = queryParams.toString();
    return fetchAPI<AutomationHistory>(`/automation/history${query ? '?' + query : ''}`);
  },

  // ============================================================================
  // Setup / Configuration
  // ============================================================================

  testSetupConnection: (request: {
    type: 'backend_db' | 'dw_db' | 'mc_db' | 'llm' | 'redis' | 'mobicontrol_api';
    config: Record<string, unknown>;
  }): Promise<{ success: boolean; message: string; latency_ms?: number }> => {
    // Always test real connections - users need actual feedback
    // regardless of mock mode (mock mode is for device/anomaly data, not setup)
    return fetchAPI<{ success: boolean; message: string; latency_ms?: number }>(
      '/setup/test-connection',
      {
        method: 'POST',
        body: JSON.stringify(request),
      }
    );
  },

  saveSetupConfig: (
    config: Record<string, unknown>
  ): Promise<{ success: boolean; message: string; file_path?: string }> => {
    // Always save real config - users need actual changes to persist
    // regardless of mock mode (mock mode is for device/anomaly data, not setup)
    return fetchAPI<{ success: boolean; message: string; file_path?: string }>(
      '/setup/save-config',
      {
        method: 'POST',
        body: JSON.stringify(config),
      }
    );
  },

  getSetupConfig: (): Promise<Record<string, unknown>> => {
    // Always fetch actual config - Setup Wizard should show real settings
    // regardless of mock mode (mock mode is for device/anomaly data, not setup)
    return fetchAPI<Record<string, unknown>>('/setup/config');
  },

  syncLocations: (): Promise<{ success: boolean; message: string }> => {
    return fetchAPI<{ success: boolean; message: string }>('/setup/sync-locations', {
      method: 'POST',
      body: JSON.stringify({}),
    });
  },

  getLocationSyncStats: (): Promise<Record<string, unknown>> => {
    return fetchAPI<Record<string, unknown>>('/setup/location-sync-stats');
  },

  // ============================================================================
  // Investigation Panel
  // ============================================================================

  getInvestigationPanel: (
    anomalyId: number,
    options?: { includeAiAnalysis?: boolean; includeSimilarCases?: boolean }
  ): Promise<InvestigationPanel> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return import('./mockData').then((m) => m.getMockInvestigationPanel(anomalyId));
    }
    const queryParams = new URLSearchParams();
    if (options?.includeAiAnalysis !== undefined) {
      queryParams.append('include_ai_analysis', options.includeAiAnalysis.toString());
    }
    if (options?.includeSimilarCases !== undefined) {
      queryParams.append('include_similar_cases', options.includeSimilarCases.toString());
    }
    const query = queryParams.toString();
    return fetchAPI<InvestigationPanel>(
      `/anomalies/${anomalyId}/investigation${query ? '?' + query : ''}`
    );
  },

  getAnomalyExplanation: (anomalyId: number): Promise<AnomalyExplanation> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return import('./mockData').then((m) => m.getMockInvestigationPanel(anomalyId)).then((p) => p.explanation);
    }
    return fetchAPI<AnomalyExplanation>(`/anomalies/${anomalyId}/explanation`);
  },

  getBaselineComparison: (
    anomalyId: number,
    baselineDays?: number
  ): Promise<BaselineComparison> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return import('./mockData').then((m) => m.getMockInvestigationPanel(anomalyId)).then((p) => {
        if (!p.baseline_comparison) {
          throw new Error('No baseline data available');
        }
        return p.baseline_comparison;
      });
    }
    const queryParams = new URLSearchParams();
    if (baselineDays) queryParams.append('baseline_days', baselineDays.toString());
    const query = queryParams.toString();
    return fetchAPI<BaselineComparison>(
      `/anomalies/${anomalyId}/baseline${query ? '?' + query : ''}`
    );
  },

  getHistoricalTimeline: (
    anomalyId: number,
    metric: string,
    days?: number
  ): Promise<HistoricalTimeline> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return import('./mockData').then((m) => m.getMockHistoricalTimeline(anomalyId, metric, days ?? 7));
    }
    const queryParams = new URLSearchParams();
    queryParams.append('metric', metric);
    if (days) queryParams.append('days', days.toString());
    return fetchAPI<HistoricalTimeline>(
      `/anomalies/${anomalyId}/timeline?${queryParams.toString()}`
    );
  },

  getAIAnalysis: (anomalyId: number, regenerate?: boolean): Promise<AIAnalysis> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return import('./mockData').then((m) => m.getMockAIAnalysis(anomalyId));
    }
    const queryParams = new URLSearchParams();
    if (regenerate) queryParams.append('regenerate', 'true');
    const query = queryParams.toString();
    return fetchAPI<AIAnalysis>(
      `/anomalies/${anomalyId}/ai-analysis${query ? '?' + query : ''}`
    );
  },

  submitAIFeedback: (
    anomalyId: number,
    feedback: AIAnalysisFeedback
  ): Promise<{ success: boolean; message: string }> => {
    // In mock mode, return success
    if (getMockModeFromStorage()) {
      return Promise.resolve({ success: true, message: 'Feedback recorded (mock mode)' });
    }
    return fetchAPI<{ success: boolean; message: string }>(
      `/anomalies/${anomalyId}/ai-analysis/feedback`,
      {
        method: 'POST',
        body: JSON.stringify(feedback),
      }
    );
  },

  getSimilarCases: (anomalyId: number, limit?: number): Promise<SimilarCase[]> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return import('./mockData').then((m) => m.getMockInvestigationPanel(anomalyId)).then((p) => p.similar_cases);
    }
    const queryParams = new URLSearchParams();
    if (limit) queryParams.append('limit', limit.toString());
    const query = queryParams.toString();
    return fetchAPI<SimilarCase[]>(
      `/anomalies/${anomalyId}/similar-cases${query ? '?' + query : ''}`
    );
  },

  getRemediations: (anomalyId: number): Promise<RemediationSuggestion[]> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return import('./mockData').then((m) => m.getMockInvestigationPanel(anomalyId)).then((p) => p.suggested_remediations);
    }
    return fetchAPI<RemediationSuggestion[]>(`/anomalies/${anomalyId}/remediations`);
  },

  recordRemediationOutcome: (
    anomalyId: number,
    remediationId: string,
    outcome: RemediationOutcome
  ): Promise<{ success: boolean; message: string; outcome_id?: number }> => {
    // In mock mode, return success
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        success: true,
        message: 'Outcome recorded (mock mode)',
        outcome_id: Math.floor(Math.random() * 1000),
      });
    }
    return fetchAPI<{ success: boolean; message: string; outcome_id?: number }>(
      `/anomalies/${anomalyId}/remediations/${remediationId}/outcome`,
      {
        method: 'POST',
        body: JSON.stringify(outcome),
      }
    );
  },

  learnFromFix: (
    anomalyId: number,
    fix: LearnFromFix
  ): Promise<{
    success: boolean;
    message: string;
    learned_remediation_id?: number;
    initial_confidence?: number;
    current_confidence?: number;
  }> => {
    // In mock mode, return success
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        success: true,
        message: 'New remediation pattern learned (mock mode)',
        learned_remediation_id: Math.floor(Math.random() * 1000),
        initial_confidence: 0.6,
      });
    }
    return fetchAPI<{
      success: boolean;
      message: string;
      learned_remediation_id?: number;
      initial_confidence?: number;
      current_confidence?: number;
    }>(`/anomalies/${anomalyId}/learn`, {
      method: 'POST',
      body: JSON.stringify(fix),
    });
  },

  // =========================================================================
  // Device Control Actions (SOTI MobiControl)
  // =========================================================================

  lockDevice: (deviceId: number, reason?: string): Promise<DeviceActionResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        success: true,
        action: 'lock',
        device_id: deviceId,
        message: `Device ${deviceId} locked successfully (mock mode)`,
        action_id: `mock-${Date.now()}`,
        timestamp: new Date().toISOString(),
      });
    }
    return fetchAPI<DeviceActionResponse>(`/devices/${deviceId}/actions/lock`, {
      method: 'POST',
      body: JSON.stringify({ reason }),
    });
  },

  restartDevice: (deviceId: number, reason?: string): Promise<DeviceActionResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        success: true,
        action: 'restart',
        device_id: deviceId,
        message: `Device ${deviceId} restart initiated (mock mode)`,
        action_id: `mock-${Date.now()}`,
        timestamp: new Date().toISOString(),
      });
    }
    return fetchAPI<DeviceActionResponse>(`/devices/${deviceId}/actions/restart`, {
      method: 'POST',
      body: JSON.stringify({ reason }),
    });
  },

  wipeDevice: (
    deviceId: number,
    factoryReset: boolean = false,
    confirm: boolean = false,
    reason?: string
  ): Promise<DeviceActionResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        success: true,
        action: factoryReset ? 'factory_reset' : 'wipe',
        device_id: deviceId,
        message: `Device ${deviceId} wipe ${confirm ? 'confirmed' : 'initiated'} (mock mode)`,
        action_id: `mock-${Date.now()}`,
        timestamp: new Date().toISOString(),
      });
    }
    return fetchAPI<DeviceActionResponse>(`/devices/${deviceId}/actions/wipe`, {
      method: 'POST',
      body: JSON.stringify({ factory_reset: factoryReset, confirm, reason }),
    });
  },

  sendMessageToDevice: (
    deviceId: number,
    message: string,
    title?: string
  ): Promise<DeviceActionResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        success: true,
        action: 'message',
        device_id: deviceId,
        message: `Message sent to device ${deviceId} (mock mode)`,
        action_id: `mock-${Date.now()}`,
        timestamp: new Date().toISOString(),
      });
    }
    return fetchAPI<DeviceActionResponse>(`/devices/${deviceId}/actions/message`, {
      method: 'POST',
      body: JSON.stringify({ message, title: title || 'Message from Admin' }),
    });
  },

  locateDevice: (deviceId: number): Promise<DeviceLocationResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        success: true,
        device_id: deviceId,
        latitude: 52.3676 + (Math.random() - 0.5) * 0.1,
        longitude: 4.9041 + (Math.random() - 0.5) * 0.1,
        accuracy: Math.floor(Math.random() * 50) + 10,
        timestamp: new Date().toISOString(),
        message: `Device ${deviceId} located (mock mode)`,
      });
    }
    return fetchAPI<DeviceLocationResponse>(`/devices/${deviceId}/actions/locate`, {
      method: 'POST',
    });
  },

  syncDevice: (deviceId: number, reason?: string): Promise<DeviceActionResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        success: true,
        action: 'sync',
        device_id: deviceId,
        message: `Device ${deviceId} sync initiated (mock mode)`,
        action_id: `mock-${Date.now()}`,
        timestamp: new Date().toISOString(),
      });
    }
    return fetchAPI<DeviceActionResponse>(`/devices/${deviceId}/actions/sync`, {
      method: 'POST',
      body: JSON.stringify({ reason }),
    });
  },

  clearDeviceCache: (
    deviceId: number,
    packageName?: string
  ): Promise<DeviceActionResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        success: true,
        action: 'clear_cache',
        device_id: deviceId,
        message: `Cache cleared for device ${deviceId}${packageName ? ` (${packageName})` : ''} (mock mode)`,
        action_id: `mock-${Date.now()}`,
        timestamp: new Date().toISOString(),
      });
    }
    return fetchAPI<DeviceActionResponse>(`/devices/${deviceId}/actions/clear-cache`, {
      method: 'POST',
      body: packageName ? JSON.stringify({ package_name: packageName }) : undefined,
    });
  },

  getDeviceActionHistory: (
    deviceId: number,
    limit?: number
  ): Promise<DeviceActionHistoryResponse> => {
    if (getMockModeFromStorage()) {
      const actions = [
        { id: 1, action_type: 'sync', initiated_by: 'admin@company.com', success: true, timestamp: new Date(Date.now() - 3600000).toISOString() },
        { id: 2, action_type: 'lock', initiated_by: 'system', reason: 'Security policy', success: true, timestamp: new Date(Date.now() - 7200000).toISOString() },
        { id: 3, action_type: 'message', initiated_by: 'admin@company.com', success: true, timestamp: new Date(Date.now() - 86400000).toISOString() },
      ].slice(0, limit || 10);
      return Promise.resolve({
        device_id: deviceId,
        total: actions.length,
        actions,
      });
    }
    const params = limit ? `?limit=${limit}` : '';
    return fetchAPI<DeviceActionHistoryResponse>(`/devices/${deviceId}/actions/history${params}`);
  },

  // ============================================================================
  // INSIGHTS API
  // Customer-facing insights aligned with Carl's vision:
  // "XSight has the data. XSight needs the story."
  // ============================================================================

  getDailyDigest: (params?: {
    digest_date?: string;
    period_days?: number;
  }): Promise<DailyDigestResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockDailyDigest());
    }
    const queryParams = new URLSearchParams();
    if (params?.digest_date) queryParams.append('digest_date', params.digest_date);
    if (params?.period_days) queryParams.append('period_days', params.period_days.toString());
    const query = queryParams.toString() ? `?${queryParams.toString()}` : '';
    return fetchAPI<DailyDigestResponse>(`/insights/daily-digest${query}`);
  },

  getLocationInsights: (
    locationId: string,
    periodDays?: number
  ): Promise<LocationInsightResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockLocationInsights(locationId));
    }
    const params = periodDays ? `?period_days=${periodDays}` : '';
    return fetchAPI<LocationInsightResponse>(`/insights/location/${locationId}${params}`);
  },

  getShiftReadiness: (
    locationId: string,
    shiftDate?: string,
    shiftName?: string
  ): Promise<ShiftReadinessResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockShiftReadiness(locationId));
    }
    const queryParams = new URLSearchParams();
    if (shiftDate) queryParams.append('shift_date', shiftDate);
    if (shiftName) queryParams.append('shift_name', shiftName);
    const query = queryParams.toString() ? `?${queryParams.toString()}` : '';
    return fetchAPI<ShiftReadinessResponse>(`/insights/location/${locationId}/shift-readiness${query}`);
  },

  getTrendingInsights: (
    lookbackDays?: number,
    limit?: number
  ): Promise<CustomerInsightResponse[]> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockTrendingInsights());
    }
    const queryParams = new URLSearchParams();
    if (lookbackDays) queryParams.append('lookback_days', lookbackDays.toString());
    if (limit) queryParams.append('limit', limit.toString());
    const query = queryParams.toString() ? `?${queryParams.toString()}` : '';
    return fetchAPI<CustomerInsightResponse[]>(`/insights/trending${query}`);
  },

  getNetworkAnalysis: (
    locationId?: string,
    periodDays?: number
  ): Promise<NetworkAnalysisResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockNetworkAnalysis());
    }
    const queryParams = new URLSearchParams();
    if (locationId) queryParams.append('location_id', locationId);
    if (periodDays) queryParams.append('period_days', periodDays.toString());
    const query = queryParams.toString() ? `?${queryParams.toString()}` : '';
    return fetchAPI<NetworkAnalysisResponse>(`/insights/network/analysis${query}`);
  },

  getDeviceAbuseAnalysis: (periodDays?: number): Promise<DeviceAbuseResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockDeviceAbuseAnalysis());
    }
    const params = periodDays ? `?period_days=${periodDays}` : '';
    return fetchAPI<DeviceAbuseResponse>(`/insights/device-abuse${params}`);
  },

  getAppAnalysis: (periodDays?: number): Promise<AppAnalysisResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockAppAnalysis());
    }
    const params = periodDays ? `?period_days=${periodDays}` : '';
    return fetchAPI<AppAnalysisResponse>(`/insights/apps/power-hungry${params}`);
  },

  getInsightsByCategory: (
    category: string,
    periodDays?: number,
    severity?: string,
    limit?: number
  ): Promise<CustomerInsightResponse[]> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockInsightsByCategory(category));
    }
    const queryParams = new URLSearchParams();
    if (periodDays) queryParams.append('period_days', periodDays.toString());
    if (severity) queryParams.append('severity', severity);
    if (limit) queryParams.append('limit', limit.toString());
    const query = queryParams.toString() ? `?${queryParams.toString()}` : '';
    return fetchAPI<CustomerInsightResponse[]>(`/insights/by-category/${category}${query}`);
  },

  compareLocations: (
    locationAId: string,
    locationBId: string,
    metrics?: string[],
    periodDays?: number
  ): Promise<LocationCompareResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockLocationComparison(locationAId, locationBId));
    }
    const queryParams = new URLSearchParams();
    if (periodDays) queryParams.append('period_days', periodDays.toString());
    const query = queryParams.toString() ? `?${queryParams.toString()}` : '';
    return fetchAPI<LocationCompareResponse>(`/insights/compare/locations${query}`, {
      method: 'POST',
      body: JSON.stringify({
        location_a_id: locationAId,
        location_b_id: locationBId,
        metrics,
      }),
    });
  },

  // ============================================================================
  // COST INTELLIGENCE API
  // Enables financial impact analysis for device anomalies and fleet insights.
  // ============================================================================

  // Hardware Costs
  getHardwareCosts: (params?: {
    page?: number;
    page_size?: number;
    search?: string;
  }): Promise<HardwareCostListResponse> => {
    if (getMockModeFromStorage()) {
      const mockHardwareCosts: HardwareCostResponse[] = [
        {
          id: 1, tenant_id: 'demo', device_model: 'Zebra TC52', purchase_cost: 599,
          replacement_cost: 649, repair_cost_avg: 125, depreciation_months: 36,
          battery_replacement_cost: 45, battery_lifespan_months: 18,
          residual_value_percent: 15, warranty_months: 12, currency_code: 'USD',
          notes: 'Enterprise-grade Android device for warehouse operations',
          device_count: 120, total_fleet_value: 71880, valid_from: '2024-01-15T00:00:00Z',
          created_at: '2024-01-15T10:30:00Z', updated_at: '2024-06-20T14:15:00Z',
        },
        {
          id: 2, tenant_id: 'demo', device_model: 'Zebra TC75x', purchase_cost: 899,
          replacement_cost: 949, repair_cost_avg: 175, depreciation_months: 36,
          battery_replacement_cost: 55, battery_lifespan_months: 24,
          residual_value_percent: 10, warranty_months: 24, currency_code: 'USD',
          notes: 'Rugged touchscreen scanner for logistics',
          device_count: 85, total_fleet_value: 76415, valid_from: '2024-02-01T00:00:00Z',
          created_at: '2024-02-01T09:00:00Z', updated_at: '2024-05-15T11:30:00Z',
        },
        {
          id: 3, tenant_id: 'demo', device_model: 'Honeywell CK65', purchase_cost: 549,
          replacement_cost: 599, repair_cost_avg: 110, depreciation_months: 48,
          battery_replacement_cost: 65, battery_lifespan_months: 24,
          residual_value_percent: 20, warranty_months: 12, currency_code: 'USD',
          notes: 'Cold storage rated mobile computer',
          device_count: 45, total_fleet_value: 24705, valid_from: '2024-03-10T00:00:00Z',
          created_at: '2024-03-10T08:45:00Z', updated_at: '2024-07-01T16:00:00Z',
        },
        {
          id: 4, tenant_id: 'demo', device_model: 'Samsung Galaxy XCover 6', purchase_cost: 449,
          replacement_cost: 479, repair_cost_avg: 89, depreciation_months: 24,
          battery_replacement_cost: 35, battery_lifespan_months: 18,
          residual_value_percent: 25, warranty_months: 12, currency_code: 'USD',
          notes: 'Rugged smartphone for field workers',
          device_count: 62, total_fleet_value: 27838, valid_from: '2024-04-01T00:00:00Z',
          created_at: '2024-04-01T14:00:00Z', updated_at: '2024-08-10T09:30:00Z',
        },
        {
          id: 5, tenant_id: 'demo', device_model: 'iPad Pro 11', purchase_cost: 799,
          replacement_cost: 849, repair_cost_avg: 250, depreciation_months: 36,
          battery_replacement_cost: 129, battery_lifespan_months: 24,
          residual_value_percent: 30, warranty_months: 12, currency_code: 'USD',
          notes: 'Customer-facing kiosk and POS device (non-removable battery)',
          device_count: 28, total_fleet_value: 22372, valid_from: '2024-01-20T00:00:00Z',
          created_at: '2024-01-20T11:00:00Z', updated_at: '2024-06-05T13:45:00Z',
        },
        {
          id: 6, tenant_id: 'demo', device_model: 'Panasonic Toughbook N1', purchase_cost: 1299,
          replacement_cost: 1399, repair_cost_avg: 320, depreciation_months: 48,
          battery_replacement_cost: 89, battery_lifespan_months: 30,
          residual_value_percent: 15, warranty_months: 36, currency_code: 'USD',
          notes: 'Extreme environment handheld for outdoor use',
          device_count: 18, total_fleet_value: 23382, valid_from: '2024-05-15T00:00:00Z',
          created_at: '2024-05-15T10:00:00Z', updated_at: '2024-09-01T08:00:00Z',
        },
      ];
      return Promise.resolve({
        costs: mockHardwareCosts,
        total: mockHardwareCosts.length,
        page: 1,
        page_size: 50,
        total_pages: 1,
        total_fleet_value: mockHardwareCosts.reduce((sum, c) => sum + c.total_fleet_value, 0),
      });
    }
    const queryParams = new URLSearchParams();
    if (params?.page) queryParams.append('page', params.page.toString());
    if (params?.page_size) queryParams.append('page_size', params.page_size.toString());
    if (params?.search) queryParams.append('search', params.search);
    const query = queryParams.toString() ? `?${queryParams.toString()}` : '';
    return fetchAPI<HardwareCostListResponse>(`/costs/hardware${query}`);
  },

  getDeviceModelTypes: (): Promise<DeviceModelsResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        models: [
          { device_model: 'Zebra TC52', device_count: 120, has_cost_entry: true },
          { device_model: 'Zebra TC75x', device_count: 85, has_cost_entry: true },
          { device_model: 'Honeywell CK65', device_count: 45, has_cost_entry: true },
          { device_model: 'Samsung Galaxy XCover 6', device_count: 62, has_cost_entry: true },
          { device_model: 'iPad Pro 11', device_count: 28, has_cost_entry: true },
          { device_model: 'Panasonic Toughbook N1', device_count: 18, has_cost_entry: true },
          { device_model: 'Zebra ET51', device_count: 12, has_cost_entry: false },
        ],
        total: 7,
      });
    }
    return fetchAPI<DeviceModelsResponse>('/costs/hardware/types');
  },

  createHardwareCost: (cost: HardwareCostCreate): Promise<HardwareCostResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        ...cost,
        id: Math.floor(Math.random() * 1000),
        tenant_id: 'mock',
        device_count: 0,
        total_fleet_value: 0,
        valid_from: new Date().toISOString(),
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      });
    }
    return fetchAPI<HardwareCostResponse>('/costs/hardware', {
      method: 'POST',
      body: JSON.stringify(cost),
    });
  },

  updateHardwareCost: (id: number, cost: HardwareCostUpdate): Promise<HardwareCostResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.reject(new Error('Update not available in mock mode'));
    }
    return fetchAPI<HardwareCostResponse>(`/costs/hardware/${id}`, {
      method: 'PUT',
      body: JSON.stringify(cost),
    });
  },

  deleteHardwareCost: (id: number): Promise<void> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve();
    }
    return fetchAPI<void>(`/costs/hardware/${id}`, { method: 'DELETE' });
  },

  // Operational Costs
  getOperationalCosts: (params?: {
    page?: number;
    page_size?: number;
    category?: string;
    is_active?: boolean;
  }): Promise<OperationalCostListResponse> => {
    if (getMockModeFromStorage()) {
      const mockOperationalCosts: OperationalCostResponse[] = [
        {
          id: 1, tenant_id: 'demo', name: 'Warehouse Worker Hourly Rate', category: 'labor',
          amount: 2500, cost_type: 'hourly', unit: 'hour', scope_type: 'tenant',
          description: 'Average hourly cost including benefits for warehouse staff',
          currency_code: 'USD', is_active: true, valid_from: '2024-01-01T00:00:00Z',
          monthly_equivalent: 4333, annual_equivalent: 52000,
          created_at: '2024-01-01T00:00:00Z', updated_at: '2024-06-01T00:00:00Z',
        },
        {
          id: 2, tenant_id: 'demo', name: 'IT Support Hourly Rate', category: 'support',
          amount: 7500, cost_type: 'hourly', unit: 'hour', scope_type: 'tenant',
          description: 'IT helpdesk and device support labor rate',
          currency_code: 'USD', is_active: true, valid_from: '2024-01-01T00:00:00Z',
          monthly_equivalent: 13000, annual_equivalent: 156000,
          created_at: '2024-01-01T00:00:00Z', updated_at: '2024-03-15T00:00:00Z',
        },
        {
          id: 3, tenant_id: 'demo', name: 'Device Downtime Cost', category: 'downtime',
          amount: 15000, cost_type: 'hourly', unit: 'hour', scope_type: 'tenant',
          description: 'Lost productivity cost per hour of device downtime',
          currency_code: 'USD', is_active: true, valid_from: '2024-01-01T00:00:00Z',
          monthly_equivalent: 26000, annual_equivalent: 312000,
          created_at: '2024-01-01T00:00:00Z', updated_at: '2024-02-20T00:00:00Z',
        },
        {
          id: 4, tenant_id: 'demo', name: 'MDM Software License', category: 'infrastructure',
          amount: 800, cost_type: 'per_device', unit: 'device/year', scope_type: 'tenant',
          description: 'SOTI MobiControl annual license per device',
          currency_code: 'USD', is_active: true, valid_from: '2024-01-01T00:00:00Z',
          monthly_equivalent: 2400, annual_equivalent: 28800,
          created_at: '2024-01-01T00:00:00Z', updated_at: '2024-04-10T00:00:00Z',
        },
        {
          id: 5, tenant_id: 'demo', name: 'Quarterly Device Maintenance', category: 'maintenance',
          amount: 2500, cost_type: 'per_device', unit: 'device/quarter', scope_type: 'tenant',
          description: 'Preventive maintenance including cleaning, battery check, and updates',
          currency_code: 'USD', is_active: true, valid_from: '2024-01-01T00:00:00Z',
          monthly_equivalent: 3000, annual_equivalent: 36000,
          created_at: '2024-01-01T00:00:00Z', updated_at: '2024-05-05T00:00:00Z',
        },
        {
          id: 6, tenant_id: 'demo', name: 'Battery Replacement Cost', category: 'maintenance',
          amount: 4500, cost_type: 'per_incident', unit: 'replacement', scope_type: 'tenant',
          description: 'Average cost to replace device battery including labor',
          currency_code: 'USD', is_active: true, valid_from: '2024-01-01T00:00:00Z',
          monthly_equivalent: 1350, annual_equivalent: 16200,
          created_at: '2024-01-01T00:00:00Z', updated_at: '2024-07-20T00:00:00Z',
        },
        {
          id: 7, tenant_id: 'demo', name: 'Device Replacement Labor', category: 'labor',
          amount: 3500, cost_type: 'per_incident', unit: 'replacement', scope_type: 'tenant',
          description: 'IT labor cost to provision and deploy replacement device',
          currency_code: 'USD', is_active: true, valid_from: '2024-02-01T00:00:00Z',
          monthly_equivalent: 1050, annual_equivalent: 12600,
          created_at: '2024-02-01T00:00:00Z', updated_at: '2024-08-01T00:00:00Z',
        },
        {
          id: 8, tenant_id: 'demo', name: 'Network Infrastructure', category: 'infrastructure',
          amount: 250000, cost_type: 'fixed_monthly', unit: 'month', scope_type: 'tenant',
          description: 'WiFi access points, switches, and network management',
          currency_code: 'USD', is_active: true, valid_from: '2024-01-01T00:00:00Z',
          monthly_equivalent: 2500, annual_equivalent: 30000,
          created_at: '2024-01-01T00:00:00Z', updated_at: '2024-06-15T00:00:00Z',
        },
      ];
      return Promise.resolve({
        costs: mockOperationalCosts,
        total: mockOperationalCosts.length,
        page: 1,
        page_size: 50,
        total_pages: 1,
        total_monthly_cost: mockOperationalCosts.reduce((sum, c) => sum + c.monthly_equivalent, 0),
        total_annual_cost: mockOperationalCosts.reduce((sum, c) => sum + c.annual_equivalent, 0),
      });
    }
    const queryParams = new URLSearchParams();
    if (params?.page) queryParams.append('page', params.page.toString());
    if (params?.page_size) queryParams.append('page_size', params.page_size.toString());
    if (params?.category) queryParams.append('category', params.category);
    if (params?.is_active !== undefined) queryParams.append('is_active', params.is_active.toString());
    const query = queryParams.toString() ? `?${queryParams.toString()}` : '';
    return fetchAPI<OperationalCostListResponse>(`/costs/operational${query}`);
  },

  createOperationalCost: (cost: OperationalCostCreate): Promise<OperationalCostResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        ...cost,
        id: Math.floor(Math.random() * 1000),
        tenant_id: 'mock',
        valid_from: new Date().toISOString(),
        monthly_equivalent: cost.amount,
        annual_equivalent: cost.amount * 12,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      });
    }
    return fetchAPI<OperationalCostResponse>('/costs/operational', {
      method: 'POST',
      body: JSON.stringify(cost),
    });
  },

  updateOperationalCost: (id: number, cost: OperationalCostUpdate): Promise<OperationalCostResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.reject(new Error('Update not available in mock mode'));
    }
    return fetchAPI<OperationalCostResponse>(`/costs/operational/${id}`, {
      method: 'PUT',
      body: JSON.stringify(cost),
    });
  },

  deleteOperationalCost: (id: number): Promise<void> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve();
    }
    return fetchAPI<void>(`/costs/operational/${id}`, { method: 'DELETE' });
  },

  // Cost Summary & Impact
  getCostSummary: (period?: string): Promise<CostSummaryResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        tenant_id: 'mock',
        summary_period: period || 'current_month',
        total_hardware_value: 125000,
        total_operational_monthly: 4500,
        total_operational_annual: 54000,
        total_anomaly_impact_mtd: 2340,
        total_anomaly_impact_ytd: 28500,
        by_category: [
          { category: 'labor', total_cost: 2000, item_count: 3, percentage_of_total: 44.4 },
          { category: 'downtime', total_cost: 1500, item_count: 2, percentage_of_total: 33.3 },
          { category: 'support', total_cost: 1000, item_count: 2, percentage_of_total: 22.2 },
        ],
        by_device_model: [
          { device_model: 'Zebra TC52', device_count: 120, unit_cost: 500, total_value: 60000 },
          { device_model: 'Zebra TC75x', device_count: 80, unit_cost: 650, total_value: 52000 },
          { device_model: 'Honeywell CK65', device_count: 25, unit_cost: 520, total_value: 13000 },
        ],
        cost_trend_30d: -5.2,
        anomaly_cost_trend_30d: 12.5,
        calculated_at: new Date().toISOString(),
        device_count: 225,
        anomaly_count_period: 47,
      });
    }
    const params = period ? `?period=${period}` : '';
    return fetchAPI<CostSummaryResponse>(`/costs/summary${params}`);
  },

  getAnomalyImpact: (anomalyId: number): Promise<AnomalyImpactResponse> => {
    if (getMockModeFromStorage()) {
      // Generate varied impact scenarios based on anomaly ID for demo variety
      const scenarios = [
        {
          // High impact: Battery failure causing shift disruption
          severity: 'high' as const,
          total: 2850,
          components: [
            { type: 'Worker Productivity Loss', description: 'Picker unable to fulfill orders for 4.5 hours', amount: 1125, calculation_method: '4.5 hours x $250/hr avg order value', confidence: 0.92 },
            { type: 'Emergency Battery Replacement', description: 'Rush replacement including expedited shipping', amount: 95, calculation_method: '$45 battery + $50 overnight shipping', confidence: 0.95 },
            { type: 'IT Support Time', description: 'Device troubleshooting, battery swap, reconfiguration', amount: 225, calculation_method: '3 hours x $75/hr IT rate', confidence: 0.88 },
            { type: 'Missed SLA Penalties', description: 'Late shipments due to scanning delays', amount: 1405, calculation_method: '5 orders x $281 avg penalty', confidence: 0.75 },
          ],
          deviceCost: 599,
          replacementCost: 649,
          depreciatedValue: 425,
          confidence: 0.85,
          explanation: 'High confidence based on historical battery failures, worker productivity metrics, and SLA penalty data from this location',
          level: 'high' as const,
        },
        {
          // Medium impact: Excessive reboots affecting operations
          severity: 'medium' as const,
          total: 1680,
          components: [
            { type: 'Cumulative Downtime', description: '23 reboots averaging 4 min each, device offline 92 min', amount: 575, calculation_method: '1.53 hours x $375/hr warehouse throughput', confidence: 0.88 },
            { type: 'Worker Frustration', description: 'Extended task completion times from interruptions', amount: 380, calculation_method: '15% efficiency loss x 8 hours x $32/hr', confidence: 0.72 },
            { type: 'IT Investigation', description: 'Root cause analysis of firmware issue', amount: 375, calculation_method: '5 hours x $75/hr IT rate', confidence: 0.90 },
            { type: 'Potential Hardware Damage', description: 'Accelerated wear from repeated cold boots', amount: 350, calculation_method: '5% increased failure probability x $599 device cost + labor', confidence: 0.65 },
          ],
          deviceCost: 599,
          replacementCost: 649,
          depreciatedValue: 380,
          confidence: 0.78,
          explanation: 'Medium confidence; reboot patterns match known firmware bug affecting TC52 devices running OS build 13.2.1',
          level: 'medium' as const,
        },
        {
          // High impact: Device damage from excessive drops
          severity: 'critical' as const,
          total: 3420,
          components: [
            { type: 'Screen Replacement', description: 'Cracked display from drop #47 (1.8m onto concrete)', amount: 285, calculation_method: 'OEM screen + labor from repair partner', confidence: 0.95 },
            { type: 'Device Out of Service', description: 'Worker without device for 2 business days', amount: 1600, calculation_method: '16 hours x $100/hr productivity', confidence: 0.85 },
            { type: 'Loaner Device Logistics', description: 'Provisioning, shipping, and recovery of loaner', amount: 185, calculation_method: '$35 shipping + 2 hours IT time', confidence: 0.92 },
            { type: 'Accelerated Depreciation', description: 'Internal damage reducing device lifespan by ~8 months', amount: 1350, calculation_method: '($599 / 36 months) x 8 months remaining value loss', confidence: 0.70 },
          ],
          deviceCost: 599,
          replacementCost: 649,
          depreciatedValue: 220,
          confidence: 0.82,
          explanation: 'Device has 47 recorded drops in 6 months (fleet avg: 8). Location Harbor Point has 3.2x higher drop rate - consider protective cases',
          level: 'high' as const,
        },
      ];

      const scenario = scenarios[anomalyId % scenarios.length];

      return Promise.resolve({
        anomaly_id: anomalyId,
        device_id: 1001 + (anomalyId % 100),
        device_model: 'Zebra TC52',
        anomaly_severity: scenario.severity,
        total_estimated_impact: scenario.total,
        impact_components: scenario.components,
        device_unit_cost: scenario.deviceCost,
        device_replacement_cost: scenario.replacementCost,
        device_age_months: 18,
        device_depreciated_value: scenario.depreciatedValue,
        estimated_downtime_hours: 4.5,
        productivity_cost_per_hour: 100,
        productivity_impact: 450,
        average_resolution_time_hours: 3,
        support_cost_per_hour: 75,
        estimated_support_cost: 225,
        similar_cases_count: 12,
        similar_cases_avg_cost: scenario.total * 0.95,
        overall_confidence: scenario.confidence,
        confidence_explanation: scenario.explanation,
        impact_level: scenario.level,
        using_defaults: false,  // Mock mode simulates configured costs
        calculated_at: new Date().toISOString(),
      });
    }
    return fetchAPI<AnomalyImpactResponse>(`/costs/impact/${anomalyId}`);
  },

  getDeviceImpact: (deviceId: number): Promise<DeviceImpactResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        device_id: deviceId,
        device_model: 'Zebra TC52',
        device_name: `Device ${deviceId}`,
        summary: {
          device_id: deviceId,
          device_model: 'Zebra TC52',
          device_name: `Device ${deviceId}`,
          location: 'Warehouse A',
          unit_cost: 500,
          current_value: 350,
          total_anomalies: 5,
          open_anomalies: 1,
          resolved_anomalies: 4,
          total_estimated_impact: 1250,
          impact_mtd: 450,
          impact_ytd: 1250,
          risk_score: 0.65,
          risk_level: 'medium',
        },
        recent_anomalies: [],
        monthly_impact_trend: {},
        cost_saving_recommendations: [
          'Consider replacing battery if health is below 80%',
          'Schedule preventive maintenance to reduce downtime',
        ],
      });
    }
    return fetchAPI<DeviceImpactResponse>(`/costs/device/${deviceId}/impact`);
  },

  // Cost History / Audit
  getCostHistory: (params?: {
    page?: number;
    page_size?: number;
    entity_type?: string;
    action?: string;
  }): Promise<CostHistoryResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        changes: [],
        total: 0,
        page: 1,
        page_size: 50,
        total_pages: 0,
        total_creates: 0,
        total_updates: 0,
        total_deletes: 0,
        unique_users: 0,
      });
    }
    const queryParams = new URLSearchParams();
    if (params?.page) queryParams.append('page', params.page.toString());
    if (params?.page_size) queryParams.append('page_size', params.page_size.toString());
    if (params?.entity_type) queryParams.append('entity_type', params.entity_type);
    if (params?.action) queryParams.append('action', params.action);
    const query = queryParams.toString() ? `?${queryParams.toString()}` : '';
    return fetchAPI<CostHistoryResponse>(`/costs/history${query}`);
  },

  // ============================================================================
  // BATTERY FORECASTING API
  // Predicts upcoming battery replacements based on lifespan data
  // ============================================================================

  getBatteryForecast: (): Promise<BatteryForecastResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        forecasts: [
          {
            device_model: 'Zebra TC52',
            device_count: 120,
            battery_replacement_cost: 45,
            battery_lifespan_months: 18,
            devices_due_this_month: 12,
            devices_due_next_month: 8,
            devices_due_in_90_days: 28,
            estimated_cost_30_days: 540,
            estimated_cost_90_days: 1260,
            avg_battery_age_months: 14,
            oldest_battery_months: 22,
            avg_battery_health_percent: 72,
            data_quality: 'real' as const,
          },
          {
            device_model: 'Zebra TC75x',
            device_count: 85,
            battery_replacement_cost: 55,
            battery_lifespan_months: 24,
            devices_due_this_month: 5,
            devices_due_next_month: 7,
            devices_due_in_90_days: 18,
            estimated_cost_30_days: 275,
            estimated_cost_90_days: 990,
            avg_battery_age_months: 18,
            oldest_battery_months: 28,
            avg_battery_health_percent: 68,
            data_quality: 'real' as const,
          },
          {
            device_model: 'Honeywell CK65',
            device_count: 45,
            battery_replacement_cost: 65,
            battery_lifespan_months: 24,
            devices_due_this_month: 3,
            devices_due_next_month: 4,
            devices_due_in_90_days: 11,
            estimated_cost_30_days: 195,
            estimated_cost_90_days: 715,
            avg_battery_age_months: 16,
            oldest_battery_months: 26,
            avg_battery_health_percent: 76,
            data_quality: 'mixed' as const,
          },
          {
            device_model: 'Samsung Galaxy XCover 6',
            device_count: 62,
            battery_replacement_cost: 35,
            battery_lifespan_months: 18,
            devices_due_this_month: 8,
            devices_due_next_month: 6,
            devices_due_in_90_days: 20,
            estimated_cost_30_days: 280,
            estimated_cost_90_days: 700,
            avg_battery_age_months: 12,
            oldest_battery_months: 20,
            avg_battery_health_percent: 81,
            data_quality: 'real' as const,
          },
          {
            device_model: 'Panasonic Toughbook N1',
            device_count: 18,
            battery_replacement_cost: 89,
            battery_lifespan_months: 30,
            devices_due_this_month: 1,
            devices_due_next_month: 2,
            devices_due_in_90_days: 4,
            estimated_cost_30_days: 89,
            estimated_cost_90_days: 356,
            avg_battery_age_months: 20,
            oldest_battery_months: 32,
            data_quality: 'estimated' as const,
          },
        ],
        total_devices_with_battery_data: 330,
        total_estimated_cost_30_days: 1379,
        total_estimated_cost_90_days: 4021,
        total_replacements_due_30_days: 29,
        total_replacements_due_90_days: 81,
        forecast_generated_at: new Date().toISOString(),
        data_quality: 'mixed' as const,
        devices_with_health_data: 267,
      });
    }
    return fetchAPI<BatteryForecastResponse>('/costs/battery-forecast');
  },

  // ============================================================================
  // COST ALERTS API
  // Configure thresholds for cost-based alerting
  // ============================================================================

  getCostAlerts: (): Promise<CostAlertListResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        alerts: [
          {
            id: 1,
            name: 'Daily Anomaly Cost Threshold',
            threshold_type: 'anomaly_cost_daily',
            threshold_value: 5000,
            is_active: true,
            notify_email: 'ops@company.com',
            last_triggered: '2025-12-28T14:30:00Z',
            trigger_count: 3,
            created_at: '2024-01-15T10:00:00Z',
            updated_at: '2024-06-20T14:00:00Z',
          },
          {
            id: 2,
            name: 'Monthly Cost Budget Alert',
            threshold_type: 'anomaly_cost_monthly',
            threshold_value: 50000,
            is_active: true,
            notify_email: 'finance@company.com',
            trigger_count: 1,
            created_at: '2024-02-01T09:00:00Z',
            updated_at: '2024-02-01T09:00:00Z',
          },
          {
            id: 3,
            name: 'Battery Replacement Budget',
            threshold_type: 'battery_forecast',
            threshold_value: 2000,
            is_active: true,
            last_triggered: '2025-12-15T09:00:00Z',
            trigger_count: 5,
            created_at: '2024-03-10T11:00:00Z',
            updated_at: '2024-09-01T16:00:00Z',
          },
        ],
        total: 3,
      });
    }
    return fetchAPI<CostAlertListResponse>('/costs/alerts');
  },

  createCostAlert: (alert: CostAlertCreate): Promise<CostAlert> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        ...alert,
        id: Math.floor(Math.random() * 1000),
        trigger_count: 0,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      });
    }
    return fetchAPI<CostAlert>('/costs/alerts', {
      method: 'POST',
      body: JSON.stringify(alert),
    });
  },

  updateCostAlert: (id: number, alert: Partial<CostAlertCreate>): Promise<CostAlert> => {
    if (getMockModeFromStorage()) {
      return Promise.reject(new Error('Update not available in mock mode'));
    }
    return fetchAPI<CostAlert>(`/costs/alerts/${id}`, {
      method: 'PUT',
      body: JSON.stringify(alert),
    });
  },

  deleteCostAlert: (id: number): Promise<void> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve();
    }
    return fetchAPI<void>(`/costs/alerts/${id}`, { method: 'DELETE' });
  },

  // ============================================================================
  // NFF (NO FAULT FOUND) TRACKING API
  // Track unnecessary repair/investigation costs
  // ============================================================================

  getNFFSummary: (): Promise<NFFSummary> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        total_nff_count: 47,
        total_nff_cost: 3525,
        avg_cost_per_nff: 75,
        nff_rate_percent: 23.5,
        by_device_model: [
          { device_model: 'Zebra TC52', count: 18, total_cost: 1350 },
          { device_model: 'Zebra TC75x', count: 12, total_cost: 900 },
          { device_model: 'Honeywell CK65', count: 9, total_cost: 675 },
          { device_model: 'Samsung Galaxy XCover 6', count: 8, total_cost: 600 },
        ],
        by_resolution: [
          { resolution: 'no_fault_found', count: 22, total_cost: 1650 },
          { resolution: 'user_error', count: 15, total_cost: 1125 },
          { resolution: 'intermittent', count: 7, total_cost: 525 },
          { resolution: 'software_issue', count: 3, total_cost: 225 },
        ],
        trend_30_days: -8.5,
      });
    }
    return fetchAPI<NFFSummary>('/costs/nff/summary');
  },

  // ============================================================================
  // SYSTEM HEALTH API
  // Fleet health monitoring with CPU, RAM, storage, and temperature metrics
  // ============================================================================

  getSystemHealthSummary: (periodDays?: number): Promise<SystemHealthSummaryResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockSystemHealthSummary());
    }
    const params = periodDays ? `?period_days=${periodDays}` : '';
    return fetchAPI<SystemHealthSummaryResponse>(`/insights/system-health/summary${params}`);
  },

  getHealthTrends: (
    metric: string,
    periodDays?: number
  ): Promise<HealthTrendsResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockHealthTrends(metric));
    }
    const queryParams = new URLSearchParams();
    queryParams.append('metric', metric);
    if (periodDays) queryParams.append('period_days', periodDays.toString());
    return fetchAPI<HealthTrendsResponse>(`/insights/system-health/trends?${queryParams.toString()}`);
  },

  getStorageForecast: (
    daysUntilFullThreshold?: number,
    limit?: number
  ): Promise<StorageForecastResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockStorageForecast());
    }
    const queryParams = new URLSearchParams();
    if (daysUntilFullThreshold) queryParams.append('days_until_full_threshold', daysUntilFullThreshold.toString());
    if (limit) queryParams.append('limit', limit.toString());
    const query = queryParams.toString() ? `?${queryParams.toString()}` : '';
    return fetchAPI<StorageForecastResponse>(`/insights/system-health/storage-forecast${query}`);
  },

  getCohortHealthBreakdown: (periodDays?: number): Promise<CohortHealthBreakdownResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockCohortHealthBreakdown());
    }
    const params = periodDays ? `?period_days=${periodDays}` : '';
    return fetchAPI<CohortHealthBreakdownResponse>(`/insights/system-health/cohort-breakdown${params}`);
  },

  // ============================================================================
  // LOCATION INTELLIGENCE API
  // WiFi coverage, GPS tracking, and mobility analytics
  // ============================================================================

  getWiFiHeatmap: (periodDays?: number): Promise<WiFiHeatmapResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockWiFiHeatmap());
    }
    const params = periodDays ? `?period_days=${periodDays}` : '';
    return fetchAPI<WiFiHeatmapResponse>(`/insights/location/wifi-heatmap${params}`);
  },

  getDeadZones: (
    signalThreshold?: number,
    minReadings?: number
  ): Promise<DeadZonesResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockDeadZones());
    }
    const queryParams = new URLSearchParams();
    if (signalThreshold) queryParams.append('signal_threshold', signalThreshold.toString());
    if (minReadings) queryParams.append('min_readings', minReadings.toString());
    const query = queryParams.toString() ? `?${queryParams.toString()}` : '';
    return fetchAPI<DeadZonesResponse>(`/insights/location/dead-zones${query}`);
  },

  getDeviceMovements: (
    deviceId: number,
    periodDays?: number
  ): Promise<DeviceMovementResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockDeviceMovements(deviceId));
    }
    const params = periodDays ? `?period_days=${periodDays}` : '';
    return fetchAPI<DeviceMovementResponse>(`/insights/location/device-movements/${deviceId}${params}`);
  },

  getDwellTime: (periodDays?: number): Promise<DwellTimeResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockDwellTime());
    }
    const params = periodDays ? `?period_days=${periodDays}` : '';
    return fetchAPI<DwellTimeResponse>(`/insights/location/dwell-time${params}`);
  },

  getCoverageSummary: (periodDays?: number): Promise<CoverageSummaryResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockCoverageSummary());
    }
    const params = periodDays ? `?period_days=${periodDays}` : '';
    return fetchAPI<CoverageSummaryResponse>(`/insights/location/coverage-summary${params}`);
  },

  // ============================================================================
  // EVENTS & ALERTS API
  // System event timeline and alert monitoring
  // ============================================================================

  getEventTimeline: (params?: {
    page?: number;
    page_size?: number;
    severity?: string;
    event_class?: string;
    device_id?: number;
    hours_back?: number;
    start_time?: string;
    end_time?: string;
  }): Promise<EventTimelineResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockEventTimeline());
    }
    const queryParams = new URLSearchParams();
    if (params?.page) queryParams.append('page', params.page.toString());
    if (params?.page_size) queryParams.append('page_size', params.page_size.toString());
    if (params?.severity) queryParams.append('severity', params.severity);
    if (params?.event_class) queryParams.append('event_class', params.event_class);
    if (params?.device_id) queryParams.append('device_id', params.device_id.toString());
    if (params?.start_time) queryParams.append('start_time', params.start_time);
    if (params?.end_time) queryParams.append('end_time', params.end_time);
    if (params?.hours_back && !params.start_time && !params.end_time) {
      // Translate hours_back to the backend's start_time/end_time filter.
      const end = new Date();
      const start = new Date(end.getTime() - params.hours_back * 60 * 60 * 1000);
      queryParams.append('start_time', start.toISOString());
      queryParams.append('end_time', end.toISOString());
    }
    const query = queryParams.toString() ? `?${queryParams.toString()}` : '';
    return fetchAPI<EventTimelineResponse>(`/insights/events/timeline${query}`);
  },

  getAlertSummary: (hoursBack?: number): Promise<AlertSummaryResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockAlertSummary());
    }
    // Backend uses period_days; convert UI hours into a day window.
    const periodDays = hoursBack ? Math.max(1, Math.ceil(hoursBack / 24)) : undefined;
    const params = periodDays ? `?period_days=${periodDays}` : '';
    return fetchAPI<AlertSummaryResponse>(`/insights/events/alerts/summary${params}`);
  },

  getAlertTrends: (
    periodDays?: number,
    granularity?: 'hourly' | 'daily'
  ): Promise<AlertTrendsResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockAlertTrends());
    }
    const queryParams = new URLSearchParams();
    if (periodDays) queryParams.append('period_days', periodDays.toString());
    if (granularity) queryParams.append('granularity', granularity);
    const query = queryParams.toString() ? `?${queryParams.toString()}` : '';
    return fetchAPI<AlertTrendsResponse>(`/insights/events/alerts/trends${query}`);
  },

  getEventCorrelation: (
    anomalyTimestamp: string,
    deviceId: number,
    windowMinutes?: number
  ): Promise<EventCorrelationResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockEventCorrelation(deviceId));
    }
    const queryParams = new URLSearchParams();
    queryParams.append('anomaly_timestamp', anomalyTimestamp);
    queryParams.append('device_id', deviceId.toString());
    if (windowMinutes) queryParams.append('window_minutes', windowMinutes.toString());
    return fetchAPI<EventCorrelationResponse>(`/insights/events/correlation?${queryParams.toString()}`);
  },

  getEventStatistics: (hoursBack?: number): Promise<EventStatisticsResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockEventStatistics());
    }
    // Backend uses period_days; convert UI hours into a day window.
    const periodDays = hoursBack ? Math.max(1, Math.ceil(hoursBack / 24)) : undefined;
    const params = periodDays ? `?period_days=${periodDays}` : '';
    return fetchAPI<EventStatisticsResponse>(`/insights/events/statistics${params}`);
  },

  // ============================================================================
  // TEMPORAL ANALYSIS API
  // Time-based patterns, peaks, and period comparisons
  // ============================================================================

  getHourlyBreakdown: (
    metric: string,
    periodDays?: number
  ): Promise<HourlyBreakdownResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockHourlyBreakdown(metric));
    }
    const queryParams = new URLSearchParams();
    queryParams.append('metric', metric);
    if (periodDays) queryParams.append('period_days', periodDays.toString());
    return fetchAPI<HourlyBreakdownResponse>(`/insights/temporal/hourly-breakdown?${queryParams.toString()}`);
  },

  getPeakDetection: (
    metric: string,
    periodDays?: number,
    stdThreshold?: number
  ): Promise<PeakDetectionResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockPeakDetection(metric));
    }
    const queryParams = new URLSearchParams();
    queryParams.append('metric', metric);
    if (periodDays) queryParams.append('period_days', periodDays.toString());
    if (stdThreshold) queryParams.append('std_threshold', stdThreshold.toString());
    return fetchAPI<PeakDetectionResponse>(`/insights/temporal/peak-detection?${queryParams.toString()}`);
  },

  getTemporalComparison: (params: {
    metric: string;
    period_a_start: string;
    period_a_end: string;
    period_b_start: string;
    period_b_end: string;
  }): Promise<TemporalComparisonResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockTemporalComparison(params.metric));
    }
    const queryParams = new URLSearchParams();
    queryParams.append('metric', params.metric);
    queryParams.append('period_a_start', params.period_a_start);
    queryParams.append('period_a_end', params.period_a_end);
    queryParams.append('period_b_start', params.period_b_start);
    queryParams.append('period_b_end', params.period_b_end);
    return fetchAPI<TemporalComparisonResponse>(`/insights/temporal/comparison?${queryParams.toString()}`);
  },

  getDayOverDay: (
    metric: string,
    lookbackDays?: number
  ): Promise<DayOverDayResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockDayOverDay(metric));
    }
    const queryParams = new URLSearchParams();
    queryParams.append('metric', metric);
    if (lookbackDays) queryParams.append('lookback_days', lookbackDays.toString());
    return fetchAPI<DayOverDayResponse>(`/insights/temporal/day-over-day?${queryParams.toString()}`);
  },

  getWeekOverWeek: (
    metric: string,
    lookbackWeeks?: number
  ): Promise<WeekOverWeekResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockWeekOverWeek(metric));
    }
    const queryParams = new URLSearchParams();
    queryParams.append('metric', metric);
    if (lookbackWeeks) queryParams.append('lookback_weeks', lookbackWeeks.toString());
    return fetchAPI<WeekOverWeekResponse>(`/insights/temporal/week-over-week?${queryParams.toString()}`);
  },

  // ============================================================================
  // CORRELATION INTELLIGENCE
  // ============================================================================

  /**
   * Get correlation matrix for numeric metrics.
   * Returns N x N correlation matrix and list of strong correlations.
   */
  getCorrelationMatrix: (
    params?: CorrelationMatrixParams
  ): Promise<CorrelationMatrixResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockCorrelationMatrix(params?.domain));
    }
    const queryParams = new URLSearchParams();
    if (params?.domain) queryParams.append('domain', params.domain);
    if (params?.method) queryParams.append('method', params.method);
    if (params?.threshold) queryParams.append('threshold', params.threshold.toString());
    if (params?.max_metrics) queryParams.append('max_metrics', params.max_metrics.toString());
    const query = queryParams.toString();
    return fetchAPI<CorrelationMatrixResponse>(
      `/correlations/matrix${query ? `?${query}` : ''}`
    );
  },

  /**
   * Get scatter plot data for two metrics.
   * Returns data points, correlation coefficient, and regression line parameters.
   */
  getScatterData: (
    metricX: string,
    metricY: string,
    colorBy: 'anomaly' | 'cohort' = 'anomaly',
    limit: number = 500
  ): Promise<ScatterPlotResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockScatterData(metricX, metricY, limit));
    }
    const queryParams = new URLSearchParams({
      metric_x: metricX,
      metric_y: metricY,
      color_by: colorBy,
      limit: limit.toString(),
    });
    return fetchAPI<ScatterPlotResponse>(
      `/correlations/scatter?${queryParams.toString()}`
    );
  },

  /**
   * Get causal relationship network from domain knowledge.
   * Returns nodes and edges representing known causal relationships.
   */
  getCausalGraph: (includeInferred: boolean = true): Promise<CausalGraphResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockCausalGraph());
    }
    return fetchAPI<CausalGraphResponse>(
      `/correlations/causal-graph?include_inferred=${includeInferred}`
    );
  },

  /**
   * Get auto-discovered correlation insights.
   * Returns ranked list of insights about metric relationships.
   */
  getCorrelationInsights: (
    topK: number = 10,
    minStrength: number = 0.5
  ): Promise<CorrelationInsightsResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockCorrelationInsights());
    }
    return fetchAPI<CorrelationInsightsResponse>(
      `/correlations/insights?top_k=${topK}&min_strength=${minStrength}`
    );
  },

  /**
   * Get cohort-specific correlation patterns.
   * Identifies cohorts with unusual correlation patterns compared to fleet average.
   */
  getCohortCorrelationPatterns: (): Promise<CohortCorrelationPatternsResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockCohortCorrelationPatterns());
    }
    return fetchAPI<CohortCorrelationPatternsResponse>(
      '/correlations/cohort-patterns'
    );
  },

  /**
   * Get time-lagged correlations for predictive insights.
   * Analyzes how metrics at time T correlate with other metrics at time T+lag.
   */
  getTimeLaggedCorrelations: (
    maxLag: number = 7,
    minCorrelation: number = 0.3
  ): Promise<TimeLagCorrelationsResponse> => {
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockTimeLaggedCorrelations());
    }
    return fetchAPI<TimeLagCorrelationsResponse>(
      `/correlations/time-lagged?max_lag=${maxLag}&min_correlation=${minCorrelation}`
    );
  },
};

// Device action response types
export interface DeviceActionResponse {
  success: boolean;
  action: string;
  device_id: number;
  message: string;
  action_id?: string;
  timestamp: string;
}

export interface DeviceLocationResponse {
  success: boolean;
  device_id: number;
  latitude?: number;
  longitude?: number;
  accuracy?: number;
  timestamp?: string;
  message: string;
}

export interface DeviceActionHistoryResponse {
  device_id: number;
  total: number;
  actions: Array<{
    id: number;
    action_type: string;
    initiated_by: string;
    reason?: string;
    success: boolean;
    error_message?: string;
    timestamp?: string;
  }>;
}

// ============================================================================
// INSIGHTS API Response Types
// ============================================================================

export interface CustomerInsightResponse {
  insight_id: string;
  category: string;
  severity: string;
  headline: string;
  impact_statement: string;
  comparison_context: string;
  recommended_actions: string[];
  entity_type: string;
  entity_id: string;
  entity_name: string;
  affected_device_count: number;
  primary_metric: string;
  primary_value: number;
  trend_direction: string;
  trend_change_percent: number | null;
  detected_at: string;
  confidence_score: number;
}

export interface DailyDigestResponse {
  tenant_id: string;
  digest_date: string;
  generated_at: string;
  total_insights: number;
  critical_count: number;
  high_count: number;
  medium_count: number;
  top_insights: CustomerInsightResponse[];
  executive_summary: string;
  trending_issues: CustomerInsightResponse[];
  new_issues: CustomerInsightResponse[];
}

export interface LocationInsightResponse {
  location_id: string;
  location_name: string;
  report_date: string;
  total_devices: number;
  devices_with_issues: number;
  issue_rate: number;
  shift_readiness: ShiftReadinessResponse | null;
  insights: CustomerInsightResponse[];
  top_issues: Array<{ category: string; count: number }>;
  rank_among_locations: number;
  better_than_percent: number;
  recommendations: string[];
}

export interface ShiftReadinessResponse {
  location_id: string;
  location_name: string;
  shift_name: string;
  shift_date: string;
  readiness_percentage: number;
  total_devices: number;
  devices_ready: number;
  devices_at_risk: number;
  devices_critical: number;
  avg_battery_at_start: number;
  avg_drain_rate: number;
  devices_not_fully_charged: number;
  vs_last_week_readiness: number | null;
  device_details: Array<{
    device_id: number;
    device_name: string;
    current_battery: number;
    drain_rate_per_hour: number;
    projected_end_battery: number;
    will_complete_shift: boolean;
    estimated_dead_time: string | null;
    was_fully_charged: boolean;
    readiness_score: number;
    recommendations: string[];
  }>;
  recommendations: string[];
}

// Financial Impact Summary for Analysis Reports
export interface AnalysisFinancialImpact {
  total_estimated_cost: number;
  cost_breakdown: Array<{ category: string; amount: number; description: string }>;
  potential_savings: number;
  cost_per_incident?: number;
  monthly_trend?: number; // percentage change
}

export interface NetworkAnalysisResponse {
  tenant_id: string;
  analysis_period_days: number;
  wifi_summary: {
    total_devices: number;
    devices_with_roaming_issues: number;
    devices_with_stickiness: number;
    avg_aps_per_device: number;
    potential_dead_zones: number;
  };
  cellular_summary: {
    total_devices: number;
    devices_with_tower_hopping: number;
    devices_with_tech_fallback: number;
    best_carrier: string;
    worst_carrier: string;
    network_type_distribution: Record<string, number>;
  } | null;
  disconnect_summary: {
    total_disconnects: number;
    avg_disconnects_per_device: number;
    total_offline_hours: number;
    has_predictable_pattern: boolean;
    pattern_description: string;
  };
  hidden_devices_count: number;
  recommendations: string[];
  financial_impact?: AnalysisFinancialImpact;
}

// User abuse item for by-user analysis (Carl's "People with excessive drops")
export interface UserAbuseItem {
  user_id: string;
  user_name?: string;
  user_email?: string;
  total_drops: number;
  total_reboots: number;
  device_count: number;
  drops_per_device: number;
  drops_per_day: number;
  vs_fleet_multiplier: number;
  is_excessive: boolean;
}

export interface DeviceAbuseResponse {
  tenant_id: string;
  analysis_period_days: number;
  total_devices: number;
  total_drops: number;
  total_reboots: number;
  devices_with_excessive_drops: number;
  devices_with_excessive_reboots: number;
  worst_locations: Array<{
    location_id: string;
    drops: number;
    rate_per_device: number;
  }>;
  worst_cohorts: Array<{
    cohort_id: string;
    reboots: number;
    rate_per_device: number;
  }>;
  // Carl's "People with excessive drops" - users ranked by drop count
  worst_users?: UserAbuseItem[];
  problem_combinations: Array<{
    cohort_id: string;
    manufacturer: string;
    model: string;
    os_version: string;
    device_count: number;
    vs_fleet_multiplier: number;
    primary_issue: string;
    severity: string;
  }>;
  recommendations: string[];
  financial_impact?: AnalysisFinancialImpact;
}

export interface AppAnalysisResponse {
  tenant_id: string;
  analysis_period_days: number;
  total_apps_analyzed: number;
  apps_with_issues: number;
  total_crashes: number;
  total_anrs: number;
  top_power_consumers: Array<{
    package_name: string;
    app_name: string;
    battery_drain_percent: number;
    drain_per_hour: number;
    foreground_hours: number;
    efficiency_score: number;
  }>;
  top_crashers: Array<{
    package_name: string;
    app_name: string;
    crash_count: number;
    anr_count: number;
    devices_affected: number;
    severity: string;
  }>;
  recommendations: string[];
  financial_impact?: AnalysisFinancialImpact;
}

export interface LocationCompareResponse {
  location_a_id: string;
  location_a_name: string;
  location_b_id: string;
  location_b_name: string;
  device_count_a: number;
  device_count_b: number;
  metric_comparisons: Record<string, {
    location_a_value: number;
    location_b_value: number;
    difference_percent: number;
  }>;
  overall_winner: string | null;
  key_differences: string[];
}

// ============================================================================
// SYSTEM HEALTH API Response Types
// ============================================================================

export interface SystemHealthMetricsResponse {
  avg_cpu_usage: number;
  avg_memory_usage: number;
  avg_storage_available_pct: number;
  avg_device_temp: number;
  avg_battery_temp: number;
  devices_high_cpu: number;
  devices_high_memory: number;
  devices_low_storage: number;
  devices_high_temp: number;
  total_devices: number;
}

export interface CohortHealthResponse {
  cohort_id: string;
  cohort_name: string;
  device_count: number;
  health_score: number;
  avg_cpu: number;
  avg_memory: number;
  avg_storage_pct: number;
  devices_at_risk: number;
}

export interface SystemHealthSummaryResponse {
  tenant_id: string;
  fleet_health_score: number;
  health_trend: 'improving' | 'stable' | 'degrading' | 'unknown';
  total_devices: number;
  healthy_count: number;
  warning_count: number;
  critical_count: number;
  metrics: SystemHealthMetricsResponse;
  cohort_breakdown: CohortHealthResponse[];
  recommendations: string[];
  generated_at: string;
}

export interface HealthTrendPointResponse {
  timestamp: string;
  value: number;
  device_count: number;
}

export interface HealthTrendsResponse {
  tenant_id: string;
  metric: string;
  trends: HealthTrendPointResponse[];
  generated_at: string;
}

export interface StorageForecastDeviceResponse {
  device_id: number;
  device_name: string;
  current_storage_pct: number;
  storage_trend_gb_per_day: number;
  projected_full_date: string | null;
  days_until_full: number | null;
  confidence: number;
}

export interface StorageForecastResponse {
  tenant_id: string;
  devices_at_risk: StorageForecastDeviceResponse[];
  total_at_risk_count: number;
  avg_days_until_full: number | null;
  recommendations: string[];
  generated_at: string;
}

export interface CohortHealthBreakdownResponse {
  tenant_id: string;
  cohorts: CohortHealthResponse[];
  total_cohorts: number;
  generated_at: string;
}

// ============================================================================
// LOCATION INTELLIGENCE API Response Types
// ============================================================================

export interface GeoBoundsResponse {
  min_lat: number;
  max_lat: number;
  min_long: number;
  max_long: number;
}

export interface HeatmapCellResponse {
  lat: number;
  long: number;
  signal_strength: number;
  reading_count: number;
  is_dead_zone: boolean;
  access_point_id: string | null;
}

export interface WiFiHeatmapResponse {
  tenant_id: string;
  grid_cells: HeatmapCellResponse[];
  bounds: GeoBoundsResponse | null;
  total_readings: number;
  avg_signal_strength: number;
  dead_zone_count: number;
  generated_at: string;
}

export interface DeadZoneResponse {
  zone_id: string;
  lat: number;
  long: number;
  avg_signal: number;
  affected_devices: number;
  total_readings: number;
  first_detected: string | null;
  last_detected: string | null;
}

export interface DeadZonesResponse {
  tenant_id: string;
  dead_zones: DeadZoneResponse[];
  total_count: number;
  recommendations: string[];
  generated_at: string;
}

export interface MovementPointResponse {
  timestamp: string;
  lat: number;
  long: number;
  speed: number;
  heading: number;
}

export interface DeviceMovementResponse {
  device_id: number;
  movements: MovementPointResponse[];
  total_distance_km: number;
  avg_speed_kmh: number;
  stationary_time_pct: number;
  active_hours: number[];
}

export interface DwellZoneResponse {
  zone_id: string;
  lat: number;
  long: number;
  avg_dwell_minutes: number;
  device_count: number;
  visit_count: number;
  peak_hours: number[];
}

export interface DwellTimeResponse {
  tenant_id: string;
  dwell_zones: DwellZoneResponse[];
  total_zones: number;
  recommendations: string[];
  generated_at: string;
}

export interface CoverageSummaryResponse {
  tenant_id: string;
  total_readings: number;
  avg_signal: number;
  coverage_distribution: Record<string, number>;
  coverage_percentage: number;
  recommendations: string[];
  generated_at: string;
}

// ============================================================================
// EVENTS & ALERTS API Response Types
// ============================================================================

export interface EventEntryResponse {
  log_id: number;
  timestamp: string;
  event_id: number;
  severity: string;
  event_class: string;
  message: string;
  device_id: number | null;
  login_id: string | null;
}

export interface EventTimelineResponse {
  tenant_id: string;
  events: EventEntryResponse[];
  total: number;
  page: number;
  page_size: number;
  severity_distribution: Record<string, number>;
  event_class_distribution: Record<string, number>;
  generated_at: string;
}

export interface AlertEntryResponse {
  alert_id: number;
  alert_key: string;
  alert_name: string;
  severity: string;
  device_id: string | null;
  status: string;
  set_datetime: string | null;
  ack_datetime: string | null;
}

export interface AlertNameCountResponse {
  name: string;
  count: number;
}

export interface AlertSummaryResponse {
  tenant_id: string;
  total_active: number;
  total_acknowledged: number;
  total_resolved: number;
  by_severity: Record<string, number>;
  by_alert_name: AlertNameCountResponse[];
  recent_alerts: AlertEntryResponse[];
  avg_acknowledge_time_minutes: number;
  avg_resolution_time_minutes: number;
  generated_at: string;
}

export interface AlertTrendPointResponse {
  timestamp: string;
  count: number;
  severity: string;
}

export interface AlertTrendsResponse {
  tenant_id: string;
  trends: AlertTrendPointResponse[];
  generated_at: string;
}

export interface CorrelatedEventResponse {
  event: EventEntryResponse;
  time_before_minutes: number;
  frequency_score: number;
}

export interface EventCorrelationResponse {
  tenant_id: string;
  anomaly_timestamp: string;
  device_id: number;
  correlated_events: CorrelatedEventResponse[];
  total_events_found: number;
  generated_at: string;
}

export interface EventStatisticsResponse {
  tenant_id: string;
  total_events: number;
  events_per_day: number;
  unique_devices: number;
  top_event_classes: Array<{ class: string; count: number }>;
  generated_at: string;
}

// ============================================================================
// TEMPORAL ANALYSIS API Response Types
// ============================================================================

export interface HourlyDataPointResponse {
  hour: number;
  avg_value: number;
  min_value: number;
  max_value: number;
  std_value: number;
  sample_count: number;
}

export interface HourlyBreakdownResponse {
  tenant_id: string;
  metric: string;
  hourly_data: HourlyDataPointResponse[];
  peak_hours: number[];
  low_hours: number[];
  day_night_ratio: number;
  generated_at: string;
}

export interface PeakDetectionItemResponse {
  timestamp: string;
  value: number;
  z_score: number;
  is_significant: boolean;
}

export interface PeakDetectionResponse {
  tenant_id: string;
  metric: string;
  peaks: PeakDetectionItemResponse[];
  total_peaks: number;
  generated_at: string;
}

export interface PeriodStatsResponse {
  start: string;
  end: string;
  avg: number;
  median: number;
  std: number;
  sample_count: number;
}

export interface TemporalComparisonResponse {
  tenant_id: string;
  metric: string;
  period_a: PeriodStatsResponse;
  period_b: PeriodStatsResponse;
  change_percent: number;
  is_significant: boolean;
  p_value: number;
  generated_at: string;
}

export interface DailyComparisonPointResponse {
  date: string;
  value: number;
  sample_count: number;
  change_percent: number;
}

export interface DayOverDayResponse {
  tenant_id: string;
  metric: string;
  comparisons: DailyComparisonPointResponse[];
  generated_at: string;
}

export interface WeeklyComparisonPointResponse {
  year: number;
  week: number;
  value: number;
  sample_count: number;
  change_percent: number;
}

export interface WeekOverWeekResponse {
  tenant_id: string;
  metric: string;
  comparisons: WeeklyComparisonPointResponse[];
  generated_at: string;
}
