import type {
  AllConnectionsStatus,
  AnomalyDetail,
  AnomalyListResponse,
  DashboardStats,
  DashboardTrend,
  DeviceDetail,
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
} from './mockData';

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
    model_type: 'IsolationForest',
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

  return response.json();
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
        getMockAnomalies({
          device_id: params.device_id,
          status: params.status,
          page: params.page,
          page_size: params.page_size,
        })
      );
    }
    if (shouldReturnEmptyLiveData()) {
      const page = params.page || 1;
      const pageSize = params.page_size || 50;
      return Promise.resolve(emptyAnomalyList(page, pageSize));
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

    return fetchAPI<AnomalyListResponse>(`/anomalies?${queryParams.toString()}`);
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

  // Devices
  getDevices: (params?: {
    page?: number;
    page_size?: number;
    search?: string;
    group_by?: string;
    group_value?: string;
  }): Promise<{ devices: DeviceDetail[]; total: number }> => {
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

      return Promise.resolve({
        devices: devices.slice(start, end),
        total: devices.length
      });
    }
    if (shouldReturnEmptyLiveData()) {
      return Promise.resolve({ devices: [], total: 0 });
    }

    const queryParams = new URLSearchParams();
    if (params?.page) queryParams.append('page', params.page.toString());
    if (params?.page_size) queryParams.append('page_size', params.page_size.toString());
    if (params?.search) queryParams.append('search', params.search);
    if (params?.group_by) queryParams.append('group_by', params.group_by);
    if (params?.group_value) queryParams.append('group_value', params.group_value);

    return fetchAPI<{ devices: DeviceDetail[]; total: number }>(`/devices?${queryParams.toString()}`);
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
      return Promise.resolve(getMockIsolationForestStats());
    }
    if (shouldReturnEmptyLiveData()) {
      return Promise.resolve(emptyIsolationForestStats);
    }

    const queryParams = new URLSearchParams();
    if (days) queryParams.append('days', days.toString());
    if (queryParams.toString()) {
      return fetchAPI<IsolationForestStats>(
        `/dashboard/isolation-forest/stats?${queryParams.toString()}`
      );
    }
    return fetchAPI<IsolationForestStats>('/dashboard/isolation-forest/stats');
  },

  // Baselines
  getBaselineSuggestions: (
    source?: string,
    days?: number
  ): Promise<BaselineSuggestion[]> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockBaselineSuggestions());
    }

    const queryParams = new URLSearchParams();
    if (source) queryParams.append('source', source);
    if (days) queryParams.append('days', days.toString());
    if (queryParams.toString()) {
      return fetchAPI<BaselineSuggestion[]>(
        `/baselines/suggestions?${queryParams.toString()}`
      );
    }
    return fetchAPI<BaselineSuggestion[]>('/baselines/suggestions');
  },

  analyzeBaselinesWithLLM: (
    source?: string,
    days?: number
  ): Promise<BaselineSuggestion[]> => {
    // Return mock data when mock mode is enabled
    if (getMockModeFromStorage()) {
      return Promise.resolve(getMockBaselineSuggestions());
    }

    const queryParams = new URLSearchParams();
    if (source) queryParams.append('source', source);
    if (days) queryParams.append('days', days.toString());
    if (queryParams.toString()) {
      return fetchAPI<BaselineSuggestion[]>(
        `/baselines/analyze-with-llm?${queryParams.toString()}`
      );
    }
    return fetchAPI<BaselineSuggestion[]>('/baselines/analyze-with-llm');
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
      return Promise.resolve([
        { feature: 'BatteryDrop', baseline: 12, observed: 14.2, unit: '%/day', status: 'stable' as const, drift_percent: 18.3, mad: 2.5, sample_count: 1250, last_updated: null },
        { feature: 'OfflineTime', baseline: 30, observed: 42, unit: 'min/day', status: 'drift' as const, drift_percent: 40.0, mad: 8.0, sample_count: 1250, last_updated: null },
        { feature: 'UploadSize', baseline: 500, observed: 520, unit: 'MB/day', status: 'stable' as const, drift_percent: 4.0, mad: 45.0, sample_count: 1250, last_updated: null },
        { feature: 'DownloadSize', baseline: 1200, observed: 1180, unit: 'MB/day', status: 'stable' as const, drift_percent: -1.7, mad: 120.0, sample_count: 1250, last_updated: null },
        { feature: 'StorageFree', baseline: 8.5, observed: 7.2, unit: 'GB', status: 'warning' as const, drift_percent: -15.3, mad: 0.8, sample_count: 1250, last_updated: null },
        { feature: 'AppCrashes', baseline: 0.5, observed: 1.8, unit: '/device/day', status: 'drift' as const, drift_percent: 260.0, mad: 0.3, sample_count: 1250, last_updated: null },
      ]);
    }

    const queryParams = new URLSearchParams();
    if (source) queryParams.append('source', source);
    if (days) queryParams.append('days', days.toString());
    const query = queryParams.toString();
    return fetchAPI<BaselineFeature[]>(`/baselines/features${query ? '?' + query : ''}`);
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
    if (unacknowledged_only) queryParams.append('unacknowledged_only', 'true');
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
    // In mock mode, simulate successful connections
    if (getMockModeFromStorage()) {
      return new Promise((resolve) => {
        setTimeout(() => {
          resolve({
            success: true,
            message: `${request.type} connection successful (mock mode)`,
            latency_ms: Math.floor(Math.random() * 100) + 20,
          });
        }, 500 + Math.random() * 500);
      });
    }
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
    // In mock mode, simulate successful save
    if (getMockModeFromStorage()) {
      return Promise.resolve({
        success: true,
        message: 'Configuration saved successfully (mock mode)',
        file_path: '.env',
      });
    }
    return fetchAPI<{ success: boolean; message: string; file_path?: string }>(
      '/setup/save-config',
      {
        method: 'POST',
        body: JSON.stringify(config),
      }
    );
  },

  getSetupConfig: (): Promise<Record<string, unknown>> => {
    // In mock mode, return empty config
    if (getMockModeFromStorage()) {
      return Promise.resolve({});
    }
    return fetchAPI<Record<string, unknown>>('/setup/config');
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
    return fetchAPI<DeviceActionResponse>(`/devices/${deviceId}/actions/lock`, {
      method: 'POST',
      body: JSON.stringify({ reason }),
    });
  },

  restartDevice: (deviceId: number, reason?: string): Promise<DeviceActionResponse> => {
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
    return fetchAPI<DeviceActionResponse>(`/devices/${deviceId}/actions/message`, {
      method: 'POST',
      body: JSON.stringify({ message, title: title || 'Message from Admin' }),
    });
  },

  locateDevice: (deviceId: number): Promise<DeviceLocationResponse> => {
    return fetchAPI<DeviceLocationResponse>(`/devices/${deviceId}/actions/locate`, {
      method: 'POST',
    });
  },

  syncDevice: (deviceId: number, reason?: string): Promise<DeviceActionResponse> => {
    return fetchAPI<DeviceActionResponse>(`/devices/${deviceId}/actions/sync`, {
      method: 'POST',
      body: JSON.stringify({ reason }),
    });
  },

  clearDeviceCache: (
    deviceId: number,
    packageName?: string
  ): Promise<DeviceActionResponse> => {
    return fetchAPI<DeviceActionResponse>(`/devices/${deviceId}/actions/clear-cache`, {
      method: 'POST',
      body: packageName ? JSON.stringify({ package_name: packageName }) : undefined,
    });
  },

  getDeviceActionHistory: (
    deviceId: number,
    limit?: number
  ): Promise<DeviceActionHistoryResponse> => {
    const params = limit ? `?limit=${limit}` : '';
    return fetchAPI<DeviceActionHistoryResponse>(`/devices/${deviceId}/actions/history${params}`);
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
