import { describe, expect, it } from 'vitest';
import {
  parseAnomalyList,
  parseBaselineFeatures,
  parseBaselineSuggestions,
  parseGroupedAnomalies,
  parseIsolationForestStats,
} from '../src/api/contracts';

describe('api contracts', () => {
  it('parses isolation forest stats with defaults', () => {
    const payload = {
      config: {
        n_estimators: 256,
        contamination: 0.04,
        random_state: 42,
        scale_features: true,
        min_variance: 0.01,
        feature_count: 18,
        model_type: 'isolation_forest',
      },
      defaults: {
        n_estimators: 100,
        contamination: 0.03,
        random_state: 99,
        scale_features: false,
        min_variance: 0.02,
        feature_count: null,
        model_type: 'isolation_forest',
      },
      score_distribution: {
        bins: [
          { bin_start: -0.2, bin_end: 0.0, count: 10, is_anomaly: false },
          { bin_start: -1.0, bin_end: -0.2, count: 2, is_anomaly: true },
        ],
        total_normal: 10,
        total_anomalies: 2,
        mean_score: -0.12,
        median_score: -0.1,
        min_score: -0.9,
        max_score: 0.1,
        std_score: 0.2,
      },
      total_predictions: 12,
      anomaly_rate: 0.1667,
    };

    const result = parseIsolationForestStats(payload);
    expect(result.config.n_estimators).toBe(256);
    expect(result.defaults?.n_estimators).toBe(100);
  });

  it('parses baseline suggestions and normalizes group_key', () => {
    const payload = [
      {
        level: 'global',
        group_key: { __all__: 'all' },
        feature: 'BatteryDrop',
        baseline_median: 10,
        observed_median: 14,
        proposed_new_median: 12,
        rationale: 'Median drifted above baseline',
      },
    ];

    const result = parseBaselineSuggestions(payload);
    expect(result[0].group_key).toContain('__all__');
  });

  it('parses baseline features', () => {
    const payload = [
      {
        feature: 'BatteryDrop',
        baseline: 10,
        observed: 12,
        unit: '%/day',
        status: 'warning',
        drift_percent: 20,
        mad: 1.5,
        sample_count: 1200,
        last_updated: null,
      },
    ];

    const result = parseBaselineFeatures(payload);
    expect(result[0].feature).toBe('BatteryDrop');
    expect(result[0].status).toBe('warning');
  });

  it('parses anomaly list responses', () => {
    const payload = {
      anomalies: [
        {
          id: 1,
          device_id: 99,
          timestamp: '2025-01-01T00:00:00Z',
          anomaly_score: -0.55,
          anomaly_label: -1,
          status: 'open',
        },
      ],
      total: 1,
      page: 1,
      page_size: 50,
      total_pages: 1,
    };

    const result = parseAnomalyList(payload);
    expect(result.anomalies).toHaveLength(1);
    expect(result.anomalies[0].device_id).toBe(99);
  });

  it('parses grouped anomalies responses', () => {
    const payload = {
      groups: [
        {
          group_id: 'cohort-1',
          group_name: 'Warehouse A',
          group_category: 'battery',
          group_type: 'temporal_cluster',
          severity: 'high',
          total_count: 3,
          open_count: 2,
          device_count: 2,
          time_range_start: '2025-01-01T00:00:00Z',
          time_range_end: '2025-01-02T00:00:00Z',
          sample_anomalies: [
            {
              anomaly_id: 101,
              device_id: 5,
              anomaly_score: -0.6,
              severity: 'high',
              status: 'open',
              timestamp: '2025-01-01T01:00:00Z',
            },
          ],
          grouping_factors: ['location'],
        },
      ],
      total_anomalies: 3,
      total_groups: 1,
      ungrouped_count: 0,
      ungrouped_anomalies: [],
      grouping_method: 'smart_auto',
      computed_at: '2025-01-02T12:00:00Z',
    };

    const result = parseGroupedAnomalies(payload);
    expect(result.groups).toHaveLength(1);
    expect(result.total_anomalies).toBe(3);
  });
});
