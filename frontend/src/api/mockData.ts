/**
 * Frontend Mock Data Provider for Stella Sentinel
 *
 * Provides comprehensive mock data generation for demo/testing purposes.
 * All mock data is consistent and internally coherent (device IDs match across endpoints, etc.)
 * This runs entirely on the frontend - no backend calls needed.
 */

import type {
    AllConnectionsStatus,
    Anomaly,
    AnomalyDetail,
    AnomalyListResponse,
    DashboardStats,
    DashboardTrend,
    DeviceDetail,
    IsolationForestStats,
    BaselineSuggestion,
    LocationHeatmapResponse,
    LLMConfig,
    LLMModelsResponse,
    LLMTestResult,
    BaselineAdjustmentResponse,
    OllamaPullResponse,
    // Investigation Panel types
    InvestigationPanel,
    FeatureContribution,
    BaselineMetric,
    EvidenceEvent,
    AIAnalysis,
    RootCauseHypothesis,
    RemediationSuggestion,
    SimilarCase,
    HistoricalTimeline,
    TimeSeriesDataPoint,
    // Smart Grouping types
    GroupedAnomaliesResponse,
    AnomalyGroup,
    AnomalyGroupMember,
    Severity,
} from '../types/anomaly';

// Seeded random for consistent data
class SeededRandom {
    private seed: number;

    constructor(seed: number = 42) {
        this.seed = seed;
    }

    next(): number {
        this.seed = (this.seed * 1103515245 + 12345) & 0x7fffffff;
        return this.seed / 0x7fffffff;
    }

    int(min: number, max: number): number {
        return Math.floor(this.next() * (max - min + 1)) + min;
    }

    float(min: number, max: number): number {
        return this.next() * (max - min) + min;
    }

    choice<T>(arr: T[]): T {
        return arr[Math.floor(this.next() * arr.length)];
    }

    // Reset seed for consistent data across calls
    reset() {
        this.seed = 42;
    }
}

const rng = new SeededRandom(42);

// ============================================================================
// Mock Data Constants
// ============================================================================

const MOCK_STORES = [
    { id: 'store-001', name: 'Downtown Flagship', region: 'Northeast' },
    { id: 'store-002', name: 'Westside Mall', region: 'West' },
    { id: 'store-003', name: 'Harbor Point', region: 'Southeast' },
    { id: 'store-004', name: 'Tech Plaza', region: 'West' },
    { id: 'store-005', name: 'Central Station', region: 'Midwest' },
    { id: 'store-006', name: 'Riverside Center', region: 'Northeast' },
    { id: 'store-007', name: 'Airport Terminal', region: 'Southeast' },
    { id: 'store-008', name: 'University District', region: 'Midwest' },
];

const MOCK_DEVICE_MODELS = [
    'Samsung Galaxy Tab A8',
    'iPad Pro 11',
    'Zebra TC52',
    'Honeywell CT60',
    'Samsung Galaxy XCover 6',
    'Panasonic Toughbook N1',
];

const MOCK_ANOMALY_TYPES = [
    'battery_drain',
    'storage_critical',
    'network_instability',
    'offline_extended',
    'data_spike',
];

// ============================================================================
// Mock Device Generation
// ============================================================================

interface MockDevice {
    device_id: number;
    device_name: string;
    device_model: string;
    location: string;
    region: string;
    store_id: string;
    status: string;
    battery: number;
    is_charging: boolean;
    wifi_signal: number; // dBm
    storage_used: number; // percentage
    memory_usage: number; // percentage
    cpu_load: number; // percentage
    last_seen: string;
    os_version: string;
    agent_version: string;
    custom_attributes: Record<string, string>;
}

function generateMockDevices(count: number = 250): MockDevice[] {
    rng.reset();
    const devices: MockDevice[] = [];

    for (let i = 1; i <= count; i++) {
        const store = MOCK_STORES[i % MOCK_STORES.length];
        const model = MOCK_DEVICE_MODELS[i % MOCK_DEVICE_MODELS.length];

        // Determine status with realistic distribution
        const statusRoll = rng.next();
        let status: string;
        if (statusRoll < 0.75) {
            status = 'Active';
        } else if (statusRoll < 0.90) {
            status = 'Idle';
        } else if (statusRoll < 0.95) {
            status = 'Offline';
        } else {
            status = 'Charging';
        }

        const minutesAgo = rng.int(0, 1440);
        const lastSeen = new Date(Date.now() - minutesAgo * 60 * 1000);

        devices.push({
            device_id: i,
            device_name: `Device-${String(i).padStart(4, '0')}`,
            device_model: model,
            location: store.name,
            region: store.region,
            store_id: store.id,
            status,
            battery: rng.int(20, 100),
            is_charging: status === 'Charging',
            wifi_signal: rng.int(-85, -40), // dBm realistic range
            storage_used: rng.int(20, 95),
            memory_usage: rng.int(30, 90),
            cpu_load: rng.int(5, 80),
            last_seen: lastSeen.toISOString(),
            os_version: `Android ${rng.int(10, 14)}`,
            agent_version: `15.${rng.int(0, 5)}.${rng.int(0, 9999)}`,
            custom_attributes: {
                'Department': rng.choice(['Sales', 'Warehouse', 'Logistics']),
                'Zone': `Zone-${rng.choice(['A', 'B', 'C'])}`,
            },
        });
    }


    return devices;
}

// Helper to calculate status based on telemetry
function calculateDeviceStatus(d: MockDevice): string {
    if (d.battery < 15 && !d.is_charging) return 'Warning (Low Battery)';
    if (d.storage_used > 95) return 'Critical (Storage)';
    if (d.storage_used > 85) return 'Warning (Storage)';
    // Add more telemetry-based status logic here if needed
    return d.status;
}

export function getMockDevices(): DeviceDetail[] {
    return MOCK_DEVICES.map(d => {
        const deviceAnomalies = MOCK_ANOMALIES.filter(
            (a) => a.device_id === d.device_id && (a.status === 'new' || a.status === 'investigating' || a.status === 'open')
        );

        let derivedStatus = calculateDeviceStatus(d);
        if (deviceAnomalies.some(a => a.anomaly_score > 0.8)) {
            derivedStatus = 'Critical';
        } else if (deviceAnomalies.length > 0) {
            derivedStatus = 'Warning';
        }

        return {
            device_id: d.device_id,
            device_model: d.device_model,
            device_name: d.device_name,
            location: d.location,
            status: derivedStatus,
            last_seen: d.last_seen,
            anomaly_count: deviceAnomalies.length,
            recent_anomalies: deviceAnomalies.slice(0, 3),
            custom_attributes: d.custom_attributes,
            store_id: d.store_id,
        };
    });
}

// Cached mock devices for consistency
const MOCK_DEVICES = generateMockDevices(250);

// ============================================================================
// Mock Anomaly Generation
// ============================================================================

function generateMockAnomalies(count: number = 100): Anomaly[] {
    rng.reset();
    const anomalies: Anomaly[] = [];
    const baseTime = new Date();

    for (let i = 1; i <= count; i++) {
        const device = rng.choice(MOCK_DEVICES);
        const anomalyType = rng.choice(MOCK_ANOMALY_TYPES);

        // Status distribution
        const statusRoll = rng.next();
        let status: string;
        if (statusRoll < 0.4) {
            status = 'new';
        } else if (statusRoll < 0.7) {
            status = 'investigating';
        } else if (statusRoll < 0.9) {
            status = 'resolved';
        } else {
            status = 'dismissed';
        }

        // Generate realistic score (-0.4 to -0.99 for anomalies, lower is worse)
        // Critical: < -0.7, High: < -0.5, Medium: < -0.3
        const score = Math.round(rng.float(-0.99, -0.4) * 1000) / 1000;

        // Time offset (spread over past 14 days)
        const daysOffset = rng.int(0, 13);
        const hoursOffset = rng.int(0, 23);
        const minutesOffset = rng.int(0, 59);
        const timestamp = new Date(
            baseTime.getTime() -
            (daysOffset * 24 * 60 + hoursOffset * 60 + minutesOffset) * 60 * 1000
        );

        anomalies.push({
            id: i,
            device_id: device.device_id,
            timestamp: timestamp.toISOString(),
            anomaly_score: score,
            anomaly_label: -1,
            status,
            assigned_to: rng.choice([null, 'Admin', 'Operator', 'System']),
            total_battery_level_drop:
                anomalyType === 'battery_drain' ? rng.int(10, 80) : rng.int(5, 20),
            total_free_storage_kb:
                anomalyType === 'storage_critical'
                    ? rng.int(50000, 500000)
                    : rng.int(1000000, 5000000),
            download:
                anomalyType === 'data_spike'
                    ? rng.int(100000, 5000000)
                    : rng.int(10000, 500000),
            upload:
                anomalyType === 'data_spike'
                    ? rng.int(50000, 2000000)
                    : rng.int(5000, 100000),
            offline_time:
                anomalyType === 'offline_extended' ? rng.int(60, 480) : rng.int(0, 30),
            disconnect_count:
                anomalyType === 'network_instability'
                    ? rng.int(5, 25)
                    : rng.int(0, 5),
            wifi_signal_strength:
                anomalyType === 'network_instability'
                    ? rng.int(10, 40)
                    : rng.int(50, 90),
            connection_time: rng.int(30, 120),
            feature_values_json: null,
            created_at: timestamp.toISOString(),
            updated_at: timestamp.toISOString(),
        });
    }

    // Sort by timestamp descending
    anomalies.sort(
        (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );
    return anomalies;
}

const MOCK_ANOMALIES = generateMockAnomalies(100);

// ============================================================================
// Public Mock Data Functions
// ============================================================================

export function getMockDashboardStats(): DashboardStats {
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const todayAnomalies = MOCK_ANOMALIES.filter((a) => {
        const timestamp = new Date(a.timestamp);
        timestamp.setHours(0, 0, 0, 0);
        return timestamp.getTime() === today.getTime();
    });

    const openCases = MOCK_ANOMALIES.filter(
        (a) => a.status === 'new' || a.status === 'investigating' || a.status === 'open'
    ).length;

    // Count critical anomalies (score <= -0.7) that are still open
    const criticalIssues = MOCK_ANOMALIES.filter(
        (a) => a.anomaly_score <= -0.7 && (a.status === 'new' || a.status === 'investigating' || a.status === 'open')
    ).length;

    // Count resolved anomalies from today
    const resolvedToday = MOCK_ANOMALIES.filter((a) => {
        const timestamp = new Date(a.timestamp);
        timestamp.setHours(0, 0, 0, 0);
        return timestamp.getTime() === today.getTime() && a.status === 'resolved';
    }).length;

    return {
        anomalies_today: todayAnomalies.length || 5,
        devices_monitored: MOCK_DEVICES.length,
        critical_issues: criticalIssues,
        resolved_today: resolvedToday || 3,
        open_cases: openCases,
    };
}

export function getMockDashboardTrends(
    days: number = 7,
    startDate?: string,
    endDate?: string
): DashboardTrend[] {
    rng.reset();
    const end = endDate ? new Date(endDate) : new Date();
    const start = startDate
        ? new Date(startDate)
        : new Date(end.getTime() - days * 24 * 60 * 60 * 1000);

    const trends: DashboardTrend[] = [];
    const current = new Date(start);
    current.setHours(0, 0, 0, 0);

    while (current <= end) {
        // Generate realistic daily counts with some pattern
        const baseCount = 8;
        const dayOfWeek = current.getDay();

        // Higher on weekdays
        let count: number;
        if (dayOfWeek >= 1 && dayOfWeek <= 5) {
            count = baseCount + rng.int(2, 8);
        } else {
            count = baseCount + rng.int(-2, 3);
        }

        trends.push({
            date: current.toISOString().split('T')[0],
            anomaly_count: Math.max(0, count),
        });

        current.setDate(current.getDate() + 1);
    }

    return trends;
}

export function getMockConnectionStatus(): AllConnectionsStatus {
    return {
        backend_db: {
            connected: true,
            server: 'postgres:5432',
            error: null,
            status: 'connected',
        },
        dw_sql: {
            connected: true,
            server: 'mock-dw-server.local',
            error: null,
            status: 'connected',
        },
        mc_sql: {
            connected: true,
            server: 'mock-mc-server.local',
            error: null,
            status: 'connected',
        },
        mobicontrol_api: {
            connected: true,
            server: 'https://mock.mobicontrol.net/MobiControl',
            error: null,
            status: 'connected',
        },
        llm: {
            connected: true,
            server: 'http://localhost:11434',
            error: null,
            status: 'connected',
        },
        redis: {
            connected: true,
            server: 'redis://redis:6379',
            error: null,
            status: 'connected',
        },
        qdrant: {
            connected: true,
            server: 'qdrant:6333',
            error: null,
            status: 'connected',
        },
        last_checked: new Date().toISOString(),
    };
}

export function getMockAnomalies(params: {
    device_id?: number;
    status?: string;
    page?: number;
    page_size?: number;
}): AnomalyListResponse {
    let filtered = [...MOCK_ANOMALIES];

    if (params.device_id) {
        filtered = filtered.filter((a) => a.device_id === params.device_id);
    }

    if (params.status) {
        filtered = filtered.filter((a) => a.status === params.status);
    }

    const page = params.page || 1;
    const pageSize = params.page_size || 50;
    const total = filtered.length;
    const start = (page - 1) * pageSize;
    const end = start + pageSize;

    return {
        anomalies: filtered.slice(start, end),
        total,
        page,
        page_size: pageSize,
        total_pages: Math.ceil(total / pageSize),
    };
}

export function getMockAnomalyDetail(anomalyId: number): AnomalyDetail | null {
    // First check if it's in the main MOCK_ANOMALIES array
    const anomaly = MOCK_ANOMALIES.find((a) => a.id === anomalyId);
    if (anomaly) {
        return {
            ...anomaly,
            notes: 'Mock anomaly for demonstration purposes.',
            investigation_notes: [
                {
                    id: 1,
                    user: 'System',
                    note: 'Anomaly detected by Isolation Forest model with high confidence.',
                    action_type: 'detection',
                    created_at: anomaly.created_at,
                },
                {
                    id: 2,
                    user: 'Admin',
                    note: 'Reviewing device telemetry patterns.',
                    action_type: 'investigation',
                    created_at: new Date().toISOString(),
                },
            ],
        };
    }

    // Check if this ID comes from grouped anomalies (IDs like 1001, 2001, 3001, etc.)
    const groupedData = getMockGroupedAnomalies();

    // Search in all group sample_anomalies
    for (const group of groupedData.groups) {
        const groupedAnomaly = group.sample_anomalies.find(a => a.anomaly_id === anomalyId);
        if (groupedAnomaly) {
            return createAnomalyDetailFromGroupMember(groupedAnomaly, group.group_name);
        }
    }

    // Search in ungrouped anomalies
    const ungroupedAnomaly = groupedData.ungrouped_anomalies?.find(a => a.anomaly_id === anomalyId);
    if (ungroupedAnomaly) {
        return createAnomalyDetailFromGroupMember(ungroupedAnomaly, 'Ungrouped');
    }

    return null;
}

// Helper function to create AnomalyDetail from AnomalyGroupMember
function createAnomalyDetailFromGroupMember(member: AnomalyGroupMember, groupName: string): AnomalyDetail {
    const now = new Date();
    const metricValues = generateMetricValuesForGroupMember(member);

    return {
        id: member.anomaly_id,
        device_id: member.device_id,
        timestamp: member.timestamp,
        anomaly_score: member.anomaly_score,
        anomaly_label: -1,
        status: member.status,
        assigned_to: null,
        total_battery_level_drop: metricValues.total_battery_level_drop,
        total_free_storage_kb: metricValues.total_free_storage_kb,
        download: metricValues.download,
        upload: metricValues.upload,
        offline_time: metricValues.offline_time,
        disconnect_count: metricValues.disconnect_count,
        wifi_signal_strength: metricValues.wifi_signal_strength,
        connection_time: metricValues.connection_time,
        feature_values_json: null,
        created_at: member.timestamp,
        updated_at: now.toISOString(),
        notes: `Mock anomaly from smart group: ${groupName}`,
        investigation_notes: [
            {
                id: 1,
                user: 'System',
                note: `Anomaly detected by Isolation Forest model. Part of group: ${groupName}`,
                action_type: 'detection',
                created_at: member.timestamp,
            },
            {
                id: 2,
                user: 'Admin',
                note: `Primary metric: ${member.primary_metric || 'unknown'}. Location: ${member.location || 'unknown'}`,
                action_type: 'investigation',
                created_at: now.toISOString(),
            },
        ],
    };
}

// Generate realistic metric values based on the group member's primary metric
function generateMetricValuesForGroupMember(member: AnomalyGroupMember): {
    total_battery_level_drop: number;
    total_free_storage_kb: number;
    download: number;
    upload: number;
    offline_time: number;
    disconnect_count: number;
    wifi_signal_strength: number;
    connection_time: number;
} {
    // Use anomaly_id as seed for consistent but varied values
    const seed = member.anomaly_id;
    const baseVariation = (seed % 10) / 10;

    // Set abnormal values based on primary_metric
    const isAbnormal = (metric: string) => member.primary_metric === metric;

    return {
        total_battery_level_drop: isAbnormal('total_battery_level_drop')
            ? 35 + Math.floor(baseVariation * 40) // 35-75% (abnormal)
            : 5 + Math.floor(baseVariation * 15),  // 5-20% (normal)
        total_free_storage_kb: isAbnormal('total_free_storage_kb')
            ? 100000 + Math.floor(baseVariation * 400000) // 100KB-500KB (critical low)
            : 2000000 + Math.floor(baseVariation * 3000000), // 2GB-5GB (normal)
        download: isAbnormal('download')
            ? 500000 + Math.floor(baseVariation * 4500000) // 500KB-5MB (high)
            : 10000 + Math.floor(baseVariation * 100000),   // 10KB-110KB (normal)
        upload: isAbnormal('upload')
            ? 200000 + Math.floor(baseVariation * 1800000) // 200KB-2MB (high)
            : 5000 + Math.floor(baseVariation * 50000),    // 5KB-55KB (normal)
        offline_time: isAbnormal('offline_time')
            ? 60 + Math.floor(baseVariation * 180)  // 60-240 min (high)
            : Math.floor(baseVariation * 20),       // 0-20 min (normal)
        disconnect_count: isAbnormal('disconnect_count')
            ? 8 + Math.floor(baseVariation * 17)    // 8-25 (high)
            : Math.floor(baseVariation * 3),        // 0-3 (normal)
        wifi_signal_strength: isAbnormal('wifi_signal_strength')
            ? -85 + Math.floor(baseVariation * 15)  // -85 to -70 dBm (weak)
            : -60 + Math.floor(baseVariation * 20), // -60 to -40 dBm (good)
        connection_time: isAbnormal('connection_time')
            ? 120 + Math.floor(baseVariation * 180) // 120-300s (slow)
            : 30 + Math.floor(baseVariation * 60),  // 30-90s (normal)
    };
}

export function getMockDeviceDetail(deviceId: number): DeviceDetail | null {
    const device = MOCK_DEVICES.find((d) => d.device_id === deviceId);
    if (!device) return null;

    const deviceAnomalies = MOCK_ANOMALIES.filter(
        (a) => a.device_id === deviceId
    );

    return {
        device_id: device.device_id,
        device_model: device.device_model,
        device_name: device.device_name,
        location: device.location,
        status: device.status,
        last_seen: device.last_seen,
        anomaly_count: deviceAnomalies.length,
        recent_anomalies: deviceAnomalies.slice(0, 5),
        // Telemetry
        battery_level: device.battery,
        is_charging: device.is_charging,
        wifi_signal: device.wifi_signal,
        storage_used: device.storage_used,
        memory_usage: device.memory_usage,
        cpu_load: device.cpu_load,
        os_version: device.os_version,
        agent_version: device.agent_version,
    };
}

export function getMockIsolationForestStats(): IsolationForestStats {
    rng.reset();
    // Generate high-resolution score distribution (50 bins) from -1.0 to 1.0
    const bins = [];
    let totalNormal = 0;
    let totalAnomalies = 0;
    const step = 0.04; // Range 2.0 / 50 bins = 0.04

    for (let i = 0; i < 50; i++) {
        const x = -1.0 + (i * step) + (step / 2); // Center of bin

        // Mixture of two Gaussians: Anomaly centered at -0.65, Normal centered at 0.4
        const anomaly = 45 * Math.exp(-Math.pow(x - (-0.65), 2) / 0.03); // Tighter anomaly cluster
        const normal = 250 * Math.exp(-Math.pow(x - 0.4, 2) / 0.1); // Broader normal cluster
        const noise = rng.int(0, 5);

        const count = Math.max(0, Math.round(normal + anomaly + noise));
        const isAnomaly = x < 0; // Threshold at 0

        if (isAnomaly) totalAnomalies += count;
        else totalNormal += count;

        bins.push({
            bin_start: Number((x - step / 2).toFixed(2)),
            bin_end: Number((x + step / 2).toFixed(2)),
            count,
            is_anomaly: isAnomaly,
        });
    }

    const total = totalNormal + totalAnomalies;

    return {
        config: {
            n_estimators: 100,
            contamination: 0.05,
            random_state: 42,
            scale_features: true,
            min_variance: 0.01,
            feature_count: 8,
            model_type: 'IsolationForest',
        },
        score_distribution: {
            bins,
            total_normal: totalNormal,
            total_anomalies: totalAnomalies,
            mean_score: 0.15,
            median_score: 0.25,
            min_score: -0.95,
            max_score: 0.98,
            std_score: 0.45,
        },
        total_predictions: total,
        anomaly_rate: totalAnomalies / total,
        feedback_stats: {
            total_feedback: 142,
            false_positives: 18,
            confirmed_anomalies: 124,
            projected_accuracy_gain: 2.15,
            last_retrain: new Date(Date.now() - 1000 * 60 * 60 * 12).toISOString(), // 12h ago
        }
    };
}

export function getMockBaselineSuggestions(): BaselineSuggestion[] {
    const features = [
        {
            feature: 'TotalBatteryLevelDrop',
            level: 'Global',
            group_key: 'global',
            baseline: 15,
            observed: 18,
            proposed: 17,
        },
        {
            feature: 'OfflineTime',
            level: 'Global',
            group_key: 'global',
            baseline: 10,
            observed: 14,
            proposed: 12,
        },
        {
            feature: 'Download',
            level: 'Store: Downtown Flagship',
            group_key: 'store-001',
            baseline: 150000,
            observed: 220000,
            proposed: 180000,
        },
        {
            feature: 'WiFiSignalStrength',
            level: 'Region: Northeast',
            group_key: 'northeast',
            baseline: 65,
            observed: 58,
            proposed: 60,
        },
    ];

    return features.map((f) => ({
        level: f.level,
        group_key: f.group_key,
        feature: f.feature,
        baseline_median: f.baseline,
        observed_median: f.observed,
        proposed_new_median: f.proposed,
        rationale: `Observed ${f.feature} has drifted from baseline. Recommend adjusting threshold to reduce false positives.`,
    }));
}

export function getMockLocationHeatmap(
    attributeName?: string
): LocationHeatmapResponse {
    // rng.reset(); // Don't reset if we want consistent anomalies from the other function

    // Calculate actual stats from MOCK_DEVICES and MOCK_ANOMALIES
    const locations = MOCK_STORES.map((store) => {
        const storeDevices = MOCK_DEVICES.filter(d => d.store_id === store.id);
        const deviceCount = storeDevices.length;

        // Count active devices
        const activeCount = storeDevices.filter(d => d.status.toLowerCase() === 'active').length;
        const utilization = deviceCount > 0 ? Math.round((activeCount / deviceCount) * 1000) / 10 : 0;
        const baseline = 80; // Statistic baseline

        // Count anomalies for this store
        const storeAnomalies = MOCK_ANOMALIES.filter(a => {
            const device = MOCK_DEVICES.find(d => d.device_id === a.device_id);
            return device && device.store_id === store.id &&
                (a.status === 'new' || a.status === 'investigating' || a.status === 'open');
        });

        return {
            id: store.id,
            name: store.name,
            utilization,
            baseline,
            deviceCount,
            activeDeviceCount: activeCount,
            region: store.region,
            anomalyCount: storeAnomalies.length,
        };
    });

    return {
        locations,
        attributeName: attributeName || 'Store',
        totalLocations: locations.length,
        totalDevices: locations.reduce((sum, loc) => sum + loc.deviceCount, 0),
    };
}

export function getMockCustomAttributes(): {
    custom_attributes: string[];
    error?: string;
} {
    return {
        custom_attributes: [
            'Store',
            'Region',
            'Department',
            'Zone',
            'Warehouse',
        ],
        error: undefined,
    };
}

export function getMockLLMConfig(): LLMConfig {
    return {
        provider: 'ollama',
        model_name: 'deepseek/deepseek-r1-0528-qwen3-8b',
        base_url: 'http://localhost:11434',
        api_key_set: false,
        api_version: null,
        is_connected: true,
        available_models: ['deepseek/deepseek-r1-0528-qwen3-8b', 'llama2:7b', 'mistral:7b'],
        active_model: 'deepseek/deepseek-r1-0528-qwen3-8b',
    };
}

export function getMockLLMModels(): LLMModelsResponse {
    return {
        models: [
            {
                id: 'deepseek/deepseek-r1-0528-qwen3-8b',
                name: 'DeepSeek R1 Qwen 8B',
                size: '8B',
            },
            {
                id: 'llama2:7b',
                name: 'Llama 2 7B',
                size: '7B',
            },
            {
                id: 'mistral:7b',
                name: 'Mistral 7B',
                size: '7B',
            },
        ],
        active_model: 'deepseek/deepseek-r1-0528-qwen3-8b',
    };
}

export function getMockLLMTestResult(): LLMTestResult {
    return {
        success: true,
        message: 'LLM connection successful (mock mode)',
        response_time_ms: rng.int(50, 200),
        model_used: 'deepseek/deepseek-r1-0528-qwen3-8b',
    };
}

export function getMockBaselineAdjustmentResponse(): BaselineAdjustmentResponse {
    return {
        success: true,
        message: 'Baseline adjustment applied successfully (mock mode)',
        baseline_updated: true,
        model_retrained: false,
    };
}

export function getMockOllamaPullResponse(modelName: string): OllamaPullResponse {
    return {
        success: true,
        message: `Model ${modelName} pulled successfully (mock mode)`,
        model_name: modelName,
    };
}

// ============================================================================
// Data Discovery Mock Data
// ============================================================================

import type {
    TableProfile,
    AvailableMetric,
    MetricDistribution,
    DataDiscoveryStatus,
    TemporalPattern,
    DiscoverySummary,
} from '../types/training';

export function getMockTableProfiles(): TableProfile[] {
    return [
        {
            table_name: 'cs_BatteryStat',
            row_count: 1_250_000,
            date_range: ['2024-01-01', '2024-12-28'],
            device_count: 4_500,
            column_stats: {
                TotalBatteryLevelDrop: {
                    column_name: 'TotalBatteryLevelDrop',
                    dtype: 'int',
                    null_count: 6250,
                    null_percent: 0.5,
                    unique_count: 100,
                    min_val: 0,
                    max_val: 100,
                    mean: 35.2,
                    std: 18.5,
                    percentiles: { p5: 5, p25: 20, p50: 32, p75: 48, p95: 72, p99: 88 },
                },
                TotalDischargeTime_Sec: {
                    column_name: 'TotalDischargeTime_Sec',
                    dtype: 'int',
                    null_count: 2500,
                    null_percent: 0.2,
                    unique_count: 5000,
                    min_val: 0,
                    max_val: 86400,
                    mean: 28800,
                    std: 12000,
                    percentiles: { p5: 3600, p25: 18000, p50: 28800, p75: 39600, p95: 57600, p99: 72000 },
                },
                TotalFreeStorageKb: {
                    column_name: 'TotalFreeStorageKb',
                    dtype: 'bigint',
                    null_count: 1250,
                    null_percent: 0.1,
                    unique_count: 100000,
                    min_val: 0,
                    max_val: 16_000_000,
                    mean: 2_000_000,
                    std: 1_500_000,
                    percentiles: { p5: 200000, p25: 800000, p50: 1800000, p75: 3000000, p95: 5000000, p99: 8000000 },
                },
            },
            profiled_at: new Date().toISOString(),
        },
        {
            table_name: 'cs_AppUsage',
            row_count: 3_500_000,
            date_range: ['2024-01-01', '2024-12-28'],
            device_count: 4_500,
            column_stats: {
                VisitCount: {
                    column_name: 'VisitCount',
                    dtype: 'int',
                    null_count: 3500,
                    null_percent: 0.1,
                    unique_count: 500,
                    min_val: 0,
                    max_val: 1000,
                    mean: 45,
                    std: 32,
                    percentiles: { p5: 2, p25: 18, p50: 38, p75: 65, p95: 120, p99: 200 },
                },
                TotalForegroundTime: {
                    column_name: 'TotalForegroundTime',
                    dtype: 'int',
                    null_count: 7000,
                    null_percent: 0.2,
                    unique_count: 8000,
                    min_val: 0,
                    max_val: 28800,
                    mean: 3600,
                    std: 2400,
                    percentiles: { p5: 120, p25: 1200, p50: 3000, p75: 5400, p95: 9000, p99: 14400 },
                },
            },
            profiled_at: new Date().toISOString(),
        },
        {
            table_name: 'cs_DataUsage',
            row_count: 2_800_000,
            date_range: ['2024-01-01', '2024-12-28'],
            device_count: 4_500,
            column_stats: {
                Download: {
                    column_name: 'Download',
                    dtype: 'bigint',
                    null_count: 2800,
                    null_percent: 0.1,
                    unique_count: 500000,
                    min_val: 0,
                    max_val: 10_000_000_000,
                    mean: 150_000_000,
                    std: 250_000_000,
                    percentiles: { p5: 1000, p25: 10_000_000, p50: 80_000_000, p75: 200_000_000, p95: 600_000_000, p99: 1_500_000_000 },
                },
                Upload: {
                    column_name: 'Upload',
                    dtype: 'bigint',
                    null_count: 2800,
                    null_percent: 0.1,
                    unique_count: 300000,
                    min_val: 0,
                    max_val: 2_000_000_000,
                    mean: 25_000_000,
                    std: 50_000_000,
                    percentiles: { p5: 100, p25: 1_000_000, p50: 10_000_000, p75: 30_000_000, p95: 100_000_000, p99: 300_000_000 },
                },
            },
            profiled_at: new Date().toISOString(),
        },
        {
            table_name: 'cs_Heatmap',
            row_count: 5_200_000,
            date_range: ['2024-01-01', '2024-12-28'],
            device_count: 4_500,
            column_stats: {
                SignalStrength: {
                    column_name: 'SignalStrength',
                    dtype: 'int',
                    null_count: 130000,
                    null_percent: 2.5,
                    unique_count: 80,
                    min_val: -120,
                    max_val: -40,
                    mean: -72,
                    std: 15,
                    percentiles: { p5: -98, p25: -82, p50: -72, p75: -62, p95: -50, p99: -44 },
                },
                DropCnt: {
                    column_name: 'DropCnt',
                    dtype: 'int',
                    null_count: 52000,
                    null_percent: 1.0,
                    unique_count: 50,
                    min_val: 0,
                    max_val: 100,
                    mean: 2.5,
                    std: 5.2,
                    percentiles: { p5: 0, p25: 0, p50: 1, p75: 3, p95: 12, p99: 25 },
                },
            },
            profiled_at: new Date().toISOString(),
        },
    ];
}

export function getMockAvailableMetrics(): AvailableMetric[] {
    return [
        // Raw database metrics
        { table: 'cs_BatteryStat', column: 'TotalBatteryLevelDrop', dtype: 'int', mean: 35.2, std: 18.5, min: 0, max: 100, category: 'raw', domain: 'battery' },
        { table: 'cs_BatteryStat', column: 'TotalDischargeTime_Sec', dtype: 'int', mean: 28800, std: 12000, min: 0, max: 86400, category: 'raw', domain: 'battery' },
        { table: 'cs_BatteryStat', column: 'TotalFreeStorageKb', dtype: 'bigint', mean: 2_000_000, std: 1_500_000, min: 0, max: 16_000_000, category: 'raw', domain: 'storage' },
        { table: 'cs_AppUsage', column: 'VisitCount', dtype: 'int', mean: 45, std: 32, min: 0, max: 1000, category: 'raw', domain: 'usage' },
        { table: 'cs_AppUsage', column: 'TotalForegroundTime', dtype: 'int', mean: 3600, std: 2400, min: 0, max: 28800, category: 'raw', domain: 'usage' },
        { table: 'cs_DataUsage', column: 'Download', dtype: 'bigint', mean: 150_000_000, std: 250_000_000, min: 0, max: 10_000_000_000, category: 'raw', domain: 'throughput' },
        { table: 'cs_DataUsage', column: 'Upload', dtype: 'bigint', mean: 25_000_000, std: 50_000_000, min: 0, max: 2_000_000_000, category: 'raw', domain: 'throughput' },
        { table: 'cs_Heatmap', column: 'SignalStrength', dtype: 'int', mean: -72, std: 15, min: -120, max: -40, category: 'raw', domain: 'rf' },
        { table: 'cs_Heatmap', column: 'DropCnt', dtype: 'int', mean: 2.5, std: 5.2, min: 0, max: 100, category: 'raw', domain: 'rf' },
        // Engineered features
        { table: 'feature_engineered', column: 'TotalBatteryLevelDrop_roll_mean', dtype: 'float', mean: 35.1, std: 12.3, min: 2, max: 95, category: 'rolling', domain: 'battery', description: 'Mean of TotalBatteryLevelDrop over 14 days' },
        { table: 'feature_engineered', column: 'TotalBatteryLevelDrop_roll_std', dtype: 'float', mean: 8.2, std: 5.1, min: 0, max: 35, category: 'rolling', domain: 'battery', description: 'Std of TotalBatteryLevelDrop over 14 days' },
        { table: 'feature_engineered', column: 'BatteryDrainPerHour', dtype: 'float', mean: 4.5, std: 2.8, min: 0, max: 25, category: 'derived', domain: 'battery', description: 'Derived: TotalBatteryLevelDrop / (TotalDischargeTime_Sec / 3600 + 1)' },
        { table: 'feature_engineered', column: 'Download_delta', dtype: 'float', mean: 5_000_000, std: 50_000_000, min: -500_000_000, max: 500_000_000, category: 'delta', domain: 'throughput', description: 'Day-over-day change for Download' },
        { table: 'feature_engineered', column: 'hour_of_day', dtype: 'int', mean: 12, std: 6.9, min: 0, max: 23, category: 'temporal', domain: 'temporal', description: 'Temporal feature: hour of day' },
        { table: 'feature_engineered', column: 'day_of_week', dtype: 'int', mean: 3, std: 2, min: 0, max: 6, category: 'temporal', domain: 'temporal', description: 'Temporal feature: day of week' },
        { table: 'feature_engineered', column: 'TotalBatteryLevelDrop_cohort_z', dtype: 'float', mean: 0, std: 1, min: -3, max: 3, category: 'cohort', domain: 'battery', description: 'Cohort-normalized z-score for TotalBatteryLevelDrop' },
        { table: 'feature_engineered', column: 'Download_cv', dtype: 'float', mean: 1.2, std: 0.8, min: 0, max: 5, category: 'volatility', domain: 'throughput', description: 'Coefficient of variation (volatility) for Download' },
    ];
}

export function getMockMetricDistribution(metricName: string): MetricDistribution {
    // Generate deterministic mock distribution based on metric name
    const seed = metricName.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    const mean = 50 + (seed % 30);
    const std = 10 + (seed % 10);

    // Generate 30 bins
    const binCount = 30;
    const minVal = Math.max(0, mean - 3 * std);
    const maxVal = mean + 3 * std;
    const binWidth = (maxVal - minVal) / binCount;

    const bins: number[] = [];
    const counts: number[] = [];

    for (let i = 0; i <= binCount; i++) {
        bins.push(minVal + i * binWidth);
    }

    // Generate bell-curve-ish counts
    for (let i = 0; i < binCount; i++) {
        const binCenter = minVal + (i + 0.5) * binWidth;
        const zScore = (binCenter - mean) / std;
        const density = Math.exp(-0.5 * zScore * zScore);
        counts.push(Math.round(density * 1000 * (1 + Math.sin(seed + i) * 0.1)));
    }

    return {
        bins,
        counts,
        stats: {
            min: minVal,
            max: maxVal,
            mean,
            std,
            median: mean - 2,
            total_samples: counts.reduce((a, b) => a + b, 0),
        },
    };
}

export function getMockDiscoveryStatus(): DataDiscoveryStatus {
    return {
        status: 'completed',
        progress: 100,
        message: 'Discovery completed successfully (mock mode)',
        started_at: new Date(Date.now() - 300000).toISOString(), // 5 minutes ago
        completed_at: new Date(Date.now() - 60000).toISOString(), // 1 minute ago
        results_available: true,
    };
}

export function getMockTemporalPatterns(): TemporalPattern[] {
    const patterns: TemporalPattern[] = [];

    for (const metric of ['TotalBatteryLevelDrop', 'VisitCount', 'Download']) {
        const hourlyStats: Record<number, { mean: number; std: number; count: number }> = {};
        const dailyStats: Record<number, { mean: number; std: number; count: number }> = {};

        // Generate hourly patterns (higher during business hours)
        for (let hour = 0; hour < 24; hour++) {
            const isBusinessHour = hour >= 8 && hour <= 18;
            const baseMean = isBusinessHour ? 50 : 30;
            hourlyStats[hour] = {
                mean: baseMean + Math.sin(hour / 3) * 10,
                std: 8 + Math.random() * 4,
                count: 2000 + Math.floor(Math.random() * 1000),
            };
        }

        // Generate daily patterns (lower on weekends)
        for (let day = 0; day < 7; day++) {
            const isWeekend = day === 0 || day === 6;
            const baseMean = isWeekend ? 25 : 45;
            dailyStats[day] = {
                mean: baseMean + Math.sin(day) * 5,
                std: 10 + Math.random() * 5,
                count: 10000 + Math.floor(Math.random() * 5000),
            };
        }

        patterns.push({
            metric_name: metric,
            hourly_stats: hourlyStats,
            daily_stats: dailyStats,
        });
    }

    return patterns;
}

export function getMockDiscoverySummary(): DiscoverySummary {
    return {
        total_tables_profiled: 4,
        total_rows: 12_750_000,
        total_devices: 4_500,
        metrics_discovered: 17,
        patterns_analyzed: 3,
        date_range: { start: '2024-01-01', end: '2024-12-28' },
        discovery_completed: new Date().toISOString(),
    };
}

// ==========================================
// Investigation Panel Mock Data
// ==========================================

function generateFeatureContributions(): FeatureContribution[] {
    const features = [
        {
            feature_name: 'total_battery_level_drop',
            feature_display_name: 'Battery Drain',
            contribution_percentage: 42.5,
            contribution_direction: 'positive' as const,
            current_value: 58.3,
            current_value_display: '58.3%',
            baseline_value: 12.4,
            baseline_value_display: '12.4%',
            deviation_sigma: 4.2,
            percentile: 99.1,
            plain_text_explanation: 'Battery drain is 4.2 standard deviations above normal, representing the 99th percentile. This is the primary driver of the anomaly score.',
        },
        {
            feature_name: 'offline_time',
            feature_display_name: 'Offline Duration',
            contribution_percentage: 28.3,
            contribution_direction: 'positive' as const,
            current_value: 4.5,
            current_value_display: '4.5 hours',
            baseline_value: 0.8,
            baseline_value_display: '0.8 hours',
            deviation_sigma: 3.1,
            percentile: 97.8,
            plain_text_explanation: 'Device was offline for 4.5 hours compared to a typical 0.8 hours, suggesting network or power issues.',
        },
        {
            feature_name: 'download',
            feature_display_name: 'Download Traffic',
            contribution_percentage: 15.2,
            contribution_direction: 'positive' as const,
            current_value: 2450,
            current_value_display: '2.4 GB',
            baseline_value: 450,
            baseline_value_display: '450 MB',
            deviation_sigma: 2.8,
            percentile: 95.2,
            plain_text_explanation: 'Download traffic is significantly higher than baseline, possibly indicating large app updates or unusual data transfer.',
        },
        {
            feature_name: 'wifi_signal_strength',
            feature_display_name: 'WiFi Signal',
            contribution_percentage: 8.7,
            contribution_direction: 'negative' as const,
            current_value: -78,
            current_value_display: '-78 dBm',
            baseline_value: -55,
            baseline_value_display: '-55 dBm',
            deviation_sigma: 1.9,
            percentile: 88.5,
            plain_text_explanation: 'WiFi signal is weaker than normal, which may correlate with connectivity issues.',
        },
        {
            feature_name: 'disconnect_count',
            feature_display_name: 'Disconnections',
            contribution_percentage: 5.3,
            contribution_direction: 'positive' as const,
            current_value: 12,
            current_value_display: '12 disconnects',
            baseline_value: 2,
            baseline_value_display: '2 disconnects',
            deviation_sigma: 1.5,
            percentile: 82.3,
            plain_text_explanation: 'Device experienced 12 disconnections compared to an average of 2, indicating network instability.',
        },
    ];
    return features;
}

function generateBaselineMetrics(): BaselineMetric[] {
    return [
        {
            metric_name: 'total_battery_level_drop',
            metric_display_name: 'Battery Drain',
            metric_unit: '%',
            current_value: 58.3,
            current_value_display: '58.3%',
            baseline_mean: 12.4,
            baseline_std: 8.2,
            baseline_min: 2.1,
            baseline_max: 35.6,
            deviation_sigma: 4.2,
            deviation_percentage: 370.2,
            percentile_rank: 99.1,
            is_anomalous: true,
            anomaly_direction: 'above',
        },
        {
            metric_name: 'offline_time',
            metric_display_name: 'Offline Duration',
            metric_unit: 'hours',
            current_value: 4.5,
            current_value_display: '4.5 hours',
            baseline_mean: 0.8,
            baseline_std: 0.6,
            baseline_min: 0,
            baseline_max: 2.1,
            deviation_sigma: 3.1,
            deviation_percentage: 462.5,
            percentile_rank: 97.8,
            is_anomalous: true,
            anomaly_direction: 'above',
        },
        {
            metric_name: 'download',
            metric_display_name: 'Download Traffic',
            metric_unit: 'MB',
            current_value: 2450,
            current_value_display: '2.4 GB',
            baseline_mean: 450,
            baseline_std: 280,
            baseline_min: 50,
            baseline_max: 1200,
            deviation_sigma: 2.8,
            deviation_percentage: 444.4,
            percentile_rank: 95.2,
            is_anomalous: true,
            anomaly_direction: 'above',
        },
        {
            metric_name: 'wifi_signal_strength',
            metric_display_name: 'WiFi Signal',
            metric_unit: 'dBm',
            current_value: -78,
            current_value_display: '-78 dBm',
            baseline_mean: -55,
            baseline_std: 12,
            baseline_min: -75,
            baseline_max: -35,
            deviation_sigma: 1.9,
            deviation_percentage: 41.8,
            percentile_rank: 88.5,
            is_anomalous: false,
            anomaly_direction: 'below',
        },
        {
            metric_name: 'total_free_storage_kb',
            metric_display_name: 'Free Storage',
            metric_unit: 'GB',
            current_value: 2.1,
            current_value_display: '2.1 GB',
            baseline_mean: 8.5,
            baseline_std: 3.2,
            baseline_min: 1.5,
            baseline_max: 24.0,
            deviation_sigma: 2.0,
            deviation_percentage: -75.3,
            percentile_rank: 12.5,
            is_anomalous: true,
            anomaly_direction: 'below',
        },
    ];
}

function generateEvidenceEvents(anomalyId: number): EvidenceEvent[] {
    const baseTime = new Date();
    baseTime.setHours(baseTime.getHours() - 6);

    return [
        {
            event_id: `evt-${anomalyId}-001`,
            timestamp: new Date(baseTime.getTime() - 5 * 60 * 60 * 1000).toISOString(),
            event_type: 'app_install',
            event_category: 'apps',
            severity: 'medium',
            title: 'Large Application Installed',
            description: 'New application "DataSync Pro v3.2" (450MB) was installed',
            details: { app_name: 'DataSync Pro', version: '3.2', size_mb: 450 },
            is_contributing_event: true,
            contribution_note: 'Large app installation may explain elevated download traffic',
        },
        {
            event_id: `evt-${anomalyId}-002`,
            timestamp: new Date(baseTime.getTime() - 4 * 60 * 60 * 1000).toISOString(),
            event_type: 'battery_drain_spike',
            event_category: 'battery',
            severity: 'high',
            title: 'Rapid Battery Drain Detected',
            description: 'Battery dropped from 85% to 42% in 2 hours (21.5%/hour)',
            details: { start_level: 85, end_level: 42, duration_hours: 2, drain_rate: 21.5 },
            is_contributing_event: true,
            contribution_note: 'Primary indicator of anomalous device behavior',
        },
        {
            event_id: `evt-${anomalyId}-003`,
            timestamp: new Date(baseTime.getTime() - 3.5 * 60 * 60 * 1000).toISOString(),
            event_type: 'wifi_disconnect',
            event_category: 'network',
            severity: 'medium',
            title: 'WiFi Connection Lost',
            description: 'Device disconnected from "Store-WiFi-5G" network',
            details: { ssid: 'Store-WiFi-5G', signal_before: -58, reason: 'signal_lost' },
            is_contributing_event: true,
            contribution_note: 'Network instability correlates with offline time increase',
        },
        {
            event_id: `evt-${anomalyId}-004`,
            timestamp: new Date(baseTime.getTime() - 3 * 60 * 60 * 1000).toISOString(),
            event_type: 'storage_warning',
            event_category: 'storage',
            severity: 'low',
            title: 'Low Storage Warning',
            description: 'Free storage dropped below 3GB threshold',
            details: { free_storage_gb: 2.1, threshold_gb: 3.0 },
            is_contributing_event: false,
            contribution_note: null,
        },
        {
            event_id: `evt-${anomalyId}-005`,
            timestamp: new Date(baseTime.getTime() - 2 * 60 * 60 * 1000).toISOString(),
            event_type: 'background_sync',
            event_category: 'apps',
            severity: 'info',
            title: 'Background Sync Activity',
            description: 'DataSync Pro performed background data synchronization',
            details: { app_name: 'DataSync Pro', sync_size_mb: 1200, duration_minutes: 45 },
            is_contributing_event: true,
            contribution_note: 'Heavy background sync activity explains battery drain and data usage',
        },
    ];
}

function generateRootCauseHypothesis(isPrimary: boolean): RootCauseHypothesis {
    if (isPrimary) {
        return {
            hypothesis_id: 'hyp-001',
            title: 'Runaway Background Sync Process',
            description: 'A recently installed application (DataSync Pro) is performing excessive background synchronization, causing high battery drain and network usage.',
            likelihood: 0.82,
            evidence_for: [
                {
                    statement: 'Battery drain rate correlates with background sync activity timestamps',
                    strength: 'strong',
                    source: 'telemetry',
                    linked_event_id: 'evt-001-005',
                },
                {
                    statement: 'App was installed within 24 hours of anomaly detection',
                    strength: 'strong',
                    source: 'telemetry',
                    linked_event_id: 'evt-001-001',
                },
                {
                    statement: 'Similar pattern seen in 3 other devices with same app version',
                    strength: 'moderate',
                    source: 'pattern_match',
                    linked_event_id: null,
                },
            ],
            evidence_against: [
                {
                    statement: 'WiFi signal degradation could be independent issue',
                    strength: 'weak',
                    source: 'inference',
                    linked_event_id: null,
                },
            ],
            recommended_actions: [
                'Check DataSync Pro battery optimization settings',
                'Review app sync frequency configuration',
                'Consider updating to latest app version (3.3 available)',
                'Monitor device for 24 hours after configuration change',
            ],
        };
    }
    return {
        hypothesis_id: 'hyp-002',
        title: 'Network Infrastructure Issue',
        description: 'WiFi access point instability is causing frequent reconnections, leading to increased battery usage and incomplete data transfers.',
        likelihood: 0.45,
        evidence_for: [
            {
                statement: 'WiFi signal strength below normal range',
                strength: 'moderate',
                source: 'telemetry',
                linked_event_id: 'evt-001-003',
            },
            {
                statement: 'Multiple disconnection events recorded',
                strength: 'moderate',
                source: 'telemetry',
                linked_event_id: null,
            },
        ],
        evidence_against: [
            {
                statement: 'No other devices at this location showing similar network issues',
                strength: 'strong',
                source: 'pattern_match',
                linked_event_id: null,
            },
            {
                statement: 'Battery drain started before network issues',
                strength: 'moderate',
                source: 'telemetry',
                linked_event_id: null,
            },
        ],
        recommended_actions: [
            'Check access point health and logs',
            'Verify device is within optimal range of AP',
            'Test network connectivity from device location',
        ],
    };
}

function generateRemediations(): RemediationSuggestion[] {
    return [
        {
            remediation_id: 'rem-001',
            title: 'Configure App Battery Optimization',
            description: 'Enable battery optimization for DataSync Pro to limit background activity.',
            detailed_steps: [
                'Open MobiControl console',
                'Navigate to device > Applications',
                'Select "DataSync Pro"',
                'Enable "Restrict background battery usage"',
                'Apply configuration change',
            ],
            priority: 1,
            confidence_score: 0.88,
            confidence_level: 'high',
            source: 'learned',
            source_details: 'Resolved 12 similar cases with 91% success rate',
            historical_success_rate: 0.91,
            historical_sample_size: 12,
            estimated_impact: 'Expected to reduce battery drain by 60-70%',
            is_automated: true,
            automation_type: 'mobicontrol_action',
        },
        {
            remediation_id: 'rem-002',
            title: 'Update DataSync Pro Application',
            description: 'Update to version 3.3 which includes battery optimization fixes.',
            detailed_steps: [
                'Navigate to App Catalog in MobiControl',
                'Find DataSync Pro in enterprise apps',
                'Push version 3.3 update to device',
                'Verify installation completes successfully',
            ],
            priority: 2,
            confidence_score: 0.72,
            confidence_level: 'medium',
            source: 'ai_generated',
            source_details: 'Suggested based on release notes mentioning battery fixes',
            historical_success_rate: null,
            historical_sample_size: null,
            estimated_impact: 'May address root cause if issue is app-related bug',
            is_automated: true,
            automation_type: 'app_update',
        },
        {
            remediation_id: 'rem-003',
            title: 'Remote Device Restart',
            description: 'Restart the device to clear any stuck processes or memory issues.',
            detailed_steps: [
                'Send remote restart command via MobiControl',
                'Wait for device to come back online (2-5 minutes)',
                'Verify device connectivity and app status',
            ],
            priority: 3,
            confidence_score: 0.45,
            confidence_level: 'low',
            source: 'policy',
            source_details: 'Standard remediation step for unresponsive devices',
            historical_success_rate: 0.35,
            historical_sample_size: 150,
            estimated_impact: 'Temporary fix; may not address underlying cause',
            is_automated: true,
            automation_type: 'device_command',
        },
    ];
}

function generateSimilarCases(anomalyId: number): SimilarCase[] {
    return [
        {
            case_id: 'case-2024-1205',
            anomaly_id: anomalyId - 15,
            device_id: 1042,
            detected_at: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
            resolved_at: new Date(Date.now() - 6.5 * 24 * 60 * 60 * 1000).toISOString(),
            similarity_score: 0.94,
            similarity_factors: ['Battery drain pattern', 'Same app installed', 'Similar usage profile'],
            anomaly_type: 'Battery Anomaly',
            severity: 'high',
            resolution_status: 'resolved',
            resolution_summary: 'Battery optimization enabled for DataSync Pro, resolved within 12 hours',
            successful_remediation: 'Configure App Battery Optimization',
            time_to_resolution_hours: 12,
        },
        {
            case_id: 'case-2024-1198',
            anomaly_id: anomalyId - 28,
            device_id: 1087,
            detected_at: new Date(Date.now() - 12 * 24 * 60 * 60 * 1000).toISOString(),
            resolved_at: new Date(Date.now() - 11 * 24 * 60 * 60 * 1000).toISOString(),
            similarity_score: 0.87,
            similarity_factors: ['Battery drain pattern', 'Background sync activity'],
            anomaly_type: 'Battery Anomaly',
            severity: 'high',
            resolution_status: 'resolved',
            resolution_summary: 'App update pushed, issue resolved after sync completed',
            successful_remediation: 'Update DataSync Pro Application',
            time_to_resolution_hours: 24,
        },
        {
            case_id: 'case-2024-1156',
            anomaly_id: anomalyId - 45,
            device_id: 1023,
            detected_at: new Date(Date.now() - 21 * 24 * 60 * 60 * 1000).toISOString(),
            resolved_at: null,
            similarity_score: 0.72,
            similarity_factors: ['Network disconnection pattern', 'Similar store location'],
            anomaly_type: 'Network Anomaly',
            severity: 'medium',
            resolution_status: 'investigating',
            resolution_summary: null,
            successful_remediation: null,
            time_to_resolution_hours: null,
        },
    ];
}

export function getMockInvestigationPanel(anomalyId: number): InvestigationPanel {
    const featureContributions = generateFeatureContributions();
    const baselineMetrics = generateBaselineMetrics();
    const evidenceEvents = generateEvidenceEvents(anomalyId);

    return {
        anomaly_id: anomalyId,
        device_id: 1001,
        anomaly_score: -0.73,
        severity: 'high',
        confidence_score: 0.85,
        detected_at: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString(),
        explanation: {
            summary_text: 'This device shows unusual battery drain (4.2σ above baseline) combined with elevated offline time and network disruptions, indicating potential app malfunction or network infrastructure issues.',
            detailed_explanation: 'The anomaly detection system flagged this device due to a combination of factors, primarily driven by extreme battery drain (58.3% vs baseline 12.4%). The Isolation Forest model assigned a score of -0.73 (threshold: -0.5), placing this in the top 2% of anomalous behavior. Secondary factors include extended offline periods and higher-than-normal data transfer, suggesting a background process consuming excessive resources.',
            feature_contributions: featureContributions,
            top_contributing_features: ['Battery Drain', 'Offline Duration', 'Download Traffic'],
            explanation_method: 'shap_tree_explainer',
            explanation_generated_at: new Date().toISOString(),
        },
        baseline_comparison: {
            baseline_config: {
                baseline_type: 'peer_group',
                baseline_period_days: 14,
                comparison_window_hours: 24,
                statistical_method: 'z_score',
                peer_group_name: 'Retail Store Devices - US West',
                peer_group_size: 245,
                baseline_calculated_at: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
            },
            metrics: baselineMetrics,
            overall_deviation_score: 3.2,
        },
        evidence_events: evidenceEvents,
        evidence_event_count: evidenceEvents.length,
        ai_analysis: {
            analysis_id: `analysis-${anomalyId}`,
            generated_at: new Date().toISOString(),
            model_used: 'llama3.2',
            primary_hypothesis: generateRootCauseHypothesis(true),
            alternative_hypotheses: [generateRootCauseHypothesis(false)],
            confidence_score: 0.82,
            confidence_level: 'high',
            confidence_explanation: 'High confidence based on strong temporal correlation between app installation and battery drain onset, supported by similar resolved cases.',
            similar_cases_analyzed: 15,
            feedback_received: false,
            feedback_rating: null,
        },
        suggested_remediations: generateRemediations(),
        similar_cases: generateSimilarCases(anomalyId),
    };
}

export function getMockAIAnalysis(anomalyId: number): AIAnalysis {
    return {
        analysis_id: `analysis-${anomalyId}`,
        generated_at: new Date().toISOString(),
        model_used: 'llama3.2',
        primary_hypothesis: generateRootCauseHypothesis(true),
        alternative_hypotheses: [generateRootCauseHypothesis(false)],
        confidence_score: 0.82,
        confidence_level: 'high',
        confidence_explanation: 'High confidence based on strong temporal correlation between app installation and battery drain onset, supported by similar resolved cases.',
        similar_cases_analyzed: 15,
        feedback_received: false,
        feedback_rating: null,
    };
}

export function getMockHistoricalTimeline(
    _anomalyId: number,
    metric: string,
    days: number
): HistoricalTimeline {
    const dataPoints: TimeSeriesDataPoint[] = [];
    const now = new Date();
    const hoursToGenerate = days * 24;

    // Generate baseline values
    const baselineByMetric: Record<string, { mean: number; std: number }> = {
        total_battery_level_drop: { mean: 12.4, std: 8.2 },
        offline_time: { mean: 0.8, std: 0.6 },
        download: { mean: 450, std: 280 },
        wifi_signal_strength: { mean: -55, std: 12 },
        total_free_storage_kb: { mean: 8500000, std: 3200000 },
    };

    const baseline = baselineByMetric[metric] || { mean: 50, std: 15 };

    for (let i = hoursToGenerate; i >= 0; i--) {
        const timestamp = new Date(now.getTime() - i * 60 * 60 * 1000);
        let value = baseline.mean + (Math.random() - 0.5) * 2 * baseline.std;

        // Add anomaly spike in recent hours for battery drain
        if (i < 6 && metric === 'total_battery_level_drop') {
            value = baseline.mean + (6 - i) * 10 + Math.random() * 5;
        }

        const isAnomalous = Math.abs(value - baseline.mean) > 2 * baseline.std;

        dataPoints.push({
            timestamp: timestamp.toISOString(),
            value: Math.max(0, value),
            is_anomalous: isAnomalous,
        });
    }

    return {
        metric_name: metric,
        data_points: dataPoints,
        baseline_mean: baseline.mean,
        baseline_std: baseline.std,
        baseline_upper: baseline.mean + 2 * baseline.std,
        baseline_lower: Math.max(0, baseline.mean - 2 * baseline.std),
    };
}

// ============================================================================
// Automation Mock Data
// ============================================================================

import type {
    SchedulerConfig,
    SchedulerStatus,
    AutomationAlert,
    AutomationJob,
    TriggerJobResponse,
} from '../types/automation';

export function getMockAutomationConfig(): SchedulerConfig {
    return {
        training_enabled: true,
        training_interval: 'daily',
        training_hour: 2,
        training_day_of_week: 0,
        training_lookback_days: 90,
        training_validation_days: 7,
        scoring_enabled: true,
        scoring_interval_minutes: 15,
        auto_retrain_enabled: true,
        auto_retrain_fp_threshold: 0.15,
        auto_retrain_min_feedback: 50,
        auto_retrain_cooldown_hours: 24,
        alerting_enabled: true,
        alert_on_high_anomaly_rate: true,
        high_anomaly_rate_threshold: 0.10,
        last_training_time: new Date(Date.now() - 12 * 60 * 60 * 1000).toISOString(),
        last_scoring_time: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
        last_auto_retrain_time: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
    };
}

export function getMockAutomationStatus(): SchedulerStatus {
    const lastTrainingTime = new Date(Date.now() - 12 * 60 * 60 * 1000);
    const lastScoringTime = new Date(Date.now() - 15 * 60 * 1000);
    const nextTrainingTime = new Date(Date.now() + 12 * 60 * 60 * 1000);
    const nextScoringTime = new Date(Date.now() + 15 * 60 * 1000);

    return {
        is_running: true,
        training_status: 'idle',
        scoring_status: 'idle',
        last_training_result: {
            success: true,
            timestamp: lastTrainingTime.toISOString(),
            metrics: {
                samples_trained: 12450,
                features_used: 8,
                model_accuracy: 0.94,
                contamination: 0.05,
            },
        },
        last_scoring_result: {
            success: true,
            timestamp: lastScoringTime.toISOString(),
            total_scored: 248,
            anomalies_detected: 12,
            anomaly_rate: 0.048,
        },
        next_training_time: nextTrainingTime.toISOString(),
        next_scoring_time: nextScoringTime.toISOString(),
        total_anomalies_detected: 156,
        false_positive_rate: 0.08,
        uptime_seconds: 86400 * 3 + 12345,
        errors: [],
    };
}

export function getMockAutomationAlerts(limit: number = 20): AutomationAlert[] {
    const alerts: AutomationAlert[] = [
        {
            id: 'alert_001',
            timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
            message: 'High anomaly rate detected: 12.5% of devices flagged in last scoring run (threshold: 10%)',
            acknowledged: false,
        },
        {
            id: 'alert_002',
            timestamp: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString(),
            message: 'Auto-retrain triggered due to false positive rate exceeding 15%',
            acknowledged: true,
        },
        {
            id: 'alert_003',
            timestamp: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
            message: 'Model training completed successfully with 94% accuracy',
            acknowledged: true,
        },
        {
            id: 'alert_004',
            timestamp: new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString(),
            message: 'Scoring run completed: 248 devices scored, 8 anomalies detected',
            acknowledged: true,
        },
    ];

    return alerts.slice(0, limit);
}

export function getMockAutomationHistory(limit: number = 10): AutomationJob[] {
    const jobs: AutomationJob[] = [
        {
            type: 'scoring',
            timestamp: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
            triggered_by: 'schedule',
            success: true,
            metrics: { total_scored: 248, anomalies_detected: 12, anomaly_rate: 0.048 },
        },
        {
            type: 'scoring',
            timestamp: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
            triggered_by: 'schedule',
            success: true,
            metrics: { total_scored: 250, anomalies_detected: 8, anomaly_rate: 0.032 },
        },
        {
            type: 'training',
            timestamp: new Date(Date.now() - 12 * 60 * 60 * 1000).toISOString(),
            triggered_by: 'schedule',
            success: true,
            metrics: { samples_trained: 12450, model_accuracy: 0.94 },
        },
        {
            type: 'scoring',
            timestamp: new Date(Date.now() - 12.5 * 60 * 60 * 1000).toISOString(),
            triggered_by: 'schedule',
            success: true,
            metrics: { total_scored: 245, anomalies_detected: 15, anomaly_rate: 0.061 },
        },
        {
            type: 'auto_retrain',
            timestamp: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
            triggered_by: 'auto',
            success: true,
            metrics: { reason: 'fp_threshold_exceeded', previous_fp_rate: 0.18, new_fp_rate: 0.08 },
        },
        {
            type: 'training',
            timestamp: new Date(Date.now() - 4 * 24 * 60 * 60 * 1000).toISOString(),
            triggered_by: 'manual',
            success: true,
            metrics: { samples_trained: 11200, model_accuracy: 0.91 },
        },
        {
            type: 'scoring',
            timestamp: new Date(Date.now() - 4 * 24 * 60 * 60 * 1000 - 30 * 60 * 1000).toISOString(),
            triggered_by: 'schedule',
            success: false,
            error: 'Database connection timeout',
        },
    ];

    return jobs.slice(0, limit);
}

export function getMockTriggerJobResponse(jobType: 'training' | 'scoring'): TriggerJobResponse {
    return {
        success: true,
        job_id: `job-${Date.now()}-${jobType}`,
        message: `${jobType.charAt(0).toUpperCase() + jobType.slice(1)} job started successfully (mock mode)`,
    };
}

// ============================================================================
// Training Mock Data
// ============================================================================

import type {
    TrainingRun,
    TrainingConfigRequest,
    TrainingMetrics,
    TrainingHistoryResponse,
} from '../types/training';

export function getMockTrainingStatus(): TrainingRun {
    return {
        run_id: 'mock-run-idle',
        status: 'idle',
        progress: 0,
        message: 'No training in progress',
        stage: null,
        started_at: null,
        completed_at: null,
        estimated_completion: null,
        config: null,
        metrics: null,
        artifacts: null,
        model_version: null,
        error: null,
        stages: null,
    };
}

export function getMockStartTrainingResponse(config: TrainingConfigRequest): TrainingRun {
    const runId = `mock-run-${Date.now()}`;
    return {
        run_id: runId,
        status: 'pending',
        progress: 0,
        message: 'Training job queued (mock mode)',
        stage: 'Initialize',
        started_at: new Date().toISOString(),
        completed_at: null,
        estimated_completion: new Date(Date.now() + 15 * 60 * 1000).toISOString(),
        config: config as unknown as Record<string, unknown>,
        metrics: null,
        artifacts: null,
        model_version: null,
        error: null,
        stages: [
            { name: 'Initialize', status: 'running', started_at: new Date().toISOString() },
            { name: 'Load Data', status: 'pending' },
            { name: 'Features', status: 'pending' },
            { name: 'Training', status: 'pending' },
            { name: 'Validation', status: 'pending' },
            { name: 'Export', status: 'pending' },
        ],
    };
}

export function getMockTrainingHistory(limit: number = 10): TrainingHistoryResponse {
    const runs: TrainingRun[] = [
        {
            run_id: 'run-20241228-142530',
            status: 'completed',
            progress: 100,
            message: 'Training completed successfully',
            stage: 'Export',
            started_at: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
            completed_at: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000 + 12 * 60 * 1000).toISOString(),
            model_version: 'v1.2.0',
            metrics: {
                train_rows: 145832,
                validation_rows: 18456,
                feature_count: 24,
                anomaly_rate_train: 0.032,
                anomaly_rate_validation: 0.029,
                validation_auc: 0.892,
                precision_at_recall_80: 0.78,
                feature_importance: {
                    total_battery_level_drop: 0.18,
                    total_free_storage_kb: 0.15,
                    offline_time: 0.14,
                    disconnect_count: 0.12,
                    download: 0.10,
                    upload: 0.09,
                    wifi_signal_strength: 0.08,
                    connection_time: 0.07,
                },
            },
            artifacts: {
                model_path: 'models/isolation_forest_v1.2.0.pkl',
                onnx_path: 'models/isolation_forest_v1.2.0.onnx',
                baselines_path: 'models/baselines_v1.2.0.json',
            },
            stages: [
                { name: 'Initialize', status: 'completed' },
                { name: 'Load Data', status: 'completed' },
                { name: 'Features', status: 'completed' },
                { name: 'Training', status: 'completed' },
                { name: 'Validation', status: 'completed' },
                { name: 'Export', status: 'completed' },
            ],
        },
        {
            run_id: 'run-20241225-093015',
            status: 'completed',
            progress: 100,
            message: 'Training completed successfully',
            stage: 'Export',
            started_at: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
            completed_at: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000 + 14 * 60 * 1000).toISOString(),
            model_version: 'v1.1.0',
            metrics: {
                train_rows: 132456,
                validation_rows: 16245,
                feature_count: 24,
                anomaly_rate_train: 0.035,
                anomaly_rate_validation: 0.031,
                validation_auc: 0.878,
                precision_at_recall_80: 0.74,
                feature_importance: {
                    total_battery_level_drop: 0.17,
                    total_free_storage_kb: 0.16,
                    offline_time: 0.13,
                    disconnect_count: 0.11,
                    download: 0.11,
                    upload: 0.08,
                    wifi_signal_strength: 0.09,
                    connection_time: 0.06,
                },
            },
            artifacts: {
                model_path: 'models/isolation_forest_v1.1.0.pkl',
                onnx_path: 'models/isolation_forest_v1.1.0.onnx',
                baselines_path: 'models/baselines_v1.1.0.json',
            },
            stages: [
                { name: 'Initialize', status: 'completed' },
                { name: 'Load Data', status: 'completed' },
                { name: 'Features', status: 'completed' },
                { name: 'Training', status: 'completed' },
                { name: 'Validation', status: 'completed' },
                { name: 'Export', status: 'completed' },
            ],
        },
        {
            run_id: 'run-20241220-183045',
            status: 'failed',
            progress: 45,
            message: 'Failed to load data: Connection timeout',
            stage: 'Load Data',
            started_at: new Date(Date.now() - 10 * 24 * 60 * 60 * 1000).toISOString(),
            completed_at: new Date(Date.now() - 10 * 24 * 60 * 60 * 1000 + 3 * 60 * 1000).toISOString(),
            model_version: null,
            error: 'Database connection timeout after 30 seconds',
            metrics: null,
            artifacts: null,
            stages: [
                { name: 'Initialize', status: 'completed' },
                { name: 'Load Data', status: 'failed', message: 'Connection timeout' },
                { name: 'Features', status: 'pending' },
                { name: 'Training', status: 'pending' },
                { name: 'Validation', status: 'pending' },
                { name: 'Export', status: 'pending' },
            ],
        },
    ];

    return {
        runs: runs.slice(0, limit),
        total: runs.length,
    };
}

export function getMockTrainingMetrics(runId: string): TrainingMetrics {
    // Return metrics based on runId to simulate different runs
    const baseMetrics: TrainingMetrics = {
        train_rows: 145832,
        validation_rows: 18456,
        feature_count: 24,
        anomaly_rate_train: 0.032,
        anomaly_rate_validation: 0.029,
        validation_auc: 0.892,
        precision_at_recall_80: 0.78,
        feature_importance: {
            total_battery_level_drop: 0.18,
            total_free_storage_kb: 0.15,
            offline_time: 0.14,
            disconnect_count: 0.12,
            download: 0.10,
            upload: 0.09,
            wifi_signal_strength: 0.08,
            connection_time: 0.07,
        },
    };

    // Vary slightly based on runId hash
    const hash = runId.split('').reduce((a, b) => a + b.charCodeAt(0), 0);
    const variation = (hash % 10) / 100;

    return {
        ...baseMetrics,
        validation_auc: baseMetrics.validation_auc! - variation + 0.05,
        precision_at_recall_80: baseMetrics.precision_at_recall_80! - variation + 0.03,
    };
}

// ============================================================================
// INSIGHTS Mock Data
// Carl's vision: "XSight has the data. XSight needs the story."
// ============================================================================

import type {
    CustomerInsightResponse,
    DailyDigestResponse,
    LocationInsightResponse,
    ShiftReadinessResponse,
    NetworkAnalysisResponse,
    DeviceAbuseResponse,
    AppAnalysisResponse,
    LocationCompareResponse,
} from './client';

function generateMockInsight(index: number, category: string, severity: string): CustomerInsightResponse {
    const categories: Record<string, { headline: string; impact: string; comparison: string }> = {
        battery_shift_failure: {
            headline: `${5 + index} devices in Warehouse A won't last the morning shift`,
            impact: 'At current drain rate (8.5%/hr), battery will die by 11:30 AM. Workers may experience 2.5 hours of unplanned downtime.',
            comparison: '42% faster drain than peer devices',
        },
        excessive_drops: {
            headline: `Location "Distribution Center" has ${12 + index} device drops this week`,
            impact: 'Estimated repair costs: $2,400. Device downtime: approximately 8 hours.',
            comparison: 'This ranks #2 worst out of 8 locations',
        },
        wifi_ap_hopping: {
            headline: `Excessive WiFi roaming in Loading Dock: ${8 + index} access points per day`,
            impact: 'Frequent AP switching causes 45 minutes of connection gaps per day.',
            comparison: 'Other locations average 3 APs/day - 167% more switching',
        },
        app_crash_pattern: {
            headline: `App "InventoryPro" crashing repeatedly: ${15 + index} crashes today`,
            impact: 'Each crash causes 3 minutes of downtime. Total productivity loss: 45 minutes.',
            comparison: 'Normal crash rate is 0.5/day - this is 30x higher',
        },
    };

    const catData = categories[category] || categories.battery_shift_failure;

    return {
        insight_id: `insight-${category}-${index}`,
        category,
        severity,
        headline: catData.headline,
        impact_statement: catData.impact,
        comparison_context: catData.comparison,
        recommended_actions: [
            'Investigate root cause',
            'Review device configuration',
            'Consider hardware replacement if issue persists',
        ],
        entity_type: 'location',
        entity_id: `loc-${index}`,
        entity_name: MOCK_STORES[index % MOCK_STORES.length].name,
        affected_device_count: 5 + index * 2,
        primary_metric: 'battery_drain_rate',
        primary_value: 8.5 + index * 0.5,
        trend_direction: index % 2 === 0 ? 'degrading' : 'stable',
        trend_change_percent: index % 2 === 0 ? 15.5 : null,
        detected_at: new Date(Date.now() - index * 60 * 60 * 1000).toISOString(),
        confidence_score: 0.85 - index * 0.02,
    };
}

export function getMockDailyDigest(): DailyDigestResponse {
    const insights = [
        generateMockInsight(0, 'battery_shift_failure', 'critical'),
        generateMockInsight(1, 'excessive_drops', 'high'),
        generateMockInsight(2, 'wifi_ap_hopping', 'medium'),
        generateMockInsight(3, 'app_crash_pattern', 'high'),
        generateMockInsight(4, 'battery_shift_failure', 'medium'),
    ];

    return {
        tenant_id: 'default',
        digest_date: new Date().toISOString().split('T')[0],
        generated_at: new Date().toISOString(),
        total_insights: 23,
        critical_count: 3,
        high_count: 8,
        medium_count: 12,
        top_insights: insights,
        executive_summary: `Today's analysis identified 23 actionable insights across your fleet. 3 critical issues require immediate attention, primarily related to battery performance at warehouse locations. Battery drain rates are 42% higher than baseline in Distribution Center locations. WiFi roaming issues are affecting productivity in 2 locations. Recommend prioritizing battery charging infrastructure review at affected sites.`,
        trending_issues: insights.slice(0, 3),
        new_issues: insights.slice(3, 5),
    };
}

export function getMockLocationInsights(locationId: string): LocationInsightResponse {
    const store = MOCK_STORES.find(s => s.id === locationId) || MOCK_STORES[0];

    return {
        location_id: locationId,
        location_name: store.name,
        report_date: new Date().toISOString().split('T')[0],
        total_devices: 45,
        devices_with_issues: 8,
        issue_rate: 0.178,
        shift_readiness: null,
        insights: [
            generateMockInsight(0, 'battery_shift_failure', 'high'),
            generateMockInsight(1, 'wifi_ap_hopping', 'medium'),
        ],
        top_issues: [
            { category: 'battery_shift_failure', count: 5 },
            { category: 'wifi_ap_hopping', count: 3 },
            { category: 'excessive_drops', count: 2 },
        ],
        rank_among_locations: 3,
        better_than_percent: 62.5,
        recommendations: [
            'Review charging infrastructure - 5 devices started shift below 80%',
            'Investigate WiFi coverage in loading dock area',
            'Consider deploying protective cases for high-drop-rate devices',
        ],
    };
}

export function getMockShiftReadiness(locationId: string): ShiftReadinessResponse {
    const store = MOCK_STORES.find(s => s.id === locationId) || MOCK_STORES[0];

    const deviceDetails = Array.from({ length: 12 }, (_, i) => ({
        device_id: 1001 + i,
        device_name: `Scanner-${String(i + 1).padStart(3, '0')}`,
        current_battery: 45 + Math.floor(Math.random() * 55),
        drain_rate_per_hour: 5 + Math.random() * 8,
        projected_end_battery: 15 + Math.floor(Math.random() * 30),
        will_complete_shift: i < 8,
        estimated_dead_time: i >= 8 ? '11:30 AM' : null,
        was_fully_charged: i < 6,
        readiness_score: 0.6 + Math.random() * 0.4,
        recommendations: i >= 8 ? ['Charge before shift starts'] : [],
    }));

    return {
        location_id: locationId,
        location_name: store.name,
        shift_name: 'Morning Shift',
        shift_date: new Date().toISOString().split('T')[0],
        readiness_percentage: 66.7,
        total_devices: 12,
        devices_ready: 8,
        devices_at_risk: 3,
        devices_critical: 1,
        avg_battery_at_start: 72.5,
        avg_drain_rate: 8.3,
        devices_not_fully_charged: 6,
        vs_last_week_readiness: -5.2,
        device_details: deviceDetails,
        recommendations: [
            'Ensure all devices are fully charged before shift start',
            'Investigate devices with high drain rates (>10%/hr)',
            'Review charging infrastructure at this location',
        ],
    };
}

export function getMockTrendingInsights(): CustomerInsightResponse[] {
    return [
        generateMockInsight(0, 'battery_shift_failure', 'critical'),
        generateMockInsight(1, 'excessive_drops', 'high'),
        generateMockInsight(2, 'wifi_ap_hopping', 'high'),
    ];
}

export function getMockNetworkAnalysis(): NetworkAnalysisResponse {
    return {
        tenant_id: 'default',
        analysis_period_days: 7,
        wifi_summary: {
            total_devices: 248,
            devices_with_roaming_issues: 23,
            devices_with_stickiness: 12,
            avg_aps_per_device: 4.2,
            potential_dead_zones: 3,
        },
        cellular_summary: {
            total_devices: 156,
            devices_with_tower_hopping: 18,
            devices_with_tech_fallback: 8,
            best_carrier: 'Verizon',
            worst_carrier: 'T-Mobile',
            network_type_distribution: { '5G': 45, 'LTE': 89, '3G': 22 },
        },
        disconnect_summary: {
            total_disconnects: 342,
            avg_disconnects_per_device: 1.38,
            total_offline_hours: 156.5,
            has_predictable_pattern: true,
            pattern_description: 'Most disconnects occur between 2-4 PM in warehouse zones',
        },
        hidden_devices_count: 4,
        recommendations: [
            'Add AP coverage to loading dock dead zone',
            'Review WiFi roaming thresholds (-70dBm recommended)',
            'Investigate T-Mobile coverage issues in Zone B',
            'Review devices with suspicious offline patterns',
        ],
        financial_impact: {
            total_estimated_cost: 23475,
            cost_breakdown: [
                { category: 'Productivity Loss', amount: 15650, description: '156.5 hours offline x $100/hr avg worker cost' },
                { category: 'IT Support', amount: 5125, description: 'Investigation time for 342 disconnects' },
                { category: 'Dead Zone Impact', amount: 2700, description: '3 zones affecting 45 devices daily' },
            ],
            potential_savings: 18750,
            cost_per_incident: 69,
            monthly_trend: 12.5,
        },
    };
}

export function getMockDeviceAbuseAnalysis(): DeviceAbuseResponse {
    return {
        tenant_id: 'default',
        analysis_period_days: 7,
        total_devices: 248,
        total_drops: 156,
        total_reboots: 89,
        devices_with_excessive_drops: 12,
        devices_with_excessive_reboots: 8,
        worst_locations: [
            { location_id: 'store-003', drops: 45, rate_per_device: 3.2 },
            { location_id: 'store-001', drops: 32, rate_per_device: 2.1 },
            { location_id: 'store-005', drops: 28, rate_per_device: 1.8 },
        ],
        worst_cohorts: [
            { cohort_id: 'Zebra_TC52_12', reboots: 23, rate_per_device: 2.3 },
            { cohort_id: 'Samsung_XCover_13', reboots: 18, rate_per_device: 1.8 },
        ],
        problem_combinations: [
            {
                cohort_id: 'Zebra_TC52_Android12_fw1.2',
                manufacturer: 'Zebra',
                model: 'TC52',
                os_version: 'Android 12',
                device_count: 15,
                vs_fleet_multiplier: 2.8,
                primary_issue: 'excessive_reboots',
                severity: 'high',
            },
        ],
        recommendations: [
            'Deploy protective cases at Harbor Point (highest drop rate)',
            'Update firmware on Zebra TC52 devices (known reboot bug)',
            'Conduct device handling training at top 3 locations',
        ],
        financial_impact: {
            total_estimated_cost: 19500,
            cost_breakdown: [
                { category: 'Screen Repairs', amount: 9360, description: '12 devices with excessive drops x $780 avg repair' },
                { category: 'Productivity Loss', amount: 5400, description: '156 drops x 15 min recovery x $35/hr' },
                { category: 'IT Support', amount: 3340, description: '89 reboots investigated x $37.50/incident' },
                { category: 'Accelerated Depreciation', amount: 1400, description: 'Reduced device lifespan from damage' },
            ],
            potential_savings: 15600,
            cost_per_incident: 80,
            monthly_trend: -8.5,
        },
    };
}

export function getMockAppAnalysis(): AppAnalysisResponse {
    return {
        tenant_id: 'default',
        analysis_period_days: 7,
        total_apps_analyzed: 45,
        apps_with_issues: 8,
        total_crashes: 156,
        total_anrs: 34,
        top_power_consumers: [
            { package_name: 'com.inventory.pro', app_name: 'InventoryPro', battery_drain_percent: 28.5, drain_per_hour: 4.2, foreground_hours: 6.8, efficiency_score: 0.42 },
            { package_name: 'com.shipping.tracker', app_name: 'ShipTrack', battery_drain_percent: 18.2, drain_per_hour: 3.1, foreground_hours: 5.9, efficiency_score: 0.58 },
            { package_name: 'com.scanner.barcode', app_name: 'BarcodePlus', battery_drain_percent: 12.4, drain_per_hour: 2.8, foreground_hours: 4.4, efficiency_score: 0.65 },
        ],
        top_crashers: [
            { package_name: 'com.inventory.pro', app_name: 'InventoryPro', crash_count: 45, anr_count: 12, devices_affected: 23, severity: 'high' },
            { package_name: 'com.legacy.wms', app_name: 'WMS Legacy', crash_count: 28, anr_count: 8, devices_affected: 15, severity: 'medium' },
        ],
        recommendations: [
            'Update InventoryPro to version 3.2.1 (fixes memory leak)',
            'Consider replacing WMS Legacy - high crash rate, no recent updates',
            'Restrict BarcodePlus background activity to reduce battery drain',
        ],
        financial_impact: {
            total_estimated_cost: 28950,
            cost_breakdown: [
                { category: 'Crash Recovery Time', amount: 11700, description: '156 crashes x 15 min recovery x $30/hr avg' },
                { category: 'Battery Replacement', amount: 8750, description: 'Accelerated battery wear from power-hungry apps' },
                { category: 'Lost Productivity (ANRs)', amount: 5100, description: '34 ANRs x 10 min freeze x $30/hr x 5 users avg' },
                { category: 'IT Support', amount: 3400, description: 'App troubleshooting and reinstalls' },
            ],
            potential_savings: 23160,
            cost_per_incident: 152,
            monthly_trend: 5.3,
        },
    };
}

export function getMockInsightsByCategory(category: string): CustomerInsightResponse[] {
    return [
        generateMockInsight(0, category, 'critical'),
        generateMockInsight(1, category, 'high'),
        generateMockInsight(2, category, 'medium'),
        generateMockInsight(3, category, 'low'),
    ];
}

export function getMockLocationComparison(locationAId: string, locationBId: string): LocationCompareResponse {
    const storeA = MOCK_STORES.find(s => s.id === locationAId) || MOCK_STORES[0];
    const storeB = MOCK_STORES.find(s => s.id === locationBId) || MOCK_STORES[1];

    return {
        location_a_id: locationAId,
        location_a_name: storeA.name,
        location_b_id: locationBId,
        location_b_name: storeB.name,
        device_count_a: 45,
        device_count_b: 38,
        metric_comparisons: {
            battery_drain_rate: { location_a_value: 8.5, location_b_value: 6.2, difference_percent: 37.1 },
            drop_rate: { location_a_value: 2.3, location_b_value: 0.8, difference_percent: 187.5 },
            wifi_disconnects: { location_a_value: 12, location_b_value: 5, difference_percent: 140 },
            shift_readiness: { location_a_value: 72, location_b_value: 89, difference_percent: -19.1 },
        },
        overall_winner: locationBId,
        key_differences: [
            `${storeA.name} has 37% higher battery drain than ${storeB.name}`,
            `${storeA.name} has 2.9x more device drops per week`,
            `${storeB.name} achieves 89% shift readiness vs 72% at ${storeA.name}`,
        ],
    };
}

// ============================================================================
// Smart Anomaly Grouping Mock Data
// ============================================================================

export function getMockGroupedAnomalies(_params?: {
    status?: string;
    min_severity?: string;
    min_group_size?: number;
    temporal_window_hours?: number;
}): GroupedAnomaliesResponse {
    const now = new Date();
    const oneHourAgo = new Date(now.getTime() - 60 * 60 * 1000);
    const twoHoursAgo = new Date(now.getTime() - 2 * 60 * 60 * 1000);
    const threeHoursAgo = new Date(now.getTime() - 3 * 60 * 60 * 1000);

    // Create sample group members
    const batteryMembers: AnomalyGroupMember[] = [
        { anomaly_id: 1001, device_id: 101, anomaly_score: -0.85, severity: 'critical', status: 'open', timestamp: oneHourAgo.toISOString(), device_model: 'Zebra TC52', location: 'Downtown Flagship', primary_metric: 'total_battery_level_drop' },
        { anomaly_id: 1002, device_id: 102, anomaly_score: -0.72, severity: 'critical', status: 'investigating', timestamp: twoHoursAgo.toISOString(), device_model: 'Zebra TC52', location: 'Downtown Flagship', primary_metric: 'total_battery_level_drop' },
        { anomaly_id: 1003, device_id: 103, anomaly_score: -0.68, severity: 'high', status: 'open', timestamp: threeHoursAgo.toISOString(), device_model: 'Samsung Galaxy Tab A8', location: 'Westside Mall', primary_metric: 'total_battery_level_drop' },
        { anomaly_id: 1004, device_id: 104, anomaly_score: -0.55, severity: 'high', status: 'open', timestamp: oneHourAgo.toISOString(), device_model: 'Samsung Galaxy Tab A8', location: 'Harbor Point', primary_metric: 'total_battery_level_drop' },
        { anomaly_id: 1005, device_id: 105, anomaly_score: -0.52, severity: 'high', status: 'open', timestamp: twoHoursAgo.toISOString(), device_model: 'iPad Pro 11', location: 'Tech Plaza', primary_metric: 'total_battery_level_drop' },
    ];

    const networkMembers: AnomalyGroupMember[] = [
        { anomaly_id: 2001, device_id: 201, anomaly_score: -0.78, severity: 'critical', status: 'open', timestamp: oneHourAgo.toISOString(), device_model: 'Honeywell CT60', location: 'Harbor Point', primary_metric: 'disconnect_count' },
        { anomaly_id: 2002, device_id: 202, anomaly_score: -0.65, severity: 'high', status: 'open', timestamp: twoHoursAgo.toISOString(), device_model: 'Honeywell CT60', location: 'Harbor Point', primary_metric: 'disconnect_count' },
        { anomaly_id: 2003, device_id: 203, anomaly_score: -0.61, severity: 'high', status: 'investigating', timestamp: threeHoursAgo.toISOString(), device_model: 'Zebra TC52', location: 'Harbor Point', primary_metric: 'disconnect_count' },
        { anomaly_id: 2004, device_id: 204, anomaly_score: -0.58, severity: 'high', status: 'open', timestamp: oneHourAgo.toISOString(), device_model: 'Samsung Galaxy XCover 6', location: 'Harbor Point', primary_metric: 'wifi_signal_strength' },
    ];

    const storageMembers: AnomalyGroupMember[] = [
        { anomaly_id: 3001, device_id: 301, anomaly_score: -0.82, severity: 'critical', status: 'open', timestamp: oneHourAgo.toISOString(), device_model: 'Samsung Galaxy Tab A8', location: 'Central Station', primary_metric: 'total_free_storage_kb' },
        { anomaly_id: 3002, device_id: 302, anomaly_score: -0.75, severity: 'critical', status: 'open', timestamp: twoHoursAgo.toISOString(), device_model: 'iPad Pro 11', location: 'University District', primary_metric: 'total_free_storage_kb' },
        { anomaly_id: 3003, device_id: 303, anomaly_score: -0.71, severity: 'critical', status: 'investigating', timestamp: threeHoursAgo.toISOString(), device_model: 'Samsung Galaxy Tab A8', location: 'Riverside Center', primary_metric: 'total_free_storage_kb' },
    ];

    const cohortMembers: AnomalyGroupMember[] = [
        { anomaly_id: 4001, device_id: 401, anomaly_score: -0.62, severity: 'high', status: 'open', timestamp: oneHourAgo.toISOString(), device_model: 'Zebra TC52', location: 'Airport Terminal', primary_metric: 'offline_time' },
        { anomaly_id: 4002, device_id: 402, anomaly_score: -0.58, severity: 'high', status: 'open', timestamp: twoHoursAgo.toISOString(), device_model: 'Zebra TC52', location: 'Downtown Flagship', primary_metric: 'offline_time' },
        { anomaly_id: 4003, device_id: 403, anomaly_score: -0.55, severity: 'high', status: 'open', timestamp: threeHoursAgo.toISOString(), device_model: 'Zebra TC52', location: 'Tech Plaza', primary_metric: 'connection_time' },
    ];

    // Create groups
    const groups: AnomalyGroup[] = [
        {
            group_id: 'grp-battery-001',
            group_name: 'Investigate Battery Drain (5 devices)',
            group_category: 'battery_investigate',
            group_type: 'remediation_match',
            severity: 'critical' as Severity,
            total_count: 5,
            open_count: 4,
            device_count: 5,
            suggested_remediation: {
                remediation_id: 'rem-001',
                title: 'Investigate Battery Drain',
                description: 'Identify and address causes of excessive battery consumption',
                detailed_steps: [
                    'Check battery usage by app in device settings',
                    'Identify background apps consuming power',
                    'Review location services and sync settings',
                    'Consider disabling power-hungry features',
                ],
                priority: 1,
                confidence_score: 0.85,
                confidence_level: 'high',
                source: 'policy',
                source_details: 'Standard procedure for battery issues',
                historical_success_rate: 0.82,
                historical_sample_size: 47,
                estimated_impact: 'Could improve battery life by 20-40%',
                is_automated: false,
                automation_type: null,
            },
            time_range_start: threeHoursAgo.toISOString(),
            time_range_end: oneHourAgo.toISOString(),
            sample_anomalies: batteryMembers,
            grouping_factors: ['Same suggested fix: Investigate Battery Drain'],
        },
        {
            group_id: 'grp-network-001',
            group_name: 'Network Disconnections at Harbor Point (4 devices)',
            group_category: 'network_disconnect_pattern',
            group_type: 'category_match',
            severity: 'critical' as Severity,
            total_count: 4,
            open_count: 3,
            device_count: 4,
            common_location: 'Harbor Point',
            suggested_remediation: {
                remediation_id: 'rem-002',
                title: 'Diagnose Network Connectivity',
                description: 'Investigate and resolve network disconnection issues at Harbor Point',
                detailed_steps: [
                    'Check WiFi signal strength at Harbor Point',
                    'Verify network credentials are current',
                    'Test device connectivity in different areas',
                    'Consider network infrastructure review',
                ],
                priority: 1,
                confidence_score: 0.78,
                confidence_level: 'high',
                source: 'ai_generated',
                source_details: 'Location-specific pattern detected',
                historical_success_rate: null,
                historical_sample_size: null,
                estimated_impact: 'Could stabilize connectivity for 4+ devices',
                is_automated: false,
                automation_type: null,
            },
            time_range_start: threeHoursAgo.toISOString(),
            time_range_end: oneHourAgo.toISOString(),
            sample_anomalies: networkMembers,
            grouping_factors: ['Same category: Network Disconnections', 'Same location: Harbor Point'],
        },
        {
            group_id: 'grp-storage-001',
            group_name: 'Clear Device Storage (3 devices)',
            group_category: 'storage_clear',
            group_type: 'remediation_match',
            severity: 'critical' as Severity,
            total_count: 3,
            open_count: 2,
            device_count: 3,
            suggested_remediation: {
                remediation_id: 'rem-003',
                title: 'Clear Device Storage',
                description: 'Free up storage space by removing unnecessary files and apps',
                detailed_steps: [
                    'Review and remove unused applications',
                    'Clear application caches',
                    'Remove old downloads and media files',
                    'Consider offloading data to cloud storage',
                ],
                priority: 1,
                confidence_score: 0.92,
                confidence_level: 'high',
                source: 'policy',
                source_details: 'Standard procedure for low storage',
                historical_success_rate: 0.95,
                historical_sample_size: 128,
                estimated_impact: 'Could recover 500MB-2GB of storage',
                is_automated: false,
                automation_type: null,
            },
            time_range_start: threeHoursAgo.toISOString(),
            time_range_end: oneHourAgo.toISOString(),
            sample_anomalies: storageMembers,
            grouping_factors: ['Same suggested fix: Clear Device Storage'],
        },
        {
            group_id: 'grp-cohort-001',
            group_name: 'Issues on Zebra TC52 (3 devices)',
            group_category: 'cohort_performance_issue',
            group_type: 'temporal_cluster',
            severity: 'high' as Severity,
            total_count: 3,
            open_count: 3,
            device_count: 3,
            common_device_model: 'Zebra TC52',
            time_range_start: threeHoursAgo.toISOString(),
            time_range_end: oneHourAgo.toISOString(),
            sample_anomalies: cohortMembers,
            grouping_factors: ['Same device model: Zebra TC52', 'Within 24h time window'],
        },
    ];

    // Create ungrouped anomalies
    const ungroupedAnomalies: AnomalyGroupMember[] = [
        { anomaly_id: 5001, device_id: 501, anomaly_score: -0.35, severity: 'medium', status: 'open', timestamp: oneHourAgo.toISOString(), device_model: 'Panasonic Toughbook N1', location: 'Airport Terminal', primary_metric: 'upload' },
        { anomaly_id: 5002, device_id: 502, anomaly_score: -0.32, severity: 'medium', status: 'open', timestamp: twoHoursAgo.toISOString(), device_model: 'iPad Pro 11', location: 'Downtown Flagship', primary_metric: 'download' },
    ];

    return {
        groups,
        total_anomalies: 17,
        total_groups: 4,
        ungrouped_count: 2,
        ungrouped_anomalies: ungroupedAnomalies,
        grouping_method: 'smart_auto',
        computed_at: now.toISOString(),
    };
}

// Export all mock devices and anomalies for direct access if needed
export { MOCK_DEVICES, MOCK_ANOMALIES, MOCK_STORES };
