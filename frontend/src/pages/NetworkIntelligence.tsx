/**
 * Network Intelligence Page
 *
 * Comprehensive network analysis dashboard showing:
 * - DeviceGroup hierarchy navigation (PATH-based grouping)
 * - WiFi AP quality and coverage
 * - Cellular signal analysis
 * - Data usage patterns by application
 * - Network switching behavior
 * - Dead zone identification
 */

import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { Card } from '../components/Card';
import { useMockMode } from '../hooks/useMockMode';
import { api } from '../api/client';
import type {
  APQualityResponse,
  NetworkSummaryResponse,
  PerAppUsageResponse,
  CarrierStatsResponse,
  DeviceGroupHierarchyResponse,
} from '../api/client';
import { DeviceGroupTreeNav } from '../components/network/DeviceGroupTreeNav';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts';

// Re-export types for local use
type APQuality = APQualityResponse;
type NetworkSummary = NetworkSummaryResponse;
type PerAppUsage = PerAppUsageResponse;
type CarrierStats = CarrierStatsResponse;

// Mock data generators for fallback
function generateMockNetworkSummary(): NetworkSummary {
  return {
    tenant_id: 'demo',
    device_group_id: null,
    device_group_name: null,
    device_group_path: null,
    total_devices: 1250,
    total_aps: 156,
    good_aps: 128,
    problematic_aps: 28,
    avg_signal_strength: -62,
    avg_drop_rate: 0.034,
    fleet_network_score: 78.5,
    wifi_vs_cellular_ratio: 0.73,
    devices_in_dead_zones: 12,
    recommendations: [
      'Consider relocating or boosting signal for 3 APs in warehouse section',
      'High data usage detected from unknown app on 8 devices - investigate',
      'Cellular fallback increasing - verify WiFi coverage in loading dock area',
    ],
    generated_at: new Date().toISOString(),
  };
}

function generateMockAPQuality(): APQuality[] {
  const locations = ['Main Office', 'Warehouse A', 'Warehouse B', 'Loading Dock', 'Break Room', 'Server Room'];
  return Array.from({ length: 20 }, (_, i) => ({
    ssid: `CORP-WiFi-${String(i + 1).padStart(2, '0')}`,
    bssid: `AA:BB:CC:DD:EE:${String(i).padStart(2, '0')}`,
    avg_signal_dbm: -45 - Math.random() * 40,
    drop_rate: Math.random() * 0.1,
    device_count: Math.floor(5 + Math.random() * 30),
    quality_score: 60 + Math.random() * 40,
    location: locations[i % locations.length],
  })).sort((a, b) => b.device_count - a.device_count);
}

function generateMockPerAppUsage(): PerAppUsage[] {
  return [
    { app_name: 'MobiControl Agent', data_download_mb: 1250, data_upload_mb: 890, device_count: 450, is_background: true },
    { app_name: 'Google Play Services', data_download_mb: 980, data_upload_mb: 120, device_count: 445, is_background: true },
    { app_name: 'Chrome', data_download_mb: 2100, data_upload_mb: 180, device_count: 320, is_background: false },
    { app_name: 'Outlook', data_download_mb: 650, data_upload_mb: 420, device_count: 280, is_background: false },
    { app_name: 'Teams', data_download_mb: 1800, data_upload_mb: 1200, device_count: 190, is_background: false },
    { app_name: 'Warehouse Scanner', data_download_mb: 45, data_upload_mb: 890, device_count: 150, is_background: false },
    { app_name: 'Unknown App', data_download_mb: 2500, data_upload_mb: 1800, device_count: 8, is_background: true },
    { app_name: 'Spotify', data_download_mb: 3200, data_upload_mb: 5, device_count: 45, is_background: false },
  ];
}

function generateMockCarrierStats(): CarrierStats[] {
  return [
    { carrier_name: 'Verizon', device_count: 180, avg_signal: -78, avg_latency_ms: 45, reliability_score: 92 },
    { carrier_name: 'AT&T', device_count: 145, avg_signal: -82, avg_latency_ms: 52, reliability_score: 88 },
    { carrier_name: 'T-Mobile', device_count: 98, avg_signal: -75, avg_latency_ms: 38, reliability_score: 85 },
    { carrier_name: 'Sprint/T-Mobile', device_count: 27, avg_signal: -88, avg_latency_ms: 65, reliability_score: 72 },
  ];
}

// SVG Tab Icons
function ChartBarIcon({ className = 'w-4 h-4' }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
        d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
      />
    </svg>
  );
}

function WifiIcon({ className = 'w-4 h-4' }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
        d="M8.111 16.404a5.5 5.5 0 017.778 0M12 20h.01m-7.08-7.071c3.904-3.905 10.236-3.905 14.141 0M1.394 9.393c5.857-5.857 15.355-5.857 21.213 0"
      />
    </svg>
  );
}

function DevicePhoneMobileIcon({ className = 'w-4 h-4' }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
        d="M10.5 1.5H8.25A2.25 2.25 0 006 3.75v16.5a2.25 2.25 0 002.25 2.25h7.5A2.25 2.25 0 0018 20.25V3.75a2.25 2.25 0 00-2.25-2.25H13.5m-3 0V3h3V1.5m-3 0h3m-3 18.75h3"
      />
    </svg>
  );
}

function SignalIcon({ className = 'w-4 h-4' }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
        d="M3 4h2v16H3V4zm4 2h2v14H7V6zm4 4h2v10h-2V10zm4 2h2v8h-2v-8zm4 2h2v6h-2v-6z"
      />
    </svg>
  );
}

// KPI Card component
function KpiCard({
  label,
  value,
  subValue,
  icon,
  trend,
  color = 'stellar',
}: {
  label: string;
  value: string | number;
  subValue?: string;
  icon: React.ReactNode;
  trend?: 'up' | 'down' | 'stable';
  color?: 'stellar' | 'aurora' | 'nebula' | 'cosmic';
}) {
  const colorClasses = {
    stellar: 'text-stellar-400 bg-stellar-500/10',
    aurora: 'text-aurora-400 bg-aurora-500/10',
    nebula: 'text-nebula-400 bg-nebula-500/10',
    cosmic: 'text-cosmic-400 bg-cosmic-500/10',
  };

  return (
    <Card className="p-4">
      <div className="flex items-start gap-3">
        <div className={`p-2 rounded-lg ${colorClasses[color]}`}>{icon}</div>
        <div className="flex-1 min-w-0">
          <p className="text-xs text-slate-500 uppercase tracking-wide">{label}</p>
          <div className="flex items-baseline gap-2">
            <p className={`text-2xl font-bold mt-0.5 ${colorClasses[color].split(' ')[0]}`}>
              {value}
            </p>
            {trend && (
              <span className={`text-xs ${trend === 'up' ? 'text-emerald-400' : trend === 'down' ? 'text-red-400' : 'text-slate-500'}`}>
                {trend === 'up' ? '+' : trend === 'down' ? '-' : '='}{subValue}
              </span>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
}

// Signal Quality Badge
function SignalBadge({ dbm }: { dbm: number }) {
  let color = 'bg-emerald-500/20 text-emerald-400';
  let label = 'Excellent';

  if (dbm < -80) {
    color = 'bg-red-500/20 text-red-400';
    label = 'Poor';
  } else if (dbm < -70) {
    color = 'bg-orange-500/20 text-orange-400';
    label = 'Fair';
  } else if (dbm < -60) {
    color = 'bg-yellow-500/20 text-yellow-400';
    label = 'Good';
  }

  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${color}`}>
      {label} ({dbm} dBm)
    </span>
  );
}

export default function NetworkIntelligence() {
  const { mockMode } = useMockMode();
  const [selectedTab, setSelectedTab] = useState<'overview' | 'aps' | 'apps' | 'carriers'>('overview');
  const [selectedGroupId, setSelectedGroupId] = useState<number | null>(null);

  // Query for device group hierarchy
  const { data: hierarchyData, isLoading: isHierarchyLoading } = useQuery({
    queryKey: ['network', 'hierarchy'],
    queryFn: () => api.getNetworkHierarchy(),
    staleTime: 300000, // 5 minutes - hierarchy doesn't change often
  });

  // Find the selected group info from hierarchy
  const findGroupById = (groups: DeviceGroupHierarchyResponse['groups'], id: number | null): {
    name: string | null;
    path: string | null;
  } => {
    if (id === null) return { name: null, path: null };

    const search = (nodes: DeviceGroupHierarchyResponse['groups']): { name: string; path: string } | null => {
      for (const node of nodes) {
        if (node.device_group_id === id) {
          return { name: node.group_name, path: node.full_path };
        }
        if (node.children.length > 0) {
          const found = search(node.children);
          if (found) return found;
        }
      }
      return null;
    };

    return search(groups) ?? { name: null, path: null };
  };

  const selectedGroupInfo = hierarchyData ? findGroupById(hierarchyData.groups, selectedGroupId) : { name: null, path: null };

  // Queries - use real API with device_group_id filter
  const { data: summary } = useQuery({
    queryKey: ['network', 'summary', selectedGroupId],
    queryFn: () => mockMode
      ? Promise.resolve(generateMockNetworkSummary())
      : api.getNetworkSummary({ device_group_id: selectedGroupId ?? undefined }),
    staleTime: 60000,
  });

  const { data: apListResponse } = useQuery({
    queryKey: ['network', 'aps', selectedGroupId],
    queryFn: () => mockMode
      ? Promise.resolve({ aps: generateMockAPQuality(), total_count: 20 })
      : api.getAPQuality({ device_group_id: selectedGroupId ?? undefined }),
    staleTime: 60000,
  });
  const apList = apListResponse?.aps;

  const { data: appUsageResponse } = useQuery({
    queryKey: ['network', 'apps', selectedGroupId],
    queryFn: () => mockMode
      ? Promise.resolve({ apps: generateMockPerAppUsage(), total_download_mb: 0, total_upload_mb: 0 })
      : api.getPerAppUsage({ device_group_id: selectedGroupId ?? undefined }),
    staleTime: 60000,
  });
  const appUsage = appUsageResponse?.apps;

  const { data: carrierStatsResponse } = useQuery({
    queryKey: ['network', 'carriers', selectedGroupId],
    queryFn: () => mockMode
      ? Promise.resolve({ carriers: generateMockCarrierStats() })
      : api.getCarrierStats({ device_group_id: selectedGroupId ?? undefined }),
    staleTime: 60000,
  });
  const carrierStats = carrierStatsResponse?.carriers;

  // Prepare chart data
  const networkTypeData = [
    { name: 'WiFi', value: summary ? Math.round(summary.wifi_vs_cellular_ratio * 100) : 73, color: '#10B981' },
    { name: 'Cellular', value: summary ? Math.round((1 - summary.wifi_vs_cellular_ratio) * 100) : 27, color: '#6366F1' },
  ];

  const apQualityDistribution = apList ? [
    { range: 'Excellent (>-60)', count: apList.filter(ap => ap.avg_signal_dbm > -60).length },
    { range: 'Good (-60 to -70)', count: apList.filter(ap => ap.avg_signal_dbm > -70 && ap.avg_signal_dbm <= -60).length },
    { range: 'Fair (-70 to -80)', count: apList.filter(ap => ap.avg_signal_dbm > -80 && ap.avg_signal_dbm <= -70).length },
    { range: 'Poor (<-80)', count: apList.filter(ap => ap.avg_signal_dbm <= -80).length },
  ] : [];

  const tabs = [
    { id: 'overview' as const, label: 'Overview', icon: <ChartBarIcon /> },
    { id: 'aps' as const, label: 'WiFi Networks', icon: <WifiIcon /> },
    { id: 'apps' as const, label: 'App Usage', icon: <DevicePhoneMobileIcon /> },
    { id: 'carriers' as const, label: 'Carriers', icon: <SignalIcon /> },
  ];

  return (
    <div className="flex h-screen">
      {/* Left Sidebar: Device Group Tree Navigation */}
      <DeviceGroupTreeNav
        groups={hierarchyData?.groups ?? []}
        selectedGroupId={selectedGroupId}
        onSelectGroup={setSelectedGroupId}
        isLoading={isHierarchyLoading}
      />

      {/* Main Content */}
      <div className="flex-1 min-h-screen p-6 space-y-6 overflow-y-auto">
        {/* Breadcrumb */}
        {selectedGroupId !== null && selectedGroupInfo.path && (
          <div className="flex items-center gap-2 text-sm">
            <button
              onClick={() => setSelectedGroupId(null)}
              className="text-slate-400 hover:text-white transition-colors"
            >
              All Devices
            </button>
            <svg className="w-4 h-4 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
            <span className="text-amber-400">{selectedGroupInfo.path}</span>
          </div>
        )}

        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white">Network Intelligence</h1>
            <p className="text-slate-400 text-sm mt-1">
              {selectedGroupId !== null && selectedGroupInfo.name
                ? `Metrics for: ${selectedGroupInfo.name} (${summary?.total_devices?.toLocaleString() ?? 0} devices)`
                : 'Fleet-wide WiFi, cellular, and data usage analytics'
              }
            </p>
          </div>
          {mockMode && (
            <span className="px-3 py-1 bg-amber-500/20 text-amber-400 text-xs rounded-full">
              Demo Mode
            </span>
          )}
        </div>

        {/* Tab Navigation */}
        <div className="flex gap-2 border-b border-slate-700/50 pb-2">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setSelectedTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                selectedTab === tab.id
                  ? 'bg-stellar-500/20 text-stellar-400'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
              }`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>

        {/* Overview Tab */}
        {selectedTab === 'overview' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            {/* KPI Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <KpiCard
                label="Fleet Network Score"
                value={`${summary?.fleet_network_score?.toFixed(1) ?? 0}%`}
                icon={<svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>}
                color="stellar"
              />
              <KpiCard
                label="WiFi Networks (SSIDs)"
                value={summary?.total_aps ?? 0}
                subValue={`${summary?.good_aps ?? 0} healthy`}
                icon={<svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.111 16.404a5.5 5.5 0 017.778 0M12 20h.01m-7.08-7.071c3.904-3.905 10.236-3.905 14.141 0M1.394 9.393c5.857-5.857 15.355-5.857 21.213 0" />
                </svg>}
                color="aurora"
              />
              <KpiCard
                label="Avg Signal Strength"
                value={`${summary?.avg_signal_strength?.toFixed(0) ?? 0} dBm`}
                icon={<svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>}
                color="nebula"
              />
              <KpiCard
                label="Devices in Dead Zones"
                value={summary?.devices_in_dead_zones ?? 0}
                icon={<svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>}
                color="cosmic"
              />
            </div>

            {/* Charts Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Network Type Distribution */}
              <Card className="p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Network Type Distribution</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={networkTypeData}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={80}
                        dataKey="value"
                        label={({ name, value }) => `${name}: ${value}%`}
                      >
                        {networkTypeData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </Card>

              {/* WiFi Network Signal Quality Distribution */}
              <Card className="p-6">
                <h3 className="text-lg font-semibold text-white mb-4">WiFi Signal Quality by Network</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={apQualityDistribution}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="range" tick={{ fill: '#9CA3AF', fontSize: 10 }} />
                      <YAxis tick={{ fill: '#9CA3AF' }} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }}
                      />
                      <Bar dataKey="count" fill="#F59E0B" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </div>

            {/* Recommendations */}
            {summary?.recommendations && summary.recommendations.length > 0 && (
              <Card className="p-6">
                <h3 className="text-lg font-semibold text-white mb-4">
                  {selectedGroupId !== null ? 'Group-Specific Insights' : 'Fleet Recommendations'}
                </h3>
                <div className="space-y-3">
                  {summary.recommendations.map((rec, i) => (
                    <div key={i} className="flex items-start gap-3 p-3 bg-slate-800/50 rounded-lg">
                      <div className="p-1.5 bg-amber-500/20 rounded">
                        <svg className="w-4 h-4 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                        </svg>
                      </div>
                      <p className="text-sm text-slate-300">{rec}</p>
                    </div>
                  ))}
                </div>
              </Card>
            )}
          </motion.div>
        )}

        {/* WiFi Networks Tab */}
        {selectedTab === 'aps' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <Card className="p-6">
              <div className="mb-4">
                <h3 className="text-lg font-semibold text-white">WiFi Networks</h3>
                <p className="text-sm text-slate-400 mt-1">Network performance aggregated by SSID across all connected devices</p>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="text-left text-xs text-slate-500 uppercase tracking-wide border-b border-slate-700">
                      <th className="pb-3 pr-4">Network Name (SSID)</th>
                      <th className="pb-3 pr-4">Avg Signal</th>
                      <th className="pb-3 pr-4">Drop Rate</th>
                      <th className="pb-3 pr-4">Connected Devices</th>
                      <th className="pb-3">Quality Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {apList?.map((ap, i) => (
                      <tr key={i} className="border-b border-slate-800 hover:bg-slate-800/50">
                        <td className="py-3 pr-4">
                          <div className="font-medium text-white">{ap.ssid}</div>
                        </td>
                        <td className="py-3 pr-4">
                          <SignalBadge dbm={Math.round(ap.avg_signal_dbm)} />
                        </td>
                        <td className="py-3 pr-4 text-slate-400">{(ap.drop_rate * 100).toFixed(1)}%</td>
                        <td className="py-3 pr-4 text-slate-400">{ap.device_count}</td>
                        <td className="py-3">
                          <div className="flex items-center gap-2">
                            <div className="w-24 h-2 bg-slate-700 rounded-full overflow-hidden">
                              <div
                                className={`h-full rounded-full ${
                                  ap.quality_score >= 80 ? 'bg-emerald-500' :
                                  ap.quality_score >= 60 ? 'bg-yellow-500' : 'bg-red-500'
                                }`}
                                style={{ width: `${ap.quality_score}%` }}
                              />
                            </div>
                            <span className="text-xs text-slate-500">{ap.quality_score.toFixed(0)}%</span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </motion.div>
        )}

        {/* App Usage Tab */}
        {selectedTab === 'apps' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            <Card className="p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Data Usage by Application</h3>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={appUsage} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis type="number" tick={{ fill: '#9CA3AF' }} />
                    <YAxis dataKey="app_name" type="category" tick={{ fill: '#9CA3AF', fontSize: 11 }} width={120} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }}
                      formatter={(value: number) => `${value.toLocaleString()} MB`}
                    />
                    <Legend />
                    <Bar dataKey="data_download_mb" name="Download" fill="#10B981" stackId="a" />
                    <Bar dataKey="data_upload_mb" name="Upload" fill="#6366F1" stackId="a" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>

            {/* Suspicious activity alert */}
            {appUsage?.some(app => app.app_name.includes('Unknown')) && (
              <Card className="p-4 border-red-500/30 bg-red-500/5">
                <div className="flex items-start gap-3">
                  <div className="p-2 bg-red-500/20 rounded-lg">
                    <svg className="w-5 h-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                  </div>
                  <div>
                    <h4 className="font-semibold text-red-400">Suspicious Data Activity Detected</h4>
                    <p className="text-sm text-slate-400 mt-1">
                      Unknown application with high upload activity detected on 8 devices. This may indicate data exfiltration.
                    </p>
                    <Link
                      to="/investigations"
                      className="mt-2 inline-block text-sm text-red-400 hover:text-red-300"
                    >
                      Investigate &rarr;
                    </Link>
                  </div>
                </div>
              </Card>
            )}
          </motion.div>
        )}

        {/* Carriers Tab */}
        {selectedTab === 'carriers' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <Card className="p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Carrier Performance Comparison</h3>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Radar Chart */}
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={carrierStats}>
                      <PolarGrid stroke="#374151" />
                      <PolarAngleAxis dataKey="carrier_name" tick={{ fill: '#9CA3AF', fontSize: 11 }} />
                      <PolarRadiusAxis tick={{ fill: '#6B7280' }} />
                      <Radar
                        name="Reliability Score"
                        dataKey="reliability_score"
                        stroke="#F59E0B"
                        fill="#F59E0B"
                        fillOpacity={0.3}
                      />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }}
                      />
                      <Legend />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>

                {/* Stats Table */}
                <div>
                  <table className="w-full">
                    <thead>
                      <tr className="text-left text-xs text-slate-500 uppercase tracking-wide border-b border-slate-700">
                        <th className="pb-3">Carrier</th>
                        <th className="pb-3">Devices</th>
                        <th className="pb-3">Avg Signal</th>
                        <th className="pb-3">Latency</th>
                        <th className="pb-3">Reliability</th>
                      </tr>
                    </thead>
                    <tbody>
                      {carrierStats?.map((carrier, i) => (
                        <tr key={i} className="border-b border-slate-800">
                          <td className="py-3 font-medium text-white">{carrier.carrier_name}</td>
                          <td className="py-3 text-slate-400">{carrier.device_count}</td>
                          <td className="py-3 text-slate-400">{carrier.avg_signal} dBm</td>
                          <td className="py-3 text-slate-400">{carrier.avg_latency_ms} ms</td>
                          <td className="py-3">
                            <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                              carrier.reliability_score >= 90 ? 'bg-emerald-500/20 text-emerald-400' :
                              carrier.reliability_score >= 80 ? 'bg-yellow-500/20 text-yellow-400' :
                              'bg-red-500/20 text-red-400'
                            }`}>
                              {carrier.reliability_score}%
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </Card>
          </motion.div>
        )}
      </div>
    </div>
  );
}
