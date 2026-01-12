/**
 * Security Posture Page
 *
 * Fleet security compliance dashboard showing:
 * - Overall security score
 * - Device compliance status
 * - Encryption status
 * - Rooting/jailbreak detection
 * - Security patch levels
 * - Policy violations
 */

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { Card } from '../components/Card';
import { useMockMode } from '../hooks/useMockMode';
import { api } from '../api/client';
import type {
  SecuritySummaryResponse,
  DeviceSecurityStatusResponse,
  ComplianceBreakdownResponse,
  SecurityTrendResponse,
  PathHierarchyResponse,
  PathNodeResponse,
  RiskClustersResponse,
  PathComparisonResponse,
} from '../api/client';
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
  RadialBarChart,
  RadialBar,
  AreaChart,
  Area,
} from 'recharts';

// Re-export types for local use
type SecuritySummary = SecuritySummaryResponse;
type DeviceSecurityStatus = DeviceSecurityStatusResponse;
type ComplianceBreakdown = ComplianceBreakdownResponse;
type SecurityTrend = SecurityTrendResponse;

// Mock data generators
function generateMockSecuritySummary(): SecuritySummary {
  return {
    tenant_id: 'demo',
    fleet_security_score: 72.5,
    total_devices: 450,
    compliant_devices: 385,
    at_risk_devices: 52,
    critical_risk_devices: 13,
    encrypted_devices: 438,
    rooted_devices: 8,
    outdated_patch_devices: 45,
    usb_debugging_enabled: 23,
    developer_mode_enabled: 31,
    no_passcode_devices: 5,
    recommendations: [
      'Enforce encryption on 12 remaining unencrypted devices',
      'Investigate 8 rooted devices for potential security compromise',
      'Push security patches to 45 devices with outdated patches (>60 days)',
      'Disable developer mode on 31 devices in production environment',
      'Enforce passcode policy on 5 non-compliant devices',
    ],
    generated_at: new Date().toISOString(),
  };
}

function generateMockDeviceList(): DeviceSecurityStatus[] {
  return Array.from({ length: 25 }, (_, i) => {
    const isRooted = Math.random() < 0.02;
    const isEncrypted = Math.random() > 0.03;
    const hasPasscode = Math.random() > 0.01;
    const usbDebugging = Math.random() < 0.05;
    const devMode = Math.random() < 0.07;
    const patchAge = Math.floor(Math.random() * 120);

    const violations: string[] = [];
    if (isRooted) violations.push('Device is rooted');
    if (!isEncrypted) violations.push('Storage not encrypted');
    if (!hasPasscode) violations.push('No passcode set');
    if (usbDebugging) violations.push('USB debugging enabled');
    if (devMode) violations.push('Developer mode enabled');
    if (patchAge > 60) violations.push(`Security patch ${patchAge} days old`);

    let score = 100;
    if (isRooted) score -= 40;
    if (!isEncrypted) score -= 25;
    if (!hasPasscode) score -= 15;
    if (usbDebugging) score -= 10;
    if (devMode) score -= 5;
    if (patchAge > 60) score -= Math.min(20, patchAge / 3);

    let riskLevel: 'low' | 'medium' | 'high' | 'critical' = 'low';
    if (score < 50) riskLevel = 'critical';
    else if (score < 70) riskLevel = 'high';
    else if (score < 85) riskLevel = 'medium';

    return {
      device_id: 1000 + i,
      device_name: `Device-${String(i + 1).padStart(3, '0')}`,
      security_score: Math.max(0, score),
      is_encrypted: isEncrypted,
      is_rooted: isRooted,
      has_passcode: hasPasscode,
      patch_age_days: patchAge,
      usb_debugging: usbDebugging,
      developer_mode: devMode,
      risk_level: riskLevel,
      violations,
    };
  }).sort((a, b) => a.security_score - b.security_score);
}

function generateMockComplianceBreakdown(): ComplianceBreakdown[] {
  return [
    { category: 'Encryption', compliant: 438, non_compliant: 12, total: 450, compliance_pct: 97.3 },
    { category: 'Passcode', compliant: 445, non_compliant: 5, total: 450, compliance_pct: 98.9 },
    { category: 'Patch Level', compliant: 405, non_compliant: 45, total: 450, compliance_pct: 90.0 },
    { category: 'Root Status', compliant: 442, non_compliant: 8, total: 450, compliance_pct: 98.2 },
    { category: 'Developer Mode', compliant: 419, non_compliant: 31, total: 450, compliance_pct: 93.1 },
    { category: 'USB Debugging', compliant: 427, non_compliant: 23, total: 450, compliance_pct: 94.9 },
  ];
}

function generateMockSecurityTrend(): SecurityTrend[] {
  const today = new Date();
  return Array.from({ length: 30 }, (_, i) => {
    const date = new Date(today);
    date.setDate(date.getDate() - (29 - i));
    return {
      date: date.toISOString().split('T')[0],
      score: 65 + Math.random() * 15 + (i * 0.2),
      compliant_pct: 80 + Math.random() * 10 + (i * 0.1),
    };
  });
}

function generateMockPathHierarchy(): PathHierarchyResponse {
  return {
    tenant_id: 'demo',
    hierarchy: [
      {
        path_id: 'group-1',
        path_name: 'North America',
        full_path: 'North America',
        device_count: 300,
        security_score: 74.2,
        compliant_count: 255,
        at_risk_count: 35,
        critical_count: 7,
        children: [
          {
            path_id: 'group-2',
            path_name: 'East Region',
            full_path: 'North America / East Region',
            device_count: 150,
            security_score: 78.5,
            compliant_count: 135,
            at_risk_count: 15,
            critical_count: 3,
            children: [
              { path_id: 'group-3', path_name: 'Store-NYC-001', full_path: 'North America / East Region / Store-NYC-001', device_count: 45, security_score: 82.1, compliant_count: 41, at_risk_count: 4, critical_count: 0, children: [] },
              { path_id: 'group-4', path_name: 'Store-NYC-002', full_path: 'North America / East Region / Store-NYC-002', device_count: 38, security_score: 75.3, compliant_count: 33, at_risk_count: 5, critical_count: 1, children: [] },
              { path_id: 'group-5', path_name: 'Warehouse-NYC', full_path: 'North America / East Region / Warehouse-NYC', device_count: 67, security_score: 71.0, compliant_count: 56, at_risk_count: 11, critical_count: 2, children: [] },
            ],
          },
          {
            path_id: 'group-6',
            path_name: 'West Region',
            full_path: 'North America / West Region',
            device_count: 150,
            security_score: 68.1,
            compliant_count: 118,
            at_risk_count: 25,
            critical_count: 7,
            children: [
              { path_id: 'group-7', path_name: 'Store-LA-001', full_path: 'North America / West Region / Store-LA-001', device_count: 52, security_score: 71.2, compliant_count: 44, at_risk_count: 8, critical_count: 1, children: [] },
              { path_id: 'group-8', path_name: 'Distribution-LA', full_path: 'North America / West Region / Distribution-LA', device_count: 98, security_score: 62.8, compliant_count: 77, at_risk_count: 21, critical_count: 6, children: [] },
            ],
          },
        ],
      },
      {
        path_id: 'group-10',
        path_name: 'Europe',
        full_path: 'Europe',
        device_count: 150,
        security_score: 76.8,
        compliant_count: 133,
        at_risk_count: 17,
        critical_count: 6,
        children: [
          { path_id: 'group-11', path_name: 'UK', full_path: 'Europe / UK', device_count: 85, security_score: 79.2, compliant_count: 76, at_risk_count: 9, critical_count: 2, children: [] },
          { path_id: 'group-12', path_name: 'Germany', full_path: 'Europe / Germany', device_count: 65, security_score: 73.5, compliant_count: 57, at_risk_count: 8, critical_count: 4, children: [] },
        ],
      },
    ],
    total_paths: 12,
    total_devices: 450,
  };
}

function generateMockRiskClusters(): RiskClustersResponse {
  return {
    tenant_id: 'demo',
    clusters: [
      { violation_type: 'Rooted Devices', device_count: 8, avg_security_score: 35.2, device_ids: [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008], common_paths: ['Warehouse-NYC', 'Distribution-LA'], severity: 'critical', recommendation: 'Investigate for potential compromise and consider device wipe' },
      { violation_type: 'Unencrypted Storage', device_count: 12, avg_security_score: 52.1, device_ids: [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012], common_paths: ['Distribution-LA', 'Store-Berlin-001'], severity: 'critical', recommendation: 'Enable encryption policy and verify compliance' },
      { violation_type: 'Outdated Patches >60 days', device_count: 45, avg_security_score: 62.1, device_ids: [], common_paths: ['Store-NYC-001', 'Store-LA-001', 'Warehouse-Manchester'], severity: 'high', recommendation: 'Schedule security patch deployment' },
      { violation_type: 'USB Debugging Enabled', device_count: 23, avg_security_score: 78.5, device_ids: [], common_paths: ['Store-NYC-001', 'Store-LA-001'], severity: 'medium', recommendation: 'Push policy to disable USB debugging' },
      { violation_type: 'Developer Mode Enabled', device_count: 31, avg_security_score: 81.2, device_ids: [], common_paths: ['Warehouse-NYC', 'Store-Berlin-001'], severity: 'medium', recommendation: 'Disable developer mode on production devices' },
    ],
    total_devices_affected: 119,
    coverage_percent: 26.4,
  };
}

function generateMockPathComparison(paths: string[]): PathComparisonResponse {
  const scores: Record<string, number> = {
    'Store-NYC-001': 82.1,
    'Warehouse-NYC': 71.0,
    'Store-LA-001': 71.2,
    'Distribution-LA': 62.8,
    'Store-London-001': 79.2,
    'Store-Berlin-001': 73.5,
  };
  const fleetAvg = 72.5;

  return {
    tenant_id: 'demo',
    paths: paths.map((path) => {
      const pathName = path.split(' / ').pop() || path;
      const score = scores[pathName] || 70;
      return {
        path,
        path_name: pathName,
        security_score: score,
        device_count: Math.floor(Math.random() * 50) + 30,
        compliance_pct: score + Math.random() * 10,
        vs_fleet_delta: score - fleetAvg,
      };
    }),
    fleet_average_score: fleetAvg,
    insights: paths.length >= 2 ? [
      `${paths[0].split(' / ').pop()} leads with highest security score`,
      `${paths[paths.length - 1].split(' / ').pop()} needs attention - below fleet average`,
    ] : [],
  };
}

// Colors
const RISK_COLORS = {
  low: '#10B981',
  medium: '#F59E0B',
  high: '#F97316',
  critical: '#EF4444',
};

// KPI Card component
function KpiCard({
  label,
  value,
  subValue,
  icon,
  color = 'stellar',
}: {
  label: string;
  value: string | number;
  subValue?: string;
  icon: React.ReactNode;
  color?: 'stellar' | 'aurora' | 'nebula' | 'cosmic' | 'danger';
}) {
  const colorClasses = {
    stellar: 'text-stellar-400 bg-stellar-500/10',
    aurora: 'text-aurora-400 bg-aurora-500/10',
    nebula: 'text-nebula-400 bg-nebula-500/10',
    cosmic: 'text-cosmic-400 bg-cosmic-500/10',
    danger: 'text-red-400 bg-red-500/10',
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
            {subValue && (
              <span className="text-xs text-slate-500">{subValue}</span>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
}

// Security Score Gauge
function SecurityScoreGauge({ score }: { score: number }) {
  const data = [{ name: 'Score', value: score, fill: score >= 80 ? '#10B981' : score >= 60 ? '#F59E0B' : '#EF4444' }];

  return (
    <div className="relative h-48">
      <ResponsiveContainer width="100%" height="100%">
        <RadialBarChart
          cx="50%"
          cy="50%"
          innerRadius="60%"
          outerRadius="80%"
          data={data}
          startAngle={180}
          endAngle={0}
        >
          <RadialBar
            background={{ fill: '#1F2937' }}
            dataKey="value"
            cornerRadius={10}
          />
        </RadialBarChart>
      </ResponsiveContainer>
      <div className="absolute inset-0 flex flex-col items-center justify-center pt-8">
        <span className={`text-4xl font-bold ${score >= 80 ? 'text-emerald-400' : score >= 60 ? 'text-amber-400' : 'text-red-400'}`}>
          {score.toFixed(1)}
        </span>
        <span className="text-xs text-slate-500 uppercase">Fleet Score</span>
      </div>
    </div>
  );
}


// PATH Tree Node component - uses PathNodeResponse from API
function PathTreeNode({
  node,
  selectedPath,
  expandedPaths,
  onSelect,
  onToggle,
  depth = 0,
}: {
  node: PathNodeResponse;
  selectedPath: string | null;
  expandedPaths: Set<string>;
  onSelect: (path: string) => void;
  onToggle: (pathId: string) => void;
  depth?: number;
}) {
  const hasChildren = node.children && node.children.length > 0;
  const isExpanded = expandedPaths.has(node.path_id);
  const isSelected = selectedPath === node.full_path;

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-emerald-400';
    if (score >= 70) return 'text-amber-400';
    return 'text-red-400';
  };

  return (
    <div>
      <div
        className={`flex items-center gap-2 px-2 py-1.5 rounded cursor-pointer transition-colors ${
          isSelected ? 'bg-stellar-500/20' : 'hover:bg-slate-800/50'
        }`}
        style={{ paddingLeft: `${depth * 16 + 8}px` }}
        onClick={() => onSelect(node.full_path)}
      >
        {/* Expand/Collapse Button */}
        {hasChildren ? (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onToggle(node.path_id);
            }}
            className="p-0.5 hover:bg-slate-700 rounded"
          >
            <svg
              className={`w-4 h-4 text-slate-500 transition-transform ${isExpanded ? 'rotate-90' : ''}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>
        ) : (
          <div className="w-5" />
        )}

        {/* Node Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className={`text-sm font-medium ${isSelected ? 'text-stellar-400' : 'text-white'}`}>
              {node.path_name}
            </span>
            <span className="text-xs text-slate-500">({node.device_count})</span>
          </div>
        </div>

        {/* Score Badge */}
        <span className={`text-xs font-medium ${getScoreColor(node.security_score)}`}>
          {node.security_score.toFixed(0)}
        </span>

        {/* Risk Indicator */}
        {node.critical_count > 0 && (
          <span className="w-2 h-2 rounded-full bg-red-500" title={`${node.critical_count} critical`} />
        )}
      </div>

      {/* Children */}
      {hasChildren && isExpanded && (
        <div>
          {node.children.map((child) => (
            <PathTreeNode
              key={child.path_id}
              node={child}
              selectedPath={selectedPath}
              expandedPaths={expandedPaths}
              onSelect={onSelect}
              onToggle={onToggle}
              depth={depth + 1}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export default function SecurityPosture() {
  const { mockMode } = useMockMode();
  const [selectedTab, setSelectedTab] = useState<'overview' | 'paths' | 'clusters' | 'compare' | 'trends'>('overview');
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [expandedPaths, setExpandedPaths] = useState<Set<string>>(new Set(['group-1', 'group-10']));
  const [comparisonPaths, setComparisonPaths] = useState<string[]>([]);

  // Queries - use real API, fallback to mock in demo mode
  const { data: summary } = useQuery({
    queryKey: ['security', 'summary'],
    queryFn: () => mockMode
      ? Promise.resolve(generateMockSecuritySummary())
      : api.getSecuritySummary(),
    staleTime: 60000,
  });

  const { data: deviceListResponse } = useQuery({
    queryKey: ['security', 'devices'],
    queryFn: () => mockMode
      ? Promise.resolve({ devices: generateMockDeviceList(), total_count: 25 })
      : api.getDeviceSecurity(),
    staleTime: 60000,
  });
  const deviceList = deviceListResponse?.devices;

  const { data: complianceResponse } = useQuery({
    queryKey: ['security', 'compliance'],
    queryFn: () => mockMode
      ? Promise.resolve({ categories: generateMockComplianceBreakdown() })
      : api.getComplianceBreakdown(),
    staleTime: 60000,
  });
  const compliance = complianceResponse?.categories;

  const { data: trendsResponse } = useQuery({
    queryKey: ['security', 'trends'],
    queryFn: () => mockMode
      ? Promise.resolve({ trends: generateMockSecurityTrend(), period_days: 30 })
      : api.getSecurityTrends(),
    staleTime: 60000,
  });
  const trends = trendsResponse?.trends;

  // Path hierarchy query
  const { data: pathHierarchy } = useQuery({
    queryKey: ['security', 'paths'],
    queryFn: () => mockMode
      ? Promise.resolve(generateMockPathHierarchy())
      : api.getSecurityPathHierarchy(),
    staleTime: 60000,
  });

  // Risk clusters query
  const { data: riskClusters } = useQuery({
    queryKey: ['security', 'risk-clusters'],
    queryFn: () => mockMode
      ? Promise.resolve(generateMockRiskClusters())
      : api.getRiskClusters(),
    staleTime: 60000,
  });

  // Path comparison query (only when paths selected)
  const { data: pathComparison } = useQuery({
    queryKey: ['security', 'path-comparison', comparisonPaths],
    queryFn: () => mockMode
      ? Promise.resolve(generateMockPathComparison(comparisonPaths))
      : api.compareSecurityByPath(comparisonPaths),
    staleTime: 60000,
    enabled: comparisonPaths.length >= 2,
  });

  // Risk distribution data
  const riskDistribution = deviceList ? [
    { name: 'Low Risk', value: deviceList.filter(d => d.risk_level === 'low').length, fill: RISK_COLORS.low },
    { name: 'Medium Risk', value: deviceList.filter(d => d.risk_level === 'medium').length, fill: RISK_COLORS.medium },
    { name: 'High Risk', value: deviceList.filter(d => d.risk_level === 'high').length, fill: RISK_COLORS.high },
    { name: 'Critical', value: deviceList.filter(d => d.risk_level === 'critical').length, fill: RISK_COLORS.critical },
  ] : [];

  const tabs = [
    {
      id: 'overview' as const,
      label: 'Overview',
      icon: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
            d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
          />
        </svg>
      ),
    },
    {
      id: 'paths' as const,
      label: 'By Location',
      icon: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
            d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4"
          />
        </svg>
      ),
    },
    {
      id: 'clusters' as const,
      label: 'Risk Clusters',
      icon: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
          />
        </svg>
      ),
    },
    {
      id: 'compare' as const,
      label: 'Compare',
      icon: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
            d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
          />
        </svg>
      ),
    },
    {
      id: 'trends' as const,
      label: 'Trends',
      icon: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
            d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"
          />
        </svg>
      ),
    },
  ];

  return (
    <div className="min-h-screen p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Security Posture</h1>
          <p className="text-slate-400 text-sm mt-1">Fleet security compliance and risk assessment</p>
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
          {/* Security Score and KPIs */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Score Gauge */}
            <Card className="p-6">
              <h3 className="text-lg font-semibold text-white mb-2">Fleet Security Score</h3>
              <SecurityScoreGauge score={summary?.fleet_security_score ?? 0} />
              <p className="text-center text-sm text-slate-400 mt-2">
                {summary?.compliant_devices ?? 0} of {summary?.total_devices ?? 0} devices compliant
              </p>
            </Card>

            {/* KPI Grid */}
            <div className="lg:col-span-2 grid grid-cols-2 gap-4">
              <KpiCard
                label="At-Risk Devices"
                value={summary?.at_risk_devices ?? 0}
                subValue={`${summary?.critical_risk_devices ?? 0} critical`}
                icon={<svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>}
                color="danger"
              />
              <KpiCard
                label="Encrypted"
                value={`${summary?.encrypted_devices ?? 0}/${summary?.total_devices ?? 0}`}
                icon={<svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>}
                color="aurora"
              />
              <KpiCard
                label="Rooted/Jailbroken"
                value={summary?.rooted_devices ?? 0}
                icon={<svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
                </svg>}
                color="danger"
              />
              <KpiCard
                label="Outdated Patches"
                value={summary?.outdated_patch_devices ?? 0}
                subValue=">60 days old"
                icon={<svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>}
                color="stellar"
              />
            </div>
          </div>

          {/* Charts Row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Risk Distribution */}
            <Card className="p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Risk Distribution</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={riskDistribution}
                      cx="50%"
                      cy="50%"
                      innerRadius={50}
                      outerRadius={80}
                      dataKey="value"
                      label={({ name, value }) => `${name}: ${value}`}
                    >
                      {riskDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }}
                    />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </Card>

            {/* Compliance Breakdown */}
            <Card className="p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Compliance by Category</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={compliance} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis type="number" tick={{ fill: '#9CA3AF' }} />
                    <YAxis dataKey="category" type="category" tick={{ fill: '#9CA3AF', fontSize: 11 }} width={100} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }}
                    />
                    <Legend />
                    <Bar dataKey="compliant" name="Compliant" fill="#10B981" stackId="a" />
                    <Bar dataKey="non_compliant" name="Non-Compliant" fill="#EF4444" stackId="a" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>
          </div>

          {/* Recommendations */}
          {summary?.recommendations && summary.recommendations.length > 0 && (
            <Card className="p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Security Recommendations</h3>
              <div className="space-y-3">
                {summary.recommendations.map((rec, i) => (
                  <div key={i} className="flex items-start gap-3 p-3 bg-slate-800/50 rounded-lg">
                    <div className={`p-1.5 rounded ${i < 2 ? 'bg-red-500/20' : 'bg-amber-500/20'}`}>
                      <svg className={`w-4 h-4 ${i < 2 ? 'text-red-400' : 'text-amber-400'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
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

      {/* By Location Tab - PATH Hierarchy */}
      {selectedTab === 'paths' && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="grid grid-cols-1 lg:grid-cols-3 gap-6"
        >
          {/* PATH Hierarchy Tree */}
          <Card className="p-4 lg:col-span-1">
            <h3 className="text-lg font-semibold text-white mb-4">Location Hierarchy</h3>
            <div className="space-y-1">
              {/* Hierarchy tree from API */}
              {pathHierarchy?.hierarchy && pathHierarchy.hierarchy.length > 0 ? (
                pathHierarchy.hierarchy.map((node) => (
                  <PathTreeNode
                    key={node.path_id}
                    node={node}
                    selectedPath={selectedPath}
                    expandedPaths={expandedPaths}
                    onSelect={(path) => setSelectedPath(path)}
                    onToggle={(pathId) => {
                      const newExpanded = new Set(expandedPaths);
                      if (newExpanded.has(pathId)) {
                        newExpanded.delete(pathId);
                      } else {
                        newExpanded.add(pathId);
                      }
                      setExpandedPaths(newExpanded);
                    }}
                    depth={0}
                  />
                ))
              ) : (
                <div className="text-center text-slate-500 py-8">
                  <svg className="w-10 h-10 mx-auto mb-2 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                  </svg>
                  <p className="text-sm">No location hierarchy available</p>
                </div>
              )}
            </div>
          </Card>

          {/* Selected Path Detail */}
          <Card className="p-6 lg:col-span-2">
            {selectedPath ? (
              (() => {
                // Find the selected node in the hierarchy
                const findNode = (nodes: PathNodeResponse[], path: string): PathNodeResponse | null => {
                  for (const node of nodes) {
                    if (node.full_path === path) return node;
                    if (node.children) {
                      const found = findNode(node.children, path);
                      if (found) return found;
                    }
                  }
                  return null;
                };
                const selectedNode = pathHierarchy?.hierarchy ? findNode(pathHierarchy.hierarchy, selectedPath) : null;

                return (
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-4">{selectedPath}</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                      <div className="bg-slate-800/50 rounded-lg p-3">
                        <p className="text-xs text-slate-500 uppercase">Security Score</p>
                        <p className="text-2xl font-bold text-amber-400">{selectedNode?.security_score?.toFixed(1) ?? 'N/A'}</p>
                      </div>
                      <div className="bg-slate-800/50 rounded-lg p-3">
                        <p className="text-xs text-slate-500 uppercase">Devices</p>
                        <p className="text-2xl font-bold text-white">{selectedNode?.device_count ?? 0}</p>
                      </div>
                      <div className="bg-slate-800/50 rounded-lg p-3">
                        <p className="text-xs text-slate-500 uppercase">At Risk</p>
                        <p className="text-2xl font-bold text-orange-400">{selectedNode?.at_risk_count ?? 0}</p>
                      </div>
                      <div className="bg-slate-800/50 rounded-lg p-3">
                        <p className="text-xs text-slate-500 uppercase">Critical</p>
                        <p className="text-2xl font-bold text-red-400">{selectedNode?.critical_count ?? 0}</p>
                      </div>
                    </div>
                    <h4 className="text-sm font-medium text-white mb-3">Compliance Status</h4>
                    <div className="space-y-3">
                      {selectedNode ? (
                        <>
                          <div className="flex items-center gap-3">
                            <span className="text-sm text-slate-400 w-24">Compliant</span>
                            <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
                              <div
                                className="h-full rounded-full bg-emerald-500"
                                style={{ width: `${selectedNode.device_count > 0 ? (selectedNode.compliant_count / selectedNode.device_count) * 100 : 0}%` }}
                              />
                            </div>
                            <span className="text-sm text-slate-400 w-12 text-right">{selectedNode.compliant_count}</span>
                          </div>
                          <div className="flex items-center gap-3">
                            <span className="text-sm text-slate-400 w-24">At Risk</span>
                            <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
                              <div
                                className="h-full rounded-full bg-orange-500"
                                style={{ width: `${selectedNode.device_count > 0 ? (selectedNode.at_risk_count / selectedNode.device_count) * 100 : 0}%` }}
                              />
                            </div>
                            <span className="text-sm text-slate-400 w-12 text-right">{selectedNode.at_risk_count}</span>
                          </div>
                          <div className="flex items-center gap-3">
                            <span className="text-sm text-slate-400 w-24">Critical</span>
                            <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
                              <div
                                className="h-full rounded-full bg-red-500"
                                style={{ width: `${selectedNode.device_count > 0 ? (selectedNode.critical_count / selectedNode.device_count) * 100 : 0}%` }}
                              />
                            </div>
                            <span className="text-sm text-slate-400 w-12 text-right">{selectedNode.critical_count}</span>
                          </div>
                        </>
                      ) : (
                        <p className="text-slate-500 text-sm">No data available for this location</p>
                      )}
                    </div>
                  </div>
                );
              })()
            ) : (
              <div className="flex items-center justify-center h-64 text-slate-500">
                <div className="text-center">
                  <svg className="w-12 h-12 mx-auto mb-3 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                  </svg>
                  <p>Select a location to view details</p>
                </div>
              </div>
            )}
          </Card>
        </motion.div>
      )}

      {/* Risk Clusters Tab */}
      {selectedTab === 'clusters' && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-4"
        >
          {/* Risk Cluster Cards - from API */}
          {riskClusters?.clusters && riskClusters.clusters.length > 0 ? (
            riskClusters.clusters.map((cluster) => (
              <Card key={cluster.violation_type} className="p-4">
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-4">
                    <div className={`p-2 rounded-lg ${
                      cluster.severity === 'critical' ? 'bg-red-500/20' :
                      cluster.severity === 'high' ? 'bg-orange-500/20' : 'bg-yellow-500/20'
                    }`}>
                      <svg className={`w-5 h-5 ${
                        cluster.severity === 'critical' ? 'text-red-400' :
                        cluster.severity === 'high' ? 'text-orange-400' : 'text-yellow-400'
                      }`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                      </svg>
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <h4 className="font-medium text-white">{cluster.violation_type}</h4>
                        <span className={`px-2 py-0.5 rounded text-xs font-medium capitalize ${
                          cluster.severity === 'critical' ? 'bg-red-500/20 text-red-400' :
                          cluster.severity === 'high' ? 'bg-orange-500/20 text-orange-400' : 'bg-yellow-500/20 text-yellow-400'
                        }`}>
                          {cluster.severity}
                        </span>
                      </div>
                      <p className="text-sm text-slate-400 mt-1">
                        {cluster.device_count} devices affected | Avg score: {cluster.avg_security_score}
                      </p>
                      <p className="text-xs text-slate-500 mt-1">
                        Locations: {cluster.common_paths.join(', ')}
                      </p>
                    </div>
                  </div>
                  <button className="px-3 py-1.5 bg-stellar-500/20 text-stellar-400 rounded text-sm hover:bg-stellar-500/30 transition-colors">
                    View Devices
                  </button>
                </div>
                <div className="mt-3 p-3 bg-slate-800/50 rounded-lg">
                  <p className="text-sm text-slate-300">
                    <span className="text-slate-500">Recommendation:</span> {cluster.recommendation}
                  </p>
                </div>
              </Card>
            ))
          ) : (
            <Card className="p-8">
              <div className="text-center text-slate-500">
                <svg className="w-12 h-12 mx-auto mb-3 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <p className="text-lg font-medium">No Risk Clusters Found</p>
                <p className="text-sm mt-1">No security violations detected in your fleet</p>
              </div>
            </Card>
          )}
        </motion.div>
      )}

      {/* Compare Tab */}
      {selectedTab === 'compare' && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
        >
          {/* Path Selection - get available paths from hierarchy */}
          <Card className="p-4">
            <h3 className="text-lg font-semibold text-white mb-4">Select Locations to Compare</h3>
            <div className="flex flex-wrap gap-2">
              {/* Extract all leaf paths from hierarchy */}
              {(() => {
                const getAllPaths = (nodes: PathNodeResponse[]): string[] => {
                  let paths: string[] = [];
                  for (const node of nodes) {
                    if (node.children && node.children.length > 0) {
                      paths = paths.concat(getAllPaths(node.children));
                    } else {
                      paths.push(node.full_path);
                    }
                  }
                  return paths;
                };
                const availablePaths = pathHierarchy?.hierarchy ? getAllPaths(pathHierarchy.hierarchy) : [];

                if (availablePaths.length === 0) {
                  return (
                    <p className="text-slate-500 text-sm">No locations available for comparison</p>
                  );
                }

                return availablePaths.map((path) => {
                  const isSelected = comparisonPaths.includes(path);
                  const pathName = path.split(' / ').pop() || path;
                  return (
                    <button
                      key={path}
                      onClick={() => {
                        if (isSelected) {
                          setComparisonPaths(comparisonPaths.filter(p => p !== path));
                        } else if (comparisonPaths.length < 4) {
                          setComparisonPaths([...comparisonPaths, path]);
                        }
                      }}
                      className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
                        isSelected
                          ? 'bg-stellar-500/30 text-stellar-400 border border-stellar-500/50'
                          : 'bg-slate-800/50 text-slate-400 hover:bg-slate-700/50'
                      }`}
                    >
                      {pathName}
                    </button>
                  );
                });
              })()}
            </div>
            <p className="text-xs text-slate-500 mt-2">Select 2-4 locations to compare</p>
          </Card>

          {/* Comparison Chart - use API data */}
          {comparisonPaths.length >= 2 && pathComparison && (
            <Card className="p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Security Score Comparison</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={pathComparison.paths.map((p) => ({
                      name: p.path_name,
                      score: p.security_score,
                      fleet: pathComparison.fleet_average_score,
                    }))}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="name" tick={{ fill: '#9CA3AF', fontSize: 11 }} />
                    <YAxis tick={{ fill: '#9CA3AF' }} domain={[0, 100]} />
                    <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }} />
                    <Legend />
                    <Bar dataKey="score" name="Security Score" fill="#F59E0B" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="fleet" name="Fleet Average" fill="#374151" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Insights - from API */}
              {pathComparison.insights && pathComparison.insights.length > 0 && (
                <div className="mt-4 p-4 bg-slate-800/50 rounded-lg">
                  <h4 className="text-sm font-medium text-white mb-2">Insights</h4>
                  <ul className="space-y-1 text-sm text-slate-400">
                    {pathComparison.insights.map((insight, i) => (
                      <li key={i}>{insight}</li>
                    ))}
                  </ul>
                </div>
              )}
            </Card>
          )}
        </motion.div>
      )}

      {/* Trends Tab */}
      {selectedTab === 'trends' && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <Card className="p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Security Score Trend (30 Days)</h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={trends}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis
                    dataKey="date"
                    tick={{ fill: '#9CA3AF', fontSize: 10 }}
                    tickFormatter={(val) => new Date(val).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                  />
                  <YAxis tick={{ fill: '#9CA3AF' }} domain={[50, 100]} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }}
                    labelFormatter={(val) => new Date(val).toLocaleDateString()}
                  />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="score"
                    name="Security Score"
                    stroke="#F59E0B"
                    fill="#F59E0B"
                    fillOpacity={0.2}
                  />
                  <Area
                    type="monotone"
                    dataKey="compliant_pct"
                    name="Compliance %"
                    stroke="#10B981"
                    fill="#10B981"
                    fillOpacity={0.2}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </motion.div>
      )}
    </div>
  );
}
