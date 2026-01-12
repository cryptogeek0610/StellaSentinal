/**
 * ImpactedDevicesPanel - Slide-over panel showing devices affected by an insight
 *
 * Features:
 * - Toggle between grouping views (flat, location, model, AI pattern)
 * - Device cards with quick actions
 * - Search/filter within device list
 * - AI pattern analysis display
 */

import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import clsx from 'clsx';
import { SlideOverPanel } from '../unified/SlideOverPanel';
import { api } from '../../api/client';
import type { CustomerInsightResponse } from '../../api/client';
import type { ImpactedDevice, DeviceGrouping } from '../../types/anomaly';

type GroupingMode = 'flat' | 'by_location' | 'by_model' | 'by_pattern';

interface ImpactedDevicesPanelProps {
  isOpen: boolean;
  onClose: () => void;
  insight: CustomerInsightResponse;
}

// Icons
const ListIcon = () => (
  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
  </svg>
);

const MapPinIcon = () => (
  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
  </svg>
);

const DeviceIcon = () => (
  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 18h.01M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z" />
  </svg>
);

const SparklesIcon = () => (
  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
  </svg>
);

const ChevronDownIcon = () => (
  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
  </svg>
);

const SEVERITY_COLORS: Record<string, string> = {
  critical: 'bg-red-500/20 text-red-400 border-red-500/30',
  high: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
  medium: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  low: 'bg-slate-500/20 text-slate-400 border-slate-500/30',
};

const STATUS_COLORS: Record<string, string> = {
  online: 'bg-emerald-500',
  offline: 'bg-slate-500',
  unknown: 'bg-amber-500',
};

function DeviceCard({
  device,
  onViewDevice,
}: {
  device: ImpactedDevice;
  onViewDevice: (deviceId: number) => void;
}) {
  const severityClass = SEVERITY_COLORS[device.severity || 'low'] || SEVERITY_COLORS.low;
  const statusColor = STATUS_COLORS[device.status] || STATUS_COLORS.unknown;

  return (
    <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/50 hover:border-slate-600/50 transition-colors">
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          {/* Device name and status */}
          <div className="flex items-center gap-2">
            <span className={clsx('w-2 h-2 rounded-full', statusColor)} title={device.status} />
            <span className="font-medium text-white text-sm truncate">
              {device.device_name || `Device-${device.device_id}`}
            </span>
            {device.severity && (
              <span className={clsx('px-1.5 py-0.5 rounded text-xs font-medium border', severityClass)}>
                {device.severity}
              </span>
            )}
          </div>

          {/* Device details */}
          <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs text-slate-400">
            {device.device_model && (
              <span className="flex items-center gap-1">
                <DeviceIcon />
                {device.device_model}
              </span>
            )}
            {device.location && (
              <span className="flex items-center gap-1">
                <MapPinIcon />
                {device.location}
              </span>
            )}
            {device.os_version && <span>{device.os_version}</span>}
          </div>

          {/* Last seen */}
          {device.last_seen && (
            <div className="mt-1 text-xs text-slate-500">
              Last seen: {new Date(device.last_seen).toLocaleString()}
            </div>
          )}
        </div>

        {/* Actions */}
        <button
          onClick={() => onViewDevice(device.device_id)}
          className="px-2 py-1 text-xs font-medium text-amber-400 hover:text-amber-300 hover:bg-amber-500/10 rounded transition-colors"
        >
          View
        </button>
      </div>
    </div>
  );
}

function DeviceGroupSection({
  grouping,
  onViewDevice,
  defaultExpanded = true,
}: {
  grouping: DeviceGrouping;
  onViewDevice: (deviceId: number) => void;
  defaultExpanded?: boolean;
}) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  return (
    <div className="border border-slate-700/50 rounded-lg overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-3 bg-slate-800/30 hover:bg-slate-800/50 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="font-medium text-white">{grouping.group_label}</span>
          <span className="px-2 py-0.5 rounded-full bg-slate-700/50 text-xs text-slate-400">
            {grouping.device_count} device{grouping.device_count !== 1 ? 's' : ''}
          </span>
        </div>
        <span
          className={clsx(
            'text-slate-400 transition-transform',
            isExpanded ? 'rotate-180' : ''
          )}
        >
          <ChevronDownIcon />
        </span>
      </button>

      {isExpanded && (
        <div className="p-3 space-y-2 bg-slate-900/50">
          {grouping.devices.map((device) => (
            <DeviceCard
              key={device.device_id}
              device={device}
              onViewDevice={onViewDevice}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function AIAnalysisBanner({ analysis }: { analysis: string }) {
  return (
    <div className="p-3 rounded-lg bg-gradient-to-r from-purple-500/10 to-blue-500/10 border border-purple-500/20">
      <div className="flex items-start gap-2">
        <span className="text-purple-400 mt-0.5">
          <SparklesIcon />
        </span>
        <div>
          <h4 className="text-sm font-medium text-purple-300">AI Pattern Analysis</h4>
          <p className="text-sm text-slate-300 mt-1">{analysis}</p>
        </div>
      </div>
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="space-y-3">
      {[1, 2, 3, 4, 5].map((i) => (
        <div key={i} className="p-3 rounded-lg bg-slate-800/50 animate-pulse">
          <div className="flex items-start gap-3">
            <div className="w-2 h-2 rounded-full bg-slate-700 mt-1.5" />
            <div className="flex-1 space-y-2">
              <div className="h-4 w-1/3 bg-slate-700 rounded" />
              <div className="h-3 w-2/3 bg-slate-700 rounded" />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <div className="w-12 h-12 rounded-full bg-slate-800 flex items-center justify-center mb-3">
        <DeviceIcon />
      </div>
      <p className="text-slate-400">No devices found for this insight</p>
      <p className="text-slate-600 text-sm mt-1">Device data may not be available yet</p>
    </div>
  );
}

export function ImpactedDevicesPanel({
  isOpen,
  onClose,
  insight,
}: ImpactedDevicesPanelProps) {
  const navigate = useNavigate();
  const [groupingMode, setGroupingMode] = useState<GroupingMode>('flat');
  const [searchQuery, setSearchQuery] = useState('');

  // Fetch device data
  const { data, isLoading, error } = useQuery({
    queryKey: ['insight-devices', insight.insight_id, groupingMode === 'by_pattern'],
    queryFn: () =>
      api.getInsightDevices(insight.insight_id, {
        include_ai_grouping: groupingMode === 'by_pattern',
      }),
    enabled: isOpen,
    staleTime: 60_000,
  });

  // Filter devices by search query
  const filteredDevices = useMemo(() => {
    if (!data?.devices) return [];
    if (!searchQuery.trim()) return data.devices;

    const query = searchQuery.toLowerCase();
    return data.devices.filter(
      (d) =>
        (d.device_name?.toLowerCase().includes(query)) ||
        (d.device_model?.toLowerCase().includes(query)) ||
        (d.location?.toLowerCase().includes(query))
    );
  }, [data?.devices, searchQuery]);

  // Get appropriate groupings based on mode
  const groupings = useMemo(() => {
    if (!data?.groupings) return [];
    switch (groupingMode) {
      case 'by_location':
        return data.groupings.by_location;
      case 'by_model':
        return data.groupings.by_model;
      case 'by_pattern':
        return data.groupings.by_pattern;
      default:
        return [];
    }
  }, [data?.groupings, groupingMode]);

  const handleViewDevice = (deviceId: number) => {
    // Navigate to device detail page
    navigate(`/devices/${deviceId}`);
    onClose();
  };

  const groupingOptions: { key: GroupingMode; label: string; icon: () => JSX.Element }[] = [
    { key: 'flat', label: 'All', icon: ListIcon },
    { key: 'by_location', label: 'Location', icon: MapPinIcon },
    { key: 'by_model', label: 'Model', icon: DeviceIcon },
    { key: 'by_pattern', label: 'AI Patterns', icon: SparklesIcon },
  ];

  return (
    <SlideOverPanel
      isOpen={isOpen}
      onClose={onClose}
      title={`Impacted Devices (${data?.total_devices ?? insight.affected_device_count})`}
      subtitle={insight.headline}
      width="lg"
    >
      <div className="flex flex-col h-full">
        {/* Grouping Mode Tabs */}
        <div className="flex gap-1 mb-4 p-1 bg-slate-800/50 rounded-lg">
          {groupingOptions.map((opt) => {
            const Icon = opt.icon;
            return (
              <button
                key={opt.key}
                onClick={() => setGroupingMode(opt.key)}
                className={clsx(
                  'flex-1 px-3 py-2 rounded-md text-sm font-medium transition-colors',
                  'flex items-center justify-center gap-2',
                  groupingMode === opt.key
                    ? 'bg-amber-500/20 text-amber-400'
                    : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
                )}
              >
                <Icon />
                {opt.label}
              </button>
            );
          })}
        </div>

        {/* Search */}
        <div className="mb-4">
          <input
            type="text"
            placeholder="Search devices..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/50 outline-none transition-colors"
          />
        </div>

        {/* AI Pattern Analysis Banner */}
        {groupingMode === 'by_pattern' && data?.ai_pattern_analysis && (
          <div className="mb-4">
            <AIAnalysisBanner analysis={data.ai_pattern_analysis} />
          </div>
        )}

        {/* Content */}
        <div className="flex-1 overflow-y-auto">
          {isLoading ? (
            <LoadingSkeleton />
          ) : error ? (
            <div className="p-4 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400">
              Failed to load device data. Please try again.
            </div>
          ) : !data?.devices.length ? (
            <EmptyState />
          ) : groupingMode === 'flat' ? (
            <div className="space-y-2">
              {filteredDevices.map((device) => (
                <DeviceCard
                  key={device.device_id}
                  device={device}
                  onViewDevice={handleViewDevice}
                />
              ))}
              {filteredDevices.length === 0 && searchQuery && (
                <p className="text-center text-slate-400 py-4">
                  No devices match "{searchQuery}"
                </p>
              )}
            </div>
          ) : (
            <div className="space-y-3">
              {groupings.length > 0 ? (
                groupings.map((grouping, idx) => (
                  <DeviceGroupSection
                    key={grouping.group_key}
                    grouping={grouping}
                    onViewDevice={handleViewDevice}
                    defaultExpanded={idx < 3}
                  />
                ))
              ) : (
                <p className="text-center text-slate-400 py-4">
                  {groupingMode === 'by_pattern'
                    ? 'AI pattern analysis requires at least 3 devices'
                    : 'No groupings available'}
                </p>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="pt-4 mt-4 border-t border-slate-700/50">
          <div className="flex items-center justify-between text-sm text-slate-400">
            <span>
              {data?.total_devices ?? 0} device{(data?.total_devices ?? 0) !== 1 ? 's' : ''} affected
            </span>
            {data?.generated_at && (
              <span>
                Updated {new Date(data.generated_at).toLocaleTimeString()}
              </span>
            )}
          </div>
        </div>
      </div>
    </SlideOverPanel>
  );
}

export default ImpactedDevicesPanel;
