/**
 * Location Heatmap - Visual location performance overview
 * 
 * Color-coded grid showing utilization vs baseline
 * Configurable attribute name (e.g., "Store", "Warehouse", "Location")
 */

import { useState, useMemo } from 'react';

export interface LocationData {
  id: string;
  name: string;
  utilization: number;
  baseline: number;
  deviceCount: number;
  activeDeviceCount: number;
  region?: string;
  anomalyCount?: number;
}

interface LocationHeatmapProps {
  locations: LocationData[];
  onLocationClick: (location: LocationData) => void;
  attributeName?: string; // e.g., "Store", "Warehouse", "Location"
  className?: string;
  statusFilter?: 'all' | 'above' | 'below' | 'critical';
}

const getUtilizationColor = (utilization: number, baseline: number) => {
  const diff = utilization - baseline;
  if (diff < -20) return 'border-red-500 bg-red-500/10 hover:bg-red-500/20';
  if (diff < 0) return 'border-yellow-500 bg-yellow-500/10 hover:bg-yellow-500/20';
  return 'border-green-500 bg-green-500/10 hover:bg-green-500/20';
};

const getUtilizationHex = (level: number) => {
  if (level >= 80) return '#10b981'; // Green
  if (level >= 60) return '#3b82f6'; // Blue
  if (level >= 40) return '#fbbf24'; // Yellow
  return '#ef4444'; // Red
};

const LocationTooltip = ({ location }: { location: LocationData }) => (
  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 z-50 pointer-events-none animate-in fade-in zoom-in-95 duration-200">
    <div className="bg-gray-900/95 backdrop-blur-sm rounded-lg shadow-xl border border-gray-700 p-3 text-xs">
      <div className="font-bold text-white mb-2 pb-2 border-b border-gray-800">{location.name}</div>
      <div className="space-y-1.5 text-gray-400">
        {location.region && (
          <div className="flex justify-between items-center">
            <span>Region:</span>
            <span className="text-white font-medium">{location.region}</span>
          </div>
        )}
        <div className="flex justify-between items-center">
          <span>Utilization:</span>
          <span className={`font-bold ${location.utilization >= location.baseline ? 'text-green-400' : 'text-red-400'}`}>
            {location.utilization}%
          </span>
        </div>
        <div className="flex justify-between items-center">
          <span>Baseline:</span>
          <span className="text-white font-medium">{location.baseline}%</span>
        </div>
        <div className="flex justify-between items-center">
          <span>Devices:</span>
          <span className="text-white font-medium">{location.deviceCount}</span>
        </div>
        <div className="flex justify-between items-center pt-1 mt-1 border-t border-gray-800">
          <span>Deviation:</span>
          <span className={`font-bold ${(location.utilization - location.baseline) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {(location.utilization - location.baseline) > 0 ? '+' : ''}{location.utilization - location.baseline}%
          </span>
        </div>
      </div>
      {/* Arrow */}
      <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-gray-900/95" />
    </div>
  </div>
);

export const LocationHeatmap = ({ 
  locations, 
  onLocationClick, 
  attributeName = 'Location',
  className = '', 
  statusFilter: propStatusFilter 
}: LocationHeatmapProps) => {
  const [hoveredLocation, setHoveredLocation] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<'name' | 'utilization' | 'deviation'>('name');
  const [groupBy, setGroupBy] = useState<'none' | 'region' | 'status'>('none');
  const [internalStatusFilter, setInternalStatusFilter] = useState<'all' | 'above' | 'below' | 'critical'>('all');

  // Use prop filter if available, otherwise internal
  const statusFilter = propStatusFilter || internalStatusFilter;

  // Helper to set filter (only work if not controlled)
  const setStatusFilter = (filter: 'all' | 'above' | 'below' | 'critical') => {
    if (!propStatusFilter) {
      setInternalStatusFilter(filter);
    }
  };

  // Filter locations by status
  const filteredLocations = useMemo(() => {
    if (statusFilter === 'all') return locations;

    return locations.filter(location => {
      const deviation = location.utilization - location.baseline;
      switch (statusFilter) {
        case 'above':
          return location.utilization >= location.baseline;
        case 'below':
          return deviation < 0 && deviation >= -20;
        case 'critical':
          return deviation < -20;
        default:
          return true;
      }
    });
  }, [locations, statusFilter]);

  // Group locations logic
  const groupedLocations = useMemo(() => {
    // Sort first
    const sorted = [...filteredLocations].sort((a, b) => {
      switch (sortBy) {
        case 'utilization': return b.utilization - a.utilization;
        case 'deviation': return (a.utilization - a.baseline) - (b.utilization - b.baseline);
        case 'name': default: return a.name.localeCompare(b.name);
      }
    });

    if (groupBy === 'none') return { [`All ${attributeName}s`]: sorted };

    return sorted.reduce((groups, location) => {
      let key = 'Other';
      if (groupBy === 'region') key = location.region || 'Unknown';
      if (groupBy === 'status') {
        const dev = location.utilization - location.baseline;
        if (dev < -20) key = 'Critical Under-utilization';
        else if (dev < 0) key = 'Below Baseline';
        else if (dev < 20) key = 'Above Baseline';
        else key = 'High Utilization';
      }

      if (!groups[key]) groups[key] = [];
      groups[key].push(location);
      return groups;
    }, {} as Record<string, LocationData[]>);
  }, [filteredLocations, sortBy, groupBy, attributeName]);

  const sortedLocations = useMemo(() => {
    return groupedLocations[`All ${attributeName}s`] || Object.values(groupedLocations).flat();
  }, [groupedLocations, attributeName]);

  // Calculate stats (always based on all locations, not filtered)
  const stats = useMemo(() => {
    const total = locations.length;
    const aboveBaseline = locations.filter(l => l.utilization >= l.baseline).length;
    const belowBaseline = locations.filter(l => {
      const dev = l.utilization - l.baseline;
      return dev < 0 && dev >= -20;
    }).length;
    const critical = locations.filter(l => (l.utilization - l.baseline) < -20).length;
    const avgUtilization = Math.round(locations.reduce((sum, l) => sum + l.utilization, 0) / Math.max(1, total));

    return { total, aboveBaseline, belowBaseline, critical, avgUtilization };
  }, [locations]);

  return (
    <div className={`glass-panel rounded-xl p-6 border border-slate-700/50 ${className}`}>
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between mb-6 gap-4">
        <div>
          <h3 className="text-lg font-bold text-white">{attributeName} Performance Map</h3>
          <p className="text-xs text-slate-500 mt-1">
            {statusFilter === 'all'
              ? `${stats.total} ${attributeName.toLowerCase()}s • ${stats.avgUtilization}% avg utilization`
              : `${filteredLocations.length} of ${stats.total} ${attributeName.toLowerCase()}s • ${statusFilter === 'above' ? 'Above Baseline' : statusFilter === 'below' ? 'Below Baseline' : 'Critical'}`
            }
          </p>
        </div>

        <div className="flex items-center gap-3">
          {/* Group By Dropdown */}
          <div className="flex items-center gap-2 bg-slate-800/50 p-1 rounded-lg border border-slate-700">
            <span className="text-xs text-slate-400 pl-2">Group by:</span>
            <select
              value={groupBy}
              onChange={(e) => setGroupBy(e.target.value as any)}
              className="bg-transparent text-xs text-white font-medium focus:outline-none cursor-pointer"
            >
              <option value="none">None</option>
              <option value="region">Region</option>
              <option value="status">Status</option>
            </select>
          </div>

          {/* Sort Controls */}
          <div className="flex bg-slate-800 rounded-lg p-1 border border-slate-700">
            {(['name', 'utilization', 'deviation'] as const).map((sort) => (
              <button
                key={sort}
                onClick={() => setSortBy(sort)}
                className={`px-3 py-1 rounded-md text-xs font-medium transition-all ${sortBy === sort
                  ? 'bg-slate-700 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
                  }`}
              >
                {sort.charAt(0).toUpperCase() + sort.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Stats Row - Clickable Filters */}
      <div className="grid grid-cols-4 gap-3 mb-6">
        <button
          onClick={() => setStatusFilter(statusFilter === 'above' ? 'all' : 'above')}
          className={`p-3 rounded-lg border transition-all cursor-pointer hover:scale-105 ${statusFilter === 'above'
            ? 'bg-green-900/40 border-green-500/60 shadow-lg shadow-green-500/20'
            : 'bg-green-900/20 border-green-500/30 hover:bg-green-900/30'
            }`}
        >
          <div className="text-xl font-bold text-green-400">{stats.aboveBaseline}</div>
          <div className="text-xs text-slate-400">Above Baseline</div>
        </button>
        <button
          onClick={() => setStatusFilter(statusFilter === 'below' ? 'all' : 'below')}
          className={`p-3 rounded-lg border transition-all cursor-pointer hover:scale-105 ${statusFilter === 'below'
            ? 'bg-yellow-900/40 border-yellow-500/60 shadow-lg shadow-yellow-500/20'
            : 'bg-yellow-900/20 border-yellow-500/30 hover:bg-yellow-900/30'
            }`}
        >
          <div className="text-xl font-bold text-yellow-400">{stats.belowBaseline}</div>
          <div className="text-xs text-slate-400">Below Baseline</div>
        </button>
        <button
          onClick={() => setStatusFilter(statusFilter === 'critical' ? 'all' : 'critical')}
          className={`p-3 rounded-lg border transition-all cursor-pointer hover:scale-105 ${statusFilter === 'critical'
            ? 'bg-red-900/40 border-red-500/60 shadow-lg shadow-red-500/20'
            : 'bg-red-900/20 border-red-500/30 hover:bg-red-900/30'
            }`}
        >
          <div className="text-xl font-bold text-red-400">{stats.critical}</div>
          <div className="text-xs text-slate-400">Critical</div>
        </button>
        <div className="p-3 rounded-lg bg-blue-900/20 border border-blue-500/30">
          <div className="text-xl font-bold text-blue-400">{stats.avgUtilization}%</div>
          <div className="text-xs text-slate-400">Avg Utilization</div>
        </div>
      </div>

      {/* Heatmap Grid */}
      <div className="grid grid-cols-8 gap-2">
        {sortedLocations.map((location) => (
          <div
            key={location.id}
            className="relative"
            onMouseEnter={() => setHoveredLocation(location.id)}
            onMouseLeave={() => setHoveredLocation(null)}
          >
            <button
              onClick={() => onLocationClick(location)}
              className={`
                w-full aspect-square rounded-xl p-2 flex flex-col items-center justify-center
                transition-all duration-200 transform hover:scale-105 hover:z-10
                border ${getUtilizationColor(location.utilization, location.baseline)}
              `}
            >
              <span className="text-xs font-bold text-white truncate w-full text-center">
                {location.name}
              </span>
              <span className="text-[10px] text-white/70">
                {location.utilization}%
              </span>

              {/* Warning indicator for critical deviation */}
              {(location.utilization - location.baseline) < -15 && (
                <span className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full animate-pulse" />
              )}
            </button>

            {/* Tooltip */}
            {hoveredLocation === location.id && (
              <LocationTooltip location={location} />
            )}
          </div>
        ))}
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-1 mt-6">
        <span className="text-xs text-slate-500 mr-2">Utilization:</span>
        <div className="flex rounded overflow-hidden">
          {[20, 40, 60, 80, 100].map((level) => (
            <div
              key={level}
              className="w-8 h-4"
              style={{ backgroundColor: getUtilizationHex(level) }}
              title={`${level}%`}
            />
          ))}
        </div>
        <div className="flex items-center gap-4 ml-4 text-xs text-slate-500">
          <span>▼ Below Baseline</span>
          <span>▲ Above Baseline</span>
        </div>
      </div>
    </div>
  );
};

// Compact version for sidebar or smaller areas
export const LocationHeatmapCompact = ({ 
  locations, 
  onLocationClick, 
  attributeName = 'Location' 
}: Omit<LocationHeatmapProps, 'statusFilter'>) => {
  return (
    <div className="glass-panel rounded-xl p-4 border border-slate-700/50">
      <h4 className="text-sm font-bold text-white mb-3">{attributeName} Health</h4>
      <div className="grid grid-cols-6 gap-1">
        {locations.slice(0, 24).map((location) => (
          <button
            key={location.id}
            onClick={() => onLocationClick(location)}
            title={`${location.name}: ${location.utilization}%`}
            className={`
              aspect-square rounded transition-transform hover:scale-125
              ${getUtilizationColor(location.utilization, location.baseline)}
            `}
          />
        ))}
      </div>
      {locations.length > 24 && (
        <p className="text-xs text-slate-500 mt-2 text-center">
          +{locations.length - 24} more {attributeName.toLowerCase()}s
        </p>
      )}
    </div>
  );
};

