/**
 * Location Center - Geographic Analytics Dashboard
 *
 * Comprehensive location-based analytics:
 * - Location heatmap with anomaly distribution
 * - Device counts and health by location
 * - Anomalies grouped by store/region
 * - Location health metrics and trends
 */

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { api } from '../api/client';
import { Card } from '../components/Card';
import { useLocationAttribute } from '../hooks/useLocationAttribute';

// Location card with health indicator
function LocationCard({
  location,
  isSelected,
  onClick,
}: {
  location: {
    id: string;
    name: string;
    utilization: number;
    baseline: number;
    deviceCount: number;
    activeDeviceCount: number;
    region?: string;
    anomalyCount?: number;
  };
  isSelected: boolean;
  onClick: () => void;
}) {
  const anomalyCount = location.anomalyCount ?? 0;
  const healthStatus =
    anomalyCount > 3
      ? 'critical'
      : anomalyCount > 0
      ? 'warning'
      : 'healthy';

  const utilizationStatus =
    location.utilization < location.baseline * 0.7
      ? 'low'
      : location.utilization > location.baseline * 1.1
      ? 'high'
      : 'normal';

  return (
    <motion.button
      onClick={onClick}
      className={`w-full text-left p-4 rounded-xl border transition-all ${
        isSelected
          ? 'bg-stellar-500/20 border-stellar-500/50'
          : 'bg-slate-800/40 border-slate-700/50 hover:bg-slate-800/60 hover:border-slate-600'
      }`}
      whileHover={{ scale: 1.01 }}
      whileTap={{ scale: 0.99 }}
    >
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-200">{location.name}</h3>
          <p className="text-xs text-slate-500">{location.region || 'Unknown Region'}</p>
        </div>
        <div
          className={`w-3 h-3 rounded-full ${
            healthStatus === 'healthy'
              ? 'bg-emerald-400 shadow-[0_0_8px_rgba(16,185,129,0.6)]'
              : healthStatus === 'warning'
              ? 'bg-orange-400 shadow-[0_0_8px_rgba(251,146,60,0.6)]'
              : 'bg-red-400 shadow-[0_0_8px_rgba(248,113,113,0.6)] animate-pulse'
          }`}
        />
      </div>

      <div className="grid grid-cols-3 gap-2 text-center">
        <div className="p-2 bg-slate-900/50 rounded-lg">
          <p className="text-lg font-bold text-white">{location.deviceCount}</p>
          <p className="text-[10px] text-slate-500">Devices</p>
        </div>
        <div className="p-2 bg-slate-900/50 rounded-lg">
          <p className="text-lg font-bold text-emerald-400">{location.activeDeviceCount}</p>
          <p className="text-[10px] text-slate-500">Active</p>
        </div>
        <div className="p-2 bg-slate-900/50 rounded-lg">
          <p
            className={`text-lg font-bold ${
              anomalyCount > 0 ? 'text-orange-400' : 'text-slate-400'
            }`}
          >
            {anomalyCount}
          </p>
          <p className="text-[10px] text-slate-500">Anomalies</p>
        </div>
      </div>

      <div className="mt-3">
        <div className="flex items-center justify-between mb-1">
          <span className="text-[10px] text-slate-500">Utilization</span>
          <span
            className={`text-xs font-medium ${
              utilizationStatus === 'low'
                ? 'text-blue-400'
                : utilizationStatus === 'high'
                ? 'text-orange-400'
                : 'text-emerald-400'
            }`}
          >
            {location.utilization.toFixed(1)}%
          </span>
        </div>
        <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
          <motion.div
            className={`h-full rounded-full ${
              utilizationStatus === 'low'
                ? 'bg-blue-500'
                : utilizationStatus === 'high'
                ? 'bg-orange-500'
                : 'bg-emerald-500'
            }`}
            initial={{ width: 0 }}
            animate={{ width: `${Math.min(100, location.utilization)}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
      </div>
    </motion.button>
  );
}

// Region summary card
function RegionSummary({
  region,
  locations,
}: {
  region: string;
  locations: Array<{
    id: string;
    name: string;
    deviceCount: number;
    activeDeviceCount: number;
    anomalyCount?: number;
  }>;
}) {
  const totalDevices = locations.reduce((sum, l) => sum + l.deviceCount, 0);
  const totalActive = locations.reduce((sum, l) => sum + l.activeDeviceCount, 0);
  const totalAnomalies = locations.reduce((sum, l) => sum + (l.anomalyCount ?? 0), 0);
  // Calculate healthy devices: active devices minus those with anomalies
  const healthyDevices = Math.max(0, totalActive - totalAnomalies);

  return (
    <div className="p-4 bg-slate-800/40 rounded-xl border border-slate-700/50">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-slate-200">{region}</h3>
        <span className="px-2 py-0.5 text-xs font-medium bg-slate-700 text-slate-300 rounded-full">
          {locations.length} locations
        </span>
      </div>

      <div className="grid grid-cols-4 gap-3">
        <div className="text-center">
          <p className="text-xl font-bold text-white">{totalDevices}</p>
          <p className="text-[10px] text-slate-500">Total Devices</p>
        </div>
        <div className="text-center">
          <p className="text-xl font-bold text-emerald-400">{totalActive}</p>
          <p className="text-[10px] text-slate-500">Active</p>
        </div>
        <div className="text-center">
          <p
            className={`text-xl font-bold ${
              totalAnomalies > 0 ? 'text-orange-400' : 'text-slate-400'
            }`}
          >
            {totalAnomalies}
          </p>
          <p className="text-[10px] text-slate-500">Anomalies</p>
        </div>
        <div className="text-center">
          <p className="text-xl font-bold text-stellar-400">{healthyDevices}</p>
          <p className="text-[10px] text-slate-500">Healthy</p>
        </div>
      </div>
    </div>
  );
}

// Location detail panel
function LocationDetailPanel({
  location,
  onClose,
}: {
  location: {
    id: string;
    name: string;
    utilization: number;
    baseline: number;
    deviceCount: number;
    activeDeviceCount: number;
    region?: string;
    anomalyCount?: number;
  };
  onClose: () => void;
}) {
  const anomalyCount = location.anomalyCount ?? 0;
  const offlineCount = location.deviceCount - location.activeDeviceCount;
  const offlinePercent = location.deviceCount > 0
    ? ((offlineCount / location.deviceCount) * 100).toFixed(1)
    : '0';

  return (
    <Card className="p-6">
      <div className="flex items-start justify-between mb-6">
        <div>
          <h2 className="text-lg font-semibold text-slate-200">{location.name}</h2>
          <p className="text-sm text-slate-500">{location.region || 'Unknown'} Region</p>
        </div>
        <button
          onClick={onClose}
          className="p-2 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-colors"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="p-4 bg-slate-800/50 rounded-xl">
          <p className="text-3xl font-bold text-white">{location.deviceCount}</p>
          <p className="text-sm text-slate-500">Total Devices</p>
        </div>
        <div className="p-4 bg-emerald-500/10 border border-emerald-500/20 rounded-xl">
          <p className="text-3xl font-bold text-emerald-400">{location.activeDeviceCount}</p>
          <p className="text-sm text-slate-500">Active Devices</p>
        </div>
        <div className="p-4 bg-slate-800/50 rounded-xl">
          <p className="text-3xl font-bold text-slate-300">{offlineCount}</p>
          <p className="text-sm text-slate-500">Offline ({offlinePercent}%)</p>
        </div>
        <div
          className={`p-4 rounded-xl ${
            anomalyCount > 0
              ? 'bg-orange-500/10 border border-orange-500/20'
              : 'bg-slate-800/50'
          }`}
        >
          <p
            className={`text-3xl font-bold ${
              anomalyCount > 0 ? 'text-orange-400' : 'text-slate-400'
            }`}
          >
            {anomalyCount}
          </p>
          <p className="text-sm text-slate-500">Open Anomalies</p>
        </div>
      </div>

      {/* Utilization */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-slate-400">Device Utilization</span>
          <span className="text-sm font-medium text-white">{location.utilization.toFixed(1)}%</span>
        </div>
        <div className="h-3 bg-slate-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-stellar-500 to-stellar-400 rounded-full"
            style={{ width: `${Math.min(100, location.utilization)}%` }}
          />
        </div>
        <div className="flex justify-between mt-1">
          <span className="text-xs text-slate-600">0%</span>
          <span className="text-xs text-slate-500">Baseline: {location.baseline}%</span>
          <span className="text-xs text-slate-600">100%</span>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="flex gap-3">
        <Link
          to={`/fleet?location=${encodeURIComponent(location.name)}`}
          className="flex-1 px-4 py-2.5 bg-stellar-600 hover:bg-stellar-500 text-white rounded-lg text-sm font-medium text-center transition-colors"
        >
          View Devices
        </Link>
        {anomalyCount > 0 && (
          <Link
            to={`/investigations?location=${encodeURIComponent(location.name)}`}
            className="flex-1 px-4 py-2.5 bg-orange-600 hover:bg-orange-500 text-white rounded-lg text-sm font-medium text-center transition-colors"
          >
            View Anomalies
          </Link>
        )}
      </div>
    </Card>
  );
}

// Main component
export default function LocationCenter() {
  // Use shared location attribute from System settings
  const [selectedAttribute, setSelectedAttribute] = useLocationAttribute();
  const [selectedLocation, setSelectedLocation] = useState<string | null>(null);

  // Fetch location heatmap data
  const { data: heatmapData, isLoading: heatmapLoading } = useQuery({
    queryKey: ['location-heatmap', selectedAttribute],
    queryFn: () => api.getLocationHeatmap(selectedAttribute),
  });

  // Fetch custom attributes for filter
  const { data: attributesData } = useQuery({
    queryKey: ['custom-attributes'],
    queryFn: () => api.getCustomAttributes(),
  });

  // Group locations by region
  const locationsByRegion = heatmapData?.locations.reduce(
    (acc, loc) => {
      const region = loc.region || 'Other';
      if (!acc[region]) acc[region] = [];
      acc[region].push(loc);
      return acc;
    },
    {} as Record<string, typeof heatmapData.locations>
  ) || {};

  const selectedLocationData = heatmapData?.locations.find(
    (l) => l.id === selectedLocation
  );

  // Calculate totals
  const totalDevices = heatmapData?.totalDevices || 0;
  const totalLocations = heatmapData?.totalLocations || 0;
  const totalAnomalies = heatmapData?.locations.reduce((sum, l) => sum + (l.anomalyCount ?? 0), 0) || 0;
  // Calculate healthy devices: total devices minus devices with anomalies
  // Note: anomalyCount represents the number of anomalies at a location, but we approximate
  // healthy devices as (total active devices) since devices with anomalies are counted in totalAnomalies
  const totalActiveDevices = heatmapData?.locations.reduce((sum, l) => sum + l.activeDeviceCount, 0) || 0;
  const healthyDevices = Math.max(0, totalActiveDevices - totalAnomalies);

  if (heatmapLoading) {
    return (
      <div className="space-y-6">
        <div className="animate-pulse">
          <div className="h-8 w-48 bg-slate-700 rounded mb-2" />
          <div className="h-4 w-64 bg-slate-700 rounded" />
        </div>
        <div className="grid grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-24 bg-slate-800 rounded-xl animate-pulse" />
          ))}
        </div>
        <div className="grid grid-cols-3 gap-6">
          <div className="col-span-2 h-96 bg-slate-800 rounded-xl animate-pulse" />
          <div className="h-96 bg-slate-800 rounded-xl animate-pulse" />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Location Center</h1>
          <p className="text-slate-400 mt-1">
            Geographic analytics and device distribution
          </p>
        </div>

        {/* Attribute Selector */}
        <div className="flex items-center gap-3">
          <span className="text-sm text-slate-500">Group by:</span>
          <select
            value={selectedAttribute}
            onChange={(e) => {
              setSelectedAttribute(e.target.value);
              setSelectedLocation(null);
            }}
            className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-slate-200 focus:border-stellar-500 focus:outline-none"
          >
            {attributesData?.custom_attributes.map((attr) => (
              <option key={attr} value={attr}>
                {attr}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Card className="p-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-stellar-500/20 flex items-center justify-center flex-shrink-0">
              <svg className="w-5 h-5 text-stellar-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            </div>
            <div>
              <p className="text-xl font-bold text-white">{totalLocations}</p>
              <p className="text-xs text-slate-500">Locations</p>
            </div>
          </div>
        </Card>

        <Card className="p-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-indigo-500/20 flex items-center justify-center flex-shrink-0">
              <svg className="w-5 h-5 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 18h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
              </svg>
            </div>
            <div>
              <p className="text-xl font-bold text-white">{totalDevices}</p>
              <p className="text-xs text-slate-500">Total Devices</p>
            </div>
          </div>
        </Card>

        <Card className="p-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-emerald-500/20 flex items-center justify-center flex-shrink-0">
              <svg className="w-5 h-5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div>
              <p className="text-xl font-bold text-emerald-400">{healthyDevices}</p>
              <p className="text-xs text-slate-500">Healthy Devices</p>
            </div>
          </div>
        </Card>

        <Card className="p-3">
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${
              totalAnomalies > 0 ? 'bg-orange-500/20' : 'bg-slate-700/50'
            }`}>
              <svg className={`w-5 h-5 ${totalAnomalies > 0 ? 'text-orange-400' : 'text-slate-500'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
            <div>
              <p className={`text-xl font-bold ${totalAnomalies > 0 ? 'text-orange-400' : 'text-slate-400'}`}>
                {totalAnomalies}
              </p>
              <p className="text-xs text-slate-500">Anomalies</p>
            </div>
          </div>
        </Card>
      </div>

      {/* Main Content - Responsive Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Region Summaries */}
        <Card className="p-4 lg:col-span-2">
          <h2 className="text-sm font-semibold text-slate-200 mb-3">Regional Overview</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {Object.entries(locationsByRegion).map(([region, locations]) => (
              <RegionSummary key={region} region={region} locations={locations} />
            ))}
          </div>
        </Card>

        {/* Anomaly Concentration */}
        <Card className="p-4">
          <h2 className="text-sm font-semibold text-slate-200 mb-3">Anomaly Concentration</h2>
          {heatmapData?.locations
            .filter((l) => (l.anomalyCount ?? 0) > 0)
            .sort((a, b) => (b.anomalyCount ?? 0) - (a.anomalyCount ?? 0))
            .slice(0, 5).length === 0 ? (
            <div className="text-center py-6">
              <svg
                className="w-10 h-10 text-emerald-500/50 mx-auto mb-2"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <p className="text-sm text-slate-400">No anomaly concentrations</p>
              <p className="text-xs text-slate-500">All locations are healthy</p>
            </div>
          ) : (
            <div className="space-y-2">
              {heatmapData?.locations
                .filter((l) => (l.anomalyCount ?? 0) > 0)
                .sort((a, b) => (b.anomalyCount ?? 0) - (a.anomalyCount ?? 0))
                .slice(0, 5)
                .map((location, index) => (
                  <button
                    key={location.id}
                    onClick={() => setSelectedLocation(location.id)}
                    className="w-full flex items-center gap-2 p-2 bg-slate-800/40 hover:bg-slate-800/60 rounded-lg transition-colors text-left"
                  >
                    <span
                      className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold ${
                        index === 0
                          ? 'bg-red-500/20 text-red-400'
                          : index === 1
                          ? 'bg-orange-500/20 text-orange-400'
                          : 'bg-yellow-500/20 text-yellow-400'
                      }`}
                    >
                      {index + 1}
                    </span>
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-medium text-slate-200 truncate">
                        {location.name}
                      </p>
                      <p className="text-[10px] text-slate-500">{location.region || 'Unknown'}</p>
                    </div>
                    <span className="px-1.5 py-0.5 bg-orange-500/20 text-orange-400 rounded text-[10px] font-medium">
                      {location.anomalyCount ?? 0}
                    </span>
                  </button>
                ))}
            </div>
          )}
        </Card>

        {/* All Locations */}
        <Card className="p-4 lg:col-span-2">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-slate-200">All Locations</h2>
            {selectedLocation && (
              <button
                onClick={() => setSelectedLocation(null)}
                className="text-xs text-slate-400 hover:text-slate-300"
              >
                Clear selection
              </button>
            )}
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3 max-h-[400px] overflow-y-auto pr-1">
            {heatmapData?.locations.map((location) => (
              <LocationCard
                key={location.id}
                location={location}
                isSelected={selectedLocation === location.id}
                onClick={() => setSelectedLocation(
                  selectedLocation === location.id ? null : location.id
                )}
              />
            ))}
          </div>
        </Card>

        {/* Detail Panel */}
        <div>
          {selectedLocationData ? (
            <LocationDetailPanel
              location={selectedLocationData}
              onClose={() => setSelectedLocation(null)}
            />
          ) : (
            <Card className="p-4">
              <div className="text-center py-8">
                <svg
                  className="w-12 h-12 text-slate-600 mx-auto mb-3"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1}
                    d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"
                  />
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1}
                    d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"
                  />
                </svg>
                <h3 className="text-sm font-medium text-slate-400 mb-1">Select a Location</h3>
                <p className="text-xs text-slate-500">
                  Click on any location card to view details
                </p>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
