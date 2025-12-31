import { useState, useMemo } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from '../components/Card';
import { formatDistanceToNowStrict } from 'date-fns';
import { useLocationAttribute } from '../hooks/useLocationAttribute';

type ViewMode = 'grid' | 'list';

function Fleet() {
  const [viewMode, setViewMode] = useState<ViewMode>('list');
  const [anomalyFilter, setAnomalyFilter] = useState<'all' | 'anomaly' | 'clean'>('all');
  const [searchParams, setSearchParams] = useSearchParams();
  const [searchQuery, setSearchQuery] = useState('');
  const [attributeName] = useLocationAttribute();
  const navigate = useNavigate();

  const selectedGroup = searchParams.get('group');

  // Fetch Grouping Data (Heatmap/Store Stats)
  const { data: groupData } = useQuery({
    queryKey: ['fleet', 'groups', attributeName],
    queryFn: () => api.getLocationHeatmap(attributeName),
    enabled: !!attributeName,
  });

  // Fetch Devices (Filtered)
  const { data: devicesResponse, isLoading: isLoadingDevices } = useQuery({
    queryKey: ['devices', selectedGroup, searchQuery],
    queryFn: () => api.getDevices({
      search: searchQuery,
      group_by: selectedGroup ? attributeName : undefined,
      group_value: selectedGroup || undefined,
      page_size: 100
    }),
  });

  const devices = devicesResponse?.devices || [];
  const filteredDevices = useMemo(() => {
    if (anomalyFilter === 'all') {
      return devices;
    }
    return devices.filter((device) => {
      const count = Number(device.anomaly_count || 0);
      return anomalyFilter === 'anomaly' ? count > 0 : count === 0;
    });
  }, [devices, anomalyFilter]);

  // Filter group cards based on search query
  const filteredGroups = useMemo(() => {
    if (!groupData?.locations) return [];
    if (!searchQuery.trim()) return groupData.locations;

    const query = searchQuery.toLowerCase();
    return groupData.locations.filter((group) =>
      group.name.toLowerCase().includes(query) ||
      group.id.toLowerCase().includes(query) ||
      (group.region && group.region.toLowerCase().includes(query))
    );
  }, [groupData, searchQuery]);

  const handleGroupSelect = (groupId: string | null) => {
    if (groupId) {
      setSearchParams({ group: groupId });
    } else {
      setSearchParams({});
    }
  };

  const currentGroupInfo = useMemo(() => {
    return groupData?.locations.find(g => g.id === selectedGroup);
  }, [groupData, selectedGroup]);

  // Aggregate status counts from the fetched filtered list (computed but not currently displayed)
  // const statusCounts = useMemo(() => {
  //   const counts = { critical: 0, warning: 0, healthy: 0, total: 0 };
  //   devices.forEach((d) => {
  //     const status = d.status.toLowerCase() as keyof typeof counts;
  //     if (counts[status] !== undefined) {
  //       counts[status]++;
  //     } else {
  //       if (d.anomaly_count > 0) counts.warning++;
  //       else counts.healthy++;
  //     }
  //     counts.total++;
  //   });
  //   return counts;
  // }, [devices]);

  if (isLoadingDevices && !devicesResponse) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="relative w-16 h-16 mx-auto mb-4">
            <div className="absolute inset-0 rounded-full border-2 border-amber-500/20"></div>
            <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-amber-500 animate-spin"></div>
          </div>
          <p className="text-slate-400 font-mono text-sm">Loading fleet data...</p>
        </div>
      </div>
    );
  }

  return (
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      {/* Header */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h1 className="text-3xl font-bold text-white">Fleet Overview</h1>
            {selectedGroup && (
              <span className="px-3 py-1 bg-amber-500/10 border border-amber-500/20 text-amber-400 rounded-full text-xs font-mono">
                {attributeName}: {currentGroupInfo?.name || selectedGroup}
              </span>
            )}
          </div>
          <p className="text-slate-500">
            {selectedGroup
              ? `Monitoring ${devices.length} devices in ${currentGroupInfo?.name || selectedGroup}`
              : `Total ${devicesResponse?.total || 0} devices online`}
          </p>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-4">
          {/* Search */}
          <div className="relative">
            <input
              type="text"
              placeholder="Search devices..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="input-stellar w-56 pl-9"
            />
            <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>

          {/* Anomaly Filter */}
          <div className="flex items-center gap-1 p-1 bg-slate-800/50 rounded-lg border border-slate-700/50">
            <button
              onClick={() => setAnomalyFilter('all')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${anomalyFilter === 'all'
                ? 'bg-amber-500/20 text-amber-400'
                : 'text-slate-400 hover:text-white'
                }`}
            >
              All
            </button>
            <button
              onClick={() => setAnomalyFilter('anomaly')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${anomalyFilter === 'anomaly'
                ? 'bg-red-500/20 text-red-300'
                : 'text-slate-400 hover:text-white'
                }`}
            >
              Anomalies
            </button>
            <button
              onClick={() => setAnomalyFilter('clean')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${anomalyFilter === 'clean'
                ? 'bg-emerald-500/20 text-emerald-300'
                : 'text-slate-400 hover:text-white'
                }`}
            >
              Healthy
            </button>
          </div>

          {/* View Toggle */}
          <div className="flex items-center gap-1 p-1 bg-slate-800/50 rounded-lg border border-slate-700/50">
            <button
              onClick={() => setViewMode('grid')}
              className={`px-3 py-1.5 text-sm font-medium rounded-md transition-all ${viewMode === 'grid'
                ? 'bg-amber-500/20 text-amber-400'
                : 'text-slate-400 hover:text-white'
                }`}
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
              </svg>
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`px-3 py-1.5 text-sm font-medium rounded-md transition-all ${viewMode === 'list'
                ? 'bg-amber-500/20 text-amber-400'
                : 'text-slate-400 hover:text-white'
                }`}
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Grouping Overview (If Attribute Set) */}
      {attributeName && groupData && !selectedGroup && (
        <section>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-xl font-bold text-white">Groups ({attributeName})</h2>
            <span className="text-xs text-slate-500">
              {searchQuery ? `${filteredGroups.length} of ${groupData.locations.length} groups` : 'Select a group to see devices'}
            </span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {filteredGroups.slice(0, 8).map((group) => {
              const hasCritical = group.anomalyCount && group.anomalyCount > 0;
              return (
                <button
                  key={group.id}
                  onClick={() => handleGroupSelect(group.id)}
                  className={`text-left p-4 rounded-xl border transition-all group relative overflow-hidden ${hasCritical
                      ? 'border-red-500/30 bg-red-500/5 hover:border-red-500/60'
                      : 'border-slate-800 bg-slate-900/50 hover:border-amber-500/50 hover:bg-slate-800'
                    }`}
                >
                  <div className="relative z-10">
                    <div className="flex justify-between items-start mb-3">
                      <h3 className="font-semibold text-white group-hover:text-amber-400 transition-colors truncate pr-2 text-lg" title={group.name}>
                        {group.name}
                      </h3>
                      {hasCritical && (
                        <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-red-500/20 border border-red-500/30 text-red-400 text-xs font-bold animate-pulse">
                          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
                          {group.anomalyCount}
                        </div>
                      )}
                    </div>

                    <div className="flex items-end justify-between">
                      <div>
                        <p className="text-slate-500 text-xs mb-0.5">Total Devices</p>
                        <p className="font-mono text-xl text-slate-200">{group.deviceCount}</p>
                      </div>
                      <div className="text-right">
                        <p className="text-slate-500 text-xs mb-0.5">Active</p>
                        <p className={`font-mono text-xl ${hasCritical ? 'text-red-300' : 'text-emerald-400'}`}>
                          {group.activeDeviceCount}
                        </p>
                      </div>
                    </div>

                    {/* Mini Status Bar */}
                    <div className="mt-4 flex h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className="bg-emerald-500"
                        style={{ width: `${Math.max(0, 100 - ((group.anomalyCount || 0) / group.deviceCount * 100))}%` }}
                      />
                      <div
                        className="bg-red-500"
                        style={{ width: `${Math.min(100, ((group.anomalyCount || 0) / group.deviceCount * 100))}%` }}
                      />
                    </div>
                    <div className="mt-1 flex justify-between text-[10px] text-slate-500">
                      <span>Normal behavior</span>
                      {hasCritical && <span className="text-red-400 font-bold">{group.anomalyCount} Anomalies</span>}
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </section>
      )}

      {/* Breadcrumb / Back Button */}
      {selectedGroup && (
        <div className="flex items-center gap-2 text-sm text-slate-500 bg-slate-900/50 p-3 rounded-lg border border-slate-800/50">
          <button
            onClick={() => handleGroupSelect(null)}
            className="hover:text-amber-400 hover:underline flex items-center gap-1 font-medium text-white"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" /></svg>
            Back to All Groups
          </button>
          <span className="text-slate-600">/</span>
          <span className="text-amber-400 font-semibold">{currentGroupInfo?.name || selectedGroup}</span>
        </div>
      )}

      {/* Device List Header */}
      <div className="flex items-center justify-between pt-4 border-t border-slate-700/50">
        <h2 className="text-lg font-bold text-white">
          {selectedGroup ? `Devices in ${currentGroupInfo?.name}` : 'All Devices'}
        </h2>
        <span className="text-slate-500 text-sm font-mono">
          {filteredDevices.length} / {devices.length} results
        </span>
      </div>

      {/* Device Grid/List */}
      {viewMode === 'grid' ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <AnimatePresence mode="popLayout">
            {filteredDevices.map((device, index) => (
              <motion.div
                key={device.device_id}
                layout
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ delay: index * 0.02 }}
              >
                <DeviceCard device={device} />
              </motion.div>
            ))}
          </AnimatePresence>

          {filteredDevices.length === 0 && (
            <div className="col-span-4 p-12 text-center">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-slate-800 flex items-center justify-center">
                <svg className="w-8 h-8 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                    d="M12 18h.01M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z" />
                </svg>
              </div>
              <p className="text-slate-400 font-medium">No devices found</p>
            </div>
          )}
        </div>
      ) : (
        <Card noPadding>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                  <tr className="border-b border-slate-800/50">
                    <th className="table-header">Status</th>
                    <th className="table-header">Device ID</th>
                    <th className="table-header">Name</th>
                    <th className="table-header">Anomalies</th>
                    <th className="table-header">Model</th>
                    <th className="table-header">Last Seen</th>
                    {attributeName && <th className="table-header">{attributeName}</th>}
                    <th className="table-header text-right">Actions</th>
                  </tr>
                </thead>
              <tbody>
                <AnimatePresence>
                  {filteredDevices.map((device, index) => {
                    const anomalyCount = Number(device.anomaly_count || 0);
                    const hasAnomaly = anomalyCount > 0;
                    const statusValue = String(device.status || '').toLowerCase();
                    const statusDotClass = hasAnomaly
                      ? 'bg-red-500 animate-pulse shadow-[0_0_10px_rgba(239,68,68,0.7)]'
                      : statusValue.includes('critical')
                        ? 'bg-red-500 animate-pulse shadow-[0_0_8px_rgba(239,68,68,0.6)]'
                        : statusValue.includes('warning')
                          ? 'bg-orange-500 shadow-[0_0_8px_rgba(249,115,22,0.4)]'
                          : statusValue.includes('offline')
                            ? 'bg-slate-500'
                            : 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.4)]';

                    return (
                      <motion.tr
                        key={device.device_id}
                        className={`table-row ${hasAnomaly ? 'bg-red-500/5 hover:bg-red-500/10' : ''} cursor-pointer`}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 20 }}
                        transition={{ delay: index * 0.02 }}
                        onClick={() => navigate(`/devices/${device.device_id}`)}
                        onKeyDown={(event) => {
                          if (event.key === 'Enter' || event.key === ' ') {
                            event.preventDefault();
                            navigate(`/devices/${device.device_id}`);
                          }
                        }}
                        tabIndex={0}
                        role="link"
                      >
                        <td className="table-cell">
                          <div className={`w-3 h-3 rounded-full ${statusDotClass}`} />
                        </td>
                        <td className="table-cell">
                          <span className="font-mono text-slate-400">#{device.device_id}</span>
                        </td>
                        <td className="table-cell">
                          <span className="font-semibold text-white">{device.device_name}</span>
                        </td>
                        <td className="table-cell">
                          {hasAnomaly ? (
                            <span className="inline-flex items-center gap-1 rounded-full border border-red-500/40 bg-red-500/10 px-2 py-1 text-xs font-semibold text-red-300">
                              <span className="inline-block h-1.5 w-1.5 rounded-full bg-red-400" />
                              {anomalyCount} {anomalyCount === 1 ? 'Anomaly' : 'Anomalies'}
                            </span>
                          ) : (
                            <span className="text-xs text-slate-500">No anomalies</span>
                          )}
                        </td>
                        <td className="table-cell text-slate-400 text-sm">
                          {device.device_model}
                        </td>
                        <td className="table-cell text-slate-500 text-xs">
                          {device.last_seen ? `${formatDistanceToNowStrict(new Date(device.last_seen))} ago` : 'Never'}
                        </td>
                        {attributeName && (
                          <td className="table-cell text-amber-400/80 text-xs font-mono">
                            {device.custom_attributes?.[attributeName] || selectedGroup || '-'}
                          </td>
                        )}
                        <td className="table-cell text-right">
                          <Link
                            to={`/devices/${device.device_id}`}
                            className="text-xs font-medium text-amber-400 hover:text-amber-300"
                          >
                            View →
                          </Link>
                        </td>
                      </motion.tr>
                    );
                  })}
                </AnimatePresence>
                {filteredDevices.length === 0 && (
                  <tr className="table-row">
                    <td className="table-cell text-slate-500 text-sm" colSpan={attributeName ? 8 : 7}>
                      No devices match the current filters.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </motion.div>
  );
}

// Device Card Component
function DeviceCard({
  device,
}: {
  device: any; // Using any for simplicity as DeviceDetail might vary
}) {
  const statusValue = String(device.status || '').toLowerCase();
  const anomalyCount = Number(device.anomaly_count || 0);
  const hasAnomaly = anomalyCount > 0;
  const isHealthy = statusValue === 'active' || statusValue === 'healthy' || statusValue === 'charging';

  return (
    <Link
      to={`/devices/${device.device_id}`}
      className={`block p-4 rounded-xl border stellar-card ${hasAnomaly ? 'border-red-500/50 bg-red-500/5 shadow-[0_0_24px_rgba(239,68,68,0.08)]' : 'border-slate-700/50 bg-slate-800/30'} hover:border-amber-500/50 transition-all group`}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className={`w-2.5 h-2.5 rounded-full ${hasAnomaly ? 'bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.6)]' : isHealthy ? 'bg-emerald-500' : 'bg-orange-500'}`} />
          <span className="font-semibold text-white group-hover:text-amber-400 transition-colors">
            {device.device_name}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {hasAnomaly && (
            <span className="inline-flex items-center rounded-full border border-red-500/40 bg-red-500/10 px-2 py-0.5 text-[10px] font-semibold text-red-300">
              Anomaly {anomalyCount}
            </span>
          )}
          <span className="text-xs font-mono text-slate-500">#{device.device_id}</span>
        </div>
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-500">Model</span>
          <span className="text-slate-300">{device.device_model}</span>
        </div>
        {device.custom_attributes && Object.entries(device.custom_attributes).slice(0, 1).map(([k, v]) => (
          <div key={k} className="flex items-center justify-between text-xs">
            <span className="text-slate-500">{k}</span>
            <span className="text-amber-400 font-mono">{String(v)}</span>
          </div>
        ))}
        <div className="pt-2 border-t border-slate-700/50 text-[10px] text-slate-600">
          Last seen {device.last_seen ? formatDistanceToNowStrict(new Date(device.last_seen)) : 'never'} ago
        </div>
      </div>
    </Link>
  );
}

export default Fleet;
