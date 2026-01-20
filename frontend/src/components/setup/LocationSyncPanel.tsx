/**
 * Location sync panel component for Setup page.
 * Allows syncing location data from connected systems.
 */

import { useMutation, useQuery } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import { api } from '../../api/client';
import { LoadingSpinner } from './FormComponents';

export function LocationSyncPanel() {
  const syncMutation = useMutation({
    mutationFn: () => api.syncLocations(),
    onSuccess: (data) => {
      toast.success(data.message || 'Locations synced successfully');
    },
    onError: (error: Error) => {
      toast.error(error.message || 'Location sync failed');
    },
  });

  const { data: syncStats, isLoading: isLoadingStats } = useQuery({
    queryKey: ['setup', 'location-sync-stats'],
    queryFn: () => api.getLocationSyncStats(),
    refetchInterval: 60000,
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h3 className="text-lg font-semibold text-white">Location Sync</h3>
          <p className="text-sm text-slate-500">
            Sync location data from connected systems to keep dashboards aligned.
          </p>
        </div>
        <button
          type="button"
          onClick={() => syncMutation.mutate()}
          disabled={syncMutation.isPending}
          className="btn-stellar flex items-center gap-2"
        >
          {syncMutation.isPending ? (
            <>
              <LoadingSpinner />
              Syncing...
            </>
          ) : (
            <>
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Sync Now
            </>
          )}
        </button>
      </div>

      <div className="bg-slate-800/40 border border-slate-700/50 rounded-xl p-4 text-sm text-slate-300">
        {isLoadingStats ? (
          <div className="flex items-center gap-2 text-slate-400">
            <LoadingSpinner />
            Loading sync status...
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <div className="text-xs text-slate-500">Last Sync</div>
              <div className="mt-1">
                {syncStats?.last_sync ? new Date(syncStats.last_sync as string).toLocaleString() : 'Never'}
              </div>
            </div>
            <div>
              <div className="text-xs text-slate-500">Locations</div>
              <div className="mt-1">{String(syncStats?.locations_count ?? '—')}</div>
            </div>
            <div>
              <div className="text-xs text-slate-500">Devices Mapped</div>
              <div className="mt-1">{String(syncStats?.devices_mapped ?? '—')}</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
