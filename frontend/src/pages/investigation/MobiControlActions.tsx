/**
 * MobiControl Action Buttons component.
 */

import {
  SyncIcon,
  MessageIcon,
  LocationIcon,
  LockIcon,
  RestartIcon,
  ClearCacheIcon,
} from './Icons';

interface MobiControlActionsProps {
  deviceId: number;
  onAction: (action: string) => void;
  loadingAction?: string | null;
}

export function MobiControlActions({ onAction, loadingAction }: MobiControlActionsProps) {
  const actions = [
    { id: 'sync', label: 'Sync Device', icon: <SyncIcon />, description: 'Force sync telemetry data' },
    { id: 'message', label: 'Send Message', icon: <MessageIcon />, description: 'Send notification to device' },
    { id: 'locate', label: 'Locate', icon: <LocationIcon />, description: 'Get current location' },
    { id: 'lock', label: 'Lock Device', icon: <LockIcon />, description: 'Remotely lock device' },
    { id: 'restart', label: 'Restart', icon: <RestartIcon />, description: 'Restart device remotely' },
    { id: 'clearCache', label: 'Clear Cache', icon: <ClearCacheIcon />, description: 'Clear app caches' },
  ];

  return (
    <div className="grid grid-cols-3 gap-2">
      {actions.map((action) => {
        const isLoading = loadingAction === action.id;
        const isDisabled = loadingAction !== null;

        return (
          <button
            key={action.id}
            onClick={() => onAction(action.id)}
            disabled={isDisabled}
            className={`p-3 rounded-xl bg-slate-800/50 border border-slate-700/50 transition-all group text-left
                       ${isDisabled ? 'opacity-50 cursor-not-allowed' : 'hover:border-amber-500/30 hover:bg-amber-500/5'}
                       ${isLoading ? 'border-amber-500/50' : ''}`}
          >
            <span className={`mb-2 block ${isLoading ? 'text-amber-400 animate-pulse' : 'text-slate-400 group-hover:text-amber-400'}`}>
              {isLoading ? (
                <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
                </svg>
              ) : action.icon}
            </span>
            <p className={`text-xs font-medium transition-colors ${isLoading ? 'text-amber-400' : 'text-white group-hover:text-amber-400'}`}>
              {isLoading ? 'Executing...' : action.label}
            </p>
            <p className="text-[10px] text-slate-500 mt-0.5">{action.description}</p>
          </button>
        );
      })}
    </div>
  );
}
