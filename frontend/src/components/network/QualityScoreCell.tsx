/**
 * Quality Score Cell Component
 *
 * Displays the WiFi network quality score with a progress bar and
 * an interactive tooltip showing the score breakdown.
 */

import { useState } from 'react';
import clsx from 'clsx';

interface QualityScoreCellProps {
  qualityScore: number;    // 0-100
  avgSignalDbm: number;    // e.g., -65
  dropRate: number;        // 0-1 normalized
}

function getSignalLabel(dbm: number): string {
  if (dbm >= -50) return 'Excellent';
  if (dbm >= -60) return 'Good';
  if (dbm >= -70) return 'Fair';
  return 'Poor';
}

function getScoreColor(score: number): string {
  if (score >= 80) return 'bg-emerald-500';
  if (score >= 60) return 'bg-yellow-500';
  return 'bg-red-500';
}

function getScoreTextColor(score: number): string {
  if (score >= 80) return 'text-emerald-400';
  if (score >= 60) return 'text-yellow-400';
  return 'text-red-400';
}

export function QualityScoreCell({ qualityScore, avgSignalDbm, dropRate }: QualityScoreCellProps) {
  const [tooltipVisible, setTooltipVisible] = useState(false);

  // Calculate score breakdown using the same formula as backend
  // Signal: -30 dBm = 100, -90 dBm = 0
  const signalScore = Math.max(0, Math.min(100, (avgSignalDbm + 90) / 60 * 100));
  // Drop score: 0% drops = 100, 100% drops = 0
  const dropScore = (1 - Math.min(dropRate, 1)) * 100;

  // Weighted contributions
  const signalContribution = signalScore * 0.7;
  const stabilityContribution = dropScore * 0.3;

  const signalLabel = getSignalLabel(avgSignalDbm);
  const dropRatePercent = (dropRate * 100).toFixed(1);

  return (
    <div className="relative">
      <div
        className="flex items-center gap-2 cursor-help"
        onMouseEnter={() => setTooltipVisible(true)}
        onMouseLeave={() => setTooltipVisible(false)}
      >
        <div className="w-24 h-2 bg-slate-700 rounded-full overflow-hidden">
          <div
            className={clsx('h-full rounded-full', getScoreColor(qualityScore))}
            style={{ width: `${qualityScore}%` }}
          />
        </div>
        <span className="text-xs text-slate-500">{qualityScore.toFixed(0)}%</span>
      </div>

      {/* Tooltip */}
      {tooltipVisible && (
        <div className="absolute bottom-full mb-2 left-0 z-50 w-72 p-3 rounded-lg bg-slate-900 border border-slate-700 shadow-xl">
          <div className="text-sm font-medium text-white mb-3">Quality Score Breakdown</div>

          {/* Signal Strength */}
          <div className="mb-3">
            <div className="flex justify-between text-xs mb-1">
              <span className="text-slate-400">Signal Strength (70%)</span>
              <span className="text-white">{Math.round(signalScore)}%</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                <div
                  className={clsx('h-full rounded-full', getScoreColor(signalScore))}
                  style={{ width: `${signalScore}%` }}
                />
              </div>
              <span className="text-xs text-slate-500 w-24 text-right">
                {avgSignalDbm.toFixed(0)} dBm ({signalLabel})
              </span>
            </div>
          </div>

          {/* Connection Stability */}
          <div className="mb-3">
            <div className="flex justify-between text-xs mb-1">
              <span className="text-slate-400">Connection Stability (30%)</span>
              <span className="text-white">{Math.round(dropScore)}%</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                <div
                  className={clsx('h-full rounded-full', getScoreColor(dropScore))}
                  style={{ width: `${dropScore}%` }}
                />
              </div>
              <span className="text-xs text-slate-500 w-24 text-right">
                {dropRatePercent}% drop rate
              </span>
            </div>
          </div>

          {/* Total Calculation */}
          <div className="pt-2 border-t border-slate-700/50">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-400">
                {Math.round(signalContribution)} + {Math.round(stabilityContribution)} =
              </span>
              <span className={clsx('font-medium', getScoreTextColor(qualityScore))}>
                {qualityScore.toFixed(0)}% Total
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default QualityScoreCell;
