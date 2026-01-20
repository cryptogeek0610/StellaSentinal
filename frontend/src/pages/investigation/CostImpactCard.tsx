/**
 * Cost Impact Card component.
 */

import { Link } from 'react-router-dom';
import { DollarIcon } from './Icons';
import { formatCurrency } from './investigationUtils';
import type { AnomalyImpactResponse } from '../../types/cost';

interface CostImpactCardProps {
  costImpact?: AnomalyImpactResponse;
  isLoading: boolean;
}

export function CostImpactCard({ costImpact, isLoading }: CostImpactCardProps) {
  if (isLoading) {
    return (
      <div className="p-6 rounded-xl bg-slate-800/30 border border-slate-700/50">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full border-2 border-amber-500/30 border-t-amber-500 animate-spin" />
          <div>
            <p className="text-amber-400 font-medium text-sm">Calculating impact...</p>
            <p className="text-xs text-slate-500">Analyzing financial data</p>
          </div>
        </div>
      </div>
    );
  }

  if (!costImpact || costImpact.using_defaults) {
    return (
      <div className="p-6 rounded-xl bg-slate-800/30 border border-slate-700/50 text-center">
        <div className="w-10 h-10 mx-auto mb-2 rounded-full bg-slate-700/50 flex items-center justify-center text-slate-400">
          <DollarIcon />
        </div>
        <p className="text-slate-400 text-sm">No cost data configured</p>
        <p className="text-xs text-slate-500 mt-1">Configure your labor rates and device costs to see financial impact</p>
        <Link
          to="/costs"
          className="mt-3 inline-block text-xs text-amber-400 hover:text-amber-300 transition-colors"
        >
          Configure Costs →
        </Link>
      </div>
    );
  }

  const impactColors = {
    high: { bg: 'bg-red-500/10', border: 'border-red-500/30', text: 'text-red-400', badge: 'bg-red-500/20 text-red-300' },
    medium: { bg: 'bg-amber-500/10', border: 'border-amber-500/30', text: 'text-amber-400', badge: 'bg-amber-500/20 text-amber-300' },
    low: { bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-400', badge: 'bg-emerald-500/20 text-emerald-300' },
  };

  const colors = impactColors[costImpact.impact_level] || impactColors.medium;

  return (
    <div className={`p-4 rounded-xl ${colors.bg} border ${colors.border}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${colors.bg} ${colors.text}`}>
            <DollarIcon />
          </div>
          <span className="text-sm font-semibold text-white">Financial Impact</span>
        </div>
        <span className={`px-2 py-0.5 rounded text-xs font-bold uppercase ${colors.badge}`}>
          {costImpact.impact_level}
        </span>
      </div>

      {/* Total Impact */}
      <div className="text-center p-4 rounded-lg bg-slate-800/30 mb-4">
        <p className="text-xs text-slate-500 mb-1">Estimated Total Impact</p>
        <p className={`text-2xl font-bold font-mono ${colors.text}`}>
          {formatCurrency(costImpact.total_estimated_impact)}
        </p>
        <p className="text-[10px] text-slate-600 mt-1">
          {(costImpact.overall_confidence * 100).toFixed(0)}% confidence
        </p>
      </div>

      {/* Cost Breakdown */}
      {costImpact.impact_components && costImpact.impact_components.length > 0 && (
        <div className="space-y-2 mb-4">
          <p className="text-xs font-medium text-slate-400">Cost Breakdown</p>
          {costImpact.impact_components.map((component, idx) => (
            <div
              key={idx}
              className="flex items-center justify-between p-2 rounded-lg bg-slate-800/30"
            >
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-slate-300 truncate">{component.type}</p>
                <p className="text-[10px] text-slate-500 truncate">{component.description}</p>
              </div>
              <span className="text-xs font-mono text-slate-300 ml-2">
                {formatCurrency(component.amount)}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Device Value Context */}
      {(costImpact.device_unit_cost || costImpact.device_depreciated_value) && (
        <div className="pt-3 border-t border-slate-700/30">
          <p className="text-xs font-medium text-slate-400 mb-2">Device Context</p>
          <div className="grid grid-cols-2 gap-2">
            {costImpact.device_unit_cost && (
              <div className="text-center p-2 rounded bg-slate-800/30">
                <p className="text-[10px] text-slate-500">Purchase Cost</p>
                <p className="text-xs font-mono text-slate-300">{formatCurrency(costImpact.device_unit_cost)}</p>
              </div>
            )}
            {costImpact.device_depreciated_value && (
              <div className="text-center p-2 rounded bg-slate-800/30">
                <p className="text-[10px] text-slate-500">Current Value</p>
                <p className="text-xs font-mono text-emerald-400">{formatCurrency(costImpact.device_depreciated_value)}</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Explanation */}
      <div className="mt-3 pt-3 border-t border-slate-700/30">
        <p className="text-[10px] text-slate-500">{costImpact.confidence_explanation}</p>
      </div>
    </div>
  );
}
