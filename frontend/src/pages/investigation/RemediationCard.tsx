/**
 * Remediation Card component.
 */

import { useState } from 'react';
import type { RemediationSuggestion } from '../../types/anomaly';

interface RemediationCardProps {
  remediation: RemediationSuggestion;
  onApply: () => void;
  onMarkSuccess: () => void;
}

export function RemediationCard({ remediation, onApply, onMarkSuccess }: RemediationCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/50 hover:border-amber-500/20 transition-colors">
      <div className="flex items-start justify-between">
        <div className="flex items-start gap-3">
          <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
            remediation.confidence_level === 'high' ? 'bg-emerald-500/20 text-emerald-400' :
            remediation.confidence_level === 'medium' ? 'bg-amber-500/20 text-amber-400' :
            'bg-slate-700/50 text-slate-400'
          }`}>
            <span className="text-sm font-bold">#{remediation.priority}</span>
          </div>
          <div className="flex-1">
            <h4 className="text-sm font-medium text-white">{remediation.title}</h4>
            <p className="text-xs text-slate-400 mt-1">{remediation.description}</p>
            {remediation.estimated_impact && (
              <p className="text-xs text-emerald-400 mt-1">Impact: {remediation.estimated_impact}</p>
            )}
          </div>
        </div>
        <span className={`text-xs px-2 py-0.5 rounded ${
          remediation.source === 'policy' ? 'bg-blue-500/20 text-blue-400' :
          remediation.source === 'ai_generated' ? 'bg-purple-500/20 text-purple-400' :
          'bg-slate-700/50 text-slate-400'
        }`}>
          {remediation.source === 'ai_generated' ? 'AI' : remediation.source}
        </span>
      </div>

      {expanded && (
        <div className="mt-4 pt-4 border-t border-slate-700/30">
          <p className="text-xs text-slate-500 mb-2">Steps:</p>
          <ol className="space-y-1">
            {remediation.detailed_steps.map((step, i) => (
              <li key={i} className="text-xs text-slate-300 flex items-start gap-2">
                <span className="text-amber-400 font-mono">{i + 1}.</span>
                {step}
              </li>
            ))}
          </ol>
        </div>
      )}

      <div className="mt-3 flex items-center gap-2">
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-xs text-slate-400 hover:text-white transition-colors"
        >
          {expanded ? 'Show less' : 'Show steps'}
        </button>
        <div className="flex-1" />
        <button
          onClick={onMarkSuccess}
          className="text-xs text-emerald-400 hover:text-emerald-300 transition-colors"
        >
          Mark as fixed
        </button>
        <button
          onClick={onApply}
          className="px-3 py-1 text-xs font-medium text-amber-400 bg-amber-500/10 rounded-lg hover:bg-amber-500/20 transition-colors"
        >
          Apply
        </button>
      </div>
    </div>
  );
}
