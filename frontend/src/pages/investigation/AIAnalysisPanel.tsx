/**
 * AI Root Cause Analysis Panel component.
 */

import { AIAnalysisDisplay } from '../../components/AIAnalysisDisplay';
import type { RootCauseHypothesis } from '../../types/anomaly';

interface AIAnalysisPanelProps {
  hypothesis?: RootCauseHypothesis;
  isLoading: boolean;
  onRegenerate: () => void;
}

export function AIAnalysisPanel({ hypothesis, isLoading, onRegenerate }: AIAnalysisPanelProps) {
  if (isLoading) {
    return (
      <div className="p-6 rounded-xl bg-gradient-to-br from-indigo-500/10 to-purple-500/10 border border-indigo-500/20">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full border-2 border-indigo-500/30 border-t-indigo-500 animate-spin" />
          <div>
            <p className="text-indigo-400 font-medium">Analyzing anomaly...</p>
            <p className="text-xs text-slate-500">AI is determining root cause</p>
          </div>
        </div>
      </div>
    );
  }

  if (!hypothesis) {
    return (
      <div className="p-6 rounded-xl bg-gradient-to-br from-indigo-500/10 to-purple-500/10 border border-indigo-500/20">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full border-2 border-indigo-500/30 border-t-indigo-500 animate-spin" />
          <div>
            <p className="text-indigo-400 font-medium">Generating AI analysis...</p>
            <p className="text-xs text-slate-500">This may take a moment</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 rounded-xl bg-gradient-to-br from-indigo-500/10 to-purple-500/10 border border-indigo-500/20">
      <AIAnalysisDisplay hypothesis={hypothesis} onRegenerate={onRegenerate} />
    </div>
  );
}
