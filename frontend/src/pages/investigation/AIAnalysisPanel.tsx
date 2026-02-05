/**
 * AI Root Cause Analysis Panel component.
 */

import { AIAnalysisDisplay } from '../../components/AIAnalysisDisplay';
import type { RootCauseHypothesis } from '../../types/anomaly';

interface AIAnalysisPanelProps {
  hypothesis?: RootCauseHypothesis;
  analysisSource?: 'llm' | 'rule_based' | 'unavailable';
  isLoading: boolean;
  onRegenerate: () => void;
}

function RuleBasedBanner() {
  return (
    <div className="mb-4 flex items-start gap-3 rounded-lg bg-amber-500/10 border border-amber-500/20 px-4 py-3">
      <svg className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
      <div>
        <p className="text-sm font-medium text-amber-400">Rule-based analysis</p>
        <p className="text-xs text-amber-400/70 mt-0.5">
          The AI model is currently unavailable. This analysis uses pattern-matching rules
          instead of AI. Results may be less detailed. Ensure the LLM service is running for
          AI-powered insights.
        </p>
      </div>
    </div>
  );
}

export function AIAnalysisPanel({ hypothesis, analysisSource, isLoading, onRegenerate }: AIAnalysisPanelProps) {
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

  const isRuleBased = analysisSource === 'rule_based' || analysisSource === 'unavailable';

  return (
    <div className="p-6 rounded-xl bg-gradient-to-br from-indigo-500/10 to-purple-500/10 border border-indigo-500/20">
      {isRuleBased && <RuleBasedBanner />}
      <AIAnalysisDisplay hypothesis={hypothesis} onRegenerate={onRegenerate} />
    </div>
  );
}
