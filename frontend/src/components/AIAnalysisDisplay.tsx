import React from 'react';
import { RootCauseHypothesis, EvidenceHypothesis } from '../types/anomaly';
import { parseAIResponse, needsParsing, ParsedAnalysis } from '../utils/llmResponseParser';

// Icons
const BrainIcon = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
  </svg>
);

const RefreshIcon = () => (
  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
  </svg>
);

const CheckCircleIcon = () => (
  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);

const XCircleIcon = () => (
  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);

const AlertTriangleIcon = () => (
  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
  </svg>
);

// Confidence Badge Component
const ConfidenceBadge: React.FC<{
  level: 'high' | 'medium' | 'low' | string;
  percentage?: number;
}> = ({ level, percentage }) => {
  const normalizedLevel = level.toLowerCase();

  const colorClasses = {
    high: 'bg-green-500/20 text-green-400 border-green-500/30',
    medium: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
    low: 'bg-red-500/20 text-red-400 border-red-500/30',
  };

  const classes = colorClasses[normalizedLevel as keyof typeof colorClasses] || colorClasses.medium;
  const displayText = percentage !== undefined ? `${percentage}%` : level;

  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-medium border ${classes}`}>
      {displayText} confidence
    </span>
  );
};

// Urgency Badge Component
const UrgencyBadge: React.FC<{ urgency: string }> = ({ urgency }) => {
  const normalizedUrgency = urgency.toLowerCase();

  const config = {
    immediate: { bg: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-500/30', label: 'Immediate' },
    soon: { bg: 'bg-amber-500/20', text: 'text-amber-400', border: 'border-amber-500/30', label: 'Soon' },
    monitor: { bg: 'bg-blue-500/20', text: 'text-blue-400', border: 'border-blue-500/30', label: 'Monitor' },
  };

  const c = config[normalizedUrgency as keyof typeof config] || config.soon;

  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-medium border ${c.bg} ${c.text} ${c.border}`}>
      {c.label}
    </span>
  );
};

// Strength Badge Component
const StrengthBadge: React.FC<{ strength: string }> = ({ strength }) => {
  const config = {
    strong: 'text-green-400',
    moderate: 'text-amber-400',
    weak: 'text-slate-400',
  };

  return (
    <span className={`text-xs ${config[strength as keyof typeof config] || config.moderate}`}>
      ({strength})
    </span>
  );
};

// Hypothesis Card Component
const HypothesisCard: React.FC<{
  title: string;
  confidence: string;
  confidencePercentage?: number;
  description: string;
  isPrimary?: boolean;
}> = ({ title, confidence, confidencePercentage, description, isPrimary = true }) => {
  return (
    <div className={`rounded-lg ${isPrimary ? 'bg-gradient-to-br from-indigo-500/10 to-purple-500/10 border border-indigo-500/20' : 'bg-slate-800/30 border border-slate-700/50'}`}>
      <div className="p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            {isPrimary && (
              <div className="w-8 h-8 rounded-lg bg-indigo-500/20 flex items-center justify-center text-indigo-400">
                <BrainIcon />
              </div>
            )}
            <span className={`text-xs font-medium ${isPrimary ? 'text-indigo-400' : 'text-slate-500'}`}>
              {isPrimary ? 'Primary Hypothesis' : 'Alternative Hypothesis'}
            </span>
          </div>
          <ConfidenceBadge level={confidence} percentage={confidencePercentage} />
        </div>
        <h4 className="text-base font-semibold text-white mb-2">{title}</h4>
        <p className="text-sm text-slate-300 leading-relaxed">{description}</p>
      </div>
    </div>
  );
};

// Alternative Hypothesis Card
const AlternativeHypothesisCard: React.FC<{
  title: string;
  confidence: string;
  whyLessLikely: string;
}> = ({ title, confidence, whyLessLikely }) => {
  return (
    <div className="rounded-lg bg-slate-800/30 border border-slate-700/50 p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-slate-500">Alternative Hypothesis</span>
        <ConfidenceBadge level={confidence} />
      </div>
      <h5 className="text-sm font-medium text-white mb-1">{title}</h5>
      <p className="text-xs text-slate-400">
        <span className="text-slate-500">Why less likely:</span> {whyLessLikely}
      </p>
    </div>
  );
};

// Evidence Section Component
const EvidenceSection: React.FC<{
  evidenceFor: EvidenceHypothesis[];
  evidenceAgainst: EvidenceHypothesis[];
  parsedEvidence?: string[];
}> = ({ evidenceFor, evidenceAgainst, parsedEvidence }) => {
  const hasEvidence = evidenceFor.length > 0 || evidenceAgainst.length > 0 || (parsedEvidence && parsedEvidence.length > 0);

  if (!hasEvidence) return null;

  return (
    <div className="rounded-lg bg-slate-800/30 border border-slate-700/50 p-4">
      <h5 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-3">Supporting Evidence</h5>

      <div className="space-y-3">
        {/* Parsed evidence from LLM text */}
        {parsedEvidence && parsedEvidence.length > 0 && (
          <ul className="space-y-2">
            {parsedEvidence.map((item, index) => (
              <li key={`parsed-${index}`} className="flex items-start gap-2 text-sm">
                <CheckCircleIcon />
                <span className="text-slate-300">{item}</span>
              </li>
            ))}
          </ul>
        )}

        {/* Structured evidence FOR */}
        {evidenceFor.length > 0 && (
          <div>
            <p className="text-xs text-green-400 font-medium mb-2 flex items-center gap-1">
              <CheckCircleIcon /> Evidence For
            </p>
            <ul className="space-y-2 ml-5">
              {evidenceFor.map((ev, index) => (
                <li key={`for-${index}`} className="text-sm text-slate-300">
                  <span>{ev.statement}</span>
                  <StrengthBadge strength={ev.strength} />
                  <span className="text-xs text-slate-500 ml-1">({ev.source})</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Structured evidence AGAINST */}
        {evidenceAgainst.length > 0 && (
          <div>
            <p className="text-xs text-red-400 font-medium mb-2 flex items-center gap-1">
              <XCircleIcon /> Evidence Against
            </p>
            <ul className="space-y-2 ml-5">
              {evidenceAgainst.map((ev, index) => (
                <li key={`against-${index}`} className="text-sm text-slate-300">
                  <span>{ev.statement}</span>
                  <StrengthBadge strength={ev.strength} />
                  <span className="text-xs text-slate-500 ml-1">({ev.source})</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};

// Actions Panel Component
const ActionsPanel: React.FC<{
  urgency?: string;
  actions: string[];
}> = ({ urgency, actions }) => {
  if (!actions || actions.length === 0) return null;

  return (
    <div className="rounded-lg bg-slate-800/30 border border-slate-700/50 p-4">
      <div className="flex items-center justify-between mb-3">
        <h5 className="text-xs font-medium text-slate-400 uppercase tracking-wider">Recommended Actions</h5>
        {urgency && <UrgencyBadge urgency={urgency} />}
      </div>
      <ol className="space-y-2">
        {actions.map((action, index) => (
          <li key={index} className="flex items-start gap-3 text-sm">
            <span className="flex-shrink-0 w-5 h-5 rounded-full bg-indigo-500/20 text-indigo-400 text-xs flex items-center justify-center font-medium">
              {index + 1}
            </span>
            <span className="text-slate-300">{action}</span>
          </li>
        ))}
      </ol>
    </div>
  );
};

// Business Impact Banner Component
const BusinessImpactBanner: React.FC<{ impact: string }> = ({ impact }) => {
  if (!impact || impact.trim() === '') return null;

  return (
    <div className="rounded-lg bg-amber-500/10 border border-amber-500/20 p-4">
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0 text-amber-400">
          <AlertTriangleIcon />
        </div>
        <div>
          <h5 className="text-xs font-medium text-amber-400 uppercase tracking-wider mb-1">Business Impact</h5>
          <p className="text-sm text-amber-300/90">{impact}</p>
        </div>
      </div>
    </div>
  );
};

// Main AIAnalysisDisplay Component
export interface AIAnalysisDisplayProps {
  hypothesis: RootCauseHypothesis;
  onRegenerate?: () => void;
}

export const AIAnalysisDisplay: React.FC<AIAnalysisDisplayProps> = ({
  hypothesis,
  onRegenerate,
}) => {
  // Check if the description needs parsing (contains raw LLM output)
  const shouldParse = needsParsing(hypothesis.description);
  const parsed: ParsedAnalysis | null = shouldParse ? parseAIResponse(hypothesis.description) : null;

  // Determine confidence level from likelihood
  const getConfidenceLevel = (likelihood: number): 'high' | 'medium' | 'low' => {
    if (likelihood >= 0.7) return 'high';
    if (likelihood >= 0.4) return 'medium';
    return 'low';
  };

  const confidencePercentage = Math.round(hypothesis.likelihood * 100);
  const confidenceLevel = getConfidenceLevel(hypothesis.likelihood);

  return (
    <div className="space-y-4">
      {/* Header with regenerate button */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-indigo-500/20 flex items-center justify-center text-indigo-400">
            <BrainIcon />
          </div>
          <span className="text-sm font-medium text-indigo-400">AI Root Cause Analysis</span>
        </div>
        {onRegenerate && (
          <button
            onClick={onRegenerate}
            className="p-2 text-slate-400 hover:text-indigo-400 hover:bg-indigo-500/10 rounded-lg transition-colors"
            title="Regenerate analysis"
          >
            <RefreshIcon />
          </button>
        )}
      </div>

      {parsed ? (
        // Render parsed LLM content
        <>
          <HypothesisCard
            title={parsed.primaryHypothesis.title}
            confidence={parsed.primaryHypothesis.confidence}
            confidencePercentage={confidencePercentage}
            description={parsed.primaryHypothesis.description}
            isPrimary={true}
          />

          <EvidenceSection
            evidenceFor={hypothesis.evidence_for || []}
            evidenceAgainst={hypothesis.evidence_against || []}
            parsedEvidence={parsed.supportingEvidence}
          />

          {parsed.alternativeHypothesis && (
            <AlternativeHypothesisCard
              title={parsed.alternativeHypothesis.title}
              confidence={parsed.alternativeHypothesis.confidence}
              whyLessLikely={parsed.alternativeHypothesis.whyLessLikely}
            />
          )}

          <ActionsPanel
            urgency={parsed.recommendedActions.urgency}
            actions={parsed.recommendedActions.actions.length > 0
              ? parsed.recommendedActions.actions
              : hypothesis.recommended_actions || []
            }
          />

          {parsed.businessImpact && (
            <BusinessImpactBanner impact={parsed.businessImpact} />
          )}
        </>
      ) : (
        // Render structured data (fallback for already-parsed or mock data)
        <>
          <HypothesisCard
            title={hypothesis.title}
            confidence={confidenceLevel}
            confidencePercentage={confidencePercentage}
            description={hypothesis.description}
            isPrimary={true}
          />

          <EvidenceSection
            evidenceFor={hypothesis.evidence_for || []}
            evidenceAgainst={hypothesis.evidence_against || []}
          />

          <ActionsPanel actions={hypothesis.recommended_actions || []} />
        </>
      )}
    </div>
  );
};

export default AIAnalysisDisplay;
