/**
 * AI Insights Panel - Stellar Operations Intelligence Display
 * 
 * Displays AI-generated insights with severity filtering,
 * explainability sections, and actionable recommendations
 */

import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AIInsight, AIInsightSeverity, FinancialImpact } from '../types/anomaly';

// ============================================================================
// Financial Impact Helpers
// ============================================================================

const formatCurrency = (amount: number, includeCents = false): string => {
  if (includeCents) {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(amount);
  }
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(amount);
};

const getFinancialImpactColor = (level: string) => {
  switch (level) {
    case 'high':
      return {
        bg: 'bg-red-500/10',
        border: 'border-red-500/30',
        text: 'text-red-400',
        badge: 'bg-red-500/20 text-red-300',
      };
    case 'medium':
      return {
        bg: 'bg-amber-500/10',
        border: 'border-amber-500/30',
        text: 'text-amber-400',
        badge: 'bg-amber-500/20 text-amber-300',
      };
    case 'low':
    default:
      return {
        bg: 'bg-emerald-500/10',
        border: 'border-emerald-500/30',
        text: 'text-emerald-400',
        badge: 'bg-emerald-500/20 text-emerald-300',
      };
  }
};

// Financial Impact Section Component
const FinancialImpactSection: React.FC<{ impact: FinancialImpact }> = ({ impact }) => {
  const colors = getFinancialImpactColor(impact.impact_level);

  return (
    <div className={`p-3 rounded-lg ${colors.bg} border ${colors.border} mb-3`}>
      <div className="flex items-center gap-2 mb-2">
        <svg className={`w-4 h-4 ${colors.text}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
        <span className={`text-xs font-semibold ${colors.text}`}>Financial Impact</span>
        <span className={`ml-auto px-1.5 py-0.5 rounded text-[10px] font-bold uppercase ${colors.badge}`}>
          {impact.impact_level}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2">
        {/* Total Impact */}
        <div className="text-center p-2 rounded bg-slate-800/30">
          <p className="text-[10px] text-slate-500 mb-0.5">Total Impact</p>
          <p className={`text-sm font-bold font-mono ${colors.text}`}>
            {formatCurrency(impact.total_impact_usd)}
          </p>
        </div>

        {/* Potential Savings */}
        {impact.potential_savings_usd !== undefined && impact.potential_savings_usd > 0 && (
          <div className="text-center p-2 rounded bg-slate-800/30">
            <p className="text-[10px] text-slate-500 mb-0.5">Potential Savings</p>
            <p className="text-sm font-bold font-mono text-emerald-400">
              {formatCurrency(impact.potential_savings_usd)}
            </p>
          </div>
        )}

        {/* Monthly Recurring */}
        {impact.monthly_recurring_usd !== undefined && impact.monthly_recurring_usd > 0 && (
          <div className="text-center p-2 rounded bg-slate-800/30">
            <p className="text-[10px] text-slate-500 mb-0.5">Monthly</p>
            <p className="text-sm font-bold font-mono text-amber-400">
              {formatCurrency(impact.monthly_recurring_usd)}/mo
            </p>
          </div>
        )}

        {/* Payback Period */}
        {impact.payback_months !== undefined && impact.payback_months > 0 && (
          <div className="text-center p-2 rounded bg-slate-800/30">
            <p className="text-[10px] text-slate-500 mb-0.5">ROI Payback</p>
            <p className="text-sm font-bold font-mono text-cyan-400">
              {impact.payback_months} mo
            </p>
          </div>
        )}
      </div>

      {/* Breakdown (if available and has items) */}
      {impact.breakdown && impact.breakdown.length > 0 && (
        <div className="mt-2 pt-2 border-t border-slate-700/30">
          <p className="text-[10px] text-slate-500 mb-1">Cost Breakdown</p>
          <div className="space-y-1">
            {impact.breakdown.slice(0, 3).map((item, idx) => (
              <div key={idx} className="flex items-center justify-between text-[10px]">
                <span className="text-slate-400 truncate flex-1 mr-2">{item.description}</span>
                <span className="font-mono text-slate-300 whitespace-nowrap">
                  {formatCurrency(item.amount)}
                  {item.is_recurring && <span className="text-slate-500">/mo</span>}
                </span>
              </div>
            ))}
            {impact.breakdown.length > 3 && (
              <p className="text-[10px] text-slate-500">
                +{impact.breakdown.length - 3} more items
              </p>
            )}
          </div>
        </div>
      )}

      {/* Confidence indicator */}
      {impact.confidence_score !== undefined && (
        <div className="mt-2 pt-2 border-t border-slate-700/30 flex items-center gap-2">
          <span className="text-[10px] text-slate-500">Confidence:</span>
          <div className="flex-1 h-1 bg-slate-700/50 rounded-full overflow-hidden">
            <div
              className={`h-full ${impact.confidence_score >= 0.7 ? 'bg-emerald-500' : impact.confidence_score >= 0.4 ? 'bg-amber-500' : 'bg-red-500'}`}
              style={{ width: `${impact.confidence_score * 100}%` }}
            />
          </div>
          <span className="text-[10px] text-slate-400 font-mono">
            {(impact.confidence_score * 100).toFixed(0)}%
          </span>
        </div>
      )}
    </div>
  );
};

interface AIInsightsPanelProps {
  title?: string;
  insights: AIInsight[];
  loading?: boolean;
  className?: string;
  onInsightAction?: (insight: AIInsight, action: 'apply' | 'dismiss') => void;
}

const getSeverityConfig = (severity: AIInsightSeverity) => {
  switch (severity) {
    case 'critical':
      return {
        bg: 'bg-red-500/10',
        border: 'border-red-500/30',
        text: 'text-red-400',
        badge: 'bg-red-500/20 text-red-300 border border-red-500/40',
        glow: 'shadow-[0_0_15px_rgba(239,68,68,0.2)]',
      };
    case 'high':
      return {
        bg: 'bg-orange-500/10',
        border: 'border-orange-500/30',
        text: 'text-orange-400',
        badge: 'bg-orange-500/20 text-orange-300 border border-orange-500/40',
        glow: '',
      };
    case 'medium':
    case 'warning':
      return {
        bg: 'bg-amber-500/10',
        border: 'border-amber-500/20',
        text: 'text-amber-400',
        badge: 'bg-amber-500/20 text-amber-300 border border-amber-500/40',
        glow: '',
      };
    case 'low':
    case 'info':
      return {
        bg: 'bg-cyan-500/10',
        border: 'border-cyan-500/20',
        text: 'text-cyan-400',
        badge: 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40',
        glow: '',
      };
    default:
      return {
        bg: 'bg-slate-500/10',
        border: 'border-slate-500/20',
        text: 'text-slate-400',
        badge: 'bg-slate-500/20 text-slate-300 border border-slate-500/40',
        glow: '',
      };
  }
};

const getTypeIcon = (type: string) => {
  const iconClass = 'w-4 h-4';
  switch (type) {
    case 'workload':
      return (
        <svg className={iconClass} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      );
    case 'efficiency':
      return (
        <svg className={iconClass} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
        </svg>
      );
    case 'optimization':
      return (
        <svg className={iconClass} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
        </svg>
      );
    case 'pattern':
    case 'trend':
      return (
        <svg className={iconClass} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      );
    case 'anomaly':
    case 'warning':
      return (
        <svg className={iconClass} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
      );
    default:
      return (
        <svg className={iconClass} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
      );
  }
};

export const AIInsightsPanel: React.FC<AIInsightsPanelProps> = ({
  title = 'AI Insights',
  insights,
  loading = false,
  className = '',
  onInsightAction,
}) => {
  const [severityFilter, setSeverityFilter] = useState<'all' | 'high' | 'medium' | 'low'>('all');
  const [expandedInsight, setExpandedInsight] = useState<string | null>(null);

  // Group insights by severity
  const highPriority = insights.filter((i) => i.severity === 'high' || i.severity === 'critical');
  const mediumPriority = insights.filter((i) => i.severity === 'medium' || i.severity === 'warning');
  const lowPriority = insights.filter((i) => i.severity === 'low' || i.severity === 'info');

  // Filter insights based on selected severity
  const filteredInsights = useMemo(() => {
    if (severityFilter === 'all') return insights;
    if (severityFilter === 'high') return highPriority;
    if (severityFilter === 'medium') return mediumPriority;
    if (severityFilter === 'low') return lowPriority;
    return insights;
  }, [insights, severityFilter, highPriority, mediumPriority, lowPriority]);

  if (loading) {
    return (
      <div className={`stellar-card rounded-2xl p-6 ${className}`}>
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-600 to-purple-600 flex items-center justify-center">
            <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <h3 className="text-lg font-bold text-white">{title}</h3>
        </div>
        <div className="flex items-center justify-center py-8">
          <div className="w-8 h-8 border-4 border-amber-500 border-t-transparent rounded-full animate-spin" />
        </div>
      </div>
    );
  }

  if (insights.length === 0) {
    return (
      <div className={`stellar-card rounded-2xl p-6 ${className}`}>
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-600 to-purple-600 flex items-center justify-center">
            <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <h3 className="text-lg font-bold text-white">{title}</h3>
        </div>
        <div className="text-center py-8">
          <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-emerald-500/10 flex items-center justify-center">
            <svg className="w-6 h-6 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          </div>
          <p className="text-slate-400 text-sm">No insights detected. All systems operating normally.</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`stellar-card rounded-2xl p-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-600 to-purple-600 flex items-center justify-center">
            <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <div>
            <h3 className="text-lg font-bold text-white">{title}</h3>
            <p className="text-xs text-slate-500">AI-powered fleet analysis</p>
          </div>
        </div>

        {/* Filter Badges */}
        <div className="flex items-center gap-2">
          {highPriority.length > 0 && (
            <button
              onClick={() => setSeverityFilter(severityFilter === 'high' ? 'all' : 'high')}
              className={`px-2 py-1 rounded-full text-xs font-semibold transition-all cursor-pointer ${
                severityFilter === 'high'
                  ? 'bg-red-500/30 text-red-300 ring-2 ring-red-500/50'
                  : 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
              }`}
            >
              {highPriority.length} High
            </button>
          )}
          {mediumPriority.length > 0 && (
            <button
              onClick={() => setSeverityFilter(severityFilter === 'medium' ? 'all' : 'medium')}
              className={`px-2 py-1 rounded-full text-xs font-semibold transition-all cursor-pointer ${
                severityFilter === 'medium'
                  ? 'bg-amber-500/30 text-amber-300 ring-2 ring-amber-500/50'
                  : 'bg-amber-500/20 text-amber-400 hover:bg-amber-500/30'
              }`}
            >
              {mediumPriority.length} Medium
            </button>
          )}
          <button
            onClick={() => setSeverityFilter('all')}
            className={`px-2 py-1 rounded-full text-xs font-semibold transition-all cursor-pointer ${
              severityFilter === 'all'
                ? 'bg-indigo-500/30 text-indigo-300 ring-2 ring-indigo-500/50'
                : 'bg-indigo-500/20 text-indigo-400 hover:bg-indigo-500/30'
            }`}
          >
            {insights.length} Total
          </button>
        </div>
      </div>

      {/* Insights List */}
      <div className="space-y-4">
        <AnimatePresence mode="popLayout">
          {filteredInsights.map((insight, index) => {
            const config = getSeverityConfig(insight.severity);
            const isExpanded = expandedInsight === insight.id;

            return (
              <motion.div
                key={insight.id}
                layout
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ delay: index * 0.05 }}
                className={`p-4 rounded-xl border ${config.bg} ${config.border} ${config.glow} transition-all hover:scale-[1.01]`}
              >
                {/* Insight Header */}
                <div className="flex items-start gap-3">
                  <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${config.bg} ${config.text}`}>
                    {getTypeIcon(insight.type)}
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1 flex-wrap">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${config.badge}`}>
                        {insight.severity}
                      </span>
                      <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-slate-700/50 text-slate-400 uppercase">
                        {insight.type}
                      </span>
                      <h4 className="text-sm font-semibold text-white">{insight.title}</h4>
                    </div>

                    <p className="text-xs text-slate-400 leading-relaxed mb-3">{insight.description}</p>

                    {/* Financial Impact */}
                    {insight.financialImpact && (
                      <FinancialImpactSection impact={insight.financialImpact} />
                    )}

                    {/* Recommendation */}
                    {insight.recommendation && (
                      <div className="p-2.5 rounded-lg bg-slate-800/50 border border-cyan-500/20 mb-3">
                        <p className="text-xs font-semibold text-cyan-400 mb-1 flex items-center gap-1">
                          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                          </svg>
                          AI Recommendation
                        </p>
                        <p className="text-xs text-slate-300">{insight.recommendation}</p>
                      </div>
                    )}

                    {/* Expandable Why/How/WhatToDo */}
                    {(insight.why || insight.how || insight.whatToDo) && (
                      <div className="mb-3">
                        <button
                          onClick={() => setExpandedInsight(isExpanded ? null : insight.id)}
                          className="text-xs font-semibold text-indigo-400 hover:text-indigo-300 flex items-center gap-1"
                        >
                          <svg
                            className={`w-3.5 h-3.5 transition-transform ${isExpanded ? 'rotate-90' : ''}`}
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                          </svg>
                          {isExpanded ? 'Hide details' : 'Why this matters'}
                        </button>

                        <AnimatePresence>
                          {isExpanded && (
                            <motion.div
                              initial={{ height: 0, opacity: 0 }}
                              animate={{ height: 'auto', opacity: 1 }}
                              exit={{ height: 0, opacity: 0 }}
                              className="overflow-hidden"
                            >
                              <div className="mt-3 space-y-2">
                                {insight.why && (
                                  <div className="p-2.5 rounded-lg bg-red-900/20 border border-red-500/20">
                                    <p className="text-xs font-semibold text-red-400 mb-1 flex items-center gap-1">
                                      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                      </svg>
                                      Why This Matters
                                    </p>
                                    <p className="text-xs text-slate-300">{insight.why}</p>
                                  </div>
                                )}
                                {insight.how && (
                                  <div className="p-2.5 rounded-lg bg-cyan-900/20 border border-cyan-500/20">
                                    <p className="text-xs font-semibold text-cyan-400 mb-1 flex items-center gap-1">
                                      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                                      </svg>
                                      How Detected
                                    </p>
                                    <p className="text-xs text-slate-300">{insight.how}</p>
                                  </div>
                                )}
                                {insight.whatToDo && (
                                  <div className="p-2.5 rounded-lg bg-emerald-900/20 border border-emerald-500/20">
                                    <p className="text-xs font-semibold text-emerald-400 mb-1 flex items-center gap-1">
                                      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                      </svg>
                                      What To Do
                                    </p>
                                    <p className="text-xs text-slate-300">{insight.whatToDo}</p>
                                  </div>
                                )}
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>
                    )}

                    {/* Impact Metrics */}
                    {insight.impact && (
                      <div className="flex items-center gap-4 text-xs mb-3">
                        {insight.impact.confidence && (
                          <div className="flex items-center gap-1">
                            <span className="text-slate-500">Confidence:</span>
                            <span className={`font-semibold ${config.text}`}>{insight.impact.confidence}%</span>
                          </div>
                        )}
                        {insight.affectedCount && (
                          <div className="flex items-center gap-1">
                            <span className="text-slate-500">Affected:</span>
                            <span className="font-semibold text-white">{insight.affectedCount} items</span>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Affected Stores */}
                    {insight.affectedStores && insight.affectedStores.length > 0 && (
                      <div className="flex flex-wrap gap-1 mb-3">
                        {insight.affectedStores.slice(0, 3).map((store) => (
                          <span
                            key={store}
                            className="px-2 py-0.5 rounded text-[10px] font-mono bg-amber-500/10 text-amber-400 border border-amber-500/20"
                          >
                            {store}
                          </span>
                        ))}
                        {insight.affectedStores.length > 3 && (
                          <span className="text-[10px] text-slate-500">+{insight.affectedStores.length - 3} more</span>
                        )}
                      </div>
                    )}

                    {/* Action Buttons */}
                    {onInsightAction && (
                      <div className="flex items-center gap-3">
                        <button
                          onClick={() => onInsightAction(insight, 'apply')}
                          className="text-xs font-semibold text-amber-400 hover:text-amber-300 transition-colors"
                        >
                          Apply →
                        </button>
                        <button
                          onClick={() => onInsightAction(insight, 'dismiss')}
                          className="text-xs text-slate-500 hover:text-slate-400 transition-colors"
                        >
                          Dismiss
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>

        {/* No results message */}
        {filteredInsights.length === 0 && (
          <div className="text-center py-8 text-slate-500">
            <p className="text-sm">No insights match the selected filter.</p>
            <button
              onClick={() => setSeverityFilter('all')}
              className="mt-2 text-xs text-cyan-400 hover:text-cyan-300 underline"
            >
              Show all insights
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default AIInsightsPanel;

