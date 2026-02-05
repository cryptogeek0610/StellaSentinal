/**
 * Investigation Detail Page - AIOps Dashboard
 *
 * Comprehensive anomaly investigation with:
 * - Feature contribution breakdown (why the anomaly was detected)
 * - Baseline comparison visualization
 * - LLM-powered root cause analysis
 * - Device telemetry & historical trends
 * - SOTI MobiControl remediation actions
 * - Similar case reference
 */

import { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../api/client';
import { format, formatDistanceToNowStrict } from 'date-fns';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from '../components/Card';
import { Breadcrumb } from '../components/Breadcrumb';
import { useMockMode } from '../hooks/useMockMode';
import { showSuccess, showError } from '../utils/toast';
import type {
  FeatureContribution,
  RemediationSuggestion,
  LearnFromFix,
  InvestigationNote,
} from '../types/anomaly';

// Import extracted components
import {
  severityConfig,
  statusConfig,
  getSeverityKey,
  BrainIcon,
  ChartIcon,
  TargetIcon,
  WrenchIcon,
  DollarIcon,
  FeatureContributionChart,
  BaselineComparisonViz,
  AIAnalysisPanel,
  CostImpactCard,
  MobiControlActions,
  RemediationCard,
  SimilarCasesList,
  TelemetryTab,
} from './investigation';

function InvestigationDetail() {
  const { id } = useParams<{ id: string }>();

  const queryClient = useQueryClient();
  useMockMode(); // Used for conditional mock data in API client
  const anomalyId = parseInt(id || '0');

  const [noteText, setNoteText] = useState('');
  const [showActions, setShowActions] = useState(false);
  const [activeTab, setActiveTab] = useState<'analysis' | 'telemetry' | 'actions'>('analysis');

  // Fetch anomaly details
  const { data: anomaly, isLoading: anomalyLoading } = useQuery({
    queryKey: ['anomaly', anomalyId],
    queryFn: () => api.getAnomaly(anomalyId),
    enabled: !!anomalyId,
  });

  // Fetch investigation panel data
  const { data: investigation } = useQuery({
    queryKey: ['investigation', anomalyId],
    queryFn: () => api.getInvestigationPanel(anomalyId),
    enabled: !!anomalyId,
  });

  // Fetch AI analysis
  const { data: aiAnalysis, isLoading: aiLoading, refetch: refetchAI } = useQuery({
    queryKey: ['aiAnalysis', anomalyId],
    queryFn: () => api.getAIAnalysis(anomalyId),
    enabled: !!anomalyId,
  });

  // Fetch remediations
  const { data: remediations } = useQuery({
    queryKey: ['remediations', anomalyId],
    queryFn: () => api.getRemediations(anomalyId),
    enabled: !!anomalyId,
  });

  // Fetch similar cases
  const { data: similarCases } = useQuery({
    queryKey: ['similarCases', anomalyId],
    queryFn: () => api.getSimilarCases(anomalyId),
    enabled: !!anomalyId,
  });

  // Fetch cost impact
  const { data: costImpact, isLoading: costImpactLoading } = useQuery({
    queryKey: ['anomalyImpact', anomalyId],
    queryFn: () => api.getAnomalyImpact(anomalyId),
    enabled: !!anomalyId,
  });

  // Status update mutation
  const resolveMutation = useMutation({
    mutationFn: ({ status, notes }: { status: string; notes?: string }) =>
      api.resolveAnomaly(anomalyId, status, notes),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['anomaly', anomalyId] });
      queryClient.invalidateQueries({ queryKey: ['anomalies'] });
      setShowActions(false);
      const statusLabel = variables.status === 'resolved' ? 'Resolved'
        : variables.status === 'investigating' ? 'Investigating'
        : variables.status === 'false_positive' ? 'False Positive'
        : variables.status;
      showSuccess(`Status updated to ${statusLabel}`);
    },
    onError: (error) => {
      showError(`Failed to update status: ${error instanceof Error ? error.message : 'Unknown error'}`);
    },
  });

  // Add note mutation
  const addNoteMutation = useMutation({
    mutationFn: (note: string) => api.addNote(anomalyId, note),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['anomaly', anomalyId] });
      setNoteText('');
      showSuccess('Note added successfully');
    },
    onError: (error) => {
      showError(`Failed to add note: ${error instanceof Error ? error.message : 'Unknown error'}`);
    },
  });

  // Learn from fix mutation
  const learnMutation = useMutation({
    mutationFn: (data: LearnFromFix) => api.learnFromFix(anomalyId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['remediations', anomalyId] });
    },
  });

  // State for action execution
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [messageDialogOpen, setMessageDialogOpen] = useState(false);
  const [messageText, setMessageText] = useState('');

  // Handle MobiControl actions
  const handleMobiControlAction = async (action: string) => {
    if (!anomaly?.device_id) return;

    const deviceId = anomaly.device_id;
    setActionLoading(action);

    try {
      let result;
      const reason = `Triggered from investigation #${anomalyId}`;

      switch (action) {
        case 'sync':
          result = await api.syncDevice(deviceId, reason);
          break;
        case 'lock':
          result = await api.lockDevice(deviceId, reason);
          break;
        case 'restart':
          result = await api.restartDevice(deviceId, reason);
          break;
        case 'locate':
          result = await api.locateDevice(deviceId);
          break;
        case 'clearCache':
          result = await api.clearDeviceCache(deviceId);
          break;
        case 'message':
          // Open message dialog instead
          setMessageDialogOpen(true);
          setActionLoading(null);
          return;
        default:
          console.warn(`Unknown action: ${action}`);
          setActionLoading(null);
          return;
      }

      // Log the action as a note and show success toast
      if (result?.success) {
        showSuccess(`${action.charAt(0).toUpperCase() + action.slice(1)} action completed`);
        addNoteMutation.mutate(`Executed MobiControl action: ${action} - ${result.message}`);
      }
    } catch (error) {
      console.error(`Failed to execute ${action}:`, error);
      showError(`Failed to execute ${action}: ${error instanceof Error ? error.message : 'Unknown error'}`);
      addNoteMutation.mutate(`Failed to execute MobiControl action: ${action} - ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setActionLoading(null);
    }
  };

  // Handle sending message
  const handleSendMessage = async () => {
    if (!anomaly?.device_id || !messageText.trim()) return;

    setActionLoading('message');
    try {
      const result = await api.sendMessageToDevice(anomaly.device_id, messageText.trim());
      if (result?.success) {
        showSuccess('Message sent to device');
        addNoteMutation.mutate(`Sent message to device: "${messageText.substring(0, 50)}..."`);
      }
      setMessageDialogOpen(false);
      setMessageText('');
    } catch (error) {
      console.error('Failed to send message:', error);
      showError(`Failed to send message: ${error instanceof Error ? error.message : 'Unknown error'}`);
      addNoteMutation.mutate(`Failed to send message: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setActionLoading(null);
    }
  };

  if (anomalyLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="relative w-16 h-16 mx-auto mb-4">
            <div className="absolute inset-0 rounded-full border-2 border-amber-500/20"></div>
            <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-amber-500 animate-spin"></div>
          </div>
          <p className="text-slate-400 font-mono text-sm">Loading investigation...</p>
        </div>
      </div>
    );
  }

  if (!anomaly) {
    return (
      <div className="flex flex-col items-center justify-center h-96">
        <div className="w-16 h-16 mb-4 rounded-full bg-red-500/20 flex items-center justify-center">
          <svg className="w-8 h-8 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </div>
        <p className="text-red-400 font-medium mb-4">Investigation not found</p>
        <Link to="/investigations" className="text-amber-400 hover:text-amber-300 text-sm">
          ← Back to Investigations
        </Link>
      </div>
    );
  }

  const severityKey = getSeverityKey(anomaly.anomaly_score);
  const severity = severityConfig[severityKey];
  const status = statusConfig[anomaly.status as keyof typeof statusConfig] || statusConfig.open;

  return (
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      {/* Breadcrumb */}
      <Breadcrumb
        items={[
          { label: 'Investigations', to: '/investigations' },
          { label: `Investigation #${anomaly.id}` },
        ]}
      />

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-white">Investigation #{anomaly.id}</h1>
            <span className={`px-2 py-1 text-xs font-bold rounded ${severity.bg} ${severity.color} border ${severity.border}`}>
              {severity.label}
            </span>
          </div>
          <p className="text-slate-500 text-sm mt-1">
            {anomaly.device_name || `Device #${anomaly.device_id}`} • Detected {formatDistanceToNowStrict(new Date(anomaly.timestamp))} ago
          </p>
        </div>

        {/* Status Update Dropdown */}
        <div className="relative">
          <button
            onClick={() => setShowActions(!showActions)}
            className="btn-stellar flex items-center gap-2"
          >
            Update Status
            <svg className={`w-4 h-4 transition-transform ${showActions ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          <AnimatePresence>
            {showActions && (
              <motion.div
                initial={{ opacity: 0, y: -10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -10, scale: 0.95 }}
                className="absolute right-0 top-full mt-2 w-64 stellar-card rounded-xl p-2 z-10"
              >
                {[
                  { status: 'investigating', label: 'Mark as Investigating', colorClass: 'hover:bg-orange-500/10 hover:text-orange-400' },
                  { status: 'resolved', label: 'Mark as Resolved', colorClass: 'hover:bg-emerald-500/10 hover:text-emerald-400' },
                  { status: 'false_positive', label: 'Mark as False Positive', colorClass: 'hover:bg-slate-700/50 hover:text-slate-300' },
                ].map((action) => (
                  <button
                    key={action.status}
                    onClick={() => resolveMutation.mutate({ status: action.status })}
                    disabled={anomaly.status === action.status || resolveMutation.isPending}
                    className={`w-full px-4 py-3 text-left text-sm font-medium rounded-lg transition-colors
                              text-slate-300 ${action.colorClass}
                              disabled:opacity-30 disabled:cursor-not-allowed`}
                  >
                    {action.label}
                  </button>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Status Banner */}
      <div className={`p-4 rounded-xl border ${status.bg} border-slate-700/50`}>
        <div className="flex items-center gap-4">
          <span className={`text-2xl ${status.color}`}>{status.icon}</span>
          <div>
            <p className={`text-lg font-bold ${status.color}`}>{status.label}</p>
            <p className="text-sm text-slate-400">
              Last updated {format(new Date(anomaly.updated_at), 'MMM d, HH:mm')}
            </p>
          </div>
          <div className="flex-1" />
          <div className="text-right">
            <p className="text-xs text-slate-500">Anomaly Score</p>
            <p className={`text-2xl font-bold font-mono ${severity.color}`}>
              {anomaly.anomaly_score.toFixed(4)}
            </p>
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex gap-2 border-b border-slate-800">
        {[
          { id: 'analysis', label: 'AI Analysis', icon: <BrainIcon /> },
          { id: 'telemetry', label: 'Telemetry', icon: <ChartIcon /> },
          { id: 'actions', label: 'Actions', icon: <WrenchIcon /> },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as typeof activeTab)}
            className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === tab.id
                ? 'border-amber-500 text-amber-400'
                : 'border-transparent text-slate-400 hover:text-white'
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Left Column - Main Content */}
        <div className="xl:col-span-2 space-y-6">
          {/* Analysis Tab */}
          {activeTab === 'analysis' && (
            <>
              {/* AI Root Cause Analysis */}
              <Card title={<span className="flex items-center gap-2"><BrainIcon /> Root Cause Analysis</span>}>
                <AIAnalysisPanel
                  hypothesis={aiAnalysis?.primary_hypothesis}
                  isLoading={aiLoading}
                  onRegenerate={() => refetchAI()}
                />
              </Card>

              {/* Feature Contributions */}
              <Card title={<span className="flex items-center gap-2"><ChartIcon /> Why This Anomaly Was Detected</span>}>
                <p className="text-sm text-slate-400 mb-4">
                  These metrics deviated most significantly from the device's baseline behavior:
                </p>
                <FeatureContributionChart contributions={investigation?.explanation?.feature_contributions || []} />

                {investigation?.explanation?.feature_contributions && investigation.explanation.feature_contributions.length > 0 && (
                  <div className="mt-6 pt-4 border-t border-slate-700/50">
                    <h4 className="text-sm font-medium text-slate-300 mb-3">Key Findings</h4>
                    <div className="space-y-2">
                      {investigation.explanation.feature_contributions.slice(0, 3).map((fc: FeatureContribution, i: number) => (
                        <div key={i} className="flex items-start gap-2 text-sm">
                          <span className="text-amber-400 mt-0.5">→</span>
                          <span className="text-slate-300">{fc.plain_text_explanation}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </Card>

              {/* Baseline Comparison */}
              <Card title={<span className="flex items-center gap-2"><TargetIcon /> Baseline Comparison</span>}>
                <p className="text-sm text-slate-400 mb-4">
                  Current values compared to 30-day rolling average:
                </p>
                <BaselineComparisonViz metrics={investigation?.baseline_comparison?.metrics || []} />
              </Card>
            </>
          )}

          {/* Telemetry Tab */}
          {activeTab === 'telemetry' && <TelemetryTab anomaly={anomaly} />}

          {/* Actions Tab */}
          {activeTab === 'actions' && (
            <>
              {/* MobiControl Actions */}
              <Card title={<span className="flex items-center gap-2"><WrenchIcon /> SOTI MobiControl Actions</span>}>
                <p className="text-sm text-slate-400 mb-4">
                  Execute remote actions on this device via MobiControl:
                </p>
                <MobiControlActions deviceId={anomaly.device_id} onAction={handleMobiControlAction} loadingAction={actionLoading} />

                {/* Send Message Dialog */}
                {messageDialogOpen && (
                  <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
                    <div className="bg-slate-800 rounded-xl p-6 w-full max-w-md border border-slate-700">
                      <h3 className="text-lg font-semibold text-white mb-4">Send Message to Device</h3>
                      <textarea
                        value={messageText}
                        onChange={(e) => setMessageText(e.target.value)}
                        placeholder="Enter message to send to device..."
                        className="w-full h-32 p-3 bg-slate-900 border border-slate-700 rounded-lg text-white placeholder-slate-500 resize-none focus:outline-none focus:ring-2 focus:ring-amber-500"
                      />
                      <div className="flex justify-end gap-3 mt-4">
                        <button
                          onClick={() => { setMessageDialogOpen(false); setMessageText(''); }}
                          className="px-4 py-2 text-slate-400 hover:text-white transition-colors"
                        >
                          Cancel
                        </button>
                        <button
                          onClick={handleSendMessage}
                          disabled={!messageText.trim() || actionLoading === 'message'}
                          className="px-4 py-2 bg-amber-500 text-white rounded-lg hover:bg-amber-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          {actionLoading === 'message' ? 'Sending...' : 'Send Message'}
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </Card>

              {/* Remediation Suggestions */}
              <Card title="AI-Powered Remediation Suggestions">
                <div className="space-y-3">
                  {remediations && remediations.length > 0 ? (
                    remediations.map((r: RemediationSuggestion) => (
                      <RemediationCard
                        key={r.remediation_id}
                        remediation={r}
                        onApply={() => handleMobiControlAction('remediation')}
                        onMarkSuccess={() => learnMutation.mutate({ remediation_description: r.title, tags: [] })}
                      />
                    ))
                  ) : (
                    <p className="text-sm text-slate-500 text-center py-4">
                      No remediation suggestions available
                    </p>
                  )}
                </div>
              </Card>
            </>
          )}
        </div>

        {/* Right Column - Sidebar */}
        <div className="space-y-6">
          {/* Cost Impact */}
          <Card title={<span className="flex items-center gap-2"><DollarIcon /> Cost Impact</span>}>
            <CostImpactCard costImpact={costImpact} isLoading={costImpactLoading} />
          </Card>

          {/* Similar Cases */}
          <Card title="Similar Cases">
            <SimilarCasesList cases={similarCases || []} />
          </Card>

          {/* Investigation Notes */}
          <Card title="Investigation Log">
            <div className="space-y-3 max-h-[300px] overflow-y-auto mb-4">
              {anomaly.investigation_notes && anomaly.investigation_notes.length > 0 ? (
                anomaly.investigation_notes.map((note: InvestigationNote) => (
                  <div
                    key={note.id}
                    className="relative pl-4 border-l-2 border-amber-500/30"
                  >
                    <div className="absolute left-0 top-1 w-2 h-2 -translate-x-[5px] rounded-full bg-amber-500" />
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-medium text-amber-400">{note.user}</span>
                      <span className="text-[10px] text-slate-600 font-mono">
                        {format(new Date(note.created_at), 'MMM d, HH:mm')}
                      </span>
                    </div>
                    <p className="text-sm text-slate-300">{note.note}</p>
                  </div>
                ))
              ) : (
                <p className="text-sm text-slate-500 text-center py-4">No notes yet</p>
              )}
            </div>

            {/* Add Note Form */}
            <div className="pt-4 border-t border-slate-700/50">
              <textarea
                value={noteText}
                onChange={(e) => setNoteText(e.target.value)}
                placeholder="Add investigation note..."
                className="input-stellar w-full resize-none"
                rows={3}
                maxLength={1000}
              />
              <div className="flex justify-between items-center mt-2">
                <span className="text-xs text-slate-500">{noteText.length}/1000</span>
                <button
                  onClick={() => noteText.trim() && addNoteMutation.mutate(noteText)}
                  disabled={!noteText.trim() || noteText.trim().length < 5 || addNoteMutation.isPending}
                  className="btn-ghost text-sm disabled:opacity-30"
                >
                  {addNoteMutation.isPending ? 'Adding...' : 'Add Note'}
                </button>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </motion.div>
  );
}

export default InvestigationDetail;
