import { useState, useRef, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { AnimatePresence, motion } from 'framer-motion';
import { IsolationForestViz } from '../components/IsolationForestViz';
import { Card } from '../components/Card';
import { useMockMode } from '../hooks/useMockMode';
import { formatDistanceToNowStrict, format } from 'date-fns';
import { getDOMPathInfo, formatDOMPathInfo } from '../utils/domPath';
import { Link } from 'react-router-dom';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

export default function ModelStatus() {
    const { mockMode } = useMockMode();
    const [activeStage, setActiveStage] = useState<string>('model');
    
    // Refs for DOM path tracking
    const dataCollectionRef = useRef<HTMLButtonElement>(null);
    const preprocessingRef = useRef<HTMLButtonElement>(null);
    const isolationForestRef = useRef<HTMLButtonElement>(null);
    const alertingRef = useRef<HTMLButtonElement>(null);
    const detailViewRef = useRef<HTMLDivElement>(null);
    
    // Log DOM paths on mount and when active stage changes (development only)
    useEffect(() => {
        if (import.meta.env.DEV) {
            const logPaths = () => {
                const refs = [
                    { name: 'Data Collection', ref: dataCollectionRef },
                    { name: 'Preprocessing', ref: preprocessingRef },
                    { name: 'Isolation Forest', ref: isolationForestRef },
                    { name: 'Alerting', ref: alertingRef },
                    { name: 'Detail View', ref: detailViewRef },
                ];

                refs.forEach(({ name, ref }) => {
                    if (ref.current) {
                        const info = getDOMPathInfo(ref.current);
                        if (info) {
                            console.log(`\n=== ${name} ===`);
                            console.log(formatDOMPathInfo(info));
                        }
                    }
                });
            };

            // Log after a short delay to ensure DOM is rendered
            const timeout = setTimeout(logPaths, 500);
            return () => clearTimeout(timeout);
        }
    }, [activeStage]);

    const { data: modelStats } = useQuery({
        queryKey: ['isolationForest'],
        queryFn: () => api.getIsolationForestStats(),
    });
    const {
        data: baselineSuggestions,
        isLoading: baselineLoading,
        isError: baselineError,
    } = useQuery({
        queryKey: ['baselines', 'suggestions', 30],
        queryFn: () => api.getBaselineSuggestions(undefined, 30),
        refetchInterval: 60000,
    });
    const { data: llmConfig, isLoading: llmLoading, isError: llmError } = useQuery({
        queryKey: ['llm', 'config'],
        queryFn: () => api.getLLMConfig(),
        refetchInterval: 60000,
    });

    // Automation status for detection metrics
    const { data: automationStatus } = useQuery({
        queryKey: ['automation', 'status'],
        queryFn: () => api.getAutomationStatus(),
        refetchInterval: 30000,
    });

    // Dashboard trends for visualization
    const { data: trends } = useQuery({
        queryKey: ['dashboard', 'trends', 7],
        queryFn: () => api.getDashboardTrends({ days: 7 }),
        refetchInterval: 60000,
    });

    // Dashboard stats (available for future use)
    useQuery({
        queryKey: ['dashboard', 'stats'],
        queryFn: () => api.getDashboardStats(),
        refetchInterval: 30000,
    });

    const baselineCount = baselineSuggestions?.length || 0;
    const baselineStatus = baselineError
        ? 'Unavailable'
        : baselineLoading
            ? 'Checking'
            : baselineCount > 0
                ? 'Drift detected'
                : 'Stable';
    const baselineTopFeatures = baselineSuggestions
        ? Array.from(new Set(baselineSuggestions.map((suggestion) => suggestion.feature))).slice(0, 3)
        : [];
    const modelStatus = modelStats?.total_predictions
        ? 'Running'
        : 'Idle';
    const lastRetrain = modelStats?.feedback_stats?.last_retrain
        ? formatDistanceToNowStrict(new Date(modelStats.feedback_stats.last_retrain), { addSuffix: false })
        : null;

    const pipelineStages = [
        {
            id: 'source',
            title: 'Data Collection',
            icon: (
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
                </svg>
            ),
            status: 'active',
            details: {
                description: 'Telemetry data is collected from SOTI MobiControl managed devices.',
                metrics: [
                    { label: 'Source', value: mockMode ? 'Mock Generator' : 'SOTI MobiControl API' },
                    { label: 'Frequency', value: '15 minutes' },
                    { label: 'Protocol', value: 'HTTPS / REST' },
                ]
            }
        },
        {
            id: 'processing',
            title: 'Preprocessing',
            icon: (
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
                </svg>
            ),
            status: 'active',
            details: {
                description: 'Raw data is cleaned, normalized, and feature engineered.',
                metrics: [
                    { label: 'Scaling', value: 'StandardScaler (Z-Score)' },
                    { label: 'Missing Data', value: 'Imputation (Median)' },
                    { label: 'Categorical', value: 'One-Hot Encoding' },
                ]
            }
        },
        {
            id: 'model',
            title: 'Isolation Forest',
            icon: (
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                </svg>
            ),
            status: 'active',
            details: {
                description: 'Unsupervised learning algorithm that isolates anomalies by randomly partitioning the data.',
                metrics: [
                    { label: 'Algorithm', value: 'Isolation Forest' },
                    { label: 'Contamination', value: `${(modelStats?.config?.contamination || 0.05) * 100}% ` },
                    { label: 'Estimators', value: modelStats?.config?.n_estimators || 100 },
                    { label: 'Features', value: modelStats?.config?.feature_count || 8 },
                ]
            }
        },
        {
            id: 'alerting',
            title: 'Alerting',
            icon: (
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
                </svg>
            ),
            status: 'active',
            details: {
                description: 'Anomalies are scored (-1.0 to 1.0) and alerts are triggered based on severity thresholds.',
                metrics: [
                    { label: 'Critical Threshold', value: '> 0.70' },
                    { label: 'Warning Threshold', value: '> 0.50' },
                    { label: 'Baseline', value: 'Dynamic (Store/Region)' },
                ]
            }
        }
    ];

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="space-y-6"
        >
            <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                <div>
                    <h1 className="text-3xl font-bold text-white">Model Status</h1>
                    <p className="text-slate-400 mt-1">ML pipeline architecture, data flow, and model configuration</p>
                </div>
                <Link to="/system" className="btn-ghost flex items-center gap-2 w-fit">
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    </svg>
                    System Settings
                </Link>
            </div>

            {/* Detection Performance Overview */}
            <div className="stellar-card rounded-xl p-6 border border-amber-500/20 bg-gradient-to-br from-amber-500/5 to-transparent">
                <div className="flex items-center gap-3 mb-6">
                    <div className="w-12 h-12 rounded-xl bg-amber-500/20 flex items-center justify-center">
                        <svg className="w-6 h-6 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                    </div>
                    <div>
                        <h2 className="text-xl font-bold text-white">Anomaly Detection Performance</h2>
                        <p className="text-sm text-slate-400">Real-time ML model effectiveness and detection metrics</p>
                    </div>
                </div>

                <div className="grid grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
                    {/* Detection Rate */}
                    <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/50">
                        <div className="flex items-center gap-2 mb-2">
                            <div className="w-2 h-2 rounded-full bg-amber-400" />
                            <span className="text-xs text-slate-500">Detection Rate</span>
                        </div>
                        <p className="text-2xl font-bold font-mono text-amber-400">
                            {automationStatus?.last_scoring_result
                                ? `${((automationStatus.last_scoring_result.anomaly_rate || 0) * 100).toFixed(1)}%`
                                : modelStats ? `${(modelStats.anomaly_rate * 100).toFixed(1)}%` : '—'}
                        </p>
                        <p className="text-[10px] text-slate-500 mt-1">anomalies / total scored</p>
                    </div>

                    {/* Total Scored */}
                    <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/50">
                        <div className="flex items-center gap-2 mb-2">
                            <div className="w-2 h-2 rounded-full bg-indigo-400" />
                            <span className="text-xs text-slate-500">Last Run Scored</span>
                        </div>
                        <p className="text-2xl font-bold font-mono text-white">
                            {automationStatus?.last_scoring_result?.total_scored || 0}
                        </p>
                        <p className="text-[10px] text-slate-500 mt-1">devices analyzed</p>
                    </div>

                    {/* Anomalies Detected */}
                    <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/50">
                        <div className="flex items-center gap-2 mb-2">
                            <div className={`w-2 h-2 rounded-full ${(automationStatus?.last_scoring_result?.anomalies_detected || 0) > 0 ? 'bg-red-400' : 'bg-emerald-400'}`} />
                            <span className="text-xs text-slate-500">Anomalies Found</span>
                        </div>
                        <p className={`text-2xl font-bold font-mono ${(automationStatus?.last_scoring_result?.anomalies_detected || 0) > 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                            {automationStatus?.last_scoring_result?.anomalies_detected || 0}
                        </p>
                        <p className="text-[10px] text-slate-500 mt-1">in last detection run</p>
                    </div>

                    {/* False Positive Rate */}
                    <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/50">
                        <div className="flex items-center gap-2 mb-2">
                            <div className="w-2 h-2 rounded-full bg-cyan-400" />
                            <span className="text-xs text-slate-500">False Positive Rate</span>
                        </div>
                        <p className="text-2xl font-bold font-mono text-cyan-400">
                            {automationStatus?.false_positive_rate !== undefined
                                ? `${(automationStatus.false_positive_rate * 100).toFixed(1)}%`
                                : '0.0%'}
                        </p>
                        <p className="text-[10px] text-slate-500 mt-1">improving with feedback</p>
                    </div>

                    {/* Automation Uptime */}
                    <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/50">
                        <div className="flex items-center gap-2 mb-2">
                            <div className={`w-2 h-2 rounded-full ${automationStatus?.is_running ? 'bg-emerald-400 animate-pulse' : 'bg-slate-500'}`} />
                            <span className="text-xs text-slate-500">ML Engine</span>
                        </div>
                        <p className={`text-2xl font-bold font-mono ${automationStatus?.is_running ? 'text-emerald-400' : 'text-slate-400'}`}>
                            {automationStatus?.is_running ? 'ACTIVE' : 'IDLE'}
                        </p>
                        <p className="text-[10px] text-slate-500 mt-1">
                            {automationStatus?.uptime_seconds
                                ? `uptime: ${Math.floor(automationStatus.uptime_seconds / 60)}m`
                                : 'not running'}
                        </p>
                    </div>
                </div>

                {/* 7-Day Trend Chart */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/50">
                        <h3 className="text-sm font-semibold text-slate-300 mb-4">7-Day Anomaly Trend</h3>
                        <div className="h-40">
                            {trends && trends.length > 0 ? (
                                <ResponsiveContainer width="100%" height="100%">
                                    <AreaChart data={trends}>
                                        <defs>
                                            <linearGradient id="anomalyGradient" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="0%" stopColor="#f59e0b" stopOpacity={0.4} />
                                                <stop offset="100%" stopColor="#f59e0b" stopOpacity={0} />
                                            </linearGradient>
                                        </defs>
                                        <XAxis
                                            dataKey="date"
                                            tickFormatter={(v) => format(new Date(v), 'EEE')}
                                            stroke="#475569"
                                            fontSize={10}
                                            tickLine={false}
                                            axisLine={false}
                                        />
                                        <YAxis hide />
                                        <Tooltip
                                            contentStyle={{
                                                backgroundColor: 'rgba(14, 17, 23, 0.95)',
                                                border: '1px solid rgba(245, 166, 35, 0.2)',
                                                borderRadius: '8px',
                                                fontSize: '11px',
                                            }}
                                            labelFormatter={(v) => format(new Date(v), 'MMM d, yyyy')}
                                            formatter={(value: number) => [value, 'Anomalies']}
                                        />
                                        <Area
                                            type="monotone"
                                            dataKey="anomaly_count"
                                            stroke="#f59e0b"
                                            strokeWidth={2}
                                            fill="url(#anomalyGradient)"
                                        />
                                    </AreaChart>
                                </ResponsiveContainer>
                            ) : (
                                <div className="h-full flex items-center justify-center text-slate-500 text-sm">
                                    No trend data available yet
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Model Learning Progress */}
                    <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/50">
                        <h3 className="text-sm font-semibold text-slate-300 mb-4">Model Learning Progress</h3>
                        <div className="space-y-4">
                            <div>
                                <div className="flex justify-between text-xs mb-1">
                                    <span className="text-slate-400">Total Predictions Made</span>
                                    <span className="text-white font-mono">{modelStats?.total_predictions?.toLocaleString() || 0}</span>
                                </div>
                                <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                                    <motion.div
                                        className="h-full bg-gradient-to-r from-amber-500 to-orange-500 rounded-full"
                                        initial={{ width: 0 }}
                                        animate={{ width: `${Math.min((modelStats?.total_predictions || 0) / 1000 * 100, 100)}%` }}
                                        transition={{ duration: 1, ease: 'easeOut' }}
                                    />
                                </div>
                                <p className="text-[10px] text-slate-500 mt-1">Goal: 1,000 predictions for stable baseline</p>
                            </div>

                            <div>
                                <div className="flex justify-between text-xs mb-1">
                                    <span className="text-slate-400">Feedback Collected</span>
                                    <span className="text-white font-mono">{modelStats?.feedback_stats?.total_feedback || 0}</span>
                                </div>
                                <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                                    <motion.div
                                        className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full"
                                        initial={{ width: 0 }}
                                        animate={{ width: `${Math.min((modelStats?.feedback_stats?.total_feedback || 0) / 50 * 100, 100)}%` }}
                                        transition={{ duration: 1, ease: 'easeOut' }}
                                    />
                                </div>
                                <p className="text-[10px] text-slate-500 mt-1">Goal: 50 feedback items for retraining</p>
                            </div>

                            <div>
                                <div className="flex justify-between text-xs mb-1">
                                    <span className="text-slate-400">Model Confidence</span>
                                    <span className="text-emerald-400 font-mono">
                                        {modelStats?.total_predictions && modelStats.total_predictions > 100
                                            ? 'High'
                                            : modelStats?.total_predictions && modelStats.total_predictions > 10
                                                ? 'Medium'
                                                : 'Learning...'}
                                    </span>
                                </div>
                                <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                                    <motion.div
                                        className="h-full bg-gradient-to-r from-emerald-500 to-green-500 rounded-full"
                                        initial={{ width: 0 }}
                                        animate={{ width: `${Math.min((modelStats?.total_predictions || 0) / 100 * 100, 100)}%` }}
                                        transition={{ duration: 1, ease: 'easeOut' }}
                                    />
                                </div>
                                <p className="text-[10px] text-slate-500 mt-1">Improves with more predictions and user feedback</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
                <Card title="Model Status" accent="orange">
                    <div className="space-y-3">
                        <div className="flex items-center justify-between text-sm">
                            <span className="text-slate-500">State</span>
                            <span className="font-mono text-slate-200">{modelStatus}</span>
                        </div>
                        <div className="flex items-center justify-between text-sm">
                            <span className="text-slate-500">Anomaly Rate</span>
                            <span className="font-mono text-slate-200">
                                {modelStats ? `${(modelStats.anomaly_rate * 100).toFixed(2)}%` : '—'}
                            </span>
                        </div>
                        <div className="flex items-center justify-between text-sm">
                            <span className="text-slate-500">Predictions</span>
                            <span className="font-mono text-slate-200">
                                {modelStats ? modelStats.total_predictions.toLocaleString() : '—'}
                            </span>
                        </div>
                        <div className="flex items-center justify-between text-sm">
                            <span className="text-slate-500">Features</span>
                            <span className="font-mono text-slate-200">
                                {modelStats?.config?.feature_count ?? '—'}
                            </span>
                        </div>
                        <div className="flex items-center justify-between text-sm">
                            <span className="text-slate-500">Last Retrain</span>
                            <span className="font-mono text-slate-200">
                                {lastRetrain ? `${lastRetrain} ago` : 'N/A'}
                            </span>
                        </div>
                    </div>
                </Card>

                <Card title="Baseline Health" accent="purple">
                    <div className="space-y-3">
                        <div className="flex items-center justify-between text-sm">
                            <span className="text-slate-500">Status</span>
                            <span className="font-mono text-slate-200">{baselineStatus}</span>
                        </div>
                        <div className="flex items-center justify-between text-sm">
                            <span className="text-slate-500">Suggestions (30d)</span>
                            <span className="font-mono text-slate-200">
                                {baselineLoading ? '…' : baselineCount}
                            </span>
                        </div>
                        <div className="text-xs text-slate-500">
                            {baselineTopFeatures.length > 0 ? (
                                <span className="font-mono text-slate-300">
                                    Top drift: {baselineTopFeatures.join(', ')}
                                </span>
                            ) : (
                                <span>No drift indicators detected.</span>
                            )}
                        </div>
                        {baselineError && (
                            <p className="text-xs text-red-400">Baseline service unavailable.</p>
                        )}
                    </div>
                </Card>

                <Card title="LLM Explainability" accent="cyan">
                    <div className="space-y-3">
                        <div className="flex items-center justify-between text-sm">
                            <span className="text-slate-500">Provider</span>
                            <span className="font-mono text-slate-200">{llmConfig?.provider || '—'}</span>
                        </div>
                        <div className="flex items-center justify-between text-sm">
                            <span className="text-slate-500">Active Model</span>
                            <span className="font-mono text-slate-200">
                                {llmConfig?.active_model || llmConfig?.model_name || '—'}
                            </span>
                        </div>
                        <div className="flex items-center justify-between text-sm">
                            <span className="text-slate-500">Connection</span>
                            <span className={`font-mono ${llmConfig?.is_connected ? 'text-emerald-400' : 'text-red-400'}`}>
                                {llmLoading ? 'Checking' : llmConfig?.is_connected ? 'Connected' : 'Offline'}
                            </span>
                        </div>
                        <div className="flex items-center justify-between text-xs text-slate-500">
                            <span>Endpoint</span>
                            <span className="font-mono text-slate-400 truncate max-w-[180px]" title={llmConfig?.base_url}>
                                {llmConfig?.base_url || '—'}
                            </span>
                        </div>
                        {llmError && (
                            <p className="text-xs text-red-400">LLM status unavailable.</p>
                        )}
                    </div>
                </Card>
            </div>

            <div className="w-full space-y-6">
                {/* Horizontal Navigation: Pipeline Stages */}
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 w-full">
                    {pipelineStages.map((stage) => {
                        const isActive = activeStage === stage.id;
                        const getStageRef = () => {
                            if (stage.id === 'source') return dataCollectionRef;
                            if (stage.id === 'processing') return preprocessingRef;
                            if (stage.id === 'model') return isolationForestRef;
                            if (stage.id === 'alerting') return alertingRef;
                            return null;
                        };
                        const stageRef = getStageRef();
                        
                        return (
                            <button
                                key={stage.id}
                                ref={stageRef}
                                onClick={() => {
                                    setActiveStage(stage.id);
                                    // Log DOM path on click
                                    if (stageRef?.current) {
                                        const info = getDOMPathInfo(stageRef.current);
                                        if (info) {
                                            console.log(`\n=== ${stage.title} ===`);
                                            console.log(formatDOMPathInfo(info));
                                        }
                                    }
                                }}
                                className={`relative flex items-center justify-center gap-3 px-4 py-3 rounded-xl border transition-all duration-200 w-full ${isActive
                                    ? 'bg-amber-500/10 border-amber-500/50 shadow-[0_0_15px_rgba(245,158,11,0.15)]'
                                    : 'bg-slate-800/50 border-slate-700/50 hover:bg-slate-800 hover:border-slate-600'
                                    }`}
                            >
                                <div className={`p-2 rounded-lg ${isActive ? 'bg-amber-500 text-white' : 'bg-slate-700 text-slate-400'}`}>
                                    {stage.icon}
                                </div>
                                <div className="text-left">
                                    <h3 className={`font-semibold text-sm ${isActive ? 'text-white' : 'text-slate-300'}`}>{stage.title}</h3>
                                    <p className="text-xs text-slate-500">{stage.details.metrics[0].value}</p>
                                </div>
                                {isActive && (
                                    <div className="absolute -top-1 -right-1 w-3 h-3 rounded-full bg-amber-500 shadow-[0_0_10px_rgba(245,158,11,0.8)]" />
                                )}
                            </button>
                        );
                    })}
                </div>

                {/* Content View */}
                <div ref={detailViewRef} className="w-full">
                    <Card className="w-full h-full relative overflow-hidden">
                        <div className="absolute top-0 right-0 p-32 bg-amber-500/5 blur-3xl rounded-full pointer-events-none transform translate-x-1/2 -translate-y-1/2" />

                        <AnimatePresence mode="wait">
                            {pipelineStages.filter(s => s.id === activeStage).map((stage) => (
                                <motion.div
                                    key={stage.id}
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: -20 }}
                                    transition={{ duration: 0.2 }}
                                    className="space-y-8 relative z-10"
                                >
                                    <div>
                                        <div className="flex items-center gap-3 mb-4">
                                            <div className="p-2 bg-slate-800 rounded-lg text-amber-400 border border-slate-700">
                                                {stage.icon}
                                            </div>
                                            <h2 className="text-2xl font-bold text-white">{stage.title}</h2>
                                        </div>
                                        <p className="text-lg text-slate-300 leading-relaxed max-w-2xl">
                                            {stage.details.description}
                                        </p>
                                    </div>

                                    {stage.id === 'model' ? (
                                        <IsolationForestViz />
                                    ) : (
                                        <>
                                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                                {stage.details.metrics.map((metric, idx) => (
                                                    <div key={idx} className="p-4 bg-slate-900/50 rounded-lg border border-slate-800">
                                                        <p className="text-sm text-slate-500 mb-1">{metric.label}</p>
                                                        <p className="text-lg font-mono font-medium text-white">{metric.value}</p>
                                                    </div>
                                                ))}
                                            </div>

                                            <div className="p-5 rounded-xl bg-slate-800/30 border border-slate-700/50">
                                                <h4 className="text-sm font-bold text-white uppercase tracking-wider mb-3 flex items-center gap-2">
                                                    <svg className="w-4 h-4 text-sky-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                                                    How it works
                                                </h4>
                                                <p className="text-sm text-slate-400">
                                                    {stage.id === 'source' && "The system polls the SOTI MobiControl API for device inventory, battery status, storage, and network usage. This data is augmented with 'Store' and 'Region' metadata provided by the Enterprise Resource Planning (ERP) link."}
                                                    {stage.id === 'processing' && "Continuous numerical features (e.g. Battery Drop) are scaled using Z-Score normalization. Categorical features like 'Device Model' are One-Hot encoded. Missing values are imputed using the localized median of the peer group to prevent skew."}
                                                    {stage.id === 'alerting' && "Raw anomaly scores are converted to a normalized probability. If the probability exceeds the dynamic baseline for that specific location/model group, an alert is generated. This reduces false positives in high-variance environments."}
                                                </p>
                                            </div>
                                        </>
                                    )}
                                </motion.div>
                            ))}
                        </AnimatePresence>
                    </Card>
                </div>
            </div>
        </motion.div>
    );
}
