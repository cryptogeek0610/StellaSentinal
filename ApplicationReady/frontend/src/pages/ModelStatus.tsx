
import { useMemo, useState, useRef, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { AnimatePresence, motion } from 'framer-motion';
import { IsolationForestViz } from '../components/IsolationForestViz';
import { Card } from '../components/Card';
import { useMockMode } from '../hooks/useMockMode';
import { ConnectionCards } from '../components/settings/ConnectionCards';
import { formatDistanceToNowStrict } from 'date-fns';
import { getDOMPathInfo, formatDOMPathInfo } from '../utils/domPath';

export default function ModelStatus() {
    const { mockMode } = useMockMode();
    const [activeStage, setActiveStage] = useState<string>('model');
    
    // Refs for DOM path tracking
    const dataCollectionRef = useRef<HTMLButtonElement>(null);
    const preprocessingRef = useRef<HTMLButtonElement>(null);
    const isolationForestRef = useRef<HTMLButtonElement>(null);
    const alertingRef = useRef<HTMLButtonElement>(null);
    const detailViewRef = useRef<HTMLDivElement>(null);
    
    // Log DOM paths on mount and when active stage changes
    useEffect(() => {
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
    }, [activeStage]);

    const { data: modelStats } = useQuery({
        queryKey: ['isolationForest'],
        queryFn: () => api.getIsolationForestStats(),
    });
    const { data: connections, isLoading: connectionsLoading } = useQuery({
        queryKey: ['dashboard', 'connections'],
        queryFn: () => api.getConnectionStatus(),
        refetchInterval: 30000,
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

    const connectedCount = useMemo(() => {
        if (!connections) {
            return 0;
        }
        return [connections.dw_sql, connections.mc_sql, connections.mobicontrol_api, connections.llm].filter(
            (service) => service.connected
        ).length;
    }, [connections]);
    const systemStatus = connectionsLoading
        ? 'CHECKING'
        : connectedCount === 4
            ? 'OPERATIONAL'
            : connectedCount >= 2
                ? 'DEGRADED'
                : 'OFFLINE';
    const systemStatusClass = connectionsLoading
        ? 'bg-slate-800 text-slate-300 border-slate-700'
        : connectedCount === 4
            ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40'
            : connectedCount >= 2
                ? 'bg-amber-500/20 text-amber-400 border-amber-500/40'
                : 'bg-red-500/20 text-red-400 border-red-500/40';

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
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold text-white mb-2">Anomaly Detection Status</h1>
                    <p className="text-slate-400">System architecture, data flow, and model configuration.</p>
                </div>
                <div className={`flex items-center gap-2 px-3 py-1 rounded-full border ${systemStatusClass}`}>
                    <div className={`w-2 h-2 rounded-full ${connectedCount === 4 ? 'bg-emerald-400' : connectedCount >= 2 ? 'bg-amber-400' : 'bg-red-400'} ${connectionsLoading ? 'animate-pulse' : ''}`}></div>
                    <span className="text-xs font-mono">{systemStatus}</span>
                    {connections && (
                        <span className="text-[10px] text-slate-400">({connectedCount}/4)</span>
                    )}
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

            <section className="space-y-3">
                <div className="flex items-center justify-between">
                    <h2 className="text-xl font-bold text-white">Backend Connections</h2>
                    <span className="text-xs text-slate-500">
                        {connections?.last_checked
                            ? `Last checked ${formatDistanceToNowStrict(new Date(connections.last_checked))} ago`
                            : 'Awaiting status'}
                    </span>
                </div>
                <ConnectionCards connections={connections} isChecking={connectionsLoading} />
            </section>

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
