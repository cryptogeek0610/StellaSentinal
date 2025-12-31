/**
 * LLM Settings Component
 *
 * Configure AI model service and select models for anomaly explanations.
 * Extracted from System.tsx for better maintainability.
 */

import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { motion, AnimatePresence } from 'framer-motion';
import { api } from '../../api/client';
import { Card } from '../Card';
import type { LLMConfig } from '../../types/anomaly';

// Provider display configuration
const PROVIDER_CONFIG: Record<string, { name: string; color: string; icon: JSX.Element }> = {
  ollama: {
    name: 'Ollama',
    color: 'text-emerald-400',
    icon: (
      <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" />
      </svg>
    ),
  },
  lmstudio: {
    name: 'LM Studio',
    color: 'text-purple-400',
    icon: (
      <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
      </svg>
    ),
  },
  azure: {
    name: 'Azure OpenAI',
    color: 'text-blue-400',
    icon: (
      <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
        <path d="M22.379 23.343a.96.96 0 0 1-1.065.209l-6.93-2.85a.96.96 0 0 1-.605-.893v-5.668a.96.96 0 0 1 .605-.893l6.93-2.85a.96.96 0 0 1 1.352.884v11.178a.96.96 0 0 1-.287.883zM5.08 8.505l6.93-2.85a.96.96 0 0 1 1.352.884v11.178a.96.96 0 0 1-1.352.884l-6.93-2.85a.96.96 0 0 1-.605-.893V9.398a.96.96 0 0 1 .605-.893zM7.467.448l6.93 2.85a.96.96 0 0 1 0 1.776l-6.93 2.85a.96.96 0 0 1-.735 0l-6.93-2.85a.96.96 0 0 1 0-1.776l6.93-2.85a.96.96 0 0 1 .735 0z" />
      </svg>
    ),
  },
  openai: {
    name: 'OpenAI',
    color: 'text-green-400',
    icon: (
      <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
        <path d="M22.2819 9.8211a5.9847 5.9847 0 0 0-.5157-4.9108 6.0462 6.0462 0 0 0-6.5098-2.9A6.0651 6.0651 0 0 0 4.9807 4.1818a5.9847 5.9847 0 0 0-3.9977 2.9 6.0462 6.0462 0 0 0 .7427 7.0966 5.98 5.98 0 0 0 .511 4.9107 6.051 6.051 0 0 0 6.5146 2.9001A5.9847 5.9847 0 0 0 13.2599 24a6.0557 6.0557 0 0 0 5.7718-4.2058 5.9894 5.9894 0 0 0 3.9977-2.9001 6.0557 6.0557 0 0 0-.7475-7.0729zm-9.022 12.6081a4.4755 4.4755 0 0 1-2.8764-1.0408l.1419-.0804 4.7783-2.7582a.7948.7948 0 0 0 .3927-.6813v-6.7369l2.02 1.1686a.071.071 0 0 1 .038.052v5.5826a4.504 4.504 0 0 1-4.4945 4.4944zm-9.6607-4.1254a4.4708 4.4708 0 0 1-.5346-3.0137l.142.0852 4.783 2.7582a.7712.7712 0 0 0 .7806 0l5.8428-3.3685v2.3324a.0804.0804 0 0 1-.0332.0615L9.74 19.9502a4.4992 4.4992 0 0 1-6.1408-1.6464zM2.3408 7.8956a4.485 4.485 0 0 1 2.3655-1.9728V11.6a.7664.7664 0 0 0 .3879.6765l5.8144 3.3543-2.0201 1.1685a.0757.0757 0 0 1-.071 0l-4.8303-2.7865A4.504 4.504 0 0 1 2.3408 7.8956zm16.0993 3.8558L12.6 8.3829l2.02-1.1638a.0757.0757 0 0 1 .071 0l4.8303 2.7913a4.4944 4.4944 0 0 1-.6765 8.1042v-5.6772a.79.79 0 0 0-.407-.667zm2.0107-3.0231l-.142-.0852-4.7735-2.7818a.7759.7759 0 0 0-.7854 0L9.409 9.2297V6.8974a.0662.0662 0 0 1 .0284-.0615l4.8303-2.7866a4.4992 4.4992 0 0 1 6.6802 4.66zM8.3065 12.863l-2.02-1.1638a.0804.0804 0 0 1-.038-.0567V6.0742a4.4992 4.4992 0 0 1 7.3757-3.4537l-.142.0805L8.704 5.459a.7948.7948 0 0 0-.3927.6813zm1.0976-2.3654l2.602-1.4998 2.6069 1.4998v2.9994l-2.5974 1.4997-2.6067-1.4997Z" />
      </svg>
    ),
  },
  unknown: {
    name: 'Custom',
    color: 'text-slate-400',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    ),
  },
};

// Preset URL configurations
const URL_PRESETS = [
  { url: 'http://ollama:11434', label: 'Ollama (Docker)', color: 'indigo' },
  { url: 'http://host.docker.internal:1234', label: 'LM Studio (Host)', color: 'purple' },
  { url: 'http://host.docker.internal:11434', label: 'Ollama (Host)', color: 'emerald' },
];

// Format model size for display
function formatModelSize(size: string | null | undefined): string {
  if (!size) return '';
  if (size.match(/^\d+\.?\d*\s*(GB|MB|KB|TB)$/i)) return size;
  const bytes = parseInt(size);
  if (!isNaN(bytes)) {
    if (bytes >= 1e12) return `${(bytes / 1e12).toFixed(2)} TB`;
    if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
    if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(2)} MB`;
    if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(2)} KB`;
    return `${bytes} B`;
  }
  return size;
}

// Loading spinner component
function LoadingSpinner({ size = 'md' }: { size?: 'sm' | 'md' | 'lg' }) {
  const sizeClasses = { sm: 'h-4 w-4', md: 'h-6 w-6', lg: 'h-8 w-8' };
  return (
    <svg className={`animate-spin ${sizeClasses[size]}`} viewBox="0 0 24 24" aria-hidden="true">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
    </svg>
  );
}

export function LLMSettings() {
  const queryClient = useQueryClient();
  const [customUrl, setCustomUrl] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [showPullModal, setShowPullModal] = useState(false);
  const [pullModelName, setPullModelName] = useState('');
  const [urlError, setUrlError] = useState<string | null>(null);

  // Fetch LLM config
  const { data: llmConfig, isLoading: isLoadingConfig, error: configError } = useQuery<LLMConfig>({
    queryKey: ['llm', 'config'],
    queryFn: () => api.getLLMConfig(),
    retry: false,
  });

  // Fetch available models
  const { data: modelsData, isLoading: isLoadingModels, refetch: refetchModels } = useQuery({
    queryKey: ['llm', 'models'],
    queryFn: () => api.getLLMModels(),
    enabled: !!llmConfig?.is_connected,
    retry: false,
  });

  // Fetch popular models for suggestions
  const { data: popularModelsData } = useQuery({
    queryKey: ['llm', 'popular-models'],
    queryFn: () => api.getPopularModels(),
  });

  // Update config mutation
  const updateConfigMutation = useMutation({
    mutationFn: api.updateLLMConfig,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['llm'] });
      queryClient.invalidateQueries({ queryKey: ['dashboard', 'connections'] });
      setUrlError(null);
    },
    onError: (error) => {
      setUrlError(error instanceof Error ? error.message : 'Failed to update configuration');
    },
  });

  // Test connection mutation
  const testConnectionMutation = useMutation({
    mutationFn: api.testLLMConnection,
  });

  // Pull model mutation
  const pullModelMutation = useMutation({
    mutationFn: api.pullOllamaModel,
    onSuccess: () => {
      setShowPullModal(false);
      setPullModelName('');
      refetchModels();
    },
  });

  // Set initial values when config loads
  useEffect(() => {
    if (llmConfig) {
      setCustomUrl(llmConfig.base_url || '');
      setSelectedModel(llmConfig.model_name || llmConfig.active_model || '');
    }
  }, [llmConfig]);

  // URL validation
  const validateUrl = (url: string): boolean => {
    if (!url) return false;
    try {
      new URL(url);
      return true;
    } catch {
      return false;
    }
  };

  const handleUrlChange = () => {
    if (!validateUrl(customUrl)) {
      setUrlError('Please enter a valid URL (e.g., http://ollama:11434)');
      return;
    }
    const currentUrl = llmConfig?.base_url || '';
    if (customUrl !== currentUrl) {
      updateConfigMutation.mutate({ base_url: customUrl });
    }
  };

  const handleModelChange = (modelName: string) => {
    setSelectedModel(modelName);
    updateConfigMutation.mutate({ model_name: modelName });
  };

  const provider: string = llmConfig?.provider || 'unknown';
  const providerInfo = PROVIDER_CONFIG[provider] || PROVIDER_CONFIG.unknown;

  return (
    <div>
      <Card
        title={
          <div className="flex items-center justify-between w-full">
            <span className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
              LLM Service
            </span>
            <div className="flex items-center gap-2">
              {llmConfig?.is_connected ? (
                <span className="badge-success text-[10px]" role="status" aria-label="Connected">
                  Connected
                </span>
              ) : (
                <span className="badge-critical text-[10px]" role="status" aria-label="Disconnected">
                  Disconnected
                </span>
              )}
            </div>
          </div>
        }
        accent="plasma"
      >
        {isLoadingConfig ? (
          <div className="flex items-center justify-center py-8" role="status" aria-label="Loading configuration">
            <LoadingSpinner />
          </div>
        ) : configError ? (
          <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg" role="alert">
            <p className="text-sm text-red-400">
              Failed to load LLM configuration: {configError instanceof Error ? configError.message : 'Unknown error'}
            </p>
            <button
              onClick={() => window.location.reload()}
              className="btn-secondary text-xs mt-2"
            >
              Retry
            </button>
          </div>
        ) : (
          <div className="space-y-6">
            {/* Provider Status */}
            <div className="flex items-center gap-4 p-4 bg-slate-800/30 rounded-lg border border-slate-700/50">
              <div className={`p-3 rounded-lg bg-slate-800/50 ${providerInfo.color}`} aria-hidden="true">
                {providerInfo.icon}
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <span className={`font-semibold ${providerInfo.color}`}>{providerInfo.name}</span>
                  {llmConfig?.active_model && (
                    <span className="text-xs text-slate-500 font-mono">• {llmConfig.active_model}</span>
                  )}
                </div>
                <p className="text-xs text-slate-500 font-mono truncate mt-1">{llmConfig?.base_url || ''}</p>
              </div>
              <button
                onClick={() => testConnectionMutation.mutate()}
                disabled={testConnectionMutation.isPending}
                className="group relative inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-indigo-600/20 border border-indigo-500/50 rounded-lg hover:bg-indigo-600/30 hover:border-indigo-400/70 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-[0_0_15px_rgba(99,102,241,0.3)]"
                aria-label="Test LLM connection"
              >
                {testConnectionMutation.isPending ? (
                  <>
                    <LoadingSpinner size="sm" />
                    <span>Testing...</span>
                  </>
                ) : (
                  <>
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span>Test Connection</span>
                  </>
                )}
              </button>
            </div>

            {/* Test Result */}
            <AnimatePresence>
              {testConnectionMutation.data && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className={`p-4 rounded-lg border ${
                    testConnectionMutation.data.success
                      ? 'bg-emerald-500/10 border-emerald-500/30'
                      : 'bg-red-500/10 border-red-500/30'
                  }`}
                  role="alert"
                >
                  <div className="flex items-start gap-3">
                    {testConnectionMutation.data.success ? (
                      <svg className="w-5 h-5 text-emerald-400 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                    ) : (
                      <svg className="w-5 h-5 text-red-400 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                      </svg>
                    )}
                    <div>
                      <p className={`text-sm font-medium ${testConnectionMutation.data.success ? 'text-emerald-400' : 'text-red-400'}`}>
                        {testConnectionMutation.data.message}
                      </p>
                      {testConnectionMutation.data.response_time_ms && (
                        <p className="text-xs text-slate-500 mt-1">
                          Response time: {testConnectionMutation.data.response_time_ms}ms
                          {testConnectionMutation.data.model_used && ` • Model: ${testConnectionMutation.data.model_used}`}
                        </p>
                      )}
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* URL Configuration */}
            <div>
              <label htmlFor="llm-url" className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                LLM Service URL
              </label>
              <div className="flex gap-2">
                <input
                  id="llm-url"
                  type="url"
                  value={customUrl}
                  onChange={(e) => {
                    setCustomUrl(e.target.value);
                    setUrlError(null);
                  }}
                  placeholder="http://ollama:11434"
                  className={`input-field flex-1 font-mono text-sm ${urlError ? 'border-red-500 focus:border-red-500 focus:ring-red-500/50' : ''}`}
                  aria-describedby={urlError ? 'url-error' : undefined}
                  aria-invalid={urlError ? 'true' : 'false'}
                />
                <button
                  onClick={handleUrlChange}
                  disabled={updateConfigMutation.isPending || customUrl === (llmConfig?.base_url || '')}
                  className="inline-flex items-center gap-2 px-5 py-2.5 text-sm font-semibold text-white bg-gradient-to-r from-indigo-600 to-purple-600 rounded-lg hover:from-indigo-500 hover:to-purple-500 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-indigo-500/25 hover:shadow-indigo-500/40"
                >
                  {updateConfigMutation.isPending ? (
                    <>
                      <LoadingSpinner size="sm" />
                      <span>Saving...</span>
                    </>
                  ) : (
                    <>
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      <span>Apply URL</span>
                    </>
                  )}
                </button>
              </div>
              {urlError && (
                <p id="url-error" className="mt-2 text-sm text-red-400" role="alert">
                  {urlError}
                </p>
              )}
              <div className="flex flex-wrap gap-2 mt-3" role="group" aria-label="Quick URL presets">
                {URL_PRESETS.map((preset) => (
                  <button
                    key={preset.url}
                    onClick={() => setCustomUrl(preset.url)}
                    className={`group inline-flex items-center gap-2 px-4 py-2 text-xs font-medium text-${preset.color}-300 bg-${preset.color}-500/10 border border-${preset.color}-500/30 rounded-lg hover:bg-${preset.color}-500/20 hover:border-${preset.color}-400/50 hover:text-${preset.color}-200 transition-all duration-200`}
                  >
                    <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                    </svg>
                    <span>{preset.label}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Model Selection */}
            <div>
              <label htmlFor="model-select" className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                Active Model
              </label>
              {isLoadingModels ? (
                <div className="flex items-center gap-2 text-sm text-slate-500" role="status">
                  <LoadingSpinner size="sm" />
                  Loading models...
                </div>
              ) : modelsData?.models && modelsData.models.length > 0 ? (
                <div className="space-y-3">
                  <select
                    id="model-select"
                    value={selectedModel}
                    onChange={(e) => handleModelChange(e.target.value)}
                    className="select-field w-full font-mono text-sm"
                  >
                    <option value="">Select a model...</option>
                    {modelsData.models.map((model) => {
                      const formattedSize = formatModelSize(model.size);
                      return (
                        <option key={model.id} value={model.id}>
                          {model.id} {formattedSize && `(${formattedSize})`}
                        </option>
                      );
                    })}
                  </select>

                  {/* Model cards */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-3" role="radiogroup" aria-label="Available models">
                    {modelsData.models.map((model) => {
                      const formattedSize = formatModelSize(model.size);
                      const isActive = selectedModel === model.id;
                      return (
                        <motion.button
                          key={model.id}
                          onClick={() => handleModelChange(model.id)}
                          className={`group relative p-4 pr-20 rounded-xl border text-left transition-all ${
                            isActive
                              ? 'border-indigo-500 bg-gradient-to-br from-indigo-500/20 to-purple-500/10 shadow-lg shadow-indigo-500/20'
                              : 'border-slate-700/50 bg-slate-800/30 hover:border-indigo-500/50 hover:bg-slate-800/50'
                          }`}
                          whileHover={{ scale: 1.02, y: -2 }}
                          whileTap={{ scale: 0.98 }}
                          role="radio"
                          aria-checked={isActive}
                        >
                          {isActive && (
                            <div className="absolute top-3 right-3">
                              <span className="inline-flex items-center gap-1 px-2.5 py-1 text-[10px] font-semibold text-indigo-300 bg-indigo-500/20 border border-indigo-500/30 rounded-full">
                                <svg className="w-2.5 h-2.5" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
                                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                                </svg>
                                Active
                              </span>
                            </div>
                          )}
                          <div className="flex flex-col gap-2">
                            <div className="font-mono text-sm font-semibold text-white truncate pr-16">
                              {model.name}
                            </div>
                            {formattedSize && (
                              <div className="flex items-center gap-1.5">
                                <svg className="w-3.5 h-3.5 text-slate-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                                </svg>
                                <span className={`text-xs font-semibold ${isActive ? 'text-indigo-300' : 'text-slate-400'}`}>
                                  {formattedSize}
                                </span>
                              </div>
                            )}
                          </div>
                        </motion.button>
                      );
                    })}
                  </div>
                </div>
              ) : llmConfig?.is_connected ? (
                <p className="text-sm text-slate-500">No models available. Pull a model to get started.</p>
              ) : (
                <p className="text-sm text-slate-500">Connect to LLM service to see available models.</p>
              )}
            </div>

            {/* Pull Model (Ollama only) */}
            {provider === 'ollama' && (
              <div>
                <div className="flex items-center justify-between mb-3">
                  <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    Download New Model
                  </label>
                  <button
                    onClick={() => setShowPullModal(true)}
                    className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-slate-700/50 border border-slate-600/50 rounded-lg hover:bg-slate-700/70 hover:border-slate-500/70 transition-all duration-200 hover:shadow-lg hover:shadow-slate-500/10"
                  >
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                    <span>Pull Model</span>
                  </button>
                </div>

                {/* Popular models */}
                {popularModelsData?.models && (
                  <div className="flex flex-wrap gap-2.5" role="group" aria-label="Popular models to download">
                    {popularModelsData.models.map((model) => {
                      const formattedSize = formatModelSize(model.size);
                      return (
                        <button
                          key={model.id}
                          onClick={() => {
                            setPullModelName(model.id);
                            setShowPullModal(true);
                          }}
                          className="group inline-flex items-center gap-2.5 px-4 py-2.5 rounded-lg border border-slate-700/50 bg-slate-800/40 hover:border-indigo-500/60 hover:bg-indigo-500/10 transition-all duration-200 hover:shadow-md hover:shadow-indigo-500/10"
                        >
                          <div className="flex-1 text-left min-w-0">
                            <div className="text-xs font-semibold text-slate-200 group-hover:text-white truncate">
                              {model.name}
                            </div>
                            {formattedSize && (
                              <div className="flex items-center gap-1.5 mt-0.5">
                                <svg className="w-3 h-3 text-slate-500 group-hover:text-indigo-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                                </svg>
                                <span className="text-[10px] font-medium text-slate-400 group-hover:text-indigo-300">
                                  {formattedSize}
                                </span>
                              </div>
                            )}
                          </div>
                          <svg className="w-4 h-4 text-slate-500 group-hover:text-indigo-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                          </svg>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            )}

            {/* Pull Modal */}
            <AnimatePresence>
              {showPullModal && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4"
                  onClick={() => setShowPullModal(false)}
                  role="dialog"
                  aria-modal="true"
                  aria-labelledby="pull-modal-title"
                >
                  <motion.div
                    initial={{ scale: 0.95, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0.95, opacity: 0 }}
                    className="bg-slate-900 border border-slate-700 rounded-xl p-6 w-full max-w-md"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <h3 id="pull-modal-title" className="text-lg font-semibold text-white mb-4">
                      Pull Ollama Model
                    </h3>
                    <p className="text-sm text-slate-400 mb-4">
                      Enter the model name to download. Large models may take several minutes.
                    </p>
                    <input
                      type="text"
                      value={pullModelName}
                      onChange={(e) => setPullModelName(e.target.value)}
                      placeholder="e.g., llama3.2, deepseek-r1:8b"
                      className="input-field w-full font-mono text-sm mb-4"
                      autoFocus
                      aria-label="Model name to pull"
                    />

                    {pullModelMutation.data && (
                      <div
                        className={`p-3 rounded-lg mb-4 ${
                          pullModelMutation.data.success
                            ? 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-400'
                            : 'bg-red-500/10 border border-red-500/30 text-red-400'
                        }`}
                        role="alert"
                      >
                        <p className="text-sm">{pullModelMutation.data.message}</p>
                      </div>
                    )}

                    <div className="flex justify-end gap-3">
                      <button
                        onClick={() => setShowPullModal(false)}
                        className="inline-flex items-center gap-2 px-5 py-2.5 text-sm font-medium text-slate-300 bg-slate-800/50 border border-slate-700/50 rounded-lg hover:bg-slate-800/70 hover:border-slate-600/70 transition-all duration-200"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={() => pullModelMutation.mutate(pullModelName)}
                        disabled={!pullModelName || pullModelMutation.isPending}
                        className="inline-flex items-center gap-2 px-5 py-2.5 text-sm font-semibold text-white bg-gradient-to-r from-indigo-600 to-purple-600 rounded-lg hover:from-indigo-500 hover:to-purple-500 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-indigo-500/25 hover:shadow-indigo-500/40"
                      >
                        {pullModelMutation.isPending ? (
                          <>
                            <LoadingSpinner size="sm" />
                            <span>Pulling...</span>
                          </>
                        ) : (
                          <>
                            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                            </svg>
                            <span>Pull Model</span>
                          </>
                        )}
                      </button>
                    </div>
                  </motion.div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}
      </Card>
    </div>
  );
}
