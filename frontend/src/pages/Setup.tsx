/**
 * Setup - Environment Configuration
 *
 * A settings page for configuring the .env file
 * with individually saveable sections.
 */

import { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useMutation, useQuery } from '@tanstack/react-query';
import { clsx } from 'clsx';
import toast from 'react-hot-toast';
import { Card } from '../components/Card';
import { ToggleSwitch } from '../components/ui';
import type {
  EnvironmentConfig,
  AppEnvironment,
  UserRole,
  TestConnectionRequest,
} from '../types/setup';
import { defaultEnvironmentConfig } from '../types/setup';
import { api } from '../api/client';

// Section definitions for navigation
const SECTIONS = [
  {
    id: 'environment',
    title: 'Environment',
    description: 'Application settings',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
      </svg>
    ),
  },
  {
    id: 'backend-db',
    title: 'Backend Database',
    description: 'PostgreSQL configuration',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
      </svg>
    ),
  },
  {
    id: 'xsight-db',
    title: 'XSight Database',
    description: 'SOTI telemetry source',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01" />
      </svg>
    ),
  },
  {
    id: 'mobicontrol-db',
    title: 'MobiControl DB',
    description: 'Device inventory (optional)',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 18h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
      </svg>
    ),
  },
  {
    id: 'llm',
    title: 'AI / LLM',
    description: 'AI-powered insights',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    ),
  },
  {
    id: 'mobicontrol-api',
    title: 'MobiControl API',
    description: 'Real-time data (optional)',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
      </svg>
    ),
  },
  {
    id: 'streaming',
    title: 'Streaming',
    description: 'Real-time processing',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
  },
  {
    id: 'security',
    title: 'Security',
    description: 'API keys & access',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
      </svg>
    ),
  },
];

// Input field component
interface FormFieldProps {
  label: string;
  description?: string;
  required?: boolean;
  error?: string;
  children: React.ReactNode;
}

function FormField({ label, description, required, error, children }: FormFieldProps) {
  return (
    <div className="space-y-2">
      <label className="block">
        <span className="text-sm font-medium text-white">
          {label}
          {required && <span className="text-amber-400 ml-1">*</span>}
        </span>
        {description && (
          <span className="block text-xs text-slate-500 mt-0.5">{description}</span>
        )}
      </label>
      {children}
      {error && (
        <p className="text-xs text-red-400 flex items-center gap-1">
          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          {error}
        </p>
      )}
    </div>
  );
}

// Text input component
interface TextInputProps {
  value: string;
  onChange: (value: string) => void;
  type?: 'text' | 'password' | 'number';
  placeholder?: string;
  disabled?: boolean;
}

function TextInput({ value, onChange, type = 'text', placeholder, disabled }: TextInputProps) {
  return (
    <input
      type={type}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      disabled={disabled}
      className={clsx(
        'w-full px-4 py-2.5 rounded-lg',
        'bg-slate-800/50 border border-slate-700/50',
        'text-white placeholder-slate-500',
        'focus:outline-none focus:ring-2 focus:ring-amber-500/50 focus:border-amber-500/50',
        'transition-all duration-200',
        disabled && 'opacity-50 cursor-not-allowed'
      )}
    />
  );
}

// Select input component
interface SelectInputProps {
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
  disabled?: boolean;
}

function SelectInput({ value, onChange, options, disabled }: SelectInputProps) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      disabled={disabled}
      className={clsx(
        'w-full px-4 py-2.5 rounded-lg',
        'bg-slate-800/50 border border-slate-700/50',
        'text-white',
        'focus:outline-none focus:ring-2 focus:ring-amber-500/50 focus:border-amber-500/50',
        'transition-all duration-200',
        disabled && 'opacity-50 cursor-not-allowed'
      )}
    >
      {options.map((opt) => (
        <option key={opt.value} value={opt.value}>
          {opt.label}
        </option>
      ))}
    </select>
  );
}

// Connection test button
interface TestConnectionButtonProps {
  type: TestConnectionRequest['type'];
  config: Partial<EnvironmentConfig>;
  label?: string;
}

function TestConnectionButton({ type, config, label = 'Test Connection' }: TestConnectionButtonProps) {
  const [status, setStatus] = useState<'idle' | 'testing' | 'success' | 'error'>('idle');
  const [message, setMessage] = useState('');

  const testMutation = useMutation({
    mutationFn: () => api.testSetupConnection({ type, config }),
    onMutate: () => {
      setStatus('testing');
      setMessage('');
    },
    onSuccess: (data) => {
      setStatus(data.success ? 'success' : 'error');
      setMessage(data.message);
      if (data.success) {
        toast.success(data.message);
      } else {
        toast.error(data.message);
      }
    },
    onError: (error: Error) => {
      setStatus('error');
      setMessage(error.message);
      toast.error(error.message);
    },
  });

  return (
    <div className="flex items-center gap-3">
      <button
        type="button"
        onClick={() => testMutation.mutate()}
        disabled={status === 'testing'}
        className={clsx(
          'px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200',
          'flex items-center gap-2',
          status === 'testing' && 'opacity-50 cursor-not-allowed',
          status === 'success' && 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30',
          status === 'error' && 'bg-red-500/20 text-red-400 border border-red-500/30',
          status === 'idle' && 'bg-slate-700/50 text-slate-300 hover:bg-slate-700 border border-slate-600/50'
        )}
      >
        {status === 'testing' ? (
          <>
            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            Testing...
          </>
        ) : status === 'success' ? (
          <>
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
            Connected
          </>
        ) : status === 'error' ? (
          <>
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
            Failed
          </>
        ) : (
          <>
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            {label}
          </>
        )}
      </button>
      {message && status !== 'idle' && (
        <span className={clsx(
          'text-xs',
          status === 'success' && 'text-emerald-400',
          status === 'error' && 'text-red-400'
        )}>
          {message}
        </span>
      )}
    </div>
  );
}

// Loading spinner
function LoadingSpinner() {
  return (
    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
    </svg>
  );
}

// Section save button component
interface SaveSectionButtonProps {
  sectionId: string;
  config: Partial<EnvironmentConfig>;
  isPending: boolean;
  onSave: () => void;
}

function SaveSectionButton({ sectionId, isPending, onSave }: SaveSectionButtonProps) {
  return (
    <div className="flex items-center justify-end pt-4 border-t border-slate-700/30 mt-6">
      <button
        type="button"
        onClick={onSave}
        disabled={isPending}
        className={clsx(
          'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all',
          isPending ? 'opacity-50 cursor-not-allowed' : '',
          'bg-amber-500/20 text-amber-400 border border-amber-500/30 hover:bg-amber-500/30'
        )}
      >
        {isPending ? (
          <>
            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            Saving...
          </>
        ) : (
          <>
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
            Save {sectionId === 'environment' ? 'Environment' : sectionId === 'backend-db' ? 'Database' : sectionId === 'xsight-db' ? 'XSight' : sectionId === 'mobicontrol-db' ? 'MobiControl DB' : sectionId === 'llm' ? 'LLM' : sectionId === 'mobicontrol-api' ? 'API' : sectionId === 'streaming' ? 'Streaming' : 'Security'} Settings
          </>
        )}
      </button>
    </div>
  );
}

export default function Setup() {
  const [activeSection, setActiveSection] = useState('environment');
  const [config, setConfig] = useState<EnvironmentConfig>(defaultEnvironmentConfig);
  const [showPasswords, setShowPasswords] = useState<Record<string, boolean>>({});
  const [savingSection, setSavingSection] = useState<string | null>(null);
  const [configLoaded, setConfigLoaded] = useState(false);

  // Fetch current configuration from backend
  const { data: savedConfig, isLoading: isLoadingConfig } = useQuery({
    queryKey: ['setup-config'],
    queryFn: () => api.getSetupConfig(),
    staleTime: 0, // Always fetch fresh data
    refetchOnWindowFocus: false,
  });

  // Merge saved config with defaults when data is loaded
  useEffect(() => {
    if (savedConfig && !configLoaded) {
      // Merge saved config with defaults (saved values take precedence)
      const mergedConfig = {
        ...defaultEnvironmentConfig,
        ...savedConfig,
      } as EnvironmentConfig;
      setConfig(mergedConfig);
      setConfigLoaded(true);
    }
  }, [savedConfig, configLoaded]);

  // Update config helper
  const updateConfig = useCallback(<K extends keyof EnvironmentConfig>(
    key: K,
    value: EnvironmentConfig[K]
  ) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  }, []);

  // Toggle password visibility
  const togglePassword = (field: string) => {
    setShowPasswords((prev) => ({ ...prev, [field]: !prev[field] }));
  };

  // Save section mutation
  const saveSectionMutation = useMutation({
    mutationFn: (section: string) => api.saveSetupConfig({ section, config: config as unknown as Record<string, unknown> }),
    onMutate: (section) => {
      setSavingSection(section);
    },
    onSuccess: (data) => {
      if (data.success) {
        toast.success('Settings saved successfully!');
      } else {
        toast.error(data.message);
      }
      setSavingSection(null);
    },
    onError: (error: Error) => {
      toast.error(`Failed to save: ${error.message}`);
      setSavingSection(null);
    },
  });

  // Save full configuration mutation
  const saveMutation = useMutation({
    mutationFn: () => api.saveSetupConfig(config as unknown as Record<string, unknown>),
    onSuccess: (data) => {
      if (data.success) {
        toast.success('Configuration saved successfully!');
      } else {
        toast.error(data.message);
      }
    },
    onError: (error: Error) => {
      toast.error(`Failed to save: ${error.message}`);
    },
  });

  // Save a specific section
  const saveSection = (sectionId: string) => {
    saveSectionMutation.mutate(sectionId);
  };

  // Download as .env file
  const downloadEnvFile = () => {
    const envContent = generateEnvContent(config);
    const blob = new Blob([envContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = '.env';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast.success('.env file downloaded!');
  };

  // Generate .env file content
  const generateEnvContent = (cfg: EnvironmentConfig): string => {
    return `# =============================================================================
# SOTI Stella Sentinel - Environment Configuration
# =============================================================================
# Generated by Setup Wizard
# NEVER commit this .env file to version control.
# =============================================================================

# Application Environment
# Options: local, development, staging, production
APP_ENV=${cfg.app_env}

# =============================================================================
# API Security & Tenant Context
# =============================================================================
REQUIRE_API_KEY=${cfg.require_api_key}
API_KEY=${cfg.api_key}
API_KEY_ROLE=${cfg.api_key_role}
TRUST_CLIENT_HEADERS=${cfg.trust_client_headers}
REQUIRE_TENANT_HEADER=${cfg.require_tenant_header}
DEFAULT_TENANT_ID=${cfg.default_tenant_id}
DEFAULT_USER_ROLE=${cfg.default_user_role}
TENANT_ID_ALLOWLIST=${cfg.tenant_id_allowlist}
ALLOW_IN_PROCESS_TRAINING=${cfg.allow_in_process_training}

# Observability
ENABLE_OTEL=${cfg.enable_otel}
OTEL_EXPORTER_OTLP_ENDPOINT=${cfg.otel_exporter_otlp_endpoint}
OTEL_EXPORTER_OTLP_INSECURE=${cfg.otel_exporter_otlp_insecure}
OTEL_SERVICE_NAME=${cfg.otel_service_name}
ENABLE_METRICS=${cfg.enable_metrics}
ENABLE_DB_METRICS=${cfg.enable_db_metrics}
ENABLE_TENANT_METRICS=${cfg.enable_tenant_metrics}
TENANT_METRICS_ALLOWLIST=${cfg.tenant_metrics_allowlist}
RESULTS_DB_USE_MIGRATIONS=${cfg.results_db_use_migrations}

# =============================================================================
# Frontend Configuration
# =============================================================================
FRONTEND_PORT=${cfg.frontend_port}
FRONTEND_DEV_PORT=${cfg.frontend_dev_port}
VITE_TENANT_ID=${cfg.vite_tenant_id}
VITE_API_KEY=${cfg.vite_api_key}
VITE_USER_ID=${cfg.vite_user_id}
VITE_USER_ROLE=${cfg.vite_user_role}

# =============================================================================
# PostgreSQL Backend Database
# =============================================================================
BACKEND_DB_HOST=${cfg.backend_db_host}
BACKEND_DB_PORT=${cfg.backend_db_port}
BACKEND_DB_NAME=${cfg.backend_db_name}
BACKEND_DB_USER=${cfg.backend_db_user}
BACKEND_DB_PASS=${cfg.backend_db_pass}
BACKEND_DB_CONNECT_TIMEOUT=${cfg.backend_db_connect_timeout}
BACKEND_DB_STATEMENT_TIMEOUT_MS=${cfg.backend_db_statement_timeout_ms}

# =============================================================================
# SOTI XSight SQL Server Configuration
# =============================================================================
DW_DB_HOST=${cfg.dw_db_host}
DW_DB_PORT=${cfg.dw_db_port}
DW_DB_NAME=${cfg.dw_db_name}
DW_DB_USER=${cfg.dw_db_user}
DW_DB_PASS=${cfg.dw_db_pass}
DW_DB_DRIVER=${cfg.dw_db_driver}
DW_DB_CONNECT_TIMEOUT=${cfg.dw_db_connect_timeout}
DW_DB_QUERY_TIMEOUT=${cfg.dw_db_query_timeout}
DW_TRUST_SERVER_CERT=${cfg.dw_trust_server_cert}

# =============================================================================
# SOTI MobiControl SQL Server Configuration (Optional)
# =============================================================================
MC_DB_HOST=${cfg.mc_db_host}
MC_DB_PORT=${cfg.mc_db_port}
MC_DB_NAME=${cfg.mc_db_name}
MC_DB_USER=${cfg.mc_db_user}
MC_DB_PASS=${cfg.mc_db_pass}
MC_DB_DRIVER=${cfg.mc_db_driver}
MC_DB_CONNECT_TIMEOUT=${cfg.mc_db_connect_timeout}
MC_DB_QUERY_TIMEOUT=${cfg.mc_db_query_timeout}
MC_TRUST_SERVER_CERT=${cfg.mc_trust_server_cert}

# =============================================================================
# LLM Configuration
# =============================================================================
ENABLE_LLM=${cfg.enable_llm}
LLM_BASE_URL=${cfg.llm_base_url}
LLM_API_KEY=${cfg.llm_api_key}
LLM_MODEL_NAME=${cfg.llm_model_name}
LLM_API_VERSION=${cfg.llm_api_version}
LLM_BASE_URL_ALLOWLIST=${cfg.llm_base_url_allowlist}

# =============================================================================
# Real-Time Streaming Configuration
# =============================================================================
ENABLE_STREAMING=${cfg.enable_streaming}
REDIS_URL=${cfg.redis_url}
REDIS_DB=${cfg.redis_db}
STREAM_BUFFER_SIZE=${cfg.stream_buffer_size}
STREAM_FLUSH_INTERVAL_MS=${cfg.stream_flush_interval_ms}

# =============================================================================
# SOTI MobiControl API Configuration (Optional)
# =============================================================================
MOBICONTROL_SERVER_URL=${cfg.mobicontrol_server_url}
MOBICONTROL_CLIENT_ID=${cfg.mobicontrol_client_id}
MOBICONTROL_CLIENT_SECRET=${cfg.mobicontrol_client_secret}
MOBICONTROL_USERNAME=${cfg.mobicontrol_username}
MOBICONTROL_PASSWORD=${cfg.mobicontrol_password}
MOBICONTROL_TENANT_ID=${cfg.mobicontrol_tenant_id}
`;
  };

  // Render section content
  const renderSectionContent = () => {
    switch (activeSection) {
      case 'environment':
        return (
          <div className="space-y-6">
            <FormField
              label="Application Environment"
              description="Select the environment mode for your deployment"
              required
            >
              <SelectInput
                value={config.app_env}
                onChange={(v) => updateConfig('app_env', v as AppEnvironment)}
                options={[
                  { value: 'local', label: 'Local Development' },
                  { value: 'development', label: 'Development' },
                  { value: 'staging', label: 'Staging' },
                  { value: 'production', label: 'Production' },
                ]}
              />
            </FormField>

            <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/30">
              <h4 className="text-sm font-medium text-white mb-3">Frontend Ports</h4>
              <div className="grid grid-cols-2 gap-4">
                <FormField label="Production Port" description="Nginx frontend (default: 3000)">
                  <TextInput
                    type="number"
                    value={config.frontend_port.toString()}
                    onChange={(v) => updateConfig('frontend_port', parseInt(v) || 3000)}
                    placeholder="3000"
                  />
                </FormField>
                <FormField label="Dev Server Port" description="Vite dev server (default: 5173)">
                  <TextInput
                    type="number"
                    value={config.frontend_dev_port.toString()}
                    onChange={(v) => updateConfig('frontend_dev_port', parseInt(v) || 5173)}
                    placeholder="5173"
                  />
                </FormField>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/30">
              <h4 className="text-sm font-medium text-white mb-3">Observability</h4>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-white">Enable Metrics</p>
                    <p className="text-xs text-slate-500">Expose Prometheus metrics endpoint</p>
                  </div>
                  <ToggleSwitch
                    enabled={config.enable_metrics}
                    onChange={(v) => updateConfig('enable_metrics', v)}
                    variant="stellar"
                  />
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-white">Database Metrics</p>
                    <p className="text-xs text-slate-500">Include database query metrics</p>
                  </div>
                  <ToggleSwitch
                    enabled={config.enable_db_metrics}
                    onChange={(v) => updateConfig('enable_db_metrics', v)}
                    variant="stellar"
                  />
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-white">OpenTelemetry</p>
                    <p className="text-xs text-slate-500">Enable distributed tracing</p>
                  </div>
                  <ToggleSwitch
                    enabled={config.enable_otel}
                    onChange={(v) => updateConfig('enable_otel', v)}
                    variant="stellar"
                  />
                </div>
                {config.enable_otel && (
                  <FormField label="OTLP Endpoint" description="OpenTelemetry collector endpoint">
                    <TextInput
                      value={config.otel_exporter_otlp_endpoint}
                      onChange={(v) => updateConfig('otel_exporter_otlp_endpoint', v)}
                      placeholder="http://localhost:4317"
                    />
                  </FormField>
                )}
              </div>
            </div>
            <SaveSectionButton
              sectionId="environment"
              config={config}
              isPending={savingSection === 'environment'}
              onSave={() => saveSection('environment')}
            />
          </div>
        );

      case 'backend-db':
        return (
          <div className="space-y-6">
            <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30">
              <div className="flex items-start gap-3">
                <svg className="w-5 h-5 text-emerald-400 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div>
                  <p className="text-sm text-emerald-400 font-medium">PostgreSQL Backend</p>
                  <p className="text-xs text-slate-400 mt-1">
                    This database stores application data including detected anomalies, baselines, and investigation notes.
                    For Docker deployments, the default settings work with the included PostgreSQL container.
                  </p>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <FormField label="Host" required>
                <TextInput
                  value={config.backend_db_host}
                  onChange={(v) => updateConfig('backend_db_host', v)}
                  placeholder="postgres"
                />
              </FormField>
              <FormField label="Port" required>
                <TextInput
                  type="number"
                  value={config.backend_db_port.toString()}
                  onChange={(v) => updateConfig('backend_db_port', parseInt(v) || 5432)}
                  placeholder="5432"
                />
              </FormField>
            </div>

            <FormField label="Database Name" required>
              <TextInput
                value={config.backend_db_name}
                onChange={(v) => updateConfig('backend_db_name', v)}
                placeholder="anomaly_detection"
              />
            </FormField>

            <div className="grid grid-cols-2 gap-4">
              <FormField label="Username" required>
                <TextInput
                  value={config.backend_db_user}
                  onChange={(v) => updateConfig('backend_db_user', v)}
                  placeholder="postgres"
                />
              </FormField>
              <FormField label="Password" required>
                <div className="relative">
                  <TextInput
                    type={showPasswords['backend_db'] ? 'text' : 'password'}
                    value={config.backend_db_pass}
                    onChange={(v) => updateConfig('backend_db_pass', v)}
                    placeholder="changeme"
                  />
                  <button
                    type="button"
                    onClick={() => togglePassword('backend_db')}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
                  >
                    {showPasswords['backend_db'] ? (
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                      </svg>
                    ) : (
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                      </svg>
                    )}
                  </button>
                </div>
              </FormField>
            </div>

            <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/30">
              <h4 className="text-sm font-medium text-white mb-3">Timeouts</h4>
              <div className="grid grid-cols-2 gap-4">
                <FormField label="Connect Timeout (seconds)">
                  <TextInput
                    type="number"
                    value={config.backend_db_connect_timeout.toString()}
                    onChange={(v) => updateConfig('backend_db_connect_timeout', parseInt(v) || 5)}
                    placeholder="5"
                  />
                </FormField>
                <FormField label="Statement Timeout (ms)">
                  <TextInput
                    type="number"
                    value={config.backend_db_statement_timeout_ms.toString()}
                    onChange={(v) => updateConfig('backend_db_statement_timeout_ms', parseInt(v) || 30000)}
                    placeholder="30000"
                  />
                </FormField>
              </div>
            </div>

            <TestConnectionButton
              type="backend_db"
              config={config}
              label="Test PostgreSQL Connection"
            />
            <SaveSectionButton
              sectionId="backend-db"
              config={config}
              isPending={savingSection === 'backend-db'}
              onSave={() => saveSection('backend-db')}
            />
          </div>
        );

      case 'xsight-db':
        return (
          <div className="space-y-6">
            <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30">
              <div className="flex items-start gap-3">
                <svg className="w-5 h-5 text-amber-400 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div>
                  <p className="text-sm text-amber-400 font-medium">SOTI XSight Data Warehouse</p>
                  <p className="text-xs text-slate-400 mt-1">
                    Source telemetry data including battery, network, app usage, and device metrics.
                    For Docker, use <code className="text-amber-400">host.docker.internal</code> (Mac/Windows)
                    or <code className="text-amber-400">172.17.0.1</code> (Linux).
                  </p>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <FormField label="Host" required>
                <TextInput
                  value={config.dw_db_host}
                  onChange={(v) => updateConfig('dw_db_host', v)}
                  placeholder="your-server.example.com"
                />
              </FormField>
              <FormField label="Port" required>
                <TextInput
                  type="number"
                  value={config.dw_db_port.toString()}
                  onChange={(v) => updateConfig('dw_db_port', parseInt(v) || 1433)}
                  placeholder="1433"
                />
              </FormField>
            </div>

            <FormField label="Database Name" required>
              <TextInput
                value={config.dw_db_name}
                onChange={(v) => updateConfig('dw_db_name', v)}
                placeholder="XSight"
              />
            </FormField>

            <div className="grid grid-cols-2 gap-4">
              <FormField label="Username" required>
                <TextInput
                  value={config.dw_db_user}
                  onChange={(v) => updateConfig('dw_db_user', v)}
                  placeholder="xsight_reader"
                />
              </FormField>
              <FormField label="Password" required>
                <div className="relative">
                  <TextInput
                    type={showPasswords['dw_db'] ? 'text' : 'password'}
                    value={config.dw_db_pass}
                    onChange={(v) => updateConfig('dw_db_pass', v)}
                    placeholder="Enter password"
                  />
                  <button
                    type="button"
                    onClick={() => togglePassword('dw_db')}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
                  >
                    {showPasswords['dw_db'] ? (
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                      </svg>
                    ) : (
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                      </svg>
                    )}
                  </button>
                </div>
              </FormField>
            </div>

            <FormField label="ODBC Driver">
              <SelectInput
                value={config.dw_db_driver}
                onChange={(v) => updateConfig('dw_db_driver', v)}
                options={[
                  { value: 'ODBC Driver 18 for SQL Server', label: 'ODBC Driver 18 for SQL Server' },
                  { value: 'ODBC Driver 17 for SQL Server', label: 'ODBC Driver 17 for SQL Server' },
                  { value: 'FreeTDS', label: 'FreeTDS' },
                ]}
              />
            </FormField>

            <div className="flex items-center justify-between p-4 rounded-xl bg-slate-800/30 border border-slate-700/30">
              <div>
                <p className="text-sm text-white">Trust Server Certificate</p>
                <p className="text-xs text-slate-500">Enable only for self-signed certificates</p>
              </div>
              <ToggleSwitch
                enabled={config.dw_trust_server_cert}
                onChange={(v) => updateConfig('dw_trust_server_cert', v)}
                variant="stellar"
              />
            </div>

            <TestConnectionButton
              type="dw_db"
              config={config}
              label="Test XSight Connection"
            />
            <SaveSectionButton
              sectionId="xsight-db"
              config={config}
              isPending={savingSection === 'xsight-db'}
              onSave={() => saveSection('xsight-db')}
            />
          </div>
        );

      case 'mobicontrol-db':
        return (
          <div className="space-y-6">
            <div className="p-4 rounded-xl bg-slate-700/30 border border-slate-600/30">
              <div className="flex items-start gap-3">
                <svg className="w-5 h-5 text-slate-400 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div>
                  <p className="text-sm text-slate-300 font-medium">SOTI MobiControl Database (Optional)</p>
                  <p className="text-xs text-slate-500 mt-1">
                    Device inventory, compliance status, and policy information.
                    Used for enrichment data - not required for core anomaly detection.
                  </p>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <FormField label="Host">
                <TextInput
                  value={config.mc_db_host}
                  onChange={(v) => updateConfig('mc_db_host', v)}
                  placeholder="your-server.example.com"
                />
              </FormField>
              <FormField label="Port">
                <TextInput
                  type="number"
                  value={config.mc_db_port.toString()}
                  onChange={(v) => updateConfig('mc_db_port', parseInt(v) || 1433)}
                  placeholder="1433"
                />
              </FormField>
            </div>

            <FormField label="Database Name">
              <TextInput
                value={config.mc_db_name}
                onChange={(v) => updateConfig('mc_db_name', v)}
                placeholder="MobiControlDB"
              />
            </FormField>

            <div className="grid grid-cols-2 gap-4">
              <FormField label="Username">
                <TextInput
                  value={config.mc_db_user}
                  onChange={(v) => updateConfig('mc_db_user', v)}
                  placeholder="mc_reader"
                />
              </FormField>
              <FormField label="Password">
                <div className="relative">
                  <TextInput
                    type={showPasswords['mc_db'] ? 'text' : 'password'}
                    value={config.mc_db_pass}
                    onChange={(v) => updateConfig('mc_db_pass', v)}
                    placeholder="Enter password"
                  />
                  <button
                    type="button"
                    onClick={() => togglePassword('mc_db')}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
                  >
                    {showPasswords['mc_db'] ? (
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                      </svg>
                    ) : (
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                      </svg>
                    )}
                  </button>
                </div>
              </FormField>
            </div>

            <div className="flex items-center justify-between p-4 rounded-xl bg-slate-800/30 border border-slate-700/30">
              <div>
                <p className="text-sm text-white">Trust Server Certificate</p>
                <p className="text-xs text-slate-500">Enable only for self-signed certificates</p>
              </div>
              <ToggleSwitch
                enabled={config.mc_trust_server_cert}
                onChange={(v) => updateConfig('mc_trust_server_cert', v)}
                variant="stellar"
              />
            </div>

            {config.mc_db_host && (
              <TestConnectionButton
                type="mc_db"
                config={config}
                label="Test MobiControl DB Connection"
              />
            )}
            <SaveSectionButton
              sectionId="mobicontrol-db"
              config={config}
              isPending={savingSection === 'mobicontrol-db'}
              onSave={() => saveSection('mobicontrol-db')}
            />
          </div>
        );

      case 'llm':
        return (
          <div className="space-y-6">
            <div className="p-4 rounded-xl bg-indigo-500/10 border border-indigo-500/30">
              <div className="flex items-start gap-3">
                <svg className="w-5 h-5 text-indigo-400 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
                <div>
                  <p className="text-sm text-indigo-400 font-medium">AI-Powered Insights</p>
                  <p className="text-xs text-slate-400 mt-1">
                    Configure the LLM provider for intelligent anomaly explanations and troubleshooting advice.
                    Supports Ollama (recommended), Azure OpenAI, OpenAI, LM Studio, or vLLM.
                  </p>
                </div>
              </div>
            </div>

            {/* Enable LLM Toggle */}
            <div className="flex items-center justify-between p-4 rounded-lg bg-slate-800/50 border border-slate-700/50">
              <div>
                <p className="text-sm text-white font-medium">Enable AI/LLM Features</p>
                <p className="text-xs text-slate-400 mt-1">Turn on AI-powered anomaly explanations and troubleshooting</p>
              </div>
              <ToggleSwitch
                enabled={config.enable_llm}
                onChange={(v) => updateConfig('enable_llm', v)}
                variant="stellar"
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* LLM Provider Quick Select */}
              <div className="md:col-span-2">
                <p className="text-sm font-medium text-white mb-3">Quick Setup</p>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                  {[
                    { name: 'Ollama (Docker)', url: 'http://ollama:11434', model: 'llama3.2', key: 'not-needed' },
                    { name: 'Ollama (Local)', url: 'http://host.docker.internal:11434', model: 'llama3.2', key: 'not-needed' },
                    { name: 'LM Studio', url: 'http://host.docker.internal:1234', model: 'local-model', key: 'not-needed' },
                    { name: 'OpenAI', url: 'https://api.openai.com/v1', model: 'gpt-4-turbo-preview', key: '' },
                  ].map((preset) => (
                    <button
                      key={preset.name}
                      type="button"
                      onClick={() => {
                        updateConfig('llm_base_url', preset.url);
                        updateConfig('llm_model_name', preset.model);
                        updateConfig('llm_api_key', preset.key);
                      }}
                      className={clsx(
                        'p-3 rounded-lg text-left transition-all',
                        'border',
                        config.llm_base_url === preset.url
                          ? 'bg-indigo-500/20 border-indigo-500/50 text-indigo-400'
                          : 'bg-slate-800/50 border-slate-700/50 text-slate-400 hover:border-slate-600'
                      )}
                    >
                      <p className="text-sm font-medium">{preset.name}</p>
                      <p className="text-xs opacity-60 truncate">{preset.model}</p>
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <FormField label="Base URL" required description="LLM API endpoint">
              <TextInput
                value={config.llm_base_url}
                onChange={(v) => updateConfig('llm_base_url', v)}
                placeholder="http://ollama:11434"
              />
            </FormField>

            <div className="grid grid-cols-2 gap-4">
              <FormField label="Model Name" required>
                <TextInput
                  value={config.llm_model_name}
                  onChange={(v) => updateConfig('llm_model_name', v)}
                  placeholder="llama3.2"
                />
              </FormField>
              <FormField label="API Key" description="Leave 'not-needed' for Ollama">
                <div className="relative">
                  <TextInput
                    type={showPasswords['llm'] ? 'text' : 'password'}
                    value={config.llm_api_key}
                    onChange={(v) => updateConfig('llm_api_key', v)}
                    placeholder="not-needed"
                  />
                  <button
                    type="button"
                    onClick={() => togglePassword('llm')}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
                  >
                    {showPasswords['llm'] ? (
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                      </svg>
                    ) : (
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                      </svg>
                    )}
                  </button>
                </div>
              </FormField>
            </div>

            <FormField label="API Version" description="Required for Azure OpenAI (e.g., 2024-12-01-preview)">
              <TextInput
                value={config.llm_api_version}
                onChange={(v) => updateConfig('llm_api_version', v)}
                placeholder="2024-12-01-preview"
              />
            </FormField>

            <TestConnectionButton
              type="llm"
              config={config}
              label="Test LLM Connection"
            />
            <SaveSectionButton
              sectionId="llm"
              config={config}
              isPending={savingSection === 'llm'}
              onSave={() => saveSection('llm')}
            />
          </div>
        );

      case 'mobicontrol-api':
        return (
          <div className="space-y-6">
            <div className="p-4 rounded-xl bg-slate-700/30 border border-slate-600/30">
              <div className="flex items-start gap-3">
                <svg className="w-5 h-5 text-slate-400 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
                </svg>
                <div>
                  <p className="text-sm text-slate-300 font-medium">MobiControl REST API (Optional)</p>
                  <p className="text-xs text-slate-500 mt-1">
                    For real-time device data and actions via MobiControl REST API.
                    Leave blank if not using MobiControl API integration.
                  </p>
                </div>
              </div>
            </div>

            <FormField label="Server URL">
              <TextInput
                value={config.mobicontrol_server_url}
                onChange={(v) => updateConfig('mobicontrol_server_url', v)}
                placeholder="https://mobicontrol.example.com"
              />
            </FormField>

            <div className="grid grid-cols-2 gap-4">
              <FormField label="Client ID">
                <TextInput
                  value={config.mobicontrol_client_id}
                  onChange={(v) => updateConfig('mobicontrol_client_id', v)}
                  placeholder="your-client-id"
                />
              </FormField>
              <FormField label="Client Secret">
                <div className="relative">
                  <TextInput
                    type={showPasswords['mc_api_secret'] ? 'text' : 'password'}
                    value={config.mobicontrol_client_secret}
                    onChange={(v) => updateConfig('mobicontrol_client_secret', v)}
                    placeholder="your-client-secret"
                  />
                  <button
                    type="button"
                    onClick={() => togglePassword('mc_api_secret')}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
                  >
                    {showPasswords['mc_api_secret'] ? (
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                      </svg>
                    ) : (
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                      </svg>
                    )}
                  </button>
                </div>
              </FormField>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <FormField label="Username">
                <TextInput
                  value={config.mobicontrol_username}
                  onChange={(v) => updateConfig('mobicontrol_username', v)}
                  placeholder="api-user"
                />
              </FormField>
              <FormField label="Password">
                <div className="relative">
                  <TextInput
                    type={showPasswords['mc_api_pass'] ? 'text' : 'password'}
                    value={config.mobicontrol_password}
                    onChange={(v) => updateConfig('mobicontrol_password', v)}
                    placeholder="Enter password"
                  />
                  <button
                    type="button"
                    onClick={() => togglePassword('mc_api_pass')}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
                  >
                    {showPasswords['mc_api_pass'] ? (
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                      </svg>
                    ) : (
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                      </svg>
                    )}
                  </button>
                </div>
              </FormField>
            </div>

            <FormField label="Tenant ID" description="MobiControl tenant identifier">
              <TextInput
                value={config.mobicontrol_tenant_id}
                onChange={(v) => updateConfig('mobicontrol_tenant_id', v)}
                placeholder="your-tenant-id"
              />
            </FormField>

            {config.mobicontrol_server_url && (
              <TestConnectionButton
                type="mobicontrol_api"
                config={config}
                label="Test MobiControl API"
              />
            )}
            <SaveSectionButton
              sectionId="mobicontrol-api"
              config={config}
              isPending={savingSection === 'mobicontrol-api'}
              onSave={() => saveSection('mobicontrol-api')}
            />
          </div>
        );

      case 'streaming':
        return (
          <div className="space-y-6">
            <div className="flex items-center justify-between p-4 rounded-xl bg-slate-800/30 border border-slate-700/30">
              <div>
                <p className="text-sm text-white font-medium">Enable Real-Time Streaming</p>
                <p className="text-xs text-slate-500">Process telemetry data in real-time using Redis pub/sub</p>
              </div>
              <ToggleSwitch
                enabled={config.enable_streaming}
                onChange={(v) => updateConfig('enable_streaming', v)}
                variant="stellar"
                size="lg"
              />
            </div>

            <AnimatePresence>
              {config.enable_streaming && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="space-y-6"
                >
                  <div className="p-4 rounded-xl bg-orange-500/10 border border-orange-500/30">
                    <div className="flex items-start gap-3">
                      <svg className="w-5 h-5 text-orange-400 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                      <div>
                        <p className="text-sm text-orange-400 font-medium">Redis Required</p>
                        <p className="text-xs text-slate-400 mt-1">
                          Real-time streaming requires Redis for pub/sub messaging.
                          The included docker-compose.yml provides a Redis container.
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <FormField label="Redis URL" required>
                      <TextInput
                        value={config.redis_url}
                        onChange={(v) => updateConfig('redis_url', v)}
                        placeholder="redis://redis:6379"
                      />
                    </FormField>
                    <FormField label="Redis Database">
                      <TextInput
                        type="number"
                        value={config.redis_db.toString()}
                        onChange={(v) => updateConfig('redis_db', parseInt(v) || 0)}
                        placeholder="0"
                      />
                    </FormField>
                  </div>

                  <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/30">
                    <h4 className="text-sm font-medium text-white mb-3">Buffer Settings</h4>
                    <div className="grid grid-cols-2 gap-4">
                      <FormField label="Buffer Size" description="Maximum messages in buffer">
                        <TextInput
                          type="number"
                          value={config.stream_buffer_size.toString()}
                          onChange={(v) => updateConfig('stream_buffer_size', parseInt(v) || 1000)}
                          placeholder="1000"
                        />
                      </FormField>
                      <FormField label="Flush Interval (ms)" description="How often to flush the buffer">
                        <TextInput
                          type="number"
                          value={config.stream_flush_interval_ms.toString()}
                          onChange={(v) => updateConfig('stream_flush_interval_ms', parseInt(v) || 100)}
                          placeholder="100"
                        />
                      </FormField>
                    </div>
                  </div>

                  <TestConnectionButton
                    type="redis"
                    config={config}
                    label="Test Redis Connection"
                  />
                </motion.div>
              )}
            </AnimatePresence>
            <SaveSectionButton
              sectionId="streaming"
              config={config}
              isPending={savingSection === 'streaming'}
              onSave={() => saveSection('streaming')}
            />
          </div>
        );

      case 'security':
        return (
          <div className="space-y-6">
            <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30">
              <div className="flex items-start gap-3">
                <svg className="w-5 h-5 text-amber-400 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                <div>
                  <p className="text-sm text-amber-400 font-medium">Security Settings</p>
                  <p className="text-xs text-slate-400 mt-1">
                    For production deployments, enable API key enforcement and disable trust of client headers.
                    Local development can use relaxed settings for convenience.
                  </p>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 rounded-xl bg-slate-800/30 border border-slate-700/30">
                <div>
                  <p className="text-sm text-white">Require API Key</p>
                  <p className="text-xs text-slate-500">Enforce API key for all requests (recommended for production)</p>
                </div>
                <ToggleSwitch
                  enabled={config.require_api_key}
                  onChange={(v) => updateConfig('require_api_key', v)}
                  variant="stellar"
                />
              </div>

              {config.require_api_key && (
                <FormField label="API Key" required description="Secret key for API authentication">
                  <div className="relative">
                    <TextInput
                      type={showPasswords['api_key'] ? 'text' : 'password'}
                      value={config.api_key}
                      onChange={(v) => updateConfig('api_key', v)}
                      placeholder="Generate a secure random key"
                    />
                    <button
                      type="button"
                      onClick={() => togglePassword('api_key')}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
                    >
                      {showPasswords['api_key'] ? (
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                        </svg>
                      ) : (
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                        </svg>
                      )}
                    </button>
                  </div>
                </FormField>
              )}

              <FormField label="API Key Role" description="Default role assigned to API key requests">
                <SelectInput
                  value={config.api_key_role}
                  onChange={(v) => updateConfig('api_key_role', v as UserRole)}
                  options={[
                    { value: 'viewer', label: 'Viewer (read-only)' },
                    { value: 'operator', label: 'Operator (read + actions)' },
                    { value: 'admin', label: 'Admin (full access)' },
                  ]}
                />
              </FormField>

              <div className="flex items-center justify-between p-4 rounded-xl bg-slate-800/30 border border-slate-700/30">
                <div>
                  <p className="text-sm text-white">Trust Client Headers</p>
                  <p className="text-xs text-slate-500">Trust X-User-Id / X-User-Role headers (local dev only)</p>
                </div>
                <ToggleSwitch
                  enabled={config.trust_client_headers}
                  onChange={(v) => updateConfig('trust_client_headers', v)}
                  variant={config.trust_client_headers ? 'danger' : 'stellar'}
                />
              </div>

              <div className="flex items-center justify-between p-4 rounded-xl bg-slate-800/30 border border-slate-700/30">
                <div>
                  <p className="text-sm text-white">Require Tenant Header</p>
                  <p className="text-xs text-slate-500">Enable multi-tenant isolation</p>
                </div>
                <ToggleSwitch
                  enabled={config.require_tenant_header}
                  onChange={(v) => updateConfig('require_tenant_header', v)}
                  variant="stellar"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <FormField label="Default Tenant ID">
                  <TextInput
                    value={config.default_tenant_id}
                    onChange={(v) => updateConfig('default_tenant_id', v)}
                    placeholder="default"
                  />
                </FormField>
                <FormField label="Default User Role">
                  <SelectInput
                    value={config.default_user_role}
                    onChange={(v) => updateConfig('default_user_role', v as UserRole)}
                    options={[
                      { value: 'viewer', label: 'Viewer' },
                      { value: 'operator', label: 'Operator' },
                      { value: 'admin', label: 'Admin' },
                    ]}
                  />
                </FormField>
              </div>

              <FormField label="Tenant ID Allowlist" description="Comma-separated list of allowed tenant IDs (leave empty to allow all)">
                <TextInput
                  value={config.tenant_id_allowlist}
                  onChange={(v) => updateConfig('tenant_id_allowlist', v)}
                  placeholder="tenant-a,tenant-b,tenant-c"
                />
              </FormField>
            </div>
            <SaveSectionButton
              sectionId="security"
              config={config}
              isPending={savingSection === 'security'}
              onSave={() => saveSection('security')}
            />
          </div>
        );

      default:
        return null;
    }
  };

  const currentSection = SECTIONS.find(s => s.id === activeSection) || SECTIONS[0];

  // Show loading state while fetching config
  if (isLoadingConfig) {
    return (
      <motion.div
        className="flex items-center justify-center min-h-[400px]"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <div className="text-center">
          <LoadingSpinner />
          <p className="text-slate-400 mt-4">Loading current configuration...</p>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-white">Settings</h1>
          <p className="text-slate-500 mt-1">Configure your environment settings individually</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={downloadEnvFile}
            className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium bg-slate-700/50 text-slate-300 hover:bg-slate-700 border border-slate-600/50 transition-all"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Export .env
          </button>
          <button
            type="button"
            onClick={() => saveMutation.mutate()}
            disabled={saveMutation.isPending}
            className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium btn-stellar"
          >
            {saveMutation.isPending ? (
              <>
                <LoadingSpinner />
                Saving...
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" />
                </svg>
                Save All
              </>
            )}
          </button>
        </div>
      </div>

      {/* Main Content - Two Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Section Navigation Sidebar */}
        <div className="lg:col-span-1">
          <Card noPadding>
            <nav className="p-2 space-y-1">
              {SECTIONS.map((section) => (
                <button
                  key={section.id}
                  type="button"
                  onClick={() => setActiveSection(section.id)}
                  className={clsx(
                    'w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all text-left',
                    activeSection === section.id
                      ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                      : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-300 border border-transparent'
                  )}
                >
                  <div
                    className={clsx(
                      'w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0',
                      activeSection === section.id
                        ? 'bg-amber-500/30 text-amber-400'
                        : 'bg-slate-800 text-slate-500'
                    )}
                  >
                    {section.icon}
                  </div>
                  <div className="min-w-0">
                    <p className="text-sm font-medium truncate">{section.title}</p>
                    <p className="text-xs text-slate-500 truncate">{section.description}</p>
                  </div>
                </button>
              ))}
            </nav>
          </Card>
        </div>

        {/* Section Content */}
        <div className="lg:col-span-3">
          <Card>
            <div className="mb-6">
              <div className="flex items-center gap-3 mb-2">
                <div className="w-10 h-10 rounded-lg bg-amber-500/20 flex items-center justify-center text-amber-400">
                  {currentSection.icon}
                </div>
                <div>
                  <h2 className="text-xl font-bold text-white">{currentSection.title}</h2>
                  <p className="text-sm text-slate-500">{currentSection.description}</p>
                </div>
              </div>
            </div>

            <AnimatePresence mode="wait">
              <motion.div
                key={activeSection}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.2 }}
              >
                {renderSectionContent()}
              </motion.div>
            </AnimatePresence>
          </Card>
        </div>
      </div>
    </motion.div>
  );
}

