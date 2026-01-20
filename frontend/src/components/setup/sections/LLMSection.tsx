/**
 * LLM/AI configuration section component.
 */

import { clsx } from 'clsx';
import { ToggleSwitch } from '../../ui';
import { FormField, TextInput, PasswordInput, SaveSectionButton } from '../FormComponents';
import { TestConnectionButton } from '../TestConnectionButton';
import { LLM_PRESETS } from '../setupConstants';
import type { EnvironmentConfig } from '../../../types/setup';

interface LLMSectionProps {
  config: EnvironmentConfig;
  updateConfig: <K extends keyof EnvironmentConfig>(key: K, value: EnvironmentConfig[K]) => void;
  showPasswords: Record<string, boolean>;
  togglePassword: (field: string) => void;
  savingSection: string | null;
  onSave: () => void;
}

export function LLMSection({
  config,
  updateConfig,
  showPasswords,
  togglePassword,
  savingSection,
  onSave
}: LLMSectionProps) {
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
            {LLM_PRESETS.map((preset) => (
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
          <PasswordInput
            value={config.llm_api_key}
            onChange={(v) => updateConfig('llm_api_key', v)}
            placeholder="not-needed"
            showPassword={showPasswords['llm'] || false}
            onTogglePassword={() => togglePassword('llm')}
          />
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
        isPending={savingSection === 'llm'}
        onSave={onSave}
      />
    </div>
  );
}
