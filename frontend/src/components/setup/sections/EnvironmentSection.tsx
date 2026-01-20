/**
 * Environment settings section component.
 */

import { ToggleSwitch } from '../../ui';
import { FormField, TextInput, SelectInput, SaveSectionButton } from '../FormComponents';
import type { EnvironmentConfig, AppEnvironment } from '../../../types/setup';

interface EnvironmentSectionProps {
  config: EnvironmentConfig;
  updateConfig: <K extends keyof EnvironmentConfig>(key: K, value: EnvironmentConfig[K]) => void;
  savingSection: string | null;
  onSave: () => void;
}

export function EnvironmentSection({ config, updateConfig, savingSection, onSave }: EnvironmentSectionProps) {
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
        isPending={savingSection === 'environment'}
        onSave={onSave}
      />
    </div>
  );
}
