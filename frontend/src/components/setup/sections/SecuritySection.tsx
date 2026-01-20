/**
 * Security configuration section component.
 */

import { ToggleSwitch } from '../../ui';
import { FormField, TextInput, SelectInput, PasswordInput, SaveSectionButton } from '../FormComponents';
import type { EnvironmentConfig, UserRole } from '../../../types/setup';

interface SecuritySectionProps {
  config: EnvironmentConfig;
  updateConfig: <K extends keyof EnvironmentConfig>(key: K, value: EnvironmentConfig[K]) => void;
  showPasswords: Record<string, boolean>;
  togglePassword: (field: string) => void;
  savingSection: string | null;
  onSave: () => void;
}

export function SecuritySection({
  config,
  updateConfig,
  showPasswords,
  togglePassword,
  savingSection,
  onSave
}: SecuritySectionProps) {
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
            <PasswordInput
              value={config.api_key}
              onChange={(v) => updateConfig('api_key', v)}
              placeholder="Generate a secure random key"
              showPassword={showPasswords['api_key'] || false}
              onTogglePassword={() => togglePassword('api_key')}
            />
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
        isPending={savingSection === 'security'}
        onSave={onSave}
      />
    </div>
  );
}
