/**
 * MobiControl API configuration section component.
 */

import { FormField, TextInput, PasswordInput, SaveSectionButton } from '../FormComponents';
import { TestConnectionButton } from '../TestConnectionButton';
import type { EnvironmentConfig } from '../../../types/setup';

interface MobiControlAPISectionProps {
  config: EnvironmentConfig;
  updateConfig: <K extends keyof EnvironmentConfig>(key: K, value: EnvironmentConfig[K]) => void;
  showPasswords: Record<string, boolean>;
  togglePassword: (field: string) => void;
  savingSection: string | null;
  onSave: () => void;
}

export function MobiControlAPISection({
  config,
  updateConfig,
  showPasswords,
  togglePassword,
  savingSection,
  onSave
}: MobiControlAPISectionProps) {
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
          <PasswordInput
            value={config.mobicontrol_client_secret}
            onChange={(v) => updateConfig('mobicontrol_client_secret', v)}
            placeholder="your-client-secret"
            showPassword={showPasswords['mc_api_secret'] || false}
            onTogglePassword={() => togglePassword('mc_api_secret')}
          />
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
          <PasswordInput
            value={config.mobicontrol_password}
            onChange={(v) => updateConfig('mobicontrol_password', v)}
            placeholder="Enter password"
            showPassword={showPasswords['mc_api_pass'] || false}
            onTogglePassword={() => togglePassword('mc_api_pass')}
          />
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
        isPending={savingSection === 'mobicontrol-api'}
        onSave={onSave}
      />
    </div>
  );
}
