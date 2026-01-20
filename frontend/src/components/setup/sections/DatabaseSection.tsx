/**
 * Database configuration sections (Backend, XSight, MobiControl).
 */

import { ToggleSwitch } from '../../ui';
import { FormField, TextInput, SelectInput, PasswordInput, SaveSectionButton } from '../FormComponents';
import { TestConnectionButton } from '../TestConnectionButton';
import type { EnvironmentConfig } from '../../../types/setup';

interface DatabaseSectionProps {
  config: EnvironmentConfig;
  updateConfig: <K extends keyof EnvironmentConfig>(key: K, value: EnvironmentConfig[K]) => void;
  showPasswords: Record<string, boolean>;
  togglePassword: (field: string) => void;
  savingSection: string | null;
  onSave: () => void;
}

// ============================================================================
// Backend Database Section (PostgreSQL)
// ============================================================================

export function BackendDatabaseSection({
  config,
  updateConfig,
  showPasswords,
  togglePassword,
  savingSection,
  onSave
}: DatabaseSectionProps) {
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
          <PasswordInput
            value={config.backend_db_pass}
            onChange={(v) => updateConfig('backend_db_pass', v)}
            placeholder="changeme"
            showPassword={showPasswords['backend_db'] || false}
            onTogglePassword={() => togglePassword('backend_db')}
          />
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
        isPending={savingSection === 'backend-db'}
        onSave={onSave}
      />
    </div>
  );
}

// ============================================================================
// XSight Database Section (SQL Server)
// ============================================================================

export function XSightDatabaseSection({
  config,
  updateConfig,
  showPasswords,
  togglePassword,
  savingSection,
  onSave
}: DatabaseSectionProps) {
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
          <PasswordInput
            value={config.dw_db_pass}
            onChange={(v) => updateConfig('dw_db_pass', v)}
            placeholder="Enter password"
            showPassword={showPasswords['dw_db'] || false}
            onTogglePassword={() => togglePassword('dw_db')}
          />
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
        isPending={savingSection === 'xsight-db'}
        onSave={onSave}
      />
    </div>
  );
}

// ============================================================================
// MobiControl Database Section (SQL Server - Optional)
// ============================================================================

export function MobiControlDatabaseSection({
  config,
  updateConfig,
  showPasswords,
  togglePassword,
  savingSection,
  onSave
}: DatabaseSectionProps) {
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
          <PasswordInput
            value={config.mc_db_pass}
            onChange={(v) => updateConfig('mc_db_pass', v)}
            placeholder="Enter password"
            showPassword={showPasswords['mc_db'] || false}
            onTogglePassword={() => togglePassword('mc_db')}
          />
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
        isPending={savingSection === 'mobicontrol-db'}
        onSave={onSave}
      />
    </div>
  );
}
